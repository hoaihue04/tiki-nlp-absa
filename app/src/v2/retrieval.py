from __future__ import annotations

import hashlib
import json
from typing import Any

from app.src.v2.cache import RedisCache
from app.src.v2.config import settings
from app.src.v2.query_analyzer import QueryAnalysis
from app.src.v2.vector_store import QdrantOpinionStore


STATISTICAL_INTENTS = {
    "SUMMARY",
    "COMPLAINT_SUMMARY",
    "PRICE_VALUE",
    "DELIVERY_SERVICE",
}

SEMANTIC_INTENTS = {
    "ASPECT_QA",
    "PRODUCT_FIT",
    "PRODUCT_QA",
    "RISK_CHECK",
    "RECOMMENDATION",
    "UNKNOWN",
}


class HybridRetrievalService:
    def __init__(
        self,
        vector_store: QdrantOpinionStore | None = None,
        cache: RedisCache | None = None,
    ) -> None:
        self.vector_store = vector_store or QdrantOpinionStore()
        self.cache = cache or RedisCache()

    def retrieve(
        self,
        product_id: str,
        question: str,
        query: QueryAnalysis,
        structured_evidence: list[dict[str, Any]],
        limit: int = 6,
    ) -> dict[str, Any]:
        if not settings.hybrid_retrieval_enabled:
            return {
                "strategy": "postgres_only",
                "evidence": structured_evidence[:limit],
                "qdrant_used": False,
                "cache_hit": False,
            }

        cache_key = _cache_key(product_id, question, query)
        cached = self.cache.get_json(cache_key)
        if isinstance(cached, dict):
            cached["cache_hit"] = True
            return cached

        strategy = "postgres_first" if query.intent in STATISTICAL_INTENTS else "qdrant_first"
        qdrant_rows = self._search_qdrant(product_id, question, query, limit=max(limit, 8))

        if strategy == "postgres_first":
            merged = _hybrid_rank(
                structured_evidence + qdrant_rows,
                query=query,
                qdrant_weight=0.35,
                structured_weight=0.65,
            )
        else:
            merged = _hybrid_rank(
                qdrant_rows + structured_evidence,
                query=query,
                qdrant_weight=0.65,
                structured_weight=0.35,
            )

        result = {
            "strategy": strategy,
            "evidence": merged[:limit],
            "qdrant_used": bool(qdrant_rows),
            "cache_hit": False,
        }
        self.cache.set_json(cache_key, result, ttl_seconds=settings.retrieval_cache_ttl_seconds)
        return result

    def _search_qdrant(
        self,
        product_id: str,
        question: str,
        query: QueryAnalysis,
        limit: int,
    ) -> list[dict[str, Any]]:
        sentiments = _sentiments_for_intent(query.intent)
        aspects = query.target_aspects or None
        return self.vector_store.search(
            product_id=product_id,
            query=question,
            limit=limit,
            sentiments=sentiments,
            aspects=aspects,
        )


def _sentiments_for_intent(intent: str) -> list[str] | None:
    if intent in {"COMPLAINT_SUMMARY", "RISK_CHECK"}:
        return ["negative"]
    if intent in {"SUMMARY", "RECOMMENDATION", "PRODUCT_FIT", "PRODUCT_QA", "ASPECT_QA"}:
        return ["negative", "positive", "neutral"]
    return None


def _hybrid_rank(
    rows: list[dict[str, Any]],
    query: QueryAnalysis,
    qdrant_weight: float,
    structured_weight: float,
) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        key = str(row.get("review_id") or "") or _text_key(text)
        current = deduped.get(key)
        scored = {**row, "hybrid_score": _score_row(row, query, qdrant_weight, structured_weight)}
        if current is None or scored["hybrid_score"] > current.get("hybrid_score", 0):
            deduped[key] = scored
    out = list(deduped.values())
    out.sort(key=lambda row: row.get("hybrid_score", 0), reverse=True)
    return out


def _score_row(row: dict[str, Any], query: QueryAnalysis, qdrant_weight: float, structured_weight: float) -> float:
    semantic_score = float(row.get("semantic_score") or 0.0)
    confidence = float(row.get("confidence") or 0.0)
    source = str(row.get("source") or "postgres")
    aspect = str(row.get("aspect_code") or row.get("aspect") or "")
    sentiment = str(row.get("sentiment") or "neutral")

    aspect_bonus = 0.0
    if query.target_aspects and aspect in query.target_aspects:
        aspect_bonus = 0.2

    sentiment_bonus = 0.0
    if query.intent in {"COMPLAINT_SUMMARY", "RISK_CHECK"} and sentiment == "negative":
        sentiment_bonus = 0.25
    elif query.intent == "ASPECT_QA" and sentiment in {"negative", "positive"}:
        sentiment_bonus = 0.15
    elif query.intent in {"SUMMARY", "RECOMMENDATION"} and sentiment in {"positive", "negative"}:
        sentiment_bonus = 0.1

    source_weight = qdrant_weight if source == "qdrant" else structured_weight
    return round(source_weight + semantic_score * 0.35 + confidence * 0.25 + aspect_bonus + sentiment_bonus, 4)


def _cache_key(product_id: str, question: str, query: QueryAnalysis) -> str:
    raw = json.dumps({
        "product_id": str(product_id),
        "question": str(question).strip().lower(),
        "intent": query.intent,
        "aspects": query.target_aspects,
    }, ensure_ascii=False, sort_keys=True)
    return "retrieval:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _text_key(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
