from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.assistant.llm import LLMError
from app.assistant.query_analyzer import ALLOWED_ASPECTS, QueryAnalysis, normalize_text


ALLOWED_METRICS = {
    "top_aspects_by_sentiment",
    "aspect_sentiment_distribution",
    "negative_ratio_by_aspect",
    "positive_ratio_by_aspect",
    "review_statistics",
    "risk_summary",
    "evidence_by_aspect",
    "rating_sentiment_summary",
}

ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}
ALLOWED_GROUP_BY = {"aspect", "sentiment", "rating", "risk_type", "severity"}
ALLOWED_SORTS = {
    "count_desc",
    "negative_ratio_desc",
    "positive_ratio_desc",
    "avg_rating_desc",
    "confidence_desc",
}

STATISTICAL_INTENTS = {
    "SUMMARY",
    "COMPLAINT_SUMMARY",
    "PRICE_VALUE",
    "DELIVERY_SERVICE",
}

STATISTICAL_MARKERS = {
    "bao nhieu",
    "ty le",
    "phan tram",
    "nhieu nhat",
    "it nhat",
    "thong ke",
    "xep hang",
    "top",
    "dem",
    "so luong",
}

POSITIVE_MARKERS = {
    "tot",
    "khen",
    "hai long",
    "tich cuc",
    "uu diem",
    "danh gia tot",
    "phan hoi tot",
}

NEGATIVE_MARKERS = {
    "xau",
    "te",
    "che",
    "phan nan",
    "khong hai long",
    "tieu cuc",
    "diem yeu",
    "co van de",
}


@dataclass(frozen=True)
class StatisticsQueryPlan:
    metric: str
    sentiment: str | None = None
    target_aspects: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    sort_by: str = "count_desc"
    limit: int = 5
    include_examples: bool = True
    source: str = "fallback"

    def to_prompt_context(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "sentiment": self.sentiment,
            "target_aspects": self.target_aspects,
            "group_by": self.group_by,
            "sort_by": self.sort_by,
            "limit": self.limit,
            "include_examples": self.include_examples,
            "source": self.source,
        }


class StatisticsPlanner:
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def should_plan(self, question: str, query: QueryAnalysis) -> bool:
        if query.intent in STATISTICAL_INTENTS:
            return True
        normalized = normalize_text(question)
        return any(marker in normalized for marker in STATISTICAL_MARKERS)

    def plan(self, question: str, query: QueryAnalysis) -> StatisticsQueryPlan | None:
        if not self.should_plan(question, query):
            return None

        if self.llm is not None and self.llm.is_configured():
            llm_plan = self._plan_by_llm(question, query)
            if llm_plan is not None:
                return _align_plan_with_question(llm_plan, question, query)

        return default_plan_for_query(query, question)

    def _plan_by_llm(self, question: str, query: QueryAnalysis) -> StatisticsQueryPlan | None:
        system_prompt = (
            "You are a query planner for TikiInsight, a Vietnamese product review analytics system. "
            "Convert the user question and query analysis into a structured JSON query plan. "
            "Do not write SQL. Return only valid JSON. Do not explain."
        )
        user_prompt = (
            f"Allowed metrics: {sorted(ALLOWED_METRICS)}\n"
            f"Allowed sentiments: {sorted(ALLOWED_SENTIMENTS)} plus null\n"
            f"Allowed aspects: {sorted(ALLOWED_ASPECTS)}\n"
            f"Allowed group_by: {sorted(ALLOWED_GROUP_BY)}\n"
            f"Allowed sort_by: {sorted(ALLOWED_SORTS)}\n"
            "Schema: {\n"
            '  "metric": "",\n'
            '  "sentiment": null,\n'
            '  "target_aspects": [],\n'
            '  "group_by": [],\n'
            '  "sort_by": "count_desc",\n'
            '  "limit": 5,\n'
            '  "include_examples": true\n'
            "}\n"
            "Rules:\n"
            "- Do not generate SQL.\n"
            "- Use only allowed enum values.\n"
            "- If the question asks about complaints, use sentiment=\"negative\".\n"
            "- If the question asks about praises, use sentiment=\"positive\".\n"
            "- If the question asks how many users/reviews were positive or negative about one aspect, use metric=\"top_aspects_by_sentiment\" with that sentiment and target aspect.\n"
            "- If the question targets a specific aspect, include it in target_aspects.\n"
            "- limit must be between 1 and 20.\n"
            f"User question: {question}\n"
            f"Query analysis: {json.dumps(query.to_prompt_context(), ensure_ascii=False)}"
        )
        try:
            raw = self.llm.generate_json(system_prompt, user_prompt)
        except LLMError:
            return None
        return coerce_statistics_plan(raw, source="llm")


def coerce_statistics_plan(raw: dict[str, Any], source: str = "llm") -> StatisticsQueryPlan | None:
    metric = str(raw.get("metric") or "").strip()
    if metric not in ALLOWED_METRICS:
        return None

    sentiment_raw = raw.get("sentiment")
    sentiment = None
    if sentiment_raw is not None:
        value = str(sentiment_raw).strip().lower()
        sentiment = value if value in ALLOWED_SENTIMENTS else None

    target_aspects = [
        str(aspect)
        for aspect in raw.get("target_aspects") or []
        if str(aspect) in ALLOWED_ASPECTS
    ]

    group_by = [
        str(item)
        for item in raw.get("group_by") or []
        if str(item) in ALLOWED_GROUP_BY
    ]

    sort_by = str(raw.get("sort_by") or "count_desc").strip()
    if sort_by not in ALLOWED_SORTS:
        sort_by = "count_desc"

    try:
        limit = int(raw.get("limit") or 5)
    except Exception:
        limit = 5
    limit = max(1, min(limit, 20))

    include_examples = bool(raw.get("include_examples", True))

    return StatisticsQueryPlan(
        metric=metric,
        sentiment=sentiment,
        target_aspects=list(dict.fromkeys(target_aspects)),
        group_by=list(dict.fromkeys(group_by)),
        sort_by=sort_by,
        limit=limit,
        include_examples=include_examples,
        source=source,
    )


def default_plan_for_query(query: QueryAnalysis, question: str = "") -> StatisticsQueryPlan:
    inferred_sentiment = _infer_sentiment_from_question(question)

    if query.intent == "COMPLAINT_SUMMARY":
        return StatisticsQueryPlan(
            metric="top_aspects_by_sentiment",
            sentiment="negative",
            target_aspects=query.target_aspects,
            group_by=["aspect"],
            sort_by="count_desc",
            limit=5,
            source="fallback",
        )

    if query.target_aspects and inferred_sentiment:
        return StatisticsQueryPlan(
            metric="top_aspects_by_sentiment",
            sentiment=inferred_sentiment,
            target_aspects=query.target_aspects,
            group_by=["aspect"],
            sort_by="count_desc",
            limit=5,
            source="fallback",
        )

    if query.intent == "PRICE_VALUE":
        return StatisticsQueryPlan(
            metric="aspect_sentiment_distribution",
            target_aspects=query.target_aspects or ["PRICE#AFFORDABILITY", "PRODUCT#VALUE", "PRODUCT#QUALITY"],
            group_by=["aspect", "sentiment"],
            sort_by="count_desc",
            limit=8,
            source="fallback",
        )

    if query.intent == "DELIVERY_SERVICE":
        return StatisticsQueryPlan(
            metric="aspect_sentiment_distribution",
            target_aspects=query.target_aspects or ["DELIVERY#SPEED", "DELIVERY#PACKAGING", "DELIVERY#ACCURACY"],
            group_by=["aspect", "sentiment"],
            sort_by="count_desc",
            limit=8,
            source="fallback",
        )

    if query.intent == "RISK_CHECK":
        return StatisticsQueryPlan(
            metric="risk_summary",
            group_by=["risk_type", "severity"],
            sort_by="count_desc",
            limit=5,
            source="fallback",
        )

    if query.target_aspects:
        return StatisticsQueryPlan(
            metric="aspect_sentiment_distribution",
            target_aspects=query.target_aspects,
            group_by=["aspect", "sentiment"],
            sort_by="count_desc",
            limit=8,
            source="fallback",
        )

    return StatisticsQueryPlan(
        metric="aspect_sentiment_distribution",
        group_by=["aspect", "sentiment"],
        sort_by="count_desc",
        limit=8,
        source="fallback",
    )


def _align_plan_with_question(
    plan: StatisticsQueryPlan,
    question: str,
    query: QueryAnalysis,
) -> StatisticsQueryPlan:
    inferred_sentiment = _infer_sentiment_from_question(question)
    target_aspects = plan.target_aspects or query.target_aspects
    metric = plan.metric
    sentiment = plan.sentiment or inferred_sentiment
    group_by = plan.group_by

    if target_aspects and sentiment and _is_count_or_ratio_question(question):
        metric = "top_aspects_by_sentiment"
        group_by = ["aspect"]

    return StatisticsQueryPlan(
        metric=metric,
        sentiment=sentiment,
        target_aspects=target_aspects,
        group_by=group_by,
        sort_by=plan.sort_by,
        limit=plan.limit,
        include_examples=plan.include_examples,
        source=plan.source,
    )


def _infer_sentiment_from_question(question: str) -> str | None:
    normalized = normalize_text(question)
    if any(marker in normalized for marker in NEGATIVE_MARKERS):
        return "negative"
    if any(marker in normalized for marker in POSITIVE_MARKERS):
        return "positive"
    return None


def _is_count_or_ratio_question(question: str) -> bool:
    normalized = normalize_text(question)
    return any(marker in normalized for marker in STATISTICAL_MARKERS)
