from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import text

from app.assistant.config import settings
from app.assistant.db import get_engine
from app.assistant.embedding import EmbeddingService


class PgVectorOpinionStore:
    def __init__(self, embedding: EmbeddingService | None = None) -> None:
        self.embedding = embedding or EmbeddingService()
        self.enabled = bool(settings.database_url and self.embedding.enabled)
        self._initialized = False

    def _ensure_schema(self) -> bool:
        if self._initialized:
            return True
        engine = get_engine()
        if engine is None:
            return False
        try:
            with engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS opinion_chunks (
                        id BIGINT PRIMARY KEY,
                        product_id VARCHAR(128) NOT NULL,
                        review_id VARCHAR(128) NOT NULL DEFAULT '',
                        aspect VARCHAR(128) NOT NULL,
                        aspect_label VARCHAR(255) NOT NULL DEFAULT '',
                        sentiment VARCHAR(32) NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                        text TEXT NOT NULL,
                        embedding vector(384) NOT NULL
                    )
                """))
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS opinion_chunks_product_idx "
                    "ON opinion_chunks (product_id)"
                ))
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS opinion_chunks_embedding_hnsw_idx "
                    "ON opinion_chunks USING hnsw (embedding vector_cosine_ops)"
                ))
            self._initialized = True
            return True
        except Exception:
            return False

    def index_product_opinions(self, product_id: str, chunks: list[dict[str, Any]]) -> int:
        chunks = [chunk for chunk in chunks if _is_indexable_chunk(chunk)]
        if not self.enabled or not chunks or not self._ensure_schema():
            return 0
        vectors = self.embedding.embed_texts([chunk["text"] for chunk in chunks])
        if not vectors:
            self.enabled = False
            return 0
        try:
            engine = get_engine()
            if engine is None:
                return 0
            rows = []
            for chunk, vector in zip(chunks, vectors):
                rows.append({
                    "id": _point_id(product_id, chunk),
                    "product_id": str(product_id),
                    "review_id": str(chunk.get("review_id") or ""),
                    "aspect": str(chunk.get("aspect") or ""),
                    "aspect_label": str(chunk.get("aspect_label") or chunk.get("aspect") or ""),
                    "sentiment": str(chunk.get("sentiment") or "neutral"),
                    "confidence": float(chunk.get("confidence") or 0.0),
                    "text": str(chunk.get("text") or ""),
                    "embedding": _vector_literal(vector),
                })
            with engine.begin() as connection:
                connection.execute(text("""
                    INSERT INTO opinion_chunks
                        (id, product_id, review_id, aspect, aspect_label,
                         sentiment, confidence, text, embedding)
                    VALUES
                        (:id, :product_id, :review_id, :aspect, :aspect_label,
                         :sentiment, :confidence, :text, CAST(:embedding AS vector))
                    ON CONFLICT (id) DO UPDATE SET
                        product_id = EXCLUDED.product_id,
                        review_id = EXCLUDED.review_id,
                        aspect = EXCLUDED.aspect,
                        aspect_label = EXCLUDED.aspect_label,
                        sentiment = EXCLUDED.sentiment,
                        confidence = EXCLUDED.confidence,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding
                """), rows)
            return len(rows)
        except Exception:
            self.enabled = False
            return 0

    def search(
        self,
        product_id: str,
        query: str,
        limit: int = 8,
        sentiments: list[str] | None = None,
        aspects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled or not self._ensure_schema():
            return []
        vector = self.embedding.embed_text(query)
        if not vector:
            self.enabled = False
            return []
        try:
            engine = get_engine()
            if engine is None:
                return []
            filters = ["product_id = :product_id"]
            params: dict[str, Any] = {
                "product_id": str(product_id),
                "query_embedding": _vector_literal(vector),
                "limit": int(limit),
            }
            if sentiments:
                filters.append("sentiment = ANY(:sentiments)")
                params["sentiments"] = list(sentiments)
            if aspects:
                filters.append("aspect = ANY(:aspects)")
                params["aspects"] = list(aspects)
            query = text(f"""
                SELECT review_id, aspect, aspect_label, sentiment,
                       confidence, text,
                       1 - (embedding <=> CAST(:query_embedding AS vector)) AS semantic_score
                FROM opinion_chunks
                WHERE {' AND '.join(filters)}
                ORDER BY embedding <=> CAST(:query_embedding AS vector)
                LIMIT :limit
            """)
            with engine.connect() as connection:
                hits = connection.execute(query, params).mappings().all()
        except Exception:
            self.enabled = False
            return []

        rows: list[dict[str, Any]] = []
        for hit in hits:
            rows.append({
                "review_id": str(hit.get("review_id") or ""),
                "aspect": str(hit.get("aspect_label") or hit.get("aspect") or ""),
                "aspect_code": str(hit.get("aspect") or ""),
                "aspect_label": str(hit.get("aspect_label") or hit.get("aspect") or ""),
                "sentiment": str(hit.get("sentiment") or "neutral"),
                "confidence": float(hit.get("confidence") or 0.0),
                "text": str(hit.get("text") or ""),
                "semantic_score": float(hit.get("semantic_score") or 0.0),
                "source": "pgvector",
            })
        return rows


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _point_id(product_id: str, payload: dict[str, Any]) -> int:
    raw = "|".join([
        str(product_id),
        str(payload.get("review_id") or ""),
        str(payload.get("aspect") or ""),
        str(payload.get("sentiment") or ""),
        str(payload.get("text") or ""),
    ])
    # Keep the deterministic key inside PostgreSQL BIGINT's signed range.
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:15], 16)


def _is_indexable_chunk(chunk: dict[str, Any]) -> bool:
    text = str(chunk.get("text") or "").strip()
    aspect = str(chunk.get("aspect") or "").strip()
    sentiment = str(chunk.get("sentiment") or "").strip()
    return bool(text and aspect and sentiment in {"positive", "neutral", "negative"} and len(text) >= 24)


def opinion_chunks_from_aspect_items(aspect_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in aspect_items:
        aspect = str(item.get("aspect") or item.get("category") or "").strip()
        sentiment = str(item.get("sentiment") or item.get("polarity") or "neutral").strip()
        text = _compress_text(str(item.get("sentence") or item.get("text") or item.get("evidence") or ""))
        review_id = str(item.get("review_id") or "").strip()
        if not aspect or sentiment not in {"positive", "neutral", "negative"} or len(text) < 24:
            continue
        key = "|".join([review_id, aspect, sentiment, " ".join(text.lower().split())])
        if key in seen:
            continue
        seen.add(key)
        chunks.append({
            "review_id": review_id,
            "aspect": aspect,
            "aspect_label": _aspect_label(aspect),
            "sentiment": sentiment,
            "confidence": float(item.get("confidence") or 0.0),
            "text": text,
        })
    return chunks


def _compress_text(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip()


def _aspect_label(aspect: str) -> str:
    labels = {
        "PRODUCT#SAFETY": "Độ an toàn",
        "PRODUCT#MATERIAL": "Chất liệu",
        "PRODUCT#SCENT": "Mùi sản phẩm",
        "PRODUCT#SIZE": "Kích thước",
        "PRODUCT#ABSORPTION": "Khả năng thấm hút",
        "PRODUCT#FUNCTION": "Công năng sử dụng",
        "PRODUCT#QUALITY": "Chất lượng",
        "PRODUCT#VALUE": "Độ đáng tiền",
        "PRODUCT#DURABILITY": "Độ bền",
        "PRODUCT#COMFORT": "Độ thoải mái",
        "PRICE#AFFORDABILITY": "Giá cả",
        "DELIVERY#SPEED": "Giao hàng",
        "DELIVERY#PACKAGING": "Bao bì",
        "DELIVERY#ACCURACY": "Giao đúng/đủ hàng",
        "SELLER#AUTHENTICITY": "Tính chính hãng",
    }
    return labels.get(aspect, aspect)
