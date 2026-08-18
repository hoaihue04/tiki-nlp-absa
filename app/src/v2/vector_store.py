from __future__ import annotations

import hashlib
from typing import Any

from app.src.v2.config import settings
from app.src.v2.embedding import EmbeddingService


class QdrantOpinionStore:
    def __init__(self, embedding: EmbeddingService | None = None) -> None:
        self.embedding = embedding or EmbeddingService()
        self.collection = settings.qdrant_collection
        self.enabled = bool(settings.qdrant_url and self.embedding.enabled)
        self._client = None

    def index_product_opinions(self, product_id: str, chunks: list[dict[str, Any]]) -> int:
        chunks = [chunk for chunk in chunks if _is_indexable_chunk(chunk)]
        if not self.enabled or not chunks:
            return 0
        vectors = self.embedding.embed_texts([chunk["text"] for chunk in chunks])
        if not vectors:
            self.enabled = False
            return 0
        try:
            client = self._get_client(vector_size=len(vectors[0]))
            from qdrant_client import models

            points = []
            for chunk, vector in zip(chunks, vectors):
                payload = {
                    "product_id": str(product_id),
                    "review_id": str(chunk.get("review_id") or ""),
                    "aspect": str(chunk.get("aspect") or ""),
                    "aspect_label": str(chunk.get("aspect_label") or chunk.get("aspect") or ""),
                    "sentiment": str(chunk.get("sentiment") or "neutral"),
                    "confidence": float(chunk.get("confidence") or 0.0),
                    "text": str(chunk.get("text") or ""),
                }
                points.append(models.PointStruct(
                    id=_point_id(product_id, payload),
                    vector=vector,
                    payload=payload,
                ))
            client.upsert(collection_name=self.collection, points=points)
            return len(points)
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
        if not self.enabled:
            return []
        vector = self.embedding.embed_text(query)
        if not vector:
            self.enabled = False
            return []
        try:
            client = self._get_client(vector_size=len(vector))
            query_filter = self._build_filter(product_id, sentiments, aspects)
            hits = client.search(
                collection_name=self.collection,
                query_vector=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except Exception:
            self.enabled = False
            return []

        rows: list[dict[str, Any]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            rows.append({
                "review_id": str(payload.get("review_id") or ""),
                "aspect": str(payload.get("aspect_label") or payload.get("aspect") or ""),
                "aspect_code": str(payload.get("aspect") or ""),
                "aspect_label": str(payload.get("aspect_label") or payload.get("aspect") or ""),
                "sentiment": str(payload.get("sentiment") or "neutral"),
                "confidence": float(payload.get("confidence") or 0.0),
                "text": str(payload.get("text") or ""),
                "semantic_score": float(hit.score or 0.0),
                "source": "qdrant",
            })
        return rows

    def _get_client(self, vector_size: int):
        if self._client is None:
            from qdrant_client import QdrantClient, models

            self._client = QdrantClient(url=settings.qdrant_url)
            try:
                self._client.get_collection(self.collection)
            except Exception:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
        return self._client

    def _build_filter(self, product_id: str, sentiments: list[str] | None, aspects: list[str] | None):
        from qdrant_client import models

        must = [
            models.FieldCondition(
                key="product_id",
                match=models.MatchValue(value=str(product_id)),
            )
        ]
        if sentiments:
            must.append(models.FieldCondition(
                key="sentiment",
                match=models.MatchAny(any=list(sentiments)),
            ))
        if aspects:
            must.append(models.FieldCondition(
                key="aspect",
                match=models.MatchAny(any=list(aspects)),
            ))
        return models.Filter(must=must)


def _point_id(product_id: str, payload: dict[str, Any]) -> int:
    raw = "|".join([
        str(product_id),
        str(payload.get("review_id") or ""),
        str(payload.get("aspect") or ""),
        str(payload.get("sentiment") or ""),
        str(payload.get("text") or ""),
    ])
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16], 16)


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
