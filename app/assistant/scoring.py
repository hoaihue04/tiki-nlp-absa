from __future__ import annotations

from collections import defaultdict
from typing import Any


ASPECT_WEIGHTS = {
    "PRODUCT#SAFETY": 2.0,
    "SELLER#AUTHENTICITY": 1.8,
    "PRODUCT#MATERIAL": 1.6,
    "PRODUCT#SIZE": 1.5,
    "PRODUCT#DURABILITY": 1.4,
    "PRODUCT#COMFORT": 1.3,
    "PRODUCT#QUALITY": 1.2,
    "PRODUCT#VALUE": 1.1,
}

SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def compute_purchase_score(aspect_items: list[dict[str, Any]], review_count: int = 0) -> dict[str, Any]:
    weighted_sum = 0.0
    total_weight = 0.0
    aspect_summary: dict[str, dict[str, float]] = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})

    for item in aspect_items:
        aspect = str(item.get("aspect") or item.get("category") or "")
        sentiment = str(item.get("sentiment") or item.get("polarity") or "neutral")
        confidence = float(item.get("confidence") or 0.5)
        aspect_weight = ASPECT_WEIGHTS.get(aspect, 1.0)
        score = SENTIMENT_SCORE.get(sentiment, 0.0)
        weight = max(confidence, 0.05) * aspect_weight
        weighted_sum += score * weight
        total_weight += weight
        if sentiment in aspect_summary[aspect]:
            aspect_summary[aspect][sentiment] += 1

    if total_weight <= 0:
        normalized = 0.5
    else:
        normalized = (weighted_sum / total_weight + 1.0) / 2.0
        normalized = max(0.0, min(1.0, normalized))

    evidence_confidence = min(1.0, max(review_count, len(aspect_items)) / 30.0)
    if normalized >= 0.68 and evidence_confidence >= 0.35:
        label = "Nen mua"
    elif normalized < 0.42:
        label = "Nen tranh"
    else:
        label = "Can nhac"

    return {
        "label": label,
        "score": round(normalized, 4),
        "confidence": round(evidence_confidence, 4),
        "aspect_summary": dict(aspect_summary),
    }


def select_evidence(aspect_items: list[dict[str, Any]], sentiment: str | None = None, limit: int = 4) -> list[dict[str, Any]]:
    pool = []
    for item in aspect_items:
        if sentiment and item.get("sentiment") != sentiment and item.get("polarity") != sentiment:
            continue
        text = str(item.get("sentence") or item.get("evidence") or "").strip()
        if not text:
            continue
        pool.append({
            "review_id": str(item.get("review_id") or ""),
            "aspect": str(item.get("aspect") or item.get("category") or ""),
            "sentiment": str(item.get("sentiment") or item.get("polarity") or "neutral"),
            "confidence": float(item.get("confidence") or 0.0),
            "text": text[:350],
        })
    pool.sort(key=lambda row: row["confidence"], reverse=True)
    return pool[:limit]

