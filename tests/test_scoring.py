from __future__ import annotations

from app.assistant.scoring import compute_purchase_score, select_evidence


def test_compute_purchase_score_returns_buy_label_when_positive_evidence_is_strong() -> None:
    aspect_items = [
        {"aspect": "PRODUCT#SAFETY", "sentiment": "positive", "confidence": 0.9},
        {"aspect": "PRODUCT#QUALITY", "sentiment": "positive", "confidence": 0.8},
        {"aspect": "PRODUCT#MATERIAL", "sentiment": "positive", "confidence": 0.7},
    ]

    result = compute_purchase_score(aspect_items, review_count=30)

    assert result["label"] == "Nen mua"
    assert result["score"] >= 0.68
    assert result["confidence"] == 1.0


def test_compute_purchase_score_avoids_when_negative_evidence_is_strong() -> None:
    aspect_items = [
        {"aspect": "PRODUCT#SAFETY", "sentiment": "negative", "confidence": 0.9},
        {"aspect": "SELLER#AUTHENTICITY", "sentiment": "negative", "confidence": 0.8},
    ]

    result = compute_purchase_score(aspect_items, review_count=10)

    assert result["label"] == "Nen tranh"
    assert result["score"] < 0.42


def test_select_evidence_filters_and_sorts_by_confidence() -> None:
    aspect_items = [
        {"review_id": "1", "aspect": "PRODUCT#SIZE", "sentiment": "negative", "confidence": 0.4, "sentence": "Hoi chat"},
        {"review_id": "2", "aspect": "PRODUCT#SIZE", "sentiment": "negative", "confidence": 0.9, "sentence": "Qua nho"},
        {"review_id": "3", "aspect": "PRODUCT#QUALITY", "sentiment": "positive", "confidence": 1.0, "sentence": "Tot"},
    ]

    evidence = select_evidence(aspect_items, sentiment="negative", limit=1)

    assert evidence == [
        {
            "review_id": "2",
            "aspect": "PRODUCT#SIZE",
            "sentiment": "negative",
            "confidence": 0.9,
            "text": "Qua nho",
        }
    ]
