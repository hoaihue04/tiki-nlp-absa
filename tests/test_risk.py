from __future__ import annotations

from app.assistant.risk import detect_risks


def test_detect_risks_groups_negative_safety_evidence() -> None:
    aspect_items = [
        {
            "review_id": "r1",
            "aspect": "PRODUCT#SAFETY",
            "sentiment": "negative",
            "confidence": 0.9,
            "sentence": "Mui hoa chat, khong an toan cho be",
        },
        {
            "review_id": "r2",
            "aspect": "PRODUCT#QUALITY",
            "sentiment": "positive",
            "confidence": 0.9,
            "sentence": "San pham tot",
        },
    ]

    risks = detect_risks(aspect_items)

    assert len(risks) == 1
    assert risks[0]["risk_type"] == "safety"
    assert risks[0]["severity"] == "medium"
    assert risks[0]["review_id"] == "r1"


def test_detect_risks_ignores_non_risk_or_non_negative_items() -> None:
    aspect_items = [
        {"aspect": "PRODUCT#SAFETY", "sentiment": "positive", "sentence": "An toan"},
        {"aspect": "PRODUCT#QUALITY", "sentiment": "negative", "sentence": "Khong tot"},
    ]

    assert detect_risks(aspect_items) == []
