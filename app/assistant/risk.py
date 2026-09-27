from __future__ import annotations

from collections import defaultdict
from typing import Any


RISK_ASPECTS = {
    "PRODUCT#SAFETY": ("safety", 3),
    "SELLER#AUTHENTICITY": ("authenticity", 3),
    "PRODUCT#MATERIAL": ("material", 2),
    "PRODUCT#SIZE": ("size_fit", 2),
    "PRODUCT#DURABILITY": ("durability", 2),
    "DELIVERY#PACKAGING": ("delivery_packaging", 1),
    "DELIVERY#ACCURACY": ("delivery_accuracy", 1),
}

RISK_KEYWORDS = {
    "safety": ["kich ung", "di ung", "nguy hiem", "doc", "mui hoa chat", "khong an toan"],
    "authenticity": ["fake", "gia", "nhai", "khong chinh hang", "tem"],
    "material": ["cung", "thon", "nong", "bi xu", "mui", "chat lieu kem"],
    "size_fit": ["chat", "nho", "rong", "khong vua", "sai size"],
    "durability": ["hong", "rach", "vo", "ro ri", "bung", "dut"],
    "delivery_packaging": ["vo hop", "mop", "rach hop", "dong goi te"],
    "delivery_accuracy": ["giao sai", "thieu hang", "nham hang"],
}


def detect_risks(aspect_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in aspect_items:
        aspect = str(item.get("aspect") or item.get("category") or "")
        sentiment = str(item.get("sentiment") or item.get("polarity") or "neutral")
        if sentiment != "negative" or aspect not in RISK_ASPECTS:
            continue
        risk_type, base_weight = RISK_ASPECTS[aspect]
        sentence = str(item.get("sentence") or "").lower()
        keyword_boost = 1 if any(keyword in sentence for keyword in RISK_KEYWORDS.get(risk_type, [])) else 0
        buckets[risk_type].append({**item, "_risk_weight": base_weight + keyword_boost})

    risks = []
    for risk_type, items in buckets.items():
        total = sum(float(item.get("confidence") or 0.5) * float(item.get("_risk_weight") or 1) for item in items)
        count = len(items)
        if total >= 5 or count >= 4:
            severity = "high"
        elif total >= 2.5 or count >= 2:
            severity = "medium"
        else:
            severity = "low"
        best = sorted(items, key=lambda x: float(x.get("confidence") or 0), reverse=True)[0]
        risks.append({
            "risk_type": risk_type,
            "severity": severity,
            "confidence": round(min(1.0, total / 8.0), 4),
            "evidence": str(best.get("sentence") or "")[:350],
            "review_id": str(best.get("review_id") or ""),
            "source": "phobert_rules",
        })

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    risks.sort(key=lambda row: (severity_rank.get(row["severity"], 0), row["confidence"]), reverse=True)
    return risks

