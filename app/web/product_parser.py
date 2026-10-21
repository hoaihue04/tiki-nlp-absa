from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = str(val).strip()
        return float(v) if v and v not in ("nan", "", "None") else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        v = str(val).strip()
        return int(float(v)) if v and v not in ("nan", "", "None") else default
    except (ValueError, TypeError):
        return default


def _safe_str(val: Any, default: str = "") -> str:
    v = str(val).strip()
    return v if v and v not in ("nan", "None") else default


def extract_product_id_from_url(product_url: str) -> Optional[str]:
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(product_url)
    m = re.search(r"-p(\d+)\.html", parsed.path)
    if m:
        return m.group(1)
    query = parse_qs(parsed.query)
    for key in ("spid", "product_id", "id"):
        if key in query and query[key]:
            return str(query[key][0])
    m2 = re.search(r"(\d+)", parsed.path)
    return m2.group(1) if m2 else None


def split_sentences(text: str) -> List[str]:
    text = _safe_str(text).strip()
    if not text:
        return []
    try:
        from underthesea import sent_tokenize
        sents = sent_tokenize(text)
        sents = [s.strip() for s in sents if s and s.strip()]
        if sents:
            return sents
    except Exception:
        pass
    parts = re.split(r"[\n\.!?;]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _sentiment_label_from_score(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def _advice_from_score(score: float) -> Dict[str, str]:
    if score >= 0.6:
        return {"label": "Nên mua", "tone": "positive"}
    if score >= 0.3:
        return {"label": "Cân nhắc", "tone": "neutral"}
    return {"label": "Thận trọng", "tone": "negative"}


