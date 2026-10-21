from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.assistant.llm import LLMError


ALLOWED_INTENTS = {
    "SUMMARY",
    "PRODUCT_QA",
    "ASPECT_QA",
    "COMPLAINT_SUMMARY",
    "RISK_CHECK",
    "PRODUCT_FIT",
    "RECOMMENDATION",
    "PRICE_VALUE",
    "DELIVERY_SERVICE",
    "UNKNOWN",
}

ALLOWED_ASPECTS = {
    "PRODUCT#SAFETY",
    "PRODUCT#MATERIAL",
    "PRODUCT#SCENT",
    "PRODUCT#SIZE",
    "PRODUCT#ABSORPTION",
    "PRODUCT#FUNCTION",
    "PRODUCT#QUALITY",
    "PRODUCT#VALUE",
    "PRODUCT#DURABILITY",
    "PRODUCT#COMFORT",
    "PRICE#AFFORDABILITY",
    "DELIVERY#SPEED",
    "DELIVERY#PACKAGING",
    "DELIVERY#ACCURACY",
    "SELLER#AUTHENTICITY",
}


@dataclass(frozen=True)
class QueryAnalysis:
    intent: str = "UNKNOWN"
    target_aspects: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    source: str = "rule"

    def to_prompt_context(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "target_aspects": self.target_aspects,
            "constraints": self.constraints,
            "source": self.source,
        }


class QueryAnalyzer:
    def __init__(self, mapping_path: str | Path | None = None, llm: Any | None = None) -> None:
        self.mapping_path = Path(mapping_path or Path(__file__).with_name("aspect_mapping.json"))
        self.mapping = _load_mapping(self.mapping_path)
        self.llm = llm

    def analyze(self, question: str) -> QueryAnalysis:
        llm_result = self._analyze_by_llm(question)
        if llm_result:
            return llm_result
        normalized = normalize_text(question)
        rule_result = self._analyze_by_rule(normalized)
        return rule_result

    def _analyze_by_rule(self, normalized: str) -> QueryAnalysis:
        intent = _detect_intent(normalized)
        aspects: list[str] = []
        constraints: list[str] = []

        for phrase, config in self.mapping.items():
            if _mapping_phrase_in_text(phrase, normalized):
                for aspect in config.get("aspects") or []:
                    if aspect in ALLOWED_ASPECTS and aspect not in aspects:
                        aspects.append(aspect)
                for constraint in config.get("constraints") or []:
                    if constraint not in constraints:
                        constraints.append(str(constraint))

        if intent != "PRODUCT_FIT" and aspects and _is_aspect_question(normalized):
            intent = "ASPECT_QA"

        if intent == "PRICE_VALUE":
            aspects = _merge_aspects(aspects, ["PRICE#AFFORDABILITY", "PRODUCT#VALUE", "PRODUCT#QUALITY"])
        elif intent == "DELIVERY_SERVICE":
            aspects = _merge_aspects(aspects, ["DELIVERY#SPEED", "DELIVERY#PACKAGING", "DELIVERY#ACCURACY"])
        elif intent == "RISK_CHECK":
            aspects = _merge_aspects(aspects, ["PRODUCT#SAFETY", "PRODUCT#MATERIAL", "SELLER#AUTHENTICITY"])
        elif intent == "PRODUCT_FIT" and not aspects:
            aspects = ["PRODUCT#SAFETY", "PRODUCT#MATERIAL", "PRODUCT#SIZE"]
        elif intent in {"PRODUCT_QA", "COMPLAINT_SUMMARY"} and aspects:
            intent = "ASPECT_QA"

        return QueryAnalysis(
            intent=intent,
            target_aspects=aspects,
            constraints=constraints,
            source="rule",
        )

    def _analyze_by_llm(self, question: str) -> QueryAnalysis | None:
        if self.llm is None or not self.llm.is_configured():
            return None
        system_prompt = (
            "Ban la query analyzer cho chatbot review san pham Me & Be. "
            "Map cau hoi nguoi dung sang intent va aspect ABSA hop le. "
            "Chi tra ve JSON, khong giai thich, khong xuat <think>."
        )
        user_prompt = (
            f"Allowed intents: {sorted(ALLOWED_INTENTS)}\n"
            f"Allowed aspects: {sorted(ALLOWED_ASPECTS)}\n"
            "Schema: {\"intent\":\"\", \"target_aspects\":[], \"constraints\":[]}\n"
            f"Question: {question}"
        )
        try:
            raw = self.llm.generate_json(system_prompt, user_prompt)
        except LLMError:
            return None
        return _coerce_llm_result(raw)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9#\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_mapping(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {normalize_text(key): value for key, value in raw.items() if isinstance(value, dict)}


def _detect_intent(q: str) -> str:
    if _has_any_phrase(q, ["tom tat", "tong quan", "review", "danh gia chung"]):
        return "SUMMARY"
    if _has_any_phrase(q, ["co bi", "co gay", "co mui", "co mem", "co tham", "bi tran", "gay ham"]):
        return "ASPECT_QA"
    if _has_any_phrase(q, ["phan nan", "che", "khong hai long", "diem yeu", "te nhat"]):
        return "COMPLAINT_SUMMARY"
    if _has_any_phrase(q, ["luu y", "rui ro", "canh bao", "co van de", "nguy hiem"]):
        return "RISK_CHECK"
    if _has_any_phrase(q, ["co nen mua", "nen mua", "mua khong", "dang mua"]):
        return "RECOMMENDATION"
    if _has_any_phrase(q, ["phu hop", "hop voi", "dung cho", "da nhay cam", "tre so sinh", "em be"]):
        return "PRODUCT_FIT"
    if _has_any_phrase(q, ["dang tien", "gia ca", "gia tien", "muc gia", "re", "dat", "ngan sach"]) or _price_word_in_text(q):
        return "PRICE_VALUE"
    if _has_any_phrase(q, ["giao hang", "ship", "dong goi", "van chuyen"]):
        return "DELIVERY_SERVICE"
    if q:
        return "PRODUCT_QA"
    return "UNKNOWN"


def _is_aspect_question(q: str) -> bool:
    return any(
        phrase in q
        for phrase in [
            "co bi",
            "co gay",
            "co mui",
            "co mem",
            "co tham",
            "bi tran",
            "gay ham",
        ]
    )


def _phrase_in_text(phrase: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _mapping_phrase_in_text(phrase: str, text: str) -> bool:
    if phrase == "gia":
        return _price_word_in_text(text)
    return _phrase_in_text(phrase, text)


def _price_word_in_text(text: str) -> bool:
    without_review_phrase = re.sub(r"(?<![a-z0-9])danh\s+gia(?![a-z0-9])", " ", text)
    return _phrase_in_text("gia", without_review_phrase)


def _has_any_phrase(text: str, phrases: list[str]) -> bool:
    return any(_phrase_in_text(phrase, text) for phrase in phrases)


def _merge_aspects(current: list[str], extra: list[str]) -> list[str]:
    out = list(current)
    for aspect in extra:
        if aspect in ALLOWED_ASPECTS and aspect not in out:
            out.append(aspect)
    return out


def _coerce_llm_result(raw: dict[str, Any]) -> QueryAnalysis | None:
    intent = str(raw.get("intent") or "UNKNOWN").strip().upper()
    if intent not in ALLOWED_INTENTS:
        intent = "UNKNOWN"
    aspects = [
        str(aspect)
        for aspect in raw.get("target_aspects") or []
        if str(aspect) in ALLOWED_ASPECTS
    ]
    constraints_raw = raw.get("constraints") or []
    if isinstance(constraints_raw, dict):
        constraints = [f"{key}:{value}" for key, value in constraints_raw.items() if value]
    else:
        constraints = [str(item) for item in constraints_raw if item]
    return QueryAnalysis(
        intent=intent,
        target_aspects=list(dict.fromkeys(aspects)),
        constraints=list(dict.fromkeys(constraints)),
        source="llm",
    )
