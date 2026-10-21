from __future__ import annotations

import math
from typing import Any

from app.assistant.config import settings


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, enabled: bool | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self.enabled = settings.reranker_enabled if enabled is None else enabled
        self._model = None

    def rerank(self, question: str, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if not self.enabled or not rows:
            return rows[:limit]

        pairs: list[tuple[str, str]] = []
        candidates: list[dict[str, Any]] = []
        for row in rows:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            candidates.append(row)
            pairs.append((question, text))

        if not pairs:
            return rows[:limit]

        try:
            model = self._load_model()
            raw_scores = model.predict(pairs, show_progress_bar=False)
        except Exception:
            self.enabled = False
            return rows[:limit]

        scored: list[dict[str, Any]] = []
        for row, score in zip(candidates, raw_scores):
            value = _to_float(score)
            scored.append({
                **row,
                "reranker_score": _sigmoid(value),
                "reranker_raw_score": value,
            })
        scored.sort(key=lambda item: item.get("reranker_score", 0.0), reverse=True)
        return scored[:limit]

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model


def _to_float(value: Any) -> float:
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return 0.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)
