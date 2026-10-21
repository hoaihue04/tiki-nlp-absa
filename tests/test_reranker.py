from app.assistant.reranker import CrossEncoderReranker
from app.assistant.retrieval import _score_row
from app.assistant.query_analyzer import QueryAnalysis


class FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        del show_progress_bar
        return [0.0 if "less relevant" in text else 4.0 for _, text in pairs]


def test_cross_encoder_reranker_orders_rows_by_model_score() -> None:
    reranker = CrossEncoderReranker(enabled=True)
    reranker._model = FakeCrossEncoder()
    rows = [
        {"text": "less relevant evidence", "semantic_score": 0.95},
        {"text": "highly relevant evidence", "semantic_score": 0.5},
    ]

    ranked = reranker.rerank("question", rows, limit=2)

    assert ranked[0]["text"] == "highly relevant evidence"
    assert ranked[0]["reranker_score"] > ranked[1]["reranker_score"]


def test_hybrid_score_prefers_reranker_score_when_available() -> None:
    query = QueryAnalysis(intent="ASPECT_QA", target_aspects=["PRODUCT#COMFORT"])
    row = {
        "source": "qdrant",
        "semantic_score": 0.1,
        "reranker_score": 0.9,
        "confidence": 0.8,
        "aspect_code": "PRODUCT#COMFORT",
        "sentiment": "positive",
    }

    score = _score_row(row, query, qdrant_weight=0.65, structured_weight=0.35)

    assert score > 1.2
