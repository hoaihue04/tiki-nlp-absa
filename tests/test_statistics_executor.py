from app.assistant.statistics_executor import StatisticsQueryExecutor
from app.assistant.statistics_planner import StatisticsQueryPlan


class StubRepository:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.calls: list[tuple[str, str, int]] = []

    def get_negative_ratio_by_aspect(self, product_id: str, limit: int) -> list[dict]:
        self.calls.append(("negative_ratio", product_id, limit))
        return [
            {"aspect": "PRODUCT#SCENT", "negative_ratio": 0.75, "negative": 3, "total": 4},
            {"aspect": "PRODUCT#SIZE", "negative_ratio": 0.5, "negative": 2, "total": 4},
        ]

    def get_positive_ratio_by_aspect(self, product_id: str, limit: int) -> list[dict]:
        self.calls.append(("positive_ratio", product_id, limit))
        return [
            {"aspect": "PRODUCT#QUALITY", "positive_ratio": 0.9, "positive": 9, "total": 10},
        ]

    def get_review_statistics(self, product_id: str) -> dict:
        self.calls.append(("review_statistics", product_id, 0))
        return {"stored_review_count": 10, "aspect_record_count": 20, "avg_review_rating": 4.5}


def test_executor_returns_empty_when_repository_disabled() -> None:
    executor = StatisticsQueryExecutor(StubRepository(enabled=False))
    plan = StatisticsQueryPlan(metric="review_statistics")

    assert executor.execute("123", plan) == {}


def test_executor_filters_ratio_rows_by_target_aspects() -> None:
    repository = StubRepository()
    executor = StatisticsQueryExecutor(repository)
    plan = StatisticsQueryPlan(
        metric="negative_ratio_by_aspect",
        target_aspects=["PRODUCT#SCENT"],
        limit=5,
    )

    result = executor.execute("123", plan)

    assert repository.calls == [("negative_ratio", "123", 1000)]
    assert result["metric"] == "negative_ratio_by_aspect"
    assert result["rows"] == [
        {"aspect": "PRODUCT#SCENT", "negative_ratio": 0.75, "negative": 3, "total": 4}
    ]


def test_executor_dispatches_review_statistics() -> None:
    repository = StubRepository()
    executor = StatisticsQueryExecutor(repository)
    plan = StatisticsQueryPlan(metric="review_statistics")

    result = executor.execute("123", plan)

    assert repository.calls == [("review_statistics", "123", 0)]
    assert result["metric"] == "review_statistics"
    assert result["data"]["stored_review_count"] == 10
