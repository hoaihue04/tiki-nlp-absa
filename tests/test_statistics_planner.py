from app.assistant.query_analyzer import QueryAnalysis
from app.assistant.statistics_planner import StatisticsPlanner, coerce_statistics_plan, default_plan_for_query


def test_coerce_statistics_plan_rejects_unknown_metric() -> None:
    assert coerce_statistics_plan({"metric": "drop_reviews"}) is None


def test_coerce_statistics_plan_filters_values_and_clamps_limit() -> None:
    plan = coerce_statistics_plan({
        "metric": "top_aspects_by_sentiment",
        "sentiment": "negative",
        "target_aspects": ["PRODUCT#SCENT", "BAD#ASPECT"],
        "group_by": ["aspect", "table_name"],
        "sort_by": "bad_sort",
        "limit": 999,
        "include_examples": True,
    })

    assert plan is not None
    assert plan.sentiment == "negative"
    assert plan.target_aspects == ["PRODUCT#SCENT"]
    assert plan.group_by == ["aspect"]
    assert plan.sort_by == "count_desc"
    assert plan.limit == 20


def test_default_plan_for_complaint_summary_uses_negative_top_aspects() -> None:
    query = QueryAnalysis(intent="COMPLAINT_SUMMARY")
    plan = default_plan_for_query(query)

    assert plan.metric == "top_aspects_by_sentiment"
    assert plan.sentiment == "negative"
    assert plan.group_by == ["aspect"]


def test_fallback_plan_counts_positive_feedback_for_specific_aspect_question() -> None:
    query = QueryAnalysis(intent="ASPECT_QA", target_aspects=["PRODUCT#COMFORT"])
    plan = default_plan_for_query(query, "Bao nhieu nguoi danh gia tot ve do mem san pham?")

    assert plan.metric == "top_aspects_by_sentiment"
    assert plan.sentiment == "positive"
    assert plan.target_aspects == ["PRODUCT#COMFORT"]


def test_statistics_planner_uses_marker_for_aspect_question() -> None:
    planner = StatisticsPlanner()
    query = QueryAnalysis(intent="ASPECT_QA", target_aspects=["PRODUCT#COMFORT"])

    assert planner.should_plan("Bao nhieu nguoi danh gia tot ve do mem san pham?", query)
