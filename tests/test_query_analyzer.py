from app.assistant.query_analyzer import QueryAnalyzer


def test_rule_analyzer_does_not_treat_danh_gia_as_price() -> None:
    query = QueryAnalyzer(llm=None).analyze("Co bao nhieu danh gia tich cuc ve chat luong?")

    assert query.intent == "ASPECT_QA"
    assert "PRICE#AFFORDABILITY" not in query.target_aspects
    assert "PRODUCT#QUALITY" in query.target_aspects


def test_rule_analyzer_detects_delivery_without_price_noise() -> None:
    query = QueryAnalyzer(llm=None).analyze("Co bao nhieu danh gia tich cuc ve giao hang?")

    assert query.intent == "DELIVERY_SERVICE"
    assert "DELIVERY#SPEED" in query.target_aspects
    assert "PRICE#AFFORDABILITY" not in query.target_aspects
