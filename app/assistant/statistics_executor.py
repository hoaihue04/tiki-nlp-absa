from __future__ import annotations

from typing import Any

from sqlalchemy import desc, func

from app.assistant.db import session_scope
from app.assistant.models import AspectSentiment, Product, Review, RiskFlag
from app.assistant.statistics_planner import StatisticsQueryPlan


class StatisticsQueryExecutor:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def execute(self, tiki_product_id: str, plan: StatisticsQueryPlan | None) -> dict[str, Any]:
        if plan is None or not self.repository.enabled:
            return {}

        if plan.metric == "top_aspects_by_sentiment":
            rows = self._top_aspects_by_sentiment(tiki_product_id, plan)
        elif plan.metric == "aspect_sentiment_distribution":
            rows = self._aspect_sentiment_distribution(tiki_product_id, plan)
        elif plan.metric == "negative_ratio_by_aspect":
            rows = self.repository.get_negative_ratio_by_aspect(tiki_product_id, limit=1000)
            rows = _filter_rows_by_aspect(rows, plan.target_aspects)
            rows = rows[:plan.limit]
        elif plan.metric == "positive_ratio_by_aspect":
            rows = self.repository.get_positive_ratio_by_aspect(tiki_product_id, limit=1000)
            rows = _filter_rows_by_aspect(rows, plan.target_aspects)
            rows = rows[:plan.limit]
        elif plan.metric == "review_statistics":
            return {
                "plan": plan.to_prompt_context(),
                "metric": plan.metric,
                "data": self.repository.get_review_statistics(tiki_product_id),
            }
        elif plan.metric == "risk_summary":
            rows = self._risk_summary(tiki_product_id, plan)
        elif plan.metric == "evidence_by_aspect":
            rows = self._evidence_by_aspect(tiki_product_id, plan)
        elif plan.metric == "rating_sentiment_summary":
            rows = self._rating_sentiment_summary(tiki_product_id, plan)
        else:
            rows = []

        return {
            "plan": plan.to_prompt_context(),
            "metric": plan.metric,
            "rows": rows,
        }

    def _get_product_db_id(self, session: Any, tiki_product_id: str) -> int | None:
        product_id = (
            session.query(Product.id)
            .filter(Product.tiki_product_id == str(tiki_product_id))
            .scalar()
        )
        return int(product_id) if product_id is not None else None

    def _top_aspects_by_sentiment(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            query = (
                session.query(
                    AspectSentiment.category.label("aspect"),
                    func.count(AspectSentiment.id).label("total"),
                    func.avg(AspectSentiment.confidence).label("avg_confidence"),
                )
                .filter(AspectSentiment.product_id == product_db_id)
            )
            query = _apply_aspect_sentiment_filters(query, plan)

            rows = (
                query.group_by(AspectSentiment.category)
                .order_by(desc("total"))
                .limit(plan.limit)
                .all()
            )

        return [
            {
                "aspect": row.aspect,
                "sentiment": plan.sentiment or "",
                "total": int(row.total or 0),
                "avg_confidence": round(float(row.avg_confidence or 0.0), 4),
            }
            for row in rows
        ]

    def _aspect_sentiment_distribution(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            query = (
                session.query(
                    AspectSentiment.category.label("aspect"),
                    AspectSentiment.polarity.label("sentiment"),
                    func.count(AspectSentiment.id).label("total"),
                    func.avg(AspectSentiment.confidence).label("avg_confidence"),
                )
                .filter(AspectSentiment.product_id == product_db_id)
            )
            query = _apply_aspect_sentiment_filters(query, plan)

            rows = (
                query.group_by(AspectSentiment.category, AspectSentiment.polarity)
                .order_by(desc("total"))
                .limit(plan.limit * 3)
                .all()
            )

        return [
            {
                "aspect": row.aspect,
                "sentiment": row.sentiment,
                "total": int(row.total or 0),
                "avg_confidence": round(float(row.avg_confidence or 0.0), 4),
            }
            for row in rows
        ]

    def _risk_summary(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            rows = (
                session.query(
                    RiskFlag.risk_type,
                    RiskFlag.severity,
                    func.count(RiskFlag.id).label("total"),
                    func.avg(RiskFlag.confidence).label("avg_confidence"),
                )
                .filter(RiskFlag.product_id == product_db_id)
                .group_by(RiskFlag.risk_type, RiskFlag.severity)
                .order_by(desc("total"))
                .limit(plan.limit)
                .all()
            )

        return [
            {
                "risk_type": row.risk_type,
                "severity": row.severity,
                "total": int(row.total or 0),
                "avg_confidence": round(float(row.avg_confidence or 0.0), 4),
            }
            for row in rows
        ]

    def _evidence_by_aspect(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            query = (
                session.query(
                    AspectSentiment.category.label("aspect"),
                    AspectSentiment.polarity.label("sentiment"),
                    AspectSentiment.confidence,
                    AspectSentiment.sentence,
                    Review.tiki_review_id,
                    Review.rating,
                )
                .outerjoin(Review, Review.id == AspectSentiment.review_id)
                .filter(AspectSentiment.product_id == product_db_id)
            )
            query = _apply_aspect_sentiment_filters(query, plan)

            rows = (
                query.order_by(AspectSentiment.confidence.desc())
                .limit(plan.limit)
                .all()
            )

        return [
            {
                "aspect": row.aspect,
                "sentiment": row.sentiment,
                "confidence": round(float(row.confidence or 0.0), 4),
                "text": row.sentence,
                "review_id": row.tiki_review_id or "",
                "rating": row.rating,
            }
            for row in rows
        ]

    def _rating_sentiment_summary(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            query = (
                session.query(
                    Review.rating,
                    AspectSentiment.polarity.label("sentiment"),
                    func.count(AspectSentiment.id).label("total"),
                )
                .join(Review, Review.id == AspectSentiment.review_id)
                .filter(
                    AspectSentiment.product_id == product_db_id,
                    Review.rating.isnot(None),
                )
            )
            query = _apply_aspect_sentiment_filters(query, plan)

            rows = (
                query.group_by(Review.rating, AspectSentiment.polarity)
                .order_by(Review.rating.desc(), desc("total"))
                .limit(plan.limit * 3)
                .all()
            )

        return [
            {
                "rating": int(row.rating or 0),
                "sentiment": row.sentiment,
                "total": int(row.total or 0),
            }
            for row in rows
        ]


def _apply_aspect_sentiment_filters(query: Any, plan: StatisticsQueryPlan) -> Any:
    if plan.sentiment:
        query = query.filter(AspectSentiment.polarity == plan.sentiment)
    if plan.target_aspects:
        query = query.filter(AspectSentiment.category.in_(plan.target_aspects))
    return query


def _filter_rows_by_aspect(rows: list[dict[str, Any]], target_aspects: list[str]) -> list[dict[str, Any]]:
    if not target_aspects:
        return rows
    allowed = set(target_aspects)
    return [row for row in rows if str(row.get("aspect") or "") in allowed]
