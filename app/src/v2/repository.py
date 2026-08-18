from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import func

from app.src.v2.db import get_database_init_error, init_db, is_database_available, is_database_configured, session_scope


class V2Repository:
    def __init__(self) -> None:
        self.enabled = is_database_available()
        self.configured = is_database_configured()
        if self.enabled:
            self.enabled = init_db()

    @property
    def status(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "enabled": self.enabled,
            "error": get_database_init_error(),
        }

    def persist_analysis(
        self,
        product_info: dict[str, Any],
        reviews: list[dict[str, Any]],
        aspect_items: list[dict[str, Any]],
        risks: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        if not self.enabled:
            return False

        from app.src.v2.models import AspectSentiment, Product, Review, RiskFlag

        tiki_product_id = str(product_info.get("product_id") or "").strip()
        if not tiki_product_id:
            return False

        with session_scope() as session:
            product = session.query(Product).filter(Product.tiki_product_id == tiki_product_id).one_or_none()
            if product is None:
                product = Product(tiki_product_id=tiki_product_id)
                session.add(product)
                session.flush()

            product.url = str(product_info.get("product_url") or "")
            product.name = str(product_info.get("name") or "")
            product.brand = str(product_info.get("brand_name") or product_info.get("brand") or "")
            product.seller = str(product_info.get("seller_name") or product_info.get("seller") or "")
            product.seller_is_official = bool(product_info.get("seller_is_official") or False)
            product.price = _to_float_or_none(product_info.get("price"))
            product.original_price = _to_float_or_none(product_info.get("original_price"))
            product.rating_average = _to_float_or_none(product_info.get("rating_average"))
            product.review_count = int(_to_float_or_none(product_info.get("review_count")) or 0)
            product.description = str(product_info.get("short_description") or product_info.get("description") or "")
            product.image_urls = list(product_info.get("images") or [])
            product.category_lv1 = str(product_info.get("category_lv1") or "")
            product.category_lv2 = str(product_info.get("category_lv2") or "")
            product.category_lv3 = str(product_info.get("category_lv3") or "")
            product.raw_payload = dict(product_info)
            session.flush()

            review_by_tiki_id: dict[str, Review] = {}
            for row in reviews:
                tiki_review_id = str(row.get("review_id") or "").strip()
                if not tiki_review_id:
                    continue
                review = (
                    session.query(Review)
                    .filter(Review.product_id == product.id, Review.tiki_review_id == tiki_review_id)
                    .one_or_none()
                )
                if review is None:
                    review = Review(product_id=product.id, tiki_review_id=tiki_review_id)
                    session.add(review)
                    session.flush()
                review.rating = int(_to_float_or_none(row.get("rating")) or 0) or None
                review.title = str(row.get("title") or "")
                review.content = str(row.get("content") or row.get("full_text") or "")
                review.cleaned_content = str(row.get("clean_text") or row.get("cleaned_content") or "")
                review.created_by = str(row.get("customer_name") or row.get("created_by") or "")
                review.helpful_count = int(_to_float_or_none(row.get("helpful_count")) or 0)
                review.is_verified = bool(row.get("is_verified") or False)
                review.created_at_tiki = str(row.get("created_at_ts") or row.get("created_at_tiki") or "")
                review.raw_payload = _json_safe_dict(row)
                review_by_tiki_id[tiki_review_id] = review

            session.query(AspectSentiment).filter(AspectSentiment.product_id == product.id).delete()
            for item in aspect_items:
                tiki_review_id = str(item.get("review_id") or "")
                review = review_by_tiki_id.get(tiki_review_id)
                session.add(AspectSentiment(
                    review_id=review.id if review is not None else None,
                    product_id=product.id,
                    category=str(item.get("aspect") or item.get("category") or ""),
                    polarity=str(item.get("sentiment") or item.get("polarity") or "neutral"),
                    confidence=float(item.get("confidence") or 0.0),
                    sentence=str(item.get("sentence") or ""),
                    model_name="phobert",
                    model_version="best_model.pt",
                ))

            session.query(RiskFlag).filter(RiskFlag.product_id == product.id).delete()
            for risk in risks or []:
                review = review_by_tiki_id.get(str(risk.get("review_id") or ""))
                session.add(RiskFlag(
                    product_id=product.id,
                    review_id=review.id if review is not None else None,
                    risk_type=str(risk.get("risk_type") or ""),
                    severity=str(risk.get("severity") or "low"),
                    confidence=float(risk.get("confidence") or 0.0),
                    evidence=str(risk.get("evidence") or ""),
                    source=str(risk.get("source") or "phobert_rules"),
                ))
        return True

    def load_product_context(self, tiki_product_id: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None

        from app.src.v2.models import AspectSentiment, Product, Review, RiskFlag

        with session_scope() as session:
            product = session.query(Product).filter(Product.tiki_product_id == str(tiki_product_id)).one_or_none()
            if product is None:
                return None
            reviews = session.query(Review).filter(Review.product_id == product.id).all()
            review_by_id = {row.id: row for row in reviews}
            aspects = session.query(AspectSentiment).filter(AspectSentiment.product_id == product.id).all()
            risks = session.query(RiskFlag).filter(RiskFlag.product_id == product.id).all()
            return {
                "product": {
                    "product_id": product.tiki_product_id,
                    "name": product.name,
                    "brand_name": product.brand,
                    "seller_name": product.seller,
                    "price": product.price,
                    "rating_average": product.rating_average,
                    "review_count": product.review_count,
                    "short_description": product.description,
                    "product_url": product.url,
                },
                "reviews": [
                    {
                        "review_id": review.tiki_review_id,
                        "rating": review.rating,
                        "content": review.content,
                        "clean_text": review.cleaned_content,
                    }
                    for review in reviews
                ],
                "aspects": [
                    {
                        "review_id": review_by_id.get(item.review_id).tiki_review_id if item.review_id in review_by_id else "",
                        "aspect": item.category,
                        "sentiment": item.polarity,
                        "confidence": item.confidence,
                        "sentence": item.sentence,
                    }
                    for item in aspects
                ],
                "risks": [
                    {
                        "risk_type": risk.risk_type,
                        "severity": risk.severity,
                        "confidence": risk.confidence,
                        "evidence": risk.evidence,
                        "source": risk.source,
                    }
                    for risk in risks
                ],
            }

    def get_product_analytics(self, tiki_product_id: str, limit: int = 8) -> dict[str, Any]:
        if not self.enabled:
            return {}

        stats = self.get_aspect_sentiment_stats(tiki_product_id, limit=limit)
        return {
            "aspect_sentiment_stats": stats,
            "top_complaints": self.get_top_complaints(tiki_product_id, limit=limit),
            "top_praises": self.get_top_praises(tiki_product_id, limit=limit),
            "negative_ratio_by_aspect": self.get_negative_ratio_by_aspect(tiki_product_id, limit=limit),
            "positive_ratio_by_aspect": self.get_positive_ratio_by_aspect(tiki_product_id, limit=limit),
            "aspect_distribution": self.get_aspect_distribution(tiki_product_id, limit=limit),
            "review_statistics": self.get_review_statistics(tiki_product_id),
        }

    def get_aspect_sentiment_stats(self, tiki_product_id: str, limit: int = 8) -> list[dict[str, Any]]:
        product_db_id = self._get_product_db_id(tiki_product_id)
        if product_db_id is None:
            return []

        from app.src.v2.models import AspectSentiment

        with session_scope() as session:
            rows = (
                session.query(
                    AspectSentiment.category,
                    AspectSentiment.polarity,
                    func.count(AspectSentiment.id).label("total"),
                    func.avg(AspectSentiment.confidence).label("avg_confidence"),
                )
                .filter(AspectSentiment.product_id == product_db_id)
                .group_by(AspectSentiment.category, AspectSentiment.polarity)
                .all()
            )

        grouped: dict[str, dict[str, Any]] = {}
        for category, polarity, total, avg_confidence in rows:
            aspect = str(category or "")
            sentiment = str(polarity or "neutral")
            bucket = grouped.setdefault(
                aspect,
                {
                    "aspect": aspect,
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "total": 0,
                    "avg_confidence": 0.0,
                },
            )
            count = int(total or 0)
            if sentiment not in ("positive", "neutral", "negative"):
                sentiment = "neutral"
            bucket[sentiment] += count
            bucket["total"] += count
            bucket["avg_confidence"] = max(bucket["avg_confidence"], float(avg_confidence or 0.0))

        out = []
        for row in grouped.values():
            total = row["total"] or 1
            row["positive_ratio"] = round(row["positive"] / total, 4)
            row["negative_ratio"] = round(row["negative"] / total, 4)
            row["avg_confidence"] = round(row["avg_confidence"], 4)
            out.append(row)
        out.sort(key=lambda row: row["total"], reverse=True)
        return out[:limit]

    def get_top_complaints(self, tiki_product_id: str, limit: int = 6) -> list[dict[str, Any]]:
        return self._get_top_by_sentiment(tiki_product_id, "negative", limit=limit)

    def get_top_praises(self, tiki_product_id: str, limit: int = 6) -> list[dict[str, Any]]:
        return self._get_top_by_sentiment(tiki_product_id, "positive", limit=limit)

    def get_negative_ratio_by_aspect(self, tiki_product_id: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.get_aspect_sentiment_stats(tiki_product_id, limit=1000)
        rows.sort(key=lambda row: (row["negative_ratio"], row["negative"], row["total"]), reverse=True)
        return rows[:limit]

    def get_positive_ratio_by_aspect(self, tiki_product_id: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.get_aspect_sentiment_stats(tiki_product_id, limit=1000)
        rows.sort(key=lambda row: (row["positive_ratio"], row["positive"], row["total"]), reverse=True)
        return rows[:limit]

    def get_aspect_distribution(self, tiki_product_id: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.get_aspect_sentiment_stats(tiki_product_id, limit=1000)
        rows.sort(key=lambda row: row["total"], reverse=True)
        return rows[:limit]

    def get_review_statistics(self, tiki_product_id: str) -> dict[str, Any]:
        product_db_id = self._get_product_db_id(tiki_product_id)
        if product_db_id is None:
            return {}

        from app.src.v2.models import AspectSentiment, Review

        with session_scope() as session:
            review_count = (
                session.query(func.count(Review.id))
                .filter(Review.product_id == product_db_id)
                .scalar()
            )
            aspect_count = (
                session.query(func.count(AspectSentiment.id))
                .filter(AspectSentiment.product_id == product_db_id)
                .scalar()
            )
            avg_rating = (
                session.query(func.avg(Review.rating))
                .filter(Review.product_id == product_db_id, Review.rating.isnot(None))
                .scalar()
            )

        return {
            "stored_review_count": int(review_count or 0),
            "aspect_record_count": int(aspect_count or 0),
            "avg_review_rating": round(float(avg_rating or 0.0), 4) if avg_rating is not None else None,
        }

    def _get_top_by_sentiment(self, tiki_product_id: str, sentiment: str, limit: int) -> list[dict[str, Any]]:
        product_db_id = self._get_product_db_id(tiki_product_id)
        if product_db_id is None:
            return []

        from app.src.v2.models import AspectSentiment

        with session_scope() as session:
            rows = (
                session.query(
                    AspectSentiment.category,
                    func.count(AspectSentiment.id).label("total"),
                    func.avg(AspectSentiment.confidence).label("avg_confidence"),
                )
                .filter(AspectSentiment.product_id == product_db_id, AspectSentiment.polarity == sentiment)
                .group_by(AspectSentiment.category)
                .order_by(func.count(AspectSentiment.id).desc())
                .limit(limit)
                .all()
            )
        return [
            {
                "aspect": str(category or ""),
                "sentiment": sentiment,
                "total": int(total or 0),
                "avg_confidence": round(float(avg_confidence or 0.0), 4),
            }
            for category, total, avg_confidence in rows
        ]

    def _get_product_db_id(self, tiki_product_id: str) -> Optional[int]:
        if not self.enabled:
            return None

        from app.src.v2.models import Product

        with session_scope() as session:
            product_id = (
                session.query(Product.id)
                .filter(Product.tiki_product_id == str(tiki_product_id))
                .scalar()
            )
        return int(product_id) if product_id is not None else None


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _json_safe_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    if not isinstance(row, dict):
        return {}
    out = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                value = str(value)
        out[str(key)] = value
    return out
