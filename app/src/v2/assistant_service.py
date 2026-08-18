from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.src.v2.config import settings
from app.src.v2.llm import ChatLLMClient, LLMError
from app.src.v2.query_analyzer import QueryAnalysis, QueryAnalyzer
from app.src.v2.repository import V2Repository
from app.src.v2.retrieval import HybridRetrievalService
from app.src.v2.risk import detect_risks
from app.src.v2.scoring import compute_purchase_score, select_evidence


ASPECT_LABELS = {
    "PRODUCT#SAFETY": "Độ an toàn",
    "PRODUCT#MATERIAL": "Chất liệu",
    "PRODUCT#SCENT": "Mùi sản phẩm",
    "PRODUCT#SIZE": "Kích thước",
    "PRODUCT#ABSORPTION": "Khả năng thấm hút",
    "PRODUCT#FUNCTION": "Công năng sử dụng",
    "PRODUCT#QUALITY": "Chất lượng",
    "PRODUCT#VALUE": "Độ đáng tiền",
    "PRODUCT#DURABILITY": "Độ bền",
    "PRODUCT#COMFORT": "Độ thoải mái",
    "PRICE#AFFORDABILITY": "Giá cả",
    "DELIVERY#SPEED": "Giao hàng",
    "DELIVERY#PACKAGING": "Bao bì",
    "DELIVERY#ACCURACY": "Giao đúng/đủ hàng",
    "SELLER#AUTHENTICITY": "Tính chính hãng",
}

RISK_LABELS = {
    "safety": "Độ an toàn",
    "authenticity": "Tính chính hãng",
    "material": "Chất liệu",
    "size_fit": "Kích thước",
    "durability": "Độ bền",
    "delivery_packaging": "Bao bì",
    "delivery_accuracy": "Giao đúng/đủ hàng",
}

BABY_FIT_ASPECTS = {
    "PRODUCT#SAFETY",
    "PRODUCT#MATERIAL",
    "PRODUCT#SCENT",
    "PRODUCT#SIZE",
    "PRODUCT#ABSORPTION",
    "PRODUCT#FUNCTION",
    "PRODUCT#COMFORT",
}

BABY_FIT_RISKS = {"safety", "material", "size_fit"}


class ParentAssistantService:
    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()
        self.llm = ChatLLMClient()
        self.query_analyzer = QueryAnalyzer(llm=self.llm)
        self.retrieval = HybridRetrievalService()

    def build_from_analysis(self, analysis: dict[str, Any], use_llm: bool = True) -> dict[str, Any]:
        product = analysis.get("product_info") or {}
        aspects = _aspect_items_from_analysis(analysis)
        review_count = int((analysis.get("metrics") or {}).get("total_reviews_used") or product.get("review_count") or 0)
        score = compute_purchase_score(aspects, review_count=review_count)
        risks = analysis.get("risk_flags") or detect_risks(aspects)
        positive = select_evidence(aspects, "positive", limit=3)
        negative = select_evidence(aspects, "negative", limit=3)
        evidence = (negative + positive)[:5]

        structured = _fallback_structured_answer(product, "SUMMARY", score, risks, evidence, stats=None)
        llm_used = False
        if use_llm:
            llm_structured = self._try_llm_purchase_summary(product, score, risks, evidence)
            if llm_structured:
                structured = llm_structured
                llm_used = True

        return {
            "product_id": str(product.get("product_id") or ""),
            "conclusion": score["label"],
            "score": score["score"],
            "confidence": score["confidence"],
            "summary": _format_structured_answer(structured),
            "structured_answer": structured,
            "reasons_to_buy": [_evidence_to_reason(row) for row in positive[:3]],
            "cautions": [_risk_to_text(row) for row in risks[:3]] or [_evidence_to_reason(row) for row in negative[:3]],
            "evidence": evidence,
            "llm_used": llm_used,
            "llm_model": self.llm.model if llm_used else "",
        }

    def build_from_db(self, product_id: str, use_llm: bool = True) -> Optional[dict[str, Any]]:
        ctx = self.repository.load_product_context(product_id)
        if not ctx:
            return None
        analysis = {
            "product_info": ctx["product"],
            "metrics": {"total_reviews_used": len(ctx.get("reviews") or [])},
            "opinion_table": [
                {
                    "aspect": item["aspect"],
                    "sentiment": item["sentiment"],
                    "confidence": item["confidence"],
                    "example": item["sentence"],
                }
                for item in ctx.get("aspects") or []
            ],
            "risk_flags": ctx.get("risks") or [],
        }
        return self.build_from_analysis(analysis, use_llm=use_llm)

    def answer_chat(self, product_id: str, question: str, use_llm: bool = True) -> Optional[dict[str, Any]]:
        ctx = self.repository.load_product_context(product_id)
        if not ctx:
            return None
        aspects = ctx.get("aspects") or []
        risks = ctx.get("risks") or detect_risks(aspects)
        query = self.query_analyzer.analyze(question)
        analytics = self.repository.get_product_analytics(product_id)
        stats = _stats_from_analytics(analytics) if analytics else _aspect_statistics(aspects)
        scoped_stats = _scope_stats_for_query(stats, query)
        scoped_risks = _scope_risks_for_query(risks, query)
        scoped_analytics = _scope_analytics_for_query(analytics or {}, query)
        structured_evidence = _retrieve_for_query(aspects, risks, query, limit=6)
        retrieval_result = self.retrieval.retrieve(
            product_id=product_id,
            question=question,
            query=query,
            structured_evidence=structured_evidence,
            limit=6,
        )
        evidence = retrieval_result.get("evidence") or structured_evidence
        product = ctx.get("product") or {}
        fallback = _fallback_structured_answer(product, query.intent, None, scoped_risks, evidence, scoped_stats, scoped_analytics)
        llm_used = False
        structured = fallback
        if use_llm:
            llm_answer = self._try_llm_chat(product, question, evidence, query, scoped_stats, scoped_risks, scoped_analytics)
            if llm_answer:
                structured = llm_answer
                llm_used = True
        return {
            "product_id": product_id,
            "answer": _format_structured_answer(structured, intent=query.intent),
            "structured_answer": structured,
            "evidence": evidence,
            "retrieval_strategy": retrieval_result.get("strategy", "postgres_only"),
            "qdrant_used": bool(retrieval_result.get("qdrant_used")),
            "cache_hit": bool(retrieval_result.get("cache_hit")),
            "llm_used": llm_used,
            "llm_model": self.llm.model if llm_used else "",
        }

    def _try_llm_purchase_summary(
        self,
        product: dict[str, Any],
        score: dict[str, Any],
        risks: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.llm.is_configured():
            return {}
        system_prompt = _shopping_assistant_system_prompt()
        user_prompt = (
            _json_response_contract()
            + "\nCONTEXT_JSON:\n"
            + json.dumps({
                "product_info": _compact_product(product),
                "shopping_signal": {
                    "conclusion": score["label"],
                    "score": score["score"],
                    "confidence": score["confidence"],
                },
                "risk_summary": [_friendly_risk_row(row) for row in risks[:3]],
                "representative_opinions": [_opinion_summary_row(row) for row in evidence[:5]],
            }, ensure_ascii=False, indent=2)
        )
        try:
            return _sanitize_structured_answer(self.llm.generate_json(system_prompt, user_prompt), evidence)
        except LLMError:
            return {}

    def _try_llm_chat(
        self,
        product: dict[str, Any],
        question: str,
        evidence: list[dict[str, Any]],
        query: QueryAnalysis,
        stats: dict[str, Any],
        risks: list[dict[str, Any]],
        analytics: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.llm.is_configured():
            return {}
        system_prompt = _shopping_assistant_system_prompt()
        user_prompt = (
            _build_chat_prompt(product, question, evidence, query, stats, risks, analytics)
        )
        try:
            return _sanitize_structured_answer(self.llm.generate_json(system_prompt, user_prompt), evidence)
        except LLMError:
            return {}


def _aspect_items_from_analysis(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in analysis.get("opinion_table") or []:
        rows.append({
            "review_id": "",
            "aspect": row.get("aspect"),
            "sentiment": row.get("sentiment"),
            "confidence": row.get("confidence"),
            "sentence": row.get("example") or "",
        })
    for group_name in ("strengths", "weaknesses"):
        for row in ((analysis.get("aspect") or {}).get(group_name) or []):
            sentiment = "positive" if group_name == "strengths" else "negative"
            rows.append({
                "review_id": "",
                "aspect": row.get("aspect"),
                "sentiment": sentiment,
                "confidence": 0.8,
                "sentence": row.get("best_positive_quote") or row.get("best_negative_quote") or "",
            })
    return rows


def _build_chat_prompt(
    product: dict[str, Any],
    question: str,
    evidence: list[dict[str, Any]],
    query: QueryAnalysis,
    stats: dict[str, Any],
    risks: list[dict[str, Any]],
    analytics: dict[str, Any] | None = None,
) -> str:
    context = {
        "product_info": _compact_product(product),
        "question": question,
        "intent": query.intent,
        "question_focus": _friendly_query_context(query),
        "absa_statistics": _compact_stats(stats),
        "sql_analytics": _compact_analytics(analytics or {}),
        "risk_summary": [_friendly_risk_row(row) for row in risks[:3]],
        "representative_opinions": [_opinion_summary_row(row) for row in evidence[:6]],
    }

    if query.intent == "SUMMARY":
        instruction = (
            "Tra loi nhu mot tro ly mua sam, toi da 150 tu, khong viet kieu report. "
            "Gom 3 doan ngan: tong quan diem duoc khen, diem can luu y, va ket luan mua hang. "
            "Khong chep nguyen van review dai, khong xuat aspect code, khong dung bullet neu khong can. "
            "Co the nhac bang chung bang so luong/tong quan, vi du 'nhieu review ve chat luong'."
        )
    elif query.intent == "COMPLAINT_SUMMARY":
        instruction = (
            "Tra loi dung format:\n"
            "Phan nan chinh:\n"
            "- Ten khia canh: so negative/tong so neu co, kem ty le.\n"
            "Bang chung:\n"
            "- Tom tat evidence ngan, khong copy review dai.\n"
            "Muc do can luu y:\n"
            "- ...\n"
            "Chi dung negative evidence, khong liet ke diem khen neu cau hoi khong yeu cau. "
            "Khong xuat aspect code."
        )
    elif query.intent == "RISK_CHECK":
        instruction = (
            "Tra loi dung format:\n"
            "Diem can luu y:\n"
            "- ...\n"
            "Bang chung:\n"
            "- ...\n"
            "Khuyen nghi:\n"
            "- ...\n"
            "Uu tien risk flags va negative evidence. Khong xuat ten risk/aspect ky thuat."
        )
    elif query.intent == "PRODUCT_FIT":
        instruction = (
            "Tra loi dung format:\n"
            "Muc do phu hop:\n"
            "- Phu hop / Can nhac / Chua du du lieu\n"
            "Ly do:\n"
            "- ...\n"
            "Can luu y:\n"
            "- ...\n"
            "Chi dua tren target_aspects va evidence. Neu co canh bao ve giao hang/chinh hang, hay tach thanh luu y khi mua/nhan hang; "
            "khong noi be 'nhay cam voi' cac khia canh nay. Tra loi tu nhien, mo dau bang Co/Can nhac/Chua du du lieu."
        )
    elif query.intent == "ASPECT_QA":
        instruction = (
            "Tra loi dung cau hoi ve khia canh cu the, khong chuyen sang tom tat phan nan chung. "
            "Chi dung target_aspects, absa_statistics va representative_opinions trong context. "
            "Neu cau hoi hoi 'co bi/co gay ... khong', hay tra loi Co/Khong thay ro/Chua du du lieu, "
            "sau do can bang so evidence tich cuc va tieu cuc neu co. Khong nhac delivery/seller neu khong nam trong target_aspects."
        )
    else:
        instruction = (
            "Tra loi ngan gon 3-5 cau. Khong chep lai evidence tho. "
            "Neu thieu du lieu, noi ro chua du du lieu."
        )

    return (
        _json_response_contract()
        + "\nCONTEXT_JSON:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        "INSTRUCTION:\n"
        f"{instruction}"
    )


def _shopping_assistant_system_prompt() -> str:
    return (
        "Bạn là AI Shopping Assistant hỗ trợ phụ huynh lựa chọn sản phẩm.\n"
        "QUY TẮC:\n"
        "- Chỉ sử dụng thông tin có trong context.\n"
        "- Không suy diễn ngoài dữ liệu.\n"
        "- Không hiển thị chain-of-thought hoặc nội dung <think>.\n"
        "- Không hiển thị tên aspect nội bộ hoặc risk flag nội bộ.\n"
        "- Không sao chép nguyên văn review.\n"
        "- Phải tổng hợp từ nhiều review khi có đủ evidence.\n"
        "- Luôn sử dụng tiếng Việt có dấu.\n"
        "- Ngôn ngữ tự nhiên, thân thiện, dễ hiểu.\n"
        "- Trả lời như một chuyên gia tư vấn mua sắm.\n"
        "- Nếu dữ liệu chưa đủ thì nói rõ \"Chưa có đủ dữ liệu để kết luận\".\n"
        "- Không dùng các câu khuyến nghị chung chung như \"Nên cân nhắc thêm\" hoặc \"Nên đọc kỹ review\".\n"
        "- Mỗi khuyến nghị phải dựa trên dữ liệu thực tế của sản phẩm.\n"
        "- Mỗi kết luận quan trọng cần có ít nhất một bằng chứng hỗ trợ.\n"
        "- Không lặp lại cùng một evidence nhiều lần.\n"
        "- Tối đa 150 từ.\n"
        "- Chỉ trả về JSON hợp lệ."
    )


def _json_response_contract() -> str:
    return (
        "Return JSON with exactly this schema:\n"
        "{\n"
        '  "summary": "short natural answer",\n'
        '  "pros": ["short positive point"],\n'
        '  "cons": ["short caution"],\n'
        '  "recommendation": "buying advice or insufficient-data note",\n'
        '  "confidence": "low|medium|high",\n'
        '  "confidence_reasons": ["short reason why confidence has this level"],\n'
        '  "citations": [{"aspect": "friendly aspect label", "evidence": "short paraphrased evidence"}]\n'
        "}\n"
        "Do not include markdown fences or any extra keys."
    )


def _compact_product(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": product.get("name"),
        "price": product.get("price"),
        "rating_average": product.get("rating_average"),
        "review_count": product.get("review_count"),
        "brand_name": product.get("brand_name"),
        "seller_name": product.get("seller_name"),
    }


def _friendly_query_context(query: QueryAnalysis) -> dict[str, Any]:
    return {
        "target_aspects": [_aspect_label(aspect) for aspect in query.target_aspects],
        "constraints": query.constraints,
        "confidence": query.confidence,
        "source": query.source,
    }


def _compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_aspect_records": stats.get("total_aspect_records", 0),
        "top_positive_aspects": [_friendly_stat_row(row) for row in (stats.get("top_positive_aspects") or [])[:4]],
        "top_negative_aspects": [_friendly_stat_row(row) for row in (stats.get("top_negative_aspects") or [])[:4]],
    }


def _compact_analytics(analytics: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_statistics": analytics.get("review_statistics") or {},
        "top_complaints": [_friendly_stat_row(row) for row in (analytics.get("top_complaints") or [])[:5]],
        "top_praises": [_friendly_stat_row(row) for row in (analytics.get("top_praises") or [])[:5]],
        "negative_ratio_by_aspect": [_friendly_stat_row(row) for row in (analytics.get("negative_ratio_by_aspect") or [])[:5]],
        "positive_ratio_by_aspect": [_friendly_stat_row(row) for row in (analytics.get("positive_ratio_by_aspect") or [])[:5]],
    }


def _stats_from_analytics(analytics: dict[str, Any]) -> dict[str, Any]:
    rows = analytics.get("aspect_sentiment_stats") or []
    return {
        "total_aspect_records": int((analytics.get("review_statistics") or {}).get("aspect_record_count") or 0),
        "top_negative_aspects": sorted(
            rows,
            key=lambda row: (row.get("negative", 0), row.get("negative_ratio", 0), row.get("total", 0)),
            reverse=True,
        ),
        "top_positive_aspects": sorted(
            rows,
            key=lambda row: (row.get("positive", 0), row.get("positive_ratio", 0), row.get("total", 0)),
            reverse=True,
        ),
    }


def _scope_stats_for_query(stats: dict[str, Any], query: QueryAnalysis) -> dict[str, Any]:
    target_aspects = set(query.target_aspects)
    if query.intent not in {"PRODUCT_FIT", "ASPECT_QA", "PRICE_VALUE", "DELIVERY_SERVICE"} or not target_aspects:
        return stats
    positive = [row for row in _filter_stat_rows(stats.get("top_positive_aspects") or [], target_aspects) if int(row.get("positive") or 0) > 0]
    negative = [row for row in _filter_stat_rows(stats.get("top_negative_aspects") or [], target_aspects) if int(row.get("negative") or 0) > 0]
    return {
        "total_aspect_records": sum(int(row.get("total") or 0) for row in _merge_stat_rows(positive, negative)),
        "top_positive_aspects": positive,
        "top_negative_aspects": negative,
    }


def _scope_analytics_for_query(analytics: dict[str, Any], query: QueryAnalysis) -> dict[str, Any]:
    target_aspects = set(query.target_aspects)
    if not analytics or query.intent not in {"PRODUCT_FIT", "ASPECT_QA", "PRICE_VALUE", "DELIVERY_SERVICE"} or not target_aspects:
        return analytics
    out = dict(analytics)
    for key in (
        "top_complaints",
        "top_praises",
        "negative_ratio_by_aspect",
        "positive_ratio_by_aspect",
        "aspect_sentiment_stats",
    ):
        out[key] = _filter_stat_rows(analytics.get(key) or [], target_aspects)
    out["top_praises"] = [row for row in out.get("top_praises") or [] if int(row.get("positive") or row.get("total") or 0) > 0]
    out["top_complaints"] = [row for row in out.get("top_complaints") or [] if int(row.get("negative") or row.get("total") or 0) > 0]
    out["positive_ratio_by_aspect"] = [row for row in out.get("positive_ratio_by_aspect") or [] if int(row.get("positive") or row.get("total") or 0) > 0]
    out["negative_ratio_by_aspect"] = [row for row in out.get("negative_ratio_by_aspect") or [] if int(row.get("negative") or row.get("total") or 0) > 0]
    aspect_stats = out.get("aspect_sentiment_stats") or []
    out["review_statistics"] = {
        **(analytics.get("review_statistics") or {}),
        "aspect_record_count": sum(int(row.get("total") or 0) for row in aspect_stats),
    }
    return out


def _scope_risks_for_query(risks: list[dict[str, Any]], query: QueryAnalysis) -> list[dict[str, Any]]:
    if query.intent == "PRODUCT_FIT":
        return [row for row in risks if _is_baby_fit_risk(row)]
    if query.intent == "ASPECT_QA":
        target_risks = {_risk_type_for_aspect(aspect) for aspect in query.target_aspects}
        target_risks.discard("")
        return [row for row in risks if str(row.get("risk_type") or "") in target_risks]
    return risks


def _filter_stat_rows(rows: list[dict[str, Any]], target_aspects: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("aspect") or "") in target_aspects]


def _merge_stat_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_aspect: dict[str, dict[str, Any]] = {}
    for rows in groups:
        for row in rows:
            aspect = str(row.get("aspect") or "")
            if aspect and aspect not in by_aspect:
                by_aspect[aspect] = row
    return list(by_aspect.values())


def _friendly_stat_row(row: dict[str, Any]) -> dict[str, Any]:
    aspect = str(row.get("aspect") or "")
    total = int(row.get("total") or 0)
    negative = int(row.get("negative") or (row.get("total") if row.get("sentiment") == "negative" else 0) or 0)
    positive = int(row.get("positive") or (row.get("total") if row.get("sentiment") == "positive" else 0) or 0)
    negative_ratio = row.get("negative_ratio")
    positive_ratio = row.get("positive_ratio")
    return {
        "aspect": _aspect_label(aspect),
        "sentiment": row.get("sentiment") or "",
        "negative": negative,
        "positive": positive,
        "total": total,
        "avg_confidence": row.get("avg_confidence") or 0.0,
        "negative_percent": _percent(negative_ratio) if negative_ratio is not None else "",
        "positive_percent": _percent(positive_ratio) if positive_ratio is not None else "",
    }


def _friendly_risk_row(row: dict[str, Any]) -> dict[str, Any]:
    risk_type = str(row.get("risk_type") or "")
    return {
        "risk": _risk_label(risk_type),
        "severity": row.get("severity") or "low",
        "confidence": row.get("confidence") or 0.0,
        "evidence": _summarize_opinion_text(str(row.get("evidence") or "")),
    }


def _retrieve_for_query(
    aspects: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    query: QueryAnalysis,
    limit: int = 6,
) -> list[dict[str, Any]]:
    target_aspects = set(query.target_aspects)

    if query.intent == "SUMMARY":
        return _select_diverse_evidence(aspects, limit=limit)

    if query.intent == "COMPLAINT_SUMMARY":
        return _select_diverse_evidence(aspects, sentiment_order=("negative",), limit=limit, fill_remaining=False)

    if query.intent == "RISK_CHECK":
        risk_evidence = [_risk_to_evidence(row) for row in risks[:3] if row.get("evidence")]
        negative = _select_diverse_evidence(
            _filter_by_aspects(aspects, target_aspects, strict=bool(target_aspects)),
            sentiment_order=("negative",),
            limit=limit,
            fill_remaining=False,
        )
        return _dedupe_evidence(risk_evidence + negative, limit=limit)

    if query.intent == "RECOMMENDATION":
        pool = _filter_by_aspects(aspects, target_aspects) if target_aspects else aspects
        negative = _select_diverse_evidence(pool, sentiment_order=("negative",), limit=3)
        positive = _select_diverse_evidence(pool, sentiment_order=("positive",), limit=3)
        return _dedupe_evidence(negative + positive, limit=limit)

    if query.intent in {"PRODUCT_FIT", "ASPECT_QA", "PRICE_VALUE", "DELIVERY_SERVICE"}:
        pool = _filter_by_aspects(aspects, target_aspects, strict=bool(target_aspects)) if target_aspects else aspects
        return _select_diverse_evidence(pool, limit=limit)

    if query.intent == "PRODUCT_QA":
        pool = _filter_by_aspects(aspects, target_aspects) if target_aspects else aspects
        return _select_diverse_evidence(pool, limit=limit)

    return _select_diverse_evidence(aspects, limit=limit)


def _select_diverse_evidence(
    aspects: list[dict[str, Any]],
    sentiment_order: tuple[str, ...] = ("negative", "positive", "neutral"),
    limit: int = 5,
    fill_remaining: bool = True,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    seen_aspects: set[str] = set()
    seen_reviews: set[str] = set()
    seen_texts: set[str] = set()

    for sentiment in sentiment_order:
        for item in select_evidence(aspects, sentiment, limit=len(aspects)):
            aspect = str(item.get("aspect") or "")
            text = str(item.get("text") or item.get("sentence") or "")
            row_sentiment = str(item.get("sentiment") or item.get("polarity") or sentiment or "")
            review_id = str(item.get("review_id") or "").strip()
            text_key = _normalize_evidence_key(text)
            if (
                not _is_quality_evidence(aspect, row_sentiment, text)
                or aspect in seen_aspects
                or text_key in seen_texts
                or (review_id and review_id in seen_reviews)
            ):
                continue
            picked.append(_normalize_evidence_row(item))
            seen_aspects.add(aspect)
            seen_texts.add(text_key)
            if review_id:
                seen_reviews.add(review_id)
            if len(picked) >= limit:
                return picked

    if not fill_remaining:
        return picked

    for item in select_evidence(aspects, None, limit=limit):
        aspect = str(item.get("aspect") or "")
        text = str(item.get("text") or item.get("sentence") or "")
        row_sentiment = str(item.get("sentiment") or item.get("polarity") or "")
        review_id = str(item.get("review_id") or "").strip()
        text_key = _normalize_evidence_key(text)
        if (
            _is_quality_evidence(aspect, row_sentiment, text)
            and text_key not in seen_texts
            and not (review_id and review_id in seen_reviews)
        ):
            picked.append(_normalize_evidence_row(item))
            seen_texts.add(text_key)
            if review_id:
                seen_reviews.add(review_id)
        if len(picked) >= limit:
            break
    return picked


def _filter_by_aspects(aspects: list[dict[str, Any]], target_aspects: set[str], strict: bool = False) -> list[dict[str, Any]]:
    if not target_aspects:
        return aspects
    pool = [item for item in aspects if str(item.get("aspect") or item.get("category") or "") in target_aspects]
    if strict:
        return pool
    return pool or aspects


def _dedupe_evidence(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_reviews: set[str] = set()
    seen_texts: set[str] = set()
    for row in rows:
        text = str(row.get("text") or row.get("sentence") or row.get("evidence") or "").strip()
        review_id = str(row.get("review_id") or "").strip()
        aspect = str(row.get("aspect") or row.get("category") or row.get("risk_type") or "")
        sentiment = str(row.get("sentiment") or row.get("polarity") or "negative")
        normalized_text = _normalize_evidence_key(text)
        if (
            not _is_quality_evidence(aspect, sentiment, text)
            or normalized_text in seen_texts
            or (review_id and review_id in seen_reviews)
        ):
            continue
        if review_id:
            seen_reviews.add(review_id)
        seen_texts.add(normalized_text)
        out.append({
            "review_id": review_id,
            "aspect_code": aspect,
            "aspect": _aspect_label(aspect),
            "aspect_label": _aspect_label(aspect),
            "sentiment": sentiment,
            "confidence": float(row.get("confidence") or 0.0),
            "text": _summarize_opinion_text(text),
        })
        if len(out) >= limit:
            break
    return out


def _risk_to_evidence(row: dict[str, Any]) -> dict[str, Any]:
    risk_type = str(row.get("risk_type") or "risk")
    return {
        "review_id": str(row.get("review_id") or ""),
        "aspect": _risk_label(risk_type),
        "aspect_label": _risk_label(risk_type),
        "sentiment": "negative",
        "confidence": float(row.get("confidence") or 0.0),
        "text": _summarize_opinion_text(str(row.get("evidence") or "")),
    }


def _normalize_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    aspect = str(row.get("aspect") or row.get("category") or "")
    return {
        "review_id": str(row.get("review_id") or ""),
        "aspect_code": aspect,
        "aspect": _aspect_label(aspect),
        "aspect_label": _aspect_label(aspect),
        "sentiment": str(row.get("sentiment") or row.get("polarity") or "neutral"),
        "confidence": float(row.get("confidence") or 0.0),
        "text": _summarize_opinion_text(str(row.get("text") or row.get("sentence") or row.get("evidence") or "")),
    }


def _compress_evidence_text(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    for marker in [". ", "; ", ", "]:
        idx = text.find(marker, 80)
        if 0 < idx <= limit:
            return text[:idx].strip()
    return text[:limit].rsplit(" ", 1)[0].strip()


def _summarize_opinion_text(text: str, limit: int = 150) -> str:
    text = _compress_evidence_text(text, limit=limit)
    text = re.sub(r"\b(PRODUCT|PRICE|DELIVERY|SELLER)#[A-Z_]+\b", "", text)
    text = re.sub(r"\b(delivery_accuracy|delivery_packaging|size_fit)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_evidence_key(text: str) -> str:
    return re.sub(r"\W+", " ", _summarize_opinion_text(text, limit=120).lower()).strip()


def _is_quality_evidence(aspect: str, sentiment: str, text: str) -> bool:
    if not aspect or _aspect_label(aspect) == "Khía cạnh sản phẩm":
        return False
    if sentiment not in {"positive", "neutral", "negative"}:
        return False
    summary = _summarize_opinion_text(text, limit=180)
    normalized = summary.lower().strip(" .,!?:;")
    if len(summary) < 28 or len(normalized.split()) < 5:
        return False
    weak_phrases = {
        "rất không hài lòng",
        "không hài lòng",
        "không thích",
        "tệ",
        "ổn",
        "ok",
        "good",
        "bad",
    }
    return normalized not in weak_phrases


def _opinion_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "aspect": str(row.get("aspect_label") or row.get("aspect") or "Bằng chứng"),
        "opinion": _summarize_opinion_text(str(row.get("text") or "")),
        "sentiment": str(row.get("sentiment") or "neutral"),
        "confidence": row.get("confidence") or 0.0,
    }


def _aspect_statistics(aspects: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for item in aspects:
        aspect = str(item.get("aspect") or item.get("category") or "")
        sentiment = str(item.get("sentiment") or item.get("polarity") or "neutral")
        if not aspect:
            continue
        bucket = buckets.setdefault(aspect, {"positive": 0, "neutral": 0, "negative": 0, "total": 0})
        if sentiment not in ("positive", "neutral", "negative"):
            sentiment = "neutral"
        bucket[sentiment] += 1
        bucket["total"] += 1

    rows = []
    for aspect, counts in buckets.items():
        total = counts["total"] or 1
        rows.append({
            "aspect": aspect,
            "positive": counts["positive"],
            "neutral": counts["neutral"],
            "negative": counts["negative"],
            "negative_ratio": round(counts["negative"] / total, 3),
            "total": counts["total"],
        })
    rows.sort(key=lambda row: (row["negative"], row["total"]), reverse=True)
    return {
        "total_aspect_records": len(aspects),
        "top_negative_aspects": rows,
        "top_positive_aspects": sorted(rows, key=lambda row: (row["positive"], row["total"]), reverse=True),
    }


def _deterministic_summary(product: dict[str, Any], score: dict[str, Any], risks: list[dict[str, Any]]) -> str:
    name = product.get("name") or "Sản phẩm này"
    if risks:
        return (
            f"{name}: ket luan {score['label'].lower()}. "
            f"Co mot so diem can luu y ve {_risk_label(str(risks[0].get('risk_type') or 'rui ro'))}."
        )
    return f"{name}: kết luận {score['label'].lower()} dựa trên các review hiện có."


def _fallback_structured_answer(
    product: dict[str, Any],
    intent: str,
    score: dict[str, Any] | None,
    risks: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    stats: dict[str, Any] | None,
    analytics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = product.get("name") or "Sản phẩm này"
    if not evidence and not stats:
        return {
            "summary": "Chưa có đủ dữ liệu để kết luận về sản phẩm này.",
            "pros": [],
            "cons": [],
            "recommendation": "Bạn nên phân tích thêm review hoặc chọn sản phẩm có nhiều phản hồi rõ ràng hơn trước khi quyết định.",
            "confidence": "low",
            "confidence_reasons": ["Chưa có evidence đủ rõ để tổng hợp thành kết luận đáng tin cậy."],
            "citations": [],
        }

    stats = stats or {}
    analytics = analytics or {}
    top_pos = (stats.get("top_positive_aspects") or analytics.get("top_praises") or [])[:2]
    top_neg = (stats.get("top_negative_aspects") or analytics.get("top_complaints") or [])[:2]
    positive_evidence = [row for row in evidence if row.get("sentiment") == "positive"]
    negative_evidence = [row for row in evidence if row.get("sentiment") == "negative"]
    risk_names = _risk_names(risks[:2])

    pros = [
        f"Nhiều phản hồi tích cực nhắc đến {_aspect_names(top_pos)}." if top_pos else "",
        *[_evidence_to_reason(row) for row in positive_evidence[:2]],
    ]
    cons = [
        f"Cần lưu ý về {_aspect_names(top_neg)}." if top_neg else "",
        f"Có tín hiệu cần xem kỹ về {risk_names}." if risk_names else "",
        *[_evidence_to_reason(row) for row in negative_evidence[:2]],
    ]
    if intent == "PRODUCT_FIT":
        baby_cons = [
            f"Cần lưu ý về {_aspect_names(_baby_fit_rows(top_neg))}." if _baby_fit_rows(top_neg) else "",
            *[_evidence_to_reason(row) for row in _baby_fit_rows(negative_evidence)[:2]],
        ]
        operational_note = _join_nonempty([
            _risk_names([row for row in risks if not _is_baby_fit_risk(row)]),
            _aspect_names([row for row in top_neg if not _is_baby_fit_aspect(str(row.get("aspect") or ""))]),
        ])
        if operational_note != "chưa đủ dữ liệu":
            baby_cons.append(f"Lưu ý khi mua/nhận hàng: {operational_note}.")
        cons = baby_cons
    pros = [item for item in pros if item]
    cons = [item for item in cons if item]

    conclusion = score.get("label") if score else "Can nhac"
    confidence = _confidence_label(float(score.get("confidence") or 0.5)) if score else _confidence_from_context(stats, evidence)
    confidence_reasons = _confidence_reasons(confidence, stats, evidence, top_pos, top_neg)

    if intent == "COMPLAINT_SUMMARY":
        summary = _complaint_summary_text(top_neg, negative_evidence)
        recommendation = _specific_recommendation(conclusion, top_pos, top_neg, risks, intent)
    elif intent == "ASPECT_QA":
        summary = _aspect_qa_summary(top_pos, top_neg, positive_evidence, negative_evidence)
        recommendation = _aspect_qa_recommendation(top_pos, top_neg, positive_evidence, negative_evidence)
    elif intent == "RISK_CHECK":
        summary = (
            f"Các điểm cần lưu ý nổi bật liên quan đến {risk_names}."
            if risk_names else "Chưa thấy rủi ro nổi bật trong các review đã phân tích."
        )
        recommendation = _specific_recommendation(conclusion, top_pos, top_neg, risks, intent)
    elif intent == "PRODUCT_FIT":
        fit_positive = _baby_fit_rows(positive_evidence) or _baby_fit_rows(top_pos) or positive_evidence[:2] or top_pos
        fit_aspects = _aspect_names(fit_positive)
        summary = (
            f"{name} có thể phù hợp với nhu cầu này nếu bé hợp với sản phẩm, "
            f"vì review đang có tín hiệu tích cực về {fit_aspects}."
        )
        recommendation = _specific_recommendation(conclusion, top_pos, top_neg, risks, intent)
    else:
        summary = (
            f"Dựa trên các đánh giá hiện có, {name} nhìn chung có tín hiệu tốt về {_aspect_names(top_pos)}. "
            f"Tuy nhiên, điểm cần lưu ý chính là {_aspect_names(top_neg)}."
        )
        recommendation = _specific_recommendation(conclusion, top_pos, top_neg, risks, intent)

    return _sanitize_structured_answer({
        "summary": summary,
        "pros": pros[:3],
        "cons": cons[:3],
        "recommendation": recommendation,
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "citations": _citations_from_evidence(evidence[:4]),
    }, evidence)


def _format_structured_answer(data: dict[str, Any], intent: str = "SUMMARY") -> str:
    data = _sanitize_structured_answer(data, [])
    lines = [data["summary"]]
    if data["pros"]:
        lines.append("Ưu điểm: " + "; ".join(data["pros"][:3]))
    if data["cons"]:
        heading = "Người mua chủ yếu phản ánh" if intent == "COMPLAINT_SUMMARY" else "Điểm cần lưu ý"
        lines.append(f"{heading}: " + "; ".join(data["cons"][:3]))
    if data["recommendation"]:
        lines.append("Khuyến nghị: " + data["recommendation"])
    if data["confidence"]:
        confidence_label = {"high": "Cao", "medium": "Trung bình", "low": "Thấp"}.get(data["confidence"], "Trung bình")
        reason_text = "; ".join(data.get("confidence_reasons") or [])
        lines.append(f"Độ tin cậy: {confidence_label}" + (f" - {reason_text}" if reason_text else ""))
    return "\n\n".join(line for line in lines if line).strip()


def _sanitize_structured_answer(data: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    out = {
        "summary": _clean_public_text(str(data.get("summary") or ""))[:600],
        "pros": [_clean_public_text(str(item))[:220] for item in (data.get("pros") or []) if item],
        "cons": [_clean_public_text(str(item))[:220] for item in (data.get("cons") or []) if item],
        "recommendation": _clean_public_text(str(data.get("recommendation") or ""))[:300],
        "confidence": str(data.get("confidence") or "medium").lower(),
        "confidence_reasons": [_clean_public_text(str(item))[:180] for item in (data.get("confidence_reasons") or []) if item],
        "citations": [],
    }
    if out["confidence"] not in {"low", "medium", "high"}:
        out["confidence"] = "medium"
    citations = data.get("citations") or _citations_from_evidence(evidence)
    for citation in citations[:4]:
        if not isinstance(citation, dict):
            continue
        raw_aspect = str(citation.get("aspect") or "Bang chung")
        aspect = _aspect_label(raw_aspect) if raw_aspect in ASPECT_LABELS else _clean_public_text(raw_aspect)
        evidence_text = _summarize_opinion_text(str(citation.get("evidence") or ""))
        if evidence_text:
            out["citations"].append({"aspect": aspect[:80], "evidence": evidence_text[:180]})
    if _is_generic_recommendation(out["recommendation"]):
        out["recommendation"] = _recommendation_from_citations(out["citations"])
    if not out["summary"] or len(out["summary"].split()) < 5:
        out["summary"] = "Chưa có đủ dữ liệu để kết luận về sản phẩm này."
    if not out["confidence_reasons"]:
        out["confidence_reasons"] = ["Mức độ tin cậy được ước tính từ số lượng và độ rõ của evidence hiện có."]
    return out


def _is_generic_recommendation(text: str) -> bool:
    normalized = text.lower().strip(" .")
    generic_phrases = {
        "nên cân nhắc thêm",
        "nên xem xét",
        "nên đọc kỹ review",
        "nên đọc kỹ các điểm bị chê nhiều nhất",
        "nên cân nhắc thêm trước khi mua",
    }
    return not normalized or normalized in generic_phrases


def _recommendation_from_citations(citations: list[dict[str, str]]) -> str:
    if not citations:
        return "Chưa có đủ dữ liệu để kết luận; bạn nên ưu tiên sản phẩm có nhiều phản hồi rõ ràng hơn."
    first = citations[0]
    aspect = first.get("aspect") or "điểm được nhắc đến nhiều"
    evidence = first.get("evidence") or "phản hồi hiện có chưa đủ rõ"
    return f"Hãy quyết định dựa trên mức độ quan trọng của {aspect.lower()} với bé, vì evidence hiện có cho thấy: {evidence}"


def _citations_from_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "aspect": str(row.get("aspect_label") or row.get("aspect") or "Bang chung"),
            "evidence": _summarize_opinion_text(str(row.get("text") or "")),
        }
        for row in evidence
        if row.get("text")
    ]


def _clean_public_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\b(PRODUCT|PRICE|DELIVERY|SELLER)#[A-Z_]+\b", "", text)
    text = re.sub(r"\b(delivery_accuracy|delivery_packaging|size_fit)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _confidence_label(value: float) -> str:
    if value >= 0.67:
        return "high"
    if value >= 0.34:
        return "medium"
    return "low"


def _confidence_from_context(stats: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    total_records = int(stats.get("total_aspect_records") or 0)
    evidence_count = len(evidence)
    rows = _merge_stat_rows(stats.get("top_positive_aspects") or [], stats.get("top_negative_aspects") or [])
    avg_absa_confidence = _average([float(row.get("avg_confidence") or 0.0) for row in rows if row.get("avg_confidence") is not None])
    negative_total = sum(int(row.get("negative") or 0) for row in rows)
    row_total = sum(int(row.get("total") or 0) for row in rows)
    negative_ratio = negative_total / row_total if row_total else 0.0

    coverage_score = min(1.0, total_records / 12.0)
    evidence_score = min(1.0, evidence_count / 4.0)
    absa_score = avg_absa_confidence if avg_absa_confidence > 0 else 0.6
    conflict_penalty = 0.2 if 0.25 <= negative_ratio <= 0.55 and row_total >= 4 else 0.0
    confidence_score = coverage_score * 0.45 + evidence_score * 0.3 + absa_score * 0.25 - conflict_penalty
    return _confidence_label(confidence_score)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _confidence_reasons(
    confidence: str,
    stats: dict[str, Any],
    evidence: list[dict[str, Any]],
    top_pos: list[dict[str, Any]],
    top_neg: list[dict[str, Any]],
) -> list[str]:
    total_records = int(stats.get("total_aspect_records") or 0)
    clear_evidence = len(evidence)
    if confidence == "high":
        return [
            f"Có {total_records} lượt ý kiến theo khía cạnh được dùng để tổng hợp." if total_records else "Có nhiều evidence đủ rõ để đối chiếu.",
            f"Các phản hồi nổi bật tập trung vào {_aspect_names((top_pos + top_neg)[:3])}.",
        ]
    if confidence == "low":
        return [
            f"Chỉ có {clear_evidence} evidence đủ rõ cho câu hỏi này.",
            "Kết luận nên được xem như gợi ý ban đầu, chưa phải nhận định chắc chắn.",
        ]
    return [
        f"Có {clear_evidence} evidence đại diện và {total_records or 'một số'} lượt ý kiến theo khía cạnh.",
        "Dữ liệu đủ để đưa ra gợi ý, nhưng vẫn nên đối chiếu với nhu cầu thực tế của bé.",
    ]


def _complaint_summary_text(top_neg: list[dict[str, Any]], negative_evidence: list[dict[str, Any]]) -> str:
    if not top_neg and not negative_evidence:
        return "Chưa có đủ dữ liệu để kết luận người mua phàn nàn nhiều nhất về điểm nào."
    main = top_neg[0] if top_neg else {}
    main_aspect = _aspect_label(str(main.get("aspect") or negative_evidence[0].get("aspect") or "khía cạnh sản phẩm"))
    ratio = main.get("negative_percent") or _percent(main.get("negative_ratio"))
    ratio_text = f" Khoảng {ratio} phản hồi liên quan đến {main_aspect.lower()} mang sắc thái tiêu cực." if ratio else ""
    example = _summarize_opinion_text(str((negative_evidence[0] if negative_evidence else {}).get("text") or ""))
    second = _aspect_names(top_neg[1:3])
    extra = f" Ngoài ra cũng có một số ý kiến liên quan đến {second}, nhưng mức độ nổi bật thấp hơn." if second != "chưa đủ dữ liệu" else ""
    evidence_text = f" Một số phụ huynh phản ánh rằng {example[0].lower() + example[1:] if example else 'điểm này chưa đáp ứng kỳ vọng'}."
    return f"Người mua chủ yếu phản ánh về {main_aspect.lower()}.{evidence_text}{ratio_text}{extra}"


def _aspect_qa_summary(
    top_pos: list[dict[str, Any]],
    top_neg: list[dict[str, Any]],
    positive_evidence: list[dict[str, Any]],
    negative_evidence: list[dict[str, Any]],
) -> str:
    focus = _aspect_names((top_pos + top_neg)[:2])
    positive_count = sum(int(row.get("positive") or 0) for row in top_pos)
    negative_count = sum(int(row.get("negative") or 0) for row in top_neg)
    if negative_evidence:
        opening = f"Có ghi nhận một số phản hồi chưa tốt về {focus.lower()}."
        balance = (
            f"Tuy vậy cũng có {positive_count} ý kiến tích cực cùng nhóm aspect."
            if positive_count else "Chưa thấy nhiều phản hồi tích cực đủ rõ để cân bằng điểm này."
        )
        return f"{opening} {balance}"
    if positive_evidence:
        return f"Chưa thấy phản hồi tiêu cực nổi bật về {focus.lower()}; các evidence hiện có nghiêng về trải nghiệm tích cực."
    return f"Chưa có đủ evidence rõ ràng về {focus.lower()} để trả lời chắc chắn."


def _aspect_qa_recommendation(
    top_pos: list[dict[str, Any]],
    top_neg: list[dict[str, Any]],
    positive_evidence: list[dict[str, Any]],
    negative_evidence: list[dict[str, Any]],
) -> str:
    focus = _aspect_names((top_pos + top_neg)[:2])
    if negative_evidence and positive_evidence:
        return f"Nên xem {focus.lower()} là điểm cần kiểm tra thêm: có cả phản hồi tốt lẫn phản hồi chưa hài lòng, nên quyết định theo mức độ quan trọng của khía cạnh này với bé."
    if negative_evidence:
        return f"Nếu {focus.lower()} là tiêu chí quan trọng, nên thận trọng hơn vì evidence đang có phản hồi tiêu cực trực tiếp."
    if positive_evidence:
        return f"Có thể yên tâm hơn về {focus.lower()}, nhưng vẫn nên đối chiếu với cách dùng thực tế của bé."
    return f"Chưa nên kết luận về {focus.lower()} vì dữ liệu hiện có còn thiếu evidence trực tiếp."


def _specific_recommendation(
    conclusion: str,
    top_pos: list[dict[str, Any]],
    top_neg: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    intent: str,
) -> str:
    positives = _aspect_names(top_pos)
    negatives = _aspect_names(top_neg)
    risk_text = _risk_names(risks[:2])
    caution_source = risk_text if risk_text else negatives

    if conclusion == "Nen mua":
        if negatives != "chưa đủ dữ liệu":
            return (
                f"Có thể mua nếu bạn ưu tiên {positives}; riêng {negatives.lower()} nên kiểm tra kỹ "
                "trước khi chọn size hoặc nhận hàng."
            )
        return f"Có thể mua nếu bạn đang tìm sản phẩm mạnh về {positives}, vì các phản hồi hiện có nghiêng về hướng tích cực."

    if conclusion == "Nen tranh":
        return (
            f"Nên tránh nếu {caution_source.lower()} là tiêu chí quan trọng với bé, vì nhóm phản hồi tiêu cực này có thể ảnh hưởng trực tiếp đến trải nghiệm sử dụng."
        )

    if intent == "PRODUCT_FIT":
        baby_risk_text = _risk_names([row for row in risks if _is_baby_fit_risk(row)])
        baby_negative_text = _aspect_names(_baby_fit_rows(top_neg))
        baby_caution = baby_risk_text if baby_risk_text else baby_negative_text
        if baby_caution != "chưa đủ dữ liệu":
            return (
                f"Sản phẩm vẫn đáng cân nhắc, nhưng với bé có da dễ kích ứng bạn nên chú ý thêm phần {baby_caution.lower()} và theo dõi phản ứng da trong vài lần dùng đầu."
            )
        return "Dựa trên các đánh giá hiện có, sản phẩm đáng cân nhắc cho bé có làn da nhạy cảm vì phản hồi liên quan đến an toàn/chất liệu đang nghiêng tích cực; bạn vẫn nên theo dõi phản ứng da trong vài lần sử dụng đầu tiên."
    if intent == "COMPLAINT_SUMMARY":
        return (
            f"Khi vẫn muốn mua, hãy ưu tiên kiểm tra {caution_source.lower()} trước; nếu điểm này là nhu cầu chính, nên so sánh thêm với sản phẩm có phản hồi ổn định hơn."
        )
    if intent == "RISK_CHECK":
        return (
            f"Có thể mua thận trọng nếu các điểm về {caution_source.lower()} không phải rủi ro lớn với bé; khi nhận hàng nên kiểm tra ngay bao bì, size và tình trạng sản phẩm."
        )
    return (
        f"Nên cân nhắc theo nhu cầu chính của bạn: sản phẩm có điểm tốt về {positives}, nhưng cần xem kỹ {caution_source.lower()} trước khi quyết định."
    )


def _evidence_to_reason(row: dict[str, Any]) -> str:
    aspect = row.get("aspect_label") or row.get("aspect") or "Khía cạnh sản phẩm"
    text = row.get("text") or ""
    return f"{aspect}: {_summarize_opinion_text(text, limit=140)}"


def _risk_to_text(row: dict[str, Any]) -> str:
    label = _risk_label(str(row.get("risk_type") or "rui ro"))
    return f"{label}: {_summarize_opinion_text(str(row.get('evidence') or ''), limit=140)}"


def _fallback_chat_answer(
    product: dict[str, Any],
    question: str,
    evidence: list[dict[str, Any]],
    query: QueryAnalysis,
    stats: dict[str, Any],
    risks: list[dict[str, Any]],
    analytics: dict[str, Any] | None = None,
) -> str:
    del question
    if not evidence:
        return "Chua du du lieu review/aspect de tra loi cau hoi nay cho san pham hien tai."
    name = product.get("name") or "san pham nay"
    snippets = "; ".join(row["text"][:120] for row in evidence[:3])
    if query.intent == "SUMMARY":
        top_pos = (stats.get("top_positive_aspects") or [])[:2]
        top_neg = (stats.get("top_negative_aspects") or [])[:2]
        risk_text = _risk_names(risks[:2])
        return (
            f"Tom tat: {name} duoc khen nhieu ve {_aspect_names(top_pos)}. "
            f"Cac bang chung tieu bieu cho thay: {snippets[:180]}.\n\n"
            f"Diem can luu y la {_aspect_names(top_neg)}"
            f"{f', dac biet lien quan den {risk_text}' if risk_text else ''}. "
            "Nhung diem nay nen duoc xem nhu canh bao truoc khi mua, khong phai ket luan tuyet doi.\n\n"
            "Ket luan: san pham van dang can nhac neu phu huynh chon dung size, kiem tra hang khi nhan "
            "va theo doi phan ung cua be trong lan dau su dung."
        )
    if query.intent == "COMPLAINT_SUMMARY":
        complaints = (stats.get("top_negative_aspects") or []) or (analytics or {}).get("top_complaints") or []
        complaint_lines = _aspect_count_lines(complaints[:3], default_sentiment="negative")
        return (
            "Phan nan chinh:\n"
            f"{complaint_lines}\n"
            "Bang chung:\n"
            f"- {snippets}\n"
            "Muc do can luu y:\n"
            "- Nen uu tien kiem tra cac aspect co nhieu negative truoc khi mua."
        )
    if query.intent == "RISK_CHECK":
        if risks:
            risk = risks[0]
            return (
                f"Diem can luu y chinh la {_risk_label(str(risk.get('risk_type') or 'risk'))} "
                f"(muc {risk.get('severity')}). Bang chung tom tat: {snippets}. "
                "Phu huynh nen kiem tra ky san pham khi nhan va theo doi phan ung cua be khi dung lan dau."
            )
        return f"Chua thay canh bao noi bat. Bang chung lien quan hien co: {snippets}"
    if query.intent == "PRODUCT_FIT":
        target = ", ".join(_aspect_label(aspect) for aspect in query.target_aspects) or "nhu cau nay"
        negative_count = sum(1 for row in evidence if row.get("sentiment") == "negative")
        fit = "Co the phu hop" if negative_count == 0 else "Can nhac"
        return (
            f"{fit}. Dua tren cac review ve {target}, san pham co mot so tin hieu phu hop nhu: {snippets}. "
            "Tuy nhien du lieu tu review khong thay the viec theo doi thuc te, nhat la voi be co da nhay cam."
        )
    if query.intent == "RECOMMENDATION":
        return f"Dua tren review hien co cua {name}, nen can nhac ca diem tot va diem can luu y. Evidence: {snippets}"
    return f"Dua tren review cua {name}, cac bang chung lien quan la: {snippets}"


def _aspect_names(rows: list[dict[str, Any]]) -> str:
    names = list(dict.fromkeys(_aspect_label(str(row.get("aspect") or "")) for row in rows if row.get("aspect")))
    return ", ".join(names) if names else "chưa đủ dữ liệu"


def _baby_fit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _is_baby_fit_aspect(str(row.get("aspect") or ""))]


def _is_baby_fit_aspect(aspect: str) -> bool:
    return aspect in BABY_FIT_ASPECTS


def _is_baby_fit_risk(row: dict[str, Any]) -> bool:
    return str(row.get("risk_type") or "") in BABY_FIT_RISKS


def _join_nonempty(parts: list[str]) -> str:
    values: list[str] = []
    for part in parts:
        if part and part != "chưa đủ dữ liệu":
            values.extend(item.strip() for item in part.split(",") if item.strip())
    deduped = list(dict.fromkeys(values))
    return ", ".join(deduped) if deduped else "chưa đủ dữ liệu"


def _aspect_count_lines(rows: list[dict[str, Any]], default_sentiment: str) -> str:
    if not rows:
        return "- Chưa có đủ dữ liệu thống kê."
    lines = []
    for row in rows:
        aspect = _aspect_label(str(row.get("aspect") or "khía cạnh sản phẩm"))
        sentiment_count = int(row.get(default_sentiment) or row.get("total") or 0)
        total = int(row.get("total") or sentiment_count or 0)
        ratio = row.get(f"{default_sentiment}_ratio")
        ratio_text = f" ({_percent(ratio)})" if ratio is not None else ""
        sentiment_label = "tiêu cực" if default_sentiment == "negative" else "tích cực"
        lines.append(f"- {aspect}: {sentiment_count}/{total} ý kiến {sentiment_label}{ratio_text}.")
    return "\n".join(lines)


def _aspect_label(aspect: str) -> str:
    if aspect in ASPECT_LABELS:
        return ASPECT_LABELS[aspect]
    aliases = {
        "delivery_accuracy": "Giao đúng/đủ hàng",
        "delivery_packaging": "Bao bì",
        "size_fit": "Kích thước",
        "safety": "An toàn cho bé",
        "authenticity": "Tính chính hãng",
        "material": "Chất liệu",
        "durability": "Độ bền",
    }
    if aspect in aliases:
        return aliases[aspect]
    if "#" in aspect or "_" in aspect:
        return "Khía cạnh sản phẩm"
    return aspect or "Khía cạnh sản phẩm"


def _risk_label(risk_type: str) -> str:
    return RISK_LABELS.get(risk_type, risk_type or "rủi ro")


def _risk_type_for_aspect(aspect: str) -> str:
    return {
        "PRODUCT#SAFETY": "safety",
        "PRODUCT#MATERIAL": "material",
        "PRODUCT#SIZE": "size_fit",
        "PRODUCT#DURABILITY": "durability",
        "DELIVERY#PACKAGING": "delivery_packaging",
        "DELIVERY#ACCURACY": "delivery_accuracy",
        "SELLER#AUTHENTICITY": "authenticity",
    }.get(aspect, "")


def _risk_names(rows: list[dict[str, Any]]) -> str:
    names = [_risk_label(str(row.get("risk_type") or "")) for row in rows if row.get("risk_type")]
    return ", ".join(names)


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except Exception:
        return ""


def _is_usable_llm_answer(answer: str) -> bool:
    normalized = answer.strip().lower()
    if len(normalized) < 20:
        return False
    blocked = ["<think", "</think>", "here's a thinking", "thinking process", "the user wants"]
    return not any(marker in normalized for marker in blocked)
