from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import (
    get_assistant_service,
    get_dashboard_service,
    resolve_product_id,
)
from app.assistant.assistant_service import ParentAssistantService
from app.assistant.schemas import (
    ChatRequest,
    ChatResponse,
    ProductFitRequest,
    ProductFitResponse,
    PurchaseAdviceRequest,
    PurchaseAdviceResponse,
)
from app.web.dashboard_service import DashboardService

router = APIRouter(prefix="/api/assistant")


@router.post("/purchase-advice", response_model=PurchaseAdviceResponse)
def purchase_advice(
    payload: PurchaseAdviceRequest,
    service: DashboardService = Depends(get_dashboard_service),
    assistant_service: ParentAssistantService = Depends(get_assistant_service),
):
    product_id = resolve_product_id(payload.product_url, payload.product_id)
    if not product_id:
        raise HTTPException(status_code=400, detail="Can cung cap product_url hoac product_id hop le")

    try:
        if payload.product_url:
            analysis = service.analyze_product(product_url=payload.product_url, product_id=product_id)
            return assistant_service.build_from_analysis(analysis, use_llm=payload.use_llm)

        advice = assistant_service.build_from_db(product_id, use_llm=payload.use_llm)
        if advice is not None:
            return advice

        analysis = service.analyze_product(product_id=product_id)
        return assistant_service.build_from_analysis(analysis, use_llm=payload.use_llm)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/product-fit", response_model=ProductFitResponse)
def product_fit(
    payload: ProductFitRequest,
    service: DashboardService = Depends(get_dashboard_service),
    assistant_service: ParentAssistantService = Depends(get_assistant_service),
):
    product_id = resolve_product_id(payload.product_url, payload.product_id)
    if not product_id:
        raise HTTPException(status_code=400, detail="Can cung cap product_url hoac product_id hop le")

    try:
        if payload.product_url:
            analysis = service.analyze_product(product_url=payload.product_url, product_id=product_id)
            advice = assistant_service.build_from_analysis(analysis, use_llm=payload.use_llm)
        else:
            advice = assistant_service.build_from_db(product_id, use_llm=payload.use_llm)
            if advice is None:
                analysis = service.analyze_product(product_id=product_id)
                advice = assistant_service.build_from_analysis(analysis, use_llm=payload.use_llm)

        fit_level = "Phu hop" if advice["conclusion"] == "Nen mua" else "Can nhac"
        if advice["confidence"] < 0.25:
            fit_level = "Chua du du lieu"
        question_note = f" Nhu cau/cau hoi: {payload.question}" if payload.question else ""
        return {
            "product_id": product_id,
            "fit_level": fit_level,
            "answer": advice["summary"] + question_note,
            "structured_answer": advice.get("structured_answer") or {},
            "evidence": advice["evidence"],
            "llm_used": advice["llm_used"],
            "llm_model": advice["llm_model"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
def assistant_chat(
    payload: ChatRequest,
    service: DashboardService = Depends(get_dashboard_service),
    assistant_service: ParentAssistantService = Depends(get_assistant_service),
):
    product_id = resolve_product_id(payload.product_url, payload.product_id)
    if not product_id:
        raise HTTPException(status_code=400, detail="Can cung cap product_url hoac product_id hop le")

    try:
        if payload.product_url:
            service.analyze_product(product_url=payload.product_url, product_id=product_id)
        answer = assistant_service.answer_chat(product_id, payload.question, use_llm=payload.use_llm)
        if answer is None:
            latest_analysis = service.get_latest_analysis(product_id)
            if latest_analysis:
                advice = assistant_service.build_from_analysis(latest_analysis, use_llm=payload.use_llm)
                answer = {
                    "product_id": product_id,
                    "answer": advice["summary"],
                    "structured_answer": advice.get("structured_answer") or {},
                    "evidence": advice.get("evidence") or [],
                    "query_analysis": {},
                    "statistics_plan": {},
                    "statistics_result": {},
                    "retrieval_strategy": "in_memory_analysis",
                    "vector_used": False,
                    "cache_hit": True,
                    "llm_used": advice.get("llm_used", False),
                    "llm_model": advice.get("llm_model", ""),
                }
        if answer is None:
            raise ValueError(
                "Chưa có context cho sản phẩm này. Hãy phân tích sản phẩm lại rồi đặt câu hỏi."
            )
        return answer
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
