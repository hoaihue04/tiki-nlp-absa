from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_dashboard_service
from app.assistant.schemas import RiskResponse
from app.web.dashboard_service import DashboardService

router = APIRouter(prefix="/api/products")


@router.get("/{product_id}/risk", response_model=RiskResponse)
def product_risk(
    product_id: str,
    service: DashboardService = Depends(get_dashboard_service),
):
    ctx = service._v2_repository.load_product_context(product_id)
    if ctx is None:
        return {"product_id": product_id, "risks": []}
    return {"product_id": product_id, "risks": ctx.get("risks") or []}
