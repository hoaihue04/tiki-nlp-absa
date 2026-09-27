from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_dashboard_service, get_job_manager
from app.api.schemas import AnalyzeRequest
from app.core.analysis_jobs import AnalysisJobManager
from app.core.serialization import make_json_serializable
from app.web.dashboard_service import DashboardService

router = APIRouter(prefix="/api/analyze")


@router.post("/start")
async def analyze_start(
    payload: AnalyzeRequest,
    job_manager: AnalysisJobManager = Depends(get_job_manager),
):
    return await job_manager.start(payload.product_url, payload.product_id)


@router.get("/progress/{job_id}")
async def analyze_progress(
    job_id: str,
    job_manager: AnalysisJobManager = Depends(get_job_manager),
):
    return job_manager.progress_stream(job_id)


@router.post("")
def analyze_sync(
    payload: AnalyzeRequest,
    service: DashboardService = Depends(get_dashboard_service),
):
    try:
        result = service.analyze_product(
            product_url=payload.product_url,
            product_id=payload.product_id,
        )
        return {"ok": True, "data": make_json_serializable(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
