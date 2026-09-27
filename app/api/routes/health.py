from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_dashboard_service
from app.assistant.db import get_database_init_error
from app.web.dashboard_service import DashboardService

router = APIRouter(prefix="/api")


@router.get("/health")
async def health(service: DashboardService = Depends(get_dashboard_service)):
    return {
        "status": "ok",
        "version": "2.1.0",
        "v2_database": service._v2_repository.status,
        "v2_database_error": get_database_init_error(),
        "description": "Tiki ABSA — Live crawl mode (no static CSV required for reviews)",
    }
