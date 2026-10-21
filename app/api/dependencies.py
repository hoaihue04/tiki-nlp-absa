from __future__ import annotations

from starlette.requests import Request

from app.core.analysis_jobs import AnalysisJobManager
from app.assistant.assistant_service import ParentAssistantService
from app.web.dashboard_service import DashboardService, extract_product_id_from_url


def get_dashboard_service(request: Request) -> DashboardService:
    return request.app.state.dashboard_service


def get_assistant_service(request: Request) -> ParentAssistantService:
    return request.app.state.parent_assistant_service


def get_job_manager(request: Request) -> AnalysisJobManager:
    return request.app.state.analysis_job_manager


def resolve_product_id(product_url: str | None, product_id: str | None) -> str | None:
    if product_id:
        return str(product_id).strip()
    if product_url:
        return extract_product_id_from_url(product_url)
    return None
