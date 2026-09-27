from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import analyze, assistant, health, products, web
from app.core.analysis_jobs import AnalysisJobManager
from app.assistant.assistant_service import ParentAssistantService
from app.web.dashboard_service import DashboardService

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


def create_app() -> FastAPI:
    fastapi_app = FastAPI(title="Tiki ABSA Dashboard", version="2.1.0")

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    fastapi_app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    dashboard_service = DashboardService()
    fastapi_app.state.dashboard_service = dashboard_service
    fastapi_app.state.parent_assistant_service = ParentAssistantService(
        dashboard_service._v2_repository
    )
    fastapi_app.state.analysis_job_manager = AnalysisJobManager(dashboard_service)

    fastapi_app.include_router(web.router)
    fastapi_app.include_router(analyze.router)
    fastapi_app.include_router(assistant.router)
    fastapi_app.include_router(products.router)
    fastapi_app.include_router(health.router)

    return fastapi_app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    print("Starting Tiki ABSA Dashboard v2.1 (Live crawl mode)")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
