
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
import uuid
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

# ─── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent          # → TIKI/app/
ROOT_DIR = BASE_DIR.parent                           # → TIKI/

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.src.web.dashboard_service import DashboardService, extract_product_id_from_url
from app.src.v2.assistant_service import ParentAssistantService
from app.src.v2.db import get_database_init_error
from app.src.v2.schemas import (
    ChatRequest,
    ChatResponse,
    ProductFitRequest,
    ProductFitResponse,
    PurchaseAdviceRequest,
    PurchaseAdviceResponse,
    RiskResponse,
)

# ─── JSON serialization helper ────────────────────────────────────────────────
def make_json_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


# ─── App setup ────────────────────────────────────────────────────────────────
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"

app = FastAPI(title="Tiki ABSA Dashboard", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Thread pool riêng — không block uvicorn event loop
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Singleton service — PhoBERT chỉ load 1 lần duy nhất
service = DashboardService()
app.state.dashboard_service = service
assistant_service = ParentAssistantService(service._v2_repository)
app.state.parent_assistant_service = assistant_service

# In-memory job store  {job_id: {status, progress, message, data, error}}
_jobs: dict[str, dict] = {}


# ─── Schemas ──────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    product_url: Optional[str] = None
    product_id: Optional[str] = None


def _resolve_product_id(product_url: Optional[str], product_id: Optional[str]) -> Optional[str]:
    if product_id:
        return str(product_id).strip()
    if product_url:
        return extract_product_id_from_url(product_url)
    return None


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.post("/api/analyze/start")
async def analyze_start(payload: AnalyzeRequest):
    """
    Khởi chạy job phân tích bất đồng bộ.
    Trả về job_id ngay lập tức để frontend theo dõi qua SSE.
    """
    # Validate input
    if not payload.product_url and not payload.product_id:
        raise HTTPException(status_code=400, detail="Cần cung cấp product_url hoặc product_id")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Khởi động...",
        "data": None,
        "error": None,
    }

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_analysis,
        job_id,
        payload.product_url,
        payload.product_id,
    )
    return {"job_id": job_id}


def _run_analysis(
    job_id: str,
    product_url: Optional[str],
    product_id: Optional[str],
):
    """Chạy trong thread-pool. Ghi progress vào _jobs[job_id]."""

    def progress(pct: int, msg: str):
        if job_id in _jobs:
            _jobs[job_id]["progress"] = pct
            _jobs[job_id]["message"]  = msg

    try:
        result = service.analyze_product(
            product_url=product_url,
            product_id=product_id,
            progress_callback=progress,
        )
        serializable_result = make_json_serializable(result)
        progress(100, "Hoàn thành!")
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["data"]   = serializable_result

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"]  = str(exc)


@app.get("/api/analyze/progress/{job_id}")
async def analyze_progress(job_id: str):
    """
    SSE stream — client nhận progress real-time.
    Tự đóng khi job kết thúc (done / error).
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        while True:
            job = _jobs.get(job_id)
            if job is None:
                break

            status = job.get("status", "running")
            payload_data = {
                "progress": int(job.get("progress", 0)),
                "message":  str(job.get("message", "")),
                "status":   str(status),
                "data":     None,
                "error":    str(job.get("error")) if job.get("error") else None,
            }

            if status == "done" and job.get("data") is not None:
                payload_data["data"] = make_json_serializable(job.get("data"))

            try:
                payload = json.dumps(payload_data, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            except TypeError as e:
                print(f"[SSE] JSON serialization error: {e}")
                fallback = {
                    "progress": payload_data["progress"],
                    "message":  payload_data["message"],
                    "status":   status,
                    "data":     None,
                    "error":    f"Serialization error: {e}",
                }
                yield f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n"

            if status in ("done", "error"):
                await asyncio.sleep(2)
                _jobs.pop(job_id, None)
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/proxy-image")
async def proxy_image(url: str):
    """
    Proxy ảnh từ Tiki CDN — tránh CORS / hotlink protection.
    Frontend dùng: /api/proxy-image?url=<encoded_tiki_image_url>
    """
    if not url:
        raise HTTPException(status_code=400, detail="Missing url param")

    from urllib.parse import urlparse
    parsed = urlparse(url)
    allowed_domains = ("salt.tikicdn.com", "tikicdn.com", "tiki.vn")
    if not any(parsed.netloc.endswith(d) for d in allowed_domains):
        raise HTTPException(status_code=403, detail="Domain not allowed")

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                url,
                headers={
                    "Referer": "https://tiki.vn/",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36"
                    ),
                },
                follow_redirects=True,
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Upstream error")

        content_type = resp.headers.get("content-type", "image/jpeg")
        from fastapi.responses import Response
        return Response(
            content=resp.content,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Proxy fetch error: {e}") from e


@app.post("/api/analyze")
def analyze_sync(payload: AnalyzeRequest):
    """Fallback endpoint đồng bộ (backward compat / testing)."""
    try:
        result = service.analyze_product(
            product_url=payload.product_url,
            product_id=payload.product_id,
        )
        return {"ok": True, "data": make_json_serializable(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ── Health check endpoint ──────────────────────────────────────────────────────
@app.post("/api/assistant/purchase-advice", response_model=PurchaseAdviceResponse)
def purchase_advice(payload: PurchaseAdviceRequest):
    product_id = _resolve_product_id(payload.product_url, payload.product_id)
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


@app.post("/api/assistant/product-fit", response_model=ProductFitResponse)
def product_fit(payload: ProductFitRequest):
    product_id = _resolve_product_id(payload.product_url, payload.product_id)
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


@app.post("/api/assistant/chat", response_model=ChatResponse)
def assistant_chat(payload: ChatRequest):
    product_id = _resolve_product_id(payload.product_url, payload.product_id)
    if not product_id:
        raise HTTPException(status_code=400, detail="Can cung cap product_url hoac product_id hop le")

    try:
        if payload.product_url:
            service.analyze_product(product_url=payload.product_url, product_id=product_id)
        answer = assistant_service.answer_chat(product_id, payload.question, use_llm=payload.use_llm)
        if answer is None:
            raise ValueError(
                "Chua co context trong PostgreSQL cho product_id nay. Hay phan tich san pham truoc "
                "hoac cau hinh DATABASE_URL de luu ket qua."
            )
        return answer
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/products/{product_id}/risk", response_model=RiskResponse)
def product_risk(product_id: str):
    ctx = service._v2_repository.load_product_context(product_id)
    if ctx is None:
        return {"product_id": product_id, "risks": []}
    return {"product_id": product_id, "risks": ctx.get("risks") or []}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "v2_database": service._v2_repository.status,
        "v2_database_error": get_database_init_error(),
        "description": "Tiki ABSA — Live crawl mode (no static CSV required for reviews)",
    }


# ── Dev server ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Tiki ABSA Dashboard v2.1 (Live crawl mode)")
    print("📍 Open http://localhost:8000 in your browser")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
