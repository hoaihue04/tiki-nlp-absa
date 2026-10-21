from __future__ import annotations

import asyncio
import concurrent.futures
import json
import traceback
import uuid
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.core.serialization import make_json_serializable
from app.web.dashboard_service import DashboardService


class AnalysisJobManager:
    def __init__(self, service: DashboardService, max_workers: int = 4) -> None:
        self._service = service
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, dict] = {}

    async def start(self, product_url: Optional[str], product_id: Optional[str]) -> dict[str, str]:
        if not product_url and not product_id:
            raise HTTPException(status_code=400, detail="Cần cung cấp product_url hoặc product_id")

        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "Khởi động...",
            "data": None,
            "error": None,
        }

        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            self._executor,
            self._run_analysis,
            job_id,
            product_url,
            product_id,
        )
        return {"job_id": job_id}

    def _run_analysis(
        self,
        job_id: str,
        product_url: Optional[str],
        product_id: Optional[str],
    ) -> None:
        def progress(pct: int, msg: str) -> None:
            if job_id in self._jobs:
                self._jobs[job_id]["progress"] = pct
                self._jobs[job_id]["message"] = msg

        try:
            result = self._service.analyze_product(
                product_url=product_url,
                product_id=product_id,
                progress_callback=progress,
            )
            serializable_result = make_json_serializable(result)
            progress(100, "Hoàn thành!")
            self._jobs[job_id]["status"] = "done"
            self._jobs[job_id]["data"] = serializable_result
        except Exception as exc:
            traceback.print_exc()
            self._jobs[job_id]["status"] = "error"
            self._jobs[job_id]["error"] = str(exc)

    def progress_stream(self, job_id: str) -> StreamingResponse:
        if job_id not in self._jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        async def event_stream():
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    break

                status = job.get("status", "running")
                payload_data = {
                    "progress": int(job.get("progress", 0)),
                    "message": str(job.get("message", "")),
                    "status": str(status),
                    "data": None,
                    "error": str(job.get("error")) if job.get("error") else None,
                }

                if status == "done" and job.get("data") is not None:
                    payload_data["data"] = make_json_serializable(job.get("data"))

                try:
                    payload = json.dumps(payload_data, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                except TypeError as e:
                    fallback = {
                        "progress": payload_data["progress"],
                        "message": payload_data["message"],
                        "status": status,
                        "data": None,
                        "error": f"Serialization error: {e}",
                    }
                    yield f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n"

                if status in ("done", "error"):
                    await asyncio.sleep(2)
                    self._jobs.pop(job_id, None)
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
