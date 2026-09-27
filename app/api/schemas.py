from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    product_url: Optional[str] = None
    product_id: Optional[str] = None
