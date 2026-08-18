from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ParentProfile(BaseModel):
    age_months: Optional[int] = Field(default=None, ge=0, le=216)
    weight_kg: Optional[float] = Field(default=None, ge=0)
    sensitive_skin: bool = False
    budget_vnd: Optional[int] = Field(default=None, ge=0)
    priorities: list[str] = Field(default_factory=list)


class PurchaseAdviceRequest(BaseModel):
    product_id: Optional[str] = None
    product_url: Optional[str] = None
    parent_profile: ParentProfile = Field(default_factory=ParentProfile)
    use_llm: bool = True


class ProductFitRequest(BaseModel):
    product_id: Optional[str] = None
    product_url: Optional[str] = None
    question: str = ""
    parent_profile: ParentProfile = Field(default_factory=ParentProfile)
    use_llm: bool = True


class ChatRequest(BaseModel):
    product_id: Optional[str] = None
    product_url: Optional[str] = None
    question: str
    use_llm: bool = True


class Evidence(BaseModel):
    review_id: str = ""
    aspect: str = ""
    aspect_label: str = ""
    sentiment: str = ""
    confidence: float = 0.0
    text: str = ""


class Citation(BaseModel):
    aspect: str = ""
    evidence: str = ""


class AssistantStructuredAnswer(BaseModel):
    summary: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    recommendation: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"
    confidence_reasons: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class PurchaseAdviceResponse(BaseModel):
    product_id: str
    conclusion: Literal["Nen mua", "Can nhac", "Nen tranh"]
    score: float
    confidence: float
    summary: str
    structured_answer: AssistantStructuredAnswer = Field(default_factory=AssistantStructuredAnswer)
    reasons_to_buy: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    llm_used: bool = False
    llm_model: str = ""


class ProductFitResponse(BaseModel):
    product_id: str
    fit_level: Literal["Phu hop", "Can nhac", "Chua du du lieu"]
    answer: str
    structured_answer: AssistantStructuredAnswer = Field(default_factory=AssistantStructuredAnswer)
    evidence: list[Evidence] = Field(default_factory=list)
    llm_used: bool = False
    llm_model: str = ""


class ChatResponse(BaseModel):
    product_id: str
    answer: str
    structured_answer: AssistantStructuredAnswer = Field(default_factory=AssistantStructuredAnswer)
    evidence: list[Evidence] = Field(default_factory=list)
    retrieval_strategy: str = "postgres_only"
    qdrant_used: bool = False
    cache_hit: bool = False
    llm_used: bool = False
    llm_model: str = ""


class RiskResponse(BaseModel):
    product_id: str
    risks: list[dict[str, Any]] = Field(default_factory=list)
