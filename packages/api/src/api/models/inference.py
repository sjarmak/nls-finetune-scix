"""Inference request/response models."""

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    """Request for query generation."""

    query: str
    date: str | None = None
    model_id: str = "fine-tuned"  # or "gpt-4o-mini", etc.


class InferenceResponse(BaseModel):
    """Response from query generation."""

    sourcegraph_query: str
    model_id: str
    latency_ms: float


class CompareRequest(BaseModel):
    """Request for side-by-side comparison."""

    query: str
    date: str | None = None
    model_ids: list[str]  # e.g., ["fine-tuned", "gpt-4o-mini"]


class CompareResponse(BaseModel):
    """Response with multiple model outputs."""

    results: list[InferenceResponse]
