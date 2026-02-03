"""Inference API routes."""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from api.models.inference import (
    CompareRequest,
    CompareResponse,
    InferenceRequest,
    InferenceResponse,
)
from api.services.openai_client import openai_inference
from api.services.pipeline_client import model_inference

router = APIRouter(prefix="/inference", tags=["inference"])

SUPPORTED_MODELS = {"fine-tuned", "gpt-4o-mini", "base-qwen"}


async def _dispatch(query: str, date: str, model_id: str) -> InferenceResponse:
    """Route a query to the appropriate inference backend."""
    if model_id in ("fine-tuned", "base-qwen"):
        return await model_inference(query, date, model_id)
    if model_id == "gpt-4o-mini":
        return await openai_inference(query, date)
    raise HTTPException(
        status_code=400,
        detail=f"Unknown model_id '{model_id}'. Supported: {sorted(SUPPORTED_MODELS)}",
    )


@router.post("/generate", response_model=InferenceResponse)
async def generate_query(request: InferenceRequest) -> InferenceResponse:
    """Generate an ADS query from natural language."""
    date = request.date or datetime.now().strftime("%Y-%m-%d")
    return await _dispatch(request.query, date, request.model_id)


@router.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest) -> CompareResponse:
    """Compare multiple models on the same query."""
    date = request.date or datetime.now().strftime("%Y-%m-%d")
    results = [await _dispatch(request.query, date, model_id) for model_id in request.model_ids]
    return CompareResponse(results=results)
