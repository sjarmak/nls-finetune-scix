"""Inference API routes."""

from datetime import datetime

from fastapi import APIRouter

from api.models.inference import (
    CompareRequest,
    CompareResponse,
    InferenceRequest,
    InferenceResponse,
)
from api.services.modal_client import modal_inference
from api.services.openai_client import openai_inference

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/generate", response_model=InferenceResponse)
async def generate_query(request: InferenceRequest) -> InferenceResponse:
    """Generate a Sourcegraph query from natural language."""
    date = request.date or datetime.now().strftime("%Y-%m-%d")

    if request.model_id == "gpt-4o-mini":
        return await openai_inference(request.query, date)
    else:
        return await modal_inference(request.query, date, request.model_id)


@router.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest) -> CompareResponse:
    """Compare multiple models on the same query."""
    date = request.date or datetime.now().strftime("%Y-%m-%d")
    results = []

    for model_id in request.model_ids:
        if model_id == "gpt-4o-mini":
            result = await openai_inference(request.query, date)
        else:
            result = await modal_inference(request.query, date, model_id)
        results.append(result)

    return CompareResponse(results=results)
