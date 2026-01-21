"""Modal inference client."""

import time

import httpx

from api.config import settings
from api.models.inference import InferenceResponse


async def modal_inference(
    query: str,
    date: str,
    model_id: str = "fine-tuned",
) -> InferenceResponse:
    """Call Modal inference endpoint."""
    if not settings.modal_inference_endpoint:
        # Return placeholder for development
        return InferenceResponse(
            sourcegraph_query=f"[PLACEHOLDER] repo:example {query}",
            model_id=model_id,
            latency_ms=0,
        )

    start = time.time()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.modal_inference_endpoint,
            json={
                "query": query,
                "date": date,
                "model_id": model_id,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

    latency_ms = (time.time() - start) * 1000

    return InferenceResponse(
        sourcegraph_query=data.get("sourcegraph_query", ""),
        model_id=model_id,
        latency_ms=latency_ms,
    )
