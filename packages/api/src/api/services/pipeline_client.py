"""Model inference client for the fine-tuned Qwen model server."""

import json
import re
import time

import httpx

from api.config import settings
from api.models.inference import InferenceResponse


async def model_inference(query: str, date: str, model_id: str = "llm") -> InferenceResponse:
    """Call the fine-tuned model server (docker/server.py or vLLM).

    The server is expected to be OpenAI-compatible at MODEL_ENDPOINT.

    Args:
        query: Natural language query
        date: Current date string (YYYY-MM-DD)
        model_id: Model identifier for the response (does not affect routing)

    Returns:
        InferenceResponse with the generated ADS query
    """
    if not settings.model_endpoint:
        return InferenceResponse(
            sourcegraph_query="[MODEL_ENDPOINT not configured]",
            model_id=model_id,
            latency_ms=0,
        )

    user_content = f"Query: {query}\nDate: {date}"
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.model_endpoint}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        return InferenceResponse(
            sourcegraph_query=f"[Model server unreachable: {e}]",
            model_id=model_id,
            latency_ms=0,
        )

    latency_ms = (time.time() - start) * 1000
    data = resp.json()
    content = _extract_content(data)

    usage = data.get("usage", {})
    debug_info = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }

    return InferenceResponse(
        sourcegraph_query=content,
        model_id=model_id,
        latency_ms=latency_ms,
        debug_info=debug_info,
    )


def _extract_content(data: dict) -> str:
    """Extract the query string from an OpenAI-compatible response.

    Handles:
    - <think>...</think> tags (Qwen reasoning traces)
    - JSON {"query": "..."} wrapper
    - Plain text fallback
    """
    raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "query" in parsed:
            return parsed["query"]
    except (json.JSONDecodeError, TypeError):
        pass

    return cleaned
