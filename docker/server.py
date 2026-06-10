#!/usr/bin/env python3
"""NLS Inference Server - Local deployment without Modal.

Routing: requests are served by the hybrid NER pipeline first (deterministic,
~5ms). The fine-tuned model is a fallback for queries the pipeline extracts
with low confidence. Both /v1/chat/completions (what nectar consumes) and
/pipeline route this way, so integrations get the fast path without changes.

Endpoints:
    POST /v1/chat/completions - OpenAI-compatible chat endpoint (vLLM style)
    POST /pipeline - Hybrid NER pipeline endpoint (includes debug info)
    GET /health - Health check
    GET /v1/models - List available models

Configuration (environment variables):
    MODEL_NAME       HuggingFace model id (default: adsabs/scix-nls-translator)
    DEVICE           cuda | mps | cpu (default: auto-detect)
    PORT             Server port (default: 8000)
    ROUTING_MODE     hybrid | pipeline | model (default: hybrid)
                     hybrid: pipeline first, model fallback on low confidence
                     pipeline: pipeline only (model never loaded into the path)
                     model: fine-tuned model only (pre-hybrid behavior)
    PIPELINE_CONFIDENCE_THRESHOLD
                     Fall back to the model when pipeline confidence is below
                     this value (default: 0.5)
    TELEMETRY_LOG    Optional path to a JSONL file; one record is appended per
                     request (path taken, confidence, latency, queries) to feed
                     the retraining data flywheel.

Usage:
    # With Docker (GPU):
    docker run --gpus all -p 8000:8000 nls-server

    # With Docker (CPU):
    docker run -p 8000:8000 -e DEVICE=cpu nls-server

    # Direct Python:
    MODEL_NAME=adsabs/scix-nls-translator python docker/server.py
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nls-server")

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "adsabs/scix-nls-translator")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.environ.get("PORT", 8000))
ROUTING_MODE = os.environ.get("ROUTING_MODE", "hybrid")
CONFIDENCE_THRESHOLD = float(os.environ.get("PIPELINE_CONFIDENCE_THRESHOLD", "0.5"))
TELEMETRY_LOG = os.environ.get("TELEMETRY_LOG", "")

if ROUTING_MODE not in ("hybrid", "pipeline", "model"):
    raise ValueError(f"ROUTING_MODE must be hybrid, pipeline, or model; got {ROUTING_MODE!r}")

# Try to import pipeline components (optional, for full pipeline mode)
try:
    sys.path.insert(0, "/app")
    from finetune.domains.scix.pipeline import process_query

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    logger.warning("Pipeline modules not available - using model-only mode")

if ROUTING_MODE == "pipeline" and not PIPELINE_AVAILABLE:
    raise RuntimeError("ROUTING_MODE=pipeline but pipeline modules are not importable")

app = FastAPI(
    title="NLS Inference Server",
    description="Natural Language to ADS Query translation",
    version="2.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "llm"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    chat_template_kwargs: dict = {}


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


class PipelineRequest(BaseModel):
    model: str = "pipeline"
    messages: list[ChatMessage]


class PipelineDebugInfo(BaseModel):
    ner_time_ms: float = 0
    retrieval_time_ms: float = 0
    assembly_time_ms: float = 0
    total_time_ms: float = 0
    constraint_corrections: list[str] = []
    fallback_reason: str | None = None
    raw_extracted: dict | None = None


class PipelineResult(BaseModel):
    query: str
    intent: dict = {}
    retrieved_examples: list[dict] = []
    debug_info: PipelineDebugInfo
    confidence: float = 1.0
    success: bool = True
    error: str | None = None


class PipelineResponse(BaseModel):
    choices: list[ChatChoice]
    pipeline_result: PipelineResult | None = None
    error: str | None = None
    fallback: bool = False
    path: str = "pipeline"


@dataclass
class RoutedResult:
    """Outcome of routing a query through pipeline and/or model."""

    query: str
    path: str  # "pipeline" | "model"
    confidence: float
    fallback_reason: str | None
    latency_ms: float
    pipeline_result: PipelineResult | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_model() -> None:
    """Load the fine-tuned model."""
    global model, tokenizer

    logger.info("Loading model: %s (device=%s)", MODEL_NAME, DEVICE)

    dtype = torch.float16 if DEVICE != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=DEVICE if DEVICE != "cpu" else None,
        trust_remote_code=True,
    )

    if DEVICE == "cpu":
        model = model.to("cpu")

    logger.info("Model loaded successfully on %s", DEVICE)


def generate_query(messages: list[ChatMessage], max_tokens: int = 256) -> tuple[str, int, int]:
    """Generate ADS query from chat messages with the fine-tuned model.

    Returns:
        Tuple of (generated_text, prompt_tokens, completion_tokens)
    """
    # Build prompt from messages
    message_dicts = [{"role": m.role, "content": m.content} for m in messages]
    prompt = tokenizer.apply_chat_template(
        message_dicts,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if DEVICE != "cpu":
        inputs = inputs.to(model.device)

    prompt_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated_ids = outputs[0][prompt_tokens:]
    completion_tokens = len(generated_ids)

    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Handle thinking mode output
    if "<think>" in response:
        parts = response.split("</think>")
        if len(parts) > 1:
            response = parts[-1].strip()

    # Try to extract JSON query
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            response = data.get("query", response)
    except json.JSONDecodeError:
        pass

    return response, prompt_tokens, completion_tokens


def extract_nl_query(messages: list[ChatMessage]) -> str:
    """Extract the natural-language query from chat messages.

    Handles the nectar message format: "Query: <NL>\\nDate: <date>".
    """
    user_message = next((m.content for m in messages if m.role == "user"), "")

    if "Query:" in user_message:
        return user_message.split("Query:")[1].split("\n")[0].strip()
    return user_message


def write_telemetry(record: dict) -> None:
    """Append a telemetry record to the JSONL flywheel log, if configured."""
    if not TELEMETRY_LOG:
        return
    try:
        with open(TELEMETRY_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.warning("Failed to write telemetry to %s: %s", TELEMETRY_LOG, e)


def run_pipeline(nl_query: str) -> tuple[RoutedResult | None, str | None]:
    """Run the deterministic pipeline.

    Returns:
        (RoutedResult, None) when the pipeline produced a query. The result
        carries the pipeline's confidence; the caller decides whether to
        fall back to the model.
        (None, error_reason) when the pipeline errored or produced nothing.
    """
    start_time = time.perf_counter()
    try:
        result = process_query(nl_query)
    except Exception as e:
        logger.exception("Pipeline raised for query %r", nl_query)
        return None, f"pipeline error: {e}"

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not result.final_query.strip():
        return None, "pipeline produced empty query"

    debug_info = PipelineDebugInfo(
        ner_time_ms=result.debug_info.ner_time_ms,
        retrieval_time_ms=result.debug_info.retrieval_time_ms,
        assembly_time_ms=result.debug_info.assembly_time_ms,
        total_time_ms=elapsed_ms,
        constraint_corrections=result.debug_info.constraint_corrections,
        fallback_reason=result.debug_info.fallback_reason,
    )

    pipeline_result = PipelineResult(
        query=result.final_query,
        intent=result.intent.to_dict(),
        retrieved_examples=[ex.to_dict() for ex in result.retrieved_examples],
        debug_info=debug_info,
        confidence=result.confidence,
        success=True,
    )

    return (
        RoutedResult(
            query=result.final_query,
            path="pipeline",
            confidence=result.confidence,
            fallback_reason=None,
            latency_ms=elapsed_ms,
            pipeline_result=pipeline_result,
        ),
        None,
    )


def run_model(messages: list[ChatMessage], max_tokens: int) -> RoutedResult:
    """Run the fine-tuned model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()
    response_text, prompt_tokens, completion_tokens = generate_query(messages, max_tokens)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return RoutedResult(
        query=response_text,
        path="model",
        confidence=0.0,
        fallback_reason=None,
        latency_ms=elapsed_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def route_query(messages: list[ChatMessage], max_tokens: int = 256) -> RoutedResult:
    """Route a request: pipeline first, model fallback on low confidence.

    Routing honors ROUTING_MODE:
        hybrid   - pipeline first; model when pipeline is unavailable, errors,
                   or reports confidence below PIPELINE_CONFIDENCE_THRESHOLD
        pipeline - pipeline only; low confidence is served anyway (logged)
        model    - model only (pre-hybrid behavior)
    """
    nl_query = extract_nl_query(messages)
    fallback_reason: str | None = None

    if ROUTING_MODE != "model" and PIPELINE_AVAILABLE:
        routed, error_reason = run_pipeline(nl_query)

        if routed is not None:
            low_confidence = routed.confidence < CONFIDENCE_THRESHOLD
            can_fall_back = ROUTING_MODE == "hybrid" and model is not None

            if not low_confidence or not can_fall_back:
                if low_confidence:
                    routed.fallback_reason = (
                        routed.pipeline_result.debug_info.fallback_reason
                        if routed.pipeline_result
                        else None
                    ) or "low confidence served without model fallback"
                    logger.warning(
                        "Serving low-confidence pipeline result (%.2f < %.2f): %s",
                        routed.confidence,
                        CONFIDENCE_THRESHOLD,
                        routed.fallback_reason,
                    )
                _log_routing(nl_query, routed)
                return routed

            fallback_reason = (
                routed.pipeline_result.debug_info.fallback_reason
                if routed.pipeline_result
                else None
            ) or (f"confidence {routed.confidence:.2f} below threshold {CONFIDENCE_THRESHOLD:.2f}")
        else:
            fallback_reason = error_reason
            if ROUTING_MODE == "pipeline" or model is None:
                raise HTTPException(status_code=500, detail=f"Pipeline failed: {error_reason}")

    routed = run_model(messages, max_tokens)
    routed.fallback_reason = fallback_reason
    _log_routing(nl_query, routed)
    return routed


def _log_routing(nl_query: str, routed: RoutedResult) -> None:
    """Emit one structured log line + telemetry record per request."""
    logger.info(
        "path=%s confidence=%.2f latency_ms=%.0f fallback_reason=%r nl=%r query=%r",
        routed.path,
        routed.confidence,
        routed.latency_ms,
        routed.fallback_reason,
        nl_query[:200],
        routed.query[:200],
    )
    write_telemetry(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "nl_query": nl_query,
            "generated_query": routed.query,
            "path": routed.path,
            "confidence": routed.confidence,
            "fallback_reason": routed.fallback_reason,
            "latency_ms": round(routed.latency_ms, 1),
            "routing_mode": ROUTING_MODE,
        }
    )


@app.on_event("startup")
async def startup():
    """Load model on startup (skipped in pipeline-only mode)."""
    if ROUTING_MODE == "pipeline":
        logger.info("ROUTING_MODE=pipeline; skipping model load")
        return
    try:
        load_model()
    except Exception:
        if ROUTING_MODE == "model":
            raise
        logger.exception("Model failed to load; continuing in pipeline-only degraded mode")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": model is not None,
        "device": DEVICE,
        "pipeline_available": PIPELINE_AVAILABLE,
        "routing_mode": ROUTING_MODE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "llm",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "adsabs",
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint (vLLM style).

    Routes through the hybrid pipeline first; the fine-tuned model handles
    low-confidence queries. Response shape is unchanged from the model-only
    server: choices[0].message.content is the bare ADS query string.
    """
    try:
        routed = route_query(request.messages, request.max_tokens)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("chat_completions failed")
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=routed.query))],
        usage=ChatUsage(
            prompt_tokens=routed.prompt_tokens,
            completion_tokens=routed.completion_tokens,
            total_tokens=routed.prompt_tokens + routed.completion_tokens,
        ),
    )


@app.post("/pipeline", response_model=PipelineResponse)
@app.post("/", response_model=PipelineResponse)
async def pipeline_endpoint(request: PipelineRequest):
    """Hybrid NER pipeline endpoint with debug info.

    Same routing as /v1/chat/completions, but the response includes the
    pipeline's IntentSpec, retrieved examples, and timing breakdown.
    """
    try:
        routed = route_query(request.messages)
    except HTTPException as e:
        return PipelineResponse(choices=[], error=str(e.detail))
    except Exception as e:
        logger.exception("pipeline_endpoint failed")
        return PipelineResponse(choices=[], error=str(e))

    pipeline_result = routed.pipeline_result or PipelineResult(
        query=routed.query,
        debug_info=PipelineDebugInfo(
            total_time_ms=routed.latency_ms,
            fallback_reason=routed.fallback_reason,
        ),
        confidence=routed.confidence,
    )

    return PipelineResponse(
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=routed.query))],
        pipeline_result=pipeline_result,
        fallback=routed.path == "model",
        path=routed.path,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
