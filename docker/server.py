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
    DEVICE           cuda | mps | cpu (default: auto-detect when the model
                     loads; cpu when torch is not installed). torch and
                     transformers are imported only when the model loads, so
                     ROUTING_MODE=pipeline runs without them.
    PORT             Server port (default: 8000)
    ROUTING_MODE     hybrid | pipeline | model (default: hybrid)
                     hybrid: pipeline first, model fallback on low confidence
                     pipeline: pipeline only (model never loaded into the path)
                     model: fine-tuned model only (pre-hybrid behavior)
    PIPELINE_CONFIDENCE_THRESHOLD
                     Fall back to the model when pipeline confidence is below
                     this value (default: 0.5). When Jev answered, pipeline
                     confidence is min(structural confidence, Jev operator
                     confidence); otherwise it is the structural confidence.
    TELEMETRY_LOG    Optional path to a JSONL file; one record is appended per
                     request (path taken, confidence, latency, queries) to feed
                     the retraining data flywheel. Request rows carry
                     record_type "request"; shadow rows carry
                     record_type "intent_shadow" and share the request_id.
    INTENT_BACKEND   regex | jev | jev_gated (default: regex). The Jev backends
                     classify operator and enum fields with TypeSafe System One
                     and also pick recency, highly cited, first author and the
                     topic span; they need TYPESAFE_API_KEY. jev_gated calls
                     it only when the regex extractor finds no operator or
                     scores below the confidence threshold. Relative years
                     end at the prompt's "Date: YYYY-MM-DD" line (400 when
                     malformed, the current year when absent).
                     A failed Jev call (HTTP error, timeout, network error or
                     malformed response) falls back to the regex intent; the
                     reason is in debug_info.classifier_error and telemetry.
                     With ADS_API_KEY set, the Jev backends also resolve a
                     paper the request names ("papers citing the original
                     TRAPPIST-1 paper"): ADS offers the most-cited candidates
                     and Jev picks one, which becomes citations(bibcode:...).
                     A failed lookup keeps the topic search; the reason is in
                     debug_info.paper_lookup_error.
    PAPER_LOOKUP_TIMEOUT_S
                     httpx timeout for each ADS candidate search in seconds
                     (default: 3.0).
    JEV_CACHE_PATH   Append-only JSONL cache for System One responses
                     (default: data/cache/jev_systemone.jsonl, relative to the
                     working directory). Set it empty to disable the cache;
                     the Docker image does, because the file grows without
                     bound and a container loses it on restart anyway.
    JEV_TIMEOUT_S    Wall-clock bound on each System One call in seconds
                     (default: 2.0; measured p95 is about 223 ms). A timeout
                     falls back to the regex intent.
    SHADOW_INTENT_BACKEND
                     Unset (default) | jev | jev_gated. Shadow mode: the served
                     response still uses the regex intent (requires
                     INTENT_BACKEND=regex and TELEMETRY_LOG), and every pipeline
                     request also queues a run of this backend on a background
                     thread pool, off the request path, so it adds no latency to
                     the response. Each run appends an "intent_shadow" row with
                     both IntentSpecs, the disagreeing fields (operator, enum
                     fields, years, topic terms, first author, citation
                     floor), Jev's operator confidence and the shadow latency.
                     At most 32 runs may be pending; beyond that runs are
                     skipped and logged. Shadow failures
                     are logged, never raised. Needs TYPESAFE_API_KEY.
                     Summarize with scripts/summarize_intent_shadow.py.

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
import re
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nls-server")


def _default_device() -> str:
    """cuda when torch sees a GPU, else cpu (also when torch is not installed:
    a pipeline-only server needs no torch, and load_model raises on its own)."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "adsabs/scix-nls-translator")
# Empty means auto-detect, resolved by load_model so torch stays unimported
# until a model is actually loaded.
DEVICE = os.environ.get("DEVICE", "")
PORT = int(os.environ.get("PORT", 8000))
ROUTING_MODE = os.environ.get("ROUTING_MODE", "hybrid")
CONFIDENCE_THRESHOLD = float(os.environ.get("PIPELINE_CONFIDENCE_THRESHOLD", "0.5"))
TELEMETRY_LOG = os.environ.get("TELEMETRY_LOG", "")
INTENT_BACKEND = os.environ.get("INTENT_BACKEND", "regex")
# An empty JEV_CACHE_PATH disables the System One response cache.
JEV_CACHE_PATH = os.environ.get("JEV_CACHE_PATH", "data/cache/jev_systemone.jsonl")
JEV_TIMEOUT_S = float(os.environ.get("JEV_TIMEOUT_S", "2.0"))
SHADOW_INTENT_BACKEND = os.environ.get("SHADOW_INTENT_BACKEND", "")
# Shadow runs are fire-and-forget on a small pool; at most this many may be
# queued or running, further ones are skipped (logged) so a slow Jev cannot
# grow memory without bound.
SHADOW_WORKERS = 2
SHADOW_MAX_PENDING = 32

if ROUTING_MODE not in ("hybrid", "pipeline", "model"):
    raise ValueError(f"ROUTING_MODE must be hybrid, pipeline, or model; got {ROUTING_MODE!r}")
if INTENT_BACKEND not in ("regex", "jev", "jev_gated"):
    raise ValueError(f"INTENT_BACKEND must be regex, jev, or jev_gated; got {INTENT_BACKEND!r}")
if JEV_TIMEOUT_S <= 0:
    raise ValueError(f"JEV_TIMEOUT_S must be positive; got {JEV_TIMEOUT_S}")
if SHADOW_INTENT_BACKEND:
    if SHADOW_INTENT_BACKEND not in ("jev", "jev_gated"):
        raise ValueError(
            f"SHADOW_INTENT_BACKEND must be jev or jev_gated; got {SHADOW_INTENT_BACKEND!r}"
        )
    if INTENT_BACKEND != "regex":
        raise ValueError(
            "SHADOW_INTENT_BACKEND shadows the served regex intent; set INTENT_BACKEND=regex"
        )
    if ROUTING_MODE == "model":
        raise ValueError(
            "SHADOW_INTENT_BACKEND needs the pipeline; ROUTING_MODE=model never runs it"
        )
    if not TELEMETRY_LOG:
        raise ValueError("SHADOW_INTENT_BACKEND writes only to TELEMETRY_LOG; set TELEMETRY_LOG")

# Try to import pipeline components (optional, for full pipeline mode)
try:
    sys.path.insert(0, "/app")
    from finetune.domains.scix.pipeline import process_query

    PIPELINE_AVAILABLE = True
    PIPELINE_IMPORT_ERROR: str | None = None
except ImportError as import_error:
    PIPELINE_AVAILABLE = False
    PIPELINE_IMPORT_ERROR = f"{type(import_error).__name__}: {import_error}"
    logger.warning("Pipeline modules not available - using model-only mode: %s", import_error)

if ROUTING_MODE == "pipeline" and not PIPELINE_AVAILABLE:
    raise RuntimeError("ROUTING_MODE=pipeline but pipeline modules are not importable")

jev_client = None
if INTENT_BACKEND != "regex" or SHADOW_INTENT_BACKEND:
    if not PIPELINE_AVAILABLE:
        raise RuntimeError("the Jev intent backends need the pipeline modules")
    from pathlib import Path

    from finetune.domains.scix.jev_intent import JevClient

    jev_client = JevClient(
        api_key=os.environ.get("TYPESAFE_API_KEY", ""),
        cache_path=Path(JEV_CACHE_PATH) if JEV_CACHE_PATH else None,
        timeout_s=JEV_TIMEOUT_S,
    )

paper_search = None
if INTENT_BACKEND != "regex" and os.environ.get("ADS_API_KEY"):
    from finetune.domains.scix.paper_lookup import ADSPaperSearch

    paper_search = ADSPaperSearch(
        api_key=os.environ["ADS_API_KEY"],
        timeout_s=float(os.environ.get("PAPER_LOOKUP_TIMEOUT_S", "3.0")),
    )
elif INTENT_BACKEND != "regex":
    logger.warning("ADS_API_KEY not set: named papers stay topic searches (no paper lookup)")

shadow_executor: ThreadPoolExecutor | None = None
_shadow_slots = threading.BoundedSemaphore(SHADOW_MAX_PENDING)
if SHADOW_INTENT_BACKEND:
    from finetune.domains.scix.intent_shadow import shadow_record

    shadow_executor = ThreadPoolExecutor(
        max_workers=SHADOW_WORKERS, thread_name_prefix="intent-shadow"
    )

_telemetry_lock = threading.Lock()


def startup() -> None:
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


def shutdown() -> None:
    """Let running shadow comparisons finish (each is bounded by JEV_TIMEOUT_S
    per Jev call) and drop queued ones, then close the Jev and ADS clients."""
    if shadow_executor is not None:
        shadow_executor.shutdown(wait=True, cancel_futures=True)
    if paper_search is not None:
        paper_search.close()
    if jev_client is not None:
        jev_client.close()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    startup()
    yield
    shutdown()


app = FastAPI(
    title="NLS Inference Server",
    description="Natural Language to ADS Query translation",
    version="2.0.0",
    lifespan=lifespan,
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
    intent_backend: str = "regex"
    classifier_called: bool = False
    classifier_cached: bool = False
    classifier_error: str | None = None
    structural_confidence: float | None = None
    classifier_operator_confidence: float | None = None


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
    pipeline_debug: PipelineDebugInfo | None = None  # set whenever the pipeline ran
    # The pipeline's routing confidence when it ran; on a model fallback this is
    # the value that fell below the threshold (``confidence`` is the model's 0.0).
    pipeline_confidence: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_model() -> None:
    """Load the fine-tuned model."""
    global model, tokenizer, DEVICE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not DEVICE:
        DEVICE = _default_device()
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


# The prompt the translator was fine-tuned on (data/datasets/processed/train.jsonl).
MODEL_SYSTEM_PROMPT = 'Convert natural language to ADS search query. Output JSON: {"query": "..."}'


def model_messages(messages: list[ChatMessage]) -> list[dict]:
    """The request rewritten in the training format: system prompt, "Query:" then "Date:".

    Off that format the model answers in prose or starts a reasoning block that
    runs past the token limit, so the client's own wording is not passed on.
    """
    user_message = next((m.content for m in messages if m.role == "user"), "")
    match = re.search(r"^Date:(.*)$", user_message, re.MULTILINE)
    day = match.group(1).strip() if match else date.today().isoformat()
    return [
        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {extract_nl_query(messages)}\nDate: {day}"},
    ]


def parse_model_output(text: str) -> str | None:
    """The query from the model's first {"query": ...} answer, or None when there is none."""
    if "<think>" in text:
        text = text.partition("</think>")[2]
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        query = data.get("query") if isinstance(data, dict) else None
        if isinstance(query, str) and query.strip():
            return query.strip()
    return None


def generate_query(
    messages: list[ChatMessage], max_tokens: int = 256
) -> tuple[str | None, int, int]:
    """Generate an ADS query with the fine-tuned model.

    Returns:
        Tuple of (query or None when the output holds no query, prompt_tokens,
        completion_tokens)
    """
    import torch

    prompt = tokenizer.apply_chat_template(
        model_messages(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return parse_model_output(response), prompt_tokens, len(generated_ids)


def extract_nl_query(messages: list[ChatMessage]) -> str:
    """Extract the natural-language query from chat messages.

    Handles the nectar message format: "Query: <NL>\\nDate: <date>".
    """
    user_message = next((m.content for m in messages if m.role == "user"), "")

    if "Query:" in user_message:
        return user_message.split("Query:")[1].split("\n")[0].strip()
    return user_message


def extract_reference_year(messages: list[ChatMessage]) -> int | None:
    """Year of the "Date: YYYY-MM-DD" line in the user message, or None without one.

    Relative dates in the query ("recent", "last 5 years") end at this year.
    A malformed date is a client error (HTTP 400).
    """
    user_message = next((m.content for m in messages if m.role == "user"), "")
    match = re.search(r"^Date:(.*)$", user_message, re.MULTILINE)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1).strip()).year
    except ValueError as error:
        raise HTTPException(
            status_code=400, detail=f"Date line is not YYYY-MM-DD: {match.group(1).strip()!r}"
        ) from error


def write_telemetry(record: dict) -> None:
    """Append a telemetry record to the JSONL flywheel log, if configured."""
    if not TELEMETRY_LOG:
        return
    line = json.dumps(record) + "\n"
    try:
        with _telemetry_lock, open(TELEMETRY_LOG, "a") as f:
            f.write(line)
    except OSError as e:
        logger.warning("Failed to write telemetry to %s: %s", TELEMETRY_LOG, e)


def run_pipeline(
    nl_query: str, reference_year: int | None = None
) -> tuple[RoutedResult | None, str | None]:
    """Run the deterministic pipeline.

    Returns:
        (RoutedResult, None) when the pipeline produced a query. The result
        carries the pipeline's confidence; the caller decides whether to
        fall back to the model.
        (None, error_reason) when the pipeline errored or produced nothing.
    """
    start_time = time.perf_counter()
    try:
        result = process_query(
            nl_query, INTENT_BACKEND, jev_client, reference_year, paper_search=paper_search
        )
    except Exception as e:
        logger.exception("Pipeline raised for query %r", nl_query)
        return None, f"pipeline error: {e}"

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not result.final_query.strip():
        return None, f"pipeline produced empty query: {result.debug_info.fallback_reason}"

    debug_info = PipelineDebugInfo(
        ner_time_ms=result.debug_info.ner_time_ms,
        retrieval_time_ms=result.debug_info.retrieval_time_ms,
        assembly_time_ms=result.debug_info.assembly_time_ms,
        total_time_ms=elapsed_ms,
        constraint_corrections=result.debug_info.constraint_corrections,
        fallback_reason=result.debug_info.fallback_reason,
        intent_backend=result.debug_info.intent_backend,
        classifier_called=result.debug_info.classifier_called,
        classifier_cached=result.debug_info.classifier_cached,
        classifier_error=result.debug_info.classifier_error,
        structural_confidence=result.debug_info.structural_confidence,
        classifier_operator_confidence=result.debug_info.classifier_operator_confidence,
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
            pipeline_debug=debug_info,
        ),
        None,
    )


def schedule_shadow(
    request_id: str,
    nl_query: str,
    served_intent: dict,
    served_path: str,
    reference_year: int | None = None,
) -> Future | None:
    """Queue a shadow intent run off the request path; never blocks or raises.

    ``served_path`` is where the request was answered ("pipeline" or "model");
    only pipeline-served rows compare the shadow with what the user got.

    Returns the Future, or None when shadow mode is off or the pending limit
    is reached (the run is skipped and logged).
    """
    if shadow_executor is None:
        return None
    if not _shadow_slots.acquire(blocking=False):
        logger.warning("Shadow intent queue full (%d); skipped %r", SHADOW_MAX_PENDING, nl_query)
        return None
    try:
        future = shadow_executor.submit(
            _run_shadow, request_id, nl_query, served_intent, served_path, reference_year
        )
    except RuntimeError:
        _shadow_slots.release()
        logger.warning("Shadow intent executor is shut down; skipped %r", nl_query)
        return None
    future.add_done_callback(lambda _: _shadow_slots.release())
    return future


def _run_shadow(
    request_id: str,
    nl_query: str,
    served_intent: dict,
    served_path: str,
    reference_year: int | None,
) -> None:
    """Background task: compare the served intent with the shadow backend's.

    It runs outside any request, so it catches everything: a failure here is
    logged and must never reach a client.
    """
    try:
        record = shadow_record(
            nl_query,
            served_intent,
            SHADOW_INTENT_BACKEND,
            jev_client,
            served_path,
            reference_year,
        )
        write_telemetry(
            {"timestamp": datetime.now(UTC).isoformat(), "request_id": request_id, **record}
        )
    except Exception:
        logger.exception("Shadow intent run failed for %r", nl_query)


def run_model(messages: list[ChatMessage], max_tokens: int) -> RoutedResult | None:
    """Run the fine-tuned model; None when its output holds no query."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()
    query, prompt_tokens, completion_tokens = generate_query(messages, max_tokens)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    if query is None:
        return None

    return RoutedResult(
        query=query,
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
    reference_year = extract_reference_year(messages)
    request_id = uuid.uuid4().hex
    fallback_reason: str | None = None
    pipeline_debug: PipelineDebugInfo | None = None
    pipeline_confidence: float | None = None
    pipeline_routed: RoutedResult | None = None

    if ROUTING_MODE != "model" and PIPELINE_AVAILABLE:
        routed, error_reason = run_pipeline(nl_query, reference_year)

        if routed is not None:
            regex_intent = routed.pipeline_result.intent
            low_confidence = routed.confidence < CONFIDENCE_THRESHOLD
            can_fall_back = ROUTING_MODE == "hybrid" and model is not None

            if not low_confidence or not can_fall_back:
                if low_confidence:
                    routed.fallback_reason = (
                        f"{_low_confidence_reason(routed)}; served without model fallback"
                    )
                    logger.warning(
                        "Serving low-confidence pipeline result (%.2f < %.2f): %s",
                        routed.confidence,
                        CONFIDENCE_THRESHOLD,
                        routed.fallback_reason,
                    )
                schedule_shadow(request_id, nl_query, regex_intent, "pipeline", reference_year)
                _log_routing(request_id, nl_query, routed)
                return routed

            pipeline_routed = routed
            pipeline_debug = routed.pipeline_debug
            pipeline_confidence = routed.confidence
            fallback_reason = _low_confidence_reason(routed)
        else:
            fallback_reason = error_reason
            if ROUTING_MODE == "pipeline" or model is None:
                raise HTTPException(status_code=500, detail=f"Pipeline failed: {error_reason}")

    routed = run_model(messages, max_tokens)
    if routed is None:
        if pipeline_routed is None:
            reasons = [fallback_reason] if fallback_reason else []
            detail = "; ".join([*reasons, "model output held no query"])
            raise HTTPException(status_code=502, detail=detail)
        pipeline_routed.fallback_reason = (
            f"{fallback_reason}; model output held no query, served the pipeline result"
        )
        logger.warning("Model output held no query; serving the pipeline result")
        schedule_shadow(
            request_id, nl_query, pipeline_routed.pipeline_result.intent, "pipeline", reference_year
        )
        _log_routing(request_id, nl_query, pipeline_routed)
        return pipeline_routed
    if pipeline_routed is not None:
        schedule_shadow(
            request_id, nl_query, pipeline_routed.pipeline_result.intent, "model", reference_year
        )
    routed.fallback_reason = fallback_reason
    routed.pipeline_debug = pipeline_debug
    routed.pipeline_confidence = pipeline_confidence
    _log_routing(request_id, nl_query, routed)
    return routed


def _low_confidence_reason(routed: RoutedResult) -> str:
    """Why a pipeline result fell below the threshold, most specific first."""
    debug = routed.pipeline_debug
    if debug is not None and debug.fallback_reason:
        return debug.fallback_reason
    classifier = debug.classifier_operator_confidence if debug is not None else None
    if classifier is not None and classifier < CONFIDENCE_THRESHOLD:
        return (
            f"classifier operator confidence {classifier:.2f} "
            f"below threshold {CONFIDENCE_THRESHOLD:.2f}"
        )
    return f"confidence {routed.confidence:.2f} below threshold {CONFIDENCE_THRESHOLD:.2f}"


def _log_routing(request_id: str, nl_query: str, routed: RoutedResult) -> None:
    """Emit one structured log line + telemetry record per request."""
    debug = routed.pipeline_debug
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
            "record_type": "request",
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "nl_query": nl_query,
            "generated_query": routed.query,
            "path": routed.path,
            "confidence": routed.confidence,
            "pipeline_confidence": (
                routed.confidence if routed.path == "pipeline" else routed.pipeline_confidence
            ),
            "fallback_reason": routed.fallback_reason,
            "latency_ms": round(routed.latency_ms, 1),
            "routing_mode": ROUTING_MODE,
            "intent_backend": INTENT_BACKEND,
            "classifier_called": debug.classifier_called if debug else False,
            "classifier_cached": debug.classifier_cached if debug else False,
            "classifier_error": debug.classifier_error if debug else None,
            "structural_confidence": debug.structural_confidence if debug else None,
            "classifier_operator_confidence": (
                debug.classifier_operator_confidence if debug else None
            ),
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": model is not None,
        "device": DEVICE or "auto",
        "pipeline_available": PIPELINE_AVAILABLE,
        "pipeline_import_error": PIPELINE_IMPORT_ERROR,
        "routing_mode": ROUTING_MODE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "intent_backend": INTENT_BACKEND,
        "jev_timeout_s": JEV_TIMEOUT_S,
        "shadow_intent_backend": SHADOW_INTENT_BACKEND or None,
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
def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint (vLLM style).

    A plain ``def``: routing calls Jev and the model synchronously, so FastAPI
    runs it in its worker threadpool instead of on the event loop.

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


def _model_path_debug(routed: RoutedResult) -> PipelineDebugInfo:
    """Debug info for a model-served request, keeping the intent-stage fields
    of the pipeline run that preceded the fallback, if there was one."""
    update = {"total_time_ms": routed.latency_ms, "fallback_reason": routed.fallback_reason}
    if routed.pipeline_debug is None:
        return PipelineDebugInfo(**update)
    return routed.pipeline_debug.model_copy(update=update)


@app.post("/pipeline", response_model=PipelineResponse)
@app.post("/", response_model=PipelineResponse)
def pipeline_endpoint(request: PipelineRequest):
    """Hybrid NER pipeline endpoint with debug info (threadpool, like
    chat_completions).

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
        debug_info=_model_path_debug(routed),
        confidence=routed.confidence,
    )

    return PipelineResponse(
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=routed.query))],
        pipeline_result=pipeline_result,
        fallback=routed.path == "model",
        path=routed.path,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
