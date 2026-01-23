#!/usr/bin/env python3
"""NLS Inference Server - Local deployment without Modal.

This server provides both the pipeline and vLLM-compatible endpoints
that nectar expects. Run with Docker or directly with Python.

Endpoints:
    POST /v1/chat/completions - OpenAI-compatible chat endpoint (vLLM style)
    POST /pipeline - Hybrid NER pipeline endpoint
    GET /health - Health check
    GET /v1/models - List available models

Usage:
    # With Docker (GPU):
    docker run --gpus all -p 8000:8000 nls-server

    # With Docker (CPU):
    docker run -p 8000:8000 -e DEVICE=cpu nls-server

    # Direct Python:
    MODEL_NAME=adsabs/scix-nls-translator python docker/server.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "adsabs/scix-nls-translator")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.environ.get("PORT", 8000))

# Try to import pipeline components (optional, for full pipeline mode)
try:
    sys.path.insert(0, "/app")
    from finetune.domains.scix.pipeline import process_query
    from finetune.domains.scix.validate import lint_query, validate_field_constraints
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("Pipeline modules not available - using model-only mode")

app = FastAPI(
    title="NLS Inference Server",
    description="Natural Language to ADS Query translation",
    version="1.0.0",
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
    success: bool = True
    error: str | None = None


class PipelineResponse(BaseModel):
    choices: list[ChatChoice]
    pipeline_result: PipelineResult | None = None
    error: str | None = None
    fallback: bool = False


def load_model():
    """Load the fine-tuned model."""
    global model, tokenizer

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

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

    print(f"Model loaded successfully on {DEVICE}")


def generate_query(messages: list[ChatMessage], max_tokens: int = 256) -> tuple[str, int, int]:
    """Generate ADS query from chat messages.

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


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    load_model()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "pipeline_available": PIPELINE_AVAILABLE,
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
    """OpenAI-compatible chat completions endpoint (vLLM style)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()
        response_text, prompt_tokens, completion_tokens = generate_query(
            request.messages, request.max_tokens
        )
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"[vLLM] Generated in {elapsed_ms:.0f}ms: {response_text[:100]}...")

        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=response_text)
                )
            ],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except Exception as e:
        print(f"[vLLM] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline", response_model=PipelineResponse)
@app.post("/", response_model=PipelineResponse)
async def pipeline_endpoint(request: PipelineRequest):
    """Hybrid NER pipeline endpoint.

    If pipeline modules are available, uses the full NER + retrieval pipeline.
    Otherwise falls back to the fine-tuned model.
    """
    try:
        # Extract user query from messages
        user_message = next(
            (m.content for m in request.messages if m.role == "user"), ""
        )

        # Parse query from "Query: X\nDate: Y" format
        nl_query = user_message
        if "Query:" in user_message:
            nl_query = user_message.split("Query:")[1].split("\n")[0].strip()

        start_time = time.time()

        if PIPELINE_AVAILABLE:
            # Use full pipeline
            try:
                result = process_query(nl_query)
                elapsed_ms = (time.time() - start_time) * 1000

                debug_info = PipelineDebugInfo(
                    ner_time_ms=result.debug_info.get("ner_time_ms", 0),
                    retrieval_time_ms=result.debug_info.get("retrieval_time_ms", 0),
                    assembly_time_ms=result.debug_info.get("assembly_time_ms", 0),
                    total_time_ms=elapsed_ms,
                    constraint_corrections=result.debug_info.get("constraint_corrections", []),
                    fallback_reason=result.debug_info.get("fallback_reason"),
                )

                pipeline_result = PipelineResult(
                    query=result.final_query,
                    intent=result.intent.__dict__ if hasattr(result, 'intent') else {},
                    retrieved_examples=[],
                    debug_info=debug_info,
                    success=True,
                )

                print(f"[Pipeline] Generated in {elapsed_ms:.0f}ms: {result.final_query}")

                return PipelineResponse(
                    choices=[
                        ChatChoice(
                            message=ChatMessage(role="assistant", content=result.final_query)
                        )
                    ],
                    pipeline_result=pipeline_result,
                )
            except Exception as e:
                print(f"[Pipeline] Error, falling back to model: {e}")
                # Fall through to model fallback

        # Fallback to fine-tuned model
        response_text, _, _ = generate_query(request.messages, max_tokens=256)
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"[Pipeline-Fallback] Generated in {elapsed_ms:.0f}ms: {response_text}")

        return PipelineResponse(
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=response_text)
                )
            ],
            pipeline_result=PipelineResult(
                query=response_text,
                debug_info=PipelineDebugInfo(total_time_ms=elapsed_ms),
            ),
            fallback=True,
        )

    except Exception as e:
        print(f"[Pipeline] Error: {e}")
        return PipelineResponse(
            choices=[],
            error=str(e),
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
