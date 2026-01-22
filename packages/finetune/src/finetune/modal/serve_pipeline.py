"""Hybrid NER pipeline endpoint for ADS query generation.

This replaces the vLLM-based model endpoint with a deterministic pipeline:
1. NER extraction → IntentSpec
2. Few-shot retrieval → gold examples
3. Template assembly → valid ADS query

The pipeline is CPU-only and fast (<200ms warm, <3s cold start).

## Deployment

    modal deploy packages/finetune/src/finetune/modal/serve_pipeline.py

## Testing

    curl -X POST https://<your-workspace>--nls-finetune-pipeline-v1-query.modal.run \
        -H "Content-Type: application/json" \
        -d '{"nl_text": "papers about exoplanets"}'
"""

import time
from pathlib import Path

import modal

# Modal app
app = modal.App("nls-finetune-pipeline")

# Volume for gold_examples.json (mounted from data volume)
data_volume = modal.Volume.from_name("nls-query-data", create_if_missing=True)

# Get the path to the finetune package source (for local copy)
PACKAGE_ROOT = Path(__file__).parent.parent  # packages/finetune/src/finetune

# CPU-only image with minimal dependencies
# Copy the entire finetune package to /root/finetune
PIPELINE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pydantic>=2.0",
        "fastapi",  # Required for web endpoints
        "httpx",  # Required for resolver ADS API calls
    )
    .env(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "/root",
            # Override gold_examples path to use Modal volume
            "GOLD_EXAMPLES_PATH": "/data/gold_examples.json",
        }
    )
    .add_local_dir(str(PACKAGE_ROOT), "/root/finetune", copy=True)
)


@app.cls(
    image=PIPELINE_IMAGE,
    volumes={"/data": data_volume},
    scaledown_window=300,  # 5 minutes
    min_containers=1,  # Keep warm for low latency
    timeout=30,  # Pipeline should complete in <1s
)
class PipelineServer:
    """Modal class for hybrid NER pipeline serving."""

    _initialized: bool = False
    _retrieval_index: object = None

    @modal.enter()
    def setup(self):
        """Preload gold_examples.json, synonym maps, and FIELD_ENUMS at container startup.

        This method is called once when the container starts. All expensive
        initialization is done here to minimize latency on requests.
        """
        print("[Pipeline] Initializing hybrid NER pipeline...")
        start = time.perf_counter()

        # Preload FIELD_ENUMS for constraint validation
        from finetune.domains.scix.field_constraints import FIELD_ENUMS

        print(f"[Pipeline] Loaded FIELD_ENUMS with {len(FIELD_ENUMS)} field types")

        # Preload NER synonym maps
        from finetune.domains.scix.ner import (
            BIBGROUP_SYNONYMS,
            DATABASE_SYNONYMS,
            DOCTYPE_SYNONYMS,
            PROPERTY_SYNONYMS,
        )

        print(
            f"[Pipeline] Loaded synonym maps: "
            f"{len(PROPERTY_SYNONYMS)} property, "
            f"{len(DOCTYPE_SYNONYMS)} doctype, "
            f"{len(BIBGROUP_SYNONYMS)} bibgroup, "
            f"{len(DATABASE_SYNONYMS)} database"
        )

        # Preload retrieval index (this triggers gold_examples.json loading)
        from finetune.domains.scix.retrieval import get_index

        self._retrieval_index = get_index()
        print(
            f"[Pipeline] Retrieval index loaded with {self._retrieval_index.num_examples} examples"
        )

        self._initialized = True
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[Pipeline] Initialization complete in {elapsed:.1f}ms")

    @modal.method()
    def process_query(self, nl_text: str) -> dict:
        """Process natural language query through hybrid pipeline.

        Args:
            nl_text: Natural language search query

        Returns:
            Dictionary with keys:
            - query: Final ADS query string
            - intent: Extracted IntentSpec as dict
            - retrieved_examples: Top-k similar examples
            - debug_info: Timing and debugging information
            - success: Whether pipeline completed successfully
            - error: Error message if success is False
        """
        start = time.perf_counter()

        try:
            # Import pipeline (inline for Modal deployment)
            from finetune.domains.scix.pipeline import process_query

            result = process_query(nl_text)

            return {
                "query": result.final_query,
                "intent": result.intent.to_dict(),
                "retrieved_examples": [ex.to_dict() for ex in result.retrieved_examples],
                "debug_info": result.debug_info.to_dict(),
                "success": result.success,
                "error": result.error,
            }

        except ImportError as e:
            # Fallback: simple topic search if pipeline not available
            print(f"[Pipeline] Import error, using fallback: {e}")

            # Basic fallback: wrap in abs:""
            clean_text = nl_text.strip().replace('"', '\\"')
            fallback_query = f'abs:"{clean_text}"'

            return {
                "query": fallback_query,
                "intent": {"raw_user_text": nl_text, "free_text_terms": [nl_text]},
                "retrieved_examples": [],
                "debug_info": {
                    "total_time_ms": (time.perf_counter() - start) * 1000,
                    "fallback_reason": f"Import error: {e}",
                },
                "success": True,
                "error": None,
            }

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "query": "",
                "intent": {},
                "retrieved_examples": [],
                "debug_info": {"total_time_ms": elapsed, "error_type": type(e).__name__},
                "success": False,
                "error": str(e),
            }


@app.function(
    image=PIPELINE_IMAGE,
    volumes={"/data": data_volume},
    scaledown_window=300,
    min_containers=1,
    timeout=30,
)
@modal.fastapi_endpoint(method="POST")
def query(request: dict) -> dict:
    """HTTP endpoint for pipeline queries.

    POST /query
    Body: {"nl_text": "papers about exoplanets"}

    Response: {
        "query": "abs:exoplanets",
        "intent": {...},
        "retrieved_examples": [...],
        "debug_info": {...},
        "success": true
    }
    """
    nl_text = request.get("nl_text", "")

    if not nl_text or not isinstance(nl_text, str):
        return {
            "query": "",
            "intent": {},
            "retrieved_examples": [],
            "debug_info": {},
            "success": False,
            "error": "Missing or invalid nl_text parameter",
        }

    # Use the class method for processing
    server = PipelineServer()
    return server.process_query.remote(nl_text)


# /v1/query endpoint as specified in US-007 acceptance criteria
@app.function(
    image=PIPELINE_IMAGE,
    volumes={"/data": data_volume},
    scaledown_window=300,
    min_containers=1,
    timeout=30,
)
@modal.fastapi_endpoint(method="POST", label="v1-query")
def v1_query(request: dict) -> dict:
    """HTTP endpoint for pipeline queries at /v1/query path.

    POST /v1/query
    Body: {"nl_text": "papers about exoplanets"}

    Response: {
        "query": "abs:exoplanets",
        "intent": {...},
        "retrieved_examples": [...],
        "debug_info": {...},
        "success": true
    }
    """
    nl_text = request.get("nl_text", "")

    if not nl_text or not isinstance(nl_text, str):
        return {
            "query": "",
            "intent": {},
            "retrieved_examples": [],
            "debug_info": {},
            "success": False,
            "error": "Missing or invalid nl_text parameter",
        }

    # Process through pipeline
    try:
        from finetune.domains.scix.pipeline import process_query

        result = process_query(nl_text)

        return {
            "query": result.final_query,
            "intent": result.intent.to_dict(),
            "retrieved_examples": [ex.to_dict() for ex in result.retrieved_examples],
            "debug_info": result.debug_info.to_dict(),
            "success": result.success,
            "error": result.error,
        }

    except Exception as e:
        # Fallback to simple topic query
        clean_text = nl_text.strip().replace('"', '\\"')
        fallback = f'abs:"{clean_text}"'

        return {
            "query": fallback,
            "intent": {"raw_user_text": nl_text, "free_text_terms": [nl_text]},
            "retrieved_examples": [],
            "debug_info": {"error": str(e), "fallback": True},
            "success": True,
            "error": None,
        }


# OpenAI-compatible endpoint for drop-in replacement
@app.function(
    image=PIPELINE_IMAGE,
    volumes={"/data": data_volume},
    scaledown_window=300,
    min_containers=1,
    timeout=30,
)
@modal.fastapi_endpoint(method="POST", label="v1-chat-completions")
def chat_completions(request: dict) -> dict:
    """OpenAI-compatible chat completions endpoint.

    This allows drop-in replacement of the vLLM endpoint in nl-search.ts
    by matching the OpenAI chat completions API format.

    POST /v1/chat/completions
    Body: {
        "model": "pipeline",
        "messages": [{"role": "user", "content": "papers about exoplanets"}]
    }

    Response: {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "abs:exoplanets"
            }
        }],
        "pipeline_result": {...}  # Full pipeline result for debugging
    }
    """
    messages = request.get("messages", [])

    # Extract user message
    nl_text = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Strip date prefix if present (from nl-search.ts format)
            if content.startswith("Query:"):
                nl_text = content.split("\n")[0].replace("Query:", "").strip()
            else:
                nl_text = content.strip()
            break

    if not nl_text:
        return {
            "choices": [{"message": {"role": "assistant", "content": ""}}],
            "error": "No user message found",
        }

    # Process through pipeline
    try:
        from finetune.domains.scix.pipeline import process_query

        result = process_query(nl_text)
        query = result.final_query

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": query},
                }
            ],
            "pipeline_result": result.to_dict(),
        }

    except Exception as e:
        # Fallback to simple topic query
        clean_text = nl_text.strip().replace('"', '\\"')
        fallback = f'abs:"{clean_text}"'

        return {
            "choices": [{"message": {"role": "assistant", "content": fallback}}],
            "error": str(e),
            "fallback": True,
        }


if __name__ == "__main__":
    # Local testing
    print("Testing pipeline locally...")

    test_queries = [
        "papers about exoplanets",
        "refereed papers on dark matter since 2020",
        "papers citing Hawking radiation paper",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        from finetune.domains.scix.pipeline import process_query

        result = process_query(q)
        print(f"  → {result.final_query}")
        print(f"  Time: {result.debug_info.total_time_ms:.1f}ms")
