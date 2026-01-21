"""vLLM-based inference endpoint for optimized latency."""

import modal

# Volumes (defined directly to avoid relative import issues with modal deploy)
runs_volume = modal.Volume.from_name("scix-finetune-runs", create_if_missing=True)
vllm_cache = modal.Volume.from_name("nls-vllm-cache", create_if_missing=True)
hf_cache = modal.Volume.from_name("nls-hf-cache", create_if_missing=True)

# Standard vLLM image for serving fine-tuned models
VLLM_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.11.2",
        "huggingface-hub>=0.24.0",
        "flashinfer-python==0.5.2",
    )
    .env({
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_USE_TRITON_FLASH_ATTN": "True",
    })
)

app = modal.App("nls-finetune-serve-vllm")

# Configuration
VLLM_PORT = 8000
MAX_MODEL_LEN = 512  # Sourcegraph queries are short


@app.function(
    image=VLLM_IMAGE,
    gpu="H100",  # High-performance for lowest latency
    volumes={
        "/runs": runs_volume,
        "/root/.cache/vllm": vllm_cache,
        "/root/.cache/huggingface": hf_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,  # 5 minutes
    min_containers=1,  # Keep one container always warm to avoid cold starts
    timeout=600,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=300)
def serve():
    import subprocess
    from pathlib import Path

    # Find latest merged model (by modification time, not alphabetical)
    runs_dir = Path("/runs")
    merged_runs = [
        (d, d.stat().st_mtime) for d in runs_dir.iterdir()
        if d.is_dir() and (d / "merged").exists()
    ]
    if not merged_runs:
        raise FileNotFoundError("No merged models found")

    # Sort by modification time (newest last)
    merged_runs.sort(key=lambda x: x[1])
    latest_run = merged_runs[-1][0].name
    model_path = f"/runs/{latest_run}/merged"
    print(f"Using model: {model_path}")

    cmd = [
        "vllm", "serve",
        model_path,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.95",  # Higher util for better KV cache
        "--trust-remote-code",
        "--served-model-name", "llm",  # Simpler model name for API calls
        "--enable-prefix-caching",  # Cache system prompt KV states
        # Small model latency optimizations
        "--max-num-batched-tokens", "512",  # Optimize for short sequences
        "--max-num-seqs", "4",  # Limit concurrent batching for latency
        "--disable-log-requests",  # Reduce logging overhead
        "--disable-cascade-attn",  # Prevent numerical issues causing repetition
    ]

    print(f"Starting vLLM: {' '.join(cmd)}")
    subprocess.Popen(cmd)
