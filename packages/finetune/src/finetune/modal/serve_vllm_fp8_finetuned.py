"""vLLM-based inference endpoint for FP8 quantized fine-tuned model.

Serves our FP8-quantized fine-tuned model from the runs volume.
"""

import modal

# Volumes
runs_volume = modal.Volume.from_name("nls-query-runs", create_if_missing=True)
vllm_cache = modal.Volume.from_name("nls-vllm-cache", create_if_missing=True)
hf_cache = modal.Volume.from_name("nls-hf-cache", create_if_missing=True)

# FP8 requires compressed-tensors for loading
VLLM_FP8_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.11.2",
        "huggingface-hub>=0.24.0",
        "flashinfer-python==0.5.2",
        "compressed-tensors>=0.6.0",  # Required for FP8 quantized models
    )
    .env(
        {
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
            "VLLM_USE_TRITON_FLASH_ATTN": "True",
        }
    )
)

app = modal.App("nls-finetune-serve-vllm-fp8-finetuned")

# Configuration
VLLM_PORT = 8000
MAX_MODEL_LEN = 512


@app.function(
    image=VLLM_FP8_IMAGE,
    gpu="H100",
    volumes={
        "/runs": runs_volume,
        "/root/.cache/vllm": vllm_cache,
        "/root/.cache/huggingface": hf_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    timeout=600,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=300)
def serve():
    import subprocess
    from pathlib import Path

    # Find latest FP8 quantized model
    runs_dir = Path("/runs")
    fp8_runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir() and (d / "fp8").exists()])
    if not fp8_runs:
        raise FileNotFoundError(
            "No FP8 quantized models found. Run 'nls-finetune quantize --method fp8' first."
        )

    model_path = f"/runs/{fp8_runs[-1]}/fp8"
    print(f"Using FP8 model: {model_path}")

    # Latency-optimized for short sequences (Sourcegraph queries)
    # V0 flags outperform V1 auto-tuning for single-request, short-sequence workloads
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.95",
        "--trust-remote-code",
        "--served-model-name",
        "llm",
        "--enable-prefix-caching",
        "--max-num-batched-tokens",
        "512",  # Optimize for short sequences
        "--max-num-seqs",
        "4",  # Limit batching for single-request latency
        "--disable-log-requests",
        "--disable-cascade-attn",
    ]

    print(f"Starting vLLM: {' '.join(cmd)}")
    subprocess.Popen(cmd)
