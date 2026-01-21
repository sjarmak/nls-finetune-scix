"""Shared Modal images and volumes for fine-tuning infrastructure.

This module provides common configurations to reduce duplication across
Modal function files.
"""

import modal

# ============================================
# VOLUMES
# ============================================

# Training data (train.jsonl, val.jsonl)
data_volume = modal.Volume.from_name("nls-query-data", create_if_missing=True)

# Training runs (checkpoints, merged models, quantized models)
runs_volume = modal.Volume.from_name("nls-query-runs", create_if_missing=True)

# vLLM KV cache (speeds up serving)
vllm_cache = modal.Volume.from_name("nls-vllm-cache", create_if_missing=True)

# HuggingFace model cache
hf_cache = modal.Volume.from_name("nls-hf-cache", create_if_missing=True)


# ============================================
# IMAGES
# ============================================

# Standard vLLM image for serving fine-tuned models
# Used by: serve_vllm.py
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

# ============================================
# STANDARD VOLUME MOUNTS
# ============================================

# For serving (read-only access to models)
SERVING_VOLUMES = {
    "/runs": runs_volume,
    "/vllm-cache": vllm_cache,
    "/hf-cache": hf_cache,
}

# For training (read data, write runs)
TRAINING_VOLUMES = {
    "/runs": runs_volume,
    "/data": data_volume,
}

# For quantization (read merged models, write quantized)
QUANTIZATION_VOLUMES = {
    "/runs": runs_volume,
    "/data": data_volume,  # For calibration data
}
