"""Modal functions for FP8 quantization of fine-tuned models.

Uses llm-compressor for FP8_DYNAMIC (8-bit floating point) quantization.
FP8 is well-supported for Qwen3 dense models and provides good speedup.
"""

import modal

# Quantization image with llm-compressor 0.8.0+ for Qwen3 FP8 support
quant_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.51.0",  # Required for Qwen3
        "llmcompressor>=0.8.0",  # 0.8.0+ has Qwen3 FP8 support
        "accelerate>=0.30.0",
        "huggingface_hub>=0.24.0",
        "compressed-tensors>=0.6.0",
    )
)

# Volumes
runs_volume = modal.Volume.from_name("nls-query-runs", create_if_missing=True)

app = modal.App("nls-finetune-quantize-fp8")


@app.function(
    image=quant_image,
    gpu="H100",  # FP8 requires compute capability > 8.9 (Hopper/Ada)
    timeout=3600,
    volumes={
        "/runs": runs_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def quantize_fp8(run_name: str) -> dict:
    """Quantize a merged model using FP8 dynamic quantization.

    Uses llm-compressor's FP8_DYNAMIC scheme which:
    - Is data-free (no calibration dataset needed)
    - Works well with Qwen3 dense models
    - Provides good speedup on H100/Ada GPUs

    Args:
        run_name: Name of the training run (e.g., run-20251215-163945)

    Returns:
        Dict with quantization results
    """
    from pathlib import Path

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoTokenizer

    # Find merged model
    merged_dir = Path(f"/runs/{run_name}/merged")
    if not merged_dir.exists():
        raise FileNotFoundError(
            f"Merged model not found at {merged_dir}. Run 'nls-finetune merge' first."
        )

    print(f"✓ Found merged model: {merged_dir}")

    # Load tokenizer (needed for saving with model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(merged_dir),
        trust_remote_code=True,
    )

    # Configure FP8 dynamic quantization
    # FP8_DYNAMIC is data-free - no calibration needed
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",  # 8-bit floating point, dynamic per-token activation
        ignore=["lm_head"],  # Don't quantize output layer
    )

    print("✓ Quantization config: FP8_DYNAMIC (8-bit floating point)")

    # Output directory
    quant_dir = Path(f"/runs/{run_name}/fp8")
    quant_dir.mkdir(exist_ok=True)

    # Run quantization with oneshot (no dataset needed for FP8_DYNAMIC)
    print("\nRunning FP8 quantization...")
    oneshot(
        model=str(merged_dir),
        recipe=recipe,
        output_dir=str(quant_dir),
    )
    print("✓ Quantization complete")

    # Save tokenizer to output dir
    tokenizer.save_pretrained(str(quant_dir))

    # Commit volume
    runs_volume.commit()

    print(f"\n✓ FP8 quantized model saved to {quant_dir}")

    return {
        "run_name": run_name,
        "merged_dir": str(merged_dir),
        "fp8_dir": str(quant_dir),
        "quant_config": {"bits": 8, "scheme": "FP8_DYNAMIC", "method": "fp8"},
    }


@app.function(
    image=quant_image,
    gpu="H100",
    timeout=600,
    volumes={
        "/runs": runs_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_fp8_model(run_name: str) -> dict:
    """Test the FP8 quantized model with a sample inference."""
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fp8_dir = Path(f"/runs/{run_name}/fp8")
    if not fp8_dir.exists():
        raise FileNotFoundError(f"FP8 model not found at {fp8_dir}")

    print(f"Loading FP8 model from {fp8_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(str(fp8_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(fp8_dir),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Test inference
    test_prompt = """<|im_start|>system
Convert natural language to Sourcegraph query syntax.
<|im_end|>
<|im_start|>user
find python files with async functions<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    print("Running test inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract just the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

    print(f"Test response: {response.strip()}")

    return {
        "run_name": run_name,
        "fp8_dir": str(fp8_dir),
        "test_response": response.strip(),
        "status": "ok",
    }
