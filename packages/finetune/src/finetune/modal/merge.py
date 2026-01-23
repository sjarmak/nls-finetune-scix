"""Modal functions for merging LoRA adapters."""

import modal

# Image with merge dependencies
merge_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.45.0",
    "peft>=0.12.0",
    "accelerate>=0.30.0",
    "huggingface_hub>=0.24.0",
)

# Volumes
runs_volume = modal.Volume.from_name("scix-finetune-runs", create_if_missing=True)

app = modal.App("nls-finetune-merge")


@app.function(
    image=merge_image,
    gpu="A10G",  # A10G is plenty for merging 1.7B model (saves ~$2.85/hr vs H100)
    timeout=1800,
    volumes={"/runs": runs_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_lora_adapter(run_name: str) -> dict:
    """Merge LoRA adapter into base model."""
    from pathlib import Path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Find the run directory
    run_dir = Path(f"/runs/{run_name}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Find latest checkpoint
    checkpoints = list(run_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
    print(f"✓ Found checkpoint: {latest_checkpoint}")

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(latest_checkpoint))

    # Merge and unload
    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_dir = run_dir / "merged"
    merged_dir.mkdir(exist_ok=True)

    print(f"Saving merged model to {merged_dir}...")
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # Commit volume
    runs_volume.commit()

    return {
        "run_name": run_name,
        "checkpoint": str(latest_checkpoint),
        "merged_dir": str(merged_dir),
    }


@app.function(
    volumes={"/runs": runs_volume},
    timeout=60,
)
def list_training_runs() -> list[dict]:
    """List all training runs in the volume."""
    from pathlib import Path

    runs_dir = Path("/runs")
    runs = []

    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith("run-"):
                # Check for checkpoints
                checkpoints = list(run_dir.glob("checkpoint-*"))
                has_merged = (run_dir / "merged").exists()

                runs.append(
                    {
                        "name": run_dir.name,
                        "checkpoints": len(checkpoints),
                        "merged": has_merged,
                    }
                )

    return runs
