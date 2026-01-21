"""Training commands for NLS Query fine-tuning CLI."""

from pathlib import Path

import typer

training_app = typer.Typer(help="Training commands")

PROJECT_ROOT = Path(__file__).parent.parent.parent


def dry_run_impl(phase: str = "train", steps: int = 3) -> None:
    """Test setup without full training."""
    if phase == "model":
        _dry_run_model()
    elif phase == "train":
        _dry_run_train(steps=steps)
    else:
        print(f"Unknown phase: {phase}")
        print("Available phases: model, train")
        raise typer.Exit(1)


def _dry_run_model() -> None:
    """Test model loading on Modal GPU."""
    print("Starting model dry-run on Modal...")

    try:
        from finetune.modal.dry_run import app, test_model_loading

        with app.run():
            result = test_model_loading.remote()

        print("\nModel dry-run passed!")
        print(f"GPU: {result['gpu']}")
        print(f"Memory: {result['memory_gb']:.1f} GB")

    except ImportError as e:
        if "modal" in str(e).lower():
            print("✗ Modal not installed. Run: pip install modal")
        else:
            print(f"✗ Import error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Model dry-run failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def _dry_run_train(steps: int = 3) -> None:
    """Test training pipeline with a few steps."""
    print(f"Starting training dry-run on Modal ({steps} steps)...")

    try:
        from finetune.modal.dry_run import app_train, test_training

        with app_train.run():
            result = test_training.remote(steps)

        print("\n✓ Training dry-run passed!")
        print(f"  GPUs: {result['num_gpus']}x {result['gpu_name']}")
        print(f"  Loss: {result['initial_loss']:.3f} → {result['final_loss']:.3f}")
        if result["loss_decreased"]:
            print("  ✓ Loss decreasing")

    except ImportError as e:
        if "modal" in str(e).lower():
            print("✗ Modal not installed. Run: pip install modal")
        else:
            print(f"✗ Import error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Training dry-run failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def train_impl(
    run_name: str | None = None,
    wandb: bool = False,
    use_unsloth: bool = False,
) -> None:
    """Run full fine-tuning."""
    from datetime import datetime

    if run_name is None:
        run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if use_unsloth:
        _train_unsloth_impl(run_name, wandb)
    else:
        _train_standard_impl(run_name, wandb)


def _train_standard_impl(run_name: str, wandb: bool = False) -> None:
    """Run standard training with TRL."""
    print(f"Starting full training: {run_name}")
    print("This will use an H100 GPU and take ~12 minutes.")
    print("Estimated cost: ~$1.50")
    print()

    try:
        from finetune.modal.train import app, run_training

        with app.run():
            result = run_training.remote(run_name, wandb)

        print("\n✓ Training complete!")
        print(f"  Run name: {result['run_name']}")
        print(f"  GPUs: {result['num_gpus']}x {result['gpu_name']}")
        print(f"  Examples: {result['train_examples']}")
        print(f"  Final loss: {result.get('final_loss', 'N/A')}")
        print(f"  Output: {result['output_dir']}")
        if result.get("checkpoint"):
            print(f"  Checkpoint: {result['checkpoint']}")

    except ImportError as e:
        if "modal" in str(e).lower():
            print("✗ Modal not installed. Run: pip install modal")
        else:
            print(f"✗ Import error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def _train_unsloth_impl(run_name: str, wandb: bool = False) -> None:
    """Run training with Unsloth for 2x faster fine-tuning."""
    print(f"Starting Unsloth training: {run_name}")
    print("Using Unsloth for 2x faster training with 70% less VRAM.")
    print("This will use an H100 GPU and take ~6 minutes.")
    print()

    try:
        from finetune.modal.train_unsloth import app, run_training_unsloth

        with app.run():
            result = run_training_unsloth.remote(run_name, wandb)

        print("\n✓ Unsloth training complete!")
        print(f"  Run name: {result['run_name']}")
        print(f"  GPUs: {result['num_gpus']}x {result['gpu_name']}")
        print(f"  Examples: {result['train_examples']}")
        print(f"  Final loss: {result.get('final_loss', 'N/A')}")
        print(f"  Output: {result['output_dir']}")
        print(f"  Framework: {result.get('framework', 'unsloth')}")

    except ImportError as e:
        if "modal" in str(e).lower():
            print("✗ Modal not installed. Run: pip install modal")
        else:
            print(f"✗ Import error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Unsloth training failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def merge_impl(run_name: str, use_unsloth: bool = False) -> None:
    """Merge LoRA adapter into base model."""
    if use_unsloth:
        _merge_unsloth_impl(run_name)
    else:
        _merge_standard_impl(run_name)


def _merge_standard_impl(run_name: str) -> None:
    """Merge using standard PEFT."""
    print(f"Merging LoRA adapter for run: {run_name}")
    print()

    try:
        from finetune.modal.merge import app, merge_lora_adapter

        with app.run():
            result = merge_lora_adapter.remote(run_name)

        print("\n✓ Merge complete!")
        print(f"  Run name: {result['run_name']}")
        print(f"  Checkpoint: {result['checkpoint']}")
        print(f"  Merged model: {result['merged_dir']}")

    except ImportError as e:
        if "modal" in str(e).lower():
            print("✗ Modal not installed. Run: pip install modal")
        else:
            print(f"✗ Import error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Merge failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def _merge_unsloth_impl(run_name: str) -> None:
    """Merge using Unsloth's optimized merge."""
    print(f"Merging Unsloth model for run: {run_name}")
    print("Using Unsloth's optimized 16-bit merge for vLLM compatibility.")
    print()

    try:
        from finetune.modal.train_unsloth import app, merge_unsloth_model

        with app.run():
            result = merge_unsloth_model.remote(run_name)

        print("\n✓ Unsloth merge complete!")
        print(f"  Run name: {result['run_name']}")
        print(f"  Adapter: {result['adapter_dir']}")
        print(f"  Merged model: {result['merged_dir']}")

    except ImportError as e:
        if "modal" in str(e).lower():
            print("✗ Modal not installed. Run: pip install modal")
        else:
            print(f"✗ Import error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Unsloth merge failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def deploy_impl(run_name: str | None = None, quantized: bool = False) -> None:
    """Deploy inference endpoint."""
    import subprocess

    # Determine which serve module to deploy
    if quantized:
        serve_module = "finetune/modal/serve_vllm_fp8_finetuned.py"
        desc = "Qwen3 FP8 quantized"
    else:
        serve_module = "finetune/modal/serve_vllm.py"
        desc = "Qwen3 BF16"

    print(f"Deploying {desc} inference endpoint...")
    print("The endpoint will automatically use the latest merged model.")
    print("This may take a few minutes on first deploy...")
    print()

    # Deploy the endpoint
    result = subprocess.run(
        ["modal", "deploy", serve_module],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    output = result.stdout + result.stderr
    print(output)

    if result.returncode != 0:
        print("✗ Deploy failed")
        raise typer.Exit(1)

    # Extract endpoint URL from output
    import re

    url_match = re.search(r"(https://[^\s]+\.modal\.run[^\s]*)", output)
    if url_match:
        endpoint_url = url_match.group(1)
        print(f"\n✓ Deployed {desc} successfully!")
        print(f"  Endpoint: {endpoint_url}")
        print()
        print("Test with:")
        print(f'  curl -X POST {endpoint_url} -H "Content-Type: application/json" \\')
        print('    -d \'{"query": "find Python files with async functions"}\'')
    else:
        print(f"\n✓ Deployed {desc} successfully!")


def status_impl() -> None:
    """Show training status and recent runs."""
    print("=== NLS Query Fine-Tuning Status ===\n")

    # Show latest runs
    try:
        from finetune.modal.merge import app, list_training_runs

        with app.run():
            runs = list_training_runs.remote()

        if runs:
            # Show recent runs
            recent = runs[-3:] if len(runs) > 3 else runs
            print("Recent training runs:")
            for run in reversed(recent):
                name = run["name"]
                has_merged = run.get("merged", False)
                has_quant = any(k in run for k in ["int4", "int8", "fp8", "gptq"])
                status_parts = []
                if has_merged:
                    status_parts.append("merged")
                if has_quant:
                    status_parts.append("quantized")
                status = ", ".join(status_parts) if status_parts else "checkpoints only"
                print(f"  • {name} ({status})")
            print()

            # Latest run details
            latest = runs[-1]
            print(f"Latest run: {latest['name']}")
            if latest.get("merged"):
                print("  Merged model: ✓")
            if any(k in latest for k in ["int4", "int8", "fp8", "gptq"]):
                print("  Quantized: ✓")
        else:
            print("No training runs found.\n")
            print("Get started:")
            print("  nls-finetune train --use-unsloth")

    except ImportError:
        print("Modal not available. Install with: pip install modal")
    except Exception as e:
        print(f"Could not fetch runs: {e}")
        print("\nTo see past runs manually:")
        print("  nls-finetune list-runs")

    print()
    print("Commands:")
    print("  nls-finetune list-runs    # Show all runs")
    print("  nls-finetune train        # Start training")
    print("  nls-finetune serve        # Deploy endpoint")


def list_runs_impl(latest: bool = False) -> None:
    """List training runs."""
    try:
        from finetune.modal.merge import app, list_training_runs

        with app.run():
            runs = list_training_runs.remote()

        if not runs:
            print("No training runs found.")
            print("\nUse 'nls-finetune train' to start a training run.")
            return

        if latest:
            # Just print the latest run name
            print(runs[-1]["name"])
            return

        print("Training runs:\n")
        for run in runs:
            status = "✓ merged" if run["merged"] else f"{run['checkpoints']} checkpoints"
            print(f"  {run['name']} ({status})")

    except ImportError as e:
        if "modal" in str(e).lower():
            print("✗ Modal not installed. Run: pip install modal")
        else:
            print(f"✗ Import error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Failed to list runs: {e}")
        raise typer.Exit(1)


def quantize_impl(run_name: str, method: str = "fp8") -> None:
    """Quantize a merged model using FP8 for faster inference.

    Args:
        run_name: Training run to quantize
        method: Quantization method (only fp8 is supported)
    """
    if method != "fp8":
        print(f"✗ Unsupported quantization method: {method}")
        print("Only FP8 quantization is supported for Qwen3.")
        print()
        print("Usage: nls-finetune quantize --run-name <name> --method fp8")
        raise typer.Exit(1)

    print(f"Quantizing model with FP8: {run_name}")
    print("This will use an H100 GPU and take ~5-10 minutes.")
    print()

    try:
        from finetune.modal.quantize_fp8 import app, quantize_fp8

        print("Using llm-compressor FP8_DYNAMIC quantization...")
        with app.run():
            result = quantize_fp8.remote(run_name)

        print("\n✓ FP8 Quantization complete!")
        print(f"  Run name: {result['run_name']}")
        print(f"  Merged dir: {result['merged_dir']}")
        print(f"  FP8 dir: {result['fp8_dir']}")
        print(f"  Quantization config: {result['quant_config']}")

    except ImportError as e:
        if "modal" in str(e).lower():
            print("✗ Modal not installed. Run: pip install modal")
        else:
            print(f"✗ Import error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
