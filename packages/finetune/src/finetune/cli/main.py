"""Main CLI entry point for SciX/ADS fine-tuning."""

import typer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from finetune import __version__
from finetune.cli.data import data_app
from finetune.cli.evaluation import eval_app
from finetune.cli.training import training_app
from finetune.cli.verify import verify_app

app = typer.Typer(
    name="scix-finetune",
    help="Fine-tune Qwen models to convert natural language to ADS/SciX search queries.",
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(verify_app, name="verify", help="Run verification checks")
app.add_typer(data_app, name="data", help="Data management commands")
app.add_typer(training_app, name="training", help="Training commands")
app.add_typer(eval_app, name="eval", help="Evaluate model quality")


@app.command()
def upload_data(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be uploaded"),
) -> None:
    """Upload training data to Modal volume."""
    from finetune.cli.data import upload_data_impl

    upload_data_impl(dry_run=dry_run)


@app.command("dry-run")
def dry_run(
    phase: str = typer.Argument("train", help="Phase to dry-run: model or train"),
    steps: int = typer.Option(3, "--steps", help="Number of training steps for train phase"),
) -> None:
    """Test setup without full training."""
    from finetune.cli.training import dry_run_impl

    dry_run_impl(phase=phase, steps=steps)


@app.command()
def train(
    run_name: str = typer.Option(None, "--run-name", help="Name for this training run"),
    wandb: bool = typer.Option(False, "--wandb", help="Enable Weights & Biases logging"),
    use_unsloth: bool = typer.Option(False, "--use-unsloth", help="Use Unsloth for 2x faster"),
) -> None:
    """Run full fine-tuning on Modal with H100."""
    from finetune.cli.training import train_impl

    train_impl(run_name=run_name, wandb=wandb, use_unsloth=use_unsloth)


@app.command()
def merge(
    run_name: str = typer.Option(..., "--run-name", help="Training run to merge"),
    use_unsloth: bool = typer.Option(False, "--use-unsloth", help="Unsloth merge"),
) -> None:
    """Merge LoRA adapter into base model."""
    from finetune.cli.training import merge_impl

    merge_impl(run_name=run_name, use_unsloth=use_unsloth)


@app.command()
def deploy(
    run_name: str = typer.Option(None, "--run-name", help="Training run to deploy"),
    quantized: bool = typer.Option(False, "--quantized", help="Deploy FP8 quantized model"),
) -> None:
    """Deploy inference endpoint to Modal."""
    from finetune.cli.training import deploy_impl

    deploy_impl(run_name=run_name, quantized=quantized)


@app.command()
def quantize(
    run_name: str = typer.Option(..., "--run-name", help="Training run to quantize"),
    method: str = typer.Option("fp8", "--method", help="Quantization method (fp8)"),
) -> None:
    """Quantize a merged model using FP8 for faster inference."""
    from finetune.cli.training import quantize_impl

    quantize_impl(run_name=run_name, method=method)


@app.command()
def status() -> None:
    """Show current training status and recent runs."""
    from finetune.cli.training import status_impl

    status_impl()


@app.command("list-runs")
def list_runs(
    latest: bool = typer.Option(False, "--latest", help="Show only the latest run name"),
) -> None:
    """List all training runs."""
    from finetune.cli.training import list_runs_impl

    list_runs_impl(latest=latest)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show version"),
) -> None:
    """SciX/ADS Fine-tuning CLI."""
    if version:
        print(f"scix-finetune {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
