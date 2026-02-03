"""Main CLI entry point for SciX/ADS fine-tuning."""

import typer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from finetune import __version__
from finetune.cli.evaluation import eval_app
from finetune.cli.verify import verify_app

app = typer.Typer(
    name="scix-finetune",
    help="Fine-tune Qwen models to convert natural language to ADS/SciX search queries.",
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(verify_app, name="verify", help="Run verification checks")
app.add_typer(eval_app, name="eval", help="Evaluate model quality")

# Lazy import for dataset-agent to avoid heavy dependency loading
try:
    from finetune.cli.dataset_agent import dataset_agent_app

    app.add_typer(dataset_agent_app, name="dataset-agent", help="Dataset generation agent")
except ImportError:
    pass


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
