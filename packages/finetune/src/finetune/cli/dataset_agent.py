"""Dataset generation agent CLI commands."""

from pathlib import Path

import typer

from finetune.dataset_agent.schemas import Stage

dataset_agent_app = typer.Typer(help="Dataset generation agent commands")

# Valid stage names for CLI
STAGE_NAMES = [s.value for s in Stage]


@dataset_agent_app.command("run")
def run_dataset_agent(
    config: Path | None = typer.Option(
        None, "--config", help="Path to config file (YAML or JSON)"
    ),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for run artifacts"),
    sources: Path | None = typer.Option(
        None, "--sources", help="Path to sources.yaml configuration"
    ),
    seed: int = typer.Option(
        42, "--seed", help="Random seed for deterministic generation"
    ),
    samples_per_template: int = typer.Option(
        100,
        "--samples-per-template",
        help="Number of samples per template (0 = unlimited for max generation)",
    ),
    from_stage: str | None = typer.Option(
        None,
        "--from-stage",
        help=f"Start from this stage (skip earlier stages). Valid: {', '.join(STAGE_NAMES)}",
    ),
    to_stage: str | None = typer.Option(
        None,
        "--to-stage",
        help=f"Stop at this stage (skip later stages). Valid: {', '.join(STAGE_NAMES)}",
    ),
    skip_backend: bool = typer.Option(
        False,
        "--skip-backend",
        help="Skip backend validation stage (faster, offline)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-run of completed stages",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Execute the dataset generation pipeline.

    Creates a per-run folder structure with all artifacts organized by type.

    Example commands:

    # Full pipeline with default settings (100 samples/template)
    scix-finetune dataset-agent run --out-dir data/datasets/agent_runs --sources sources.yaml

    # Max generation (all catalog entries)
    scix-finetune dataset-agent run --out-dir data/datasets/agent_runs --samples-per-template 0

    # Skip backend validation (offline mode, faster)
    scix-finetune dataset-agent run --out-dir data/datasets/agent_runs --skip-backend

    # Run specific stages only
    scix-finetune dataset-agent run --out-dir data/datasets/agent_runs --from-stage normalize --to-stage render_pairs

    # Dry run with small sample
    scix-finetune dataset-agent run --out-dir data/datasets/agent_runs --samples-per-template 10 --skip-backend
    """
    from finetune.dataset_agent.runner import run_pipeline

    # Validate stage names if provided
    if from_stage and from_stage not in STAGE_NAMES:
        raise typer.BadParameter(
            f"Invalid stage: {from_stage}. Valid stages: {', '.join(STAGE_NAMES)}"
        )
    if to_stage and to_stage not in STAGE_NAMES:
        raise typer.BadParameter(
            f"Invalid stage: {to_stage}. Valid stages: {', '.join(STAGE_NAMES)}"
        )

    run_pipeline(
        config=config,
        out_dir=out_dir,
        sources_config_path=sources,
        seed=seed,
        samples_per_template=samples_per_template,
        from_stage=from_stage,
        to_stage=to_stage,
        skip_backend=skip_backend,
        force=force,
        verbose=verbose,
    )


@dataset_agent_app.command("list-stages")
def list_stages() -> None:
    """List all available pipeline stages."""
    typer.echo("Available pipeline stages:")
    for stage in Stage:
        typer.echo(f"  - {stage.value}")
