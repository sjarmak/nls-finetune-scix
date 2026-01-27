"""Main pipeline runner for dataset generation agent."""

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import typer

from finetune.dataset_agent.pipeline_handlers import (
    STAGE_HANDLERS,
    PipelineConfig,
    PipelineState,
)
from finetune.dataset_agent.schemas import SourceManifest, Stage
from finetune.dataset_agent.sources import (
    SourcesConfig,
    create_source_entry,
    get_deterministic_filename,
)
from finetune.dataset_agent.stage_runner import StageRunner, StageStatus

logger = logging.getLogger(__name__)

# Ordered list of all pipeline stages
ALL_STAGES = [
    Stage.FETCH,
    Stage.NORMALIZE,
    Stage.EXPAND_ALIASES,
    Stage.LOAD_TEMPLATES,
    Stage.GENERATE_INPUTS,
    Stage.RENDER_PAIRS,
    Stage.VALIDATE_LOCAL,
    Stage.VALIDATE_BACKEND,
    Stage.GENERATE_ENRICHMENT,
    Stage.REPORT,
]


def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{short_uuid}"


def create_run_directory_structure(base_dir: Path) -> dict[str, Path]:
    """
    Create the standard directory structure for a pipeline run.

    Returns a dict mapping directory purpose to Path.
    """
    directories = {
        "raw": base_dir / "raw",
        "normalized": base_dir / "normalized",
        "pairs": base_dir / "pairs",
        "enrichment": base_dir / "enrichment",
        "reports": base_dir / "reports",
        "manifests": base_dir / "manifests",
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def create_run_manifest(
    run_id: str,
    run_dir: Path,
    config_path: Path | None,
) -> dict:
    """
    Create the initial run manifest with run metadata.

    Returns the manifest dict that will be written to run_manifest.json.
    """
    now = datetime.now(UTC)

    manifest = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "started_at": now.isoformat(),
        "completed_at": None,
        "config_path": str(config_path) if config_path else None,
        "run_directory": str(run_dir),
        "status": "in_progress",
        "stages_completed": [],
        "artifacts": {},
    }

    return manifest


def generate_source_manifest_stub(
    sources_config: SourcesConfig,
    raw_dir: Path,
) -> SourceManifest:
    """Generate a stub source manifest from sources.yaml config.

    This creates source entries with expected local paths but without
    actual download metadata (checksums, retrieval times, etc.).
    The actual download will populate these fields.

    Args:
        sources_config: Parsed sources.yaml configuration
        raw_dir: Path to raw/ directory for storing downloads

    Returns:
        SourceManifest with stub entries
    """
    entries = []
    for source in sources_config.sources:
        filename = get_deterministic_filename(source)
        local_path = f"raw/{filename}"
        entry = create_source_entry(source, local_path=local_path)
        entries.append(entry)

    return SourceManifest(
        sources=entries,
        generated_at=datetime.now(UTC).isoformat(),
    )


def get_stages_to_run(
    from_stage: Stage | None = None,
    to_stage: Stage | None = None,
    skip_backend: bool = False,
) -> list[Stage]:
    """Determine which stages to run based on from/to filters.

    Args:
        from_stage: Start from this stage (inclusive)
        to_stage: Stop at this stage (inclusive)
        skip_backend: Skip backend validation stage

    Returns:
        List of stages to execute in order
    """
    stages = ALL_STAGES.copy()

    # Apply from filter
    if from_stage:
        try:
            start_idx = stages.index(from_stage)
            stages = stages[start_idx:]
        except ValueError:
            logger.warning(f"Unknown stage: {from_stage}, starting from beginning")

    # Apply to filter
    if to_stage:
        try:
            end_idx = stages.index(to_stage)
            stages = stages[: end_idx + 1]
        except ValueError:
            logger.warning(f"Unknown stage: {to_stage}, running to end")

    # Skip backend if requested
    if skip_backend and Stage.VALIDATE_BACKEND in stages:
        stages.remove(Stage.VALIDATE_BACKEND)

    return stages


def run_pipeline(
    config: Path | None,
    out_dir: Path,
    sources_config_path: Path | None = None,
    seed: int | None = 42,
    samples_per_template: int = 100,
    from_stage: str | None = None,
    to_stage: str | None = None,
    skip_backend: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Execute the full dataset generation pipeline.

    Args:
        config: Path to configuration file (optional for now)
        out_dir: Base output directory for run artifacts
        sources_config_path: Path to sources.yaml (optional, uses default if not provided)
        seed: Random seed for deterministic generation
        samples_per_template: Number of samples per template (0 = unlimited)
        from_stage: Start from this stage (skip earlier stages)
        to_stage: Stop at this stage (skip later stages)
        skip_backend: Skip backend validation stage
        force: Force re-run of completed stages
        verbose: Enable verbose logging

    Returns:
        Path to run directory
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Generate run ID and create run directory
    run_id = generate_run_id()
    run_dir = out_dir / run_id

    typer.echo(f"Starting dataset generation run: {run_id}")
    typer.echo(f"Output directory: {run_dir}")

    # Create directory structure
    directories = create_run_directory_structure(run_dir)
    typer.echo("\nCreated directory structure:")
    for name, path in directories.items():
        typer.echo(f"  {name:12} -> {path.relative_to(run_dir)}")

    # Create run manifest
    manifest = create_run_manifest(run_id, run_dir, config)
    manifest_path = directories["manifests"] / "run_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    typer.echo(f"\n✓ Created run manifest: {manifest_path.relative_to(run_dir)}")

    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        sources_config_path=sources_config_path,
        seed=seed,
        samples_per_template=samples_per_template,
        skip_backend=skip_backend,
        enable_backend_validation=not skip_backend,
    )

    # Create pipeline state
    pipeline_state = PipelineState()

    # Determine stages to run
    from_stage_enum = Stage(from_stage) if from_stage else None
    to_stage_enum = Stage(to_stage) if to_stage else None
    stages_to_run = get_stages_to_run(from_stage_enum, to_stage_enum, skip_backend)

    typer.echo(f"\nStages to run: {[s.value for s in stages_to_run]}")
    if skip_backend:
        typer.echo("  (backend validation skipped)")

    # Create stage runner
    stage_runner = StageRunner(run_dir)

    # Register handlers
    for stage, handler in STAGE_HANDLERS.items():
        # Wrap handler to match StageRunner signature
        def create_wrapped_handler(h, pc, ps):
            def wrapped(rd: Path) -> list[Path]:
                return h(rd, pc, ps)

            return wrapped

        wrapped = create_wrapped_handler(handler, pipeline_config, pipeline_state)
        stage_runner.register_handler(stage, wrapped)

    # Execute stages
    typer.echo("\n" + "=" * 60)
    typer.echo("PIPELINE EXECUTION")
    typer.echo("=" * 60)

    stages_completed = []
    failed_stage = None
    error_message = None

    for stage in stages_to_run:
        typer.echo(f"\n>>> Stage: {stage.value}")

        # Check if stage already completed (unless force is set)
        if not force and not stage_runner.should_run_stage(stage):
            typer.echo("    [SKIPPED] Stage already complete")
            stages_completed.append(stage.value)
            continue

        result = stage_runner.run_stage(stage)

        if result.status == StageStatus.COMPLETED:
            stages_completed.append(stage.value)
            typer.echo(f"    [COMPLETE] {len(result.outputs)} outputs")
            for output in result.outputs[:5]:  # Show first 5 outputs
                typer.echo(f"      - {output.relative_to(run_dir)}")
            if len(result.outputs) > 5:
                typer.echo(f"      ... and {len(result.outputs) - 5} more")
        elif result.status == StageStatus.SKIPPED:
            typer.echo("    [SKIPPED] Stage already complete")
            stages_completed.append(stage.value)
        elif result.status == StageStatus.FAILED:
            failed_stage = stage.value
            error_message = result.error
            logger.error(f"Stage {stage.value} failed: {result.error}")
            typer.echo(f"    [FAILED] {result.error}", err=True)
            break

    # Update manifest with final status
    manifest["stages_completed"] = stages_completed
    manifest["completed_at"] = datetime.now(UTC).isoformat()

    if failed_stage:
        manifest["status"] = "failed"
        manifest["error_message"] = error_message
        manifest["current_stage"] = failed_stage
    else:
        manifest["status"] = "completed"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    typer.echo("\n" + "=" * 60)
    typer.echo("SUMMARY")
    typer.echo("=" * 60)
    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Status: {manifest['status']}")
    typer.echo(f"Stages completed: {len(stages_completed)}/{len(stages_to_run)}")

    if failed_stage:
        typer.echo(f"Failed at: {failed_stage}", err=True)
        typer.echo(f"Error: {error_message}", err=True)
    else:
        # Show key outputs
        pairs_path = run_dir / "pairs" / "pairs.jsonl"
        if pairs_path.exists():
            line_count = sum(1 for _ in open(pairs_path))
            typer.echo(f"Valid pairs: {line_count}")

        report_path = run_dir / "reports" / "summary.json"
        if report_path.exists():
            typer.echo(f"Report: {report_path}")

    typer.echo(f"\nRun directory: {run_dir}")

    return run_dir
