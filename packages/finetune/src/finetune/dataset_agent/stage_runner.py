"""Stage runner for dataset generation pipeline.

This module implements stage-by-stage execution with checkpoints for the
dataset generation pipeline. Each stage writes a completion marker to
manifests/ and can be skipped if outputs already exist.

Stages:
1. fetch - Download data sources
2. normalize - Normalize raw data to catalog format
3. expand_aliases - Expand alias variants in catalogs
4. load_templates - Load and validate templates
5. generate_inputs - Generate NL inputs from templates
6. render_pairs - Render ADS queries from inputs
7. validate_local - Tier 1 local syntax validation
8. validate_backend - Tier 2/3 backend validation (optional)
9. generate_enrichment - Generate enrichment labels
10. report - Generate reports and manifest checksums

Features:
- Resumable checkpoints - skip completed stages
- Force mode - re-run stages even if outputs exist
- Partial pipelines - run subset of stages
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from finetune.dataset_agent.schemas import RunManifest, Stage


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


# Define stage order
STAGE_ORDER = [
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


@dataclass
class StageMarker:
    """Completion marker for a stage.

    Written to manifests/stage_<stage_name>.json when stage completes.
    """

    stage: str
    status: str
    started_at: str
    completed_at: str | None = None
    outputs: list[str] = field(default_factory=list)  # List of output file paths
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = {
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
        }
        if self.completed_at:
            result["completed_at"] = self.completed_at
        if self.outputs:
            result["outputs"] = self.outputs
        if self.error:
            result["error"] = self.error
        return result

    @classmethod
    def from_dict(cls, data: dict) -> StageMarker:
        """Create from dict."""
        return cls(
            stage=data["stage"],
            status=data["status"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            outputs=data.get("outputs", []),
            error=data.get("error"),
        )


@dataclass
class StageResult:
    """Result of running a stage."""

    stage: Stage
    status: StageStatus
    outputs: list[Path] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class StageRunnerConfig:
    """Configuration for stage runner.

    Attributes:
        force: Re-run stages even if outputs exist
        from_stage: Start from this stage (inclusive)
        to_stage: Stop at this stage (inclusive)
        skip_backend: Skip backend validation stage
        verbose: Enable verbose logging
    """

    force: bool = False
    from_stage: Stage | None = None
    to_stage: Stage | None = None
    skip_backend: bool = False
    verbose: bool = False


@dataclass
class StageRunnerStats:
    """Statistics from stage runner.

    Attributes:
        stages_run: Number of stages executed
        stages_skipped: Number of stages skipped
        stages_failed: Number of stages that failed
        total_duration_seconds: Total execution time
    """

    stages_run: int = 0
    stages_skipped: int = 0
    stages_failed: int = 0
    total_duration_seconds: float = 0.0


def get_stage_marker_path(manifests_dir: Path, stage: Stage) -> Path:
    """Get the path for a stage completion marker.

    Args:
        manifests_dir: Path to manifests directory
        stage: The stage

    Returns:
        Path to stage_<stage_name>.json
    """
    return manifests_dir / f"stage_{stage.value}.json"


def read_stage_marker(manifests_dir: Path, stage: Stage) -> StageMarker | None:
    """Read a stage completion marker if it exists.

    Args:
        manifests_dir: Path to manifests directory
        stage: The stage

    Returns:
        StageMarker if exists, None otherwise
    """
    marker_path = get_stage_marker_path(manifests_dir, stage)
    if not marker_path.exists():
        return None

    with open(marker_path) as f:
        data = json.load(f)
    return StageMarker.from_dict(data)


def write_stage_marker(manifests_dir: Path, marker: StageMarker) -> None:
    """Write a stage completion marker.

    Args:
        manifests_dir: Path to manifests directory
        marker: The marker to write
    """
    marker_path = get_stage_marker_path(manifests_dir, Stage(marker.stage))
    with open(marker_path, "w") as f:
        json.dump(marker.to_dict(), f, indent=2)


def is_stage_complete(manifests_dir: Path, stage: Stage) -> bool:
    """Check if a stage has been completed.

    Args:
        manifests_dir: Path to manifests directory
        stage: The stage

    Returns:
        True if stage has completed successfully
    """
    marker = read_stage_marker(manifests_dir, stage)
    return marker is not None and marker.status == StageStatus.COMPLETED.value


def get_stages_to_run(config: StageRunnerConfig) -> list[Stage]:
    """Get the list of stages to run based on config.

    Args:
        config: Stage runner configuration

    Returns:
        List of stages to run in order
    """
    stages = STAGE_ORDER.copy()

    # Skip backend if configured
    if config.skip_backend:
        stages = [s for s in stages if s != Stage.VALIDATE_BACKEND]

    # Filter by from_stage
    if config.from_stage:
        try:
            start_idx = stages.index(config.from_stage)
            stages = stages[start_idx:]
        except ValueError:
            pass

    # Filter by to_stage
    if config.to_stage:
        try:
            end_idx = stages.index(config.to_stage)
            stages = stages[: end_idx + 1]
        except ValueError:
            pass

    return stages


class StageRunner:
    """Runner for pipeline stages with checkpoint support.

    Executes stages in order, writing completion markers for each.
    Can skip stages that have already completed unless force=True.
    """

    def __init__(
        self,
        run_dir: Path,
        config: StageRunnerConfig | None = None,
    ) -> None:
        """Initialize stage runner.

        Args:
            run_dir: Run directory containing manifests/
            config: Runner configuration
        """
        self.run_dir = run_dir
        self.config = config or StageRunnerConfig()
        self.manifests_dir = run_dir / "manifests"
        self._stats = StageRunnerStats()
        self._stage_handlers: dict[Stage, callable] = {}

    def register_handler(self, stage: Stage, handler: callable) -> None:
        """Register a handler function for a stage.

        Args:
            stage: The stage to handle
            handler: Function that takes (run_dir: Path) and returns list[Path] of outputs
        """
        self._stage_handlers[stage] = handler

    def should_run_stage(self, stage: Stage) -> bool:
        """Check if a stage should be run.

        Args:
            stage: The stage to check

        Returns:
            True if stage should run
        """
        if self.config.force:
            return True

        return not is_stage_complete(self.manifests_dir, stage)

    def run_stage(self, stage: Stage) -> StageResult:
        """Run a single stage.

        Args:
            stage: The stage to run

        Returns:
            StageResult with status and outputs
        """
        # Check if we should skip
        if not self.should_run_stage(stage):
            self._stats.stages_skipped += 1
            return StageResult(
                stage=stage,
                status=StageStatus.SKIPPED,
            )

        # Get handler
        handler = self._stage_handlers.get(stage)
        if handler is None:
            # No handler registered, skip
            self._stats.stages_skipped += 1
            return StageResult(
                stage=stage,
                status=StageStatus.SKIPPED,
                error=f"No handler registered for stage {stage.value}",
            )

        # Write pending marker
        started_at = datetime.now(UTC).isoformat()
        pending_marker = StageMarker(
            stage=stage.value,
            status=StageStatus.RUNNING.value,
            started_at=started_at,
        )
        write_stage_marker(self.manifests_dir, pending_marker)

        # Run the handler
        try:
            start_time = datetime.now(UTC)
            outputs = handler(self.run_dir)
            end_time = datetime.now(UTC)
            duration = (end_time - start_time).total_seconds()

            # Write completion marker
            completed_marker = StageMarker(
                stage=stage.value,
                status=StageStatus.COMPLETED.value,
                started_at=started_at,
                completed_at=end_time.isoformat(),
                outputs=[str(p.relative_to(self.run_dir)) for p in outputs],
            )
            write_stage_marker(self.manifests_dir, completed_marker)

            self._stats.stages_run += 1
            self._stats.total_duration_seconds += duration

            return StageResult(
                stage=stage,
                status=StageStatus.COMPLETED,
                outputs=outputs,
                duration_seconds=duration,
            )

        except Exception as e:
            # Write failure marker
            failed_marker = StageMarker(
                stage=stage.value,
                status=StageStatus.FAILED.value,
                started_at=started_at,
                completed_at=datetime.now(UTC).isoformat(),
                error=str(e),
            )
            write_stage_marker(self.manifests_dir, failed_marker)

            self._stats.stages_failed += 1

            return StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error=str(e),
            )

    def run_all(self) -> list[StageResult]:
        """Run all configured stages.

        Returns:
            List of StageResult for each stage
        """
        stages = get_stages_to_run(self.config)
        results = []

        for stage in stages:
            result = self.run_stage(stage)
            results.append(result)

            # Stop on failure
            if result.status == StageStatus.FAILED:
                break

        return results

    def get_completed_stages(self) -> list[Stage]:
        """Get list of completed stages.

        Returns:
            List of Stage that have completed successfully
        """
        completed = []
        for stage in STAGE_ORDER:
            if is_stage_complete(self.manifests_dir, stage):
                completed.append(stage)
        return completed

    def update_manifest(self, manifest: RunManifest) -> RunManifest:
        """Update run manifest with completed stages.

        Args:
            manifest: Existing run manifest

        Returns:
            Updated manifest
        """
        completed = self.get_completed_stages()
        return RunManifest(
            run_id=manifest.run_id,
            created_at=manifest.created_at,
            started_at=manifest.started_at,
            completed_at=manifest.completed_at,
            run_directory=manifest.run_directory,
            config_path=manifest.config_path,
            status=manifest.status,
            stages_completed=[s.value for s in completed],
            current_stage=manifest.current_stage,
            error_message=manifest.error_message,
            artifacts=manifest.artifacts,
            reproduce=manifest.reproduce,
        )

    @property
    def stats(self) -> StageRunnerStats:
        """Get runner statistics."""
        return self._stats


def run_pipeline_stages(
    run_dir: Path,
    stage_handlers: dict[Stage, callable],
    config: StageRunnerConfig | None = None,
) -> tuple[list[StageResult], StageRunnerStats]:
    """Convenience function to run pipeline stages.

    Args:
        run_dir: Run directory
        stage_handlers: Dict mapping Stage to handler functions
        config: Runner configuration

    Returns:
        Tuple of (results, stats)
    """
    runner = StageRunner(run_dir, config)

    for stage, handler in stage_handlers.items():
        runner.register_handler(stage, handler)

    results = runner.run_all()
    return results, runner.stats
