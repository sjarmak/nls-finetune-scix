"""Report generator for dataset generation pipeline.

This module generates summary reports and updates the run manifest with
checksums for all artifacts produced by the pipeline.

Features:
- Generate reports/summary.json with counts and rates
- Compute SHA256 checksums for all artifacts
- Update run_manifest.json with artifact checksums
- Track reproduce info (config, seed, pinned revisions)

Output:
- reports/summary.json: Pipeline statistics
- manifests/run_manifest.json: Updated with checksums and reproduce info
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from finetune.dataset_agent.schemas import (
    ArtifactChecksum,
    Report,
    ReproduceInfo,
    RunManifest,
)
from finetune.dataset_agent.writers import JSONLReader, JSONWriter


@dataclass
class ReportGeneratorConfig:
    """Configuration for report generation.

    Attributes:
        include_artifact_checksums: Compute checksums for all artifacts
        include_reproduce_info: Include reproduce section in manifest
        include_timing: Include timing information in report
    """

    include_artifact_checksums: bool = True
    include_reproduce_info: bool = True
    include_timing: bool = True


@dataclass
class ReportGeneratorStats:
    """Statistics from report generation.

    Attributes:
        artifacts_processed: Number of artifact files processed
        total_size_bytes: Total size of all artifacts
        report_generated: Whether summary report was generated
        manifest_updated: Whether run manifest was updated
    """

    artifacts_processed: int = 0
    total_size_bytes: int = 0
    report_generated: bool = False
    manifest_updated: bool = False


def compute_file_checksum(file_path: Path) -> tuple[str, int]:
    """Compute SHA256 checksum and size for a file.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (SHA256 checksum hex string, size in bytes)
    """
    sha256 = hashlib.sha256()
    size = 0

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
            size += len(chunk)

    return sha256.hexdigest(), size


def count_jsonl_lines(file_path: Path) -> int:
    """Count lines in a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        Number of lines
    """
    reader = JSONLReader(file_path)
    return reader.count()


def create_artifact_checksum(
    file_path: Path,
    run_dir: Path,
) -> ArtifactChecksum:
    """Create an ArtifactChecksum for a file.

    Args:
        file_path: Absolute path to the artifact file
        run_dir: Run directory for relative path calculation

    Returns:
        ArtifactChecksum with checksum and metadata
    """
    checksum, size = compute_file_checksum(file_path)
    relative_path = str(file_path.relative_to(run_dir))

    # Count lines for JSONL files
    line_count = None
    if file_path.suffix == ".jsonl":
        line_count = count_jsonl_lines(file_path)

    return ArtifactChecksum(
        path=relative_path,
        checksum_sha256=checksum,
        size_bytes=size,
        line_count=line_count,
    )


def collect_artifacts(run_dir: Path) -> list[Path]:
    """Collect all artifact files from a run directory.

    Args:
        run_dir: Run directory to scan

    Returns:
        List of artifact file paths
    """
    artifacts = []

    # Standard artifact locations
    artifact_patterns = [
        "raw/*.json",
        "raw/*.jsonl",
        "raw/*.tar.gz",
        "normalized/*.jsonl",
        "pairs/*.jsonl",
        "enrichment/*.jsonl",
        "reports/*.json",
        "manifests/*.json",
        "manifests/*.jsonl",
    ]

    for pattern in artifact_patterns:
        artifacts.extend(run_dir.glob(pattern))

    return sorted(artifacts)


class ReportGenerator:
    """Generator for pipeline reports and manifest updates.

    Produces summary reports and updates run manifests with
    artifact checksums and reproduce information.
    """

    def __init__(self, config: ReportGeneratorConfig | None = None) -> None:
        """Initialize the report generator.

        Args:
            config: Generator configuration
        """
        self.config = config or ReportGeneratorConfig()
        self._stats = ReportGeneratorStats()

    def generate_report(
        self,
        run_dir: Path,
        sources_count: int = 0,
        topics_count: int = 0,
        entities_count: int = 0,
        entries_by_source: dict[str, int] | None = None,
        entries_by_domain: dict[str, int] | None = None,
        templates_count: int = 0,
        inputs_count: int = 0,
        pairs_valid_count: int = 0,
        pairs_quarantined_count: int = 0,
        enrichment_labels_count: int = 0,
        backend_pass_rate: float | None = None,
        pairs_by_template: dict[str, int] | None = None,
        errors_by_type: dict[str, int] | None = None,
        started_at: str | None = None,
    ) -> Report:
        """Generate a summary report.

        Args:
            run_dir: Run directory
            sources_count: Number of data sources
            topics_count: Number of topics in catalog
            entities_count: Number of entities in catalog
            entries_by_source: Entry counts keyed by source ID
            entries_by_domain: Entry counts keyed by domain tag
            templates_count: Number of templates loaded
            inputs_count: Number of NL inputs generated
            pairs_valid_count: Number of valid pairs
            pairs_quarantined_count: Number of quarantined pairs
            enrichment_labels_count: Number of enrichment labels
            backend_pass_rate: Backend validation pass rate
            pairs_by_template: Pairs count by template ID
            errors_by_type: Error count by error type
            started_at: Pipeline start time

        Returns:
            Report with all statistics
        """
        completed_at = datetime.now(UTC).isoformat()

        # Calculate duration if started_at provided
        duration_seconds = None
        if started_at and self.config.include_timing:
            try:
                start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                end = datetime.now(UTC)
                duration_seconds = (end - start).total_seconds()
            except (ValueError, TypeError):
                pass

        return Report(
            sources_count=sources_count,
            topics_count=topics_count,
            entities_count=entities_count,
            entries_by_source=entries_by_source or {},
            entries_by_domain=entries_by_domain or {},
            templates_count=templates_count,
            inputs_count=inputs_count,
            pairs_valid_count=pairs_valid_count,
            pairs_quarantined_count=pairs_quarantined_count,
            enrichment_labels_count=enrichment_labels_count,
            backend_pass_rate=backend_pass_rate,
            pairs_by_template=pairs_by_template or {},
            quarantine_by_error_type=errors_by_type or {},
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
        )

    def generate_artifact_checksums(
        self,
        run_dir: Path,
    ) -> list[ArtifactChecksum]:
        """Generate checksums for all artifacts in run directory.

        Args:
            run_dir: Run directory to scan

        Returns:
            List of ArtifactChecksum objects
        """
        if not self.config.include_artifact_checksums:
            return []

        checksums = []
        artifacts = collect_artifacts(run_dir)

        for artifact_path in artifacts:
            checksum = create_artifact_checksum(artifact_path, run_dir)
            checksums.append(checksum)
            self._stats.artifacts_processed += 1
            self._stats.total_size_bytes += checksum.size_bytes

        return checksums

    def generate_reproduce_info(
        self,
        config_path: Path | None = None,
        seed: int | None = None,
        pinned_revisions: dict[str, str] | None = None,
    ) -> ReproduceInfo:
        """Generate reproduction information.

        Args:
            config_path: Path to config file used
            seed: Random seed used
            pinned_revisions: Source ID to commit hash mapping

        Returns:
            ReproduceInfo with reproduction details
        """
        if not self.config.include_reproduce_info:
            return ReproduceInfo()

        config_checksum = None
        config_path_str = None

        if config_path and config_path.exists():
            config_checksum, _ = compute_file_checksum(config_path)
            config_path_str = str(config_path)

        return ReproduceInfo(
            config_path=config_path_str,
            config_checksum=config_checksum,
            seed=seed,
            pinned_revisions=pinned_revisions or {},
        )

    def update_manifest(
        self,
        manifest: RunManifest,
        report: Report,
        checksums: list[ArtifactChecksum],
        reproduce_info: ReproduceInfo,
    ) -> RunManifest:
        """Update run manifest with report data.

        Args:
            manifest: Existing run manifest
            report: Generated report
            checksums: Artifact checksums
            reproduce_info: Reproduction information

        Returns:
            Updated RunManifest
        """
        # Convert checksums to dict format (keyed by path)
        artifacts_dict = {c.path: c for c in checksums}

        return RunManifest(
            run_id=manifest.run_id,
            created_at=manifest.created_at,
            started_at=manifest.started_at,
            completed_at=report.completed_at,
            status="completed",
            config_path=manifest.config_path,
            stages_completed=manifest.stages_completed,
            artifacts=artifacts_dict,
            reproduce=reproduce_info,
        )

    def generate_and_save(
        self,
        run_dir: Path,
        manifest: RunManifest,
        sources_count: int = 0,
        topics_count: int = 0,
        entities_count: int = 0,
        entries_by_source: dict[str, int] | None = None,
        entries_by_domain: dict[str, int] | None = None,
        templates_count: int = 0,
        inputs_count: int = 0,
        pairs_valid_count: int = 0,
        pairs_quarantined_count: int = 0,
        enrichment_labels_count: int = 0,
        backend_pass_rate: float | None = None,
        pairs_by_template: dict[str, int] | None = None,
        errors_by_type: dict[str, int] | None = None,
        config_path: Path | None = None,
        seed: int | None = None,
        pinned_revisions: dict[str, str] | None = None,
    ) -> tuple[Report, RunManifest, ReportGeneratorStats]:
        """Generate report and update manifest, saving both to files.

        Args:
            run_dir: Run directory
            manifest: Existing run manifest
            sources_count: Number of data sources
            topics_count: Number of topics in catalog
            entities_count: Number of entities in catalog
            entries_by_source: Entry counts keyed by source ID
            entries_by_domain: Entry counts keyed by domain tag
            templates_count: Number of templates loaded
            inputs_count: Number of NL inputs generated
            pairs_valid_count: Number of valid pairs
            pairs_quarantined_count: Number of quarantined pairs
            enrichment_labels_count: Number of enrichment labels
            backend_pass_rate: Backend validation pass rate
            pairs_by_template: Pairs count by template ID
            errors_by_type: Error count by error type
            config_path: Path to config file used
            seed: Random seed used
            pinned_revisions: Source ID to commit hash mapping

        Returns:
            Tuple of (Report, updated RunManifest, stats)
        """
        # Reset stats
        self._stats = ReportGeneratorStats()

        # Generate report
        report = self.generate_report(
            run_dir=run_dir,
            sources_count=sources_count,
            topics_count=topics_count,
            entities_count=entities_count,
            entries_by_source=entries_by_source,
            entries_by_domain=entries_by_domain,
            templates_count=templates_count,
            inputs_count=inputs_count,
            pairs_valid_count=pairs_valid_count,
            pairs_quarantined_count=pairs_quarantined_count,
            enrichment_labels_count=enrichment_labels_count,
            backend_pass_rate=backend_pass_rate,
            pairs_by_template=pairs_by_template,
            errors_by_type=errors_by_type,
            started_at=manifest.started_at,
        )

        # Save report
        report_path = run_dir / "reports" / "summary.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        JSONWriter(report_path).write(report)
        self._stats.report_generated = True

        # Generate checksums (after saving report so it's included)
        checksums = self.generate_artifact_checksums(run_dir)

        # Generate reproduce info
        reproduce_info = self.generate_reproduce_info(
            config_path=config_path,
            seed=seed,
            pinned_revisions=pinned_revisions,
        )

        # Update manifest
        updated_manifest = self.update_manifest(
            manifest=manifest,
            report=report,
            checksums=checksums,
            reproduce_info=reproduce_info,
        )

        # Save updated manifest
        manifest_path = run_dir / "manifests" / "run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        JSONWriter(manifest_path).write(updated_manifest)
        self._stats.manifest_updated = True

        return report, updated_manifest, self._stats

    @property
    def stats(self) -> ReportGeneratorStats:
        """Get report generation statistics."""
        return self._stats


def generate_report(
    run_dir: Path,
    manifest: RunManifest,
    sources_count: int = 0,
    topics_count: int = 0,
    entities_count: int = 0,
    entries_by_source: dict[str, int] | None = None,
    entries_by_domain: dict[str, int] | None = None,
    templates_count: int = 0,
    inputs_count: int = 0,
    pairs_valid_count: int = 0,
    pairs_quarantined_count: int = 0,
    enrichment_labels_count: int = 0,
    backend_pass_rate: float | None = None,
    config_path: Path | None = None,
    seed: int | None = None,
    pinned_revisions: dict[str, str] | None = None,
    config: ReportGeneratorConfig | None = None,
) -> tuple[Report, RunManifest, ReportGeneratorStats]:
    """Convenience function to generate report and update manifest.

    Args:
        run_dir: Run directory
        manifest: Existing run manifest
        sources_count: Number of data sources
        topics_count: Number of topics in catalog
        entities_count: Number of entities in catalog
        entries_by_source: Entry counts keyed by source ID
        entries_by_domain: Entry counts keyed by domain tag
        templates_count: Number of templates loaded
        inputs_count: Number of NL inputs generated
        pairs_valid_count: Number of valid pairs
        pairs_quarantined_count: Number of quarantined pairs
        enrichment_labels_count: Number of enrichment labels
        backend_pass_rate: Backend validation pass rate
        config_path: Path to config file used
        seed: Random seed used
        pinned_revisions: Source ID to commit hash mapping
        config: Report generator configuration

    Returns:
        Tuple of (Report, updated RunManifest, stats)
    """
    generator = ReportGenerator(config=config)
    return generator.generate_and_save(
        run_dir=run_dir,
        manifest=manifest,
        sources_count=sources_count,
        topics_count=topics_count,
        entities_count=entities_count,
        entries_by_source=entries_by_source,
        entries_by_domain=entries_by_domain,
        templates_count=templates_count,
        inputs_count=inputs_count,
        pairs_valid_count=pairs_valid_count,
        pairs_quarantined_count=pairs_quarantined_count,
        enrichment_labels_count=enrichment_labels_count,
        backend_pass_rate=backend_pass_rate,
        config_path=config_path,
        seed=seed,
        pinned_revisions=pinned_revisions,
    )
