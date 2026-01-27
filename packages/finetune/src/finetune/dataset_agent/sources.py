"""Source configuration and manifest generation for dataset pipeline.

This module handles:
1. Loading and validating sources.yaml configuration
2. Generating source_manifest.json with provenance information
3. Creating deterministic filenames for downloaded artifacts in raw/

Sources are external data sources (HTTP or Git) that provide vocabularies,
taxonomies, and other inputs to the dataset generation pipeline.

Example sources.yaml:
```yaml
sources:
  - id: uat
    type: http
    url: https://astrothesaurus.org/uat.rdf
    license: CC-BY-SA-4.0
    notes: Unified Astronomy Thesaurus

  - id: ror
    type: git
    url: https://github.com/ror-community/ror-data.git
    pinned_revision: v1.52
    license: CC0-1.0
    notes: Research Organization Registry
```
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from finetune.dataset_agent.schemas import SourceEntry, SourceManifest, SourceType


class SourceConfigError(Exception):
    """Error loading or validating source configuration."""

    def __init__(self, message: str, file_path: str | None = None, source_id: str | None = None) -> None:
        self.file_path = file_path
        self.source_id = source_id
        context = []
        if file_path:
            context.append(f"file={file_path}")
        if source_id:
            context.append(f"source_id={source_id}")
        full_message = f"{message} [{', '.join(context)}]" if context else message
        super().__init__(full_message)


@dataclass
class SourceConfig:
    """Configuration for a single source from sources.yaml.

    This represents the declared configuration, not the actual download state.
    """

    id: str  # Unique identifier
    type: SourceType  # http or git
    url: str  # Source URL
    license: str | None = None  # License identifier
    notes: str | None = None  # Human-readable notes
    pinned_revision: str | None = None  # Git commit/tag (for git sources)
    files: list[str] = field(default_factory=list)  # Specific files to extract (for git)
    normalizer: str | None = None  # Normalizer to use (e.g., 'uat', 'ror', 'sweet', 'gcmd', 'planetary')

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "id": self.id,
            "type": self.type.value,
            "url": self.url,
        }
        if self.license:
            d["license"] = self.license
        if self.notes:
            d["notes"] = self.notes
        if self.pinned_revision:
            d["pinned_revision"] = self.pinned_revision
        if self.files:
            d["files"] = self.files
        if self.normalizer:
            d["normalizer"] = self.normalizer
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceConfig":
        """Create from dictionary."""
        # Validate required fields
        if "id" not in d:
            raise SourceConfigError("Missing required field 'id'")
        if "type" not in d:
            raise SourceConfigError("Missing required field 'type'", source_id=d.get("id"))
        if "url" not in d:
            raise SourceConfigError("Missing required field 'url'", source_id=d.get("id"))

        # Parse type
        try:
            source_type = SourceType(d["type"])
        except ValueError:
            valid_types = ", ".join(t.value for t in SourceType)
            raise SourceConfigError(
                f"Invalid source type '{d['type']}'. Valid types: {valid_types}",
                source_id=d.get("id"),
            )

        # Git sources should have pinned_revision
        if source_type == SourceType.GIT and not d.get("pinned_revision"):
            raise SourceConfigError(
                "Git sources should have 'pinned_revision' for reproducibility",
                source_id=d.get("id"),
            )

        return cls(
            id=d["id"],
            type=source_type,
            url=d["url"],
            license=d.get("license"),
            notes=d.get("notes"),
            pinned_revision=d.get("pinned_revision"),
            files=d.get("files", []),
            normalizer=d.get("normalizer"),
        )


@dataclass
class SourcesConfig:
    """Complete sources configuration from sources.yaml."""

    sources: list[SourceConfig] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"sources": [s.to_dict() for s in self.sources]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourcesConfig":
        """Create from dictionary."""
        if not isinstance(d, dict):
            raise SourceConfigError("Root must be a dictionary")
        if "sources" not in d:
            raise SourceConfigError("Missing required field 'sources' at root level")
        if not isinstance(d["sources"], list):
            raise SourceConfigError("'sources' must be a list")

        sources = []
        seen_ids: set[str] = set()
        for i, source_dict in enumerate(d["sources"]):
            if not isinstance(source_dict, dict):
                raise SourceConfigError(f"Source at index {i} must be a dictionary")
            try:
                source = SourceConfig.from_dict(source_dict)
                # Check for duplicate IDs
                if source.id in seen_ids:
                    raise SourceConfigError(f"Duplicate source ID: {source.id}")
                seen_ids.add(source.id)
                sources.append(source)
            except SourceConfigError:
                raise
            except Exception as e:
                raise SourceConfigError(f"Error parsing source at index {i}: {e}")

        return cls(sources=sources)


def load_sources_config(path: Path) -> SourcesConfig:
    """Load and validate sources configuration from YAML file.

    Args:
        path: Path to sources.yaml file

    Returns:
        Validated SourcesConfig object

    Raises:
        SourceConfigError: If configuration is invalid or file cannot be read
    """
    if not path.exists():
        raise SourceConfigError(f"Configuration file not found: {path}", file_path=str(path))

    try:
        with open(path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SourceConfigError(f"Invalid YAML syntax: {e}", file_path=str(path))
    except Exception as e:
        raise SourceConfigError(f"Error reading file: {e}", file_path=str(path))

    if raw_data is None:
        raise SourceConfigError("Empty configuration file", file_path=str(path))

    try:
        return SourcesConfig.from_dict(raw_data)
    except SourceConfigError as e:
        e.file_path = str(path)
        raise


def get_deterministic_filename(source_config: SourceConfig) -> str:
    """Generate a deterministic filename for a source's downloaded artifact.

    The filename is based on the source ID and includes the original extension
    where possible.

    Args:
        source_config: Source configuration

    Returns:
        Filename string (e.g., "uat.rdf", "ror.tar.gz", "ror.zip")
    """
    source_id = source_config.id

    # Git repos typically become tar archives (regardless of URL extension)
    if source_config.type == SourceType.GIT:
        return f"{source_id}.tar.gz"

    # Try to extract extension from URL for HTTP sources
    url_path = source_config.url.rstrip("/")

    # Strip query parameters (e.g., ?download=1)
    if "?" in url_path:
        url_path = url_path.split("?")[0]

    url_filename = url_path.split("/")[-1]

    # Extract extension - look for known archive extensions first
    known_extensions = [".tar.gz", ".tar.bz2", ".tgz", ".zip", ".gz", ".json", ".rdf", ".csv"]
    for ext in known_extensions:
        if url_filename.lower().endswith(ext):
            return f"{source_id}{ext}"

    # Fallback: use last extension if present
    if "." in url_filename:
        ext = "." + url_filename.rsplit(".", 1)[-1]
        return f"{source_id}{ext}"

    # Default: no extension
    return source_id


def create_source_entry(
    source_config: SourceConfig,
    local_path: str | None = None,
    checksum: str | None = None,
    retrieved_at: str | None = None,
    etag: str | None = None,
    last_modified: str | None = None,
    resolved_commit: str | None = None,
) -> SourceEntry:
    """Create a SourceEntry from a SourceConfig with download metadata.

    Args:
        source_config: Source configuration
        local_path: Path to downloaded artifact in raw/ directory
        checksum: SHA256 checksum of downloaded content
        retrieved_at: ISO 8601 timestamp of retrieval
        etag: HTTP ETag header value
        last_modified: HTTP Last-Modified header value
        resolved_commit: Resolved git commit hash (for git sources)

    Returns:
        SourceEntry with full provenance information
    """
    return SourceEntry(
        id=source_config.id,
        type=source_config.type.value,
        url=source_config.url,
        license=source_config.license,
        notes=source_config.notes,
        retrieved_at=retrieved_at,
        checksum_sha256=checksum,
        etag=etag,
        last_modified=last_modified,
        pinned_revision=source_config.pinned_revision,
        resolved_commit=resolved_commit,
        local_path=local_path,
    )


def create_source_manifest(
    source_entries: list[SourceEntry],
    generated_at: str | None = None,
) -> SourceManifest:
    """Create a source manifest from a list of source entries.

    Args:
        source_entries: List of SourceEntry objects with provenance info
        generated_at: ISO 8601 timestamp (defaults to now)

    Returns:
        SourceManifest ready for serialization
    """
    if generated_at is None:
        generated_at = datetime.now(UTC).isoformat()

    return SourceManifest(
        sources=source_entries,
        generated_at=generated_at,
    )


def save_source_manifest(
    manifest: SourceManifest,
    output_path: Path,
) -> str:
    """Save source manifest to JSON file.

    Args:
        manifest: SourceManifest to save
        output_path: Path to write manifest JSON

    Returns:
        SHA256 checksum of written file
    """
    from finetune.dataset_agent.writers import write_json

    return write_json(output_path, manifest)


def get_local_path_for_source(
    source_config: SourceConfig,
    raw_dir: Path,
) -> Path:
    """Get the expected local path for a source's downloaded artifact.

    Args:
        source_config: Source configuration
        raw_dir: Path to raw/ directory

    Returns:
        Path where the downloaded artifact should be stored
    """
    filename = get_deterministic_filename(source_config)
    return raw_dir / filename
