"""Canonical artifact schemas for dataset generation pipeline.

This module defines typed models for all output artifacts produced by the pipeline.
All schemas use standard Python dataclasses for consistency with the codebase.

Artifact types:
- SourceManifest: Records provenance of external data sources
- TopicCatalog: Normalized topic entries from UAT vocabulary
- EntityCatalog: Normalized entity entries (institutions from ROR, etc.)
- Template: NL→ADS query template definitions
- Pair: NL input / ADS query training pairs
- EnrichmentLabel: Entity/topic labels for enrichment training
- Report: Pipeline run summary statistics
- RunManifest: Top-level run metadata with checksums
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from json import dumps as json_dumps
from json import loads as json_loads
from typing import Any


class SourceType(str, Enum):
    """Type of external data source."""

    HTTP = "http"
    GIT = "git"


class Stage(str, Enum):
    """Pipeline stages for tracking completion."""

    FETCH = "fetch"
    NORMALIZE = "normalize"
    EXPAND_ALIASES = "expand_aliases"
    LOAD_TEMPLATES = "load_templates"
    GENERATE_INPUTS = "generate_inputs"
    RENDER_PAIRS = "render_pairs"
    VALIDATE_LOCAL = "validate_local"
    VALIDATE_BACKEND = "validate_backend"
    GENERATE_ENRICHMENT = "generate_enrichment"
    REPORT = "report"


class LabelType(str, Enum):
    """Types of enrichment labels."""

    TOPIC = "topic"
    INSTITUTION = "institution"
    AUTHOR = "author"
    DATE_RANGE = "date_range"


# =============================================================================
# Source Manifest Schemas
# =============================================================================


@dataclass
class SourceEntry:
    """A single source in the source manifest.

    Records provenance information for one external data source.
    """

    id: str  # Unique identifier for this source
    type: str  # http or git
    url: str  # Source URL
    license: str | None = None  # License identifier (e.g., CC-BY-4.0)
    notes: str | None = None  # Human-readable notes
    retrieved_at: str | None = None  # ISO 8601 timestamp
    checksum_sha256: str | None = None  # SHA256 of downloaded content
    etag: str | None = None  # HTTP ETag if available
    last_modified: str | None = None  # HTTP Last-Modified if available
    pinned_revision: str | None = None  # Git commit/tag (for git sources)
    resolved_commit: str | None = None  # Actual commit hash (for git sources)
    local_path: str | None = None  # Path in raw/ directory

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceEntry":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SourceManifest:
    """Manifest of all external sources used in a pipeline run.

    File: manifests/source_manifest.json
    """

    sources: list[SourceEntry] = field(default_factory=list)
    generated_at: str | None = None  # ISO 8601 timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "generated_at": self.generated_at,
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceManifest":
        """Create from dictionary."""
        return cls(
            generated_at=d.get("generated_at"),
            sources=[SourceEntry.from_dict(s) for s in d.get("sources", [])],
        )


# =============================================================================
# Topic Catalog Schema
# =============================================================================


@dataclass
class TopicEntry:
    """A normalized topic entry from UAT or similar vocabulary.

    File: normalized/topic_catalog.jsonl (one entry per line)
    """

    id: str  # Canonical topic ID (e.g., UAT concept URI)
    label: str  # Preferred label
    aliases: list[str] = field(default_factory=list)  # Alternative labels
    parents: list[str] = field(default_factory=list)  # Broader topic IDs
    children: list[str] = field(default_factory=list)  # Narrower topic IDs
    source_id: str | None = None  # Reference to source in source_manifest
    domain_tags: list[str] = field(default_factory=list)  # e.g., ['astronomy'], ['earthscience']
    source_vocabulary: str = ""  # e.g., 'uat', 'sweet', 'gcmd'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in d.items() if v is not None or k in ("id", "label")}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TopicEntry":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Entity Catalog Schema
# =============================================================================


@dataclass
class EntityEntry:
    """A normalized entity entry (institution, author, etc.).

    File: normalized/entity_catalog_<type>.jsonl (one entry per line)
    """

    id: str  # Canonical entity ID (e.g., ROR ID)
    label: str  # Primary name
    aliases: list[str] = field(default_factory=list)  # Alternative names
    metadata: dict[str, Any] = field(default_factory=dict)  # Type-specific metadata
    source_id: str | None = None  # Reference to source in source_manifest
    entity_subtype: str = ""  # e.g., 'crater', 'mons', 'observatory'
    domain_tags: list[str] = field(default_factory=list)  # e.g., ['planetary'], ['multidisciplinary']
    source_vocabulary: str = ""  # e.g., 'ror', 'planetary'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None or k in ("id", "label")}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EntityEntry":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Template Schema
# =============================================================================


@dataclass
class TemplateSlot:
    """Definition of a slot in a template."""

    name: str  # Slot name (e.g., "topic", "author")
    type: str  # Slot type: "topic", "entity", "literal", "date"
    required: bool = True  # Whether the slot must be filled
    constraints: dict[str, Any] = field(default_factory=dict)  # Type-specific constraints

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TemplateSlot":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Template:
    """A template for generating NL→ADS query pairs.

    Templates define:
    - Natural language patterns with slots
    - Corresponding ADS query patterns
    - Constraints on how slots can be filled

    File: templates/*.yaml (loaded into memory)
    """

    id: str  # Unique template ID
    intent: str  # Intent category (e.g., "topic_search", "author_search")
    nl_templates: list[str]  # NL patterns with {slot} placeholders
    ads_query_template: str  # ADS query pattern with {slot} placeholders
    slots: dict[str, TemplateSlot] = field(default_factory=dict)  # Slot definitions
    constraints: dict[str, Any] = field(default_factory=dict)  # Global constraints

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["slots"] = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.slots.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Template":
        """Create from dictionary."""
        slots_raw = d.get("slots", {})
        slots = {}
        for k, v in slots_raw.items():
            if isinstance(v, dict):
                slots[k] = TemplateSlot.from_dict(v)
            else:
                slots[k] = v
        return cls(
            id=d["id"],
            intent=d["intent"],
            nl_templates=d.get("nl_templates", []),
            ads_query_template=d.get("ads_query_template", ""),
            slots=slots,
            constraints=d.get("constraints", {}),
        )


# =============================================================================
# Pair Schema
# =============================================================================


@dataclass
class NLInput:
    """A generated natural language input.

    File: pairs/inputs.jsonl (one entry per line)
    """

    input_id: str  # Unique input ID
    user_text: str  # Generated NL text
    template_id: str  # ID of template used
    filled_slots: dict[str, Any] = field(default_factory=dict)  # Slot values used
    source_ids: list[str] = field(default_factory=list)  # Catalog entry IDs used

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NLInput":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Pair:
    """A NL input / ADS query training pair.

    File: pairs/pairs.jsonl (one entry per line)
    """

    pair_id: str  # Unique pair ID
    user_text: str  # Natural language input
    ads_query: str  # Generated ADS query
    template_id: str  # ID of template used
    filled_slots: dict[str, Any] = field(default_factory=dict)  # Slot values
    validation_tier: int = 0  # 0=none, 1=local, 2=backend_syntax, 3=backend_results
    validation_errors: list[str] = field(default_factory=list)  # Error messages if invalid

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Pair":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class QuarantinedPair:
    """A pair that failed validation.

    File: pairs/quarantine.jsonl (one entry per line)
    """

    pair_id: str
    user_text: str
    ads_query: str
    template_id: str
    filled_slots: dict[str, Any] = field(default_factory=dict)
    error_type: str = ""  # Category of error
    error_details: str = ""  # Detailed error message
    failed_at_tier: int = 0  # Which validation tier failed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuarantinedPair":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Enrichment Label Schema
# =============================================================================


@dataclass
class EnrichmentLabel:
    """An enrichment label for training entity/topic extraction.

    File: enrichment/enrichment_labels.jsonl (one entry per line)
    """

    example_id: str  # Unique example ID
    user_text: str  # Natural language input
    labels: list[dict[str, Any]] = field(default_factory=list)  # List of label dicts
    provenance: dict[str, Any] = field(default_factory=dict)  # Template/slot info

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnrichmentLabel":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Label:
    """A single label within an enrichment example.

    This is embedded in EnrichmentLabel.labels list.
    """

    entity_id: str  # ID from topic/entity catalog
    entity_type: str  # topic, institution, author, date_range
    text_span: str | None = None  # Text in user_text that triggered this label
    start_char: int | None = None  # Character offset start (optional)
    end_char: int | None = None  # Character offset end (optional)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Label":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Report Schema
# =============================================================================


@dataclass
class Report:
    """Pipeline run summary report.

    File: reports/summary.json
    """

    # Counts
    sources_count: int = 0
    topics_count: int = 0
    entities_count: int = 0
    templates_count: int = 0
    inputs_count: int = 0
    pairs_valid_count: int = 0
    pairs_quarantined_count: int = 0
    enrichment_labels_count: int = 0

    # Rates
    backend_pass_rate: float | None = None  # Populated if backend validation ran

    # Breakdown by source and domain
    entries_by_source: dict[str, int] = field(default_factory=dict)
    entries_by_domain: dict[str, int] = field(default_factory=dict)

    # Breakdown by category
    pairs_by_template: dict[str, int] = field(default_factory=dict)
    pairs_by_intent: dict[str, int] = field(default_factory=dict)
    quarantine_by_error_type: dict[str, int] = field(default_factory=dict)

    # Timing
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Report":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Run Manifest Schema
# =============================================================================


@dataclass
class ArtifactChecksum:
    """Checksum record for a single artifact file."""

    path: str  # Relative path from run directory
    checksum_sha256: str  # SHA256 of file contents
    size_bytes: int = 0  # File size
    line_count: int | None = None  # For JSONL files

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArtifactChecksum":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReproduceInfo:
    """Information needed to reproduce a pipeline run."""

    config_path: str | None = None  # Path to config file used
    config_checksum: str | None = None  # SHA256 of config
    seed: int | None = None  # Random seed used
    pinned_revisions: dict[str, str] = field(default_factory=dict)  # source_id -> commit

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReproduceInfo":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RunManifest:
    """Top-level manifest for a pipeline run.

    File: manifests/run_manifest.json
    """

    run_id: str
    created_at: str  # ISO 8601
    started_at: str | None = None  # ISO 8601
    completed_at: str | None = None  # ISO 8601
    run_directory: str = ""
    config_path: str | None = None
    status: str = "pending"  # pending, in_progress, completed, failed
    stages_completed: list[str] = field(default_factory=list)
    current_stage: str | None = None
    error_message: str | None = None  # If status == failed
    artifacts: dict[str, ArtifactChecksum] = field(default_factory=dict)
    reproduce: ReproduceInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "run_directory": self.run_directory,
            "config_path": self.config_path,
            "status": self.status,
            "stages_completed": self.stages_completed,
            "current_stage": self.current_stage,
            "error_message": self.error_message,
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
        }
        if self.reproduce:
            d["reproduce"] = self.reproduce.to_dict()
        # Remove None values for cleaner output
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunManifest":
        """Create from dictionary."""
        artifacts_raw = d.get("artifacts", {})
        artifacts = {}
        for k, v in artifacts_raw.items():
            if isinstance(v, dict):
                artifacts[k] = ArtifactChecksum.from_dict(v)
            else:
                artifacts[k] = v

        reproduce = None
        if d.get("reproduce"):
            reproduce = ReproduceInfo.from_dict(d["reproduce"])

        return cls(
            run_id=d["run_id"],
            created_at=d["created_at"],
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            run_directory=d.get("run_directory", ""),
            config_path=d.get("config_path"),
            status=d.get("status", "pending"),
            stages_completed=d.get("stages_completed", []),
            current_stage=d.get("current_stage"),
            error_message=d.get("error_message"),
            artifacts=artifacts,
            reproduce=reproduce,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json_dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "RunManifest":
        """Deserialize from JSON string."""
        return cls.from_dict(json_loads(json_str))
