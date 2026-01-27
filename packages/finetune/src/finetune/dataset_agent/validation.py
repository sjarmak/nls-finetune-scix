"""Schema validation for dataset generation pipeline artifacts.

This module provides validation functions that check artifacts against their
schemas before writing. Validation failures raise SchemaValidationError with
detailed error messages.

Validation is applied at write time to ensure all persisted artifacts are valid.
"""

from typing import Any, TypeVar

from .schemas import (
    ArtifactChecksum,
    EnrichmentLabel,
    EntityEntry,
    NLInput,
    Pair,
    QuarantinedPair,
    Report,
    ReproduceInfo,
    RunManifest,
    SourceEntry,
    SourceManifest,
    SourceType,
    Template,
    TemplateSlot,
    TopicEntry,
)

T = TypeVar("T")


class SchemaValidationError(Exception):
    """Raised when artifact validation fails.

    Attributes:
        schema_name: Name of the schema that failed validation
        field_name: Name of the field with invalid value (if applicable)
        message: Human-readable error description
    """

    def __init__(
        self,
        message: str,
        schema_name: str | None = None,
        field_name: str | None = None,
    ) -> None:
        self.schema_name = schema_name
        self.field_name = field_name
        self.message = message
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with context."""
        parts = []
        if self.schema_name:
            parts.append(f"[{self.schema_name}]")
        if self.field_name:
            parts.append(f"field '{self.field_name}':")
        parts.append(self.message)
        return " ".join(parts)


def validate_required_string(
    value: Any,
    field_name: str,
    schema_name: str,
    allow_empty: bool = False,
) -> None:
    """Validate that a value is a non-empty string.

    Args:
        value: Value to validate
        field_name: Name of field for error messages
        schema_name: Name of schema for error messages
        allow_empty: Whether to allow empty strings

    Raises:
        SchemaValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise SchemaValidationError(
            f"expected string, got {type(value).__name__}",
            schema_name=schema_name,
            field_name=field_name,
        )
    if not allow_empty and not value.strip():
        raise SchemaValidationError(
            "cannot be empty or whitespace-only",
            schema_name=schema_name,
            field_name=field_name,
        )


def validate_optional_string(
    value: Any,
    field_name: str,
    schema_name: str,
) -> None:
    """Validate that a value is None or a string.

    Args:
        value: Value to validate
        field_name: Name of field for error messages
        schema_name: Name of schema for error messages

    Raises:
        SchemaValidationError: If validation fails
    """
    if value is not None and not isinstance(value, str):
        raise SchemaValidationError(
            f"expected string or None, got {type(value).__name__}",
            schema_name=schema_name,
            field_name=field_name,
        )


def validate_list_of_strings(
    value: Any,
    field_name: str,
    schema_name: str,
    allow_empty_list: bool = True,
    allow_empty_strings: bool = False,
) -> None:
    """Validate that a value is a list of strings.

    Args:
        value: Value to validate
        field_name: Name of field for error messages
        schema_name: Name of schema for error messages
        allow_empty_list: Whether to allow empty lists
        allow_empty_strings: Whether to allow empty strings in list

    Raises:
        SchemaValidationError: If validation fails
    """
    if not isinstance(value, list):
        raise SchemaValidationError(
            f"expected list, got {type(value).__name__}",
            schema_name=schema_name,
            field_name=field_name,
        )
    if not allow_empty_list and not value:
        raise SchemaValidationError(
            "cannot be empty",
            schema_name=schema_name,
            field_name=field_name,
        )
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise SchemaValidationError(
                f"item {i} expected string, got {type(item).__name__}",
                schema_name=schema_name,
                field_name=field_name,
            )
        if not allow_empty_strings and not item.strip():
            raise SchemaValidationError(
                f"item {i} cannot be empty or whitespace-only",
                schema_name=schema_name,
                field_name=field_name,
            )


def validate_dict(
    value: Any,
    field_name: str,
    schema_name: str,
) -> None:
    """Validate that a value is a dict.

    Args:
        value: Value to validate
        field_name: Name of field for error messages
        schema_name: Name of schema for error messages

    Raises:
        SchemaValidationError: If validation fails
    """
    if not isinstance(value, dict):
        raise SchemaValidationError(
            f"expected dict, got {type(value).__name__}",
            schema_name=schema_name,
            field_name=field_name,
        )


def validate_iso_timestamp(
    value: Any,
    field_name: str,
    schema_name: str,
    allow_none: bool = True,
) -> None:
    """Validate that a value is an ISO 8601 timestamp string.

    Args:
        value: Value to validate
        field_name: Name of field for error messages
        schema_name: Name of schema for error messages
        allow_none: Whether to allow None values

    Raises:
        SchemaValidationError: If validation fails
    """
    if value is None:
        if not allow_none:
            raise SchemaValidationError(
                "cannot be None",
                schema_name=schema_name,
                field_name=field_name,
            )
        return

    if not isinstance(value, str):
        raise SchemaValidationError(
            f"expected ISO 8601 string, got {type(value).__name__}",
            schema_name=schema_name,
            field_name=field_name,
        )

    # Basic ISO 8601 format check (YYYY-MM-DDTHH:MM:SS)
    # Accept variations like timezone suffixes, microseconds, etc.
    import re

    iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    if not re.match(iso_pattern, value):
        raise SchemaValidationError(
            f"not a valid ISO 8601 timestamp: {value}",
            schema_name=schema_name,
            field_name=field_name,
        )


# =============================================================================
# Schema-specific validators
# =============================================================================


def validate_source_entry(entry: SourceEntry) -> None:
    """Validate a SourceEntry.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "SourceEntry"
    validate_required_string(entry.id, "id", schema)
    validate_required_string(entry.type, "type", schema)

    # Validate type is http or git
    valid_types = {t.value for t in SourceType}
    if entry.type not in valid_types:
        raise SchemaValidationError(
            f"must be one of {valid_types}, got '{entry.type}'",
            schema_name=schema,
            field_name="type",
        )

    validate_required_string(entry.url, "url", schema)
    validate_optional_string(entry.license, "license", schema)
    validate_optional_string(entry.notes, "notes", schema)
    validate_iso_timestamp(entry.retrieved_at, "retrieved_at", schema)
    validate_optional_string(entry.checksum_sha256, "checksum_sha256", schema)
    validate_optional_string(entry.pinned_revision, "pinned_revision", schema)
    validate_optional_string(entry.resolved_commit, "resolved_commit", schema)
    validate_optional_string(entry.local_path, "local_path", schema)


def validate_source_manifest(manifest: SourceManifest) -> None:
    """Validate a SourceManifest.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "SourceManifest"
    validate_iso_timestamp(manifest.generated_at, "generated_at", schema)

    if not isinstance(manifest.sources, list):
        raise SchemaValidationError(
            "expected list",
            schema_name=schema,
            field_name="sources",
        )

    for i, source in enumerate(manifest.sources):
        try:
            validate_source_entry(source)
        except SchemaValidationError as e:
            raise SchemaValidationError(
                f"sources[{i}]: {e.message}",
                schema_name=schema,
            ) from e


def validate_topic_entry(entry: TopicEntry) -> None:
    """Validate a TopicEntry.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "TopicEntry"
    validate_required_string(entry.id, "id", schema)
    validate_required_string(entry.label, "label", schema)
    validate_list_of_strings(entry.aliases, "aliases", schema, allow_empty_strings=False)
    validate_list_of_strings(entry.parents, "parents", schema)
    validate_list_of_strings(entry.children, "children", schema)
    validate_optional_string(entry.source_id, "source_id", schema)


def validate_entity_entry(entry: EntityEntry) -> None:
    """Validate an EntityEntry.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "EntityEntry"
    validate_required_string(entry.id, "id", schema)
    validate_required_string(entry.label, "label", schema)
    validate_list_of_strings(entry.aliases, "aliases", schema, allow_empty_strings=False)
    validate_dict(entry.metadata, "metadata", schema)
    validate_optional_string(entry.source_id, "source_id", schema)


def validate_template_slot(slot: TemplateSlot, slot_name: str) -> None:
    """Validate a TemplateSlot.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = f"TemplateSlot[{slot_name}]"
    validate_required_string(slot.name, "name", schema)
    validate_required_string(slot.type, "type", schema)

    if not isinstance(slot.required, bool):
        raise SchemaValidationError(
            f"expected bool, got {type(slot.required).__name__}",
            schema_name=schema,
            field_name="required",
        )

    validate_dict(slot.constraints, "constraints", schema)


def validate_template(template: Template) -> None:
    """Validate a Template.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "Template"
    validate_required_string(template.id, "id", schema)
    validate_required_string(template.intent, "intent", schema)
    validate_list_of_strings(
        template.nl_templates,
        "nl_templates",
        schema,
        allow_empty_list=False,
    )
    validate_required_string(template.ads_query_template, "ads_query_template", schema)

    if not isinstance(template.slots, dict):
        raise SchemaValidationError(
            "expected dict",
            schema_name=schema,
            field_name="slots",
        )

    for slot_name, slot in template.slots.items():
        if isinstance(slot, TemplateSlot):
            validate_template_slot(slot, slot_name)

    validate_dict(template.constraints, "constraints", schema)


def validate_nl_input(nl_input: NLInput) -> None:
    """Validate an NLInput.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "NLInput"
    validate_required_string(nl_input.input_id, "input_id", schema)
    validate_required_string(nl_input.user_text, "user_text", schema)
    validate_required_string(nl_input.template_id, "template_id", schema)
    validate_dict(nl_input.filled_slots, "filled_slots", schema)
    validate_list_of_strings(nl_input.source_ids, "source_ids", schema)


def validate_pair(pair: Pair) -> None:
    """Validate a Pair.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "Pair"
    validate_required_string(pair.pair_id, "pair_id", schema)
    validate_required_string(pair.user_text, "user_text", schema)
    validate_required_string(pair.ads_query, "ads_query", schema)
    validate_required_string(pair.template_id, "template_id", schema)
    validate_dict(pair.filled_slots, "filled_slots", schema)

    if not isinstance(pair.validation_tier, int):
        raise SchemaValidationError(
            f"expected int, got {type(pair.validation_tier).__name__}",
            schema_name=schema,
            field_name="validation_tier",
        )

    if not 0 <= pair.validation_tier <= 3:
        raise SchemaValidationError(
            f"must be 0-3, got {pair.validation_tier}",
            schema_name=schema,
            field_name="validation_tier",
        )

    validate_list_of_strings(
        pair.validation_errors,
        "validation_errors",
        schema,
        allow_empty_strings=True,
    )


def validate_quarantined_pair(pair: QuarantinedPair) -> None:
    """Validate a QuarantinedPair.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "QuarantinedPair"
    validate_required_string(pair.pair_id, "pair_id", schema)
    validate_required_string(pair.user_text, "user_text", schema)
    validate_required_string(pair.ads_query, "ads_query", schema)
    validate_required_string(pair.template_id, "template_id", schema)
    validate_dict(pair.filled_slots, "filled_slots", schema)
    validate_required_string(pair.error_type, "error_type", schema, allow_empty=True)
    validate_required_string(pair.error_details, "error_details", schema, allow_empty=True)

    if not isinstance(pair.failed_at_tier, int):
        raise SchemaValidationError(
            f"expected int, got {type(pair.failed_at_tier).__name__}",
            schema_name=schema,
            field_name="failed_at_tier",
        )


def validate_enrichment_label(label: EnrichmentLabel) -> None:
    """Validate an EnrichmentLabel.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "EnrichmentLabel"
    validate_required_string(label.example_id, "example_id", schema)
    validate_required_string(label.user_text, "user_text", schema)

    if not isinstance(label.labels, list):
        raise SchemaValidationError(
            "expected list",
            schema_name=schema,
            field_name="labels",
        )

    validate_dict(label.provenance, "provenance", schema)


def validate_report(report: Report) -> None:
    """Validate a Report.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "Report"

    # Validate count fields are non-negative integers
    count_fields = [
        "sources_count",
        "topics_count",
        "entities_count",
        "templates_count",
        "inputs_count",
        "pairs_valid_count",
        "pairs_quarantined_count",
        "enrichment_labels_count",
    ]

    for field_name in count_fields:
        value = getattr(report, field_name)
        if not isinstance(value, int):
            raise SchemaValidationError(
                f"expected int, got {type(value).__name__}",
                schema_name=schema,
                field_name=field_name,
            )
        if value < 0:
            raise SchemaValidationError(
                f"cannot be negative: {value}",
                schema_name=schema,
                field_name=field_name,
            )

    # Validate optional rate
    if report.backend_pass_rate is not None:
        if not isinstance(report.backend_pass_rate, (int, float)):
            raise SchemaValidationError(
                f"expected number, got {type(report.backend_pass_rate).__name__}",
                schema_name=schema,
                field_name="backend_pass_rate",
            )
        if not 0.0 <= report.backend_pass_rate <= 1.0:
            raise SchemaValidationError(
                f"must be between 0.0 and 1.0, got {report.backend_pass_rate}",
                schema_name=schema,
                field_name="backend_pass_rate",
            )

    validate_dict(report.pairs_by_template, "pairs_by_template", schema)
    validate_dict(report.pairs_by_intent, "pairs_by_intent", schema)
    validate_dict(report.quarantine_by_error_type, "quarantine_by_error_type", schema)
    validate_iso_timestamp(report.started_at, "started_at", schema)
    validate_iso_timestamp(report.completed_at, "completed_at", schema)


def validate_artifact_checksum(checksum: ArtifactChecksum) -> None:
    """Validate an ArtifactChecksum.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "ArtifactChecksum"
    validate_required_string(checksum.path, "path", schema)
    validate_required_string(checksum.checksum_sha256, "checksum_sha256", schema)

    if not isinstance(checksum.size_bytes, int):
        raise SchemaValidationError(
            f"expected int, got {type(checksum.size_bytes).__name__}",
            schema_name=schema,
            field_name="size_bytes",
        )

    if checksum.size_bytes < 0:
        raise SchemaValidationError(
            f"cannot be negative: {checksum.size_bytes}",
            schema_name=schema,
            field_name="size_bytes",
        )

    if checksum.line_count is not None:
        if not isinstance(checksum.line_count, int):
            raise SchemaValidationError(
                f"expected int or None, got {type(checksum.line_count).__name__}",
                schema_name=schema,
                field_name="line_count",
            )


def validate_reproduce_info(info: ReproduceInfo) -> None:
    """Validate a ReproduceInfo.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "ReproduceInfo"
    validate_optional_string(info.config_path, "config_path", schema)
    validate_optional_string(info.config_checksum, "config_checksum", schema)

    if info.seed is not None and not isinstance(info.seed, int):
        raise SchemaValidationError(
            f"expected int or None, got {type(info.seed).__name__}",
            schema_name=schema,
            field_name="seed",
        )

    validate_dict(info.pinned_revisions, "pinned_revisions", schema)


def validate_run_manifest(manifest: RunManifest) -> None:
    """Validate a RunManifest.

    Raises:
        SchemaValidationError: If validation fails
    """
    schema = "RunManifest"
    validate_required_string(manifest.run_id, "run_id", schema)
    validate_iso_timestamp(manifest.created_at, "created_at", schema, allow_none=False)
    validate_iso_timestamp(manifest.started_at, "started_at", schema)
    validate_iso_timestamp(manifest.completed_at, "completed_at", schema)
    validate_required_string(manifest.run_directory, "run_directory", schema, allow_empty=True)
    validate_optional_string(manifest.config_path, "config_path", schema)

    # Validate status
    valid_statuses = {"pending", "in_progress", "completed", "failed"}
    if manifest.status not in valid_statuses:
        raise SchemaValidationError(
            f"must be one of {valid_statuses}, got '{manifest.status}'",
            schema_name=schema,
            field_name="status",
        )

    validate_list_of_strings(manifest.stages_completed, "stages_completed", schema)
    validate_optional_string(manifest.current_stage, "current_stage", schema)
    validate_optional_string(manifest.error_message, "error_message", schema)

    if not isinstance(manifest.artifacts, dict):
        raise SchemaValidationError(
            "expected dict",
            schema_name=schema,
            field_name="artifacts",
        )

    for artifact_name, checksum in manifest.artifacts.items():
        if isinstance(checksum, ArtifactChecksum):
            try:
                validate_artifact_checksum(checksum)
            except SchemaValidationError as e:
                raise SchemaValidationError(
                    f"artifacts['{artifact_name}']: {e.message}",
                    schema_name=schema,
                ) from e

    if manifest.reproduce is not None:
        try:
            validate_reproduce_info(manifest.reproduce)
        except SchemaValidationError as e:
            raise SchemaValidationError(
                f"reproduce: {e.message}",
                schema_name=schema,
            ) from e


# =============================================================================
# Dispatcher for automatic validation
# =============================================================================

VALIDATORS = {
    SourceEntry: validate_source_entry,
    SourceManifest: validate_source_manifest,
    TopicEntry: validate_topic_entry,
    EntityEntry: validate_entity_entry,
    Template: validate_template,
    NLInput: validate_nl_input,
    Pair: validate_pair,
    QuarantinedPair: validate_quarantined_pair,
    EnrichmentLabel: validate_enrichment_label,
    Report: validate_report,
    ArtifactChecksum: validate_artifact_checksum,
    ReproduceInfo: validate_reproduce_info,
    RunManifest: validate_run_manifest,
}


def validate(obj: Any) -> None:
    """Validate an artifact against its schema.

    Automatically dispatches to the correct validator based on object type.

    Args:
        obj: Artifact to validate

    Raises:
        SchemaValidationError: If validation fails
        ValueError: If no validator exists for the object type
    """
    obj_type = type(obj)
    validator = VALIDATORS.get(obj_type)

    if validator is None:
        raise ValueError(f"No validator registered for type: {obj_type.__name__}")

    validator(obj)


def validate_all(objects: list[Any]) -> list[SchemaValidationError]:
    """Validate a list of artifacts, collecting all errors.

    Args:
        objects: List of artifacts to validate

    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []
    for i, obj in enumerate(objects):
        try:
            validate(obj)
        except (SchemaValidationError, ValueError) as e:
            errors.append(
                SchemaValidationError(
                    f"item {i}: {e}",
                    schema_name=type(obj).__name__,
                )
            )
    return errors
