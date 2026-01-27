"""Template loader for dataset generation pipeline.

This module provides functionality to load and validate template YAML files.
Templates define NL→ADS query patterns with slots for entity/topic references.

Template YAML format:
    id: topic_search
    intent: topic_search
    nl_templates:
      - "papers about {topic}"
      - "research on {topic}"
    ads_query_template: "abs:{topic}"
    slots:
      topic:
        name: topic
        type: topic
        required: true
        constraints:
          catalog: topic_catalog
    constraints:
      min_results: 1
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import Template, TemplateSlot
from .validation import SchemaValidationError, validate_template


class TemplateLoadError(Exception):
    """Raised when template loading or validation fails.

    Attributes:
        file_path: Path to the template file that failed
        line_number: Approximate line number where error occurred (if known)
        field_name: Field that caused the error (if applicable)
        message: Human-readable error description
    """

    def __init__(
        self,
        message: str,
        file_path: str | Path | None = None,
        line_number: int | None = None,
        field_name: str | None = None,
    ) -> None:
        self.file_path = str(file_path) if file_path else None
        self.line_number = line_number
        self.field_name = field_name
        self.message = message
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with context."""
        parts = []
        if self.file_path:
            parts.append(f"[{self.file_path}")
            if self.line_number:
                parts[-1] += f":{self.line_number}"
            parts[-1] += "]"
        if self.field_name:
            parts.append(f"field '{self.field_name}':")
        parts.append(self.message)
        return " ".join(parts)


# Valid slot types for templates
VALID_SLOT_TYPES = {"topic", "entity", "literal", "date", "author", "bibcode"}


@dataclass
class TemplateLoadResult:
    """Result of loading templates from a directory.

    Attributes:
        templates: List of successfully loaded templates
        errors: List of errors encountered during loading
        templates_by_id: Dict mapping template ID to template
        templates_by_intent: Dict mapping intent to list of templates
    """

    templates: list[Template]
    errors: list[TemplateLoadError]

    @property
    def templates_by_id(self) -> dict[str, Template]:
        """Get templates indexed by ID."""
        return {t.id: t for t in self.templates}

    @property
    def templates_by_intent(self) -> dict[str, list[Template]]:
        """Get templates grouped by intent."""
        result: dict[str, list[Template]] = {}
        for t in self.templates:
            if t.intent not in result:
                result[t.intent] = []
            result[t.intent].append(t)
        return result

    @property
    def success(self) -> bool:
        """Whether all templates loaded successfully."""
        return len(self.errors) == 0


def parse_template_yaml(data: dict[str, Any], file_path: Path | None = None) -> Template:
    """Parse a template from YAML data.

    Args:
        data: Dictionary parsed from YAML
        file_path: Path to source file for error messages

    Returns:
        Parsed Template object

    Raises:
        TemplateLoadError: If required fields are missing or invalid
    """
    # Validate required fields
    required_fields = ["id", "intent", "nl_templates", "ads_query_template"]
    for field in required_fields:
        if field not in data:
            raise TemplateLoadError(
                f"Missing required field '{field}'",
                file_path=file_path,
                field_name=field,
            )

    # Parse slots
    slots_raw = data.get("slots", {})
    if not isinstance(slots_raw, dict):
        raise TemplateLoadError(
            "Expected dict for 'slots'",
            file_path=file_path,
            field_name="slots",
        )

    slots: dict[str, TemplateSlot] = {}
    for slot_name, slot_data in slots_raw.items():
        if isinstance(slot_data, dict):
            # Validate slot type
            slot_type = slot_data.get("type")
            if slot_type and slot_type not in VALID_SLOT_TYPES:
                raise TemplateLoadError(
                    f"Invalid slot type '{slot_type}'. Valid types: {VALID_SLOT_TYPES}",
                    file_path=file_path,
                    field_name=f"slots.{slot_name}.type",
                )
            slots[slot_name] = TemplateSlot(
                name=slot_data.get("name", slot_name),
                type=slot_data.get("type", "literal"),
                required=slot_data.get("required", True),
                constraints=slot_data.get("constraints", {}),
            )
        else:
            raise TemplateLoadError(
                f"Slot '{slot_name}' must be a dict",
                file_path=file_path,
                field_name=f"slots.{slot_name}",
            )

    # Validate nl_templates is a non-empty list
    nl_templates = data.get("nl_templates", [])
    if not isinstance(nl_templates, list):
        raise TemplateLoadError(
            "Expected list for 'nl_templates'",
            file_path=file_path,
            field_name="nl_templates",
        )
    if not nl_templates:
        raise TemplateLoadError(
            "'nl_templates' cannot be empty",
            file_path=file_path,
            field_name="nl_templates",
        )

    # Validate all nl_templates are strings
    for i, tmpl in enumerate(nl_templates):
        if not isinstance(tmpl, str):
            raise TemplateLoadError(
                f"nl_templates[{i}] must be a string",
                file_path=file_path,
                field_name=f"nl_templates[{i}]",
            )

    # Validate slot references in templates
    _validate_slot_references(nl_templates, slots, "nl_templates", file_path)
    _validate_slot_references(
        [data["ads_query_template"]], slots, "ads_query_template", file_path
    )

    return Template(
        id=data["id"],
        intent=data["intent"],
        nl_templates=nl_templates,
        ads_query_template=data["ads_query_template"],
        slots=slots,
        constraints=data.get("constraints", {}),
    )


def _validate_slot_references(
    templates: list[str],
    slots: dict[str, TemplateSlot],
    field_name: str,
    file_path: Path | None = None,
) -> None:
    """Validate that slot references in templates match defined slots.

    Args:
        templates: List of template strings with {slot} placeholders
        slots: Defined slots
        field_name: Name of field being validated for error messages
        file_path: Source file path for error messages

    Raises:
        TemplateLoadError: If template references undefined slot
    """
    slot_pattern = re.compile(r"\{(\w+)\}")

    for i, template in enumerate(templates):
        referenced_slots = set(slot_pattern.findall(template))
        defined_slots = set(slots.keys())

        # Check for undefined slot references
        undefined = referenced_slots - defined_slots
        if undefined:
            raise TemplateLoadError(
                f"Template references undefined slots: {undefined}. "
                f"Defined slots: {defined_slots}",
                file_path=file_path,
                field_name=f"{field_name}[{i}]" if len(templates) > 1 else field_name,
            )


def load_template_file(file_path: Path) -> list[Template]:
    """Load templates from a single YAML file.

    A YAML file can contain either:
    - A single template (dict at root)
    - Multiple templates (list at root)

    Args:
        file_path: Path to YAML file

    Returns:
        List of parsed templates

    Raises:
        TemplateLoadError: If file cannot be read or parsed
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise TemplateLoadError(
            f"Cannot read file: {e}",
            file_path=file_path,
        ) from e

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise TemplateLoadError(
            f"Invalid YAML: {e}",
            file_path=file_path,
        ) from e

    if data is None:
        raise TemplateLoadError(
            "Empty YAML file",
            file_path=file_path,
        )

    # Handle single template or list of templates
    templates_data: list[dict[str, Any]]
    if isinstance(data, dict):
        templates_data = [data]
    elif isinstance(data, list):
        templates_data = data
    else:
        raise TemplateLoadError(
            f"Expected dict or list at root, got {type(data).__name__}",
            file_path=file_path,
        )

    templates = []
    for i, tmpl_data in enumerate(templates_data):
        if not isinstance(tmpl_data, dict):
            raise TemplateLoadError(
                f"Template {i} must be a dict",
                file_path=file_path,
            )
        template = parse_template_yaml(tmpl_data, file_path)
        templates.append(template)

    return templates


def load_templates_from_directory(
    templates_dir: Path,
    pattern: str = "*.yaml",
    fail_fast: bool = False,
) -> TemplateLoadResult:
    """Load all templates from a directory.

    Args:
        templates_dir: Directory containing template YAML files
        pattern: Glob pattern for template files (default: *.yaml)
        fail_fast: If True, raise on first error; if False, collect all errors

    Returns:
        TemplateLoadResult with templates and any errors

    Raises:
        TemplateLoadError: If fail_fast=True and any error occurs
    """
    if not templates_dir.exists():
        raise TemplateLoadError(
            f"Templates directory does not exist: {templates_dir}",
        )

    if not templates_dir.is_dir():
        raise TemplateLoadError(
            f"Not a directory: {templates_dir}",
        )

    templates: list[Template] = []
    errors: list[TemplateLoadError] = []

    # Find all template files
    template_files = sorted(templates_dir.glob(pattern))
    if not template_files:
        errors.append(
            TemplateLoadError(
                f"No template files found matching pattern '{pattern}'",
                file_path=templates_dir,
            )
        )
        return TemplateLoadResult(templates=templates, errors=errors)

    # Load each file
    for file_path in template_files:
        try:
            file_templates = load_template_file(file_path)

            # Validate each template with schema validator
            for template in file_templates:
                try:
                    validate_template(template)
                    templates.append(template)
                except SchemaValidationError as e:
                    error = TemplateLoadError(
                        f"Schema validation failed: {e.message}",
                        file_path=file_path,
                        field_name=e.field_name,
                    )
                    if fail_fast:
                        raise error
                    errors.append(error)

        except TemplateLoadError as e:
            if fail_fast:
                raise
            errors.append(e)

    # Check for duplicate template IDs
    seen_ids: dict[str, Path] = {}
    for template_file in template_files:
        try:
            file_templates = load_template_file(template_file)
            for template in file_templates:
                if template.id in seen_ids:
                    error = TemplateLoadError(
                        f"Duplicate template ID '{template.id}' "
                        f"(also in {seen_ids[template.id]})",
                        file_path=template_file,
                        field_name="id",
                    )
                    if fail_fast:
                        raise error
                    # Only add if not already in errors list
                    if not any(e.message.startswith(f"Duplicate template ID '{template.id}'") for e in errors):
                        errors.append(error)
                else:
                    seen_ids[template.id] = template_file
        except TemplateLoadError:
            # Already handled above
            pass

    return TemplateLoadResult(templates=templates, errors=errors)


def iterate_templates(templates_dir: Path, pattern: str = "*.yaml") -> Iterator[Template]:
    """Iterate over templates from a directory, yielding each template.

    This is a streaming interface that yields templates one at a time.
    Errors are raised as TemplateLoadError.

    Args:
        templates_dir: Directory containing template YAML files
        pattern: Glob pattern for template files

    Yields:
        Template objects

    Raises:
        TemplateLoadError: If any template fails to load
    """
    result = load_templates_from_directory(templates_dir, pattern, fail_fast=True)
    yield from result.templates


def get_default_templates_dir() -> Path:
    """Get the default templates directory path.

    Returns:
        Path to templates/ directory relative to this module
    """
    return Path(__file__).parent / "templates"
