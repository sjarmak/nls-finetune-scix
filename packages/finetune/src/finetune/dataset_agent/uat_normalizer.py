"""UAT (Unified Astronomy Thesaurus) vocabulary normalizer.

This module parses UAT vocabulary data (from UAT_list.json format) and produces
normalized TopicEntry records for the topic_catalog.jsonl output.

UAT source: https://github.com/astrothesaurus/UAT
Format: JSON array of concept objects with fields:
  - uri: Concept identifier (e.g., "http://astrothesaurus.org/uat/1000")
  - name: Preferred label
  - altNames: Alternative labels (array or null)
  - broader: Parent concepts [{name, uri}] or null
  - narrower: Child concepts [{name, uri}] or null
  - related: Related concepts [{name, uri}] or null (not used in normalization)
  - definition: Concept description (not used in normalization)
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finetune.dataset_agent.schemas import TopicEntry

if TYPE_CHECKING:
    from collections.abc import Iterator


class UATParseError(Exception):
    """Error parsing UAT vocabulary data."""

    def __init__(self, message: str, source_path: str | None = None):
        self.source_path = source_path
        super().__init__(message)


def extract_concept_id(uri: str) -> str:
    """Extract the concept ID from a UAT URI.

    Args:
        uri: Full concept URI (e.g., "http://astrothesaurus.org/uat/1000")

    Returns:
        Concept ID (e.g., "uat:1000")
    """
    # Handle various URI formats
    if "/uat/" in uri:
        # Standard format: http://astrothesaurus.org/uat/1000
        numeric_id = uri.split("/uat/")[-1].strip("/")
        return f"uat:{numeric_id}"
    # Fallback: use the whole URI
    return uri


def normalize_label(label: str) -> str:
    """Normalize a label by stripping whitespace.

    Args:
        label: Raw label string

    Returns:
        Normalized label (stripped of leading/trailing whitespace)
    """
    if not label:
        return ""
    return label.strip()


def deduplicate_aliases(
    aliases: list[str],
    preferred_label: str,
    case_insensitive: bool = True,
) -> list[str]:
    """Deduplicate aliases, removing duplicates and the preferred label.

    Args:
        aliases: List of alternative labels
        preferred_label: The preferred/primary label to exclude
        case_insensitive: Whether to compare case-insensitively

    Returns:
        Deduplicated list of aliases (empty strings and whitespace-only removed)
    """
    seen: set[str] = set()
    result: list[str] = []

    # Normalize the preferred label for comparison
    pref_norm = preferred_label.lower() if case_insensitive else preferred_label

    for alias in aliases:
        normalized = normalize_label(alias)
        if not normalized:  # Skip empty/whitespace-only
            continue

        # Create comparison key
        key = normalized.lower() if case_insensitive else normalized

        # Skip if duplicate or matches preferred label
        if key in seen or key == pref_norm:
            continue

        seen.add(key)
        result.append(normalized)

    return result


def extract_relationship_ids(
    relationships: list[dict[str, Any]] | None,
) -> list[str]:
    """Extract concept IDs from a list of relationship objects.

    Args:
        relationships: List of {name, uri} objects or None

    Returns:
        List of concept IDs (e.g., ["uat:1000", "uat:2000"])
    """
    if not relationships:
        return []

    ids: list[str] = []
    for rel in relationships:
        if isinstance(rel, dict) and "uri" in rel:
            concept_id = extract_concept_id(rel["uri"])
            ids.append(concept_id)
    return ids


def parse_uat_concept(
    concept: dict[str, Any],
    source_id: str | None = None,
) -> TopicEntry | None:
    """Parse a single UAT concept into a TopicEntry.

    Args:
        concept: Raw concept dictionary from UAT JSON
        source_id: Optional source identifier for provenance

    Returns:
        TopicEntry or None if concept is invalid
    """
    # Extract required fields
    uri = concept.get("uri")
    name = concept.get("name")

    if not uri or not name:
        return None

    # Extract concept ID
    topic_id = extract_concept_id(uri)

    # Normalize preferred label
    preferred_label = normalize_label(name)
    if not preferred_label:
        return None

    # Extract and deduplicate aliases
    raw_aliases = concept.get("altNames") or []
    if not isinstance(raw_aliases, list):
        raw_aliases = []
    aliases = deduplicate_aliases(raw_aliases, preferred_label)

    # Extract parent (broader) and child (narrower) relationships
    parents = extract_relationship_ids(concept.get("broader"))
    children = extract_relationship_ids(concept.get("narrower"))

    return TopicEntry(
        id=topic_id,
        label=preferred_label,
        aliases=aliases,
        parents=parents,
        children=children,
        source_id=source_id,
        domain_tags=["astronomy"],
        source_vocabulary="uat",
    )


def load_uat_json(path: Path) -> list[dict[str, Any]]:
    """Load UAT concepts from a JSON file.

    Supports both plain JSON files and JSON files inside tar.gz archives.

    Args:
        path: Path to UAT_list.json or a tar.gz archive containing it

    Returns:
        List of concept dictionaries

    Raises:
        UATParseError: If the file cannot be parsed
    """
    try:
        if path.suffix == ".gz" or path.name.endswith(".tar.gz"):
            # Extract from tar.gz archive
            return _load_from_tarball(path)
        else:
            # Plain JSON file
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise UATParseError(
                    f"Expected JSON array at root, got {type(data).__name__}",
                    source_path=str(path),
                )
            return data

    except json.JSONDecodeError as e:
        raise UATParseError(f"Invalid JSON: {e}", source_path=str(path)) from e
    except FileNotFoundError as e:
        raise UATParseError(f"File not found: {path}", source_path=str(path)) from e


def _load_from_tarball(path: Path) -> list[dict[str, Any]]:
    """Load UAT JSON from a tar.gz archive.

    Looks for UAT_list.json or UAT.json inside the archive.

    Args:
        path: Path to the tar.gz archive

    Returns:
        List of concept dictionaries

    Raises:
        UATParseError: If no suitable JSON file found in archive
    """
    # Files to look for, in priority order
    target_files = ["UAT_list.json", "UAT.json"]

    try:
        with tarfile.open(path, "r:gz") as tar:
            for target in target_files:
                # Look for the file at any path within the archive
                for member in tar.getmembers():
                    if member.name.endswith(target) and member.isfile():
                        f = tar.extractfile(member)
                        if f is not None:
                            data = json.load(f)
                            if isinstance(data, list):
                                return data
                            # UAT.json has nested structure with 'children'
                            if isinstance(data, dict) and "children" in data:
                                return _flatten_hierarchical_uat(data)

            # No suitable file found
            raise UATParseError(
                f"No UAT JSON file found in archive. Looked for: {target_files}",
                source_path=str(path),
            )

    except tarfile.TarError as e:
        raise UATParseError(f"Invalid tar archive: {e}", source_path=str(path)) from e


def _flatten_hierarchical_uat(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten hierarchical UAT.json format into a flat list.

    The hierarchical format has nested 'children' arrays. We need to extract
    all concepts into a flat list and preserve broader/narrower relationships.

    Args:
        data: Root UAT.json object with 'children' array

    Returns:
        Flat list of concept dictionaries with broader/narrower fields
    """
    concepts: list[dict[str, Any]] = []

    def _extract(node: dict[str, Any], parent_uri: str | None = None) -> None:
        """Recursively extract concepts from hierarchical structure."""
        # Create a copy without children for the flat list
        concept = {k: v for k, v in node.items() if k != "children"}

        # Add broader reference if we have a parent
        if parent_uri:
            # Convert altLabels to altNames for consistency
            if "altLabels" in concept and "altNames" not in concept:
                concept["altNames"] = concept.pop("altLabels")
            concept["broader"] = [{"uri": parent_uri, "name": ""}]

        # Add narrower references from children
        children = node.get("children") or []
        if children:
            concept["narrower"] = [
                {"uri": child.get("uri", ""), "name": child.get("name", "")}
                for child in children
                if child.get("uri")
            ]

        if concept.get("uri") and concept.get("name"):
            concepts.append(concept)

        # Recurse into children
        for child in children:
            _extract(child, node.get("uri"))

    # Process all top-level children
    for child in data.get("children", []):
        _extract(child)

    return concepts


def normalize_uat(
    source_path: Path,
    source_id: str | None = "uat",
) -> Iterator[TopicEntry]:
    """Normalize UAT vocabulary into TopicEntry records.

    Args:
        source_path: Path to UAT JSON file or tar.gz archive
        source_id: Source identifier for provenance tracking

    Yields:
        TopicEntry records for each valid concept
    """
    concepts = load_uat_json(source_path)

    for concept in concepts:
        entry = parse_uat_concept(concept, source_id=source_id)
        if entry is not None:
            yield entry


def normalize_uat_to_catalog(
    source_path: Path,
    output_path: Path,
    source_id: str | None = "uat",
) -> tuple[int, str]:
    """Normalize UAT vocabulary and write to topic_catalog.jsonl.

    Args:
        source_path: Path to UAT JSON file or tar.gz archive
        output_path: Path for output JSONL file

    Returns:
        Tuple of (entry count, SHA256 checksum)
    """
    from finetune.dataset_agent.writers import JSONLWriter

    writer = JSONLWriter(output_path)
    count = 0

    with writer:
        for entry in normalize_uat(source_path, source_id=source_id):
            writer.write_line(entry)
            count += 1

    return count, writer.checksum
