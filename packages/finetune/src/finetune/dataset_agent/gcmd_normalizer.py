"""GCMD (Global Change Master Directory) Science Keywords normalizer.

This module parses NASA GCMD Science Keywords (JSON format from the
adiwg/gcmd-keywords GitHub mirror) and produces normalized TopicEntry
records for the topic_catalog_gcmd.jsonl output.

GCMD source: https://github.com/adiwg/gcmd-keywords
Format: JSON tree with 5 hierarchy levels:
  Category > Topic > Term > Variable > Detailed Variable

Each node has:
  - uuid: Unique identifier
  - label: Keyword name
  - broader: Parent UUID (or null at root)
  - definition: Descriptive text (Earth Science keywords only)
  - children: Array of child nodes (empty array at leaves)

License: CC0 (public domain)
"""

from __future__ import annotations

import json
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from finetune.dataset_agent.schemas import TopicEntry

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# GCMD hierarchy level names (0-indexed from the root's children)
HIERARCHY_LEVELS = ("Category", "Topic", "Term", "Variable", "Detailed Variable")


class GCMDParseError(Exception):
    """Error parsing GCMD keyword data."""

    def __init__(self, message: str, source_path: str | None = None):
        self.source_path = source_path
        super().__init__(message)


def _build_concept_id(path_parts: list[str]) -> str:
    """Build a concept ID from the hierarchy path.

    Converts a hierarchy path like ["EARTH SCIENCE", "ATMOSPHERE", "AEROSOLS"]
    to "gcmd:EARTH_SCIENCE/ATMOSPHERE/AEROSOLS".

    Args:
        path_parts: List of label strings from root to current concept

    Returns:
        Concept ID (e.g., 'gcmd:EARTH_SCIENCE/ATMOSPHERE/AEROSOLS')
    """
    segments = [part.replace(" ", "_") for part in path_parts]
    return f"gcmd:{'/'.join(segments)}"


def _build_parent_id(path_parts: list[str]) -> str | None:
    """Build the parent concept ID from the hierarchy path.

    Args:
        path_parts: List of label strings from root to current concept

    Returns:
        Parent concept ID, or None if at root level
    """
    if len(path_parts) <= 1:
        return None
    return _build_concept_id(path_parts[:-1])


def _extract_child_ids(node: dict, current_path: list[str]) -> list[str]:
    """Extract child concept IDs from a GCMD node.

    Args:
        node: GCMD keyword node with optional 'children' array
        current_path: Hierarchy path to the current node

    Returns:
        List of child concept IDs
    """
    children = node.get("children", [])
    if not children:
        return []
    return [
        _build_concept_id([*current_path, child["label"]])
        for child in children
        if "label" in child and child["label"]
    ]


def parse_gcmd_node(
    node: dict,
    path_parts: list[str],
    source_id: str | None = None,
) -> TopicEntry | None:
    """Parse a single GCMD keyword node into a TopicEntry.

    Args:
        node: GCMD keyword node dict with uuid, label, broader, children, definition
        path_parts: Hierarchy path from root to this node (inclusive)
        source_id: Optional source identifier for provenance

    Returns:
        TopicEntry or None if node is invalid
    """
    label = node.get("label", "").strip()
    if not label:
        return None

    concept_id = _build_concept_id(path_parts)
    parent_id = _build_parent_id(path_parts)
    child_ids = _extract_child_ids(node, path_parts)

    parents = [parent_id] if parent_id else []

    return TopicEntry(
        id=concept_id,
        label=label,
        aliases=[],
        parents=parents,
        children=child_ids,
        source_id=source_id,
        domain_tags=["earthscience"],
        source_vocabulary="gcmd",
    )


def _walk_gcmd_tree(
    node: dict,
    path_parts: list[str],
    source_id: str | None = None,
) -> Iterator[tuple[dict, list[str]]]:
    """Walk the GCMD keyword tree depth-first, yielding (node, path) pairs.

    Skips the virtual root node (which has broader=null and no uuid).

    Args:
        node: Current GCMD keyword node
        path_parts: Hierarchy path to this node (inclusive)
        source_id: Source identifier (unused, passed for API consistency)

    Yields:
        Tuples of (node_dict, path_parts) for each keyword in the tree
    """
    label = node.get("label", "").strip()
    is_root = node.get("broader") is None and "uuid" not in node

    if label and not is_root:
        yield node, path_parts

    for child in node.get("children", []):
        child_label = child.get("label", "").strip()
        if not child_label:
            continue
        child_path = [*path_parts, child_label]
        yield from _walk_gcmd_tree(child, child_path, source_id=source_id)


def load_gcmd_data(path: Path) -> dict:
    """Load GCMD keyword data from a file, zip, or tar.gz archive.

    Supports:
    - Direct JSON file (sciencekeywords.json)
    - Directory containing the JSON file
    - zip archive containing the JSON file
    - tar.gz archive containing the JSON file

    Args:
        path: Path to a JSON file, directory, zip, or tar.gz archive

    Returns:
        Parsed GCMD keyword tree (root node dict)

    Raises:
        GCMDParseError: If the file cannot be parsed
    """
    try:
        if path.suffix == ".json":
            return _load_json_file(path)
        elif path.is_dir():
            return _load_from_directory(path)
        elif path.name.endswith(".tar.gz") or path.suffix == ".gz":
            return _load_from_tarball(path)
        elif path.suffix == ".zip":
            return _load_from_zip(path)
        else:
            # Try loading as JSON
            return _load_json_file(path)
    except GCMDParseError:
        raise
    except Exception as e:
        raise GCMDParseError(
            f"Failed to parse GCMD data: {e}", source_path=str(path)
        ) from e


def _load_json_file(path: Path) -> dict:
    """Load GCMD JSON from a file.

    The file can contain either a single root object or an array
    containing a single root object.

    Args:
        path: Path to the JSON file

    Returns:
        Root node dict

    Raises:
        GCMDParseError: If file is invalid
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise GCMDParseError(
            f"Cannot read file: {e}", source_path=str(path)
        ) from e

    return _parse_gcmd_json(text, str(path))


def _parse_gcmd_json(text: str, source_label: str) -> dict:
    """Parse GCMD JSON text into a root node dict.

    Handles both single-object and array-wrapped formats.

    Args:
        text: JSON text
        source_label: Label for error messages

    Returns:
        Root node dict

    Raises:
        GCMDParseError: If JSON is invalid or structure is unexpected
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise GCMDParseError(
            f"Invalid JSON in {source_label}: {e}", source_path=source_label
        ) from e

    if isinstance(data, list):
        if len(data) == 0:
            raise GCMDParseError(
                f"Empty array in {source_label}", source_path=source_label
            )
        data = data[0]

    if not isinstance(data, dict):
        raise GCMDParseError(
            f"Unexpected JSON structure in {source_label}: expected object, got {type(data).__name__}",
            source_path=source_label,
        )

    return data


def _load_from_directory(path: Path) -> dict:
    """Load GCMD data from a directory.

    Searches for sciencekeywords.json in common locations.

    Args:
        path: Directory path

    Returns:
        Root node dict

    Raises:
        GCMDParseError: If no suitable file found
    """
    candidates = [
        path / "sciencekeywords.json",
        path / "resources" / "json" / "sciencekeywords.json",
    ]

    # Also search recursively for the file
    for json_file in sorted(path.rglob("sciencekeywords.json")):
        if json_file not in candidates:
            candidates.append(json_file)

    for candidate in candidates:
        if candidate.exists():
            return _load_json_file(candidate)

    raise GCMDParseError(
        "No sciencekeywords.json found in directory", source_path=str(path)
    )


def _load_from_tarball(path: Path) -> dict:
    """Load GCMD data from a tar.gz archive.

    Args:
        path: Path to the tar.gz archive

    Returns:
        Root node dict

    Raises:
        GCMDParseError: If no suitable file found in archive
    """
    try:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("sciencekeywords.json") and member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        text = f.read().decode("utf-8")
                        return _parse_gcmd_json(text, member.name)
    except tarfile.TarError as e:
        raise GCMDParseError(
            f"Invalid tar archive: {e}", source_path=str(path)
        ) from e

    raise GCMDParseError(
        "No sciencekeywords.json found in archive", source_path=str(path)
    )


def _load_from_zip(path: Path) -> dict:
    """Load GCMD data from a zip archive.

    Args:
        path: Path to the zip archive

    Returns:
        Root node dict

    Raises:
        GCMDParseError: If no suitable file found in archive
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("sciencekeywords.json"):
                    text = zf.read(name).decode("utf-8")
                    return _parse_gcmd_json(text, name)
    except zipfile.BadZipFile as e:
        raise GCMDParseError(
            f"Invalid zip archive: {e}", source_path=str(path)
        ) from e

    raise GCMDParseError(
        "No sciencekeywords.json found in zip", source_path=str(path)
    )


def normalize_gcmd(
    source_path: Path,
    source_id: str | None = "gcmd",
) -> Iterator[TopicEntry]:
    """Normalize GCMD Science Keywords into TopicEntry records.

    Args:
        source_path: Path to GCMD data (JSON file, directory, or archive)
        source_id: Source identifier for provenance tracking

    Yields:
        TopicEntry records for each valid keyword
    """
    root = load_gcmd_data(source_path)

    seen_ids: set[str] = set()
    for node, path_parts in _walk_gcmd_tree(root, [], source_id=source_id):
        entry = parse_gcmd_node(node, path_parts, source_id=source_id)
        if entry is None:
            continue
        if entry.id in seen_ids:
            continue
        seen_ids.add(entry.id)
        yield entry


def normalize_gcmd_to_catalog(
    source_path: Path,
    output_path: Path,
    source_id: str | None = "gcmd",
) -> tuple[int, str]:
    """Normalize GCMD keywords and write to topic_catalog_gcmd.jsonl.

    Args:
        source_path: Path to GCMD data (JSON file, directory, or archive)
        output_path: Path for output JSONL file
        source_id: Source identifier for provenance

    Returns:
        Tuple of (entry count, SHA256 checksum)
    """
    from finetune.dataset_agent.writers import JSONLWriter

    writer = JSONLWriter(output_path)
    count = 0

    with writer:
        for entry in normalize_gcmd(source_path, source_id=source_id):
            writer.write_line(entry)
            count += 1

    logger.info(f"GCMD normalization complete: {count} keywords written to {output_path}")
    return count, writer.checksum
