"""Entity linker: map NER-extracted spans to catalog canonical IDs.

Provides exact and fuzzy string matching against topic and entity catalogs.
The linking cascade tries exact match first, then fuzzy (Levenshtein) match.

Usage:
    index = build_linking_index(topic_catalog_path, entity_catalog_path)
    result = link_span("Hubble Space Telescope", "entity", index)
    # result = LinkResult(canonical_id="ror:...", source_vocabulary="ror", confidence=1.0, match_type="exact")

    # Serialize to disk for reuse
    save_linking_index(index, Path("linking_index.json"))
    index2 = load_linking_index(Path("linking_index.json"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from finetune.dataset_agent.schemas import EntityEntry, TopicEntry
from finetune.dataset_agent.writers import JSONLReader

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkResult:
    """Result of linking a surface span to a catalog entry."""

    canonical_id: str
    source_vocabulary: str
    confidence: float
    match_type: str  # "exact" or "fuzzy"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass(frozen=True)
class _IndexEntry:
    """Internal: one row in the linking lookup table."""

    canonical_id: str
    source_vocabulary: str
    span_type: str  # "topic" or "entity"


@dataclass
class LinkingIndex:
    """Pre-built lookup structure for entity linking.

    Contains two parallel structures:
    - exact_map: lowercased surface -> list[_IndexEntry]  (for O(1) exact match)
    - entries: list of (surface_lower, _IndexEntry)        (for fuzzy scan)
    """

    exact_map: dict[str, list[dict[str, str]]]
    entries: list[tuple[str, dict[str, str]]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "exact_map": self.exact_map,
            "entries": self.entries,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LinkingIndex:
        """Deserialize from JSON-safe dict."""
        return cls(
            exact_map=d["exact_map"],
            entries=[(e[0], e[1]) for e in d["entries"]],
        )


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _surface_forms(label: str, aliases: list[str]) -> list[str]:
    """Return all surface forms for a catalog entry (label + aliases)."""
    forms = [label]
    for alias in aliases:
        if alias and alias != label:
            forms.append(alias)
    return forms


def _add_entry_to_index(
    exact_map: dict[str, list[dict[str, str]]],
    entries: list[tuple[str, dict[str, str]]],
    canonical_id: str,
    source_vocabulary: str,
    span_type: str,
    label: str,
    aliases: list[str],
) -> None:
    """Add one catalog entry (all its surface forms) to the index structures."""
    entry_dict = {
        "canonical_id": canonical_id,
        "source_vocabulary": source_vocabulary,
        "span_type": span_type,
    }
    for surface in _surface_forms(label, aliases):
        key = surface.lower()
        if key not in exact_map:
            exact_map[key] = []
        exact_map[key].append(entry_dict)
        entries.append((key, entry_dict))


def build_linking_index(
    topic_catalog_path: Path | str,
    entity_catalog_path: Path | str,
) -> LinkingIndex:
    """Build a linking index from topic and entity catalog JSONL files.

    Args:
        topic_catalog_path: Path to unified topic_catalog.jsonl
        entity_catalog_path: Path to unified entity_catalog.jsonl

    Returns:
        LinkingIndex ready for exact and fuzzy lookups.
    """
    exact_map: dict[str, list[dict[str, str]]] = {}
    entries: list[tuple[str, dict[str, str]]] = []

    topic_path = Path(topic_catalog_path)
    if topic_path.exists():
        for d in JSONLReader(topic_path):
            entry = TopicEntry.from_dict(d)
            _add_entry_to_index(
                exact_map, entries,
                canonical_id=entry.id,
                source_vocabulary=entry.source_vocabulary,
                span_type="topic",
                label=entry.label,
                aliases=entry.aliases,
            )

    entity_path = Path(entity_catalog_path)
    if entity_path.exists():
        for d in JSONLReader(entity_path):
            entry = EntityEntry.from_dict(d)
            _add_entry_to_index(
                exact_map, entries,
                canonical_id=entry.id,
                source_vocabulary=entry.source_vocabulary,
                span_type="entity",
                label=entry.label,
                aliases=entry.aliases,
            )

    return LinkingIndex(exact_map=exact_map, entries=entries)


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

_FUZZY_THRESHOLD = 0.85
_FUZZY_CONFIDENCE = 0.8


def link_span(
    surface: str,
    span_type: str,
    index: LinkingIndex,
    fuzzy_threshold: float = _FUZZY_THRESHOLD,
) -> LinkResult | None:
    """Link a surface span to the best matching catalog entry.

    Cascade:
    1. Exact match (case-insensitive) — confidence=1.0
    2. Fuzzy match (Levenshtein normalized similarity >= threshold) — confidence=0.8

    Type filtering: topics only match topic catalog entries, entities only
    match entity catalog entries.

    Args:
        surface: The text span to link.
        span_type: "topic" or "entity".
        index: Pre-built LinkingIndex.
        fuzzy_threshold: Minimum normalized similarity for fuzzy match (default 0.85).

    Returns:
        LinkResult if a match is found, None otherwise.
    """
    key = surface.lower()

    # --- Stage 1: Exact match ---
    candidates = index.exact_map.get(key, [])
    for cand in candidates:
        if cand["span_type"] == span_type:
            return LinkResult(
                canonical_id=cand["canonical_id"],
                source_vocabulary=cand["source_vocabulary"],
                confidence=1.0,
                match_type="exact",
            )

    # --- Stage 2: Fuzzy match ---
    best_score = 0.0
    best_entry: dict[str, str] | None = None

    for entry_surface, entry_dict in index.entries:
        if entry_dict["span_type"] != span_type:
            continue
        score = fuzz.ratio(key, entry_surface) / 100.0
        if score >= fuzzy_threshold and score > best_score:
            best_score = score
            best_entry = entry_dict

    if best_entry is not None:
        return LinkResult(
            canonical_id=best_entry["canonical_id"],
            source_vocabulary=best_entry["source_vocabulary"],
            confidence=_FUZZY_CONFIDENCE,
            match_type="fuzzy",
        )

    return None


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def save_linking_index(index: LinkingIndex, path: Path | str) -> None:
    """Serialize a LinkingIndex to a JSON file on disk.

    Args:
        index: The index to persist.
        path: Output file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index.to_dict(), f, ensure_ascii=False)


def load_linking_index(path: Path | str) -> LinkingIndex:
    """Load a LinkingIndex from a JSON file on disk.

    Args:
        path: Path to previously saved index.

    Returns:
        Deserialized LinkingIndex.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    with open(Path(path), encoding="utf-8") as f:
        return LinkingIndex.from_dict(json.load(f))
