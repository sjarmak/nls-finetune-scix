"""Entity linker: map NER-extracted spans to catalog canonical IDs.

Provides a three-stage linking cascade:
1. Exact match (case-insensitive) — confidence=1.0
2. Fuzzy match (Levenshtein, normalized similarity >= 0.85) — confidence=0.8
3. Embedding match (cosine similarity >= 0.75) — confidence=cosine_sim

Usage:
    index = build_linking_index(topic_catalog_path, entity_catalog_path)
    result = link_span("Hubble Space Telescope", "entity", index)
    # result = LinkResult(canonical_id="ror:...", source_vocabulary="ror", confidence=1.0, match_type="exact")

    # Serialize to disk for reuse
    save_linking_index(index, Path("linking_index.json"))
    index2 = load_linking_index(Path("linking_index.json"))

    # Embedding-based fallback
    emb_index = build_embedding_index(catalog_entries, model_name="all-MiniLM-L6-v2")
    result = link_span_embedding("HST observatory", "entity", emb_index, threshold=0.75)

    # Full cascade
    result = link_span_cascade("HST observatory", "entity", index, emb_index)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
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


# ---------------------------------------------------------------------------
# Embedding-based linking
# ---------------------------------------------------------------------------

_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EMBEDDING_THRESHOLD = 0.75


@dataclass(frozen=True)
class _EmbeddingEntry:
    """Metadata for one entry in the embedding index."""

    canonical_id: str
    source_vocabulary: str
    span_type: str  # "topic" or "entity"
    surface: str  # original surface form (for debugging)


@dataclass
class EmbeddingIndex:
    """Pre-built embedding index for similarity-based entity linking.

    Contains:
    - embeddings: numpy array of shape (N, dim) with L2-normalized vectors
    - entries: list of _EmbeddingEntry metadata aligned with embedding rows
    - model_name: the sentence transformer model used to generate embeddings
    """

    embeddings: np.ndarray  # shape (N, dim), float32
    entries: list[dict[str, str]]
    model_name: str

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize metadata (without embeddings) to a JSON-safe dict."""
        return {
            "entries": self.entries,
            "model_name": self.model_name,
        }

    @classmethod
    def from_parts(
        cls,
        embeddings: np.ndarray,
        metadata: dict[str, Any],
    ) -> EmbeddingIndex:
        """Reconstruct from numpy array + metadata dict."""
        return cls(
            embeddings=embeddings,
            entries=metadata["entries"],
            model_name=metadata["model_name"],
        )


@dataclass(frozen=True)
class CatalogEntry:
    """Lightweight representation of a catalog entry for embedding index building."""

    canonical_id: str
    source_vocabulary: str
    span_type: str  # "topic" or "entity"
    label: str
    aliases: list[str]


def _load_sentence_transformer(model_name: str) -> Any:
    """Lazily import and load a SentenceTransformer model.

    The import is deferred so the module can be used without
    sentence-transformers installed (for exact/fuzzy-only usage).
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def build_embedding_index(
    catalog_entries: list[CatalogEntry],
    model_name: str = _DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingIndex:
    """Build an embedding index from catalog entries.

    Encodes all surface forms (label + aliases) of each catalog entry using
    a sentence transformer model. The resulting embeddings are L2-normalized
    for cosine similarity via dot product.

    Args:
        catalog_entries: List of CatalogEntry objects to index.
        model_name: Sentence transformer model name (default: all-MiniLM-L6-v2).

    Returns:
        EmbeddingIndex ready for similarity-based lookups.
    """
    model = _load_sentence_transformer(model_name)

    surfaces: list[str] = []
    entry_metadata: list[dict[str, str]] = []

    for ce in catalog_entries:
        for surface in _surface_forms(ce.label, ce.aliases):
            surfaces.append(surface)
            entry_metadata.append({
                "canonical_id": ce.canonical_id,
                "source_vocabulary": ce.source_vocabulary,
                "span_type": ce.span_type,
                "surface": surface,
            })

    if not surfaces:
        return EmbeddingIndex(
            embeddings=np.zeros((0, 384), dtype=np.float32),
            entries=[],
            model_name=model_name,
        )

    embeddings = model.encode(surfaces, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    return EmbeddingIndex(
        embeddings=embeddings,
        entries=entry_metadata,
        model_name=model_name,
    )


def build_embedding_index_from_catalogs(
    topic_catalog_path: Path | str,
    entity_catalog_path: Path | str,
    model_name: str = _DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingIndex:
    """Build an embedding index directly from JSONL catalog files.

    Convenience wrapper that reads JSONL catalogs and delegates to
    build_embedding_index.

    Args:
        topic_catalog_path: Path to unified topic_catalog.jsonl.
        entity_catalog_path: Path to unified entity_catalog.jsonl.
        model_name: Sentence transformer model name.

    Returns:
        EmbeddingIndex ready for similarity-based lookups.
    """
    catalog_entries: list[CatalogEntry] = []

    topic_path = Path(topic_catalog_path)
    if topic_path.exists():
        for d in JSONLReader(topic_path):
            entry = TopicEntry.from_dict(d)
            catalog_entries.append(CatalogEntry(
                canonical_id=entry.id,
                source_vocabulary=entry.source_vocabulary,
                span_type="topic",
                label=entry.label,
                aliases=entry.aliases,
            ))

    entity_path = Path(entity_catalog_path)
    if entity_path.exists():
        for d in JSONLReader(entity_path):
            entry = EntityEntry.from_dict(d)
            catalog_entries.append(CatalogEntry(
                canonical_id=entry.id,
                source_vocabulary=entry.source_vocabulary,
                span_type="entity",
                label=entry.label,
                aliases=entry.aliases,
            ))

    return build_embedding_index(catalog_entries, model_name=model_name)


def link_span_embedding(
    surface: str,
    span_type: str,
    embedding_index: EmbeddingIndex,
    threshold: float = _EMBEDDING_THRESHOLD,
) -> LinkResult | None:
    """Link a surface span using embedding cosine similarity.

    Encodes the query surface with the same model used to build the index,
    then finds the most similar catalog entry above the threshold.

    Type filtering: topics only match topic entries, entities only match
    entity entries.

    Args:
        surface: The text span to link.
        span_type: "topic" or "entity".
        embedding_index: Pre-built EmbeddingIndex.
        threshold: Minimum cosine similarity for a match (default 0.75).

    Returns:
        LinkResult with confidence=cosine_similarity if match found, None otherwise.
    """
    if embedding_index.embeddings.shape[0] == 0 or not surface.strip():
        return None

    model = _load_sentence_transformer(embedding_index.model_name)

    query_embedding = model.encode([surface], normalize_embeddings=True, show_progress_bar=False)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Cosine similarity via dot product (both are L2-normalized)
    similarities = query_embedding @ embedding_index.embeddings.T
    similarities = similarities.flatten()

    # Apply type filter: set non-matching types to -1
    for i, entry in enumerate(embedding_index.entries):
        if entry["span_type"] != span_type:
            similarities[i] = -1.0

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score >= threshold:
        best_entry = embedding_index.entries[best_idx]
        return LinkResult(
            canonical_id=best_entry["canonical_id"],
            source_vocabulary=best_entry["source_vocabulary"],
            confidence=round(best_score, 4),
            match_type="embedding",
        )

    return None


# ---------------------------------------------------------------------------
# Embedding index persistence
# ---------------------------------------------------------------------------

def save_embedding_index(index: EmbeddingIndex, directory: Path | str) -> None:
    """Save an EmbeddingIndex to disk as numpy array + JSON metadata.

    Creates two files:
    - <directory>/embedding_vectors.npy — the embedding matrix
    - <directory>/embedding_metadata.json — entry metadata and model name

    Args:
        index: The embedding index to persist.
        directory: Directory path to save files into.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embedding_vectors.npy", index.embeddings)

    with open(out_dir / "embedding_metadata.json", "w", encoding="utf-8") as f:
        json.dump(index.to_metadata_dict(), f, ensure_ascii=False)


def load_embedding_index(directory: Path | str) -> EmbeddingIndex:
    """Load an EmbeddingIndex from disk.

    Reads:
    - <directory>/embedding_vectors.npy
    - <directory>/embedding_metadata.json

    Args:
        directory: Directory containing saved embedding index files.

    Returns:
        Deserialized EmbeddingIndex.

    Raises:
        FileNotFoundError: If the directory or required files don't exist.
    """
    in_dir = Path(directory)
    embeddings = np.load(in_dir / "embedding_vectors.npy")

    with open(in_dir / "embedding_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return EmbeddingIndex.from_parts(embeddings, metadata)


# ---------------------------------------------------------------------------
# Full linking cascade
# ---------------------------------------------------------------------------

def link_span_cascade(
    surface: str,
    span_type: str,
    linking_index: LinkingIndex,
    embedding_index: EmbeddingIndex | None = None,
    fuzzy_threshold: float = _FUZZY_THRESHOLD,
    embedding_threshold: float = _EMBEDDING_THRESHOLD,
) -> LinkResult | None:
    """Link a surface span using the full cascade: exact -> fuzzy -> embedding.

    Tries stages in order and returns the first successful match.

    Args:
        surface: The text span to link.
        span_type: "topic" or "entity".
        linking_index: Pre-built LinkingIndex for exact/fuzzy matching.
        embedding_index: Optional EmbeddingIndex for embedding fallback.
        fuzzy_threshold: Minimum similarity for fuzzy match (default 0.85).
        embedding_threshold: Minimum cosine similarity for embedding match (default 0.75).

    Returns:
        LinkResult from the first matching stage, or None if all stages fail.
    """
    # Stage 1 + 2: Exact and fuzzy match
    result = link_span(surface, span_type, linking_index, fuzzy_threshold=fuzzy_threshold)
    if result is not None:
        return result

    # Stage 3: Embedding match (if index provided)
    if embedding_index is not None:
        return link_span_embedding(surface, span_type, embedding_index, threshold=embedding_threshold)

    return None
