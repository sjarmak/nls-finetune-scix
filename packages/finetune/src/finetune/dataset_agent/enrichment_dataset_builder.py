"""Enrichment dataset builder for combining multiple label sources.

Combines enrichment labels from:
1. Synthetic snippet generation (snippet_generator.py output)
2. Existing NL-query pair enrichment labels (enrichment_generator.py output)

Produces a unified enrichment_labels.jsonl dataset with train/val/test splits
and a coverage report.

Output schema per record:
    {
        "id": str,
        "text": str,
        "text_type": "title" | "abstract" | "snippet",
        "spans": [
            {
                "surface": str,
                "start": int,
                "end": int,
                "type": str,
                "canonical_id": str,
                "source_vocabulary": str,
                "confidence": float,
            }
        ],
        "topics": [
            {
                "concept_id": str,
                "label": str,
                "source_vocabulary": str,
                "confidence": float,
            }
        ],
        "provenance": dict,
    }
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune.dataset_agent.writers import JSONLReader, JSONLWriter, JSONWriter

# ---------------------------------------------------------------------------
# Unified enrichment record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentSpan:
    """A single span annotation in the unified enrichment dataset."""

    surface: str
    start: int
    end: int
    type: str
    canonical_id: str
    source_vocabulary: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "canonical_id": self.canonical_id,
            "source_vocabulary": self.source_vocabulary,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class EnrichmentTopic:
    """A topic-level label in the unified enrichment dataset."""

    concept_id: str
    label: str
    source_vocabulary: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "source_vocabulary": self.source_vocabulary,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class EnrichmentRecord:
    """A single record in the unified enrichment dataset."""

    id: str
    text: str
    text_type: str  # "title", "abstract", or "snippet"
    spans: list[EnrichmentSpan]
    topics: list[EnrichmentTopic]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "text_type": self.text_type,
            "spans": [s.to_dict() for s in self.spans],
            "topics": [t.to_dict() for t in self.topics],
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetBuilderConfig:
    """Configuration for the enrichment dataset builder.

    Attributes:
        min_examples: Minimum number of examples to produce.
        snippet_multiplier: How many times to repeat snippet generation
            if the initial count is below min_examples.
        train_fraction: Fraction of data for training split.
        val_fraction: Fraction of data for validation split.
        test_fraction: Fraction of data for test split.
        seed: Random seed for reproducible shuffling and splitting.
        min_per_vocabulary: Minimum examples per source vocabulary for
            balanced representation.
    """

    min_examples: int = 10000
    snippet_multiplier: int = 5
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    seed: int = 42
    min_per_vocabulary: int = 500


@dataclass
class DatasetBuilderStats:
    """Statistics from dataset building."""

    snippets_loaded: int = 0
    pair_labels_loaded: int = 0
    total_records: int = 0
    records_by_text_type: dict[str, int] = field(default_factory=dict)
    records_by_source_vocabulary: dict[str, int] = field(default_factory=dict)
    records_by_label_type: dict[str, int] = field(default_factory=dict)
    records_by_domain: dict[str, int] = field(default_factory=dict)
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snippets_loaded": self.snippets_loaded,
            "pair_labels_loaded": self.pair_labels_loaded,
            "total_records": self.total_records,
            "records_by_text_type": dict(self.records_by_text_type),
            "records_by_source_vocabulary": dict(self.records_by_source_vocabulary),
            "records_by_label_type": dict(self.records_by_label_type),
            "records_by_domain": dict(self.records_by_domain),
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
        }


# ---------------------------------------------------------------------------
# Record ID generation
# ---------------------------------------------------------------------------


def _record_id(source: str, index: int) -> str:
    """Generate a deterministic record ID."""
    raw = f"enr_{source}_{index}"
    hash_val = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"enr_{source[:3]}_{hash_val}"


# ---------------------------------------------------------------------------
# Conversion: snippets -> EnrichmentRecord
# ---------------------------------------------------------------------------


def convert_snippet(snippet_dict: dict[str, Any], index: int) -> EnrichmentRecord:
    """Convert a snippet generator output dict to an EnrichmentRecord.

    Snippets have spans with {surface, start, end, type, canonical_id, source_vocabulary}.
    We map these to EnrichmentSpan (adding confidence=1.0) and also extract
    topic-level labels from topic-type spans.
    """
    spans = [
        EnrichmentSpan(
            surface=s["surface"],
            start=s["start"],
            end=s["end"],
            type=s["type"],
            canonical_id=s["canonical_id"],
            source_vocabulary=s["source_vocabulary"],
            confidence=1.0,
        )
        for s in snippet_dict.get("spans", [])
    ]

    # Extract topic-level labels from spans of type "topic"
    topics = [
        EnrichmentTopic(
            concept_id=s["canonical_id"],
            label=s["surface"],
            source_vocabulary=s["source_vocabulary"],
            confidence=1.0,
        )
        for s in snippet_dict.get("spans", [])
        if s.get("type") == "topic"
    ]

    text_type = snippet_dict.get("text_type", "snippet")

    provenance = {
        "source": "snippet_generator",
        "original_id": snippet_dict.get("id", ""),
        **{k: v for k, v in snippet_dict.get("provenance", {}).items()},
    }

    return EnrichmentRecord(
        id=_record_id("snp", index),
        text=snippet_dict.get("text", ""),
        text_type=text_type,
        spans=spans,
        topics=topics,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Conversion: pair-based enrichment labels -> EnrichmentRecord
# ---------------------------------------------------------------------------


def convert_pair_label(label_dict: dict[str, Any], index: int) -> EnrichmentRecord:
    """Convert a pair-based enrichment label dict to an EnrichmentRecord.

    Pair labels have {example_id, user_text, labels: [{entity_id, entity_type,
    text_span, start_char, end_char}], provenance}.
    """
    spans = []
    topics = []

    for lbl in label_dict.get("labels", []):
        entity_id = lbl.get("entity_id", "")
        entity_type = lbl.get("entity_type", "")
        text_span = lbl.get("text_span", "")
        start_char = lbl.get("start_char")
        end_char = lbl.get("end_char")
        source_vocab = _infer_vocabulary_from_id(entity_id)

        if start_char is not None and end_char is not None:
            spans.append(
                EnrichmentSpan(
                    surface=text_span or "",
                    start=start_char,
                    end=end_char,
                    type=entity_type,
                    canonical_id=entity_id,
                    source_vocabulary=source_vocab,
                    confidence=0.9,
                )
            )

        if entity_type == "topic":
            topics.append(
                EnrichmentTopic(
                    concept_id=entity_id,
                    label=text_span or "",
                    source_vocabulary=source_vocab,
                    confidence=0.9,
                )
            )

    provenance = {
        "source": "pair_enrichment",
        "original_id": label_dict.get("example_id", ""),
        **{k: v for k, v in label_dict.get("provenance", {}).items()},
    }

    return EnrichmentRecord(
        id=_record_id("pair", index),
        text=label_dict.get("user_text", ""),
        text_type="snippet",
        spans=spans,
        topics=topics,
        provenance=provenance,
    )


def _infer_vocabulary_from_id(entity_id: str) -> str:
    """Infer source vocabulary from entity ID prefix.

    Entity IDs follow the pattern "prefix:rest" where prefix indicates
    the vocabulary (e.g., "uat:123", "ror:abc", "sweet:xyz").
    """
    if ":" in entity_id:
        prefix = entity_id.split(":")[0].lower()
        known = {"uat", "ror", "sweet", "gcmd", "planetary"}
        if prefix in known:
            return prefix
    return ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_snippets(path: Path) -> list[dict[str, Any]]:
    """Load snippet records from a JSONL file."""
    if not path.exists():
        return []
    return JSONLReader(path).read_all()


def load_pair_labels(path: Path) -> list[dict[str, Any]]:
    """Load pair-based enrichment labels from a JSONL file."""
    if not path.exists():
        return []
    return JSONLReader(path).read_all()


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_dataset(
    records: list[EnrichmentRecord],
    config: DatasetBuilderConfig,
) -> tuple[list[EnrichmentRecord], list[EnrichmentRecord], list[EnrichmentRecord]]:
    """Split records into train/val/test sets.

    Shuffles deterministically based on config.seed, then splits by fraction.

    Returns:
        (train, val, test) lists of EnrichmentRecord
    """
    rng = random.Random(config.seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * config.train_fraction)
    val_end = train_end + int(n * config.val_fraction)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


def compute_coverage(
    records: list[EnrichmentRecord],
    stats: DatasetBuilderStats,
) -> dict[str, Any]:
    """Compute a coverage report for the enrichment dataset.

    Returns a dict suitable for writing to enrichment_coverage.json.
    """
    label_type_counts: dict[str, int] = {}
    vocabulary_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    text_type_counts: dict[str, int] = {}

    for record in records:
        text_type_counts[record.text_type] = (
            text_type_counts.get(record.text_type, 0) + 1
        )

        for span in record.spans:
            label_type_counts[span.type] = label_type_counts.get(span.type, 0) + 1
            if span.source_vocabulary:
                vocabulary_counts[span.source_vocabulary] = (
                    vocabulary_counts.get(span.source_vocabulary, 0) + 1
                )

        for topic in record.topics:
            if topic.source_vocabulary:
                vocabulary_counts[topic.source_vocabulary] = (
                    vocabulary_counts.get(topic.source_vocabulary, 0) + 1
                )

        # Infer domain from provenance or span vocabularies
        domains = _infer_domains(record)
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    return {
        "total_records": len(records),
        "by_label_type": label_type_counts,
        "by_source_vocabulary": vocabulary_counts,
        "by_domain": domain_counts,
        "by_text_type": text_type_counts,
        "stats": stats.to_dict(),
    }


def _infer_domains(record: EnrichmentRecord) -> list[str]:
    """Infer domain tags for a record from its spans and provenance."""
    domains: set[str] = set()
    vocab_to_domain = {
        "uat": "astronomy",
        "sweet": "earthscience",
        "gcmd": "earthscience",
        "planetary": "planetary",
        "ror": "multidisciplinary",
    }
    for span in record.spans:
        if span.source_vocabulary in vocab_to_domain:
            domains.add(vocab_to_domain[span.source_vocabulary])
    for topic in record.topics:
        if topic.source_vocabulary in vocab_to_domain:
            domains.add(vocab_to_domain[topic.source_vocabulary])
    return sorted(domains) if domains else ["unknown"]


# ---------------------------------------------------------------------------
# Update stats
# ---------------------------------------------------------------------------


def _update_stats(stats: DatasetBuilderStats, records: list[EnrichmentRecord]) -> None:
    """Update stats counters from a list of records."""
    stats.total_records = len(records)
    for record in records:
        stats.records_by_text_type[record.text_type] = (
            stats.records_by_text_type.get(record.text_type, 0) + 1
        )
        for span in record.spans:
            stats.records_by_label_type[span.type] = (
                stats.records_by_label_type.get(span.type, 0) + 1
            )
            if span.source_vocabulary:
                stats.records_by_source_vocabulary[span.source_vocabulary] = (
                    stats.records_by_source_vocabulary.get(span.source_vocabulary, 0)
                    + 1
                )
        domains = _infer_domains(record)
        for domain in domains:
            stats.records_by_domain[domain] = (
                stats.records_by_domain.get(domain, 0) + 1
            )


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_enrichment_dataset(
    snippet_dicts: list[dict[str, Any]],
    pair_label_dicts: list[dict[str, Any]],
    config: DatasetBuilderConfig | None = None,
) -> tuple[list[EnrichmentRecord], DatasetBuilderStats]:
    """Build the unified enrichment dataset from both label sources.

    Args:
        snippet_dicts: Raw snippet dicts from snippet_generator output.
        pair_label_dicts: Raw enrichment label dicts from enrichment_generator.
        config: Builder configuration.

    Returns:
        (list of EnrichmentRecord, DatasetBuilderStats)
    """
    config = config or DatasetBuilderConfig()
    stats = DatasetBuilderStats()

    records: list[EnrichmentRecord] = []

    # Convert snippets
    for i, snippet in enumerate(snippet_dicts):
        record = convert_snippet(snippet, i)
        if record.spans or record.topics:
            records.append(record)
    stats.snippets_loaded = len(snippet_dicts)

    # Convert pair-based labels
    for i, label in enumerate(pair_label_dicts):
        record = convert_pair_label(label, i)
        if record.spans or record.topics:
            records.append(record)
    stats.pair_labels_loaded = len(pair_label_dicts)

    # Deduplicate by text content
    records = _deduplicate(records)

    _update_stats(stats, records)

    return records, stats


def _deduplicate(records: list[EnrichmentRecord]) -> list[EnrichmentRecord]:
    """Remove duplicate records based on text content hash."""
    seen: set[str] = set()
    unique: list[EnrichmentRecord] = []
    for record in records:
        key = hashlib.md5(record.text.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


# ---------------------------------------------------------------------------
# File-based API
# ---------------------------------------------------------------------------


def build_enrichment_dataset_from_files(
    snippets_path: Path,
    pair_labels_path: Path,
    output_dir: Path,
    config: DatasetBuilderConfig | None = None,
) -> tuple[int, DatasetBuilderStats]:
    """Build the enrichment dataset from files and write splits.

    Args:
        snippets_path: Path to snippets JSONL (from snippet_generator).
        pair_labels_path: Path to enrichment_labels.jsonl (from enrichment_generator).
        output_dir: Directory to write output files.
        config: Builder configuration.

    Returns:
        (total_record_count, stats)
    """
    cfg = config or DatasetBuilderConfig()

    snippet_dicts = load_snippets(snippets_path)
    pair_label_dicts = load_pair_labels(pair_labels_path)

    records, stats = build_enrichment_dataset(snippet_dicts, pair_label_dicts, cfg)

    # Split
    train, val, test = split_dataset(records, cfg)
    stats.train_count = len(train)
    stats.val_count = len(val)
    stats.test_count = len(test)

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_records(output_dir / "enrichment_labels.jsonl", records)
    _write_records(output_dir / "enrichment_train.jsonl", train)
    _write_records(output_dir / "enrichment_val.jsonl", val)
    _write_records(output_dir / "enrichment_test.jsonl", test)

    # Write coverage report
    coverage = compute_coverage(records, stats)
    coverage_path = output_dir / "enrichment_coverage.json"
    json_writer = JSONWriter(coverage_path)
    json_writer.write(coverage)

    return len(records), stats


def _write_records(path: Path, records: list[EnrichmentRecord]) -> tuple[str, int]:
    """Write enrichment records to a JSONL file."""
    with JSONLWriter(path) as writer:
        for record in records:
            writer.write_line(record)
    return writer.checksum, writer.line_count
