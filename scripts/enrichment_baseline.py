"""Keyword-matching baseline for enrichment model evaluation.

Establishes floor metrics by matching catalog entry labels against enrichment
test set texts. This baseline uses exact substring matching (case-insensitive)
to find spans, then evaluates against ground-truth annotations.

Usage:
    python scripts/enrichment_baseline.py \\
        --test-file data/enrichment_test.jsonl \\
        --topic-catalog data/normalized/topic_catalog.jsonl \\
        --entity-catalog data/normalized/entity_catalog.jsonl \\
        --output-file reports/enrichment_baseline.json

If --test-file is not provided, generates synthetic test data to demonstrate
the evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictedSpan:
    """A span predicted by the keyword-matching baseline."""

    surface: str
    start: int
    end: int
    type: str
    canonical_id: str
    source_vocabulary: str


@dataclass(frozen=True)
class GoldSpan:
    """A ground-truth span from the enrichment test set."""

    surface: str
    start: int
    end: int
    type: str
    canonical_id: str
    source_vocabulary: str


@dataclass
class EvalMetrics:
    """Precision/recall/F1 metrics for a single evaluation slice."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class BaselineResult:
    """Full evaluation result from the keyword-matching baseline."""

    overall: EvalMetrics = field(default_factory=EvalMetrics)
    by_type: dict[str, EvalMetrics] = field(default_factory=dict)
    by_vocabulary: dict[str, EvalMetrics] = field(default_factory=dict)
    by_domain: dict[str, EvalMetrics] = field(default_factory=dict)
    total_records: int = 0
    total_gold_spans: int = 0
    total_predicted_spans: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_type": {k: v.to_dict() for k, v in sorted(self.by_type.items())},
            "by_vocabulary": {
                k: v.to_dict() for k, v in sorted(self.by_vocabulary.items())
            },
            "by_domain": {k: v.to_dict() for k, v in sorted(self.by_domain.items())},
            "total_records": self.total_records,
            "total_gold_spans": self.total_gold_spans,
            "total_predicted_spans": self.total_predicted_spans,
        }


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


@dataclass
class CatalogEntry:
    """A single catalog entry used for keyword matching."""

    entry_id: str
    label: str
    aliases: list[str]
    entry_type: str  # "topic" or "institution"
    source_vocabulary: str
    domain_tags: list[str]


def load_catalog_entries(path: Path, entry_type: str) -> list[CatalogEntry]:
    """Load catalog entries from a JSONL file.

    Args:
        path: Path to topic_catalog.jsonl or entity_catalog.jsonl
        entry_type: "topic" for topic catalogs, "institution" for entity catalogs

    Returns:
        List of CatalogEntry objects with labels and aliases for matching.
    """
    if not path.exists():
        return []

    entries: list[CatalogEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            entries.append(
                CatalogEntry(
                    entry_id=record.get("id", ""),
                    label=record.get("label", ""),
                    aliases=record.get("aliases", []),
                    entry_type=entry_type,
                    source_vocabulary=record.get("source_vocabulary", ""),
                    domain_tags=record.get("domain_tags", []),
                )
            )
    return entries


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------


def build_keyword_index(
    entries: list[CatalogEntry],
    min_label_length: int = 3,
) -> list[tuple[str, CatalogEntry]]:
    """Build a keyword index from catalog entries.

    Returns (lowercase_surface, entry) pairs sorted by surface length
    descending (longest match first to prefer specific terms).

    Args:
        entries: Catalog entries with labels and aliases.
        min_label_length: Minimum character length for a label to be included
            (filters out very short terms that produce false positives).

    Returns:
        Sorted list of (surface_lower, entry) tuples.
    """
    index: list[tuple[str, CatalogEntry]] = []
    for entry in entries:
        surfaces = [entry.label] + entry.aliases
        for surface in surfaces:
            if len(surface) >= min_label_length:
                index.append((surface.lower(), entry))

    # Sort by length descending so longer matches take priority
    index.sort(key=lambda x: len(x[0]), reverse=True)
    return index


def find_keyword_spans(
    text: str,
    keyword_index: list[tuple[str, CatalogEntry]],
) -> list[PredictedSpan]:
    """Find all keyword matches in the given text.

    Uses greedy longest-match-first strategy. Once a character position
    is covered by a match, it is not available for shorter matches.

    Args:
        text: Input text to search.
        keyword_index: Pre-built keyword index (sorted longest first).

    Returns:
        List of PredictedSpan objects for all matches found.
    """
    text_lower = text.lower()
    occupied: set[int] = set()
    spans: list[PredictedSpan] = []

    for surface_lower, entry in keyword_index:
        search_from = 0
        while True:
            idx = text_lower.find(surface_lower, search_from)
            if idx == -1:
                break

            span_end = idx + len(surface_lower)

            # Check if any position in this span is already occupied
            span_positions = set(range(idx, span_end))
            if span_positions & occupied:
                search_from = idx + 1
                continue

            # Check word boundaries to reduce false positives
            if not _is_word_boundary(text_lower, idx, span_end):
                search_from = idx + 1
                continue

            occupied.update(span_positions)
            actual_surface = text[idx:span_end]

            spans.append(
                PredictedSpan(
                    surface=actual_surface,
                    start=idx,
                    end=span_end,
                    type=entry.entry_type,
                    canonical_id=entry.entry_id,
                    source_vocabulary=entry.source_vocabulary,
                )
            )

            search_from = span_end

    return sorted(spans, key=lambda s: s.start)


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if the span at [start:end] falls on word boundaries.

    A word boundary exists when:
    - start is 0 or preceded by a non-alphanumeric character
    - end is len(text) or followed by a non-alphanumeric character
    """
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

VOCAB_TO_DOMAIN = {
    "uat": "astronomy",
    "sweet": "earthscience",
    "gcmd": "earthscience",
    "planetary": "planetary",
    "ror": "multidisciplinary",
}


def _spans_match(pred: PredictedSpan, gold: GoldSpan) -> bool:
    """Check if a predicted span matches a gold span.

    Uses exact boundary matching (start and end must match exactly)
    and type must match.
    """
    return pred.start == gold.start and pred.end == gold.end and pred.type == gold.type


def evaluate_record(
    record: dict[str, Any],
    keyword_index: list[tuple[str, CatalogEntry]],
) -> tuple[list[PredictedSpan], list[GoldSpan], list[bool], list[bool]]:
    """Evaluate keyword matching on a single record.

    Returns:
        (predicted_spans, gold_spans, pred_matched, gold_matched)
        where pred_matched[i] indicates if predicted span i matched any gold span,
        and gold_matched[i] indicates if gold span i was found by any prediction.
    """
    text = record.get("text", "")
    gold_span_dicts = record.get("spans", [])

    gold_spans = [
        GoldSpan(
            surface=s.get("surface", ""),
            start=s.get("start", 0),
            end=s.get("end", 0),
            type=s.get("type", ""),
            canonical_id=s.get("canonical_id", ""),
            source_vocabulary=s.get("source_vocabulary", ""),
        )
        for s in gold_span_dicts
    ]

    predicted = find_keyword_spans(text, keyword_index)

    # Match predictions to gold
    pred_matched = [False] * len(predicted)
    gold_matched = [False] * len(gold_spans)

    for pi, pred in enumerate(predicted):
        for gi, gold in enumerate(gold_spans):
            if not gold_matched[gi] and _spans_match(pred, gold):
                pred_matched[pi] = True
                gold_matched[gi] = True
                break

    return predicted, gold_spans, pred_matched, gold_matched


def evaluate_baseline(
    test_records: list[dict[str, Any]],
    keyword_index: list[tuple[str, CatalogEntry]],
) -> BaselineResult:
    """Run the keyword-matching baseline evaluation on the test set.

    Args:
        test_records: List of enrichment record dicts from enrichment_test.jsonl.
        keyword_index: Pre-built keyword index from catalog entries.

    Returns:
        BaselineResult with metrics broken down by type, vocabulary, and domain.
    """
    result = BaselineResult(total_records=len(test_records))

    for record in test_records:
        predicted, gold_spans, pred_matched, gold_matched = evaluate_record(
            record, keyword_index
        )

        result.total_gold_spans += len(gold_spans)
        result.total_predicted_spans += len(predicted)

        # Score overall
        tp = sum(1 for m in pred_matched if m)
        fp = sum(1 for m in pred_matched if not m)
        fn = sum(1 for m in gold_matched if not m)

        result.overall.true_positives += tp
        result.overall.false_positives += fp
        result.overall.false_negatives += fn

        # Score by type (from gold spans)
        for gi, gold in enumerate(gold_spans):
            _ensure_key(result.by_type, gold.type)
            if gold_matched[gi]:
                result.by_type[gold.type].true_positives += 1
            else:
                result.by_type[gold.type].false_negatives += 1

        # False positive types (from predicted spans that didn't match)
        for pi, pred in enumerate(predicted):
            if not pred_matched[pi]:
                _ensure_key(result.by_type, pred.type)
                result.by_type[pred.type].false_positives += 1

        # Score by vocabulary (from gold spans)
        for gi, gold in enumerate(gold_spans):
            vocab = gold.source_vocabulary or "unknown"
            _ensure_key(result.by_vocabulary, vocab)
            if gold_matched[gi]:
                result.by_vocabulary[vocab].true_positives += 1
            else:
                result.by_vocabulary[vocab].false_negatives += 1

        for pi, pred in enumerate(predicted):
            if not pred_matched[pi]:
                vocab = pred.source_vocabulary or "unknown"
                _ensure_key(result.by_vocabulary, vocab)
                result.by_vocabulary[vocab].false_positives += 1

        # Score by domain (inferred from vocabulary)
        for gi, gold in enumerate(gold_spans):
            domain = VOCAB_TO_DOMAIN.get(gold.source_vocabulary, "unknown")
            _ensure_key(result.by_domain, domain)
            if gold_matched[gi]:
                result.by_domain[domain].true_positives += 1
            else:
                result.by_domain[domain].false_negatives += 1

        for pi, pred in enumerate(predicted):
            if not pred_matched[pi]:
                domain = VOCAB_TO_DOMAIN.get(pred.source_vocabulary, "unknown")
                _ensure_key(result.by_domain, domain)
                result.by_domain[domain].false_positives += 1

    return result


def _ensure_key(d: dict[str, EvalMetrics], key: str) -> None:
    """Ensure a key exists in the metrics dict."""
    if key not in d:
        d[key] = EvalMetrics()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_test_records(path: Path) -> list[dict[str, Any]]:
    """Load enrichment test records from JSONL file."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_result(result: BaselineResult, path: Path) -> None:
    """Write evaluation result to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Synthetic test data (fallback when real data isn't available)
# ---------------------------------------------------------------------------


def generate_synthetic_test_data() -> tuple[list[dict[str, Any]], list[CatalogEntry]]:
    """Generate minimal synthetic test data for demonstration.

    Returns:
        (test_records, catalog_entries) for running the baseline evaluation.
    """
    catalog = [
        CatalogEntry(
            entry_id="uat:dark_matter",
            label="dark matter",
            aliases=["DM"],
            entry_type="topic",
            source_vocabulary="uat",
            domain_tags=["astronomy"],
        ),
        CatalogEntry(
            entry_id="uat:solar_wind",
            label="solar wind",
            aliases=[],
            entry_type="topic",
            source_vocabulary="uat",
            domain_tags=["astronomy"],
        ),
        CatalogEntry(
            entry_id="sweet:precipitation",
            label="precipitation",
            aliases=["rainfall"],
            entry_type="topic",
            source_vocabulary="sweet",
            domain_tags=["earthscience"],
        ),
        CatalogEntry(
            entry_id="gcmd:aerosols",
            label="aerosols",
            aliases=["atmospheric aerosols"],
            entry_type="topic",
            source_vocabulary="gcmd",
            domain_tags=["earthscience"],
        ),
        CatalogEntry(
            entry_id="planetary:Mars/1001",
            label="Gale Crater",
            aliases=["Gale"],
            entry_type="institution",
            source_vocabulary="planetary",
            domain_tags=["planetary"],
        ),
        CatalogEntry(
            entry_id="ror:abc123",
            label="Harvard University",
            aliases=["Harvard"],
            entry_type="institution",
            source_vocabulary="ror",
            domain_tags=["multidisciplinary"],
        ),
    ]

    records = [
        {
            "id": "test_001",
            "text": "Dark matter distribution in galaxy clusters observed by Chandra",
            "text_type": "title",
            "spans": [
                {
                    "surface": "dark matter",
                    "start": 0,
                    "end": 11,
                    "type": "topic",
                    "canonical_id": "uat:dark_matter",
                    "source_vocabulary": "uat",
                    "confidence": 1.0,
                },
            ],
            "topics": [],
            "provenance": {"source": "synthetic"},
        },
        {
            "id": "test_002",
            "text": "Solar wind interactions with the Martian magnetosphere near Gale Crater",
            "text_type": "title",
            "spans": [
                {
                    "surface": "solar wind",
                    "start": 0,
                    "end": 10,
                    "type": "topic",
                    "canonical_id": "uat:solar_wind",
                    "source_vocabulary": "uat",
                    "confidence": 1.0,
                },
                {
                    "surface": "Gale Crater",
                    "start": 60,
                    "end": 71,
                    "type": "institution",
                    "canonical_id": "planetary:Mars/1001",
                    "source_vocabulary": "planetary",
                    "confidence": 1.0,
                },
            ],
            "topics": [],
            "provenance": {"source": "synthetic"},
        },
        {
            "id": "test_003",
            "text": "Precipitation patterns and aerosols in tropical regions",
            "text_type": "title",
            "spans": [
                {
                    "surface": "precipitation",
                    "start": 0,
                    "end": 13,
                    "type": "topic",
                    "canonical_id": "sweet:precipitation",
                    "source_vocabulary": "sweet",
                    "confidence": 1.0,
                },
                {
                    "surface": "aerosols",
                    "start": 27,
                    "end": 35,
                    "type": "topic",
                    "canonical_id": "gcmd:aerosols",
                    "source_vocabulary": "gcmd",
                    "confidence": 1.0,
                },
            ],
            "topics": [],
            "provenance": {"source": "synthetic"},
        },
        {
            "id": "test_004",
            "text": "Researchers at Harvard University study dark matter halos",
            "text_type": "abstract",
            "spans": [
                {
                    "surface": "Harvard University",
                    "start": 15,
                    "end": 33,
                    "type": "institution",
                    "canonical_id": "ror:abc123",
                    "source_vocabulary": "ror",
                    "confidence": 1.0,
                },
                {
                    "surface": "dark matter",
                    "start": 40,
                    "end": 51,
                    "type": "topic",
                    "canonical_id": "uat:dark_matter",
                    "source_vocabulary": "uat",
                    "confidence": 1.0,
                },
            ],
            "topics": [],
            "provenance": {"source": "synthetic"},
        },
    ]

    return records, catalog


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyword-matching baseline for enrichment model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run on real data:\n"
            "  python scripts/enrichment_baseline.py \\\n"
            "    --test-file data/enrichment_test.jsonl \\\n"
            "    --topic-catalog data/normalized/topic_catalog.jsonl \\\n"
            "    --entity-catalog data/normalized/entity_catalog.jsonl\n"
            "\n"
            "  # Run with synthetic data (demo mode):\n"
            "  python scripts/enrichment_baseline.py --synthetic\n"
        ),
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Path to enrichment_test.jsonl",
    )
    parser.add_argument(
        "--topic-catalog",
        type=Path,
        help="Path to topic_catalog.jsonl",
    )
    parser.add_argument(
        "--entity-catalog",
        type=Path,
        help="Path to entity_catalog.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("reports/enrichment_baseline.json"),
        help="Path to write evaluation results (default: reports/enrichment_baseline.json)",
    )
    parser.add_argument(
        "--min-label-length",
        type=int,
        default=3,
        help="Minimum label length for keyword matching (default: 3)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic test data instead of real files",
    )

    args = parser.parse_args()

    # Load data
    if args.synthetic:
        print("Running in synthetic mode (demo)...")
        test_records, catalog_entries = generate_synthetic_test_data()
    else:
        if not args.test_file:
            print("Error: --test-file is required (or use --synthetic for demo mode)")
            sys.exit(1)

        if not args.test_file.exists():
            print(f"Error: test file not found: {args.test_file}")
            sys.exit(1)

        print(f"Loading test records from {args.test_file}...")
        test_records = load_test_records(args.test_file)
        print(f"  Loaded {len(test_records)} records")

        catalog_entries: list[CatalogEntry] = []
        if args.topic_catalog and args.topic_catalog.exists():
            topics = load_catalog_entries(args.topic_catalog, "topic")
            print(f"  Loaded {len(topics)} topic catalog entries")
            catalog_entries.extend(topics)

        if args.entity_catalog and args.entity_catalog.exists():
            entities = load_catalog_entries(args.entity_catalog, "institution")
            print(f"  Loaded {len(entities)} entity catalog entries")
            catalog_entries.extend(entities)

        if not catalog_entries:
            print("Warning: no catalog entries loaded; results will show 0 predictions")

    # Build keyword index
    keyword_index = build_keyword_index(catalog_entries, args.min_label_length)
    print(f"  Keyword index: {len(keyword_index)} surface forms")

    # Evaluate
    print("Evaluating...")
    result = evaluate_baseline(test_records, keyword_index)

    # Print summary
    print(f"\nResults ({result.total_records} records):")
    print(f"  Gold spans:      {result.total_gold_spans}")
    print(f"  Predicted spans: {result.total_predicted_spans}")
    print(f"  Overall P/R/F1:  {result.overall.precision:.4f} / "
          f"{result.overall.recall:.4f} / {result.overall.f1:.4f}")

    if result.by_type:
        print("\n  By type:")
        for type_name, metrics in sorted(result.by_type.items()):
            print(f"    {type_name:20s}  P={metrics.precision:.4f}  "
                  f"R={metrics.recall:.4f}  F1={metrics.f1:.4f}")

    if result.by_vocabulary:
        print("\n  By vocabulary:")
        for vocab, metrics in sorted(result.by_vocabulary.items()):
            print(f"    {vocab:20s}  P={metrics.precision:.4f}  "
                  f"R={metrics.recall:.4f}  F1={metrics.f1:.4f}")

    if result.by_domain:
        print("\n  By domain:")
        for domain, metrics in sorted(result.by_domain.items()):
            print(f"    {domain:20s}  P={metrics.precision:.4f}  "
                  f"R={metrics.recall:.4f}  F1={metrics.f1:.4f}")

    # Write output
    write_result(result, args.output_file)
    print(f"\nResults written to {args.output_file}")


if __name__ == "__main__":
    main()
