#!/usr/bin/env python3
"""Export human annotations from dashboard to training format.

Converts dashboard-exported annotations (JSONL) into EnrichmentRecord format
for mixed retraining with synthetic data.

Input:
    Dashboard export JSON (from the Export button):
    {
        "bibcode": str,
        "title": str,
        "abstract_clean": str,
        "domain_category": str,
        "spans": [
            {
                "surface": str,
                "start": int,
                "end": int,
                "type": str,
                "source": str,
                "canonical_id": optional str,
                "source_vocabulary": optional str,
                "confidence": optional float,
            }
        ],
        "review_status": str,
        "notes": str,
    }

Output:
    EnrichmentRecord format JSONL (enrichment_labels.jsonl schema):
    {
        "id": str,
        "text": str,
        "text_type": "abstract",
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
        "provenance": {
            "source": "human_annotation",
            "original_bibcode": str,
            "domain_category": str,
            "review_status": str,
            "notes": str,
        },
    }

Usage:
    python scripts/export_annotations_to_training.py [options]

Examples:
    # Process dashboard export (90/10 train/val split)
    python scripts/export_annotations_to_training.py \\
        --input ner_annotations_export.jsonl \\
        --output-dir data/datasets/enrichment \\
        --train-fraction 0.9

    # Custom output paths
    python scripts/export_annotations_to_training.py \\
        --input my_annotations.jsonl \\
        --train-output human_train.jsonl \\
        --val-output human_val.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SpanValidationError:
    """A single span validation failure."""

    bibcode: str
    span_surface: str
    span_start: int
    span_end: int
    expected: str
    actual: str
    text_snippet: str


@dataclass
class ConversionStats:
    """Statistics from the conversion process."""

    total_records: int = 0
    total_spans: int = 0
    spans_by_type: dict[str, int] = None
    spans_by_vocabulary: dict[str, int] = None
    spans_by_source: dict[str, int] = None
    validation_failures: list[SpanValidationError] = None
    records_with_failures: int = 0
    train_records: int = 0
    val_records: int = 0

    def __post_init__(self):
        if self.spans_by_type is None:
            self.spans_by_type = {}
        if self.spans_by_vocabulary is None:
            self.spans_by_vocabulary = {}
        if self.spans_by_source is None:
            self.spans_by_source = {}
        if self.validation_failures is None:
            self.validation_failures = []


def validate_span_offsets(
    text: str, spans: list[dict[str, Any]]
) -> list[SpanValidationError]:
    """Validate that span offsets correctly extract surface text.

    Args:
        text: The full abstract text.
        spans: List of span dicts with surface, start, end fields.

    Returns:
        List of validation errors (empty if all spans are valid).
    """
    errors = []
    for span in spans:
        surface = span["surface"]
        start = span["start"]
        end = span["end"]

        # Extract actual text at span offsets
        actual = text[start:end]

        if actual != surface:
            # Provide context: 20 chars before and after
            context_start = max(0, start - 20)
            context_end = min(len(text), end + 20)
            context = text[context_start:context_end]

            errors.append(
                SpanValidationError(
                    bibcode="",  # Will be filled in by caller
                    span_surface=surface,
                    span_start=start,
                    span_end=end,
                    expected=surface,
                    actual=actual,
                    text_snippet=context,
                )
            )

    return errors


def convert_to_enrichment_record(
    dashboard_record: dict[str, Any], seed_prefix: str
) -> dict[str, Any]:
    """Convert dashboard export record to EnrichmentRecord format.

    Args:
        dashboard_record: Dashboard export record.
        seed_prefix: Prefix for generating unique IDs (e.g., "human_train").

    Returns:
        EnrichmentRecord dict ready for JSONL serialization.
    """
    bibcode = dashboard_record["bibcode"]
    text = dashboard_record["abstract_clean"]
    domain_category = dashboard_record.get("domain_category", "unknown")
    review_status = dashboard_record.get("review_status", "reviewed")
    notes = dashboard_record.get("notes", "")

    # Generate unique ID
    id_seed = f"{seed_prefix}_{bibcode}"
    record_id = f"enr_hum_{hashlib.md5(id_seed.encode()).hexdigest()[:10]}"

    # Convert spans
    enrichment_spans = []
    topics = []
    topic_ids = set()

    for span in dashboard_record["spans"]:
        # Build enrichment span
        enrichment_span = {
            "surface": span["surface"],
            "start": span["start"],
            "end": span["end"],
            "type": span["type"],
            "canonical_id": span.get("canonical_id", ""),
            "source_vocabulary": span.get("source_vocabulary", "human"),
            "confidence": span.get("confidence", 1.0),
        }
        enrichment_spans.append(enrichment_span)

        # Build topic list (only for topic-type spans with canonical_id)
        if span["type"] == "topic" and span.get("canonical_id"):
            concept_id = span["canonical_id"]
            if concept_id not in topic_ids:
                topic_ids.add(concept_id)
                topics.append(
                    {
                        "concept_id": concept_id,
                        "label": span["surface"],
                        "source_vocabulary": span.get("source_vocabulary", "human"),
                        "confidence": span.get("confidence", 1.0),
                    }
                )

    # Build provenance
    provenance = {
        "source": "human_annotation",
        "original_bibcode": bibcode,
        "domain_category": domain_category,
        "review_status": review_status,
        "notes": notes,
    }

    return {
        "id": record_id,
        "text": text,
        "text_type": "abstract",
        "spans": enrichment_spans,
        "topics": topics,
        "provenance": provenance,
    }


def process_annotations(
    input_path: Path,
    train_output: Path,
    val_output: Path,
    train_fraction: float = 0.9,
    seed: int = 42,
) -> ConversionStats:
    """Process dashboard annotations and split into train/val sets.

    Args:
        input_path: Path to dashboard export JSONL.
        train_output: Path to write training set.
        val_output: Path to write validation set.
        train_fraction: Fraction of records for training (rest go to val).
        seed: Random seed for reproducible splitting.

    Returns:
        ConversionStats with processing statistics.
    """
    stats = ConversionStats()
    random.seed(seed)

    # Load all records
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Skipping line {line_num} (invalid JSON): {e}",
                    file=sys.stderr,
                )

    if not records:
        print("Error: No valid records found in input file", file=sys.stderr)
        return stats

    stats.total_records = len(records)

    # Shuffle for random splitting
    random.shuffle(records)

    # Split into train/val
    split_idx = int(len(records) * train_fraction)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    stats.train_records = len(train_records)
    stats.val_records = len(val_records)

    # Process and write train set
    train_output.parent.mkdir(parents=True, exist_ok=True)
    with open(train_output, "w", encoding="utf-8") as f:
        for record in train_records:
            # Skip records without required fields
            if "abstract_clean" not in record or "spans" not in record:
                print(
                    f"Warning: Skipping record {record.get('bibcode', 'unknown')} (missing required fields)",
                    file=sys.stderr,
                )
                continue

            # Validate spans
            validation_errors = validate_span_offsets(
                record["abstract_clean"], record["spans"]
            )
            if validation_errors:
                stats.records_with_failures += 1
                for error in validation_errors:
                    error.bibcode = record.get("bibcode", "unknown")
                    stats.validation_failures.append(error)

            # Convert to enrichment format
            enrichment_record = convert_to_enrichment_record(record, "train")
            f.write(json.dumps(enrichment_record) + "\n")

            # Update stats
            stats.total_spans += len(record["spans"])
            for span in record["spans"]:
                span_type = span["type"]
                stats.spans_by_type[span_type] = (
                    stats.spans_by_type.get(span_type, 0) + 1
                )

                vocab = span.get("source_vocabulary", "human")
                stats.spans_by_vocabulary[vocab] = (
                    stats.spans_by_vocabulary.get(vocab, 0) + 1
                )

                source = span.get("source", "unknown")
                stats.spans_by_source[source] = (
                    stats.spans_by_source.get(source, 0) + 1
                )

    # Process and write val set
    val_output.parent.mkdir(parents=True, exist_ok=True)
    with open(val_output, "w", encoding="utf-8") as f:
        for record in val_records:
            # Skip records without required fields
            if "abstract_clean" not in record or "spans" not in record:
                print(
                    f"Warning: Skipping record {record.get('bibcode', 'unknown')} (missing required fields)",
                    file=sys.stderr,
                )
                continue

            # Validate spans
            validation_errors = validate_span_offsets(
                record["abstract_clean"], record["spans"]
            )
            if validation_errors:
                stats.records_with_failures += 1
                for error in validation_errors:
                    error.bibcode = record.get("bibcode", "unknown")
                    stats.validation_failures.append(error)

            # Convert to enrichment format
            enrichment_record = convert_to_enrichment_record(record, "val")
            f.write(json.dumps(enrichment_record) + "\n")

    return stats


def print_report(stats: ConversionStats) -> None:
    """Print conversion statistics report.

    Args:
        stats: ConversionStats object with processing results.
    """
    print("\n" + "=" * 60)
    print("EXPORT ANNOTATIONS TO TRAINING — REPORT")
    print("=" * 60)

    print(f"\nTotal records: {stats.total_records}")
    print(f"  Train: {stats.train_records}")
    print(f"  Val: {stats.val_records}")

    print(f"\nTotal spans: {stats.total_spans}")

    print("\nSpans by type:")
    for span_type, count in sorted(
        stats.spans_by_type.items(), key=lambda x: x[1], reverse=True
    ):
        pct = (count / stats.total_spans * 100) if stats.total_spans > 0 else 0
        print(f"  {span_type}: {count} ({pct:.1f}%)")

    print("\nSpans by vocabulary:")
    for vocab, count in sorted(
        stats.spans_by_vocabulary.items(), key=lambda x: x[1], reverse=True
    ):
        pct = (count / stats.total_spans * 100) if stats.total_spans > 0 else 0
        print(f"  {vocab}: {count} ({pct:.1f}%)")

    print("\nSpans by source:")
    for source, count in sorted(
        stats.spans_by_source.items(), key=lambda x: x[1], reverse=True
    ):
        pct = (count / stats.total_spans * 100) if stats.total_spans > 0 else 0
        print(f"  {source}: {count} ({pct:.1f}%)")

    print(f"\nValidation failures: {len(stats.validation_failures)}")
    if stats.validation_failures:
        print(
            f"  Records with failures: {stats.records_with_failures} / {stats.total_records}"
        )
        print("\nFirst 10 validation failures:")
        for i, error in enumerate(stats.validation_failures[:10], start=1):
            print(f"\n  {i}. {error.bibcode}")
            print(f"     Surface: '{error.span_surface}'")
            print(f"     Offsets: [{error.span_start}:{error.span_end}]")
            print(f"     Expected: '{error.expected}'")
            print(f"     Actual: '{error.actual}'")
            print(f"     Context: ...{error.text_snippet}...")

        if len(stats.validation_failures) > 10:
            print(f"\n  ... and {len(stats.validation_failures) - 10} more failures")
    else:
        print("  ✓ All spans validated successfully")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Export human annotations from dashboard to training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ner_annotations_export.jsonl"),
        help="Path to dashboard export JSONL (default: ner_annotations_export.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for train/val files (creates human_annotated_train.jsonl and human_annotated_val.jsonl)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        help="Path to write training set JSONL (overrides --output-dir)",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        help="Path to write validation set JSONL (overrides --output-dir)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of records for training (default: 0.9)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting (default: 42)"
    )

    args = parser.parse_args()

    # Determine output paths
    if args.train_output and args.val_output:
        train_output = args.train_output
        val_output = args.val_output
    elif args.output_dir:
        train_output = args.output_dir / "human_annotated_train.jsonl"
        val_output = args.output_dir / "human_annotated_val.jsonl"
    else:
        train_output = Path("data/datasets/enrichment/human_annotated_train.jsonl")
        val_output = Path("data/datasets/enrichment/human_annotated_val.jsonl")

    # Check input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {args.input}")
    print(f"Train output: {train_output}")
    print(f"Val output: {val_output}")
    print(f"Train fraction: {args.train_fraction}")
    print(f"Seed: {args.seed}")

    # Process annotations
    stats = process_annotations(
        args.input, train_output, val_output, args.train_fraction, args.seed
    )

    # Print report
    print_report(stats)

    # Exit with error code if there were validation failures
    if stats.validation_failures:
        print(
            f"\n⚠️  Warning: {len(stats.validation_failures)} validation failures detected",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("\n✓ Export completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
