"""Clean HTML from ADS abstract files and update span offsets.

Processes:
  1. data/evaluation/ads_sample_raw.jsonl → adds ``abstract_clean`` field
  2. data/evaluation/ads_sample_annotated.jsonl → adds ``abstract_clean`` field
     and recalculates span start/end offsets to match the cleaned text

Usage:
    python scripts/clean_ads_abstracts.py

Depends on: scripts/html_utils.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sibling modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from html_utils import clean_abstract_and_remap_spans, strip_html_tags

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "evaluation"


def _process_raw(input_path: Path, output_path: Path) -> dict[str, int]:
    """Add ``abstract_clean`` to every record in the raw JSONL file."""
    records: list[dict] = []
    html_count = 0

    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            abstract = record.get("abstract", "")
            cleaned = strip_html_tags(abstract)
            record = {**record, "abstract_clean": cleaned}
            if cleaned != abstract:
                html_count += 1
            records.append(record)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"total": len(records), "html_modified": html_count}


def _process_annotated(input_path: Path, output_path: Path) -> dict[str, int]:
    """Add ``abstract_clean`` and remap span offsets in the annotated JSONL file."""
    records: list[dict] = []
    total_spans_before = 0
    total_spans_after = 0
    span_offset_changes = 0
    dropped_spans = 0

    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            abstract = record.get("abstract", "")
            spans = record.get("spans", [])

            total_spans_before += len(spans)
            clean_text, remapped_spans = clean_abstract_and_remap_spans(abstract, spans)
            total_spans_after += len(remapped_spans)
            dropped_spans += len(spans) - len(remapped_spans)

            # Count spans where offsets actually changed
            for orig, new in zip(spans, remapped_spans):
                if orig["start"] != new["start"] or orig["end"] != new["end"]:
                    span_offset_changes += 1

            # Validate all remapped spans
            validation_failures = 0
            for span in remapped_spans:
                extracted = clean_text[span["start"] : span["end"]]
                if extracted != span["surface"]:
                    validation_failures += 1

            record = {
                **record,
                "abstract_clean": clean_text,
                "spans": remapped_spans,
            }
            records.append(record)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "total_records": len(records),
        "spans_before": total_spans_before,
        "spans_after": total_spans_after,
        "spans_dropped": dropped_spans,
        "span_offset_changes": span_offset_changes,
        "validation_failures": validation_failures,
    }


def main() -> None:
    raw_path = DATA_DIR / "ads_sample_raw.jsonl"
    annotated_path = DATA_DIR / "ads_sample_annotated.jsonl"

    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found")
        sys.exit(1)

    # Process raw file
    print(f"Processing {raw_path} ...")
    raw_stats = _process_raw(raw_path, raw_path)
    print(f"  Total records: {raw_stats['total']}")
    print(f"  Records with HTML cleaned: {raw_stats['html_modified']}")

    # Process annotated file
    if annotated_path.exists():
        print(f"\nProcessing {annotated_path} ...")
        ann_stats = _process_annotated(annotated_path, annotated_path)
        print(f"  Total records: {ann_stats['total_records']}")
        print(f"  Spans before: {ann_stats['spans_before']}")
        print(f"  Spans after: {ann_stats['spans_after']}")
        print(f"  Spans dropped (unmappable): {ann_stats['spans_dropped']}")
        print(f"  Span offsets changed: {ann_stats['span_offset_changes']}")
        print(f"  Validation failures: {ann_stats['validation_failures']}")
    else:
        print(f"\nWARNING: {annotated_path} not found, skipping span remapping")

    print("\nDone.")


if __name__ == "__main__":
    main()
