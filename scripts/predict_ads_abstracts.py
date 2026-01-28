"""Run NER model predictions on HTML-cleaned ADS abstracts.

Loads a trained SciBERT NER token-classification model and runs inference
on all abstracts in the reannotated dataset, using the ``abstract_clean``
field. Produces a JSONL file with model-predicted spans including per-span
confidence scores.

Output schema per record:
    {
        "bibcode": "...",
        "title": "...",
        "abstract_clean": "...",
        "domain_category": "...",
        "spans": [
            {
                "surface": "...",
                "start": 0,
                "end": 10,
                "type": "topic",
                "confidence": 0.98
            }
        ]
    }

Usage:
    python scripts/predict_ads_abstracts.py \\
        --model-dir output/enrichment_model \\
        --input-file data/evaluation/ads_sample_reannotated.jsonl \\
        --output-file data/evaluation/ads_sample_predictions.jsonl

    python scripts/predict_ads_abstracts.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# BIO label schema (must match training script)
# ---------------------------------------------------------------------------

ENTITY_TYPES = ("topic", "institution", "author", "date_range")

BIO_LABELS: list[str] = ["O"]
for _etype in ENTITY_TYPES:
    BIO_LABELS.append(f"B-{_etype}")
    BIO_LABELS.append(f"I-{_etype}")

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(BIO_LABELS)}

# Map BIO entity types back to enrichment dataset span types.
# BIO uses "institution" but enrichment dataset uses "entity".
BIO_TO_SPAN_TYPE: dict[str, str] = {
    "institution": "entity",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictedSpan:
    """A span extracted by the NER model with confidence score."""

    surface: str
    start: int
    end: int
    type: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_spans_for_text(
    text: str,
    tokenizer: Any,
    model: Any,
    max_length: int = 512,
) -> list[PredictedSpan]:
    """Run the NER model on *text* and decode BIO tags into spans.

    For each span, confidence is the max softmax probability of the B- tag
    that initiated the span.

    Returns a list of ``PredictedSpan`` with character offsets and confidence.
    """
    import torch

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offset_mapping = encoding.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

    logits = outputs.logits[0]  # (seq_len, num_labels)
    pred_ids = torch.argmax(logits, dim=-1).tolist()

    # Compute softmax probabilities for confidence scores
    probs = torch.softmax(logits, dim=-1)

    # Decode BIO tags into character-level spans
    spans: list[PredictedSpan] = []
    current_type: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    current_confidence: float = 0.0

    for token_idx, (tag_id, (char_start, char_end)) in enumerate(
        zip(pred_ids, offset_mapping)
    ):
        # Skip special tokens (CLS, SEP, PAD have offset (0, 0))
        if char_start == 0 and char_end == 0:
            continue

        tag = ID2LABEL.get(tag_id, "O")

        if tag.startswith("B-"):
            # Close any open span
            if current_type is not None and current_start is not None:
                spans.append(
                    PredictedSpan(
                        surface=text[current_start:current_end],
                        start=current_start,
                        end=current_end or current_start,
                        type=current_type,
                        confidence=round(current_confidence, 4),
                    )
                )
            raw_type = tag[2:]
            current_type = BIO_TO_SPAN_TYPE.get(raw_type, raw_type)
            current_start = char_start
            current_end = char_end
            # Confidence = softmax probability of the B- tag for this token
            current_confidence = probs[token_idx, tag_id].item()

        elif (
            tag.startswith("I-")
            and current_type == BIO_TO_SPAN_TYPE.get(tag[2:], tag[2:])
        ):
            # Continue the current span
            current_end = char_end

        else:
            # O tag or type mismatch — close any open span
            if current_type is not None and current_start is not None:
                spans.append(
                    PredictedSpan(
                        surface=text[current_start:current_end],
                        start=current_start,
                        end=current_end or current_start,
                        type=current_type,
                        confidence=round(current_confidence, 4),
                    )
                )
            current_type = None
            current_start = None
            current_end = None
            current_confidence = 0.0

    # Close final span if open
    if current_type is not None and current_start is not None:
        spans.append(
            PredictedSpan(
                surface=text[current_start:current_end],
                start=current_start,
                end=current_end or current_start,
                type=current_type,
                confidence=round(current_confidence, 4),
            )
        )

    return spans


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from *path*."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_predictions(
    model_dir: Path,
    input_file: Path,
    output_file: Path,
    max_seq_length: int = 512,
) -> dict[str, Any]:
    """Load the NER model and run inference on all abstracts.

    Returns a summary statistics dict.
    """
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        print("Error: transformers is not installed. Install with: pip install transformers")
        sys.exit(1)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Error: torch is not installed. Install with: pip install torch")
        sys.exit(1)

    # Load model
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()
    print(f"  Labels: {model.config.num_labels}")

    # Load input data
    print(f"Loading abstracts from {input_file}...")
    records = load_records(input_file)
    print(f"  {len(records)} abstracts loaded")

    # Run inference
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_spans = 0
    spans_by_type: dict[str, int] = {}
    confidence_sum = 0.0
    confidence_count = 0
    records_with_spans = 0

    with open(output_file, "w", encoding="utf-8") as fh:
        for i, record in enumerate(records):
            bibcode = record.get("bibcode", "")
            text = record.get("abstract_clean", "")
            if not text:
                # Fall back to raw abstract if no clean version
                text = record.get("abstract", "")

            predicted = predict_spans_for_text(text, tokenizer, model, max_seq_length)

            # Build output record
            output_record = {
                "bibcode": bibcode,
                "title": record.get("title", ""),
                "abstract_clean": text,
                "domain_category": record.get("domain_category", ""),
                "spans": [span.to_dict() for span in predicted],
            }

            fh.write(json.dumps(output_record, ensure_ascii=False) + "\n")

            # Accumulate stats
            total_spans += len(predicted)
            if predicted:
                records_with_spans += 1
            for span in predicted:
                spans_by_type[span.type] = spans_by_type.get(span.type, 0) + 1
                confidence_sum += span.confidence
                confidence_count += 1

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(records)} abstracts ({total_spans} spans so far)")

    avg_confidence = (
        round(confidence_sum / confidence_count, 4) if confidence_count > 0 else 0.0
    )

    stats = {
        "total_abstracts": len(records),
        "records_with_spans": records_with_spans,
        "total_spans": total_spans,
        "spans_per_abstract": round(total_spans / max(len(records), 1), 2),
        "spans_by_type": dict(sorted(spans_by_type.items())),
        "average_confidence": avg_confidence,
        "model_dir": str(model_dir),
        "max_seq_length": max_seq_length,
    }

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run NER model predictions on HTML-cleaned ADS abstracts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/predict_ads_abstracts.py \\\n"
            "    --model-dir output/enrichment_model \\\n"
            "    --input-file data/evaluation/ads_sample_reannotated.jsonl \\\n"
            "    --output-file data/evaluation/ads_sample_predictions.jsonl\n"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("output/enrichment_model"),
        help="Path to trained model directory (default: output/enrichment_model)",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/evaluation/ads_sample_reannotated.jsonl"),
        help="Path to reannotated abstracts JSONL (default: data/evaluation/ads_sample_reannotated.jsonl)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/evaluation/ads_sample_predictions.jsonl"),
        help="Path for output predictions JSONL (default: data/evaluation/ads_sample_predictions.jsonl)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max tokenization length (default: 512 — use 512 for full abstracts)",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Error: model directory not found: {args.model_dir}")
        return 1
    if not args.input_file.exists():
        print(f"Error: input file not found: {args.input_file}")
        return 1

    stats = run_predictions(
        model_dir=args.model_dir,
        input_file=args.input_file,
        output_file=args.output_file,
        max_seq_length=args.max_seq_length,
    )

    # Print summary
    print(f"\nPredictions written to {args.output_file}")
    print(f"  Total abstracts: {stats['total_abstracts']}")
    print(f"  Abstracts with spans: {stats['records_with_spans']}")
    print(f"  Total spans: {stats['total_spans']}")
    print(f"  Spans per abstract: {stats['spans_per_abstract']}")
    print(f"  Average confidence: {stats['average_confidence']}")
    print("  Spans by type:")
    for stype, count in stats["spans_by_type"].items():
        print(f"    {stype}: {count}")

    # Also write stats file
    stats_path = args.output_file.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"\nStats written to {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
