"""End-to-end NER + entity linking inference pipeline.

Loads a trained SciBERT NER model and an entity linking index, reads
input JSONL records, extracts typed spans via NER inference, links each
span to catalog canonical IDs via the three-stage cascade (exact → fuzzy
→ embedding), and writes enriched output JSONL.

Output schema per record:
    {
        "id": "...",
        "text": "...",
        "spans": [
            {
                "surface": "...",
                "start": 0,
                "end": 10,
                "type": "topic",
                "canonical_id": "uat:...",
                "source_vocabulary": "uat",
                "confidence": 1.0
            }
        ]
    }

Usage:
    python scripts/run_enrichment_pipeline.py \
        --model-dir output/enrichment_model \
        --linking-index data/linking_index.json \
        --input-file data/datasets/enrichment/enrichment_test.jsonl \
        --output-file output/enrichment_predictions.jsonl

    # With embedding index for full cascade:
    python scripts/run_enrichment_pipeline.py \
        --model-dir output/enrichment_model \
        --linking-index data/linking_index.json \
        --embedding-index data/embedding_index/ \
        --input-file data/datasets/enrichment/enrichment_test.jsonl \
        --output-file output/enrichment_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# BIO label schema (must match the training script)
# ---------------------------------------------------------------------------

ENTITY_TYPES = ("topic", "institution", "author", "date_range")

BIO_LABELS: list[str] = ["O"]
for _etype in ENTITY_TYPES:
    BIO_LABELS.append(f"B-{_etype}")
    BIO_LABELS.append(f"I-{_etype}")

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(BIO_LABELS)}

# Map BIO entity types back to enrichment dataset span types.
BIO_TO_SPAN_TYPE: dict[str, str] = {
    "institution": "entity",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractedSpan:
    """A span extracted by the NER model."""

    surface: str
    start: int
    end: int
    span_type: str


@dataclass(frozen=True)
class EnrichedSpan:
    """A span with entity linking results attached."""

    surface: str
    start: int
    end: int
    type: str
    canonical_id: str
    source_vocabulary: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnrichmentRecord:
    """Output record for one input text."""

    id: str
    text: str
    spans: list[EnrichedSpan]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclass
class LinkingStats:
    """Aggregated linking statistics across all processed records."""

    total_spans: int = 0
    exact_matches: int = 0
    fuzzy_matches: int = 0
    embedding_matches: int = 0
    unlinked: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_spans": self.total_spans,
            "exact_matches": self.exact_matches,
            "fuzzy_matches": self.fuzzy_matches,
            "embedding_matches": self.embedding_matches,
            "unlinked": self.unlinked,
            "pct_exact": _pct(self.exact_matches, self.total_spans),
            "pct_fuzzy": _pct(self.fuzzy_matches, self.total_spans),
            "pct_embedding": _pct(self.embedding_matches, self.total_spans),
            "pct_unlinked": _pct(self.unlinked, self.total_spans),
        }


def _pct(numerator: int, denominator: int) -> float:
    """Return percentage rounded to 2 decimal places."""
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


# ---------------------------------------------------------------------------
# NER inference
# ---------------------------------------------------------------------------

def predict_spans(
    text: str,
    tokenizer: Any,
    model: Any,
    max_length: int = 256,
) -> list[ExtractedSpan]:
    """Run the NER model on *text* and decode BIO tags into spans.

    Returns a list of ExtractedSpan with character offsets.
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

    spans: list[ExtractedSpan] = []
    current_type: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    for tag_id, (char_start, char_end) in zip(pred_ids, offset_mapping):
        # Skip special tokens (offset 0,0)
        if char_start == 0 and char_end == 0:
            continue

        tag = ID2LABEL.get(tag_id, "O")

        if tag.startswith("B-"):
            # Close any open span
            if current_type is not None and current_start is not None:
                spans.append(ExtractedSpan(
                    surface=text[current_start:current_end],
                    start=current_start,
                    end=current_end or current_start,
                    span_type=current_type,
                ))
            raw_type = tag[2:]
            current_type = BIO_TO_SPAN_TYPE.get(raw_type, raw_type)
            current_start = char_start
            current_end = char_end

        elif tag.startswith("I-") and current_type == BIO_TO_SPAN_TYPE.get(tag[2:], tag[2:]):
            # Continue the current span
            current_end = char_end

        else:
            # O tag or type mismatch — close any open span
            if current_type is not None and current_start is not None:
                spans.append(ExtractedSpan(
                    surface=text[current_start:current_end],
                    start=current_start,
                    end=current_end or current_start,
                    span_type=current_type,
                ))
            current_type = None
            current_start = None
            current_end = None

    # Close final span if open
    if current_type is not None and current_start is not None:
        spans.append(ExtractedSpan(
            surface=text[current_start:current_end],
            start=current_start,
            end=current_end or current_start,
            span_type=current_type,
        ))

    return spans


# ---------------------------------------------------------------------------
# Pipeline: NER + linking
# ---------------------------------------------------------------------------

def enrich_text(
    record_id: str,
    text: str,
    tokenizer: Any,
    model: Any,
    linking_index: Any,
    embedding_index: Any | None,
    stats: LinkingStats,
    max_length: int = 256,
) -> EnrichmentRecord:
    """Run NER + entity linking on a single text and return an EnrichmentRecord."""
    from finetune.dataset_agent.entity_linker import link_span_cascade

    extracted = predict_spans(text, tokenizer, model, max_length)
    enriched_spans: list[EnrichedSpan] = []

    for span in extracted:
        stats.total_spans += 1
        link_result = link_span_cascade(
            surface=span.surface,
            span_type=span.span_type,
            linking_index=linking_index,
            embedding_index=embedding_index,
        )

        if link_result is not None:
            if link_result.match_type == "exact":
                stats.exact_matches += 1
            elif link_result.match_type == "fuzzy":
                stats.fuzzy_matches += 1
            elif link_result.match_type == "embedding":
                stats.embedding_matches += 1

            enriched_spans.append(EnrichedSpan(
                surface=span.surface,
                start=span.start,
                end=span.end,
                type=span.span_type,
                canonical_id=link_result.canonical_id,
                source_vocabulary=link_result.source_vocabulary,
                confidence=link_result.confidence,
            ))
        else:
            stats.unlinked += 1
            enriched_spans.append(EnrichedSpan(
                surface=span.surface,
                start=span.start,
                end=span.end,
                type=span.span_type,
                canonical_id="",
                source_vocabulary="",
                confidence=0.0,
            ))

    return EnrichmentRecord(id=record_id, text=text, spans=enriched_spans)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_input_records(path: Path) -> list[dict[str, Any]]:
    """Load input JSONL records. Each must have at least {id, text}."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_output_records(records: list[EnrichmentRecord], path: Path) -> None:
    """Write EnrichmentRecord objects as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end NER + entity linking enrichment pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/run_enrichment_pipeline.py \\
                --model-dir output/enrichment_model \\
                --linking-index data/linking_index.json \\
                --input-file data/datasets/enrichment/enrichment_test.jsonl \\
                --output-file output/enrichment_predictions.jsonl

              # With embedding index:
              python scripts/run_enrichment_pipeline.py \\
                --model-dir output/enrichment_model \\
                --linking-index data/linking_index.json \\
                --embedding-index data/embedding_index/ \\
                --input-file data/datasets/enrichment/enrichment_test.jsonl \\
                --output-file output/enrichment_predictions.jsonl
        """),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to trained NER model directory (contains model + tokenizer)",
    )
    parser.add_argument(
        "--linking-index",
        type=Path,
        required=True,
        help="Path to entity linking index JSON file",
    )
    parser.add_argument(
        "--embedding-index",
        type=Path,
        default=None,
        help="Path to embedding index directory (optional; enables embedding fallback)",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to input JSONL file ({id, text} records)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to output JSONL file for enrichment results",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Max tokenization length (default: 256)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Process only the first N records (default: all)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.model_dir.exists():
        print(f"Error: model directory not found: {args.model_dir}")
        return 1
    if not args.linking_index.exists():
        print(f"Error: linking index not found: {args.linking_index}")
        return 1
    if not args.input_file.exists():
        print(f"Error: input file not found: {args.input_file}")
        return 1
    if args.embedding_index is not None and not args.embedding_index.exists():
        print(f"Error: embedding index directory not found: {args.embedding_index}")
        return 1

    # --- Import ML dependencies ---
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        print("Error: transformers is not installed. Install with: pip install transformers")
        return 1
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Error: torch is not installed. Install with: pip install torch")
        return 1

    from finetune.dataset_agent.entity_linker import (
        load_embedding_index,
        load_linking_index,
    )

    # --- Load NER model ---
    print(f"Loading NER model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(args.model_dir))
    model.eval()
    print(f"  Labels: {model.config.num_labels}")

    # --- Load linking index ---
    print(f"Loading linking index from {args.linking_index}...")
    linking_index = load_linking_index(args.linking_index)
    print(f"  Exact-match entries: {len(linking_index.exact_map)}")
    print(f"  Total surface forms: {len(linking_index.entries)}")

    # --- Load embedding index (optional) ---
    embedding_index = None
    if args.embedding_index is not None:
        print(f"Loading embedding index from {args.embedding_index}...")
        embedding_index = load_embedding_index(args.embedding_index)
        print(f"  Embeddings shape: {embedding_index.embeddings.shape}")
        print(f"  Model: {embedding_index.model_name}")

    # --- Load input data ---
    print(f"Loading input records from {args.input_file}...")
    records = load_input_records(args.input_file)
    if args.max_records is not None:
        records = records[: args.max_records]
    print(f"  {len(records)} records to process")

    # --- Run pipeline ---
    print("Running NER + entity linking pipeline...")
    stats = LinkingStats()
    output_records: list[EnrichmentRecord] = []

    for i, record in enumerate(records):
        record_id = record.get("id", f"record_{i}")
        text = record.get("text", "")
        if not text:
            continue

        enriched = enrich_text(
            record_id=record_id,
            text=text,
            tokenizer=tokenizer,
            model=model,
            linking_index=linking_index,
            embedding_index=embedding_index,
            stats=stats,
            max_length=args.max_seq_length,
        )
        output_records.append(enriched)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(records)} records...")

    # --- Write output ---
    print(f"Writing {len(output_records)} enrichment records to {args.output_file}...")
    write_output_records(output_records, args.output_file)

    # --- Report statistics ---
    print("\n" + "=" * 60)
    print("Enrichment Pipeline Statistics")
    print("=" * 60)
    print(f"  Records processed: {len(output_records)}")
    print(f"  Total NER spans:   {stats.total_spans}")
    print(f"  Exact matches:     {stats.exact_matches} ({_pct(stats.exact_matches, stats.total_spans)}%)")
    print(f"  Fuzzy matches:     {stats.fuzzy_matches} ({_pct(stats.fuzzy_matches, stats.total_spans)}%)")
    print(f"  Embedding matches: {stats.embedding_matches} ({_pct(stats.embedding_matches, stats.total_spans)}%)")
    print(f"  Unlinked:          {stats.unlinked} ({_pct(stats.unlinked, stats.total_spans)}%)")
    print("=" * 60)

    # Write statistics JSON alongside output
    stats_path = args.output_file.with_suffix(".stats.json")
    stats_dict = {
        "records_processed": len(output_records),
        "linking": stats.to_dict(),
    }
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats_dict, fh, indent=2, ensure_ascii=False)
    print(f"\nStatistics saved to {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
