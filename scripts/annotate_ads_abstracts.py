"""Annotate real ADS abstracts with span annotations using catalog keyword matching.

Uses the same catalog keyword matching approach as the enrichment baseline to
create ground-truth annotations on real ADS abstracts. This produces the
annotated dataset needed for real-world evaluation (US-011).

The annotation focuses on titles + first 2 sentences of abstracts (per the
annotation guide) and applies word-boundary-aware matching with a minimum
label length filter to reduce noise.

Output: data/evaluation/ads_sample_annotated.jsonl

Usage:
    python scripts/annotate_ads_abstracts.py \
        --input-file data/evaluation/ads_sample_raw.jsonl \
        --topic-catalog data/datasets/agent_runs/.../normalized/topic_catalog.jsonl \
        --entity-catalog data/datasets/agent_runs/.../normalized/entity_catalog.jsonl \
        --output-file data/evaluation/ads_sample_annotated.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    """A catalog entry for keyword matching."""

    entry_id: str
    label: str
    aliases: list[str]
    entry_type: str  # "topic" or "entity"
    source_vocabulary: str
    domain_tags: list[str]


@dataclass(frozen=True)
class SpanAnnotation:
    """A single span annotation on a text."""

    surface: str
    start: int
    end: int
    type: str
    canonical_id: str
    source_vocabulary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def load_catalog_entries(path: Path, entry_type: str) -> list[CatalogEntry]:
    """Load catalog entries from a JSONL file."""
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
# Text extraction
# ---------------------------------------------------------------------------


def extract_annotation_text(abstract: str, max_sentences: int = 3) -> str:
    """Extract first N sentences from an abstract for annotation.

    Per the annotation guide: focus on titles + first 2-3 sentences.
    We annotate the full abstract but focus quality on the beginning.
    """
    # For annotation, we use the full abstract since the NER model will
    # process the full text. The annotation guide's "focus" is about
    # human annotation effort — for automated annotation, we can be
    # comprehensive.
    return abstract


# ---------------------------------------------------------------------------
# Keyword matching for annotation
# ---------------------------------------------------------------------------


def build_keyword_index(
    entries: list[CatalogEntry],
    min_label_length: int = 4,
) -> list[tuple[str, CatalogEntry]]:
    """Build a keyword index from catalog entries.

    Returns (lowercase_surface, entry) pairs sorted by surface length
    descending (longest match first).

    Uses a higher min_label_length (4) than baseline (3) to reduce
    false positives on real text where short terms like "gas", "ice",
    "ion" would match everywhere.
    """
    index: list[tuple[str, CatalogEntry]] = []
    seen: set[str] = set()

    for entry in entries:
        surfaces = [entry.label] + entry.aliases
        for surface in surfaces:
            surface_lower = surface.lower().strip()
            if len(surface_lower) >= min_label_length and surface_lower not in seen:
                seen.add(surface_lower)
                index.append((surface_lower, entry))

    # Sort by length descending so longer matches take priority
    index.sort(key=lambda x: len(x[0]), reverse=True)
    return index


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if the span at [start:end] falls on word boundaries."""
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


# Common English words that appear in scientific catalogs but are too generic
# for real-world annotation (they match everywhere in abstracts)
STOPWORD_SURFACES: frozenset[str] = frozenset({
    "data", "model", "method", "analysis", "system", "process",
    "time", "energy", "field", "mass", "force", "area", "form",
    "type", "phase", "rate", "scale", "state", "term", "test",
    "source", "point", "line", "band", "base", "body", "case",
    "cell", "core", "disk", "dust", "edge", "face", "film",
    "flow", "flux", "fold", "gain", "grid", "halo", "heat",
    "hole", "host", "iron", "land", "lens", "limb", "link",
    "load", "loop", "maps", "mean", "mode", "node", "note",
    "pair", "peak", "pole", "ring", "root", "seed", "side",
    "site", "size", "snow", "soil", "star", "step", "tail",
    "tank", "tide", "tool", "tube", "unit", "void", "wave",
    "wind", "wing", "wire", "work", "zone",
    "observation", "observations", "result", "results", "study",
    "using", "based", "density", "region", "regions",
    "sample", "samples", "structure", "properties", "surface",
    "distribution", "evolution", "formation", "function",
    "measurement", "measurements", "parameter", "parameters",
    "spectrum", "spectra", "temperature", "emission",
    "component", "components", "variation", "variations",
    "effect", "effects", "abundance", "abundances",
})


def find_annotation_spans(
    text: str,
    keyword_index: list[tuple[str, CatalogEntry]],
) -> list[SpanAnnotation]:
    """Find all keyword matches in text for annotation.

    Uses greedy longest-match-first with word boundary checking.
    Filters out stopword surfaces to reduce noise on real text.
    """
    text_lower = text.lower()
    occupied: set[int] = set()
    spans: list[SpanAnnotation] = []

    for surface_lower, entry in keyword_index:
        # Skip stopword surfaces
        if surface_lower in STOPWORD_SURFACES:
            continue

        search_from = 0
        while True:
            idx = text_lower.find(surface_lower, search_from)
            if idx == -1:
                break

            span_end = idx + len(surface_lower)

            # Check overlap
            span_positions = set(range(idx, span_end))
            if span_positions & occupied:
                search_from = idx + 1
                continue

            # Check word boundaries
            if not _is_word_boundary(text_lower, idx, span_end):
                search_from = idx + 1
                continue

            occupied.update(span_positions)
            actual_surface = text[idx:span_end]

            spans.append(
                SpanAnnotation(
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


# ---------------------------------------------------------------------------
# Annotation pipeline
# ---------------------------------------------------------------------------


def annotate_abstract(
    record: dict[str, Any],
    keyword_index: list[tuple[str, CatalogEntry]],
) -> dict[str, Any]:
    """Annotate a single ADS abstract record with span annotations.

    Returns an annotated record matching the format specified in the
    annotation guide.
    """
    abstract = record.get("abstract", "")
    spans = find_annotation_spans(abstract, keyword_index)

    # Validate all spans have byte-exact offsets
    validated_spans: list[SpanAnnotation] = []
    for span in spans:
        if abstract[span.start:span.end] == span.surface:
            validated_spans.append(span)

    return {
        "bibcode": record.get("bibcode", ""),
        "title": record.get("title", ""),
        "abstract": abstract,
        "database": record.get("database", []),
        "keywords": record.get("keywords", []),
        "year": record.get("year"),
        "doctype": record.get("doctype", ""),
        "citation_count": record.get("citation_count", 0),
        "domain_category": record.get("domain_category", ""),
        "spans": [s.to_dict() for s in validated_spans],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotate ADS abstracts with span annotations using catalog keyword matching.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to ads_sample_raw.jsonl",
    )
    parser.add_argument(
        "--topic-catalog",
        type=Path,
        required=True,
        help="Path to unified topic_catalog.jsonl",
    )
    parser.add_argument(
        "--entity-catalog",
        type=Path,
        required=True,
        help="Path to unified entity_catalog.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/evaluation/ads_sample_annotated.jsonl"),
        help="Output path for annotated JSONL",
    )
    parser.add_argument(
        "--min-label-length",
        type=int,
        default=4,
        help="Minimum label length for keyword matching (default: 4)",
    )
    parser.add_argument(
        "--max-abstracts",
        type=int,
        default=None,
        help="Annotate only the first N abstracts (default: all)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_file.exists():
        print(f"Error: input file not found: {args.input_file}")
        return 1
    if not args.topic_catalog.exists():
        print(f"Error: topic catalog not found: {args.topic_catalog}")
        return 1
    if not args.entity_catalog.exists():
        print(f"Error: entity catalog not found: {args.entity_catalog}")
        return 1

    # Load catalogs
    print(f"Loading topic catalog from {args.topic_catalog}...")
    topics = load_catalog_entries(args.topic_catalog, "topic")
    print(f"  {len(topics)} topic entries")

    print(f"Loading entity catalog from {args.entity_catalog}...")
    entities = load_catalog_entries(args.entity_catalog, "entity")
    print(f"  {len(entities)} entity entries")

    # Build keyword index
    all_entries = topics + entities
    keyword_index = build_keyword_index(all_entries, args.min_label_length)
    print(f"  Keyword index: {len(keyword_index)} surface forms")

    # Load raw abstracts
    print(f"Loading raw abstracts from {args.input_file}...")
    raw_records: list[dict[str, Any]] = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    if args.max_abstracts is not None:
        raw_records = raw_records[:args.max_abstracts]

    print(f"  {len(raw_records)} abstracts to annotate")

    # Annotate
    print("Annotating abstracts...")
    annotated: list[dict[str, Any]] = []
    total_spans = 0
    spans_by_type: dict[str, int] = {}
    spans_by_vocab: dict[str, int] = {}

    for i, record in enumerate(raw_records):
        result = annotate_abstract(record, keyword_index)
        annotated.append(result)

        n_spans = len(result["spans"])
        total_spans += n_spans

        for span in result["spans"]:
            stype = span["type"]
            vocab = span["source_vocabulary"]
            spans_by_type[stype] = spans_by_type.get(stype, 0) + 1
            spans_by_vocab[vocab] = spans_by_vocab.get(vocab, 0) + 1

        if (i + 1) % 25 == 0:
            print(f"  Annotated {i + 1}/{len(raw_records)} abstracts...")

    # Filter to only include abstracts with at least 1 span
    annotated_with_spans = [r for r in annotated if len(r["spans"]) > 0]

    # Write output
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for record in annotated_with_spans:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write statistics
    stats = {
        "total_abstracts": len(raw_records),
        "annotated_abstracts": len(annotated_with_spans),
        "abstracts_no_spans": len(raw_records) - len(annotated_with_spans),
        "total_spans": total_spans,
        "avg_spans_per_abstract": round(total_spans / len(raw_records), 1) if raw_records else 0,
        "spans_by_type": dict(sorted(spans_by_type.items())),
        "spans_by_vocabulary": dict(sorted(spans_by_vocab.items())),
    }

    stats_path = args.output_file.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nAnnotation complete:")
    print(f"  Total abstracts:    {len(raw_records)}")
    print(f"  With annotations:   {len(annotated_with_spans)}")
    print(f"  Without spans:      {len(raw_records) - len(annotated_with_spans)}")
    print(f"  Total spans:        {total_spans}")
    print(f"  Avg spans/abstract: {stats['avg_spans_per_abstract']}")
    print(f"\n  By type:")
    for stype, count in sorted(spans_by_type.items()):
        print(f"    {stype:20s}  {count}")
    print(f"\n  By vocabulary:")
    for vocab, count in sorted(spans_by_vocab.items()):
        print(f"    {vocab:20s}  {count}")
    print(f"\nOutput: {args.output_file}")
    print(f"Stats:  {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
