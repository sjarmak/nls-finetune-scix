"""Re-annotate ADS abstracts with curated SWEET vocabulary + HTML-cleaned text.

US-003: Uses HTML-stripped abstract_clean text from US-001 and curated SWEET
vocabulary from US-002, while keeping UAT, GCMD, ROR, and planetary catalogs
unchanged. Produces comparison stats showing span count before vs after curation.

Output: data/evaluation/ads_sample_reannotated.jsonl

Usage:
    python scripts/reannotate_ads_abstracts.py \
        --input-file data/evaluation/ads_sample_annotated.jsonl \
        --sweet-curated data/vocabularies/sweet_curated.jsonl \
        --topic-catalog-uat data/datasets/agent_runs/.../normalized/topic_catalog_uat.jsonl \
        --topic-catalog-gcmd data/datasets/agent_runs/.../normalized/topic_catalog_gcmd.jsonl \
        --entity-catalog data/datasets/agent_runs/.../normalized/entity_catalog.jsonl \
        --output-file data/evaluation/ads_sample_reannotated.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Reuse annotation logic from the existing annotator
from annotate_ads_abstracts import (
    CatalogEntry,
    SpanAnnotation,
    build_keyword_index,
    find_annotation_spans,
    load_catalog_entries,
)


# ---------------------------------------------------------------------------
# SWEET curated loader
# ---------------------------------------------------------------------------


def load_sweet_curated(path: Path) -> list[CatalogEntry]:
    """Load curated SWEET entries as CatalogEntry objects.

    The curated SWEET JSONL has the same structure as topic catalog entries.
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
                    entry_type="topic",
                    source_vocabulary=record.get("source_vocabulary", "sweet"),
                    domain_tags=record.get("domain_tags", ["earthscience"]),
                )
            )
    return entries


# ---------------------------------------------------------------------------
# Re-annotation
# ---------------------------------------------------------------------------


def reannotate_abstract(
    record: dict[str, Any],
    keyword_index: list[tuple[str, CatalogEntry]],
) -> dict[str, Any]:
    """Re-annotate a single abstract using abstract_clean and curated vocabulary.

    Uses abstract_clean (HTML-stripped) text for annotation. Falls back to
    abstract if abstract_clean is not available.
    """
    abstract_clean = record.get("abstract_clean", record.get("abstract", ""))
    spans = find_annotation_spans(abstract_clean, keyword_index)

    # Validate all spans have byte-exact offsets
    validated_spans: list[SpanAnnotation] = []
    for span in spans:
        if abstract_clean[span.start : span.end] == span.surface:
            validated_spans.append(span)

    return {
        "bibcode": record.get("bibcode", ""),
        "title": record.get("title", ""),
        "abstract": record.get("abstract", ""),
        "abstract_clean": abstract_clean,
        "database": record.get("database", []),
        "keywords": record.get("keywords", []),
        "year": record.get("year"),
        "doctype": record.get("doctype", ""),
        "citation_count": record.get("citation_count", 0),
        "domain_category": record.get("domain_category", ""),
        "spans": [s.to_dict() for s in validated_spans],
    }


# ---------------------------------------------------------------------------
# Comparison stats
# ---------------------------------------------------------------------------


def compute_comparison_stats(
    old_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute before-vs-after stats for the re-annotation."""
    old_by_bibcode = {r["bibcode"]: r for r in old_records}

    old_total = 0
    new_total = 0
    old_by_vocab: dict[str, int] = {}
    new_by_vocab: dict[str, int] = {}
    old_by_type: dict[str, int] = {}
    new_by_type: dict[str, int] = {}

    per_abstract: list[dict[str, Any]] = []

    for new_rec in new_records:
        bibcode = new_rec["bibcode"]
        old_rec = old_by_bibcode.get(bibcode, {"spans": []})
        old_spans = old_rec.get("spans", [])
        new_spans = new_rec.get("spans", [])

        old_count = len(old_spans)
        new_count = len(new_spans)
        old_total += old_count
        new_total += new_count

        for s in old_spans:
            vocab = s.get("source_vocabulary", "unknown")
            stype = s.get("type", "unknown")
            old_by_vocab[vocab] = old_by_vocab.get(vocab, 0) + 1
            old_by_type[stype] = old_by_type.get(stype, 0) + 1

        for s in new_spans:
            vocab = s.get("source_vocabulary", "unknown")
            stype = s.get("type", "unknown")
            new_by_vocab[vocab] = new_by_vocab.get(vocab, 0) + 1
            new_by_type[stype] = new_by_type.get(stype, 0) + 1

        per_abstract.append({
            "bibcode": bibcode,
            "old_spans": old_count,
            "new_spans": new_count,
            "reduction": old_count - new_count,
            "reduction_pct": (
                round((old_count - new_count) / old_count * 100, 1)
                if old_count > 0
                else 0.0
            ),
        })

    reduction_total = old_total - new_total
    reduction_pct = (
        round(reduction_total / old_total * 100, 1) if old_total > 0 else 0.0
    )

    # Per-vocabulary comparison
    all_vocabs = sorted(set(list(old_by_vocab.keys()) + list(new_by_vocab.keys())))
    vocab_comparison: list[dict[str, Any]] = []
    for vocab in all_vocabs:
        old_v = old_by_vocab.get(vocab, 0)
        new_v = new_by_vocab.get(vocab, 0)
        vocab_comparison.append({
            "vocabulary": vocab,
            "old_count": old_v,
            "new_count": new_v,
            "reduction": old_v - new_v,
            "reduction_pct": (
                round((old_v - new_v) / old_v * 100, 1) if old_v > 0 else 0.0
            ),
        })

    return {
        "summary": {
            "total_abstracts": len(new_records),
            "old_total_spans": old_total,
            "new_total_spans": new_total,
            "reduction": reduction_total,
            "reduction_pct": reduction_pct,
        },
        "by_vocabulary": vocab_comparison,
        "old_by_type": dict(sorted(old_by_type.items())),
        "new_by_type": dict(sorted(new_by_type.items())),
        "per_abstract_summary": {
            "max_reduction": max(
                (a["reduction"] for a in per_abstract), default=0
            ),
            "min_reduction": min(
                (a["reduction"] for a in per_abstract), default=0
            ),
            "avg_reduction": (
                round(
                    sum(a["reduction"] for a in per_abstract) / len(per_abstract),
                    1,
                )
                if per_abstract
                else 0.0
            ),
        },
        "per_abstract": per_abstract,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-annotate ADS abstracts with curated SWEET + HTML-cleaned text."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/evaluation/ads_sample_annotated.jsonl"),
        help="Path to ads_sample_annotated.jsonl (from US-001, with abstract_clean)",
    )
    parser.add_argument(
        "--sweet-curated",
        type=Path,
        default=Path("data/vocabularies/sweet_curated.jsonl"),
        help="Path to curated SWEET vocabulary from US-002",
    )
    parser.add_argument(
        "--topic-catalog-uat",
        type=Path,
        default=Path(
            "data/datasets/agent_runs/run_20260127_174306_999adfdd"
            "/normalized/topic_catalog_uat.jsonl"
        ),
        help="Path to UAT topic catalog (unchanged)",
    )
    parser.add_argument(
        "--topic-catalog-gcmd",
        type=Path,
        default=Path(
            "data/datasets/agent_runs/run_20260127_174306_999adfdd"
            "/normalized/topic_catalog_gcmd.jsonl"
        ),
        help="Path to GCMD topic catalog (unchanged)",
    )
    parser.add_argument(
        "--entity-catalog",
        type=Path,
        default=Path(
            "data/datasets/agent_runs/run_20260127_174306_999adfdd"
            "/normalized/entity_catalog.jsonl"
        ),
        help="Path to entity catalog (ROR + planetary, unchanged)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/evaluation/ads_sample_reannotated.jsonl"),
        help="Output path for re-annotated JSONL",
    )
    parser.add_argument(
        "--min-label-length",
        type=int,
        default=4,
        help="Minimum label length for keyword matching (default: 4)",
    )

    args = parser.parse_args()

    # Validate inputs
    missing = []
    for name, path in [
        ("input-file", args.input_file),
        ("sweet-curated", args.sweet_curated),
        ("topic-catalog-uat", args.topic_catalog_uat),
        ("topic-catalog-gcmd", args.topic_catalog_gcmd),
        ("entity-catalog", args.entity_catalog),
    ]:
        if not path.exists():
            missing.append(f"  {name}: {path}")
    if missing:
        print("Error: missing input files:")
        for m in missing:
            print(m)
        return 1

    # Load old annotated records (for comparison)
    print(f"Loading existing annotations from {args.input_file}...")
    old_records: list[dict[str, Any]] = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                old_records.append(json.loads(line))
    print(f"  {len(old_records)} records loaded")

    # Load catalogs (non-SWEET)
    print(f"Loading UAT topic catalog from {args.topic_catalog_uat}...")
    uat_topics = load_catalog_entries(args.topic_catalog_uat, "topic")
    print(f"  {len(uat_topics)} UAT entries")

    print(f"Loading GCMD topic catalog from {args.topic_catalog_gcmd}...")
    gcmd_topics = load_catalog_entries(args.topic_catalog_gcmd, "topic")
    print(f"  {len(gcmd_topics)} GCMD entries")

    print(f"Loading entity catalog from {args.entity_catalog}...")
    entities = load_catalog_entries(args.entity_catalog, "entity")
    print(f"  {len(entities)} entity entries")

    # Load curated SWEET (replaces full SWEET)
    print(f"Loading curated SWEET vocabulary from {args.sweet_curated}...")
    sweet_curated = load_sweet_curated(args.sweet_curated)
    print(f"  {len(sweet_curated)} curated SWEET entries (was ~12,986 pre-curation)")

    # Build keyword index with curated vocabulary
    all_entries = uat_topics + gcmd_topics + sweet_curated + entities
    keyword_index = build_keyword_index(all_entries, args.min_label_length)
    print(f"  Combined keyword index: {len(keyword_index)} surface forms")

    # Re-annotate all abstracts
    print("Re-annotating abstracts with curated vocabulary + clean text...")
    new_records: list[dict[str, Any]] = []
    total_spans = 0
    spans_by_type: dict[str, int] = {}
    spans_by_vocab: dict[str, int] = {}

    for i, record in enumerate(old_records):
        result = reannotate_abstract(record, keyword_index)
        new_records.append(result)

        n_spans = len(result["spans"])
        total_spans += n_spans

        for span in result["spans"]:
            stype = span["type"]
            vocab = span["source_vocabulary"]
            spans_by_type[stype] = spans_by_type.get(stype, 0) + 1
            spans_by_vocab[vocab] = spans_by_vocab.get(vocab, 0) + 1

        if (i + 1) % 25 == 0:
            print(f"  Re-annotated {i + 1}/{len(old_records)} abstracts...")

    # Write re-annotated output
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for record in new_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Compute comparison stats
    comparison = compute_comparison_stats(old_records, new_records)

    stats_path = args.output_file.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Print summary
    summary = comparison["summary"]
    print(f"\nRe-annotation complete:")
    print(f"  Total abstracts:    {summary['total_abstracts']}")
    print(f"  Old total spans:    {summary['old_total_spans']}")
    print(f"  New total spans:    {summary['new_total_spans']}")
    print(
        f"  Reduction:          {summary['reduction']} spans "
        f"({summary['reduction_pct']}%)"
    )

    print(f"\n  By vocabulary (before -> after):")
    for vc in comparison["by_vocabulary"]:
        print(
            f"    {vc['vocabulary']:20s}  {vc['old_count']:5d} -> "
            f"{vc['new_count']:5d}  ({vc['reduction']:+d}, "
            f"{vc['reduction_pct']:.1f}% reduction)"
        )

    print(f"\n  By type (after curation):")
    for stype, count in sorted(spans_by_type.items()):
        print(f"    {stype:20s}  {count}")

    print(f"\nOutput: {args.output_file}")
    print(f"Stats:  {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
