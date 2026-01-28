#!/usr/bin/env python3
"""Generate the NER annotation dashboard by injecting real data into the HTML template.

Reads:
  - data/evaluation/ads_sample_reannotated.jsonl (auto-annotations from curated SWEET)
  - data/evaluation/ads_sample_predictions.jsonl (NER model predictions with confidence)
  - scripts/annotation_dashboard_template.html (dashboard template)

Produces:
  - data/evaluation/review_ner_annotations.html (fully functional dashboard)

Usage:
    python scripts/generate_annotation_dashboard.py
    python scripts/generate_annotation_dashboard.py --reannotated PATH --predictions PATH
    python scripts/generate_annotation_dashboard.py --output PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: Skipping malformed JSON at {path}:{line_num}: {exc}",
                    file=sys.stderr,
                )
    return records


def merge_data(
    reannotated: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge auto-annotations and model predictions into the dashboard data format.

    The template expects each record to have:
      - bibcode, title, abstract_clean, domain_category, citation_count
      - auto_spans: [{surface, start, end, type, canonical_id?, source_vocabulary?}]
      - model_spans: [{surface, start, end, type, confidence}]

    Returns a new list of merged records sorted by bibcode.
    """
    # Index predictions by bibcode
    pred_by_bibcode: dict[str, dict[str, Any]] = {}
    for rec in predictions:
        pred_by_bibcode[rec["bibcode"]] = rec

    merged: list[dict[str, Any]] = []
    for reann_rec in reannotated:
        bibcode = reann_rec["bibcode"]
        pred_rec = pred_by_bibcode.get(bibcode)

        record: dict[str, Any] = {
            "bibcode": bibcode,
            "title": reann_rec.get("title", ""),
            "abstract_clean": reann_rec.get("abstract_clean", ""),
            "domain_category": reann_rec.get("domain_category", "unknown"),
            "citation_count": reann_rec.get("citation_count", 0),
            "auto_spans": reann_rec.get("spans", []),
            "model_spans": pred_rec.get("spans", []) if pred_rec else [],
        }
        merged.append(record)

    merged.sort(key=lambda r: r["bibcode"])
    return merged


def inject_data_into_template(template_html: str, data: list[dict[str, Any]]) -> str:
    """Replace the DATA_PLACEHOLDER in the template with serialized JSON data.

    The template contains: const DATA = /*DATA_PLACEHOLDER*/[];
    We replace /*DATA_PLACEHOLDER*/[] with the JSON array.
    """
    placeholder = "/*DATA_PLACEHOLDER*/[]"
    if placeholder not in template_html:
        raise ValueError(
            f"Template does not contain expected placeholder: {placeholder}"
        )

    data_json = json.dumps(data, ensure_ascii=False)
    return template_html.replace(placeholder, data_json, 1)


def generate_dashboard(
    reannotated_path: Path,
    predictions_path: Path,
    template_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Generate the annotation dashboard HTML with embedded data.

    Returns a stats dict summarizing the generated dashboard.
    """
    # Load input data
    reannotated = load_jsonl(reannotated_path)
    predictions = load_jsonl(predictions_path)

    if not reannotated:
        raise ValueError(f"No records found in {reannotated_path}")
    if not predictions:
        raise ValueError(f"No records found in {predictions_path}")

    # Merge data
    merged = merge_data(reannotated, predictions)

    # Load template
    template_html = template_path.read_text(encoding="utf-8")

    # Inject data
    dashboard_html = inject_data_into_template(template_html, merged)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dashboard_html, encoding="utf-8")

    # Compute stats
    total_auto_spans = sum(len(r["auto_spans"]) for r in merged)
    total_model_spans = sum(len(r["model_spans"]) for r in merged)
    domains = sorted({r["domain_category"] for r in merged})

    auto_types: dict[str, int] = {}
    for r in merged:
        for s in r["auto_spans"]:
            span_type = s.get("type", "unknown")
            auto_types[span_type] = auto_types.get(span_type, 0) + 1

    model_types: dict[str, int] = {}
    for r in merged:
        for s in r["model_spans"]:
            span_type = s.get("type", "unknown")
            model_types[span_type] = model_types.get(span_type, 0) + 1

    stats: dict[str, Any] = {
        "total_abstracts": len(merged),
        "total_auto_spans": total_auto_spans,
        "total_model_spans": total_model_spans,
        "auto_spans_per_abstract": round(total_auto_spans / len(merged), 1),
        "model_spans_per_abstract": round(total_model_spans / len(merged), 1),
        "domains": domains,
        "auto_span_types": dict(sorted(auto_types.items())),
        "model_span_types": dict(sorted(model_types.items())),
        "output_file": str(output_path),
        "output_size_kb": round(output_path.stat().st_size / 1024, 1),
    }

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate NER annotation dashboard with embedded data"
    )
    parser.add_argument(
        "--reannotated",
        type=Path,
        default=Path("data/evaluation/ads_sample_reannotated.jsonl"),
        help="Path to reannotated abstracts JSONL (default: data/evaluation/ads_sample_reannotated.jsonl)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/evaluation/ads_sample_predictions.jsonl"),
        help="Path to model predictions JSONL (default: data/evaluation/ads_sample_predictions.jsonl)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("scripts/annotation_dashboard_template.html"),
        help="Path to HTML template (default: scripts/annotation_dashboard_template.html)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/review_ner_annotations.html"),
        help="Output HTML path (default: data/evaluation/review_ner_annotations.html)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the dashboard generation script."""
    args = parse_args(argv)

    print(f"Loading reannotated data from {args.reannotated}...")
    print(f"Loading predictions from {args.predictions}...")
    print(f"Using template: {args.template}")

    stats = generate_dashboard(
        reannotated_path=args.reannotated,
        predictions_path=args.predictions,
        template_path=args.template,
        output_path=args.output,
    )

    print(f"\nDashboard generated: {stats['output_file']}")
    print(f"  File size: {stats['output_size_kb']} KB")
    print(f"  Abstracts: {stats['total_abstracts']}")
    print(f"  Auto spans: {stats['total_auto_spans']} ({stats['auto_spans_per_abstract']} per abstract)")
    print(f"  Model spans: {stats['total_model_spans']} ({stats['model_spans_per_abstract']} per abstract)")
    print(f"  Domains: {', '.join(stats['domains'])}")
    print(f"  Auto span types: {stats['auto_span_types']}")
    print(f"  Model span types: {stats['model_span_types']}")
    print("\nOpen in a browser to start annotating.")


if __name__ == "__main__":
    main()
