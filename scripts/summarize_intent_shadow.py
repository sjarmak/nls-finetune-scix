#!/usr/bin/env python3
"""Summarize intent shadow rows from the hybrid server's TELEMETRY_LOG.

With SHADOW_INTENT_BACKEND set, docker/server.py appends one
``record_type: "intent_shadow"`` row per pipeline request, comparing the
served regex IntentSpec with the shadow Jev backend's. This script counts
the rows, the Jev calls and errors, the disagreements per field, and lists
the disagreeing queries with both values. It aggregates only; it makes no
judgement about which side is right.

Usage:
    uv run python scripts/summarize_intent_shadow.py telemetry.jsonl
    uv run python scripts/summarize_intent_shadow.py telemetry.jsonl --json
"""

import argparse
import json
import sys
from pathlib import Path

from finetune.domains.scix.intent_shadow import SHADOW_COMPARED_FIELDS, SHADOW_RECORD_TYPE

REQUIRED_FIELDS: tuple[str, ...] = (
    "nl_query",
    "served_path",
    "served_intent",
    "shadow_intent",
    "disagreements",
    "classifier_called",
    "classifier_cached",
    "classifier_error",
    "classifier_operator_confidence",
)


def load_shadow_rows(path: Path) -> list[dict]:
    """Shadow rows from a telemetry JSONL file; other record types are skipped."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{lineno}: not valid JSON") from error
            if not isinstance(row, dict) or row.get("record_type") != SHADOW_RECORD_TYPE:
                continue
            missing = [name for name in REQUIRED_FIELDS if name not in row]
            if missing:
                raise ValueError(f"{path}:{lineno}: shadow row missing {missing}")
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Counts per field and the disagreeing queries, in log order.

    Disagreements are counted only on rows the pipeline served. When a request
    fell back to the model the regex intent never reached the user, so those
    rows are counted separately and not compared.
    """
    by_field = {name: 0 for name in SHADOW_COMPARED_FIELDS}
    disagreeing_queries = []
    served_by_pipeline = [row for row in rows if row["served_path"] == "pipeline"]
    for row in served_by_pipeline:
        fields = row["disagreements"]
        for name in fields:
            by_field[name] = by_field.get(name, 0) + 1
        if fields:
            disagreeing_queries.append(
                {
                    "nl_query": row["nl_query"],
                    "disagreements": fields,
                    "served": {name: row["served_intent"].get(name) for name in fields},
                    "shadow": {name: row["shadow_intent"].get(name) for name in fields},
                    "classifier_operator_confidence": row["classifier_operator_confidence"],
                }
            )
    return {
        "shadow_rows": len(rows),
        "served_by_model": len(rows) - len(served_by_pipeline),
        "classifier_called": sum(1 for row in rows if row["classifier_called"]),
        "classifier_billed": sum(
            1 for row in rows if row["classifier_called"] and not row["classifier_cached"]
        ),
        "classifier_errors": sum(1 for row in rows if row["classifier_error"]),
        "disagreeing": len(disagreeing_queries),
        "by_field": by_field,
        "disagreeing_queries": disagreeing_queries,
    }


def format_summary(summary: dict) -> str:
    lines = [
        f"shadow rows: {summary['shadow_rows']}",
        f"served by the model (not compared): {summary['served_by_model']}",
        f"classifier called: {summary['classifier_called']}"
        f" (billed, not from cache: {summary['classifier_billed']})",
        f"classifier errors: {summary['classifier_errors']}",
        f"disagreeing (pipeline-served rows): {summary['disagreeing']}",
        "disagreements by field:",
        *(f"  {name}: {count}" for name, count in summary["by_field"].items()),
    ]
    if summary["disagreeing_queries"]:
        lines.append("disagreeing queries:")
    for item in summary["disagreeing_queries"]:
        confidence = item["classifier_operator_confidence"]
        shown = "n/a" if confidence is None else f"{confidence:.2f}"
        lines.append(f"- {item['nl_query']!r} (jev operator confidence {shown})")
        for name in item["disagreements"]:
            lines.append(
                f"    {name}: served={item['served'][name]!r} shadow={item['shadow'][name]!r}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry", type=Path, help="TELEMETRY_LOG JSONL file")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    summary = summarize(load_shadow_rows(args.telemetry))
    print(json.dumps(summary, indent=2) if args.json else format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
