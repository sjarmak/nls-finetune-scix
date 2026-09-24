#!/usr/bin/env python3
"""Mechanical checks on data/datasets/benchmark/heldout_paraphrases.json.

The set exists to measure generalisation beyond the regex patterns, so the
check asserts, per stratum, that the regex extractor does NOT already get the
labelled value:

- operator_positive: regex returns no operator (the phrasing is out-of-pattern)
- operator_negative: regex returns no operator (no false positive by construction)
- enum_synonym:      regex does not produce the labelled value on the target field
- ambiguous:         labels are schema-valid only

Also validates every label against IntentSpec / FIELD_ENUMS and reports the
review status. Exit code 1 on any failure so it can gate the evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "finetune" / "src"))

from finetune.domains.scix.field_constraints import FIELD_ENUMS  # noqa: E402
from finetune.domains.scix.intent_spec import OPERATORS  # noqa: E402
from finetune.domains.scix.ner import extract_intent  # noqa: E402

DEFAULT_PATH = PROJECT_ROOT / "data/datasets/benchmark/heldout_paraphrases.json"
STRATA = ("operator_positive", "operator_negative", "enum_synonym", "ambiguous")
SEARCH_KINDS = {"topic", "author", "object", "paper_reference", "identifier", "mixed"}
ENUM_LABEL_FIELDS = {
    "property": "property",
    "doctype": "doctype",
    "bibgroup": "bibgroup",
    "collection": "collection",
}


def load_heldout(path: Path = DEFAULT_PATH) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != "heldout-paraphrases-v1":
        raise ValueError(f"{path}: unexpected schema {doc.get('schema')!r}")
    return doc


def validate_item(item: dict) -> list[str]:
    """Schema and enum validity problems for one item (empty list when clean)."""
    problems = []
    if item.get("stratum") not in STRATA:
        problems.append(f"stratum {item.get('stratum')!r} not in {STRATA}")
    if not isinstance(item.get("nl"), str) or not item["nl"].strip():
        problems.append("empty nl")
    labels = item.get("labels") or {}
    if labels.get("operator") not in {"none", *OPERATORS}:
        problems.append(f"operator {labels.get('operator')!r} is not legal")
    for field in ENUM_LABEL_FIELDS:
        for value in labels.get(field, []):
            if value not in FIELD_ENUMS[field]:
                problems.append(f"{field} value {value!r} not in FIELD_ENUMS")
    if labels.get("search_kind") not in SEARCH_KINDS:
        problems.append(f"search_kind {labels.get('search_kind')!r} is not legal")
    for flag in ("needs_clarification", "refers_to_specific_paper"):
        if not isinstance(labels.get(flag), bool):
            problems.append(f"{flag} must be a boolean")
    if item.get("stratum") == "enum_synonym" and item.get("target_field") not in ENUM_LABEL_FIELDS:
        problems.append("enum_synonym item needs a target_field")
    return problems


def regex_problems(item: dict) -> list[str]:
    """Out-of-pattern assertions against the shipped regex extractor."""
    intent = extract_intent(item["nl"])
    stratum = item["stratum"]
    if stratum in ("operator_positive", "operator_negative") and intent.operator is not None:
        return [f"regex already returns operator={intent.operator!r}; item is in-pattern"]
    if stratum == "enum_synonym":
        field = item["target_field"]
        labelled = set(item["labels"][field])
        got = set(getattr(intent, field))
        if labelled & got:
            return [f"regex already maps to {field}={sorted(labelled & got)}; item is in-map"]
    return []


def run_checks(doc: dict) -> tuple[list[str], dict[str, int]]:
    failures: list[str] = []
    counts: dict[str, int] = dict.fromkeys(STRATA, 0)
    seen_ids: set[str] = set()
    for item in doc["items"]:
        item_id = item.get("id", "<no id>")
        if item_id in seen_ids:
            failures.append(f"{item_id}: duplicate id")
        seen_ids.add(item_id)
        problems = validate_item(item)
        if not problems:
            problems = regex_problems(item)
            counts[item["stratum"]] += 1
        failures.extend(f"{item_id}: {p}" for p in problems)
    return failures, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    doc = load_heldout(args.path)
    failures, counts = run_checks(doc)
    statuses = [i.get("review", {}).get("status") for i in doc["items"]]
    pending = sum(1 for s in statuses if s not in ("approved", "rejected"))
    rejected = statuses.count("rejected")
    print(f"items per stratum: {counts}")
    print(
        f"items not yet reviewed by a human: {pending} of {len(doc['items'])}"
        f" (rejected and excluded: {rejected})"
    )
    for failure in failures:
        print(f"FAIL {failure}")
    print("all mechanical checks passed" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
