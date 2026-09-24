#!/usr/bin/env python3
"""Record a human review of data/datasets/benchmark/heldout_paraphrases.json.

Every item not named in --reject is stamped approved; rejected items are kept
in the file with status "rejected" and are skipped by the evaluation loader.
Edit an item's text or labels in the JSON first if it needs a fix, then run
this. Example:

    uv run python scripts/approve_heldout_paraphrases.py --reviewer Stephanie \\
        --reject hp-neg-017 hp-amb-004 --note hp-neg-017="operator word is an instruction here"
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path

DEFAULT_PATH = Path("data/datasets/benchmark/heldout_paraphrases.json")


def parse_notes(raw: list[str]) -> dict[str, str]:
    notes: dict[str, str] = {}
    for entry in raw:
        item_id, sep, text = entry.partition("=")
        if not sep or not item_id:
            raise SystemExit(f"--note expects ID=text, got {entry!r}")
        notes[item_id] = text
    return notes


def apply_review(doc: dict, reviewer: str, reject: list[str], notes: dict[str, str]) -> dict:
    """Return a new document with every item's review block filled in."""
    ids = {item["id"] for item in doc["items"]}
    unknown = sorted((set(reject) | set(notes)) - ids)
    if unknown:
        raise SystemExit(f"unknown item id(s): {', '.join(unknown)}")
    rejected = set(reject)
    out = copy.deepcopy(doc)
    for item in out["items"]:
        status = "rejected" if item["id"] in rejected else "approved"
        item["review"] = {"status": status, "reviewer": reviewer, "note": notes.get(item["id"], "")}
    approved = len(out["items"]) - len(rejected)
    today = dt.date.today().isoformat()
    out["review_status"] = (
        f"reviewed by {reviewer} on {today}: {approved} approved, {len(rejected)} rejected"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reject", nargs="*", default=[], metavar="ID")
    parser.add_argument("--note", nargs="*", default=[], metavar="ID=text")
    args = parser.parse_args(argv)
    doc = json.loads(args.path.read_text(encoding="utf-8"))
    out = apply_review(doc, args.reviewer, args.reject, parse_notes(args.note))
    args.path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out["review_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
