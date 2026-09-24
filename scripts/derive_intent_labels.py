#!/usr/bin/env python3
"""Derive IntentSpec-level gold labels from gold ADS queries.

The labels are read off the query string mechanically: the outermost operator
prefix, the values of the constrained enum fields (doctype, property,
bibgroup, database/collection), the first year range, the first-author caret,
the citation-count floor and the words in abs:/title: clauses. No semantics
are inferred. Operators that
IntentSpec cannot represent (topn) are labelled ``none`` and recorded in
``unsupported_operator`` so they can be reported separately.

Usage:
    uv run python scripts/derive_intent_labels.py \\
        --out data/datasets/evaluations/intent_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "finetune" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_benchmark import flatten_tests, load_benchmark  # noqa: E402
from intent_metrics import topic_words  # noqa: E402

from finetune.domains.scix.field_constraints import FIELD_ENUMS  # noqa: E402
from finetune.domains.scix.intent_spec import OPERATORS  # noqa: E402

DEFAULT_BENCHMARK = PROJECT_ROOT / "data/datasets/benchmark/benchmark_queries.json"
DEFAULT_VAL = PROJECT_ROOT / "data/datasets/processed/val.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data/datasets/evaluations/intent_labels.jsonl"

NONE = "none"
ENUM_FIELDS = {
    "doctype": "doctype",
    "property": "property",
    "bibgroup": "bibgroup",
    "database": "collection",
    "collection": "collection",
}
_OPERATOR_CALL = re.compile(r"\b([a-z_]+)\(")
_FIELD_VALUE = re.compile(
    r'\b(doctype|property|bibgroup|database|collection):(?:"([^"]+)"|\(([^)]*)\)|([^\s(),]+))'
)
_LIST_SEPARATOR = re.compile(r"\s+(?:OR|AND)\s+|\s+", re.IGNORECASE)
_YEAR_RANGE = re.compile(
    r"\b(?:year|pubdate):(?:\[\s*(\*|\d{4})[-\d]*\s+TO\s+(\*|\d{4})[-\d]*\s*\]"
    r"|(\d{4})(?:-\d{2})?(?:-(\d{4}))?\b)"
)
_FIRST_AUTHOR = re.compile(r'\b(?:author:\(?"\^|author:\^|first_author:)')
_CITATION_FLOOR = re.compile(r"\bcitation_count:\[\s*(\d+)\s+TO\s+\*\s*\]")
_TOPIC_CLAUSE = re.compile(r'\b(?:abs|title):(?:"([^"]*)"|\(([^)]*)\)|([^\s()"]+))')


@dataclass(frozen=True)
class IntentLabels:
    operator: str
    property: frozenset[str]
    doctype: frozenset[str]
    bibgroup: frozenset[str]
    collection: frozenset[str]
    has_year: bool
    has_author: bool
    unsupported_operator: str | None
    unparsed_values: tuple[str, ...]
    year_from: int | None = None
    year_to: int | None = None
    first_author: bool = False
    min_citations: int | None = None
    topic_tokens: frozenset[str] = frozenset()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["topic_tokens"] = sorted(self.topic_tokens)
        for key in ("property", "doctype", "bibgroup", "collection"):
            d[key] = sorted(d[key])
        d["unparsed_values"] = list(self.unparsed_values)
        return d


def _canonical(field: str, value: str) -> str | None:
    """Map a raw value onto the canonical enum spelling, or None if unknown."""
    legal = FIELD_ENUMS[field]
    if value in legal:
        return value
    lowered = value.lower()
    for candidate in legal:
        if candidate.lower() == lowered:
            return candidate
    return None


def _year_range(gold_query: str) -> tuple[int | None, int | None]:
    """First year:/pubdate: clause as (from, to); ``*`` is open. Months are dropped."""
    match = _YEAR_RANGE.search(gold_query)
    if match is None:
        return None, None
    low, high, single, single_end = match.groups()
    if single is not None:
        return int(single), int(single_end or single)
    return (None if low == "*" else int(low)), (None if high == "*" else int(high))


def derive_labels(gold_query: str) -> IntentLabels:
    """Read operator and enum labels off a gold ADS query string."""
    operator = NONE
    unsupported: str | None = None
    for name in _OPERATOR_CALL.findall(gold_query):
        if name in OPERATORS:
            operator = name
            break
        if unsupported is None and name not in ("pos",):
            unsupported = name

    values: dict[str, set[str]] = {
        "property": set(),
        "doctype": set(),
        "bibgroup": set(),
        "collection": set(),
    }
    unparsed: list[str] = []
    for raw_field, quoted, listed, bare in _FIELD_VALUE.findall(gold_query):
        field = ENUM_FIELDS[raw_field]
        raw_values = (
            [quoted] if quoted else (_LIST_SEPARATOR.split(listed.strip()) if listed else [bare])
        )
        for raw in raw_values:
            raw = raw.strip().strip('"')
            if not raw:
                continue
            canonical = _canonical(field, raw)
            if canonical is None:
                unparsed.append(f"{raw_field}:{raw}")
            else:
                values[field].add(canonical)

    year_from, year_to = _year_range(gold_query)
    floor = _CITATION_FLOOR.search(gold_query)
    return IntentLabels(
        operator=operator,
        property=frozenset(values["property"]),
        doctype=frozenset(values["doctype"]),
        bibgroup=frozenset(values["bibgroup"]),
        collection=frozenset(values["collection"]),
        has_year=bool(re.search(r"\b(year|pubdate):", gold_query)),
        has_author=bool(re.search(r"\b(author|first_author):", gold_query)),
        unsupported_operator=unsupported,
        unparsed_values=tuple(unparsed),
        year_from=year_from,
        year_to=year_to,
        first_author=bool(_FIRST_AUTHOR.search(gold_query)),
        min_citations=int(floor.group(1)) if floor else None,
        topic_tokens=topic_words(" ".join("".join(m) for m in _TOPIC_CLAUSE.findall(gold_query))),
    )


# -----------------------------------------------------------------------------
# Dataset loaders
# -----------------------------------------------------------------------------


def load_benchmark_items(path: Path = DEFAULT_BENCHMARK) -> list[dict]:
    """Benchmark items as {id, source, category, nl, gold_query, meta}."""
    items = []
    for test, category, subcategory in flatten_tests(load_benchmark(path)):
        gold = (test.get("expected_query") or "").strip()
        if not gold:
            continue
        items.append(
            {
                "id": f"bench-{test['id']}",
                "source": "benchmark",
                "category": f"{category}/{subcategory}",
                "nl": test["natural_language"],
                "gold_query": gold,
                "meta": {
                    k: test.get(k) for k in ("operator", "enum_field", "enum_value", "difficulty")
                },
            }
        )
    return items


def load_val_items(path: Path = DEFAULT_VAL) -> list[dict]:
    """val.jsonl chat rows as {id, source, category, nl, gold_query, meta}."""
    items = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            user = row["messages"][1]["content"]
            nl = user.split("\n", 1)[0].removeprefix("Query:").strip()
            gold = json.loads(row["messages"][2]["content"])["query"].strip()
            if not nl or not gold:
                continue
            items.append(
                {
                    "id": f"val-{index}",
                    "source": "val",
                    "category": "val",
                    "nl": nl,
                    "gold_query": gold,
                    "meta": {},
                }
            )
    return items


def label_items(items: list[dict]) -> list[dict]:
    return [{**item, "labels": derive_labels(item["gold_query"]).to_dict()} for item in items]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = label_items(load_benchmark_items(args.benchmark)) + label_items(load_val_items(args.val))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    unsupported = sum(1 for r in rows if r["labels"]["unsupported_operator"])
    unparsed = sum(1 for r in rows if r["labels"]["unparsed_values"])
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"  unsupported operator (labelled none): {unsupported}")
    print(f"  rows with enum values outside FIELD_ENUMS: {unparsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
