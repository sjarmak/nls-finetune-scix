#!/usr/bin/env python3
"""Audit training data coverage against the complete ADS field inventory.

This script analyzes gold_examples.json to measure how well the training data
covers all ADS fields, operators, and enum values documented in the field inventory.

Outputs:
    - data/datasets/evaluations/coverage_audit.json: Detailed coverage report
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_MODEL_PATH = PROJECT_ROOT / "data" / "model" / "ads_field_inventory.json"
GOLD_EXAMPLES_PATH = PROJECT_ROOT / "data" / "datasets" / "raw" / "gold_examples.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "datasets" / "evaluations" / "coverage_audit.json"

# Add packages to path for field_constraints
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "finetune" / "src"))
from finetune.domains.scix.field_constraints import (
    BIBGROUPS,
    DATABASES,
    DOCTYPES,
    ESOURCES,
    PROPERTIES,
)


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_fields_from_query(query: str) -> dict[str, list[str]]:
    """Extract field:value pairs from an ADS query string.

    Returns dict mapping field names to list of values used.
    """
    fields: dict[str, list[str]] = defaultdict(list)

    # Pattern to match field:value or field:"value" or field:(value)
    # Handles various ADS query syntaxes
    field_pattern = r"(\w+):(?:\"([^\"]+)\"|(\([^)]+\))|(\[[^\]]+\])|(\S+))"

    for match in re.finditer(field_pattern, query):
        field_name = match.group(1).lower()
        # Get the value from whichever capture group matched
        value = match.group(2) or match.group(3) or match.group(4) or match.group(5)
        if value:
            # Clean up parentheses and quotes
            value = value.strip("()\"'")
            fields[field_name].append(value)

    return dict(fields)


def extract_operators_from_query(query: str) -> list[str]:
    """Extract operator names from an ADS query string.

    Returns list of operator names found (e.g., ['citations', 'references']).
    """
    operators = []

    # Pattern to match operator(...)
    operator_pattern = r"\b(citations|references|trending|similar|useful|reviews|topn|pos)\s*\("

    for match in re.finditer(operator_pattern, query, re.IGNORECASE):
        operators.append(match.group(1).lower())

    return operators


def extract_trigger_phrases(
    nl_text: str, operators_in_query: list[str]
) -> dict[str, list[str]]:
    """Attempt to identify trigger phrases in NL that led to operators.

    This is heuristic-based to help understand what NL patterns trigger operators.
    """
    trigger_patterns: dict[str, list[str]] = {
        "citations": [
            r"papers? (that )?cit(e|ing|ed)",
            r"cit(e|ing|ed) (by|papers?)",
            r"cited by",
            r"citations? (of|to)",
        ],
        "references": [
            r"references? (of|by|in|from)",
            r"cited in",
            r"what (does|did) .+ cite",
            r"bibliography",
            r"sources? cited",
        ],
        "trending": [
            r"trending",
            r"popular",
            r"hot (papers?|topics?)",
            r"what's hot",
        ],
        "similar": [
            r"similar (to|papers?)",
            r"like",
            r"related (to|papers?)",
            r"resembl",
        ],
        "useful": [
            r"useful",
            r"helpful",
            r"foundational",
            r"essential",
            r"important",
        ],
        "reviews": [
            r"review (articles?|papers?)?",
            r"survey",
            r"overviews?",
            r"comprehensive",
        ],
        "topn": [
            r"top \d+",
            r"most cited",
            r"highest cited",
            r"best papers?",
        ],
    }

    found_triggers: dict[str, list[str]] = defaultdict(list)
    nl_lower = nl_text.lower()

    for op in operators_in_query:
        if op in trigger_patterns:
            for pattern in trigger_patterns[op]:
                if re.search(pattern, nl_lower):
                    # Extract the matching phrase
                    match = re.search(pattern, nl_lower)
                    if match:
                        found_triggers[op].append(match.group(0))

    return dict(found_triggers)


def audit_coverage(
    examples: list[dict], inventory: dict
) -> dict[str, Any]:
    """Perform comprehensive coverage audit.

    Args:
        examples: List of gold examples with natural_language and ads_query
        inventory: Field inventory from ads_field_inventory.json

    Returns:
        Coverage audit report dict
    """
    # Initialize counters
    field_usage: dict[str, Counter] = defaultdict(Counter)  # field -> value -> count
    field_example_count: Counter = Counter()  # field -> number of examples
    operator_usage: Counter = Counter()  # operator -> count
    operator_triggers: dict[str, Counter] = defaultdict(Counter)  # operator -> trigger phrase -> count
    category_counts: Counter = Counter()  # category -> count

    # Enum value tracking
    enum_usage = {
        "doctype": Counter(),
        "property": Counter(),
        "database": Counter(),
        "bibgroup": Counter(),
        "esources": Counter(),
    }

    # Valid enum values from constraints
    valid_enums = {
        "doctype": DOCTYPES,
        "property": PROPERTIES,
        "database": DATABASES,
        "bibgroup": BIBGROUPS,
        "esources": ESOURCES,
    }

    # Process each example
    for example in examples:
        query = example.get("ads_query", "")
        nl = example.get("natural_language", "")
        category = example.get("category", "unknown")

        category_counts[category] += 1

        # Extract fields
        fields = extract_fields_from_query(query)
        for field, values in fields.items():
            field_example_count[field] += 1
            for value in values:
                field_usage[field][value] += 1

                # Track enum usage
                if field in enum_usage:
                    # Normalize to match constraint format
                    normalized_value = value.strip().lower()
                    # Try to match against valid values
                    matched = False
                    for valid_val in valid_enums[field]:
                        if valid_val.lower() == normalized_value:
                            enum_usage[field][valid_val] += 1
                            matched = True
                            break
                    if not matched:
                        # Record as unknown/invalid
                        enum_usage[field][f"[unknown:{value}]"] += 1

        # Extract operators
        operators = extract_operators_from_query(query)
        for op in operators:
            operator_usage[op] += 1

        # Try to identify trigger phrases
        if operators:
            triggers = extract_trigger_phrases(nl, operators)
            for op, phrases in triggers.items():
                for phrase in phrases:
                    operator_triggers[op][phrase] += 1

    # Build field coverage report
    inventory_fields = set(inventory.get("fields", {}).keys())
    field_groups = inventory.get("field_groups", {})

    field_coverage = {}
    for field in sorted(inventory_fields):
        field_info = inventory["fields"].get(field, {})
        example_count = field_example_count.get(field, 0)
        unique_values = list(field_usage.get(field, {}).keys())

        field_coverage[field] = {
            "example_count": example_count,
            "unique_values_used": len(unique_values),
            "sample_values": unique_values[:10],  # First 10 unique values
            "field_type": field_info.get("type", "unknown"),
            "field_group": field_info.get("group", "unknown"),
        }

    # Build operator coverage report
    inventory_operators = list(inventory.get("operators", {}).keys())
    operator_coverage = {}
    for op in inventory_operators:
        example_count = operator_usage.get(op, 0)
        trigger_counts = dict(operator_triggers.get(op, {}))

        operator_coverage[op] = {
            "example_count": example_count,
            "trigger_phrases": trigger_counts,
            "unique_trigger_variations": len(trigger_counts),
        }

    # Build enum coverage report
    # Also get inventory valid values for comparison
    inventory_enum_values = {}
    for field_name, field_info in inventory.get("fields", {}).items():
        if field_info.get("type") == "enum" and "valid_values" in field_info:
            inventory_enum_values[field_name] = set(field_info["valid_values"])

    enum_coverage = {}
    for enum_field, valid_values in valid_enums.items():
        used_values = set(enum_usage[enum_field].keys()) - {
            v for v in enum_usage[enum_field].keys() if v.startswith("[unknown:")
        }
        unused_values = set(valid_values) - used_values
        unknown_values = [
            v for v in enum_usage[enum_field].keys() if v.startswith("[unknown:")
        ]

        # Check for values in inventory but not in constraints
        inventory_vals = inventory_enum_values.get(enum_field, set())
        constraint_vals = set(valid_values)
        missing_from_constraints = inventory_vals - constraint_vals
        missing_from_inventory = constraint_vals - inventory_vals

        enum_coverage[enum_field] = {
            "total_valid_values": len(valid_values),
            "values_represented": len(used_values),
            "coverage_percent": round(100 * len(used_values) / len(valid_values), 1) if valid_values else 0,
            "used_values": {v: enum_usage[enum_field].get(v, 0) for v in used_values},
            "unused_values": sorted(unused_values),
            "unknown_values": unknown_values,
            "inventory_vs_constraints_diff": {
                "in_inventory_not_constraints": sorted(missing_from_constraints),
                "in_constraints_not_inventory": sorted(missing_from_inventory),
            } if missing_from_constraints or missing_from_inventory else None,
        }

    # Identify gaps
    fields_with_zero_examples = [
        field for field, info in field_coverage.items() if info["example_count"] == 0
    ]
    operators_with_few_examples = [
        op for op, info in operator_coverage.items() if info["example_count"] < 10
    ]

    # Summary statistics
    summary = {
        "total_examples": len(examples),
        "total_fields_in_inventory": len(inventory_fields),
        "fields_with_examples": len([f for f in field_coverage if field_coverage[f]["example_count"] > 0]),
        "fields_with_zero_examples": len(fields_with_zero_examples),
        "total_operators": len(inventory_operators),
        "operators_with_examples": len([o for o in operator_coverage if operator_coverage[o]["example_count"] > 0]),
        "operators_with_less_than_10_examples": len(operators_with_few_examples),
        "category_distribution": dict(category_counts.most_common()),
    }

    # Enum summary
    for enum_field in enum_coverage:
        summary[f"{enum_field}_coverage"] = enum_coverage[enum_field]["coverage_percent"]

    return {
        "summary": summary,
        "field_coverage": field_coverage,
        "operator_coverage": operator_coverage,
        "enum_coverage": enum_coverage,
        "gaps": {
            "fields_with_zero_examples": sorted(fields_with_zero_examples),
            "operators_with_less_than_10_examples": sorted(operators_with_few_examples),
            "enum_values_never_used": {
                field: info["unused_values"]
                for field, info in enum_coverage.items()
                if info["unused_values"]
            },
        },
    }


def print_summary(audit: dict) -> None:
    """Print human-readable summary to console."""
    summary = audit["summary"]
    gaps = audit["gaps"]

    print("\n" + "=" * 60)
    print("TRAINING DATA COVERAGE AUDIT")
    print("=" * 60)

    print(f"\nTotal examples: {summary['total_examples']}")
    print(f"\nField Coverage:")
    print(f"  - Fields in inventory: {summary['total_fields_in_inventory']}")
    print(f"  - Fields with examples: {summary['fields_with_examples']}")
    print(f"  - Fields with ZERO examples: {summary['fields_with_zero_examples']}")

    print(f"\nOperator Coverage:")
    print(f"  - Operators in inventory: {summary['total_operators']}")
    print(f"  - Operators with examples: {summary['operators_with_examples']}")
    print(f"  - Operators with <10 examples: {summary['operators_with_less_than_10_examples']}")

    print(f"\nEnum Field Coverage:")
    for enum_field in ["doctype", "property", "database", "bibgroup"]:
        coverage = summary.get(f"{enum_field}_coverage", 0)
        info = audit["enum_coverage"].get(enum_field, {})
        used = info.get("values_represented", 0)
        total = info.get("total_valid_values", 0)
        print(f"  - {enum_field}: {used}/{total} values ({coverage}%)")

    print(f"\nCategory Distribution:")
    for cat, count in summary["category_distribution"].items():
        print(f"  - {cat}: {count}")

    print("\n" + "-" * 60)
    print("GAPS IDENTIFIED")
    print("-" * 60)

    if gaps["fields_with_zero_examples"]:
        print(f"\nFields with 0 examples ({len(gaps['fields_with_zero_examples'])}):")
        for field in gaps["fields_with_zero_examples"][:15]:
            print(f"  - {field}")
        if len(gaps["fields_with_zero_examples"]) > 15:
            print(f"  ... and {len(gaps['fields_with_zero_examples']) - 15} more")

    if gaps["operators_with_less_than_10_examples"]:
        print(f"\nOperators with <10 examples ({len(gaps['operators_with_less_than_10_examples'])}):")
        for op in gaps["operators_with_less_than_10_examples"]:
            count = audit["operator_coverage"].get(op, {}).get("example_count", 0)
            print(f"  - {op}: {count} examples")

    if gaps["enum_values_never_used"]:
        print("\nEnum values never used in training data:")
        for field, values in gaps["enum_values_never_used"].items():
            print(f"\n  {field} ({len(values)} unused):")
            for v in values[:10]:
                print(f"    - {v}")
            if len(values) > 10:
                print(f"    ... and {len(values) - 10} more")

    # Check for inventory vs constraints differences
    print("\n" + "-" * 60)
    print("INVENTORY VS CONSTRAINTS DIFFERENCES")
    print("-" * 60)
    has_diffs = False
    for field, info in audit["enum_coverage"].items():
        diff = info.get("inventory_vs_constraints_diff")
        if diff:
            has_diffs = True
            if diff.get("in_inventory_not_constraints"):
                print(f"\n  {field}: Values in inventory but NOT in field_constraints.py:")
                for v in diff["in_inventory_not_constraints"]:
                    print(f"    - {v} (needs to be added to FIELD_ENUMS)")
            if diff.get("in_constraints_not_inventory"):
                print(f"\n  {field}: Values in field_constraints.py but NOT in inventory:")
                for v in diff["in_constraints_not_inventory"]:
                    print(f"    - {v}")
    if not has_diffs:
        print("\n  No differences found between inventory and constraints.")

    print("\n" + "=" * 60)


def main():
    """Run the coverage audit."""
    print("Loading field inventory...")
    inventory = load_json(DATA_MODEL_PATH)

    print("Loading gold examples...")
    examples = load_json(GOLD_EXAMPLES_PATH)
    print(f"Loaded {len(examples)} examples")

    print("Running coverage audit...")
    audit = audit_coverage(examples, inventory)

    # Print summary to console
    print_summary(audit)

    # Save full report
    print(f"\nSaving detailed report to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(audit, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
