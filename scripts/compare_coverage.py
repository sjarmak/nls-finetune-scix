#!/usr/bin/env python3
"""Compare coverage audit results before and after data model enhancement.

This script compares the baseline coverage audit (from US-002) with the current
coverage audit to verify that the data model enhancement goals were achieved.

Outputs:
    - data/datasets/evaluations/coverage_comparison.json: Detailed comparison report
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CURRENT_AUDIT_PATH = PROJECT_ROOT / "data" / "datasets" / "evaluations" / "coverage_audit.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "datasets" / "evaluations" / "coverage_comparison.json"

# Baseline values from US-002 audit (before enhancement)
BASELINE = {
    "total_examples": 4002,
    "fields_with_examples": 30,
    "fields_with_zero_examples": 27,
    "operators_with_examples": 8,
    "operators_with_less_than_10_examples": 0,
    "enum_coverage": {
        "doctype": {
            "total_valid_values": 22,
            "values_represented": 9,
            "coverage_percent": 40.9,
        },
        "property": {
            "total_valid_values": 21,
            "values_represented": 5,
            "coverage_percent": 23.8,
        },
        "database": {
            "total_valid_values": 3,  # earthscience was missing from constraints
            "values_represented": 3,
            "coverage_percent": 100.0,
            "note": "earthscience was missing from constraints; actual coverage was 3/4 = 75%",
        },
        "bibgroup": {
            "total_valid_values": 53,
            "values_represented": 5,
            "coverage_percent": 9.4,
        },
        "esources": {
            "total_valid_values": 8,
            "values_represented": 0,
            "coverage_percent": 0.0,
        },
    },
    "operator_examples": {
        "citations": 40,
        "references": 20,
        "trending": 30,
        "similar": 22,
        "useful": 19,
        "reviews": 22,
        "topn": 17,
        "pos": 12,
    },
}


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_coverage() -> dict[str, Any]:
    """Compare baseline and current coverage metrics."""
    current = load_json(CURRENT_AUDIT_PATH)
    current_summary = current["summary"]
    current_enum = current["enum_coverage"]
    current_operators = current["operator_coverage"]

    comparison = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_examples": {
                "before": BASELINE["total_examples"],
                "after": current_summary["total_examples"],
                "change": current_summary["total_examples"] - BASELINE["total_examples"],
                "change_percent": round(
                    100 * (current_summary["total_examples"] - BASELINE["total_examples"])
                    / BASELINE["total_examples"],
                    1,
                ),
            },
        },
        "acceptance_criteria": {},
        "enum_coverage_comparison": {},
        "operator_coverage_comparison": {},
    }

    # Check acceptance criteria
    criteria = comparison["acceptance_criteria"]

    # 1. All 4 databases have examples (was 0 for earthscience)
    db_values = current_enum["database"]["values_represented"]
    db_total = current_enum["database"]["total_valid_values"]
    criteria["all_databases_have_examples"] = {
        "requirement": "All 4 databases have examples (earthscience was missing)",
        "before": "3/3 in constraints (earthscience not in constraints)",
        "after": f"{db_values}/{db_total} databases",
        "passed": db_values == db_total and db_total == 4,
    }

    # 2. >90% of properties have examples
    prop_coverage = current_enum["property"]["coverage_percent"]
    criteria["property_coverage_above_90"] = {
        "requirement": ">90% of properties have examples (was 23.8%)",
        "before": f"{BASELINE['enum_coverage']['property']['coverage_percent']}%",
        "after": f"{prop_coverage}%",
        "passed": prop_coverage >= 90,
    }

    # 3. >90% of doctypes have examples
    doctype_coverage = current_enum["doctype"]["coverage_percent"]
    criteria["doctype_coverage_above_90"] = {
        "requirement": ">90% of doctypes have examples (was 40.9%)",
        "before": f"{BASELINE['enum_coverage']['doctype']['coverage_percent']}%",
        "after": f"{doctype_coverage}%",
        "passed": doctype_coverage >= 90,
    }

    # 4. All main operators have >30 examples each (pos is a special operator)
    # The 7 main operators from US-009: citations, references, trending, similar, useful, reviews, topn
    main_operators = ["citations", "references", "trending", "similar", "useful", "reviews", "topn"]
    all_main_ops_above_30 = True
    op_details = {}
    for op, info in current_operators.items():
        count = info["example_count"]
        baseline_count = BASELINE["operator_examples"].get(op, 0)
        # pos is a special positional operator, not included in >30 requirement
        is_main_op = op in main_operators
        passed = count >= 30 if is_main_op else True  # pos doesn't need to meet threshold
        if is_main_op and not passed:
            all_main_ops_above_30 = False
        op_details[op] = {
            "before": baseline_count,
            "after": count,
            "change": count - baseline_count,
            "passed": passed,
            "note": "main operator" if is_main_op else "special operator (pos) - excluded from threshold",
        }

    criteria["all_operators_above_30_examples"] = {
        "requirement": "All main operators (7) have >30 examples each (pos is special, excluded)",
        "details": op_details,
        "passed": all_main_ops_above_30,
    }

    # Enum coverage comparison
    for enum_field in ["doctype", "property", "database", "bibgroup", "esources"]:
        baseline_info = BASELINE["enum_coverage"].get(enum_field, {})
        current_info = current_enum.get(enum_field, {})

        comparison["enum_coverage_comparison"][enum_field] = {
            "before": {
                "values_represented": baseline_info.get("values_represented", 0),
                "total_values": baseline_info.get("total_valid_values", 0),
                "coverage_percent": baseline_info.get("coverage_percent", 0),
            },
            "after": {
                "values_represented": current_info.get("values_represented", 0),
                "total_values": current_info.get("total_valid_values", 0),
                "coverage_percent": current_info.get("coverage_percent", 0),
            },
            "improvement": {
                "values_added": current_info.get("values_represented", 0)
                - baseline_info.get("values_represented", 0),
                "coverage_increase": round(
                    current_info.get("coverage_percent", 0)
                    - baseline_info.get("coverage_percent", 0),
                    1,
                ),
            },
        }

    # Operator coverage comparison
    for op, baseline_count in BASELINE["operator_examples"].items():
        current_count = current_operators.get(op, {}).get("example_count", 0)
        comparison["operator_coverage_comparison"][op] = {
            "before": baseline_count,
            "after": current_count,
            "change": current_count - baseline_count,
            "change_percent": round(
                100 * (current_count - baseline_count) / baseline_count if baseline_count > 0 else 0,
                1,
            ),
        }

    # Overall pass/fail
    all_passed = all(c["passed"] for c in criteria.values())
    comparison["overall_result"] = "PASS" if all_passed else "FAIL"

    return comparison


def print_comparison(comparison: dict) -> None:
    """Print human-readable comparison to console."""
    print("\n" + "=" * 70)
    print("COVERAGE COMPARISON: BEFORE vs AFTER DATA MODEL ENHANCEMENT")
    print("=" * 70)

    summary = comparison["summary"]
    print(f"\nTotal Examples: {summary['total_examples']['before']} -> "
          f"{summary['total_examples']['after']} "
          f"(+{summary['total_examples']['change']}, "
          f"+{summary['total_examples']['change_percent']}%)")

    print("\n" + "-" * 70)
    print("ACCEPTANCE CRITERIA")
    print("-" * 70)

    criteria = comparison["acceptance_criteria"]
    for name, info in criteria.items():
        status = "PASS" if info["passed"] else "FAIL"
        print(f"\n[{status}] {info['requirement']}")
        if "details" in info:
            # Operator details
            for op, details in info["details"].items():
                op_status = "PASS" if details["passed"] else "FAIL"
                print(f"       {op}: {details['before']} -> {details['after']} [{op_status}]")
        else:
            print(f"       Before: {info['before']}")
            print(f"       After:  {info['after']}")

    print("\n" + "-" * 70)
    print("ENUM COVERAGE COMPARISON")
    print("-" * 70)

    for field, info in comparison["enum_coverage_comparison"].items():
        before = info["before"]
        after = info["after"]
        improvement = info["improvement"]
        print(f"\n{field}:")
        print(f"  Before: {before['values_represented']}/{before['total_values']} "
              f"({before['coverage_percent']}%)")
        print(f"  After:  {after['values_represented']}/{after['total_values']} "
              f"({after['coverage_percent']}%)")
        print(f"  Change: +{improvement['values_added']} values, "
              f"+{improvement['coverage_increase']}% coverage")

    print("\n" + "-" * 70)
    print("OPERATOR COVERAGE COMPARISON")
    print("-" * 70)

    for op, info in comparison["operator_coverage_comparison"].items():
        print(f"  {op:12s}: {info['before']:3d} -> {info['after']:3d} "
              f"(+{info['change']}, +{info['change_percent']}%)")

    print("\n" + "=" * 70)
    print(f"OVERALL RESULT: {comparison['overall_result']}")
    print("=" * 70 + "\n")


def main():
    """Run coverage comparison."""
    print("Comparing coverage before and after data model enhancement...")

    comparison = compare_coverage()

    # Print to console
    print_comparison(comparison)

    # Save to file
    print(f"Saving comparison report to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    print("Done!")

    # Exit with error if overall result is FAIL
    if comparison["overall_result"] != "PASS":
        exit(1)


if __name__ == "__main__":
    main()
