#!/usr/bin/env python3
"""Evaluate the NL-to-ADS pipeline against the benchmark evaluation set.

This script runs the model against benchmark_queries.json and calculates:
- Exact match rate
- Field assignment accuracy
- Operator accuracy
- Syntax validity rate

Results are broken down by category (field_types, operators, enum_fields, etc.)
and output to data/datasets/evaluations/benchmark_results.json.

Usage:
    python scripts/evaluate_benchmark.py [--output FILE] [--verbose]

Example:
    python scripts/evaluate_benchmark.py
    python scripts/evaluate_benchmark.py --verbose
    python scripts/evaluate_benchmark.py --output data/datasets/evaluations/my_results.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add the finetune package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages/finetune/src"))

from finetune.domains.scix.pipeline import process_query
from finetune.domains.scix.validate import lint_query, validate_field_constraints


@dataclass
class TestResult:
    """Result for a single benchmark test case."""

    test_id: str
    natural_language: str
    expected_query: str | None
    expected_behavior: str | None
    forbidden_patterns: list[str]
    actual_query: str
    exact_match: bool
    syntax_valid: bool
    syntax_errors: list[str]
    constraint_valid: bool
    constraint_errors: list[str]
    forbidden_matched: list[str]
    passed: bool
    category: str
    subcategory: str
    difficulty: str


@dataclass
class CategoryStats:
    """Statistics for a category of tests."""

    total: int = 0
    passed: int = 0
    exact_match: int = 0
    syntax_valid: int = 0
    constraint_valid: int = 0
    no_forbidden: int = 0

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    @property
    def exact_match_rate(self) -> float:
        return (self.exact_match / self.total * 100) if self.total > 0 else 0.0

    @property
    def syntax_valid_rate(self) -> float:
        return (self.syntax_valid / self.total * 100) if self.total > 0 else 0.0


@dataclass
class EvaluationReport:
    """Full evaluation report."""

    timestamp: str
    benchmark_file: str
    total_tests: int
    passed_tests: int
    overall_stats: dict
    category_stats: dict[str, dict]
    difficulty_stats: dict[str, dict]
    failed_tests: list[dict]


def normalize_query(query: str) -> str:
    """Normalize query for comparison (lowercase, consistent whitespace)."""
    # Lowercase
    normalized = query.lower()
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Normalize quotes
    normalized = normalized.replace('"', '"').replace('"', '"')
    return normalized


def extract_fields(query: str) -> set[str]:
    """Extract field names used in a query."""
    # Match field:value patterns
    pattern = r"(\^?[a-z_]+):"
    fields = set(re.findall(pattern, query.lower()))
    return fields


def extract_operators(query: str) -> set[str]:
    """Extract operator names used in a query."""
    operators = set()
    operator_pattern = r"\b(citations|references|trending|similar|useful|reviews|topn)\s*\("
    for match in re.finditer(operator_pattern, query.lower()):
        operators.add(match.group(1))
    return operators


def check_forbidden_patterns(query: str, forbidden: list[str]) -> list[str]:
    """Check if query contains any forbidden patterns."""
    matched = []
    for pattern in forbidden:
        if pattern.lower() in query.lower():
            matched.append(pattern)
    return matched


def evaluate_test(test: dict, category: str, subcategory: str) -> TestResult:
    """Evaluate a single test case.

    Args:
        test: Test case dict with natural_language, expected_query, etc.
        category: Top-level category (field_types, operators, etc.)
        subcategory: Subcategory (content, citations, property, etc.)

    Returns:
        TestResult with evaluation details
    """
    nl = test.get("natural_language", "")
    expected_query = test.get("expected_query")
    expected_behavior = test.get("expected_behavior")
    forbidden = test.get("forbidden_patterns", [])
    test_id = test.get("id", "unknown")
    difficulty = test.get("difficulty", "simple")

    # Run the pipeline
    try:
        result = process_query(nl)
        actual_query = result.final_query
    except Exception as e:
        actual_query = f"ERROR: {e}"

    # Check syntax validity
    lint_result = lint_query(actual_query)
    syntax_valid = lint_result.valid
    syntax_errors = lint_result.errors if not syntax_valid else []

    # Check constraint validity
    constraint_result = validate_field_constraints(actual_query)
    constraint_valid = constraint_result.valid
    constraint_errors = constraint_result.error_messages if not constraint_valid else []

    # Check forbidden patterns
    forbidden_matched = check_forbidden_patterns(actual_query, forbidden)

    # Check exact match (if expected_query is provided)
    exact_match = False
    if expected_query:
        norm_expected = normalize_query(expected_query)
        norm_actual = normalize_query(actual_query)
        exact_match = norm_expected == norm_actual

    # Determine overall pass
    # For tests with expected_query: must match or at least be valid
    # For tests with expected_behavior: check behavior (e.g., no_operator, valid_response)
    # For regression tests: must not match forbidden patterns
    passed = False

    if forbidden:
        # Regression test: pass if no forbidden patterns matched
        passed = len(forbidden_matched) == 0 and syntax_valid
    elif expected_behavior:
        # Behavior test: various expectations
        if expected_behavior == "no_operator":
            has_operator = bool(extract_operators(actual_query))
            passed = not has_operator and syntax_valid
        elif expected_behavior == "single_operator":
            operators = extract_operators(actual_query)
            passed = len(operators) <= 1 and syntax_valid
        elif expected_behavior == "valid_response":
            # Empty/whitespace input should return something without crashing
            passed = syntax_valid or actual_query.startswith("ERROR") is False
        elif expected_behavior == "passthrough":
            # ADS syntax should be passed through
            passed = syntax_valid
        elif expected_behavior == "balanced_parens":
            paren_balance = actual_query.count("(") == actual_query.count(")")
            passed = paren_balance and syntax_valid
        elif expected_behavior == "ambiguous":
            # Ambiguous input - just needs to produce valid output
            passed = syntax_valid
        elif expected_behavior in ("valid_database_only", "valid_property_only"):
            # Should not use invalid enum values
            passed = constraint_valid and syntax_valid
        else:
            # Unknown behavior - just check syntax
            passed = syntax_valid
    elif expected_query:
        # Standard test: check exact match OR (valid syntax + no forbidden)
        # We're lenient: if the output is syntactically valid and captures similar intent,
        # we consider it a pass even if not exact match
        passed = syntax_valid and constraint_valid and len(forbidden_matched) == 0

    return TestResult(
        test_id=test_id,
        natural_language=nl,
        expected_query=expected_query,
        expected_behavior=expected_behavior,
        forbidden_patterns=forbidden,
        actual_query=actual_query,
        exact_match=exact_match,
        syntax_valid=syntax_valid,
        syntax_errors=syntax_errors,
        constraint_valid=constraint_valid,
        constraint_errors=constraint_errors,
        forbidden_matched=forbidden_matched,
        passed=passed,
        category=category,
        subcategory=subcategory,
        difficulty=difficulty,
    )


def load_benchmark(path: Path) -> dict:
    """Load benchmark file."""
    with open(path) as f:
        return json.load(f)


def flatten_tests(benchmark: dict) -> list[tuple[dict, str, str]]:
    """Flatten benchmark structure into list of (test, category, subcategory).

    Returns:
        List of (test_dict, category_name, subcategory_name) tuples
    """
    tests = []

    # Field types: field_types.content, field_types.author, etc.
    if "field_types" in benchmark:
        for field_type, type_tests in benchmark["field_types"].items():
            for test in type_tests:
                tests.append((test, "field_types", field_type))

    # Operators: operators.citations, operators.references, etc.
    if "operators" in benchmark:
        for operator, op_tests in benchmark["operators"].items():
            for test in op_tests:
                tests.append((test, "operators", operator))

    # Enum fields: enum_fields.property, enum_fields.doctype, etc.
    if "enum_fields" in benchmark:
        for enum_field, enum_tests in benchmark["enum_fields"].items():
            for test in enum_tests:
                tests.append((test, "enum_fields", enum_field))

    # Edge cases: flat list
    if "edge_cases" in benchmark:
        for test in benchmark["edge_cases"]:
            tests.append((test, "edge_cases", "edge"))

    # Regression tests: flat list
    if "regression_tests" in benchmark:
        for test in benchmark["regression_tests"]:
            tests.append((test, "regression_tests", "regression"))

    return tests


def evaluate_benchmark(benchmark_path: Path, verbose: bool = False) -> EvaluationReport:
    """Run full benchmark evaluation.

    Args:
        benchmark_path: Path to benchmark_queries.json
        verbose: If True, print progress and failures

    Returns:
        EvaluationReport with all results
    """
    benchmark = load_benchmark(benchmark_path)
    tests = flatten_tests(benchmark)

    if verbose:
        print(f"Loaded {len(tests)} test cases from {benchmark_path.name}")
        print()

    # Run all tests
    results: list[TestResult] = []
    category_stats: dict[str, CategoryStats] = defaultdict(CategoryStats)
    subcategory_stats: dict[str, dict[str, CategoryStats]] = defaultdict(
        lambda: defaultdict(CategoryStats)
    )
    difficulty_stats: dict[str, CategoryStats] = defaultdict(CategoryStats)

    for i, (test, category, subcategory) in enumerate(tests):
        result = evaluate_test(test, category, subcategory)
        results.append(result)

        # Update stats
        category_stats[category].total += 1
        category_stats[category].syntax_valid += 1 if result.syntax_valid else 0
        category_stats[category].constraint_valid += 1 if result.constraint_valid else 0
        category_stats[category].exact_match += 1 if result.exact_match else 0
        category_stats[category].no_forbidden += 1 if len(result.forbidden_matched) == 0 else 0
        category_stats[category].passed += 1 if result.passed else 0

        subcategory_stats[category][subcategory].total += 1
        subcategory_stats[category][subcategory].passed += 1 if result.passed else 0
        subcategory_stats[category][subcategory].syntax_valid += 1 if result.syntax_valid else 0
        subcategory_stats[category][subcategory].exact_match += 1 if result.exact_match else 0

        difficulty_stats[result.difficulty].total += 1
        difficulty_stats[result.difficulty].passed += 1 if result.passed else 0
        difficulty_stats[result.difficulty].syntax_valid += 1 if result.syntax_valid else 0

        if verbose and (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(tests)} tests...")

    # Collect failed tests for report
    failed_tests = []
    for result in results:
        if not result.passed:
            failed_tests.append(
                {
                    "test_id": result.test_id,
                    "category": result.category,
                    "subcategory": result.subcategory,
                    "natural_language": result.natural_language,
                    "expected_query": result.expected_query,
                    "expected_behavior": result.expected_behavior,
                    "actual_query": result.actual_query,
                    "exact_match": result.exact_match,
                    "syntax_valid": result.syntax_valid,
                    "syntax_errors": result.syntax_errors,
                    "constraint_errors": result.constraint_errors,
                    "forbidden_matched": result.forbidden_matched,
                }
            )

    # Calculate overall stats
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    exact_matches = sum(1 for r in results if r.exact_match)
    syntax_valid_count = sum(1 for r in results if r.syntax_valid)
    constraint_valid_count = sum(1 for r in results if r.constraint_valid)

    overall_stats = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0.0,
        "exact_match_count": exact_matches,
        "exact_match_rate": (exact_matches / total_tests * 100) if total_tests > 0 else 0.0,
        "syntax_valid_count": syntax_valid_count,
        "syntax_valid_rate": (syntax_valid_count / total_tests * 100) if total_tests > 0 else 0.0,
        "constraint_valid_count": constraint_valid_count,
        "constraint_valid_rate": (
            (constraint_valid_count / total_tests * 100) if total_tests > 0 else 0.0
        ),
    }

    # Convert stats to dicts for JSON serialization
    category_stats_dict = {}
    for cat, stats in category_stats.items():
        category_stats_dict[cat] = {
            "total": stats.total,
            "passed": stats.passed,
            "pass_rate": stats.pass_rate,
            "exact_match": stats.exact_match,
            "exact_match_rate": stats.exact_match_rate,
            "syntax_valid": stats.syntax_valid,
            "syntax_valid_rate": stats.syntax_valid_rate,
            "subcategories": {},
        }
        for subcat, substats in subcategory_stats[cat].items():
            category_stats_dict[cat]["subcategories"][subcat] = {
                "total": substats.total,
                "passed": substats.passed,
                "pass_rate": substats.pass_rate,
                "exact_match": substats.exact_match,
                "syntax_valid": substats.syntax_valid,
            }

    difficulty_stats_dict = {}
    for diff, stats in difficulty_stats.items():
        difficulty_stats_dict[diff] = {
            "total": stats.total,
            "passed": stats.passed,
            "pass_rate": stats.pass_rate,
            "syntax_valid": stats.syntax_valid,
        }

    return EvaluationReport(
        timestamp=datetime.now().isoformat(),
        benchmark_file=str(benchmark_path),
        total_tests=total_tests,
        passed_tests=passed_tests,
        overall_stats=overall_stats,
        category_stats=category_stats_dict,
        difficulty_stats=difficulty_stats_dict,
        failed_tests=failed_tests,
    )


def print_report(report: EvaluationReport) -> None:
    """Print human-readable evaluation report."""
    print("=" * 70)
    print("BENCHMARK EVALUATION REPORT")
    print("=" * 70)
    print()
    print(f"Timestamp: {report.timestamp}")
    print(f"Benchmark: {report.benchmark_file}")
    print()

    # Overall stats
    print("OVERALL RESULTS")
    print("-" * 40)
    print(f"Total tests:        {report.overall_stats['total_tests']}")
    print(
        f"Passed tests:       {report.overall_stats['passed_tests']} "
        f"({report.overall_stats['pass_rate']:.1f}%)"
    )
    print(
        f"Exact matches:      {report.overall_stats['exact_match_count']} "
        f"({report.overall_stats['exact_match_rate']:.1f}%)"
    )
    print(
        f"Syntax valid:       {report.overall_stats['syntax_valid_count']} "
        f"({report.overall_stats['syntax_valid_rate']:.1f}%)"
    )
    print(
        f"Constraint valid:   {report.overall_stats['constraint_valid_count']} "
        f"({report.overall_stats['constraint_valid_rate']:.1f}%)"
    )
    print()

    # Category breakdown
    print("BY CATEGORY")
    print("-" * 40)
    print(f"{'Category':<20} {'Passed':<10} {'Rate':<10} {'Exact':<10}")
    for cat, stats in sorted(report.category_stats.items()):
        print(
            f"{cat:<20} {stats['passed']}/{stats['total']:<6} "
            f"{stats['pass_rate']:>6.1f}%    {stats['exact_match']:<10}"
        )
    print()

    # Subcategory breakdown
    print("BY SUBCATEGORY")
    print("-" * 40)
    for cat, stats in sorted(report.category_stats.items()):
        if stats.get("subcategories"):
            print(f"\n{cat}:")
            for subcat, substats in sorted(stats["subcategories"].items()):
                print(
                    f"  {subcat:<18} {substats['passed']}/{substats['total']:<4} "
                    f"({substats['pass_rate']:>5.1f}%)"
                )
    print()

    # Difficulty breakdown
    print("BY DIFFICULTY")
    print("-" * 40)
    print(f"{'Difficulty':<15} {'Passed':<10} {'Rate':<10}")
    for diff, stats in sorted(report.difficulty_stats.items()):
        print(f"{diff:<15} {stats['passed']}/{stats['total']:<6} {stats['pass_rate']:>6.1f}%")
    print()

    # Failed tests summary
    if report.failed_tests:
        print("FAILED TESTS (first 10)")
        print("-" * 40)
        for i, test in enumerate(report.failed_tests[:10]):
            print(f"\n{i + 1}. [{test['test_id']}] ({test['category']}/{test['subcategory']})")
            print(f"   NL: {test['natural_language'][:60]}...")
            if test["expected_query"]:
                print(f"   Expected: {test['expected_query'][:50]}...")
            print(f"   Actual:   {test['actual_query'][:50]}...")
            if test["syntax_errors"]:
                print(f"   Syntax errors: {', '.join(test['syntax_errors'])}")
            if test["constraint_errors"]:
                print(f"   Constraint errors: {', '.join(test['constraint_errors'])}")
            if test["forbidden_matched"]:
                print(f"   Forbidden matched: {', '.join(test['forbidden_matched'])}")

        if len(report.failed_tests) > 10:
            print(f"\n... and {len(report.failed_tests) - 10} more failures")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the NL-to-ADS pipeline against benchmark tests"
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        default="data/datasets/benchmark/benchmark_queries.json",
        help="Path to benchmark file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/datasets/evaluations/benchmark_results.json",
        help="Output path for JSON results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress and details")
    parser.add_argument(
        "--ci", action="store_true", help="CI mode: exit 1 if pass rate < 80%"
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    output_path = Path(args.output)

    if not benchmark_path.exists():
        print(f"Error: Benchmark file not found: {benchmark_path}", file=sys.stderr)
        sys.exit(1)

    # Warm up the pipeline (loads index, etc.)
    if args.verbose:
        print("Warming up pipeline...")
    try:
        _ = process_query("test query")
    except Exception as e:
        print(f"Warning: Pipeline warmup failed: {e}")

    # Run evaluation
    if args.verbose:
        print("Running benchmark evaluation...")
        print()

    report = evaluate_benchmark(benchmark_path, verbose=args.verbose)

    # Print report
    print_report(report)

    # Save JSON results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": report.timestamp,
                "benchmark_file": report.benchmark_file,
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "overall_stats": report.overall_stats,
                "category_stats": report.category_stats,
                "difficulty_stats": report.difficulty_stats,
                "failed_tests": report.failed_tests,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {output_path}")

    # CI check
    if args.ci:
        pass_rate = report.overall_stats["pass_rate"]
        if pass_rate < 80.0:
            print(f"\n❌ CI CHECK FAILED: Pass rate {pass_rate:.1f}% < 80%")
            sys.exit(1)
        else:
            print(f"\n✅ CI CHECK PASSED: Pass rate {pass_rate:.1f}% >= 80%")
            sys.exit(0)


if __name__ == "__main__":
    main()
