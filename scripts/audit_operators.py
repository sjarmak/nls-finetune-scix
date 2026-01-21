#!/usr/bin/env python3
"""Audit operator syntax in training data.

This script checks for malformed operator patterns like:
- citations(abs:cosmology) instead of citations(abs:"cosmology")
- trending(abs:(topic)) instead of trending(abs:"topic")  
- reviews(abs:magnetar) instead of reviews(abs:"magnetar")

Usage:
    python scripts/audit_operators.py [--verbose]
"""

import json
import re
import sys
from pathlib import Path

# Operators that take query arguments
OPERATORS = ["citations", "trending", "useful", "reviews", "similar", "references"]

# Pattern to find operators with unquoted abs: values
# Matches: operator(abs:word) but not operator(abs:"quoted")
UNQUOTED_ABS_PATTERN = re.compile(
    r'(' + '|'.join(OPERATORS) + r')\([^)]*abs:(?!["\'])([a-zA-Z0-9_]+)'
)

# Pattern to find operator at the start of query (indicates we're inside an operator context)
OPERATOR_START = re.compile(r'^(' + '|'.join(OPERATORS) + r')\(')

# Pattern to find malformed parentheses: abs:(topic)
MALFORMED_PARENS_PATTERN = re.compile(r'abs:\(([^)]+)\)')

# Pattern to find operators with unquoted title: values inside
UNQUOTED_TITLE_PATTERN = re.compile(
    r'(' + '|'.join(OPERATORS) + r')\([^)]*title:(?!["\'])([a-zA-Z0-9_]+)'
)


def find_operator_issues(query: str) -> list[dict]:
    """Find all operator syntax issues in a query."""
    issues = []
    
    # Check for unquoted abs: values inside operators
    for match in UNQUOTED_ABS_PATTERN.finditer(query):
        issues.append({
            "type": "unquoted_abs",
            "operator": match.group(1),
            "value": match.group(2),
            "match": match.group(0)
        })
    
    # Check for malformed parentheses inside operators (query starts with operator)
    if OPERATOR_START.match(query):
        for match in MALFORMED_PARENS_PATTERN.finditer(query):
            issues.append({
                "type": "malformed_parens",
                "value": match.group(1),
                "match": match.group(0)
            })
    
    return issues


def audit_file(filepath: Path, verbose: bool = False) -> dict:
    """Audit a JSON file for operator issues."""
    with open(filepath) as f:
        data = json.load(f)
    
    total = len(data)
    issues_by_type = {
        "unquoted_abs": [],
        "malformed_parens": [],
    }
    
    for i, pair in enumerate(data):
        query = pair.get("ads_query", pair.get("query", ""))
        nl = pair.get("natural_language", pair.get("nl", ""))
        issues = find_operator_issues(query)
        
        for issue in issues:
            issue["index"] = i
            issue["query"] = query
            issue["natural_language"] = nl[:80] + "..." if len(nl) > 80 else nl
            issues_by_type[issue["type"]].append(issue)
    
    return {
        "filepath": str(filepath),
        "total": total,
        "issues": issues_by_type,
    }


def print_issues(result: dict, verbose: bool = False):
    """Print audit results."""
    filepath = Path(result["filepath"]).name
    total_issues = sum(len(v) for v in result["issues"].values())
    
    print(f"\n📄 {filepath} ({result['total']} examples)")
    print("-" * 60)
    
    if total_issues == 0:
        print("  ✅ No operator issues found")
        return
    
    for issue_type, issues in result["issues"].items():
        if not issues:
            continue
        
        label = {
            "unquoted_abs": "Unquoted abs: inside operator",
            "malformed_parens": "Malformed parentheses abs:()",
        }.get(issue_type, issue_type)
        
        print(f"\n  ❌ {label}: {len(issues)}")
        
        if verbose:
            for issue in issues[:10]:  # Show first 10
                print(f"      Line {issue['index']}: {issue['query'][:70]}...")


def main():
    """Audit operator syntax in training data."""
    project_root = Path(__file__).parent.parent
    
    files = [
        project_root / "data/datasets/processed/all_pairs.json",
        project_root / "data/datasets/raw/gold_examples.json",
    ]
    
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    print("=" * 70)
    print("OPERATOR SYNTAX AUDIT")
    print("=" * 70)
    print("Checking for:")
    print("  - Unquoted values inside operators: citations(abs:word)")
    print("  - Malformed parentheses: abs:(topic)")
    
    total_issues = 0
    all_results = []
    
    for filepath in files:
        if not filepath.exists():
            print(f"\n⚠️  File not found: {filepath}")
            continue
        
        result = audit_file(filepath, verbose)
        all_results.append(result)
        print_issues(result, verbose)
        total_issues += sum(len(v) for v in result["issues"].values())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if total_issues == 0:
        print("✅ No operator syntax issues found!")
        return 0
    else:
        print(f"❌ Found {total_issues} operator syntax issues")
        print("\nRun: python scripts/fix_operators.py to fix these issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
