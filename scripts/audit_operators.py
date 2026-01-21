#!/usr/bin/env python3
"""Audit training data for malformed operator syntax."""

import json
import re
import sys
from pathlib import Path

def find_bad_operators(query: str) -> list[tuple[str, str]]:
    """Find malformed operator patterns.
    
    Returns list of (operator, issue) tuples.
    """
    operators = ["citations", "trending", "useful", "reviews", "similar", "references"]
    issues = []
    
    for op in operators:
        # Pattern: operator(field:value without quotes)
        # Bad: citations(abs:topic) ✗
        # Good: citations(abs:"topic") ✓
        if re.search(rf'{op}\([a-z_]+:(?!["\(])[a-zA-Z]', query):
            issues.append((op, "missing quotes around field value"))
        
        # Pattern: operator(field:(value)) with extra parens
        # Bad: trending(abs:(exoplanets)) ✗
        # Good: trending(abs:"exoplanets") ✓
        if re.search(rf'{op}\([a-z_]+:\([^)]+\)\)', query):
            issues.append((op, "extra/malformed parentheses"))
        
        # Pattern: field:value without quotes inside operator
        # Already caught above but be explicit
    
    return issues

def audit_file(filepath: Path) -> dict:
    """Audit a JSON file for bad operator patterns."""
    with open(filepath) as f:
        data = json.load(f)
    
    bad_examples = []
    total = len(data)
    
    for i, pair in enumerate(data):
        query = pair.get("ads_query", "")
        nl = pair.get("natural_language", "")
        
        issues = find_bad_operators(query)
        if issues:
            bad_examples.append({
                "index": i,
                "nl": nl,
                "query": query,
                "issues": [f"{op} - {issue}" for op, issue in issues],
            })
    
    return {
        "filepath": str(filepath),
        "total": total,
        "bad_count": len(bad_examples),
        "percentage": f"{100 * len(bad_examples) / total:.1f}%" if total else "N/A",
        "examples": bad_examples,
    }

def main():
    """Audit training data files."""
    project_root = Path(__file__).parent.parent
    
    files = [
        project_root / "data/datasets/processed/all_pairs.json",
        project_root / "data/datasets/raw/gold_examples.json",
    ]
    
    print("=" * 80)
    print("OPERATOR SYNTAX AUDIT")
    print("=" * 80)
    
    all_results = {}
    total_bad = 0
    
    for filepath in files:
        if not filepath.exists():
            print(f"\n⚠️  File not found: {filepath}")
            continue
        
        print(f"\n📄 Auditing {filepath.name}...")
        result = audit_file(filepath)
        all_results[str(filepath)] = result
        
        print(f"  Total pairs: {result['total']}")
        print(f"  Bad operators: {result['bad_count']} ({result['percentage']})")
        total_bad += result['bad_count']
    
    # Show samples
    print("\n" + "=" * 80)
    print("SAMPLE BAD OPERATORS")
    print("=" * 80)
    
    shown = 0
    max_samples = 10
    
    for filepath, result in all_results.items():
        for ex in result["examples"]:
            if shown >= max_samples:
                break
            
            print(f"\n[{ex['index']}] {ex['nl']}")
            print(f"     Query: {ex['query']}")
            for issue in ex['issues']:
                print(f"     ✗ {issue}")
            
            shown += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total bad operator examples: {total_bad}")
    
    if total_bad == 0:
        print("\n✓ No malformed operators found - data looks good!")
        return 0
    else:
        print(f"\n⚠️  {total_bad} operator examples need fixing")
        print("\nTo fix, ensure operators follow this pattern:")
        print("  operator(field:\"value\") - not operator(field:value)")
        print("  operator(field:\"value\") - not operator(field:(value))")
        return 1

if __name__ == "__main__":
    sys.exit(main())
