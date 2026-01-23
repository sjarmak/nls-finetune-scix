#!/usr/bin/env python3
"""Audit training data for bare (unquoted) field values."""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

def find_bare_fields(query: str) -> list[tuple[str, str]]:
    """Find unquoted field values in a query.
    
    Returns list of (field_type, value) tuples for bare fields.
    """
    bare = []
    
    # author:value without quotes (but not ^author or author:(...) or author:"...")
    for match in re.finditer(r'author:(?!["\'^(])([a-zA-Z][a-zA-Z0-9\-\.]*)', query):
        bare.append(("author", match.group(1)))
    
    # abs:, title:, full:, keyword: without quotes
    for field in ["abs", "title", "full", "keyword"]:
        for match in re.finditer(
            rf'{field}:(?!["\(])([a-zA-Z][a-zA-Z0-9\-\s]*?)(?=\s(?:AND|OR|NOT)|$)',
            query
        ):
            bare.append((field, match.group(1)))
    
    # bibstem: without quotes
    for match in re.finditer(r'bibstem:(?!["\'])([a-zA-Z0-9\.]+)', query):
        bare.append(("bibstem", match.group(1)))
    
    return bare

def audit_file(filepath: Path) -> dict:
    """Audit a JSON file for bare fields."""
    with open(filepath) as f:
        data = json.load(f)
    
    issues = defaultdict(list)
    total = len(data)
    problematic = 0
    
    for i, pair in enumerate(data):
        query = pair.get("ads_query", "")
        nl = pair.get("natural_language", "")
        
        bare = find_bare_fields(query)
        if bare:
            problematic += 1
            issues[tuple(bare)].append({
                "index": i,
                "nl": nl,
                "query": query,
            })
    
    return {
        "filepath": str(filepath),
        "total": total,
        "problematic": problematic,
        "percentage": f"{100 * problematic / total:.1f}%" if total else "N/A",
        "issues_by_type": issues,
    }

def main():
    """Audit training data files."""
    project_root = Path(__file__).parent.parent
    
    files = [
        project_root / "data/datasets/processed/all_pairs.json",
        project_root / "data/datasets/raw/gold_examples.json",
    ]
    
    print("=" * 80)
    print("BARE FIELD AUDIT")
    print("=" * 80)
    
    all_issues = {}
    for filepath in files:
        if not filepath.exists():
            print(f"\n⚠️  File not found: {filepath}")
            continue
        
        print(f"\n📄 Auditing {filepath.name}...")
        result = audit_file(filepath)
        all_issues[str(filepath)] = result
        
        print(f"  Total pairs: {result['total']}")
        print(f"  Problematic: {result['problematic']} ({result['percentage']})")
        
        # Group by field type
        field_counts = defaultdict(int)
        for bare_fields, examples in result["issues_by_type"].items():
            for field_type, _ in bare_fields:
                field_counts[field_type] += len(examples)
        
        if field_counts:
            print(f"  Issues by field type:")
            for field_type in sorted(field_counts.keys()):
                count = field_counts[field_type]
                print(f"    - {field_type}: {count}")
    
    # Show sample issues
    print("\n" + "=" * 80)
    print("SAMPLE ISSUES")
    print("=" * 80)
    
    shown = 0
    max_samples = 5
    
    for filepath, result in all_issues.items():
        if shown >= max_samples:
            break
        
        for bare_fields, examples in result["issues_by_type"].items():
            if shown >= max_samples:
                break
            
            print(f"\n{bare_fields}:")
            for ex in examples[:2]:
                print(f"  [{ex['index']}] {ex['nl']}")
                print(f"       → {ex['query']}")
            
            shown += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_problematic = sum(r["problematic"] for r in all_issues.values())
    total_pairs = sum(r["total"] for r in all_issues.values())
    
    print(f"Total problematic pairs across all files: {total_problematic} / {total_pairs}")
    print(f"Percentage: {100 * total_problematic / total_pairs:.1f}%")
    
    if total_problematic == 0:
        print("\n✓ No bare fields found - data looks good!")
        return 0
    else:
        print(f"\n⚠️  {total_problematic} pairs need fixing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
