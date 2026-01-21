#!/usr/bin/env python3
"""Fix operator syntax in training data.

This script fixes malformed operator patterns like:
- citations(abs:cosmology) → citations(abs:"cosmology")
- trending(abs:(topic)) → trending(abs:"topic")
- similar(abs:Fermi abs:"gamma ray") → similar(abs:"Fermi" abs:"gamma ray")

Usage:
    python scripts/fix_operators.py [--dry-run]
"""

import json
import re
import sys
from pathlib import Path

# Operators that take query arguments
OPERATORS = ["citations", "trending", "useful", "reviews", "similar", "references"]


def fix_unquoted_abs_in_operators(query: str) -> tuple[str, int]:
    """Quote unquoted abs: values inside operators.
    
    Converts: citations(abs:cosmology) → citations(abs:"cosmology")
    """
    fixes = 0
    
    def fix_operator(match):
        nonlocal fixes
        operator = match.group(1)
        args = match.group(2)
        
        # Fix unquoted abs: values in the args
        def quote_abs(abs_match):
            nonlocal fixes
            value = abs_match.group(1)
            fixes += 1
            return f'abs:"{value}"'
        
        # Match abs: followed by unquoted word (not already quoted)
        fixed_args = re.sub(r'abs:(?!["\'])([a-zA-Z0-9_]+)', quote_abs, args)
        
        return f'{operator}({fixed_args})'
    
    # Match operator(anything)
    pattern = r'(' + '|'.join(OPERATORS) + r')\(([^)]+)\)'
    result = re.sub(pattern, fix_operator, query)
    
    return result, fixes


def fix_malformed_parens_in_operators(query: str) -> tuple[str, int]:
    """Fix malformed parentheses in operators only.
    
    Converts: trending(abs:(exoplanets)) → trending(abs:"exoplanets")
    Only fixes when the query is wrapped in an operator.
    """
    fixes = 0
    
    # Only fix if query starts with an operator
    operator_start = re.match(r'^(' + '|'.join(OPERATORS) + r')\(', query)
    if not operator_start:
        return query, 0
    
    # Fix abs:(value) to abs:"value"
    def fix_parens(abs_match):
        nonlocal fixes
        value = abs_match.group(1)
        fixes += 1
        return f'abs:"{value}"'
    
    result = re.sub(r'abs:\(([^)]+)\)', fix_parens, query)
    
    return result, fixes


def fix_operator_syntax(query: str) -> tuple[str, dict]:
    """Fix all operator syntax issues in a query."""
    stats = {"unquoted_abs": 0, "malformed_parens": 0}
    
    # Fix malformed parens first (they may contain unquoted values)
    query, count = fix_malformed_parens_in_operators(query)
    stats["malformed_parens"] = count
    
    # Then fix unquoted values
    query, count = fix_unquoted_abs_in_operators(query)
    stats["unquoted_abs"] = count
    
    return query, stats


def fix_file(filepath: Path, dry_run: bool = False) -> dict:
    """Fix operator syntax in a JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    total = len(data)
    fixed_count = 0
    stats = {"unquoted_abs": 0, "malformed_parens": 0}
    examples = []
    
    for pair in data:
        query_key = "ads_query" if "ads_query" in pair else "query"
        query = pair.get(query_key, "")
        
        fixed_query, fix_stats = fix_operator_syntax(query)
        
        if fixed_query != query:
            fixed_count += 1
            stats["unquoted_abs"] += fix_stats["unquoted_abs"]
            stats["malformed_parens"] += fix_stats["malformed_parens"]
            
            if len(examples) < 5:
                examples.append({
                    "before": query,
                    "after": fixed_query,
                })
            
            pair[query_key] = fixed_query
    
    if not dry_run and fixed_count > 0:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    
    return {
        "filepath": str(filepath),
        "total": total,
        "fixed_pairs": fixed_count,
        "stats": stats,
        "examples": examples,
    }


def main():
    """Fix operator syntax in training data files."""
    project_root = Path(__file__).parent.parent
    
    files = [
        project_root / "data/datasets/processed/all_pairs.json",
        project_root / "data/datasets/raw/gold_examples.json",
    ]
    
    dry_run = "--dry-run" in sys.argv
    
    print("=" * 70)
    print("FIXING OPERATOR SYNTAX" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 70)
    
    total_fixed = 0
    total_unquoted = 0
    total_parens = 0
    
    for filepath in files:
        if not filepath.exists():
            print(f"\n⚠️  File not found: {filepath}")
            continue
        
        print(f"\n📄 Processing {filepath.name}...")
        result = fix_file(filepath, dry_run=dry_run)
        
        print(f"  Total pairs: {result['total']}")
        print(f"  Fixed pairs: {result['fixed_pairs']}")
        print(f"    - Unquoted abs: fixed: {result['stats']['unquoted_abs']}")
        print(f"    - Malformed parens fixed: {result['stats']['malformed_parens']}")
        
        if result["examples"]:
            print("\n  Examples:")
            for ex in result["examples"][:3]:
                print(f"    Before: {ex['before']}")
                print(f"    After:  {ex['after']}")
                print()
        
        total_fixed += result["fixed_pairs"]
        total_unquoted += result["stats"]["unquoted_abs"]
        total_parens += result["stats"]["malformed_parens"]
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total pairs fixed: {total_fixed}")
    print(f"Total unquoted abs: values fixed: {total_unquoted}")
    print(f"Total malformed parentheses fixed: {total_parens}")
    
    if dry_run:
        print("\n(Dry run - no files modified)")
    else:
        print("\n✓ Files updated successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
