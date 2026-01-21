#!/usr/bin/env python3
"""Fix bare (unquoted) field values in training data."""

import json
import re
import sys
from pathlib import Path


def fix_bare_bibstem(query: str) -> str:
    """Quote unquoted bibstem values.
    
    Converts bibstem:ApJ to bibstem:"ApJ"
    """
    def quote_bibstem(match):
        value = match.group(1)
        return f'bibstem:"{value}"'
    
    # Match bibstem: followed by unquoted value (not already quoted)
    return re.sub(r'bibstem:(?!["\'])([a-zA-Z0-9&\.]+)', quote_bibstem, query)


def fix_bare_fields(query: str) -> str:
    """Fix all bare field values in a query."""
    query = fix_bare_bibstem(query)
    return query


def count_bare_bibstem(query: str) -> int:
    """Count bare bibstem values in a query."""
    return len(re.findall(r'bibstem:(?!["\'])([a-zA-Z0-9&\.]+)', query))


def fix_file(filepath: Path, dry_run: bool = False) -> dict:
    """Fix bare fields in a JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    total = len(data)
    fixed_count = 0
    fixed_bibstem = 0
    
    for pair in data:
        query = pair.get("ads_query", "")
        bare_count = count_bare_bibstem(query)
        
        if bare_count > 0:
            fixed_query = fix_bare_fields(query)
            if fixed_query != query:
                fixed_count += 1
                fixed_bibstem += bare_count
                pair["ads_query"] = fixed_query
    
    if not dry_run and fixed_count > 0:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    
    return {
        "filepath": str(filepath),
        "total": total,
        "fixed_pairs": fixed_count,
        "fixed_bibstem": fixed_bibstem,
    }


def main():
    """Fix bare fields in training data files."""
    project_root = Path(__file__).parent.parent
    
    files = [
        project_root / "data/datasets/processed/all_pairs.json",
        project_root / "data/datasets/raw/gold_examples.json",
    ]
    
    dry_run = "--dry-run" in sys.argv
    
    print("=" * 80)
    print("FIXING BARE FIELDS" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 80)
    
    total_fixed = 0
    total_bibstem = 0
    
    for filepath in files:
        if not filepath.exists():
            print(f"\n⚠️  File not found: {filepath}")
            continue
        
        print(f"\n📄 Processing {filepath.name}...")
        result = fix_file(filepath, dry_run=dry_run)
        
        print(f"  Total pairs: {result['total']}")
        print(f"  Fixed pairs: {result['fixed_pairs']}")
        print(f"  Fixed bibstem fields: {result['fixed_bibstem']}")
        
        total_fixed += result["fixed_pairs"]
        total_bibstem += result["fixed_bibstem"]
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total pairs fixed: {total_fixed}")
    print(f"Total bibstem fields quoted: {total_bibstem}")
    
    if dry_run:
        print("\n(Dry run - no files modified)")
    else:
        print("\n✓ Files updated successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
