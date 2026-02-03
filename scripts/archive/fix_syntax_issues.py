#!/usr/bin/env python3
"""Fix remaining syntax issues in gold_examples.json.

Issues to fix:
1. Unbalanced brackets/parentheses - remove these malformed examples
2. facility: field - change to abs: (facility is not a valid ADS field)
3. The object: coordinate syntax is actually valid - no fix needed there
"""

import json
import re


def has_unbalanced_brackets(query: str) -> bool:
    """Check if query has unbalanced brackets or parentheses."""
    # Count parentheses
    open_parens = query.count('(')
    close_parens = query.count(')')

    # Count brackets
    open_brackets = query.count('[')
    close_brackets = query.count(']')

    return open_parens != close_parens or open_brackets != close_brackets


def fix_facility_field(query: str) -> str:
    """Fix facility: field to use abs: instead."""
    # facility: is not a valid ADS field, use abs: for the content
    return re.sub(r'facility:', 'abs:', query)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix syntax issues in gold examples")
    parser.add_argument("--input", "-i", default="data/datasets/raw/gold_examples.json")
    parser.add_argument("--output", "-o", default="data/datasets/raw/gold_examples.json")
    parser.add_argument("--dry-run", "-n", action="store_true")

    args = parser.parse_args()

    # Load
    with open(args.input) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")

    # Process
    fixed_examples = []
    removed = []
    fixed = []

    for i, example in enumerate(examples):
        query = example.get("ads_query", "")
        original_query = query

        # Check for unbalanced brackets
        if has_unbalanced_brackets(query):
            removed.append({
                "index": i,
                "nl": example.get("natural_language", ""),
                "query": query,
                "reason": "Unbalanced brackets/parentheses"
            })
            continue

        # Fix facility: field
        if "facility:" in query:
            query = fix_facility_field(query)
            example = example.copy()
            example["ads_query"] = query
            fixed.append({
                "index": i,
                "nl": example.get("natural_language", ""),
                "original": original_query,
                "fixed": query,
                "reason": "Changed facility: to abs:"
            })

        fixed_examples.append(example)

    # Summary
    print(f"\nSyntax Fix Summary")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")
    print(f"Removed (unbalanced): {len(removed)}")
    print(f"Fixed (facility:): {len(fixed)}")
    print(f"Kept: {len(fixed_examples)}")

    # Show removed
    if removed:
        print(f"\n{'-'*60}")
        print("REMOVED (unbalanced brackets):")
        print(f"{'-'*60}")
        for r in removed:
            print(f"\nNL: {r['nl']}")
            print(f"Query: {r['query']}")

    # Show fixed
    if fixed:
        print(f"\n{'-'*60}")
        print("FIXED (facility: -> abs:):")
        print(f"{'-'*60}")
        for f in fixed:
            print(f"\nNL: {f['nl']}")
            print(f"Original: {f['original']}")
            print(f"Fixed: {f['fixed']}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return

    # Save
    with open(args.output, "w") as f:
        json.dump(fixed_examples, f, indent=2)
    print(f"\nFixed examples saved to: {args.output}")

    # Save removed examples
    removed_path = "data/datasets/raw/syntax_issues_removed.json"
    with open(removed_path, "w") as f:
        json.dump(removed, f, indent=2)
    print(f"Removed examples saved to: {removed_path}")


if __name__ == "__main__":
    main()
