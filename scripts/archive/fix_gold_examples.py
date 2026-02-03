#!/usr/bin/env python3
"""Fix identified issues in gold_examples.json.

Applies automated fixes for:
1. Author initials being guessed - remove fabricated initials
2. Category 'unfielded' - attempt to infer proper category
3. Syntax issues - fix common patterns
4. NL/Query alignment - fix obvious mismatches
"""

import json
import re
import sys
from pathlib import Path
from copy import deepcopy
from datetime import datetime

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))



def fix_author_initials(example: dict) -> tuple[dict, list[str]]:
    """Remove fabricated author initials from queries.

    If NL has just last name, query should not have specific initials.
    """
    fixes = []
    nl = example.get("natural_language", "")
    query = example.get("ads_query", "")

    # Find author names in NL (just last name, no initial)
    nl_authors = re.findall(r'\bby\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b(?!\s+[A-Z]\.)', nl)

    for nl_author in nl_authors:
        # Pattern: author:"LastName, X." or author:"LastName, X. Y." or author:"^LastName, X."
        pattern = rf'(author:\s*"\^?){re.escape(nl_author)}(?:-[A-Za-z]+)?,\s*[A-Z]\.?\s*[A-Z]?\.?"'

        if re.search(pattern, query, re.IGNORECASE):
            # Replace with just the last name
            new_query = re.sub(
                pattern,
                rf'\1{nl_author}"',
                query,
                flags=re.IGNORECASE
            )
            if new_query != query:
                fixes.append(f"Removed fabricated initials for author '{nl_author}'")
                query = new_query

    example = deepcopy(example)
    example["ads_query"] = query
    return example, fixes


def infer_category(example: dict) -> tuple[dict, list[str]]:
    """Infer proper category from query content."""
    fixes = []
    query = example.get("ads_query", "")
    nl = example.get("natural_language", "").lower()
    current_cat = example.get("category", "")

    if current_cat != "unfielded":
        return example, fixes

    # Infer category from query fields and NL content
    new_category = None

    # Check for operators first (highest priority)
    for op in ["citations", "references", "trending", "useful", "similar", "reviews"]:
        if f"{op}(" in query.lower():
            new_category = "operator"
            break

    if not new_category:
        # Check for specific field patterns
        if "^author:" in query.lower() or "first author" in nl:
            new_category = "first_author"
        elif "author:" in query.lower():
            new_category = "author"
        elif "object:" in query.lower():
            new_category = "object"
        elif "bibgroup:" in query.lower():
            new_category = "bibgroup"
        elif "collection:" in query.lower() or "database:" in query.lower():
            new_category = "collection"
        elif "property:" in query.lower():
            new_category = "property"
        elif "doctype:" in query.lower():
            new_category = "doctype"
        elif "bibstem:" in query.lower():
            new_category = "publication"
        elif "year:" in query.lower() or "pubdate:" in query.lower():
            new_category = "temporal"
        elif "aff:" in query.lower() or "inst:" in query.lower():
            new_category = "affiliation"
        elif "abs:" in query.lower() or "title:" in query.lower():
            new_category = "topic"
        else:
            # Default based on NL content
            if any(word in nl for word in ["cite", "citing", "cited"]):
                new_category = "operator"
            elif any(word in nl for word in ["hubble", "jwst", "chandra", "spitzer", "kepler"]):
                new_category = "bibgroup"
            elif any(word in nl for word in ["refereed", "peer", "open access", "preprint"]):
                new_category = "property"
            else:
                new_category = "topic"  # Default fallback

    if new_category:
        example = deepcopy(example)
        example["category"] = new_category
        fixes.append(f"Changed category from 'unfielded' to '{new_category}'")

    return example, fixes


def fix_syntax_issues(example: dict) -> tuple[dict, list[str]]:
    """Fix common syntax issues in queries."""
    fixes = []
    query = example.get("ads_query", "")
    original = query

    # Fix unbalanced quotes (add missing closing quote at end)
    if query.count('"') % 2 != 0:
        query = query + '"'
        fixes.append("Added missing closing quote")

    # Fix unbalanced parentheses
    open_count = query.count("(")
    close_count = query.count(")")
    if open_count > close_count:
        query = query + ")" * (open_count - close_count)
        fixes.append(f"Added {open_count - close_count} missing closing parenthesis")
    elif close_count > open_count:
        # Remove extra closing parens from the end
        while query.endswith(")") and query.count(")") > query.count("("):
            query = query[:-1]
        fixes.append("Removed extra closing parenthesis")

    # Fix common field name typos
    replacements = [
        (r'\bauthors:', 'author:'),
        (r'\babstracts:', 'abs:'),
        (r'\babstract:', 'abs:'),
        (r'\byears:', 'year:'),
        (r'\bdocuments:', 'doctype:'),
        (r'\bproperties:', 'property:'),
    ]

    for pattern, replacement in replacements:
        if re.search(pattern, query, re.IGNORECASE):
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
            fixes.append(f"Fixed field name: {pattern} -> {replacement}")

    # Fix database: -> collection: (prefer collection)
    if "database:" in query.lower():
        query = re.sub(r'\bdatabase:', 'collection:', query, flags=re.IGNORECASE)
        fixes.append("Changed database: to collection:")

    if query != original:
        example = deepcopy(example)
        example["ads_query"] = query

    return example, fixes


def fix_alignment_issues(example: dict) -> tuple[dict, list[str]]:
    """Fix NL/Query alignment issues like author in abs: field."""
    fixes = []
    nl = example.get("natural_language", "").lower()
    query = example.get("ads_query", "")

    # Check if NL mentions "by [name]" but query uses abs: with names
    if " by " in nl and "author:" not in query.lower():
        # Extract potential author name from NL
        author_match = re.search(r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                                  example.get("natural_language", ""))
        if author_match:
            author_name = author_match.group(1)
            # Check if this name appears in abs:()
            if re.search(rf'abs:\s*\([^)]*\b{author_name}\b', query, re.IGNORECASE):
                # This is likely an author being put in abs: incorrectly
                # Extract the abs content and rebuild
                abs_match = re.search(r'abs:\s*\(([^)]+)\)', query)
                if abs_match:
                    abs_content = abs_match.group(1)
                    # Remove author name from abs content
                    remaining = re.sub(rf'\b{author_name}\b', '', abs_content, flags=re.IGNORECASE)
                    remaining = re.sub(r'\s+', ' ', remaining).strip()

                    # Build new query
                    new_query = f'author:"{author_name}"'
                    if remaining:
                        new_query += f' abs:"{remaining}"'

                    # Preserve other parts of query
                    other_parts = re.sub(r'abs:\s*\([^)]+\)', '', query).strip()
                    if other_parts:
                        new_query += ' ' + other_parts

                    example = deepcopy(example)
                    example["ads_query"] = new_query.strip()
                    fixes.append(f"Moved author '{author_name}' from abs: to author: field")

    return example, fixes


def process_examples(examples: list[dict], dry_run: bool = True) -> tuple[list[dict], dict]:
    """Process all examples and apply fixes."""
    fixed_examples = []
    stats = {
        "total": len(examples),
        "modified": 0,
        "author_initials_fixed": 0,
        "category_fixed": 0,
        "syntax_fixed": 0,
        "alignment_fixed": 0,
        "all_fixes": []
    }

    for i, example in enumerate(examples):
        original = deepcopy(example)
        all_fixes = []

        # Apply fixes in order
        example, fixes = fix_author_initials(example)
        if fixes:
            all_fixes.extend(fixes)
            stats["author_initials_fixed"] += 1

        example, fixes = infer_category(example)
        if fixes:
            all_fixes.extend(fixes)
            stats["category_fixed"] += 1

        example, fixes = fix_syntax_issues(example)
        if fixes:
            all_fixes.extend(fixes)
            stats["syntax_fixed"] += 1

        example, fixes = fix_alignment_issues(example)
        if fixes:
            all_fixes.extend(fixes)
            stats["alignment_fixed"] += 1

        if all_fixes:
            stats["modified"] += 1
            stats["all_fixes"].append({
                "index": i,
                "original": original,
                "fixed": example,
                "fixes": all_fixes
            })

        fixed_examples.append(example)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples...")

    return fixed_examples, stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix issues in gold_examples.json")
    parser.add_argument("--input", "-i", default="data/datasets/raw/gold_examples.json",
                        help="Input JSON file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON file (default: overwrite input)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Don't write output, just report what would be fixed")
    parser.add_argument("--report", "-r", default="data/datasets/evaluations/fix_report.json",
                        help="Path to save fix report")
    parser.add_argument("--backup", "-b", action="store_true",
                        help="Create backup before modifying")

    args = parser.parse_args()

    # Load examples
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} examples from {input_path}")

    # Process
    print("Processing examples...")
    fixed_examples, stats = process_examples(examples, dry_run=args.dry_run)

    # Print summary
    print("\n" + "="*60)
    print("FIX SUMMARY")
    print("="*60)
    print(f"Total examples: {stats['total']}")
    print(f"Modified: {stats['modified']}")
    print(f"  - Author initials fixed: {stats['author_initials_fixed']}")
    print(f"  - Categories fixed: {stats['category_fixed']}")
    print(f"  - Syntax fixed: {stats['syntax_fixed']}")
    print(f"  - Alignment fixed: {stats['alignment_fixed']}")

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now().isoformat(),
        "input_file": str(input_path),
        "summary": {
            "total": stats["total"],
            "modified": stats["modified"],
            "author_initials_fixed": stats["author_initials_fixed"],
            "category_fixed": stats["category_fixed"],
            "syntax_fixed": stats["syntax_fixed"],
            "alignment_fixed": stats["alignment_fixed"]
        },
        "fixes": stats["all_fixes"][:100]  # First 100 for review
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFix report saved to: {report_path}")

    # Show sample fixes
    if stats["all_fixes"]:
        print("\n" + "-"*60)
        print("SAMPLE FIXES (first 5):")
        print("-"*60)
        for fix in stats["all_fixes"][:5]:
            print(f"\nOriginal NL: {fix['original']['natural_language']}")
            print(f"Original Query: {fix['original']['ads_query']}")
            print(f"Fixed Query: {fix['fixed']['ads_query']}")
            print(f"Original Category: {fix['original'].get('category', 'none')}")
            print(f"Fixed Category: {fix['fixed'].get('category', 'none')}")
            print(f"Fixes applied: {', '.join(fix['fixes'])}")

    # Write output if not dry run
    if not args.dry_run:
        output_path = Path(args.output) if args.output else input_path

        if args.backup and output_path == input_path:
            backup_path = input_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(backup_path, "w") as f:
                json.dump(examples, f, indent=2)
            print(f"\nBackup saved to: {backup_path}")

        with open(output_path, "w") as f:
            json.dump(fixed_examples, f, indent=2)
        print(f"\nFixed examples saved to: {output_path}")
    else:
        print("\n[DRY RUN] No files were modified. Run without --dry-run to apply fixes.")


if __name__ == "__main__":
    main()
