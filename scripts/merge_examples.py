#!/usr/bin/env python3
"""
Merge curated generated examples into gold_examples.json.

This script:
1. Loads all curated_*.json files from data/datasets/generated/
2. Deduplicates against existing gold_examples.json (by NL text)
3. Assigns appropriate category to each example
4. Merges into gold_examples.json preserving existing examples
5. Outputs merge report

Usage:
    python scripts/merge_examples.py [--dry-run]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
GOLD_EXAMPLES_PATH = PROJECT_ROOT / "data" / "datasets" / "raw" / "gold_examples.json"
CURATED_DIR = PROJECT_ROOT / "data" / "datasets" / "generated"
MERGE_REPORT_PATH = PROJECT_ROOT / "data" / "datasets" / "evaluations" / "merge_report.json"


def normalize_nl(text: str) -> str:
    """Normalize NL text for deduplication (case-insensitive, whitespace-normalized)."""
    return " ".join(text.lower().split())


def determine_category_from_filename(filename: str) -> str:
    """Determine category from curated filename."""
    # Map curated filenames to categories
    category_map = {
        "curated_bibgroup_examples.json": "bibgroup",
        "curated_collection_examples.json": "collection",
        "curated_doctype_examples.json": "doctype",
        "curated_operator_examples.json": "operator",
        "curated_property_examples.json": "property",
    }
    return category_map.get(filename, "generated")


def load_gold_examples() -> list[dict]:
    """Load existing gold examples."""
    if not GOLD_EXAMPLES_PATH.exists():
        return []
    with open(GOLD_EXAMPLES_PATH) as f:
        return json.load(f)


def load_curated_examples() -> dict[str, list[dict]]:
    """Load all curated example files."""
    curated_by_file = {}
    for path in sorted(CURATED_DIR.glob("curated_*.json")):
        with open(path) as f:
            curated_by_file[path.name] = json.load(f)
    return curated_by_file


def merge_examples(dry_run: bool = False) -> dict:
    """
    Merge curated examples into gold_examples.json.

    Returns merge report with statistics.
    """
    # Load existing gold examples
    print("Loading gold_examples.json...")
    gold_examples = load_gold_examples()
    original_count = len(gold_examples)
    print(f"  Loaded {original_count} existing examples")

    # Build set of normalized NL for deduplication
    existing_nl = {normalize_nl(ex["natural_language"]) for ex in gold_examples}

    # Load curated examples
    print("\nLoading curated example files...")
    curated_by_file = load_curated_examples()

    # Track statistics
    stats = {
        "original_count": original_count,
        "curated_files": {},
        "total_curated": 0,
        "duplicates_skipped": 0,
        "examples_added": 0,
        "by_category": defaultdict(int),
        "final_count": 0,
    }

    # Process each curated file
    new_examples = []
    for filename, examples in curated_by_file.items():
        category = determine_category_from_filename(filename)
        file_stats = {
            "total": len(examples),
            "duplicates": 0,
            "added": 0,
            "category": category,
        }

        print(f"  Processing {filename}: {len(examples)} examples")

        for ex in examples:
            nl_normalized = normalize_nl(ex["natural_language"])

            if nl_normalized in existing_nl:
                file_stats["duplicates"] += 1
                stats["duplicates_skipped"] += 1
            else:
                # Create new example with category
                new_ex = {
                    "natural_language": ex["natural_language"],
                    "ads_query": ex["ads_query"],
                    "category": ex.get("category", category),
                }
                new_examples.append(new_ex)
                existing_nl.add(nl_normalized)  # Prevent duplicates within curated files
                file_stats["added"] += 1
                stats["examples_added"] += 1
                stats["by_category"][new_ex["category"]] += 1

        stats["total_curated"] += file_stats["total"]
        stats["curated_files"][filename] = file_stats
        print(f"    -> Added: {file_stats['added']}, Duplicates skipped: {file_stats['duplicates']}")

    # Merge
    merged_examples = gold_examples + new_examples
    stats["final_count"] = len(merged_examples)

    # Convert defaultdict to regular dict for JSON serialization
    stats["by_category"] = dict(stats["by_category"])

    # Report
    print("\n" + "=" * 60)
    print("MERGE REPORT")
    print("=" * 60)
    print(f"Original examples: {stats['original_count']}")
    print(f"Total curated examined: {stats['total_curated']}")
    print(f"Duplicates skipped: {stats['duplicates_skipped']}")
    print(f"New examples added: {stats['examples_added']}")
    print(f"Final total: {stats['final_count']}")
    print("\nExamples added by category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  {cat}: {count}")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] No changes written to disk.")
    else:
        # Write merged examples
        print(f"\nWriting merged examples to {GOLD_EXAMPLES_PATH}...")
        with open(GOLD_EXAMPLES_PATH, "w") as f:
            json.dump(merged_examples, f, indent=2)
        print(f"  Written {len(merged_examples)} examples")

        # Write merge report
        MERGE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing merge report to {MERGE_REPORT_PATH}...")
        with open(MERGE_REPORT_PATH, "w") as f:
            json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge curated generated examples into gold_examples.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing files",
    )
    args = parser.parse_args()

    stats = merge_examples(dry_run=args.dry_run)

    # Summary
    if stats["final_count"] >= 4500:
        print(f"\n✓ Target achieved: {stats['final_count']} total examples (target: 6000+, minimum: 4500)")
    else:
        print(f"\n⚠ Below target: {stats['final_count']} total examples (target: 6000+)")


if __name__ == "__main__":
    main()
