#!/usr/bin/env python3
"""
Extract diverse Sourcegraph queries from BigQuery for training data generation.

Usage:
    python scripts/extract_queries.py --output data/datasets/raw/extracted_queries.json
"""

import argparse
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path


# Target distribution for first iteration (1,100 total)
CATEGORY_TARGETS = {
    "repo_scoped": 300,
    "lang_filtered": 150,
    "commit_search": 150,
    "diff_search": 150,
    "symbol_search": 100,
    "keyword_only": 100,
    "file_filtered": 50,
    "author_search": 50,
    "dependency_search": 50,
}


def run_bq_query(query: str) -> list[dict]:
    """Execute a BigQuery query and return results as list of dicts."""
    cmd = ["bq", "query", "--project_id=telligentsourcegraph", "--use_legacy_sql=false", "--format=json", "--max_rows=10000", query]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"BigQuery error: {result.stderr} {result.stdout}", file=sys.stderr)
        sys.exit(1)

    if not result.stdout.strip():
        return []

    return json.loads(result.stdout)


def decode_query(encoded: str) -> str:
    """URL-decode a query string."""
    if not encoded:
        return ""
    return urllib.parse.unquote_plus(encoded)





def categorize_query(encoded_query: str) -> str:
    """Categorize a query based on its filters."""
    q = encoded_query.lower() if encoded_query else ""

    # Check in order of specificity
    if "type:commit" in q or "type%3acommit" in q:
        return "commit_search"
    if "type:diff" in q or "type%3adiff" in q:
        return "diff_search"
    if "type:symbol" in q or "type%3asymbol" in q:
        return "symbol_search"
    if "author:" in q or "author%3a" in q:
        return "author_search"

    # Dependency patterns
    dep_patterns = [
        "file:package.json",
        "file:requirements",
        "file:go.mod",
        "file%3apackage",
        "file%3arequirements",
        "file%3ago.mod",
        "file:(package.json",
        "file%3a(package",
    ]
    if any(p in q for p in dep_patterns):
        return "dependency_search"

    if "lang:" in q or "lang%3a" in q:
        return "lang_filtered"
    if "file:" in q or "file%3a" in q:
        return "file_filtered"
    if "repo:" in q or "repo%3a" in q:
        return "repo_scoped"

    return "keyword_only"


def build_extraction_query(category: str, limit: int) -> str:
    """Build a BigQuery query to extract examples for a category."""

    # Category-specific WHERE clauses
    category_filters = {
        "commit_search": "(url LIKE '%type:commit%' OR url LIKE '%type%3Acommit%' OR url LIKE '%type%3acommit%')",
        "diff_search": "(url LIKE '%type:diff%' OR url LIKE '%type%3Adiff%' OR url LIKE '%type%3adiff%')",
        "symbol_search": "(url LIKE '%type:symbol%' OR url LIKE '%type%3Asymbol%' OR url LIKE '%type%3asymbol%')",
        "author_search": "(url LIKE '%author:%' OR url LIKE '%author%3A%' OR url LIKE '%author%3a%')",
        "dependency_search": "(url LIKE '%file:package.json%' OR url LIKE '%file%3Apackage%' OR url LIKE '%file:requirements%' OR url LIKE '%file:go.mod%')",
        "lang_filtered": "(url LIKE '%lang:%' OR url LIKE '%lang%3A%') AND NOT (url LIKE '%type:commit%' OR url LIKE '%type:diff%' OR url LIKE '%type:symbol%')",
        "file_filtered": "(url LIKE '%file:%' OR url LIKE '%file%3A%') AND NOT (url LIKE '%lang:%') AND NOT (url LIKE '%type:%') AND NOT (url LIKE '%file:package%' OR url LIKE '%file:requirements%' OR url LIKE '%file:go.mod%')",
        "repo_scoped": "(url LIKE '%repo:%' OR url LIKE '%repo%3A%') AND NOT (url LIKE '%lang:%') AND NOT (url LIKE '%file:%') AND NOT (url LIKE '%type:%') AND NOT (url LIKE '%author:%')",
        "keyword_only": "NOT (url LIKE '%repo:%' OR url LIKE '%repo%3A%') AND NOT (url LIKE '%lang:%') AND NOT (url LIKE '%file:%') AND NOT (url LIKE '%type:%') AND NOT (url LIKE '%author:%')",
    }

    where_clause = category_filters.get(category, "TRUE")

    return f"""
    SELECT DISTINCT
        REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') as encoded_query
    FROM `telligentsourcegraph.dotcom_events.search_urls`
    WHERE url LIKE '%/search?%'
        AND REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') IS NOT NULL
        AND LENGTH(REGEXP_EXTRACT(url, r'[?&]q=([^&]+)')) BETWEEN 10 AND 500
        AND {where_clause}
        -- Filter out junk
        AND url NOT LIKE '%<script%'
        AND url NOT LIKE '%<esi:%'
        AND REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') != 'context:global'
        AND REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') NOT LIKE 'context:global+%'
    ORDER BY RAND()
    LIMIT {limit * 2}
    """


def extract_queries_for_category(category: str, target: int) -> list[dict]:
    """Extract queries for a specific category."""
    print(f"  Extracting {category} (target: {target})...")

    query = build_extraction_query(category, target)
    results = run_bq_query(query)

    extracted = []
    seen_decoded = set()

    for row in results:
        encoded = row.get("encoded_query")
        if not encoded:
            continue

        decoded = decode_query(encoded)

        # Skip duplicates
        if decoded in seen_decoded:
            continue
        seen_decoded.add(decoded)

        # Verify category assignment
        actual_category = categorize_query(encoded)
        if actual_category != category:
            continue

        extracted.append(
            {
                "sourcegraph_query": decoded,
                "category": category,
                "encoded_query": encoded,
            }
        )

        if len(extracted) >= target:
            break

    print(f"    Got {len(extracted)} examples")
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract queries from BigQuery")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/datasets/raw/extracted_queries.json",
        help="Output file path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print queries without executing")
    args = parser.parse_args()

    print("Extracting Sourcegraph queries from BigQuery...")
    print(f"Target distribution: {sum(CATEGORY_TARGETS.values())} total examples\n")

    all_queries = []

    for category, target in CATEGORY_TARGETS.items():
        if args.dry_run:
            query = build_extraction_query(category, target)
            print(f"\n--- {category} ---")
            print(query)
            continue

        queries = extract_queries_for_category(category, target)
        all_queries.extend(queries)

    if args.dry_run:
        return

    # Summary
    print(f"\n{'=' * 50}")
    print("EXTRACTION SUMMARY")
    print(f"{'=' * 50}")

    category_counts = {}
    for q in all_queries:
        cat = q["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for category, target in CATEGORY_TARGETS.items():
        actual = category_counts.get(category, 0)
        status = "✓" if actual >= target * 0.8 else "⚠️"
        print(f"  {status} {category}: {actual}/{target}")

    print(f"\nTotal: {len(all_queries)} queries")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_queries, f, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
