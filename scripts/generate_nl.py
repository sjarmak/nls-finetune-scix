#!/usr/bin/env python3
"""
Generate natural language questions for ADS/SciX queries using Claude.

Usage:
    python scripts/generate_nl.py \
        --input data/datasets/raw/extracted_queries.json \
        --output data/datasets/raw/nl_pairs.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Add packages/finetune/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))

from anthropic import Anthropic

# Import from the scix domain module
from finetune.domains.scix.prompts import NL_GENERATION_PROMPT
from finetune.domains.scix.validate import validate_nl

# Initialize client (uses ANTHROPIC_API_KEY env var)
client: Anthropic | None = None


def get_client() -> Anthropic:
    """Get or create Anthropic client."""
    global client
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        client = Anthropic(api_key=api_key)
    return client


# ADS syntax markers that should never appear in natural language
ADS_SYNTAX_MARKERS = [
    "author:",
    "^author:",
    "abs:",
    "abstract:",
    "title:",
    "pubdate:",
    "bibstem:",
    "object:",
    "keyword:",
    "doi:",
    "arxiv:",
    "orcid:",
    "aff:",
    "inst:",
    "citation_count:",
    "read_count:",
    "property:",
    "database:",
    "doctype:",
    "full:",
    "body:",
    "bibcode:",
    "identifier:",
]


def generate_nl_for_query(query: str, category: str, retries: int = 3) -> str | None:
    """Generate a natural language question for an ADS query."""
    prompt = NL_GENERATION_PROMPT.format(query=query)

    for attempt in range(retries):
        try:
            response = get_client().messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            nl = response.content[0].text.strip()

            # Remove quotes if the model wrapped the response
            if nl.startswith('"') and nl.endswith('"'):
                nl = nl[1:-1]

            # Validate using the domain validator
            is_valid, issues = validate_nl(nl)
            if not is_valid:
                print(f"    Warning: Validation failed ({', '.join(issues)}), retrying...")
                continue

            # Additional check for ADS syntax markers
            nl_lower = nl.lower()
            if any(marker in nl_lower for marker in ADS_SYNTAX_MARKERS):
                print(f"    Warning: ADS syntax leaked, retrying... ({nl[:50]})")
                continue

            # Check for range syntax leakage
            if re.search(r"\[[^\]]+\s+TO\s+[^\]]+\]", nl, re.IGNORECASE):
                print(f"    Warning: Range syntax leaked, retrying... ({nl[:50]})")
                continue

            return nl

        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            continue

    return None


def process_batch(
    queries: list[dict],
    start_idx: int,
    batch_size: int,
    delay: float = 0.5,
) -> list[dict]:
    """Process a batch of queries."""
    results = []
    end_idx = min(start_idx + batch_size, len(queries))

    for i in range(start_idx, end_idx):
        q = queries[i]
        query = q["ads_query"]
        category = q.get("category", "unknown")

        nl = generate_nl_for_query(query, category)

        if nl:
            results.append(
                {
                    "natural_language": nl,
                    "ads_query": query,
                    "category": category,
                }
            )

        # Progress indicator
        if (i - start_idx + 1) % 10 == 0:
            print(f"    Processed {i - start_idx + 1}/{end_idx - start_idx}")

        # Rate limiting
        time.sleep(delay)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NL for ADS/SciX queries")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/datasets/raw/extracted_queries.json",
        help="Input JSON file with extracted ADS queries",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/datasets/raw/nl_pairs.json",
        help="Output JSON file with NL pairs",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Resume from this index",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of queries to process (0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between API calls in seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print first query without processing",
    )
    args = parser.parse_args()

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} queries from {input_path}")

    # Apply limit
    if args.limit > 0:
        queries = queries[: args.limit]
        print(f"Limited to {len(queries)} queries")

    if args.dry_run:
        print("\n--- Dry run: First query ---")
        print(f"Query: {queries[0]['ads_query']}")
        print(f"Category: {queries[0].get('category', 'unknown')}")
        print("\nGenerating NL...")
        nl = generate_nl_for_query(
            queries[0]["ads_query"],
            queries[0].get("category", "unknown"),
        )
        print(f"NL: {nl}")
        return

    # Load existing results if resuming
    output_path = Path(args.output)
    existing_results: list[dict] = []
    if args.resume > 0 and output_path.exists():
        with open(output_path) as f:
            existing_results = json.load(f)
        print(f"Resuming from index {args.resume}, {len(existing_results)} existing results")

    # Process in batches
    print(f"\nGenerating NL for {len(queries) - args.resume} queries...")
    print(f"Batch size: {args.batch_size}, Delay: {args.delay}s\n")

    all_results = existing_results.copy()
    start = args.resume

    while start < len(queries):
        batch_num = (start // args.batch_size) + 1
        total_batches = (len(queries) + args.batch_size - 1) // args.batch_size
        print(
            f"Batch {batch_num}/{total_batches} (indices {start}-{min(start + args.batch_size, len(queries)) - 1})"
        )

        batch_results = process_batch(queries, start, args.batch_size, args.delay)
        all_results.extend(batch_results)

        # Save after each batch (checkpoint)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  ✓ Batch complete. Total: {len(all_results)} pairs. Saved checkpoint.\n")

        start += args.batch_size

    # Final summary
    print("=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)

    category_counts: dict[str, int] = {}
    for r in all_results:
        cat = r.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nResults by category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nTotal pairs: {len(all_results)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
