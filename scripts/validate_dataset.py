#!/usr/bin/env python3
"""
Validate NL-query pairs and create training dataset for ADS/SciX.

Usage:
    python scripts/validate_dataset.py \
        --input data/datasets/raw/gold_examples.json \
        --output-dir data/datasets/processed
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Add packages/finetune/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))

from finetune.domains.scix.validate import lint_query, validate_nl


def validate_ads_query(query: str) -> tuple[bool, list[str]]:
    """Validate ADS query using offline linter."""
    issues = []

    if not query or not query.strip():
        issues.append("empty query")
        return False, issues

    result = lint_query(query)
    if not result.valid:
        issues.extend(result.errors)

    return len(issues) == 0, issues


def to_training_format(pair: dict, date: str = "2025-12-15") -> dict:
    """Convert a NL-query pair to chat training format."""
    system_content = (
        'Convert natural language to ADS search query. Output JSON: {"query": "..."}'
    )

    user_content = f"Query: {pair['natural_language']}\nDate: {date}"

    assistant_content = json.dumps({"query": pair["ads_query"]})

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and process training data")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/datasets/raw/gold_examples.json",
        help="Input JSON file with NL-query pairs",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/datasets/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Training split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2025-12-15",
        help="Date to use in training examples",
    )
    args = parser.parse_args()

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} pairs from {input_path}")

    print("=" * 50)
    print("VALIDATION")
    print("=" * 50)

    valid_pairs = []
    nl_issues: list[tuple[int, str, list[str]]] = []
    query_issues: list[tuple[int, str, list[str]]] = []

    for i, pair in enumerate(pairs):
        nl = pair.get("natural_language", "")
        query = pair.get("ads_query", "")

        nl_valid, nl_problems = validate_nl(nl)
        query_valid, query_problems = validate_ads_query(query)

        if not nl_valid:
            nl_issues.append((i, nl[:50], nl_problems))
        if not query_valid:
            query_issues.append((i, query[:50], query_problems))

        if nl_valid and query_valid:
            valid_pairs.append(pair)

    # Report validation results
    print(
        f"\nValid pairs: {len(valid_pairs)}/{len(pairs)} ({100 * len(valid_pairs) / len(pairs):.1f}%)"
    )

    if nl_issues:
        print(f"\nNL issues ({len(nl_issues)}):")
        for idx, text, problems in nl_issues[:5]:
            print(f"  [{idx}] '{text}...' - {', '.join(problems)}")
        if len(nl_issues) > 5:
            print(f"  ... and {len(nl_issues) - 5} more")

    if query_issues:
        print(f"\nQuery issues ({len(query_issues)}):")
        for idx, text, problems in query_issues[:5]:
            print(f"  [{idx}] '{text}...' - {', '.join(problems)}")
        if len(query_issues) > 5:
            print(f"  ... and {len(query_issues) - 5} more")

    # Category distribution
    print("\n" + "=" * 50)
    print("CATEGORY DISTRIBUTION")
    print("=" * 50)

    category_counts = Counter(p.get("category", "unknown") for p in valid_pairs)
    for cat, count in category_counts.most_common():
        pct = 100 * count / len(valid_pairs)
        bar = "█" * int(pct / 2)
        print(f"  {cat:20} {count:4} ({pct:5.1f}%) {bar}")

    # Check for duplicates
    print("\n" + "=" * 50)
    print("DUPLICATE CHECK")
    print("=" * 50)

    queries = [p["ads_query"] for p in valid_pairs]
    unique_queries = len(set(queries))
    print(
        f"Unique queries: {unique_queries}/{len(queries)} ({100 * unique_queries / len(queries):.1f}%)"
    )

    nls = [p["natural_language"] for p in valid_pairs]
    unique_nls = len(set(nls))
    print(f"Unique NLs: {unique_nls}/{len(nls)} ({100 * unique_nls / len(nls):.1f}%)")

    # Create train/val split
    print("\n" + "=" * 50)
    print("CREATING TRAIN/VAL SPLIT")
    print("=" * 50)

    random.seed(args.seed)
    random.shuffle(valid_pairs)

    split_idx = int(len(valid_pairs) * args.train_split)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"Train: {len(train_pairs)} examples")
    print(f"Val: {len(val_pairs)} examples")

    # Convert to training format
    train_data = [to_training_format(p, args.date) for p in train_pairs]
    val_data = [to_training_format(p, args.date) for p in val_pairs]

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    # Also save the valid pairs for inspection
    valid_pairs_path = output_dir / "valid_pairs.json"
    with open(valid_pairs_path, "w") as f:
        json.dump(valid_pairs, f, indent=2)

    print(f"\nSaved:")
    print(f"  {train_path} ({len(train_data)} examples)")
    print(f"  {val_path} ({len(val_data)} examples)")
    print(f"  {valid_pairs_path} (for inspection)")

    # Final verdict
    print("\n" + "=" * 50)
    print("VERDICT")
    print("=" * 50)

    min_required = 500
    if len(valid_pairs) >= min_required:
        print(f"\n✓ Dataset ready! {len(valid_pairs)} valid examples (>= {min_required} required)")
        sys.exit(0)
    else:
        print(f"\n✗ Need more data: {len(valid_pairs)} valid examples (< {min_required} required)")
        sys.exit(1)


if __name__ == "__main__":
    main()
