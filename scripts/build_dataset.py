#!/usr/bin/env python3
"""
Build the complete training dataset from all sources.

Combines:
1. Gold examples (hand-crafted)
2. Synthetic pairs (generated for edge cases)
3. NL pairs (generated from query logs via Claude)

Usage:
    # After generating NL pairs:
    python scripts/build_dataset.py \
        --gold data/datasets/raw/gold_examples.json \
        --synthetic data/datasets/raw/synthetic_pairs.json \
        --nl-pairs data/datasets/raw/nl_pairs.json \
        --output-dir data/datasets/processed
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Add packages/finetune/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))

from finetune.domains.scix.validate import lint_query, validate_nl


def load_json(path: Path) -> list[dict]:
    """Load JSON file, return empty list if not found."""
    if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        return []
    with open(path) as f:
        return json.load(f)


def normalize_pair(pair: dict) -> dict:
    """Normalize pair to consistent format."""
    return {
        "natural_language": pair.get("natural_language", pair.get("nl", "")),
        "ads_query": pair.get("ads_query", pair.get("query", "")),
        "category": pair.get("category", "unknown"),
    }


def validate_pair(pair: dict) -> tuple[bool, list[str]]:
    """Validate a NL-query pair."""
    issues = []
    
    nl = pair.get("natural_language", "")
    query = pair.get("ads_query", "")
    
    # Validate NL
    nl_valid, nl_issues = validate_nl(nl)
    if not nl_valid:
        issues.extend([f"NL: {i}" for i in nl_issues])
    
    # Validate query
    lint_result = lint_query(query)
    if not lint_result.valid:
        issues.extend([f"Query: {e}" for e in lint_result.errors])
    
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
    parser = argparse.ArgumentParser(description="Build complete training dataset")
    parser.add_argument(
        "--gold", "-g",
        type=str,
        default="data/datasets/raw/gold_examples.json",
        help="Gold examples file"
    )
    parser.add_argument(
        "--synthetic", "-s",
        type=str,
        default="data/datasets/raw/synthetic_pairs.json",
        help="Synthetic pairs file"
    )
    parser.add_argument(
        "--nl-pairs", "-n",
        type=str,
        default="data/datasets/raw/nl_pairs.json",
        help="NL pairs from query logs"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/datasets/processed",
        help="Output directory"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Training split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2025-12-15",
        help="Date for training examples"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("LOADING DATA SOURCES")
    print("=" * 60)
    
    # Load all sources
    gold = [normalize_pair(p) for p in load_json(Path(args.gold))]
    synthetic = [normalize_pair(p) for p in load_json(Path(args.synthetic))]
    nl_pairs = [normalize_pair(p) for p in load_json(Path(args.nl_pairs))]
    
    print(f"\n  Gold examples:    {len(gold):5}")
    print(f"  Synthetic pairs:  {len(synthetic):5}")
    print(f"  NL pairs:         {len(nl_pairs):5}")
    
    # Combine all
    all_pairs = gold + synthetic + nl_pairs
    print(f"\n  Total combined:   {len(all_pairs):5}")
    
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    valid_pairs = []
    invalid_count = 0
    
    for pair in all_pairs:
        is_valid, issues = validate_pair(pair)
        if is_valid:
            valid_pairs.append(pair)
        else:
            invalid_count += 1
            if invalid_count <= 5:
                nl = pair.get("natural_language", "")[:40]
                print(f"  Invalid: '{nl}...' - {', '.join(issues[:2])}")
    
    if invalid_count > 5:
        print(f"  ... and {invalid_count - 5} more invalid pairs")
    
    print(f"\n  Valid pairs: {len(valid_pairs)}/{len(all_pairs)}")
    
    print("\n" + "=" * 60)
    print("DEDUPLICATION")
    print("=" * 60)
    
    # Dedupe by NL text (case-insensitive)
    seen_nl = set()
    unique_pairs = []
    for pair in valid_pairs:
        nl_key = pair["natural_language"].lower().strip()
        if nl_key not in seen_nl:
            seen_nl.add(nl_key)
            unique_pairs.append(pair)
    
    # Also dedupe by query
    seen_query = set()
    final_pairs = []
    for pair in unique_pairs:
        query_key = pair["ads_query"].lower().strip()
        if query_key not in seen_query:
            seen_query.add(query_key)
            final_pairs.append(pair)
    
    print(f"  After NL dedup:    {len(unique_pairs):5}")
    print(f"  After query dedup: {len(final_pairs):5}")
    
    print("\n" + "=" * 60)
    print("CATEGORY DISTRIBUTION")
    print("=" * 60)
    
    cats = Counter(p.get("category", "unknown") for p in final_pairs)
    for cat, count in cats.most_common():
        pct = 100 * count / len(final_pairs)
        bar = "█" * int(pct / 2)
        print(f"  {cat:20} {count:4} ({pct:5.1f}%) {bar}")
    
    print("\n" + "=" * 60)
    print("TRAIN/VAL SPLIT")
    print("=" * 60)
    
    random.seed(args.seed)
    random.shuffle(final_pairs)
    
    split_idx = int(len(final_pairs) * args.train_split)
    train_pairs = final_pairs[:split_idx]
    val_pairs = final_pairs[split_idx:]
    
    print(f"  Train: {len(train_pairs):5}")
    print(f"  Val:   {len(val_pairs):5}")
    
    # Convert to training format
    train_data = [to_training_format(p, args.date) for p in train_pairs]
    val_data = [to_training_format(p, args.date) for p in val_pairs]
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    combined_path = output_dir / "all_pairs.json"
    
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    
    with open(combined_path, "w") as f:
        json.dump(final_pairs, f, indent=2)
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"\n  {train_path} ({len(train_data)} examples)")
    print(f"  {val_path} ({len(val_data)} examples)")
    print(f"  {combined_path} (all pairs for inspection)")
    
    # Final status
    print("\n" + "=" * 60)
    target = 500
    if len(final_pairs) >= target:
        print(f"✓ READY: {len(final_pairs)} pairs (>= {target} target)")
    else:
        print(f"✗ NEED MORE: {len(final_pairs)} pairs (< {target} target)")
        print(f"  Run generate_nl.py to add pairs from query logs")
    print("=" * 60)


if __name__ == "__main__":
    main()
