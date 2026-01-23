#!/usr/bin/env python3
"""Fix exact paper titles in abs: field.

These should be broken into distinctive topic terms with AND operators.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from copy import deepcopy


# Stopwords to filter out
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "and", "or", "is", "are", "was", "were", "be",
    "its", "their", "this", "that", "these", "those", "as", "into",
    "toward", "towards", "during", "using", "based", "between",
    "observations", "analysis", "study", "research", "group",
    "implications", "reanalysis", "simulating", "formation",
    "shifting", "benchmarking", "performance",
}


def extract_distinctive_terms(title: str) -> list[str]:
    """Extract the most distinctive terms from a title."""
    # Remove quotes if present
    title = title.strip('"')

    # Split into words
    words = re.findall(r'\b[A-Za-z0-9_-]+\b', title)

    # Filter and score terms
    distinctive = []
    for word in words:
        word_lower = word.lower()

        # Skip stopwords
        if word_lower in STOPWORDS:
            continue

        # Skip very short words
        if len(word) <= 2:
            continue

        # Skip pure numbers (but keep years and IDs)
        if word.isdigit() and len(word) != 4:
            continue

        distinctive.append(word)

    # Prioritize: proper nouns, scientific terms, acronyms
    def term_score(term):
        # Proper nouns/names (capitalized)
        if term[0].isupper() and not term.isupper():
            return 3
        # Acronyms
        if term.isupper() and len(term) >= 2:
            return 3
        # Scientific notation
        if re.match(r'^[A-Z][a-z]?[_\d]+$', term):
            return 3
        # Longer terms are more specific
        if len(term) > 7:
            return 2
        return 1

    # Sort by score and take top 4
    distinctive.sort(key=term_score, reverse=True)
    return distinctive[:4]


def fix_exact_title(query: str) -> tuple[str, str | None]:
    """Fix abs:(...) with exact title to use AND with distinctive terms."""

    # Pattern for abs:(...) without boolean operators (long content)
    pattern1 = r'abs:\s*\(([^)]{50,})\)'

    # Pattern for abs:"..." with long quoted string
    pattern2 = r'abs:\s*"([^"]{50,})"'

    def replace_match(match):
        content = match.group(1)

        # Skip if already has boolean operators
        if re.search(r'\b(OR|AND)\b', content):
            return match.group(0)

        terms = extract_distinctive_terms(content)

        if len(terms) == 0:
            return match.group(0)

        if len(terms) == 1:
            return f'abs:{terms[0]}'

        return f'abs:({" AND ".join(terms)})'

    original = query
    query = re.sub(pattern1, replace_match, query)
    query = re.sub(pattern2, replace_match, query)

    if query != original:
        return query, f"Fixed exact title: {original[:60]}... -> {query[:60]}..."

    return query, None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix exact titles in abs: field")
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
    changes = []

    for i, example in enumerate(examples):
        query = example.get("ads_query", "")
        fixed_query, description = fix_exact_title(query)

        if description:
            fixed_example = deepcopy(example)
            fixed_example["ads_query"] = fixed_query
            fixed_examples.append(fixed_example)
            changes.append({
                "index": i,
                "nl": example.get("natural_language", ""),
                "original": query,
                "fixed": fixed_query
            })
        else:
            fixed_examples.append(example)

    # Summary
    print(f"\nExact Title Fix Summary")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")
    print(f"Modified: {len(changes)}")

    # Show changes
    print(f"\n{'-'*60}")
    print("Changes:")
    print(f"{'-'*60}")
    for change in changes:
        print(f"\nNL: {change['nl']}")
        print(f"Original: {change['original']}")
        print(f"Fixed: {change['fixed']}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return

    # Save
    with open(args.output, "w") as f:
        json.dump(fixed_examples, f, indent=2)
    print(f"\nFixed examples saved to: {args.output}")


if __name__ == "__main__":
    main()
