#!/usr/bin/env python3
"""Fix overly permissive OR patterns to AND in gold_examples.json.

The previous strict curation incorrectly used OR when AND should be used
for topic searches. This script:
1. Finds abs:() patterns with OR
2. Changes OR to AND (more restrictive, correct for topic search)
3. Filters to only the most distinctive terms
"""

import json
import re
from pathlib import Path
from datetime import datetime
from copy import deepcopy


# Extended stopwords - these are too generic for topic searches
STOPWORDS = {
    # Articles and prepositions
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "and", "or", "is", "are", "was", "were", "be",
    "its", "their", "this", "that", "these", "those", "as", "into",
    # Very common/generic words
    "using", "based", "new", "recent", "first", "toward", "towards",
    "between", "among", "within", "about", "around",
    # Generic research terms (verbs/nouns)
    "observations", "properties", "characteristics", "nature",
    "evidence", "detection", "discovery", "search", "finding",
    "study", "studies", "research", "analysis", "data", "results",
    "survey", "model", "modeling", "simulation", "numerical",
    "mapping", "probing", "measuring", "exploring", "investigating",
    "distribution", "spatial", "temporal", "spectral",
    "inhomogeneous", "homogeneous", "structure", "dynamics",
    # Colors and generic adjectives (too common alone)
    "blue", "red", "hot", "cold", "large", "small", "high", "low",
    "early", "late", "young", "old", "bright", "faint", "massive",
    "failed", "successful", "nearby", "distant",
    # Common astronomy terms that are too generic alone
    "space", "ray", "rays", "satellite", "system", "systems",
    "origin", "origins", "formation", "evolution",
    # Generic telescope/instrument words (keep specific names)
    "telescope", "spectrograph", "detector", "camera",
}

# Domain-specific terms to KEEP (high priority)
KEEP_TERMS = {
    # Specific objects/phenomena
    "quasar", "pulsar", "magnetar", "supernova", "novae", "nova",
    "exoplanet", "protostar", "neutron", "black", "hole",
    "galaxy", "galaxies", "cluster", "nebula", "planetary",
    "asteroid", "comet", "meteor", "planet", "lunar", "solar",
    # Specific physics terms
    "gravitational", "electromagnetic", "radiation", "spectra",
    "photon", "electron", "proton", "neutrino", "cosmic",
    "gamma", "x-ray", "infrared", "ultraviolet", "radio",
    # Specific missions/instruments (proper nouns)
    "hubble", "jwst", "spitzer", "chandra", "kepler", "tess",
    "gaia", "alma", "vlt", "hst", "eso", "nasa",
    # Specific methods/codes
    "cloudy", "simbad", "astrometry", "photometry", "spectroscopy",
}


def term_priority(term: str) -> int:
    """Get priority score for a term (higher = more distinctive)."""
    term_lower = term.lower()

    # Highest priority: domain-specific terms we explicitly want to keep
    if term_lower in KEEP_TERMS:
        return 100

    # High priority: scientific notation (H_2, CO_2, Fe_II, etc.)
    if re.match(r'^[A-Z][a-z]?[_\d]+$', term) or re.match(r'^[A-Z]{1,2}\d+$', term):
        return 95

    # High priority: proper nouns (starts with capital, not all caps)
    if term[0].isupper() and not term.isupper() and len(term) > 3:
        return 80

    # High priority: acronyms (all caps, 2-6 chars)
    if term.isupper() and 2 <= len(term) <= 6:
        return 75

    # Medium priority: technical terms (longer, specific)
    if len(term) > 6 and term_lower not in STOPWORDS:
        return 60

    # Lower priority: short non-stopword terms
    if term_lower not in STOPWORDS and len(term) > 2:
        return 40

    # Lowest: stopwords or very short
    return 0


def is_distinctive_term(term: str) -> bool:
    """Check if a term is distinctive enough to be useful in a query."""
    return term_priority(term) >= 40


def fix_or_pattern(query: str) -> tuple[str, str | None]:
    """Fix abs:() with OR pattern to use AND with distinctive terms only.

    Returns:
        Tuple of (fixed_query, description of change or None if no change)
    """
    # Pattern for abs:(...) with OR inside
    pattern = r'abs:\s*\(([^)]+)\)'

    def replace_abs(match):
        content = match.group(1)

        # Skip if no OR operators
        if ' OR ' not in content:
            return match.group(0)

        # Split on OR
        terms = [t.strip() for t in content.split(' OR ')]

        # Sort terms by priority (most distinctive first)
        scored_terms = [(term_priority(t), t) for t in terms]
        scored_terms.sort(key=lambda x: -x[0])  # Highest priority first

        # Take top 3 distinctive terms (priority >= 40)
        distinctive = [t for score, t in scored_terms if score >= 40][:3]

        # If we filtered too much, take top 3 regardless of score
        if len(distinctive) < 2:
            distinctive = [t for _, t in scored_terms[:3] if len(t) > 2]

        # Limit to 3 terms
        distinctive = distinctive[:3]

        if len(distinctive) == 0:
            # No good terms, use first original term
            return f"abs:{terms[0]}" if terms else match.group(0)

        if len(distinctive) == 1:
            return f"abs:{distinctive[0]}"

        # Use AND instead of OR
        return f"abs:({' AND '.join(distinctive)})"

    fixed = re.sub(pattern, replace_abs, query)

    if fixed != query:
        return fixed, f"Changed OR to AND: {query[:50]}... -> {fixed[:50]}..."

    return query, None


def process_examples(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Process all examples and fix OR patterns.

    Returns:
        Tuple of (fixed_examples, list of changes made)
    """
    fixed_examples = []
    changes = []

    for i, example in enumerate(examples):
        query = example.get("ads_query", "")

        fixed_query, description = fix_or_pattern(query)

        if description:
            fixed_example = deepcopy(example)
            fixed_example["ads_query"] = fixed_query
            fixed_examples.append(fixed_example)
            changes.append({
                "index": i,
                "nl": example.get("natural_language", ""),
                "original": query,
                "fixed": fixed_query,
                "description": description
            })
        else:
            fixed_examples.append(example)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(examples)}...")

    return fixed_examples, changes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix OR to AND in gold examples")
    parser.add_argument("--input", "-i", default="data/datasets/raw/gold_examples.json")
    parser.add_argument("--output", "-o", default="data/datasets/raw/gold_examples.json")
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--backup", "-b", action="store_true")

    args = parser.parse_args()

    # Load
    with open(args.input) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")

    # Process
    fixed_examples, changes = process_examples(examples)

    # Summary
    print(f"\n{'='*60}")
    print("OR→AND FIX SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")
    print(f"Modified: {len(changes)}")

    # Show samples
    print(f"\n{'-'*60}")
    print("SAMPLE CHANGES (first 15):")
    print(f"{'-'*60}")
    for change in changes[:15]:
        print(f"\nNL: {change['nl']}")
        print(f"Original: {change['original']}")
        print(f"Fixed: {change['fixed']}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return

    # Backup if requested
    if args.backup:
        backup_path = Path(args.input).with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(backup_path, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"\nBackup saved to: {backup_path}")

    # Save fixed examples
    with open(args.output, "w") as f:
        json.dump(fixed_examples, f, indent=2)
    print(f"\nFixed examples saved to: {args.output}")

    # Save change report
    report_path = "data/datasets/evaluations/or_to_and_changes.json"
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_examples": len(examples),
            "modified": len(changes),
            "changes": changes[:200]  # First 200 for review
        }, f, indent=2)
    print(f"Change report saved to: {report_path}")


if __name__ == "__main__":
    main()
