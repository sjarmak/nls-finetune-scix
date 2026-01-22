#!/usr/bin/env python3
"""
Generate training data for operator-based queries.

Creates NL-query pairs for all ADS operators with diverse trigger phrases,
addressing the operator coverage requirements from US-009. Uses the trigger
patterns from ner.py (US-004) as the basis for natural language variations.

Operators covered:
- citations(): papers that cite X
- references(): papers cited by X / bibliography of X
- trending(): hot/popular papers
- similar(): papers like X
- useful(): foundational/helpful papers
- reviews(): review/survey articles
- topn(): top N papers by citation count

Usage:
    python scripts/generate_operator_examples.py \
        --output data/datasets/generated/operator_examples.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

# =============================================================================
# Operator trigger phrase variations
# These are derived from OPERATOR_PATTERNS in ner.py but formatted for NL
# =============================================================================

CITATIONS_TRIGGERS = [
    # Basic patterns
    "papers citing {target}",
    "papers that cite {target}",
    "work citing {target}",
    "research citing {target}",
    "studies citing {target}",
    "articles citing {target}",
    "works that cite {target}",
    "papers which cite {target}",
    # "cited by" patterns
    "{target} cited by",
    "who cited {target}",
    # Citation noun patterns
    "citations to {target}",
    "citations of {target}",
    "find citations to {target}",
    "get citations of {target}",
    "show citations to {target}",
    "list citations of {target}",
    # More natural patterns
    "how many papers cite {target}",
    "what research cites {target}",
]

REFERENCES_TRIGGERS = [
    # Basic patterns
    "references of {target}",
    "references from {target}",
    "papers referenced by {target}",
    "bibliography of {target}",
    # "cited in" patterns
    "papers cited in {target}",
    "sources cited by {target}",
    "works cited by {target}",
    # "what does X cite" patterns
    "what does {target} cite",
    "what papers does {target} cite",
    "papers {target} cites",
    "papers they cite in {target}",
    # Show/list patterns
    "show references of {target}",
    "list references from {target}",
    # Natural variations
    "what sources did {target} use",
    "references used in {target}",
]

TRENDING_TRIGGERS = [
    # Hot patterns
    "hot topics in {topic}",
    "hot papers in {topic}",
    "hot research in {topic}",
    "what's hot in {topic}",
    # Trending patterns
    "trending papers in {topic}",
    "trending research in {topic}",
    "trending topics in {topic}",
    "what's trending in {topic}",
    "trending now in {topic}",
    # Popular patterns
    "popular papers in {topic}",
    "popular research in {topic}",
    "currently popular in {topic}",
    "popular now in {topic}",
    "popular recently in {topic}",
    "recently popular in {topic}",
    # Generic
    "what's popular in {topic}",
    "hot in {topic}",
]

SIMILAR_TRIGGERS = [
    # Similar to patterns
    "similar to {target}",
    "similar papers to {target}",
    "papers similar to {target}",
    "work similar to {target}",
    "studies similar to {target}",
    "find similar to {target}",
    # Related patterns
    "related to {target}",
    "related papers to {target}",
    "research related to {target}",
    # Like patterns
    "papers like {target}",
    "works like {target}",
    "articles like {target}",
    "studies like {target}",
    # Resembling patterns
    "papers resembling {target}",
    "studies resembling {target}",
    # Similar work/research patterns
    "similar work to {target}",
    "similar research to {target}",
    "similar studies to {target}",
    "comparable papers to {target}",
    "comparable work to {target}",
]

USEFUL_TRIGGERS = [
    # Useful patterns
    "useful papers on {topic}",
    "most useful papers on {topic}",
    "high utility papers on {topic}",
    # Helpful patterns
    "helpful papers on {topic}",
    "helpful research on {topic}",
    "helpful work on {topic}",
    # Foundational patterns
    "foundational work on {topic}",
    "foundational papers on {topic}",
    # Essential patterns
    "essential reading on {topic}",
    "essential papers on {topic}",
    # Must-read patterns
    "must-read papers on {topic}",
    "must read papers on {topic}",
    # Key papers
    "key papers on {topic}",
    "key references on {topic}",
    # Seminal/landmark
    "seminal papers on {topic}",
    "seminal work on {topic}",
    "landmark papers in {topic}",
    # Important
    "important papers on {topic}",
]

REVIEWS_TRIGGERS = [
    # Review article patterns
    "review articles on {topic}",
    "review papers on {topic}",
    "reviews of {topic}",
    "reviews on {topic}",
    "reviews about {topic}",
    "find reviews on {topic}",
    # Survey patterns
    "survey papers on {topic}",
    "survey articles on {topic}",
    "survey of {topic}",
    "survey on {topic}",
    "survey about {topic}",
    # Overview patterns
    "overviews of {topic}",
    "overview papers on {topic}",
    # Comprehensive patterns
    "comprehensive reviews on {topic}",
    "comprehensive survey of {topic}",
    # Special review types
    "literature review on {topic}",
    "review of the literature on {topic}",
    "systematic review of {topic}",
    "state-of-the-art review on {topic}",
    # Tutorial/introduction
    "tutorial on {topic}",
    "tutorial papers on {topic}",
    "introduction to {topic}",
]

TOPN_TRIGGERS = [
    # Top N explicit
    "top 10 papers on {topic}",
    "top 10 papers about {topic}",
    "top 5 papers on {topic}",
    "top 20 papers on {topic}",
    "top 100 papers on {topic}",
    # Most cited
    "most cited papers on {topic}",
    "most cited papers in {topic}",
    "most cited articles on {topic}",
    "highest cited papers on {topic}",
    "highly cited papers on {topic}",
    # Best papers
    "best papers on {topic}",
    "best papers about {topic}",
    "best papers in {topic}",
    # Top ranked
    "top ranked papers on {topic}",
    "top papers on {topic}",
    "top papers in {topic}",
    # Citation count
    "papers with most citations on {topic}",
    "papers with highest citation count on {topic}",
]

# =============================================================================
# Topics and targets for filling templates
# =============================================================================

# General astronomy/physics topics for trending, useful, reviews, topn
TOPICS = [
    "dark matter",
    "dark energy",
    "black holes",
    "exoplanets",
    "galaxy formation",
    "gravitational waves",
    "cosmic microwave background",
    "neutron stars",
    "supernovae",
    "stellar evolution",
    "cosmology",
    "quasars",
    "AGN",
    "star formation",
    "galactic dynamics",
    "interstellar medium",
    "planet formation",
    "galaxy clusters",
    "gamma-ray bursts",
    "pulsars",
    "redshift surveys",
    "Hubble constant",
    "inflation",
    "reionization",
    "primordial nucleosynthesis",
    "machine learning in astronomy",
    "spectroscopy",
    "photometry",
    "astrometry",
    "radio astronomy",
]

# Targets for citations, references, similar (papers/authors to reference)
# These are realistic but fictional paper identifiers and author names
PAPER_TARGETS = [
    "the Riess et al. 2018 paper",
    "Planck 2018 results",
    "Einstein 1905",
    "Hawking 1974 paper on black holes",
    "the LIGO detection paper",
    "Gaia DR2 paper",
    "the original SDSS paper",
    "Hubble deep field paper",
    "the CMB discovery paper",
    "WMAP results",
    "this paper",
    "that article",
    "the 2016 gravitational wave paper",
    "Perlmutter et al. supernova paper",
    "Mayor & Queloz exoplanet paper",
    "the 2019 black hole image paper",
]

AUTHOR_TARGETS = [
    "Hawking",
    "Einstein papers",
    "Penrose publications",
    "Riess et al.",
    "Perlmutter",
    "Weinberg",
    "Rees papers",
    "Salpeter",
    "Chandrasekhar",
    "Sunyaev and Zeldovich",
]

# Years for combined queries
YEARS = list(range(2018, 2026))

# =============================================================================
# Example generation functions
# =============================================================================


def generate_citations_examples(count: int = 18) -> list[dict]:
    """Generate citations() operator examples."""
    examples = []
    targets = PAPER_TARGETS + AUTHOR_TARGETS

    # Use all trigger patterns
    for i, trigger in enumerate(CITATIONS_TRIGGERS):
        if len(examples) >= count:
            break
        target = targets[i % len(targets)]
        nl = trigger.format(target=target)

        # Build query - citations wraps a search for the target
        # For paper targets, use abs: or bibcode-like search
        # For author targets, use author:
        author_names = ["hawking", "einstein", "penrose", "riess", "perlmutter", "weinberg", "rees", "salpeter", "chandrasekhar", "sunyaev"]
        matched_author = None
        for name in author_names:
            if name.lower() in target.lower():
                matched_author = name.capitalize()
                break
        if matched_author:
            # Author target - use matched author name
            query = f"citations(author:{matched_author})"
        elif "this paper" in target.lower() or "that article" in target.lower():
            # Generic paper reference - use placeholder
            query = "citations(bibcode:PLACEHOLDER)"
        else:
            # Paper target - extract key words for abs search
            query = f'citations(abs:"{target}")'

        examples.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "operator",
            "operator": "citations",
        })

    return examples


def generate_references_examples(count: int = 16) -> list[dict]:
    """Generate references() operator examples."""
    examples = []
    targets = PAPER_TARGETS + AUTHOR_TARGETS

    for i, trigger in enumerate(REFERENCES_TRIGGERS):
        if len(examples) >= count:
            break
        target = targets[i % len(targets)]
        nl = trigger.format(target=target)

        # Build query
        author_names = ["hawking", "einstein", "penrose", "riess", "perlmutter", "weinberg", "rees", "salpeter", "chandrasekhar", "sunyaev"]
        matched_author = None
        for name in author_names:
            if name.lower() in target.lower():
                matched_author = name.capitalize()
                break
        if matched_author:
            query = f"references(author:{matched_author})"
        elif "this paper" in target.lower() or "that article" in target.lower():
            query = "references(bibcode:PLACEHOLDER)"
        else:
            query = f'references(abs:"{target}")'

        examples.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "operator",
            "operator": "references",
        })

    return examples


def generate_trending_examples(count: int = 18) -> list[dict]:
    """Generate trending() operator examples."""
    examples = []

    for i, trigger in enumerate(TRENDING_TRIGGERS):
        if len(examples) >= count:
            break
        topic = TOPICS[i % len(TOPICS)]
        nl = trigger.format(topic=topic)
        query = f'trending(abs:"{topic}")'

        examples.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "operator",
            "operator": "trending",
        })

    return examples


def generate_similar_examples(count: int = 21) -> list[dict]:
    """Generate similar() operator examples."""
    examples = []
    targets = PAPER_TARGETS + AUTHOR_TARGETS

    for i, trigger in enumerate(SIMILAR_TRIGGERS):
        if len(examples) >= count:
            break
        target = targets[i % len(targets)]
        nl = trigger.format(target=target)

        # Build query
        author_names = ["hawking", "einstein", "penrose", "riess", "perlmutter", "weinberg", "rees", "salpeter", "chandrasekhar", "sunyaev"]
        matched_author = None
        for name in author_names:
            if name.lower() in target.lower():
                matched_author = name.capitalize()
                break
        if matched_author:
            query = f"similar(author:{matched_author})"
        elif "this paper" in target.lower() or "that article" in target.lower():
            query = "similar(bibcode:PLACEHOLDER)"
        else:
            query = f'similar(abs:"{target}")'

        examples.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "operator",
            "operator": "similar",
        })

    return examples


def generate_useful_examples(count: int = 19) -> list[dict]:
    """Generate useful() operator examples."""
    examples = []

    for i, trigger in enumerate(USEFUL_TRIGGERS):
        if len(examples) >= count:
            break
        topic = TOPICS[i % len(TOPICS)]
        nl = trigger.format(topic=topic)
        query = f'useful(abs:"{topic}")'

        examples.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "operator",
            "operator": "useful",
        })

    return examples


def generate_reviews_examples(count: int = 21) -> list[dict]:
    """Generate reviews() operator examples."""
    examples = []

    for i, trigger in enumerate(REVIEWS_TRIGGERS):
        if len(examples) >= count:
            break
        topic = TOPICS[i % len(TOPICS)]
        nl = trigger.format(topic=topic)
        query = f'reviews(abs:"{topic}")'

        examples.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "operator",
            "operator": "reviews",
        })

    return examples


def generate_topn_examples(count: int = 18) -> list[dict]:
    """Generate topn() operator examples."""
    examples = []

    # Extract N from trigger patterns
    for i, trigger in enumerate(TOPN_TRIGGERS):
        if len(examples) >= count:
            break
        topic = TOPICS[i % len(TOPICS)]
        nl = trigger.format(topic=topic)

        # Determine N from trigger
        n = 10  # default
        if "top 5" in trigger:
            n = 5
        elif "top 10" in trigger:
            n = 10
        elif "top 20" in trigger:
            n = 20
        elif "top 100" in trigger:
            n = 100
        elif "most cited" in trigger or "highest cited" in trigger or "highly cited" in trigger:
            n = 10  # default for "most cited" queries

        query = f'topn({n}, abs:"{topic}", citation_count)'

        examples.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "operator",
            "operator": "topn",
        })

    return examples


def generate_combined_examples() -> list[dict]:
    """Generate combined operator + filter examples."""
    examples = []

    # Citations + date filter
    combined_templates = [
        ("citations to dark matter papers from 2020", 'citations(abs:"dark matter" year:2020)'),
        ("who cited Hawking's papers since 2015", "citations(author:Hawking) year:[2015 TO *]"),
        ("recent papers citing the CMB discovery", 'citations(abs:"cosmic microwave background") year:[2020 TO *]'),
        # References + filter
        ("refereed papers in the references of Planck 2018", 'references(abs:"Planck 2018") property:refereed'),
        ("what sources did the LIGO paper cite", 'references(abs:"LIGO gravitational wave")'),
        # Trending + date
        ("hot topics in exoplanets this year", 'trending(abs:"exoplanets") year:2025'),
        ("currently trending in machine learning astronomy", 'trending(abs:"machine learning astronomy")'),
        # Similar + author
        ("papers similar to Weinberg's cosmology work", 'similar(author:Weinberg abs:"cosmology")'),
        ("find similar to the dark energy papers", 'similar(abs:"dark energy")'),
        # Useful + filter
        ("foundational papers on galaxy formation by Rees", 'useful(abs:"galaxy formation" author:Rees)'),
        ("must-read papers on gravitational lensing", 'useful(abs:"gravitational lensing")'),
        # Reviews + date
        ("recent review articles on black holes", 'reviews(abs:"black holes") year:[2020 TO *]'),
        ("survey papers on stellar evolution from 2019", 'reviews(abs:"stellar evolution") year:2019'),
        # Topn + filter
        ("top 10 refereed papers on supernovae", 'topn(10, abs:"supernovae" property:refereed, citation_count)'),
        ("most cited papers on neutron stars from 2015", 'topn(10, abs:"neutron stars" year:[2015 TO *], citation_count)'),
    ]

    for nl, query in combined_templates:
        # Determine operator from query
        op = "unknown"
        for operator in ["citations", "references", "trending", "similar", "useful", "reviews", "topn"]:
            if query.startswith(operator + "("):
                op = operator
                break

        examples.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "operator",
            "operator": op,
        })

    return examples


def generate_all_examples() -> list[dict]:
    """Generate examples for all operators."""
    all_examples = []

    # Generate 15+ examples for each operator as required
    all_examples.extend(generate_citations_examples(18))
    all_examples.extend(generate_references_examples(16))
    all_examples.extend(generate_trending_examples(18))
    all_examples.extend(generate_similar_examples(21))
    all_examples.extend(generate_useful_examples(19))
    all_examples.extend(generate_reviews_examples(21))
    all_examples.extend(generate_topn_examples(18))

    # Add combined examples
    all_examples.extend(generate_combined_examples())

    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate operator-based training examples"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/generated/operator_examples.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Generate examples
    examples = generate_all_examples()

    # Remove duplicates based on natural_language
    seen = set()
    unique_examples = []
    for ex in examples:
        nl_normalized = ex["natural_language"].lower().strip()
        if nl_normalized not in seen:
            seen.add(nl_normalized)
            unique_examples.append(ex)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(args.output, "w") as f:
        json.dump(unique_examples, f, indent=2)

    # Print summary
    print(f"Generated {len(unique_examples)} unique operator examples")

    # Count by operator
    operator_counts = {}
    for ex in unique_examples:
        op = ex.get("operator", "unknown")
        operator_counts[op] = operator_counts.get(op, 0) + 1

    print(f"\nTotal operators covered: {len(operator_counts)}")
    print("\nExamples per operator:")
    for op, count in sorted(operator_counts.items()):
        status = "✓" if count >= 15 else "⚠"
        print(f"  {status} {op}: {count}")

    # Check for operators with < 15 examples
    under_target = [op for op, c in operator_counts.items() if c < 15]
    if under_target:
        print(f"\nWarning: Operators with < 15 examples: {under_target}")

    print(f"\nOutput written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
