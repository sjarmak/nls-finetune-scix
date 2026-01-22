#!/usr/bin/env python3
"""
Generate training data for doctype-based queries.

Creates NL-query pairs for doctype filtering, which has limited coverage
in the original training data (only 9 of 22 doctype values are represented).
This addresses the data model coverage gap identified in the US-002 audit.

Underrepresented doctypes to cover (13 values):
- abstract, bookreview, circular, editorial, erratum
- inbook, mastersthesis, misc, newsletter, obituary
- pressrelease, talk, proceedings

Usage:
    python scripts/generate_doctype_examples.py \
        --output data/datasets/generated/doctype_examples.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Doctype-specific natural language configurations
# Each doctype has triggers (NL phrases that map to it) and topics for combined queries
DOCTYPE_CONFIG = {
    # Meeting abstracts
    "abstract": {
        "nl_triggers": [
            "meeting abstracts",
            "conference abstracts",
            "AAS meeting abstracts",
            "symposium abstracts",
            "poster abstracts",
            "talk abstracts",
            "abstracts from meetings",
            "research abstracts",
        ],
        "topics": [
            "exoplanets",
            "black holes",
            "star formation",
            "galaxies",
            "cosmology",
        ],
    },
    # Book reviews
    "bookreview": {
        "nl_triggers": [
            "book reviews",
            "reviews of books",
            "published book reviews",
            "book critiques",
            "book evaluations",
            "reviews of astronomy books",
            "textbook reviews",
            "reviews of scientific books",
        ],
        "topics": [
            "cosmology",
            "astrophysics",
            "general relativity",
            "stellar evolution",
            "planetary science",
        ],
    },
    # Circulars (ATels, GCNs, etc.)
    "circular": {
        "nl_triggers": [
            "circulars",
            "astronomer's telegrams",
            "ATels",
            "GCN circulars",
            "transient circulars",
            "IAU circulars",
            "observation circulars",
            "discovery circulars",
        ],
        "topics": [
            "supernovae",
            "gamma-ray bursts",
            "transients",
            "novae",
            "fast radio bursts",
        ],
    },
    # Editorial content
    "editorial": {
        "nl_triggers": [
            "editorials",
            "editor's notes",
            "editorial content",
            "journal editorials",
            "editor comments",
            "editorial pieces",
            "editorial articles",
            "editor introductions",
        ],
        "topics": [
            "astronomy policy",
            "scientific publishing",
            "open access",
            "peer review",
            "research ethics",
        ],
    },
    # Errata to articles
    "erratum": {
        "nl_triggers": [
            "errata",
            "corrections",
            "erratum notices",
            "corrigenda",
            "published corrections",
            "article corrections",
            "erratum to papers",
            "correction notices",
        ],
        "topics": [
            "spectroscopy",
            "photometry",
            "data analysis",
            "measurements",
            "calibration",
        ],
    },
    # Book chapters
    "inbook": {
        "nl_triggers": [
            "book chapters",
            "chapters in books",
            "book sections",
            "contributions to books",
            "articles in books",
            "book contributions",
            "chapters from textbooks",
            "handbook chapters",
        ],
        "topics": [
            "stellar atmospheres",
            "interstellar medium",
            "galactic structure",
            "radiative transfer",
            "nucleosynthesis",
        ],
    },
    # Master's theses
    "mastersthesis": {
        "nl_triggers": [
            "masters theses",
            "master's thesis",
            "MS theses",
            "MSc theses",
            "graduate theses",
            "master's degree theses",
            "masters dissertations",
            "MSc dissertations",
        ],
        "topics": [
            "variable stars",
            "binary stars",
            "stellar spectra",
            "planetary atmospheres",
            "data reduction",
        ],
    },
    # Miscellaneous documents
    "misc": {
        "nl_triggers": [
            "miscellaneous documents",
            "uncategorized papers",
            "other documents",
            "miscellaneous publications",
            "other records",
            "miscellaneous entries",
            "misc documents",
            "other content",
        ],
        "topics": [
            "astronomy history",
            "observatory reports",
            "annual reports",
            "facility documentation",
            "instrument manuals",
        ],
    },
    # Newsletters
    "newsletter": {
        "nl_triggers": [
            "newsletters",
            "society newsletters",
            "observatory newsletters",
            "AAS newsletters",
            "department newsletters",
            "astronomical newsletters",
            "research newsletters",
            "institutional newsletters",
        ],
        "topics": [
            "observatory news",
            "mission updates",
            "instrument status",
            "community announcements",
            "funding news",
        ],
    },
    # Obituaries
    "obituary": {
        "nl_triggers": [
            "obituaries",
            "memorial articles",
            "tributes",
            "memorial notices",
            "biographical memoirs",
            "remembrances",
            "obituary notices",
            "in memoriam articles",
        ],
        "topics": [
            "astronomers",
            "astrophysicists",
            "Nobel laureates",
            "observatory directors",
            "pioneers",
        ],
    },
    # Press releases
    "pressrelease": {
        "nl_triggers": [
            "press releases",
            "news releases",
            "media releases",
            "NASA press releases",
            "ESA press releases",
            "observatory announcements",
            "discovery announcements",
            "public announcements",
        ],
        "topics": [
            "exoplanet discoveries",
            "black hole images",
            "gravitational waves",
            "new missions",
            "major discoveries",
        ],
    },
    # Research talks
    "talk": {
        "nl_triggers": [
            "conference talks",
            "research talks",
            "invited talks",
            "presentations",
            "talks at conferences",
            "colloquium talks",
            "seminar talks",
            "oral presentations",
        ],
        "topics": [
            "machine learning",
            "data science",
            "numerical simulations",
            "instrumentation",
            "survey results",
        ],
    },
    # Conference proceedings volumes
    "proceedings": {
        "nl_triggers": [
            "proceedings volumes",
            "conference proceedings",
            "proceedings books",
            "symposium proceedings",
            "workshop proceedings",
            "meeting proceedings",
            "proceedings of conferences",
            "conference volumes",
        ],
        "topics": [
            "IAU symposia",
            "SPIE conferences",
            "ASP conferences",
            "AAS meetings",
            "European conferences",
        ],
    },
}

# Natural language templates for generating diverse examples
SIMPLE_TEMPLATES = [
    "find {nl_trigger}",
    "show me {nl_trigger}",
    "search for {nl_trigger}",
    "{nl_trigger}",
    "looking for {nl_trigger}",
    "get {nl_trigger}",
    "I want {nl_trigger}",
]

TOPIC_TEMPLATES = [
    "{nl_trigger} about {topic}",
    "{nl_trigger} on {topic}",
    "find {nl_trigger} about {topic}",
    "show me {nl_trigger} on {topic}",
    "{nl_trigger} discussing {topic}",
    "{nl_trigger} related to {topic}",
    "{topic} {nl_trigger}",
]

COMBINED_TEMPLATES = [
    # Doctype + refereed
    "refereed {nl_trigger}",
    "peer-reviewed {nl_trigger}",
    # Doctype + date
    "{nl_trigger} from {year}",
    "recent {nl_trigger}",
    "{nl_trigger} published in {year}",
    # Doctype + topic + date
    "{nl_trigger} about {topic} from {year}",
    "recent {nl_trigger} on {topic}",
    # Doctype + author
    "{nl_trigger} by {author}",
]

# Sample authors for combined queries
AUTHORS = [
    "Hawking",
    "Penrose",
    "Perlmutter",
    "Riess",
    "Schmidt",
    "Ghez",
    "Rubin",
    "Sagan",
    "Tyson",
    "Thorne",
]

# Years for date-based queries
YEARS = list(range(2018, 2026))


def generate_simple_examples(doctype_name: str, count: int = 3) -> list[dict]:
    """Generate simple doctype-only examples."""
    examples = []
    config = DOCTYPE_CONFIG[doctype_name]

    for i in range(min(count, len(config["nl_triggers"]))):
        template = random.choice(SIMPLE_TEMPLATES)
        nl_trigger = config["nl_triggers"][i]
        nl = template.format(nl_trigger=nl_trigger)
        query = f"doctype:{doctype_name}"

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "doctype",
            }
        )

    return examples


def generate_topic_examples(doctype_name: str, count: int = 2) -> list[dict]:
    """Generate doctype + topic examples."""
    examples = []
    config = DOCTYPE_CONFIG[doctype_name]

    for _ in range(count):
        template = random.choice(TOPIC_TEMPLATES)
        nl_trigger = random.choice(config["nl_triggers"])
        topic = random.choice(config["topics"])

        nl = template.format(nl_trigger=nl_trigger, topic=topic)
        query = f'doctype:{doctype_name} abs:"{topic}"'

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "doctype",
            }
        )

    return examples


def generate_combined_examples(doctype_name: str, count: int = 2) -> list[dict]:
    """Generate doctype + other filters examples."""
    examples = []
    config = DOCTYPE_CONFIG[doctype_name]

    # Some doctypes don't make sense with refereed filter
    # (abstracts, circulars, newsletters, press releases, obituaries, misc are typically not refereed)
    non_refereed_types = {
        "abstract",
        "circular",
        "newsletter",
        "pressrelease",
        "obituary",
        "misc",
        "editorial",
    }

    # Filter templates based on doctype
    available_templates = COMBINED_TEMPLATES.copy()
    if doctype_name in non_refereed_types:
        available_templates = [
            t
            for t in available_templates
            if "refereed" not in t and "peer-reviewed" not in t
        ]

    for _ in range(count):
        template = random.choice(available_templates)
        nl_trigger = random.choice(config["nl_triggers"])
        topic = random.choice(config["topics"])
        author = random.choice(AUTHORS)
        year = random.choice(YEARS)

        nl = template.format(
            nl_trigger=nl_trigger, topic=topic, author=author, year=year
        )

        # Build query based on what's in the template
        query_parts = [f"doctype:{doctype_name}"]
        if "{topic}" in template:
            query_parts.append(f'abs:"{topic}"')
        if "refereed" in template or "peer-reviewed" in template:
            query_parts.append("property:refereed")
        if "{author}" in template:
            query_parts.append(f'author:"{author}"')
        if "{year}" in template:
            query_parts.append(f"year:{year}")
        if "recent" in template:
            query_parts.append("pubdate:[2023-01 TO 2026-12]")

        query = " ".join(query_parts)

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "doctype",
            }
        )

    return examples


def generate_all_examples() -> list[dict]:
    """Generate examples for all underrepresented doctypes."""
    all_examples = []

    for doctype_name in DOCTYPE_CONFIG:
        # Generate 5+ examples for each doctype
        # 3 simple + 2 topic + 2 combined = 7 base examples
        all_examples.extend(generate_simple_examples(doctype_name, count=3))
        all_examples.extend(generate_topic_examples(doctype_name, count=2))
        all_examples.extend(generate_combined_examples(doctype_name, count=2))

    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate doctype-based training examples"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/generated/doctype_examples.json"),
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
    print(f"Generated {len(unique_examples)} unique doctype examples")

    # Count by doctype
    import re

    doctype_counts = {}
    for ex in unique_examples:
        query = ex["ads_query"]
        # Extract doctype value from query (first doctype: match)
        match = re.search(r"doctype:(\w+)", query)
        if match:
            dt = match.group(1)
            doctype_counts[dt] = doctype_counts.get(dt, 0) + 1

    print("\nExamples per doctype:")
    for dt, count in sorted(doctype_counts.items()):
        print(f"  {dt}: {count}")

    # Check for doctypes with < 5 examples
    under_target = [d for d, c in doctype_counts.items() if c < 5]
    if under_target:
        print(f"\nWarning: Doctypes with < 5 examples: {under_target}")

    print(f"\nOutput written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
