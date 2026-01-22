#!/usr/bin/env python3
"""
Generate training data for database/collection queries.

Creates NL-query pairs for database filtering, which had limited coverage
in the original training data. This addresses the data model coverage gap
identified in the US-002 audit.

Usage:
    python scripts/generate_collection_examples.py \
        --output data/datasets/generated/collection_examples.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Database-specific topics and natural language variations
DATABASE_CONFIG = {
    "astronomy": {
        "nl_triggers": [
            "astronomy papers",
            "astrophysics papers",
            "astrophysics literature",
            "astronomical research",
            "astronomy articles",
            "astrophysics research",
            "papers from the astronomy database",
            "astronomy journal articles",
            "astronomy collection papers",
            "astronomical studies",
            "astrophysical papers",
            "papers in astrophysics",
        ],
        "topics": [
            "dark matter",
            "black holes",
            "galaxy formation",
            "stellar evolution",
            "exoplanets",
            "gravitational waves",
            "supernovae",
            "neutron stars",
            "cosmic microwave background",
            "quasars",
            "galaxy clusters",
            "star formation",
            "planetary nebulae",
            "active galactic nuclei",
            "binary stars",
        ],
    },
    "physics": {
        "nl_triggers": [
            "physics papers",
            "physics literature",
            "physics research",
            "physics articles",
            "papers from the physics database",
            "physics collection papers",
            "physics journal articles",
            "physics studies",
            "physical sciences papers",
            "papers in physics",
        ],
        "topics": [
            "quantum mechanics",
            "string theory",
            "particle physics",
            "condensed matter",
            "quantum field theory",
            "general relativity",
            "nuclear physics",
            "plasma physics",
            "atomic physics",
            "quantum computing",
            "superconductivity",
            "electromagnetism",
            "statistical mechanics",
            "high energy physics",
        ],
    },
    "general": {
        "nl_triggers": [
            "general science papers",
            "interdisciplinary papers",
            "general collection papers",
            "papers from the general database",
            "general science literature",
            "cross-disciplinary research",
            "multidisciplinary papers",
            "general science articles",
        ],
        "topics": [
            "climate science",
            "biophysics",
            "scientific computing",
            "data science",
            "numerical methods",
            "computational science",
            "science policy",
            "research methods",
            "scientific instruments",
            "measurement techniques",
        ],
    },
    "earthscience": {
        "nl_triggers": [
            "earth science papers",
            "planetary science papers",
            "geoscience papers",
            "earth science literature",
            "planetary science research",
            "geoscience research",
            "papers from the earth science database",
            "earth science collection papers",
            "earth and planetary papers",
            "geophysics papers",
            "geology papers",
            "atmospheric science papers",
            "oceanography papers",
            "climate research papers",
            "earth system science papers",
            "space weather research",
            "heliophysics papers",
        ],
        "topics": [
            "climate change",
            "planetary atmospheres",
            "Mars geology",
            "Venus atmosphere",
            "lunar science",
            "asteroid impacts",
            "meteorites",
            "plate tectonics",
            "seismology",
            "oceanography",
            "atmospheric dynamics",
            "solar wind",
            "magnetospheres",
            "space weather",
            "planetary surfaces",
            "ice sheets",
            "paleoclimate",
            "geochemistry",
            "mineral physics",
        ],
    },
}

# Natural language templates for generating diverse examples
SIMPLE_TEMPLATES = [
    # Database-only queries - only use complete phrase triggers
    "{nl_trigger}",
    "find {nl_trigger}",
    "show me {nl_trigger}",
    "search for {nl_trigger}",
    "looking for {nl_trigger}",
]

TOPIC_TEMPLATES = [
    # Database + topic
    "{nl_trigger} about {topic}",
    "{nl_trigger} on {topic}",
    "find {nl_trigger} about {topic}",
    "show me {nl_trigger} on {topic}",
    "{nl_trigger} covering {topic}",
    "{nl_trigger} discussing {topic}",
    "{nl_trigger} related to {topic}",
    "search {nl_trigger} for {topic}",
]

DATE_TEMPLATES = [
    # Database + year
    "{nl_trigger} from {year}",
    "{nl_trigger} published in {year}",
    "recent {nl_trigger}",
    "latest {nl_trigger}",
    # Database + topic + year
    "{nl_trigger} about {topic} from {year}",
    "{nl_trigger} on {topic} published in {year}",
    "recent {nl_trigger} about {topic}",
]

COMBINED_TEMPLATES = [
    # Database + refereed
    "refereed {nl_trigger}",
    "peer-reviewed {nl_trigger}",
    "refereed {nl_trigger} about {topic}",
    # Database + author
    "{nl_trigger} by {author}",
    "{author}'s {nl_trigger}",
    # Complex combinations
    "refereed {nl_trigger} about {topic} from {year}",
    "{nl_trigger} about {topic} by {author}",
    "recent refereed {nl_trigger} on {topic}",
]

# Sample authors for combined queries
AUTHORS = [
    "Smith",
    "Jones",
    "Williams",
    "Brown",
    "Davis",
    "Miller",
    "Wilson",
    "Moore",
    "Taylor",
    "Anderson",
]

# Years for date-based queries
YEARS = list(range(2018, 2026))


def generate_simple_examples(database: str, count: int = 5) -> list[dict]:
    """Generate simple database-only examples."""
    examples = []
    config = DATABASE_CONFIG[database]

    for template in random.sample(SIMPLE_TEMPLATES, min(count, len(SIMPLE_TEMPLATES))):
        nl_trigger = random.choice(config["nl_triggers"])
        nl = template.format(nl_trigger=nl_trigger)
        query = f"database:{database}"

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "collection",
            }
        )

    return examples


def generate_topic_examples(database: str, count: int = 10) -> list[dict]:
    """Generate database + topic examples."""
    examples = []
    config = DATABASE_CONFIG[database]

    for _ in range(count):
        template = random.choice(TOPIC_TEMPLATES)
        nl_trigger = random.choice(config["nl_triggers"])
        topic = random.choice(config["topics"])

        nl = template.format(nl_trigger=nl_trigger, topic=topic)
        query = f'database:{database} abs:"{topic}"'

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "collection",
            }
        )

    return examples


def generate_date_examples(database: str, count: int = 5) -> list[dict]:
    """Generate database + date examples."""
    examples = []
    config = DATABASE_CONFIG[database]

    for _ in range(count):
        template = random.choice(DATE_TEMPLATES)
        nl_trigger = random.choice(config["nl_triggers"])
        topic = random.choice(config["topics"])
        year = random.choice(YEARS)

        if "recent" in template or "latest" in template:
            nl = template.format(nl_trigger=nl_trigger, topic=topic)
            query = f"database:{database} pubdate:[2023-01 TO 2026-12]"
        elif "{topic}" in template:
            nl = template.format(nl_trigger=nl_trigger, topic=topic, year=year)
            query = f'database:{database} abs:"{topic}" year:{year}'
        else:
            nl = template.format(nl_trigger=nl_trigger, year=year)
            query = f"database:{database} year:{year}"

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "collection",
            }
        )

    return examples


def generate_combined_examples(database: str, count: int = 5) -> list[dict]:
    """Generate database + other filters examples."""
    examples = []
    config = DATABASE_CONFIG[database]

    for _ in range(count):
        template = random.choice(COMBINED_TEMPLATES)
        nl_trigger = random.choice(config["nl_triggers"])
        topic = random.choice(config["topics"])
        author = random.choice(AUTHORS)
        year = random.choice(YEARS)

        nl = template.format(nl_trigger=nl_trigger, topic=topic, author=author, year=year)

        # Build query based on what's in the template
        query_parts = [f"database:{database}"]
        if "{topic}" in template:
            query_parts.append(f'abs:"{topic}"')
        if "refereed" in template or "peer-reviewed" in template:
            query_parts.append("property:refereed")
        if "{author}" in template:
            query_parts.append(f'author:"{author}"')
        if "{year}" in template:
            query_parts.append(f"year:{year}")

        query = " ".join(query_parts)

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "collection",
            }
        )

    return examples


def generate_all_examples() -> list[dict]:
    """Generate examples for all databases."""
    all_examples = []

    # Astronomy: 20+ examples
    all_examples.extend(generate_simple_examples("astronomy", count=5))
    all_examples.extend(generate_topic_examples("astronomy", count=10))
    all_examples.extend(generate_date_examples("astronomy", count=5))
    all_examples.extend(generate_combined_examples("astronomy", count=5))

    # Physics: 20+ examples
    all_examples.extend(generate_simple_examples("physics", count=5))
    all_examples.extend(generate_topic_examples("physics", count=10))
    all_examples.extend(generate_date_examples("physics", count=5))
    all_examples.extend(generate_combined_examples("physics", count=5))

    # General: 10+ examples
    all_examples.extend(generate_simple_examples("general", count=3))
    all_examples.extend(generate_topic_examples("general", count=5))
    all_examples.extend(generate_date_examples("general", count=3))
    all_examples.extend(generate_combined_examples("general", count=2))

    # Earthscience: 20+ examples
    all_examples.extend(generate_simple_examples("earthscience", count=5))
    all_examples.extend(generate_topic_examples("earthscience", count=10))
    all_examples.extend(generate_date_examples("earthscience", count=5))
    all_examples.extend(generate_combined_examples("earthscience", count=5))

    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate database/collection training examples"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/generated/collection_examples.json"),
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
        if ex["natural_language"] not in seen:
            seen.add(ex["natural_language"])
            unique_examples.append(ex)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(args.output, "w") as f:
        json.dump(unique_examples, f, indent=2)

    # Print summary
    print(f"Generated {len(unique_examples)} unique collection/database examples")

    # Count by database
    db_counts = {}
    for ex in unique_examples:
        for db in ["astronomy", "physics", "general", "earthscience"]:
            if f"database:{db}" in ex["ads_query"]:
                db_counts[db] = db_counts.get(db, 0) + 1
                break

    print("\nExamples per database:")
    for db, count in sorted(db_counts.items()):
        print(f"  {db}: {count}")

    print(f"\nOutput written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
