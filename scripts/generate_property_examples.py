#!/usr/bin/env python3
"""
Generate training data for property-based queries.

Creates NL-query pairs for property filtering, which has limited coverage
in the original training data (only 5 of 21 property values are represented).
This addresses the data model coverage gap identified in the US-002 audit.

Underrepresented properties to cover:
- notrefereed, eprint, catalog, article, nonarticle, inproceedings
- associated, toc, presentation, esource, inspire, library_catalog
- ads_openaccess, author_openaccess, eprint_openaccess, pub_openaccess
- ocr_abstract

Usage:
    python scripts/generate_property_examples.py \
        --output data/datasets/generated/property_examples.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Property-specific natural language configurations
# Each property has triggers (NL phrases that map to it) and topics for combined queries
PROPERTY_CONFIG = {
    # Non-refereed content
    "notrefereed": {
        "nl_triggers": [
            "non-peer-reviewed papers",
            "non-refereed papers",
            "papers that aren't peer-reviewed",
            "papers without peer review",
            "unrefereed papers",
            "papers that have not been peer reviewed",
            "non-peer-reviewed research",
            "non-refereed articles",
        ],
        "negation_forms": [
            "papers that are not refereed",
            "papers not peer reviewed",
            "exclude peer-reviewed papers",
            "only non-refereed",
        ],
        "topics": ["dark matter", "exoplanets", "machine learning", "cosmology"],
    },
    # Preprints
    "eprint": {
        "nl_triggers": [
            "preprints",
            "arxiv papers",
            "preprint papers",
            "arxiv preprints",
            "unpublished preprints",
            "papers on arxiv",
            "pre-publication papers",
            "preprint articles",
            "eprints",
            "e-prints",
        ],
        "negation_forms": [],
        "topics": ["gravitational waves", "black holes", "neural networks", "quasars"],
    },
    # Catalogs
    "catalog": {
        "nl_triggers": [
            "data catalogs",
            "catalog papers",
            "astronomical catalogs",
            "catalog publications",
            "survey catalogs",
            "catalog data products",
            "catalog entries",
            "data catalog publications",
        ],
        "negation_forms": [],
        "topics": ["stellar populations", "galaxies", "variable stars", "transients"],
    },
    # Regular articles
    "article": {
        "nl_triggers": [
            "regular articles",
            "journal articles only",
            "standard articles",
            "published articles",
            "full journal articles",
            "regular papers",
            "standard journal papers",
            "traditional articles",
        ],
        "negation_forms": [],
        "topics": ["spectroscopy", "photometry", "astrometry", "high-energy physics"],
    },
    # Non-articles
    "nonarticle": {
        "nl_triggers": [
            "non-article papers",
            "non-articles",
            "papers that are not articles",
            "meeting abstracts",
            "conference abstracts",
            "posters and abstracts",
            "non-article publications",
            "abstracts and proceedings",
        ],
        "negation_forms": [
            "exclude regular articles",
            "not regular articles",
            "only non-articles",
        ],
        "topics": ["AGN", "star formation", "planetary atmospheres", "instrumentation"],
    },
    # Conference proceedings
    "inproceedings": {
        "nl_triggers": [
            "conference papers",
            "conference proceedings",
            "proceedings papers",
            "papers from conferences",
            "conference publications",
            "proceeding articles",
            "conference contributions",
            "symposium papers",
        ],
        "negation_forms": [],
        "topics": [
            "data reduction",
            "calibration",
            "telescope design",
            "detector technology",
        ],
    },
    # Associated papers
    "associated": {
        "nl_triggers": [
            "papers with associated articles",
            "articles with companion papers",
            "papers having associated content",
            "publications with related articles",
            "papers with linked articles",
            "articles with associated publications",
        ],
        "negation_forms": [],
        "topics": ["surveys", "large programs", "multi-wavelength studies"],
    },
    # Table of contents
    "toc": {
        "nl_triggers": [
            "papers with table of contents",
            "articles with toc",
            "publications with table of contents",
            "papers including toc",
            "articles having table of contents",
        ],
        "negation_forms": [],
        "topics": ["reviews", "textbooks", "comprehensive studies"],
    },
    # Presentations
    "presentation": {
        "nl_triggers": [
            "papers with presentations",
            "articles with media presentations",
            "publications with video presentations",
            "papers having presentations",
            "articles with recorded talks",
            "papers with associated presentations",
            "publications with slides",
        ],
        "negation_forms": [],
        "topics": [
            "observational techniques",
            "data analysis methods",
            "survey results",
        ],
    },
    # Electronic sources
    "esource": {
        "nl_triggers": [
            "papers with electronic sources",
            "articles with online access",
            "publications with electronic versions",
            "papers available electronically",
            "articles with e-sources",
            "papers with online versions",
            "electronically available papers",
        ],
        "negation_forms": [],
        "topics": ["recent research", "modern astrophysics", "current studies"],
    },
    # INSPIRE records
    "inspire": {
        "nl_triggers": [
            "papers in INSPIRE",
            "INSPIRE database papers",
            "articles indexed in INSPIRE",
            "INSPIRE records",
            "papers from INSPIRE",
            "INSPIRE-indexed articles",
            "publications in INSPIRE database",
        ],
        "negation_forms": [],
        "topics": [
            "particle physics",
            "high energy physics",
            "theoretical physics",
            "collider physics",
        ],
    },
    # Library catalogs
    "library_catalog": {
        "nl_triggers": [
            "papers in library catalogs",
            "library catalog entries",
            "cataloged library papers",
            "papers indexed in library catalogs",
            "library-cataloged publications",
            "articles in library systems",
        ],
        "negation_forms": [],
        "topics": [
            "historical astronomy",
            "classic papers",
            "foundational research",
        ],
    },
    # ADS open access
    "ads_openaccess": {
        "nl_triggers": [
            "papers with ADS open access",
            "ADS open access papers",
            "open access from ADS",
            "papers freely available through ADS",
            "ADS-hosted open access articles",
            "papers with free ADS access",
        ],
        "negation_forms": [],
        "topics": ["solar physics", "stellar astrophysics", "galactic astronomy"],
    },
    # Author open access
    "author_openaccess": {
        "nl_triggers": [
            "author-submitted open access papers",
            "papers with author-provided open access",
            "author open access versions",
            "papers with author-hosted PDFs",
            "author-shared open access articles",
            "papers with author-uploaded versions",
        ],
        "negation_forms": [],
        "topics": ["theoretical models", "simulations", "numerical methods"],
    },
    # Eprint open access
    "eprint_openaccess": {
        "nl_triggers": [
            "papers with arxiv open access",
            "preprint open access papers",
            "open access from preprint servers",
            "papers freely available on arxiv",
            "eprint open access versions",
            "open access preprints",
        ],
        "negation_forms": [],
        "topics": ["cosmology", "extragalactic astronomy", "dark energy"],
    },
    # Publisher open access
    "pub_openaccess": {
        "nl_triggers": [
            "publisher open access papers",
            "papers with publisher open access",
            "open access from publishers",
            "publisher-provided open access",
            "papers freely available from publisher",
            "journal open access articles",
        ],
        "negation_forms": [],
        "topics": ["nature papers", "science papers", "high-impact journals"],
    },
    # OCR abstracts
    "ocr_abstract": {
        "nl_triggers": [
            "papers with OCR abstracts",
            "articles with scanned abstracts",
            "papers with digitized abstracts",
            "OCR-processed abstracts",
            "papers with optical character recognition abstracts",
            "scanned paper abstracts",
        ],
        "negation_forms": [],
        "topics": ["historical observations", "early discoveries", "classic astronomy"],
    },
    # Software property (has some coverage, but add more)
    "software": {
        "nl_triggers": [
            "software papers",
            "papers about software",
            "software publications",
            "code papers",
            "papers describing software",
            "software documentation papers",
            "papers with software",
        ],
        "negation_forms": [],
        "topics": ["data pipelines", "analysis tools", "visualization software"],
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
    # Property + refereed
    "refereed {nl_trigger}",
    "peer-reviewed {nl_trigger}",
    # Property + date
    "{nl_trigger} from {year}",
    "recent {nl_trigger}",
    "{nl_trigger} published in {year}",
    # Property + topic + date
    "{nl_trigger} about {topic} from {year}",
    "recent {nl_trigger} on {topic}",
    # Property + author
    "{nl_trigger} by {author}",
]

NEGATION_TEMPLATES = [
    "{negation_form}",
    "find {negation_form}",
    "show me {negation_form}",
    "search for {negation_form}",
    "{negation_form} about {topic}",
]

# Sample authors for combined queries
AUTHORS = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Wilson",
    "Taylor",
]

# Years for date-based queries
YEARS = list(range(2018, 2026))


def generate_simple_examples(property_name: str, count: int = 3) -> list[dict]:
    """Generate simple property-only examples."""
    examples = []
    config = PROPERTY_CONFIG[property_name]

    for i in range(min(count, len(config["nl_triggers"]))):
        template = random.choice(SIMPLE_TEMPLATES)
        nl_trigger = config["nl_triggers"][i]
        nl = template.format(nl_trigger=nl_trigger)
        query = f"property:{property_name}"

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "property",
            }
        )

    return examples


def generate_topic_examples(property_name: str, count: int = 2) -> list[dict]:
    """Generate property + topic examples."""
    examples = []
    config = PROPERTY_CONFIG[property_name]

    for _ in range(count):
        template = random.choice(TOPIC_TEMPLATES)
        nl_trigger = random.choice(config["nl_triggers"])
        topic = random.choice(config["topics"])

        nl = template.format(nl_trigger=nl_trigger, topic=topic)
        query = f'property:{property_name} abs:"{topic}"'

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "property",
            }
        )

    return examples


def generate_combined_examples(property_name: str, count: int = 2) -> list[dict]:
    """Generate property + other filters examples."""
    examples = []
    config = PROPERTY_CONFIG[property_name]

    # Properties that conflict with "refereed" filter
    # Don't combine these with refereed/peer-reviewed templates
    refereed_conflicting = {"notrefereed", "refereed", "nonarticle", "eprint"}

    # Filter templates based on property
    available_templates = COMBINED_TEMPLATES.copy()
    if property_name in refereed_conflicting:
        available_templates = [
            t for t in available_templates
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
        query_parts = [f"property:{property_name}"]
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
                "category": "property",
            }
        )

    return examples


def generate_negation_examples(property_name: str) -> list[dict]:
    """Generate negation-based examples for properties with negation forms."""
    examples = []
    config = PROPERTY_CONFIG[property_name]

    if not config.get("negation_forms"):
        return examples

    for negation_form in config["negation_forms"]:
        # Simple negation
        template = random.choice(NEGATION_TEMPLATES[:4])  # Exclude topic variant
        nl = template.format(negation_form=negation_form)
        query = f"property:{property_name}"

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "property",
            }
        )

        # Negation with topic
        if config["topics"]:
            topic = random.choice(config["topics"])
            nl = NEGATION_TEMPLATES[4].format(negation_form=negation_form, topic=topic)
            query = f'property:{property_name} abs:"{topic}"'

            examples.append(
                {
                    "natural_language": nl,
                    "ads_query": query,
                    "category": "property",
                }
            )

    return examples


def generate_all_examples() -> list[dict]:
    """Generate examples for all underrepresented properties."""
    all_examples = []

    for property_name, config in PROPERTY_CONFIG.items():
        # Generate 5+ examples for each property
        # 3 simple + 2 topic + 2 combined = 7 base examples
        all_examples.extend(generate_simple_examples(property_name, count=3))
        all_examples.extend(generate_topic_examples(property_name, count=2))
        all_examples.extend(generate_combined_examples(property_name, count=2))

        # Add negation examples where applicable
        all_examples.extend(generate_negation_examples(property_name))

    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate property-based training examples"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/generated/property_examples.json"),
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
    print(f"Generated {len(unique_examples)} unique property examples")

    # Count by property
    prop_counts = {}
    for ex in unique_examples:
        query = ex["ads_query"]
        # Extract property value from query (first property: match)
        import re

        match = re.search(r"property:(\w+)", query)
        if match:
            prop = match.group(1)
            prop_counts[prop] = prop_counts.get(prop, 0) + 1

    print("\nExamples per property:")
    for prop, count in sorted(prop_counts.items()):
        print(f"  {prop}: {count}")

    # Check for properties with < 5 examples
    under_target = [p for p, c in prop_counts.items() if c < 5]
    if under_target:
        print(f"\nWarning: Properties with < 5 examples: {under_target}")

    print(f"\nOutput written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
