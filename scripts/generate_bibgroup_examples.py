#!/usr/bin/env python3
"""
Generate training data for bibgroup-based queries.

Creates NL-query pairs for bibgroup (telescope/survey) filtering, which has
limited coverage in the original training data (only 5 of 53 bibgroups are
represented). This addresses the data model coverage gap identified in US-002.

Uses bibgroup_synonyms.json to map common telescope names to bibgroup codes.

Usage:
    python scripts/generate_bibgroup_examples.py \
        --output data/datasets/generated/bibgroup_examples.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Load bibgroup synonyms file (expected in data/model/)
SYNONYMS_PATH = Path(__file__).parent.parent / "data/model/bibgroup_synonyms.json"

# Topics relevant to telescope/survey science
# These are general enough to work across most instruments
GENERAL_TOPICS = [
    "exoplanets",
    "galaxies",
    "black holes",
    "star formation",
    "cosmology",
    "supernovae",
    "quasars",
    "gravitational lensing",
    "stellar evolution",
    "dark matter",
]

# Telescope-specific topic mappings for better relevance
TELESCOPE_TOPICS = {
    # Space telescopes - optical/IR
    "HST": [
        "deep field",
        "galaxy morphology",
        "Cepheid variables",
        "supernovae",
        "planetary nebulae",
    ],
    "JWST": [
        "early universe",
        "protoplanetary disks",
        "exoplanet atmospheres",
        "high-z galaxies",
        "brown dwarfs",
    ],
    "Spitzer": [
        "infrared emission",
        "protostellar disks",
        "galaxy evolution",
        "infrared spectroscopy",
        "star formation",
    ],
    # X-ray observatories
    "Chandra": [
        "X-ray emission",
        "active galactic nuclei",
        "galaxy clusters",
        "neutron stars",
        "black hole accretion",
    ],
    "XMM": [
        "X-ray spectroscopy",
        "AGN variability",
        "hot gas",
        "X-ray binaries",
        "cluster mass",
    ],
    "NuSTAR": [
        "hard X-rays",
        "AGN coronae",
        "neutron star surfaces",
        "supernova remnants",
        "black hole spin",
    ],
    "RXTE": [
        "timing analysis",
        "X-ray pulsars",
        "accretion disk variability",
        "quasi-periodic oscillations",
        "X-ray bursts",
    ],
    "Swift": [
        "gamma-ray bursts",
        "transient sources",
        "afterglows",
        "tidal disruption events",
        "supernovae",
    ],
    # Gamma-ray
    "Fermi": [
        "gamma-ray emission",
        "pulsars",
        "blazars",
        "diffuse emission",
        "dark matter searches",
    ],
    # UV missions
    "GALEX": [
        "UV emission",
        "star formation rates",
        "hot stars",
        "galactic extinction",
        "UV luminosity function",
    ],
    "FUSE": [
        "far-UV spectroscopy",
        "interstellar medium",
        "O VI absorption",
        "hot gas",
        "deuterium abundance",
    ],
    "IUE": [
        "ultraviolet spectra",
        "stellar winds",
        "interstellar absorption",
        "chromospheric activity",
        "binary stars",
    ],
    "EUVE": [
        "extreme UV",
        "white dwarf atmospheres",
        "stellar coronae",
        "interstellar medium",
        "hot stars",
    ],
    # Infrared missions
    "IRAS": [
        "infrared sources",
        "dust emission",
        "circumstellar disks",
        "infrared galaxies",
        "zodiacal light",
    ],
    "WISE": [
        "brown dwarfs",
        "asteroids",
        "infrared all-sky",
        "YSOs",
        "AGN mid-IR",
    ],
    "NEOWISE": [
        "near-Earth objects",
        "asteroid properties",
        "comet surveys",
        "thermal infrared",
        "Solar System",
    ],
    # Exoplanet missions
    "Kepler": [
        "exoplanet transits",
        "stellar oscillations",
        "eclipsing binaries",
        "planet occurrence rates",
        "asteroseismology",
    ],
    "K2": [
        "extended mission",
        "ecliptic targets",
        "open clusters",
        "young stellar objects",
        "asteroids",
    ],
    "TESS": [
        "transiting planets",
        "bright stars",
        "all-sky survey",
        "stellar variability",
        "planet candidates",
    ],
    # Solar missions
    "SOHO": [
        "solar corona",
        "coronal mass ejections",
        "helioseismology",
        "solar wind",
        "comets",
    ],
    "STEREO": [
        "solar stereoscopy",
        "CME 3D structure",
        "heliospheric imaging",
        "solar energetic particles",
        "space weather",
    ],
    "SDO": [
        "solar dynamics",
        "EUV emission",
        "magnetic field",
        "flare physics",
        "coronal loops",
    ],
    # Astrometry
    "Gaia": [
        "parallaxes",
        "proper motions",
        "stellar distances",
        "galactic structure",
        "astrometric binaries",
    ],
    "Hipparcos": [
        "stellar parallaxes",
        "distance scale",
        "stellar kinematics",
        "HR diagram",
        "double stars",
    ],
    # Ground-based optical
    "VLT": [
        "spectroscopy",
        "adaptive optics",
        "galactic center",
        "exoplanet imaging",
        "high-z galaxies",
    ],
    "Keck": [
        "spectroscopy",
        "adaptive optics",
        "exoplanet imaging",
        "galaxy dynamics",
        "cosmology",
    ],
    "Gemini": [
        "mid-IR imaging",
        "AO observations",
        "transient follow-up",
        "stellar populations",
        "spectroscopy",
    ],
    "Subaru": [
        "wide-field imaging",
        "high-z surveys",
        "Lyman-alpha emitters",
        "galaxy evolution",
        "weak lensing",
    ],
    "CFHT": [
        "wide-field imaging",
        "MegaCam surveys",
        "galaxy clusters",
        "stellar streams",
        "weak lensing",
    ],
    "Pan-STARRS": [
        "transient surveys",
        "asteroid detection",
        "stellar streams",
        "photometric surveys",
        "moving objects",
    ],
    "SDSS": [
        "spectroscopic survey",
        "photometric survey",
        "galaxy redshifts",
        "quasar catalog",
        "stellar parameters",
    ],
    "2MASS": [
        "near-infrared survey",
        "point source catalog",
        "extended sources",
        "brown dwarf searches",
        "galactic structure",
    ],
    "UKIRT": [
        "infrared imaging",
        "galactic plane",
        "brown dwarf searches",
        "star forming regions",
        "spectroscopy",
    ],
    # Radio/mm observatories
    "ALMA": [
        "submillimeter",
        "protoplanetary disks",
        "molecular lines",
        "high-z dust",
        "star formation",
    ],
    "VLA": [
        "radio continuum",
        "radio transients",
        "HI surveys",
        "AGN jets",
        "pulsar timing",
    ],
    "VLBA": [
        "VLBI astrometry",
        "maser emission",
        "AGN structure",
        "proper motion",
        "parallaxes",
    ],
    "GBT": [
        "pulsar surveys",
        "HI spectroscopy",
        "fast radio bursts",
        "molecular lines",
        "CMB",
    ],
    "ARECIBO": [
        "pulsar timing",
        "radar astronomy",
        "SETI",
        "HI surveys",
        "fast radio bursts",
    ],
    "JCMT": [
        "submillimeter continuum",
        "molecular lines",
        "star forming regions",
        "debris disks",
        "polarimetry",
    ],
    "APEX": [
        "submillimeter",
        "spectral line surveys",
        "high-z sources",
        "Galactic plane",
        "dust emission",
    ],
    "LOFAR": [
        "low-frequency radio",
        "pulsars",
        "radio transients",
        "epoch of reionization",
        "galaxy clusters",
    ],
    "MeerKAT": [
        "radio continuum",
        "HI surveys",
        "pulsars",
        "transient searches",
        "galaxy evolution",
    ],
    "SKA": [
        "next-generation radio",
        "cosmic magnetism",
        "epoch of reionization",
        "pulsars",
        "HI cosmology",
    ],
    # Gravitational waves
    "LIGO": [
        "gravitational waves",
        "binary black holes",
        "neutron star mergers",
        "compact binaries",
        "detector sensitivity",
    ],
    "LISA": [
        "space-based interferometry",
        "massive black holes",
        "galactic binaries",
        "EMRI",
        "cosmological sources",
    ],
    # Observatories/institutions
    "NOAO": [
        "observing programs",
        "time allocation",
        "instrumentation",
        "survey data",
        "community access",
    ],
    "NOIRLab": [
        "observatory programs",
        "CTIO data",
        "Gemini programs",
        "Rubin Observatory",
        "community science",
    ],
    "CTIO": [
        "southern sky",
        "DECam",
        "photometry",
        "galaxy surveys",
        "transient follow-up",
    ],
    "KPNO": [
        "northern sky",
        "spectroscopy",
        "photometry",
        "stellar surveys",
        "time-domain",
    ],
    "ESO/Telescopes": [
        "VLT observations",
        "ALMA data",
        "survey programs",
        "instrumentation",
        "spectroscopy",
    ],
    "ESO": [
        "VLT observations",
        "ALMA data",
        "survey programs",
        "instrumentation",
        "spectroscopy",
    ],
    "CfA": [
        "redshift surveys",
        "stellar astrophysics",
        "cosmology",
        "theoretical research",
        "instrumentation",
    ],
    "NASA PubSpace": [
        "NASA missions",
        "public access",
        "mission data",
        "planetary science",
        "astrophysics",
    ],
    "SETI": [
        "technosignatures",
        "radio searches",
        "exoplanet biosignatures",
        "signal detection",
        "extraterrestrial intelligence",
    ],
    "Copernicus": [
        "UV spectroscopy",
        "interstellar medium",
        "stellar spectra",
        "absorption lines",
        "hot stars",
    ],
}

# Natural language templates for different query types
SIMPLE_TEMPLATES = [
    "{telescope_name} papers",
    "{telescope_name} observations",
    "{telescope_name} data",
    "papers using {telescope_name}",
    "research with {telescope_name}",
    "studies using {telescope_name} data",
    "{telescope_name} publications",
]

TOPIC_TEMPLATES = [
    "{telescope_name} {topic} papers",
    "{telescope_name} observations of {topic}",
    "{topic} studies using {telescope_name}",
    "{topic} research with {telescope_name}",
    "{telescope_name} {topic} observations",
]

COMBINED_TEMPLATES = [
    # With refereed filter
    "refereed {telescope_name} papers",
    "peer-reviewed {telescope_name} observations",
    "refereed papers using {telescope_name}",
    # With date filter
    "{telescope_name} papers from {year}",
    "recent {telescope_name} observations",
    "{telescope_name} data releases in {year}",
    # With citations operator
    "papers citing {telescope_name} observations",
    "citations of {telescope_name} results",
    # With topic + date
    "{telescope_name} {topic} papers from {year}",
    "recent {telescope_name} {topic} research",
    # With refereed + topic
    "refereed {telescope_name} {topic} studies",
]

# Years for date-based queries
YEARS = list(range(2018, 2026))


def load_bibgroup_synonyms() -> dict:
    """Load bibgroup synonyms from the JSON file."""
    with open(SYNONYMS_PATH) as f:
        data = json.load(f)
    return data


def format_bibgroup_code(bibgroup_code: str) -> str:
    """Format a bibgroup code for use in ADS query syntax.

    Bibgroup codes containing spaces or special characters need to be quoted.
    """
    if " " in bibgroup_code or "/" in bibgroup_code:
        return f'"{bibgroup_code}"'
    return bibgroup_code


def get_telescope_names(bibgroup_code: str, synonyms_data: dict) -> list[str]:
    """Get all name variations for a bibgroup code."""
    names = [bibgroup_code]  # Always include the code itself

    # Get synonyms if available
    if bibgroup_code in synonyms_data.get("synonyms", {}):
        common_names = synonyms_data["synonyms"][bibgroup_code].get("common_names", [])
        names.extend(common_names)

    return names


def get_topics_for_bibgroup(bibgroup_code: str) -> list[str]:
    """Get relevant topics for a bibgroup."""
    if bibgroup_code in TELESCOPE_TOPICS:
        return TELESCOPE_TOPICS[bibgroup_code]
    return GENERAL_TOPICS


def generate_simple_examples(
    bibgroup_code: str, telescope_names: list[str], count: int = 2
) -> list[dict]:
    """Generate simple bibgroup-only examples."""
    examples = []

    # Use different name variations
    for i in range(min(count, len(telescope_names))):
        template = random.choice(SIMPLE_TEMPLATES)
        name = telescope_names[i]

        nl = template.format(telescope_name=name)
        query = f"bibgroup:{format_bibgroup_code(bibgroup_code)}"

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "bibgroup",
            }
        )

    return examples


def generate_topic_examples(
    bibgroup_code: str, telescope_names: list[str], count: int = 2
) -> list[dict]:
    """Generate bibgroup + topic examples."""
    examples = []
    topics = get_topics_for_bibgroup(bibgroup_code)

    for _ in range(count):
        template = random.choice(TOPIC_TEMPLATES)
        name = random.choice(telescope_names[:3])  # Prefer shorter names
        topic = random.choice(topics)

        nl = template.format(telescope_name=name, topic=topic)
        query = f'bibgroup:{format_bibgroup_code(bibgroup_code)} abs:"{topic}"'

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "bibgroup",
            }
        )

    return examples


def generate_combined_examples(
    bibgroup_code: str, telescope_names: list[str], count: int = 2
) -> list[dict]:
    """Generate bibgroup + other filters examples."""
    examples = []
    topics = get_topics_for_bibgroup(bibgroup_code)

    for _ in range(count):
        template = random.choice(COMBINED_TEMPLATES)
        name = random.choice(telescope_names[:3])
        topic = random.choice(topics)
        year = random.choice(YEARS)

        nl = template.format(telescope_name=name, topic=topic, year=year)

        # Build query based on template content
        query_parts = [f"bibgroup:{format_bibgroup_code(bibgroup_code)}"]

        if "{topic}" in template:
            query_parts.append(f'abs:"{topic}"')
        if "refereed" in template or "peer-reviewed" in template:
            query_parts.append("property:refereed")
        if "{year}" in template:
            query_parts.append(f"year:{year}")
        if "recent" in template:
            query_parts.append("pubdate:[2023-01 TO 2026-12]")
        if "citing" in template or "citations of" in template.lower():
            # For citation queries, wrap in citations operator
            base_query = " ".join(query_parts)
            query = f"citations({base_query})"
        else:
            query = " ".join(query_parts)

        examples.append(
            {
                "natural_language": nl,
                "ads_query": query,
                "category": "bibgroup",
            }
        )

    return examples


def generate_all_examples() -> list[dict]:
    """Generate examples for all bibgroups."""
    all_examples = []
    synonyms_data = load_bibgroup_synonyms()

    # Get all bibgroup codes from the synonyms file
    bibgroup_codes = list(synonyms_data.get("synonyms", {}).keys())

    for bibgroup_code in bibgroup_codes:
        telescope_names = get_telescope_names(bibgroup_code, synonyms_data)

        # Generate 5+ examples per bibgroup
        # 2 simple + 2 topic + 2 combined = 6 base examples
        all_examples.extend(
            generate_simple_examples(bibgroup_code, telescope_names, count=2)
        )
        all_examples.extend(
            generate_topic_examples(bibgroup_code, telescope_names, count=2)
        )
        all_examples.extend(
            generate_combined_examples(bibgroup_code, telescope_names, count=2)
        )

    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate bibgroup-based training examples"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/generated/bibgroup_examples.json"),
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
    print(f"Generated {len(unique_examples)} unique bibgroup examples")

    # Count by bibgroup
    import re

    bibgroup_counts = {}
    for ex in unique_examples:
        query = ex["ads_query"]
        # Extract bibgroup value from query
        match = re.search(r"bibgroup:([^\s)]+)", query)
        if match:
            bg = match.group(1)
            bibgroup_counts[bg] = bibgroup_counts.get(bg, 0) + 1

    print(f"\nTotal bibgroups covered: {len(bibgroup_counts)}")
    print("\nExamples per bibgroup:")
    for bg, count in sorted(bibgroup_counts.items()):
        print(f"  {bg}: {count}")

    # Check for bibgroups with < 3 examples
    under_target = [bg for bg, c in bibgroup_counts.items() if c < 3]
    if under_target:
        print(f"\nWarning: Bibgroups with < 3 examples: {under_target}")

    print(f"\nOutput written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
