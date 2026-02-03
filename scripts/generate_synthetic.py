#!/usr/bin/env python3
"""
Generate synthetic training data for edge cases.

Creates NL-query pairs for common search patterns that may not be
well-represented in query logs, such as:
- Simple unfielded searches
- Natural conversational queries
- Ambiguous/informal phrasing

Usage:
    python scripts/generate_synthetic.py \
        --output data/datasets/raw/synthetic_pairs.json \
        --count 500
"""

import argparse
import json
import random
from pathlib import Path

# Seed topics for scientific literature searches
TOPICS = [
    # Astrophysics
    "dark matter", "dark energy", "black holes", "neutron stars", "pulsars",
    "supernovae", "gamma ray bursts", "quasars", "active galactic nuclei",
    "gravitational waves", "cosmic microwave background", "galaxy formation",
    "galaxy evolution", "stellar evolution", "star formation", "exoplanets",
    "planetary atmospheres", "protoplanetary disks", "brown dwarfs",
    "white dwarfs", "red giants", "main sequence stars", "binary stars",
    "globular clusters", "open clusters", "dwarf galaxies", "spiral galaxies",
    "elliptical galaxies", "galaxy clusters", "cosmic rays", "interstellar medium",
    "molecular clouds", "HII regions", "supernova remnants", "planetary nebulae",
    # Physics
    "string theory", "quantum gravity", "cosmological constant", "inflation",
    "big bang", "nucleosynthesis", "dark matter candidates", "WIMPs", "axions",
    "modified gravity", "MOND", "general relativity", "special relativity",
    # Methods/Instruments
    "spectroscopy", "photometry", "astrometry", "interferometry", "radio astronomy",
    "X-ray astronomy", "infrared astronomy", "optical astronomy", "gamma ray astronomy",
    "machine learning", "neural networks", "deep learning", "image processing",
    # Missions/Telescopes (use longer names to avoid <5 char NL strings)
    "JWST observations", "James Webb telescope", "Hubble Space Telescope", 
    "Chandra X-ray", "Spitzer infrared", "Kepler mission", "TESS exoplanets",
    "Gaia astrometry", "ALMA observations", "VLA radio", "LIGO gravitational waves", 
    "Virgo interferometer", "LISA mission", "Euclid survey", "Roman telescope",
    "Rubin Observatory", "LSST survey", "SKA radio", "Fermi gamma ray", "Swift bursts",
]

ASTRONOMERS = [
    ("Hawking, S.", "Stephen Hawking", "Hawking"),
    ("Einstein, A.", "Albert Einstein", "Einstein"),
    ("Hubble, E.", "Edwin Hubble", "Hubble"),
    ("Penrose, R.", "Roger Penrose", "Penrose"),
    ("Chandrasekhar, S.", "Chandrasekhar", "Chandrasekhar"),
    ("Rubin, V.", "Vera Rubin", "Rubin"),
    ("Thorne, K.", "Kip Thorne", "Thorne"),
    ("Perlmutter, S.", "Saul Perlmutter", "Perlmutter"),
    ("Schmidt, B.", "Brian Schmidt", "Schmidt"),
    ("Riess, A.", "Adam Riess", "Riess"),
    ("Mather, J.", "John Mather", "Mather"),
    ("Smoot, G.", "George Smoot", "Smoot"),
    ("Giacconi, R.", "Riccardo Giacconi", "Giacconi"),
    ("Mayor, M.", "Michel Mayor", "Mayor"),
    ("Queloz, D.", "Didier Queloz", "Queloz"),
]

OBJECTS = [
    ("M31", "Andromeda", "the Andromeda galaxy"),
    ("M87", "M87", "the M87 black hole"),
    ("Sgr A*", "Sagittarius A*", "the Milky Way's central black hole"),
    ("LMC", "Large Magellanic Cloud", "the LMC"),
    ("SMC", "Small Magellanic Cloud", "the SMC"),
    ("Crab Nebula", "Crab Nebula", "the Crab pulsar"),
    ("Orion Nebula", "Orion Nebula", "M42"),
    ("TRAPPIST-1", "TRAPPIST-1", "the TRAPPIST-1 system"),
    ("Proxima Centauri", "Proxima Centauri", "Proxima b"),
    ("HD 209458", "HD 209458 b", "the hot Jupiter HD 209458 b"),
    ("Betelgeuse", "Betelgeuse", "Alpha Orionis"),
    ("Vega", "Vega", "Alpha Lyrae"),
    ("GW150914", "GW150914", "the first gravitational wave detection"),
]

JOURNALS = [
    ("ApJ", "Astrophysical Journal", "ApJ"),
    ("MNRAS", "Monthly Notices", "MNRAS"),
    ("A&A", "Astronomy & Astrophysics", "A&A"),
    ("AJ", "Astronomical Journal", "AJ"),
    ("Nature", "Nature", "Nature"),
    ("Science", "Science", "Science"),
    ("PhRvL", "Physical Review Letters", "PRL"),
    ("PhRvD", "Physical Review D", "PRD"),
    ("ApJL", "ApJ Letters", "ApJL"),
]

INSTITUTIONS = [
    ("Harvard", "Harvard"),
    ("MIT", "MIT"),
    ("Caltech", "Caltech"),
    ("Stanford", "Stanford"),
    ("Berkeley", "UC Berkeley"),
    ("Princeton", "Princeton"),
    ("Cambridge", "Cambridge"),
    ("Oxford", "Oxford"),
    ("Max Planck", "Max Planck"),
    ("ESO", "European Southern Observatory"),
    ("STScI", "Space Telescope Science Institute"),
    ("NASA", "NASA"),
    ("JPL", "JPL"),
    ("CERN", "CERN"),
]


def generate_unfielded_pairs(count: int) -> list[dict]:
    """Generate simple unfielded search pairs."""
    pairs = []
    
    for _ in range(count):
        topic = random.choice(TOPICS)
        
        templates = [
            # Simple topic
            (topic, f'abs:"{topic}"'),
            # Topic with variation
            (f"{topic} papers", f'abs:"{topic}"'),
            (f"{topic} research", f'abs:"{topic}"'),
            (f"{topic} studies", f'abs:"{topic}"'),
            # Recent
            (f"recent {topic}", f'abs:"{topic}" pubdate:[2023 TO 2025]'),
            (f"latest {topic} research", f'abs:"{topic}" pubdate:[2024 TO 2025]'),
            (f"new {topic} papers", f'abs:"{topic}" pubdate:[2024 TO 2025]'),
            # Reviews
            (f"{topic} review", f'abs:"{topic}" doctype:article property:refereed'),
            (f"{topic} reviews", f'abs:"{topic}" doctype:article property:refereed'),
        ]
        
        nl, query = random.choice(templates)
        pairs.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "unfielded",
        })
    
    return pairs


def generate_author_pairs(count: int) -> list[dict]:
    """Generate author search pairs."""
    pairs = []
    
    for _ in range(count):
        ads_name, full_name, short_name = random.choice(ASTRONOMERS)
        topic = random.choice(TOPICS[:20])  # Use astro topics
        
        templates = [
            # Simple author
            (f"{full_name} papers", f'author:"{ads_name}"'),
            (f"papers by {full_name}", f'author:"{ads_name}"'),
            (f"{short_name}'s work", f'author:"{ads_name}"'),
            # Author + topic
            (f"{full_name} on {topic}", f'author:"{ads_name}" abs:"{topic}"'),
            (f"{short_name}'s {topic} papers", f'author:"{ads_name}" abs:"{topic}"'),
            # First author
            (f"first author {short_name}", f'^author:"{ads_name}"'),
            (f"{short_name} as lead author", f'^author:"{ads_name}"'),
        ]
        
        nl, query = random.choice(templates)
        pairs.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "author",
        })
    
    return pairs


def generate_object_pairs(count: int) -> list[dict]:
    """Generate astronomical object search pairs."""
    pairs = []
    
    for _ in range(count):
        obj_ads, obj_common, obj_alt = random.choice(OBJECTS)
        topic = random.choice(["observations", "spectroscopy", "photometry", 
                               "variability", "structure", "dynamics"])
        
        templates = [
            # Simple object
            (f"papers about {obj_common}", f'object:"{obj_ads}"'),
            (f"{obj_common} papers", f'object:"{obj_ads}"'),
            (f"studies of {obj_alt}", f'object:"{obj_ads}"'),
            # Object + topic
            (f"{obj_common} {topic}", f'object:"{obj_ads}" abs:{topic}'),
        ]
        
        nl, query = random.choice(templates)
        pairs.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "astronomy",
        })
    
    return pairs


def generate_date_pairs(count: int) -> list[dict]:
    """Generate date-filtered search pairs."""
    pairs = []
    
    for _ in range(count):
        topic = random.choice(TOPICS)
        year = random.randint(2015, 2024)
        year_end = min(year + random.randint(1, 5), 2025)
        
        templates = [
            # Single year
            (f"{topic} {year}", f'abs:"{topic}" pubdate:{year}'),
            (f"{topic} papers from {year}", f'abs:"{topic}" pubdate:{year}'),
            # Year range
            (f"{topic} {year}-{year_end}", f'abs:"{topic}" pubdate:[{year} TO {year_end}]'),
            (f"{topic} from {year} to {year_end}", f'abs:"{topic}" pubdate:[{year} TO {year_end}]'),
            (f"{topic} between {year} and {year_end}", f'abs:"{topic}" pubdate:[{year} TO {year_end}]'),
            # Decade
            (f"{topic} in the 2020s", f'abs:"{topic}" pubdate:[2020 TO 2029]'),
            (f"{topic} this decade", f'abs:"{topic}" pubdate:[2020 TO 2029]'),
        ]
        
        nl, query = random.choice(templates)
        pairs.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "publication",
        })
    
    return pairs


def generate_journal_pairs(count: int) -> list[dict]:
    """Generate journal search pairs."""
    pairs = []
    
    for _ in range(count):
        bibstem, journal_name, short = random.choice(JOURNALS)
        topic = random.choice(TOPICS)
        
        templates = [
            # Simple journal
            (f"papers in {journal_name}", f'bibstem:{bibstem}'),
            (f"{short} articles", f'bibstem:{bibstem}'),
            # Journal + topic
            (f"{topic} in {journal_name}", f'abs:"{topic}" bibstem:{bibstem}'),
            (f"{short} papers on {topic}", f'abs:"{topic}" bibstem:{bibstem}'),
        ]
        
        nl, query = random.choice(templates)
        pairs.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "publication",
        })
    
    return pairs


def generate_institution_pairs(count: int) -> list[dict]:
    """Generate institution/affiliation search pairs."""
    pairs = []
    
    for _ in range(count):
        inst_ads, inst_common = random.choice(INSTITUTIONS)
        topic = random.choice(TOPICS)
        
        templates = [
            # Simple institution
            (f"{inst_common} papers", f'aff:"{inst_ads}"'),
            (f"research from {inst_common}", f'aff:"{inst_ads}"'),
            # Institution + topic
            (f"{inst_common} {topic} research", f'aff:"{inst_ads}" abs:"{topic}"'),
            (f"{topic} papers from {inst_common}", f'aff:"{inst_ads}" abs:"{topic}"'),
        ]
        
        nl, query = random.choice(templates)
        pairs.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "affiliation",
        })
    
    return pairs


def generate_metrics_pairs(count: int) -> list[dict]:
    """Generate citation/metrics search pairs."""
    pairs = []
    
    for _ in range(count):
        topic = random.choice(TOPICS)
        citations = random.choice([50, 100, 500, 1000])
        
        templates = [
            # Highly cited
            (f"highly cited {topic} papers", f'abs:"{topic}" citation_count:[{citations} TO *]'),
            (f"influential {topic} research", f'abs:"{topic}" citation_count:[{citations} TO *]'),
            (f"top cited {topic}", f'abs:"{topic}" citation_count:[{citations} TO *]'),
            # Popular
            (f"popular {topic} papers", f'abs:"{topic}" read_count:[1000 TO *]'),
            (f"most read {topic} articles", f'abs:"{topic}" read_count:[1000 TO *]'),
        ]
        
        nl, query = random.choice(templates)
        pairs.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "metrics",
        })
    
    return pairs


def generate_conversational_pairs(count: int) -> list[dict]:
    """Generate conversational/question-style queries."""
    pairs = []
    
    for _ in range(count):
        topic = random.choice(TOPICS)
        
        templates = [
            # Question style
            (f"what's new in {topic}?", f'abs:"{topic}" pubdate:[2024 TO 2025]'),
            (f"what are the latest {topic} discoveries?", f'abs:"{topic}" pubdate:[2024 TO 2025]'),
            (f"any good {topic} reviews?", f'abs:"{topic}" doctype:article property:refereed'),
            # Looking for
            (f"looking for {topic} papers", f'abs:"{topic}"'),
            (f"need papers on {topic}", f'abs:"{topic}"'),
            (f"interested in {topic} research", f'abs:"{topic}"'),
            # Specific needs
            (f"open access {topic} papers", f'abs:"{topic}" property:openaccess'),
            (f"refereed {topic} articles", f'abs:"{topic}" property:refereed'),
            (f"{topic} preprints", f'abs:"{topic}" doctype:eprint'),
        ]
        
        nl, query = random.choice(templates)
        pairs.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "conversational",
        })
    
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/datasets/raw/synthetic_pairs.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=500,
        help="Total number of pairs to generate"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Allocate counts per category
    per_cat = args.count // 8
    
    all_pairs = []
    all_pairs.extend(generate_unfielded_pairs(per_cat * 2))  # 2x for unfielded
    all_pairs.extend(generate_author_pairs(per_cat))
    all_pairs.extend(generate_object_pairs(per_cat))
    all_pairs.extend(generate_date_pairs(per_cat))
    all_pairs.extend(generate_journal_pairs(per_cat // 2))
    all_pairs.extend(generate_institution_pairs(per_cat // 2))
    all_pairs.extend(generate_metrics_pairs(per_cat // 2))
    all_pairs.extend(generate_conversational_pairs(per_cat))
    
    # Shuffle
    random.shuffle(all_pairs)
    
    # Dedupe by NL
    seen_nl = set()
    unique_pairs = []
    for p in all_pairs:
        if p["natural_language"] not in seen_nl:
            seen_nl.add(p["natural_language"])
            unique_pairs.append(p)
    
    print(f"Generated {len(unique_pairs)} unique synthetic pairs")
    
    # Category distribution
    from collections import Counter
    cats = Counter(p["category"] for p in unique_pairs)
    print("\nCategory distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat:20} {count:4}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unique_pairs, f, indent=2)
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
