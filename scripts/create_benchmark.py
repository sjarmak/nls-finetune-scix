#!/usr/bin/env python3
"""
Create a comprehensive benchmark evaluation set for NL-to-ADS query translation.

This script generates benchmark_queries.json covering all data model elements:
- Field type examples (content, author, enum, identifier, metric, date)
- Operator examples (citations, references, trending, similar, useful, reviews, topn)
- Enum field subsets (property, doctype, bibgroup)
- Edge cases (ambiguous operator words, complex boolean logic, nested operators)
- Regression tests (malformed patterns that must be rejected)

Target: 300+ test cases total.

Usage:
    python scripts/create_benchmark.py \
        --output data/datasets/benchmark/benchmark_queries.json
"""

import argparse
import json
import sys
from pathlib import Path

# =============================================================================
# FIELD TYPE BENCHMARKS
# Group: content, author, enum, identifier, metric, date
# =============================================================================

FIELD_TYPE_BENCHMARKS = {
    # Content fields (abs, title, abstract, full, keyword)
    "content": [
        {
            "id": "content-001",
            "natural_language": "papers about dark matter halos",
            "expected_query": 'abs:"dark matter halos"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "simple",
        },
        {
            "id": "content-002",
            "natural_language": "articles with gravitational waves in the title",
            "expected_query": 'title:"gravitational waves"',
            "field_type": "content",
            "fields_expected": ["title"],
            "difficulty": "simple",
        },
        {
            "id": "content-003",
            "natural_language": "papers mentioning exoplanets in the abstract",
            "expected_query": 'abstract:"exoplanets"',
            "field_type": "content",
            "fields_expected": ["abstract"],
            "difficulty": "simple",
        },
        {
            "id": "content-004",
            "natural_language": "full text search for Cepheid variables",
            "expected_query": 'full:"Cepheid variables"',
            "field_type": "content",
            "fields_expected": ["full"],
            "difficulty": "simple",
        },
        {
            "id": "content-005",
            "natural_language": "papers with keyword black holes",
            "expected_query": 'keyword:"black holes"',
            "field_type": "content",
            "fields_expected": ["keyword"],
            "difficulty": "simple",
        },
        {
            "id": "content-006",
            "natural_language": "find papers about stellar evolution",
            "expected_query": 'abs:"stellar evolution"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "simple",
        },
        {
            "id": "content-007",
            "natural_language": "research on galaxy formation and merger events",
            "expected_query": 'abs:"galaxy formation" abs:"merger events"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "medium",
        },
        {
            "id": "content-008",
            "natural_language": "papers discussing supermassive black hole growth",
            "expected_query": 'abs:"supermassive black hole growth"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "simple",
        },
        {
            "id": "content-009",
            "natural_language": "cosmic microwave background anisotropies",
            "expected_query": 'abs:"cosmic microwave background anisotropies"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "simple",
        },
        {
            "id": "content-010",
            "natural_language": "neutron star mergers and r-process nucleosynthesis",
            "expected_query": 'abs:"neutron star mergers" abs:"r-process nucleosynthesis"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "medium",
        },
        {
            "id": "content-011",
            "natural_language": "papers about AGN feedback in clusters",
            "expected_query": 'abs:"AGN feedback" abs:"clusters"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "medium",
        },
        {
            "id": "content-012",
            "natural_language": "acknowledgments mentioning NASA grant",
            "expected_query": 'ack:"NASA grant"',
            "field_type": "content",
            "fields_expected": ["ack"],
            "difficulty": "simple",
        },
        {
            "id": "content-013",
            "natural_language": "papers about protoplanetary disks",
            "expected_query": 'abs:"protoplanetary disks"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "simple",
        },
        {
            "id": "content-014",
            "natural_language": "quasar absorption lines studies",
            "expected_query": 'abs:"quasar absorption lines"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "simple",
        },
        {
            "id": "content-015",
            "natural_language": "interstellar medium chemistry",
            "expected_query": 'abs:"interstellar medium chemistry"',
            "field_type": "content",
            "fields_expected": ["abs"],
            "difficulty": "simple",
        },
    ],
    # Author fields (author, first_author, aff, inst, orcid)
    "author": [
        {
            "id": "author-001",
            "natural_language": "papers by Hawking",
            "expected_query": "author:Hawking",
            "field_type": "author",
            "fields_expected": ["author"],
            "difficulty": "simple",
        },
        {
            "id": "author-002",
            "natural_language": "papers by Stephen Hawking",
            "expected_query": 'author:"Hawking, Stephen"',
            "field_type": "author",
            "fields_expected": ["author"],
            "difficulty": "simple",
        },
        {
            "id": "author-003",
            "natural_language": "papers where Einstein is first author",
            "expected_query": 'author:"^Einstein"',
            "field_type": "author",
            "fields_expected": ["author"],
            "difficulty": "simple",
        },
        {
            "id": "author-004",
            "natural_language": "first author Smith on cosmology",
            "expected_query": 'first_author:Smith abs:"cosmology"',
            "field_type": "author",
            "fields_expected": ["first_author", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "author-005",
            "natural_language": "papers from Harvard affiliations",
            "expected_query": "aff:Harvard",
            "field_type": "author",
            "fields_expected": ["aff"],
            "difficulty": "simple",
        },
        {
            "id": "author-006",
            "natural_language": "papers from Caltech on exoplanets",
            "expected_query": 'inst:Caltech abs:"exoplanets"',
            "field_type": "author",
            "fields_expected": ["inst", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "author-007",
            "natural_language": "papers by Einstein and Hawking",
            "expected_query": "author:Einstein author:Hawking",
            "field_type": "author",
            "fields_expected": ["author"],
            "difficulty": "medium",
        },
        {
            "id": "author-008",
            "natural_language": "papers by Riess et al. on supernovae",
            "expected_query": 'author:Riess abs:"supernovae"',
            "field_type": "author",
            "fields_expected": ["author", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "author-009",
            "natural_language": "papers from MIT or Caltech",
            "expected_query": "aff:MIT OR aff:Caltech",
            "field_type": "author",
            "fields_expected": ["aff"],
            "difficulty": "medium",
        },
        {
            "id": "author-010",
            "natural_language": "single author papers on black holes",
            "expected_query": 'author_count:1 abs:"black holes"',
            "field_type": "author",
            "fields_expected": ["author_count", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "author-011",
            "natural_language": "papers by Penrose on cosmology from Cambridge",
            "expected_query": 'author:Penrose abs:"cosmology" aff:Cambridge',
            "field_type": "author",
            "fields_expected": ["author", "abs", "aff"],
            "difficulty": "complex",
        },
        {
            "id": "author-012",
            "natural_language": "large collaboration papers with more than 100 authors",
            "expected_query": "author_count:[100 TO *]",
            "field_type": "author",
            "fields_expected": ["author_count"],
            "difficulty": "medium",
        },
        {
            "id": "author-013",
            "natural_language": "papers by Salpeter on stellar mass",
            "expected_query": 'author:Salpeter abs:"stellar mass"',
            "field_type": "author",
            "fields_expected": ["author", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "author-014",
            "natural_language": "papers with affiliation Max Planck Institute",
            "expected_query": 'aff:"Max Planck"',
            "field_type": "author",
            "fields_expected": ["aff"],
            "difficulty": "simple",
        },
        {
            "id": "author-015",
            "natural_language": "papers by Chandrasekhar",
            "expected_query": "author:Chandrasekhar",
            "field_type": "author",
            "fields_expected": ["author"],
            "difficulty": "simple",
        },
    ],
    # Identifier fields (bibcode, doi, arxiv, identifier)
    "identifier": [
        {
            "id": "identifier-001",
            "natural_language": "paper with bibcode 2020ApJ...900..100S",
            "expected_query": "bibcode:2020ApJ...900..100S",
            "field_type": "identifier",
            "fields_expected": ["bibcode"],
            "difficulty": "simple",
        },
        {
            "id": "identifier-002",
            "natural_language": "paper with DOI 10.1086/345794",
            "expected_query": "doi:10.1086/345794",
            "field_type": "identifier",
            "fields_expected": ["doi"],
            "difficulty": "simple",
        },
        {
            "id": "identifier-003",
            "natural_language": "arxiv preprint 2301.00001",
            "expected_query": "arxiv:2301.00001",
            "field_type": "identifier",
            "fields_expected": ["arxiv"],
            "difficulty": "simple",
        },
        {
            "id": "identifier-004",
            "natural_language": "find paper with identifier 2023Natur.615..605H",
            "expected_query": "identifier:2023Natur.615..605H",
            "field_type": "identifier",
            "fields_expected": ["identifier"],
            "difficulty": "simple",
        },
        {
            "id": "identifier-005",
            "natural_language": "papers in ApJ from 2023",
            "expected_query": "bibcode:2023ApJ* year:2023",
            "field_type": "identifier",
            "fields_expected": ["bibcode", "year"],
            "difficulty": "medium",
        },
        {
            "id": "identifier-006",
            "natural_language": "old arxiv paper astro-ph/0401001",
            "expected_query": "arxiv:astro-ph/0401001",
            "field_type": "identifier",
            "fields_expected": ["arxiv"],
            "difficulty": "simple",
        },
        {
            "id": "identifier-007",
            "natural_language": "Nature paper 10.1038/nature12917",
            "expected_query": "doi:10.1038/nature12917",
            "field_type": "identifier",
            "fields_expected": ["doi"],
            "difficulty": "simple",
        },
        {
            "id": "identifier-008",
            "natural_language": "papers in MNRAS volume 500",
            "expected_query": "bibstem:MNRAS volume:500",
            "field_type": "identifier",
            "fields_expected": ["bibstem", "volume"],
            "difficulty": "medium",
        },
        {
            "id": "identifier-009",
            "natural_language": "Astrophysical Journal Letters papers",
            "expected_query": "bibstem:ApJL",
            "field_type": "identifier",
            "fields_expected": ["bibstem"],
            "difficulty": "simple",
        },
        {
            "id": "identifier-010",
            "natural_language": "papers published in A&A",
            "expected_query": "bibstem:A&A",
            "field_type": "identifier",
            "fields_expected": ["bibstem"],
            "difficulty": "simple",
        },
        {
            "id": "identifier-011",
            "natural_language": "papers from Science journal",
            "expected_query": "bibstem:Sci",
            "field_type": "identifier",
            "fields_expected": ["bibstem"],
            "difficulty": "simple",
        },
        {
            "id": "identifier-012",
            "natural_language": "ARA&A annual review papers",
            "expected_query": "bibstem:ARA&A",
            "field_type": "identifier",
            "fields_expected": ["bibstem"],
            "difficulty": "simple",
        },
    ],
    # Metric fields (citation_count, read_count, author_count)
    "metric": [
        {
            "id": "metric-001",
            "natural_language": "highly cited papers with more than 100 citations",
            "expected_query": "citation_count:[100 TO *]",
            "field_type": "metric",
            "fields_expected": ["citation_count"],
            "difficulty": "simple",
        },
        {
            "id": "metric-002",
            "natural_language": "papers with between 50 and 200 citations",
            "expected_query": "citation_count:[50 TO 200]",
            "field_type": "metric",
            "fields_expected": ["citation_count"],
            "difficulty": "simple",
        },
        {
            "id": "metric-003",
            "natural_language": "highly read papers with more than 500 reads",
            "expected_query": "read_count:[500 TO *]",
            "field_type": "metric",
            "fields_expected": ["read_count"],
            "difficulty": "simple",
        },
        {
            "id": "metric-004",
            "natural_language": "papers with exactly 1000 citations",
            "expected_query": "citation_count:1000",
            "field_type": "metric",
            "fields_expected": ["citation_count"],
            "difficulty": "simple",
        },
        {
            "id": "metric-005",
            "natural_language": "short papers with fewer than 5 pages",
            "expected_query": "page_count:[1 TO 5]",
            "field_type": "metric",
            "fields_expected": ["page_count"],
            "difficulty": "simple",
        },
        {
            "id": "metric-006",
            "natural_language": "highly cited papers on cosmology",
            "expected_query": 'citation_count:[100 TO *] abs:"cosmology"',
            "field_type": "metric",
            "fields_expected": ["citation_count", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "metric-007",
            "natural_language": "papers with more than 1000 citations about dark energy",
            "expected_query": 'citation_count:[1000 TO *] abs:"dark energy"',
            "field_type": "metric",
            "fields_expected": ["citation_count", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "metric-008",
            "natural_language": "uncited papers on exoplanets",
            "expected_query": 'citation_count:0 abs:"exoplanets"',
            "field_type": "metric",
            "fields_expected": ["citation_count", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "metric-009",
            "natural_language": "moderately cited papers between 10 and 50 citations",
            "expected_query": "citation_count:[10 TO 50]",
            "field_type": "metric",
            "fields_expected": ["citation_count"],
            "difficulty": "simple",
        },
        {
            "id": "metric-010",
            "natural_language": "papers with few authors and many citations",
            "expected_query": "author_count:[1 TO 3] citation_count:[100 TO *]",
            "field_type": "metric",
            "fields_expected": ["author_count", "citation_count"],
            "difficulty": "medium",
        },
        {
            "id": "metric-011",
            "natural_language": "papers with 500 or more citations",
            "expected_query": "citation_count:[500 TO *]",
            "field_type": "metric",
            "fields_expected": ["citation_count"],
            "difficulty": "simple",
        },
        {
            "id": "metric-012",
            "natural_language": "long papers with more than 50 pages",
            "expected_query": "page_count:[50 TO *]",
            "field_type": "metric",
            "fields_expected": ["page_count"],
            "difficulty": "simple",
        },
    ],
    # Date fields (year, pubdate, entry_date)
    "date": [
        {
            "id": "date-001",
            "natural_language": "papers from 2023",
            "expected_query": "year:2023",
            "field_type": "date",
            "fields_expected": ["year"],
            "difficulty": "simple",
        },
        {
            "id": "date-002",
            "natural_language": "papers from 2020 to 2023",
            "expected_query": "year:[2020 TO 2023]",
            "field_type": "date",
            "fields_expected": ["year"],
            "difficulty": "simple",
        },
        {
            "id": "date-003",
            "natural_language": "papers published after 2020",
            "expected_query": "year:[2020 TO *]",
            "field_type": "date",
            "fields_expected": ["year"],
            "difficulty": "simple",
        },
        {
            "id": "date-004",
            "natural_language": "papers from the last 5 years on black holes",
            "expected_query": 'year:[2021 TO *] abs:"black holes"',
            "field_type": "date",
            "fields_expected": ["year", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "date-005",
            "natural_language": "recent papers on exoplanets from 2024",
            "expected_query": 'year:2024 abs:"exoplanets"',
            "field_type": "date",
            "fields_expected": ["year", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "date-006",
            "natural_language": "papers published in January 2023",
            "expected_query": "pubdate:[2023-01 TO 2023-01]",
            "field_type": "date",
            "fields_expected": ["pubdate"],
            "difficulty": "medium",
        },
        {
            "id": "date-007",
            "natural_language": "papers added to ADS in 2023",
            "expected_query": "entry_date:[2023-01-01 TO 2023-12-31]",
            "field_type": "date",
            "fields_expected": ["entry_date"],
            "difficulty": "medium",
        },
        {
            "id": "date-008",
            "natural_language": "older papers before 2000",
            "expected_query": "year:[* TO 1999]",
            "field_type": "date",
            "fields_expected": ["year"],
            "difficulty": "simple",
        },
        {
            "id": "date-009",
            "natural_language": "papers by Hawking from the 1970s",
            "expected_query": "author:Hawking year:[1970 TO 1979]",
            "field_type": "date",
            "fields_expected": ["author", "year"],
            "difficulty": "medium",
        },
        {
            "id": "date-010",
            "natural_language": "1990s papers on supernovae",
            "expected_query": 'year:[1990 TO 1999] abs:"supernovae"',
            "field_type": "date",
            "fields_expected": ["year", "abs"],
            "difficulty": "medium",
        },
        {
            "id": "date-011",
            "natural_language": "papers between 2010 and 2015",
            "expected_query": "year:[2010 TO 2015]",
            "field_type": "date",
            "fields_expected": ["year"],
            "difficulty": "simple",
        },
        {
            "id": "date-012",
            "natural_language": "papers from this year",
            "expected_query": "year:2026",
            "field_type": "date",
            "fields_expected": ["year"],
            "difficulty": "simple",
        },
    ],
}

# =============================================================================
# OPERATOR BENCHMARKS
# =============================================================================

OPERATOR_BENCHMARKS = {
    "citations": [
        {
            "id": "citations-001",
            "natural_language": "papers citing Hawking",
            "expected_query": "citations(author:Hawking)",
            "operator": "citations",
            "difficulty": "simple",
        },
        {
            "id": "citations-002",
            "natural_language": "papers that cite the dark matter paper",
            "expected_query": 'citations(abs:"dark matter")',
            "operator": "citations",
            "difficulty": "simple",
        },
        {
            "id": "citations-003",
            "natural_language": "who cited Einstein's 1905 paper",
            "expected_query": "citations(author:Einstein year:1905)",
            "operator": "citations",
            "difficulty": "medium",
        },
        {
            "id": "citations-004",
            "natural_language": "citations to the Planck 2018 cosmology results",
            "expected_query": 'citations(abs:"Planck 2018 cosmology")',
            "operator": "citations",
            "difficulty": "medium",
        },
        {
            "id": "citations-005",
            "natural_language": "papers citing gravitational wave detection papers",
            "expected_query": 'citations(abs:"gravitational wave detection")',
            "operator": "citations",
            "difficulty": "simple",
        },
        {
            "id": "citations-006",
            "natural_language": "research citing JWST observations",
            "expected_query": "citations(bibgroup:JWST)",
            "operator": "citations",
            "difficulty": "medium",
        },
        {
            "id": "citations-007",
            "natural_language": "recent papers citing the original LIGO paper",
            "expected_query": 'citations(abs:"LIGO") year:[2020 TO *]',
            "operator": "citations",
            "difficulty": "complex",
        },
        {
            "id": "citations-008",
            "natural_language": "find citations to exoplanet discovery papers",
            "expected_query": 'citations(abs:"exoplanet discovery")',
            "operator": "citations",
            "difficulty": "simple",
        },
        {
            "id": "citations-009",
            "natural_language": "articles citing Riess et al. supernova work",
            "expected_query": "citations(author:Riess)",
            "operator": "citations",
            "difficulty": "simple",
        },
        {
            "id": "citations-010",
            "natural_language": "how many papers cite the cosmic microwave background paper",
            "expected_query": 'citations(abs:"cosmic microwave background")',
            "operator": "citations",
            "difficulty": "simple",
        },
        {
            "id": "citations-011",
            "natural_language": "papers citing Hubble deep field observations",
            "expected_query": 'citations(abs:"Hubble deep field")',
            "operator": "citations",
            "difficulty": "simple",
        },
        {
            "id": "citations-012",
            "natural_language": "show citations to supernova cosmology papers",
            "expected_query": 'citations(abs:"supernova cosmology")',
            "operator": "citations",
            "difficulty": "simple",
        },
        {
            "id": "citations-013",
            "natural_language": "refereed papers citing ALMA observations",
            "expected_query": "citations(bibgroup:ALMA) property:refereed",
            "operator": "citations",
            "difficulty": "complex",
        },
        {
            "id": "citations-014",
            "natural_language": "work that cites the dark energy discovery",
            "expected_query": 'citations(abs:"dark energy")',
            "operator": "citations",
            "difficulty": "simple",
        },
        {
            "id": "citations-015",
            "natural_language": "list citations to Weinberg cosmology papers",
            "expected_query": "citations(author:Weinberg)",
            "operator": "citations",
            "difficulty": "simple",
        },
    ],
    "references": [
        {
            "id": "references-001",
            "natural_language": "references of Hawking's papers",
            "expected_query": "references(author:Hawking)",
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-002",
            "natural_language": "papers cited in the Planck 2018 paper",
            "expected_query": 'references(abs:"Planck 2018")',
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-003",
            "natural_language": "bibliography of the LIGO detection paper",
            "expected_query": 'references(abs:"LIGO detection")',
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-004",
            "natural_language": "what does Einstein cite",
            "expected_query": "references(author:Einstein)",
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-005",
            "natural_language": "sources cited by Penrose papers",
            "expected_query": "references(author:Penrose)",
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-006",
            "natural_language": "papers referenced by recent dark matter studies",
            "expected_query": 'references(abs:"dark matter" year:[2020 TO *])',
            "operator": "references",
            "difficulty": "medium",
        },
        {
            "id": "references-007",
            "natural_language": "what papers did the JWST team cite",
            "expected_query": "references(bibgroup:JWST)",
            "operator": "references",
            "difficulty": "medium",
        },
        {
            "id": "references-008",
            "natural_language": "show references of the gravitational waves paper",
            "expected_query": 'references(abs:"gravitational waves")',
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-009",
            "natural_language": "works cited by the exoplanet atmospheres paper",
            "expected_query": 'references(abs:"exoplanet atmospheres")',
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-010",
            "natural_language": "what sources did Weinberg use",
            "expected_query": "references(author:Weinberg)",
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-011",
            "natural_language": "papers cited in the Planck cosmology results",
            "expected_query": 'references(abs:"Planck cosmology")',
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-012",
            "natural_language": "what papers did Riess cite in supernova work",
            "expected_query": 'references(author:Riess abs:"supernova")',
            "operator": "references",
            "difficulty": "medium",
        },
        {
            "id": "references-013",
            "natural_language": "bibliography of the Gaia mission papers",
            "expected_query": "references(bibgroup:Gaia)",
            "operator": "references",
            "difficulty": "medium",
        },
        {
            "id": "references-014",
            "natural_language": "sources used in the black hole imaging paper",
            "expected_query": 'references(abs:"black hole imaging")',
            "operator": "references",
            "difficulty": "simple",
        },
        {
            "id": "references-015",
            "natural_language": "papers referenced by stellar evolution reviews",
            "expected_query": 'references(abs:"stellar evolution" doctype:article)',
            "operator": "references",
            "difficulty": "medium",
        },
    ],
    "trending": [
        {
            "id": "trending-001",
            "natural_language": "trending papers on dark matter",
            "expected_query": 'trending(abs:"dark matter")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-002",
            "natural_language": "hot topics in exoplanet research",
            "expected_query": 'trending(abs:"exoplanet")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-003",
            "natural_language": "what's trending in cosmology",
            "expected_query": 'trending(abs:"cosmology")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-004",
            "natural_language": "popular papers in gravitational wave astronomy",
            "expected_query": 'trending(abs:"gravitational wave")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-005",
            "natural_language": "hot research in black hole physics",
            "expected_query": 'trending(abs:"black hole")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-006",
            "natural_language": "trending JWST papers",
            "expected_query": "trending(bibgroup:JWST)",
            "operator": "trending",
            "difficulty": "medium",
        },
        {
            "id": "trending-007",
            "natural_language": "currently popular in machine learning astronomy",
            "expected_query": 'trending(abs:"machine learning astronomy")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-008",
            "natural_language": "hot papers in stellar evolution",
            "expected_query": 'trending(abs:"stellar evolution")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-009",
            "natural_language": "trending research on galaxy mergers",
            "expected_query": 'trending(abs:"galaxy mergers")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-010",
            "natural_language": "what's popular in neutron star research",
            "expected_query": 'trending(abs:"neutron star")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-011",
            "natural_language": "hot topics in AGN research",
            "expected_query": 'trending(abs:"AGN")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-012",
            "natural_language": "trending work on fast radio bursts",
            "expected_query": 'trending(abs:"fast radio bursts")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-013",
            "natural_language": "popular papers on dark energy equation of state",
            "expected_query": 'trending(abs:"dark energy equation of state")',
            "operator": "trending",
            "difficulty": "medium",
        },
        {
            "id": "trending-014",
            "natural_language": "what's trending in galactic archaeology",
            "expected_query": 'trending(abs:"galactic archaeology")',
            "operator": "trending",
            "difficulty": "simple",
        },
        {
            "id": "trending-015",
            "natural_language": "hot research on gravitational wave sources",
            "expected_query": 'trending(abs:"gravitational wave sources")',
            "operator": "trending",
            "difficulty": "simple",
        },
    ],
    "similar": [
        {
            "id": "similar-001",
            "natural_language": "papers similar to dark matter research",
            "expected_query": 'similar(abs:"dark matter")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-002",
            "natural_language": "papers like the Hawking radiation paper",
            "expected_query": 'similar(abs:"Hawking radiation")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-003",
            "natural_language": "related to exoplanet atmospheres research",
            "expected_query": 'similar(abs:"exoplanet atmospheres")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-004",
            "natural_language": "work similar to cosmological perturbation theory",
            "expected_query": 'similar(abs:"cosmological perturbation theory")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-005",
            "natural_language": "find similar papers to gravitational lensing studies",
            "expected_query": 'similar(abs:"gravitational lensing")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-006",
            "natural_language": "papers resembling ALMA observations research",
            "expected_query": 'similar(bibgroup:ALMA)',
            "operator": "similar",
            "difficulty": "medium",
        },
        {
            "id": "similar-007",
            "natural_language": "related papers to Penrose's work on singularities",
            "expected_query": 'similar(author:Penrose abs:"singularities")',
            "operator": "similar",
            "difficulty": "medium",
        },
        {
            "id": "similar-008",
            "natural_language": "comparable work to CMB anisotropy studies",
            "expected_query": 'similar(abs:"CMB anisotropy")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-009",
            "natural_language": "papers like this on stellar nucleosynthesis",
            "expected_query": 'similar(abs:"stellar nucleosynthesis")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-010",
            "natural_language": "similar research to galaxy cluster dynamics",
            "expected_query": 'similar(abs:"galaxy cluster dynamics")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-011",
            "natural_language": "papers like those on Type Ia supernovae",
            "expected_query": 'similar(abs:"Type Ia supernovae")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-012",
            "natural_language": "related to pulsar timing research",
            "expected_query": 'similar(abs:"pulsar timing")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-013",
            "natural_language": "work resembling reionization studies",
            "expected_query": 'similar(abs:"reionization")',
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-014",
            "natural_language": "find papers similar to Hawking's work",
            "expected_query": "similar(author:Hawking)",
            "operator": "similar",
            "difficulty": "simple",
        },
        {
            "id": "similar-015",
            "natural_language": "comparable work to HST deep field papers",
            "expected_query": 'similar(bibgroup:HST abs:"deep field")',
            "operator": "similar",
            "difficulty": "medium",
        },
    ],
    "useful": [
        {
            "id": "useful-001",
            "natural_language": "useful papers on cosmology",
            "expected_query": 'useful(abs:"cosmology")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-002",
            "natural_language": "foundational work on dark matter",
            "expected_query": 'useful(abs:"dark matter")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-003",
            "natural_language": "essential reading on gravitational waves",
            "expected_query": 'useful(abs:"gravitational waves")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-004",
            "natural_language": "must-read papers on black holes",
            "expected_query": 'useful(abs:"black holes")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-005",
            "natural_language": "key papers on exoplanet detection",
            "expected_query": 'useful(abs:"exoplanet detection")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-006",
            "natural_language": "seminal work on stellar evolution",
            "expected_query": 'useful(abs:"stellar evolution")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-007",
            "natural_language": "landmark papers in galaxy formation",
            "expected_query": 'useful(abs:"galaxy formation")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-008",
            "natural_language": "important papers on quasars",
            "expected_query": 'useful(abs:"quasars")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-009",
            "natural_language": "helpful papers on spectroscopy techniques",
            "expected_query": 'useful(abs:"spectroscopy")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-010",
            "natural_language": "foundational papers on inflation theory",
            "expected_query": 'useful(abs:"inflation theory")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-011",
            "natural_language": "essential work on Hubble constant",
            "expected_query": 'useful(abs:"Hubble constant")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-012",
            "natural_language": "must-read papers on baryon acoustic oscillations",
            "expected_query": 'useful(abs:"baryon acoustic oscillations")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-013",
            "natural_language": "key references on cosmic strings",
            "expected_query": 'useful(abs:"cosmic strings")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-014",
            "natural_language": "landmark papers on weak lensing",
            "expected_query": 'useful(abs:"weak lensing")',
            "operator": "useful",
            "difficulty": "simple",
        },
        {
            "id": "useful-015",
            "natural_language": "seminal work on primordial nucleosynthesis",
            "expected_query": 'useful(abs:"primordial nucleosynthesis")',
            "operator": "useful",
            "difficulty": "simple",
        },
    ],
    "reviews": [
        {
            "id": "reviews-001",
            "natural_language": "review articles on dark matter",
            "expected_query": 'reviews(abs:"dark matter")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-002",
            "natural_language": "survey papers on galaxy evolution",
            "expected_query": 'reviews(abs:"galaxy evolution")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-003",
            "natural_language": "overviews of exoplanet atmospheres",
            "expected_query": 'reviews(abs:"exoplanet atmospheres")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-004",
            "natural_language": "comprehensive reviews on gravitational lensing",
            "expected_query": 'reviews(abs:"gravitational lensing")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-005",
            "natural_language": "literature review on cosmological parameters",
            "expected_query": 'reviews(abs:"cosmological parameters")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-006",
            "natural_language": "systematic review of stellar mass function",
            "expected_query": 'reviews(abs:"stellar mass function")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-007",
            "natural_language": "tutorial on Bayesian inference in astronomy",
            "expected_query": 'reviews(abs:"Bayesian inference")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-008",
            "natural_language": "introduction to black hole thermodynamics",
            "expected_query": 'reviews(abs:"black hole thermodynamics")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-009",
            "natural_language": "state-of-the-art review on neutrino astronomy",
            "expected_query": 'reviews(abs:"neutrino astronomy")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-010",
            "natural_language": "survey of AGN variability",
            "expected_query": 'reviews(abs:"AGN variability")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-011",
            "natural_language": "review articles on supernova remnants",
            "expected_query": 'reviews(abs:"supernova remnants")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-012",
            "natural_language": "comprehensive survey of globular clusters",
            "expected_query": 'reviews(abs:"globular clusters")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-013",
            "natural_language": "overview papers on interstellar dust",
            "expected_query": 'reviews(abs:"interstellar dust")',
            "operator": "reviews",
            "difficulty": "simple",
        },
        {
            "id": "reviews-014",
            "natural_language": "recent reviews on exomoon detection",
            "expected_query": 'reviews(abs:"exomoon") year:[2020 TO *]',
            "operator": "reviews",
            "difficulty": "medium",
        },
        {
            "id": "reviews-015",
            "natural_language": "literature survey on magnetars",
            "expected_query": 'reviews(abs:"magnetars")',
            "operator": "reviews",
            "difficulty": "simple",
        },
    ],
    "topn": [
        {
            "id": "topn-001",
            "natural_language": "top 10 papers on dark matter",
            "expected_query": 'topn(10, abs:"dark matter", citation_count)',
            "operator": "topn",
            "difficulty": "simple",
        },
        {
            "id": "topn-002",
            "natural_language": "most cited papers on exoplanets",
            "expected_query": 'topn(10, abs:"exoplanets", citation_count)',
            "operator": "topn",
            "difficulty": "simple",
        },
        {
            "id": "topn-003",
            "natural_language": "top 5 papers on gravitational waves",
            "expected_query": 'topn(5, abs:"gravitational waves", citation_count)',
            "operator": "topn",
            "difficulty": "simple",
        },
        {
            "id": "topn-004",
            "natural_language": "best papers on cosmology",
            "expected_query": 'topn(10, abs:"cosmology", citation_count)',
            "operator": "topn",
            "difficulty": "simple",
        },
        {
            "id": "topn-005",
            "natural_language": "top 100 highly cited black hole papers",
            "expected_query": 'topn(100, abs:"black holes", citation_count)',
            "operator": "topn",
            "difficulty": "simple",
        },
        {
            "id": "topn-006",
            "natural_language": "top papers on stellar evolution from 2020",
            "expected_query": 'topn(10, abs:"stellar evolution" year:2020, citation_count)',
            "operator": "topn",
            "difficulty": "medium",
        },
        {
            "id": "topn-007",
            "natural_language": "most read papers on supernovae",
            "expected_query": 'topn(10, abs:"supernovae", read_count)',
            "operator": "topn",
            "difficulty": "medium",
        },
        {
            "id": "topn-008",
            "natural_language": "highest cited papers by Hawking",
            "expected_query": "topn(10, author:Hawking, citation_count)",
            "operator": "topn",
            "difficulty": "medium",
        },
        {
            "id": "topn-009",
            "natural_language": "top 20 papers on neutron stars",
            "expected_query": 'topn(20, abs:"neutron stars", citation_count)',
            "operator": "topn",
            "difficulty": "simple",
        },
        {
            "id": "topn-010",
            "natural_language": "top papers with most citations on galaxy clusters",
            "expected_query": 'topn(10, abs:"galaxy clusters", citation_count)',
            "operator": "topn",
            "difficulty": "simple",
        },
        {
            "id": "topn-011",
            "natural_language": "top 10 most read papers on AGN",
            "expected_query": 'topn(10, abs:"AGN", read_count)',
            "operator": "topn",
            "difficulty": "medium",
        },
        {
            "id": "topn-012",
            "natural_language": "most cited JWST papers",
            "expected_query": "topn(10, bibgroup:JWST, citation_count)",
            "operator": "topn",
            "difficulty": "medium",
        },
        {
            "id": "topn-013",
            "natural_language": "top 50 papers on machine learning in astronomy",
            "expected_query": 'topn(50, abs:"machine learning astronomy", citation_count)',
            "operator": "topn",
            "difficulty": "simple",
        },
        {
            "id": "topn-014",
            "natural_language": "highest cited papers on cosmic rays",
            "expected_query": 'topn(10, abs:"cosmic rays", citation_count)',
            "operator": "topn",
            "difficulty": "simple",
        },
        {
            "id": "topn-015",
            "natural_language": "top 10 papers by Einstein",
            "expected_query": "topn(10, author:Einstein, citation_count)",
            "operator": "topn",
            "difficulty": "medium",
        },
    ],
}

# =============================================================================
# ENUM FIELD BENCHMARKS (property, doctype, bibgroup)
# =============================================================================

ENUM_BENCHMARKS = {
    "property": [
        {
            "id": "property-001",
            "natural_language": "peer reviewed papers on cosmology",
            "expected_query": 'property:refereed abs:"cosmology"',
            "enum_field": "property",
            "enum_value": "refereed",
            "difficulty": "simple",
        },
        {
            "id": "property-002",
            "natural_language": "open access papers on dark matter",
            "expected_query": 'property:openaccess abs:"dark matter"',
            "enum_field": "property",
            "enum_value": "openaccess",
            "difficulty": "simple",
        },
        {
            "id": "property-003",
            "natural_language": "preprints on exoplanets",
            "expected_query": 'property:eprint abs:"exoplanets"',
            "enum_field": "property",
            "enum_value": "eprint",
            "difficulty": "simple",
        },
        {
            "id": "property-004",
            "natural_language": "papers with associated data",
            "expected_query": "property:data",
            "enum_field": "property",
            "enum_value": "data",
            "difficulty": "simple",
        },
        {
            "id": "property-005",
            "natural_language": "non-refereed papers on galaxy formation",
            "expected_query": 'property:notrefereed abs:"galaxy formation"',
            "enum_field": "property",
            "enum_value": "notrefereed",
            "difficulty": "simple",
        },
        {
            "id": "property-006",
            "natural_language": "software papers in astronomy",
            "expected_query": "property:software database:astronomy",
            "enum_field": "property",
            "enum_value": "software",
            "difficulty": "medium",
        },
        {
            "id": "property-007",
            "natural_language": "catalog entries in the ADS",
            "expected_query": "property:catalog",
            "enum_field": "property",
            "enum_value": "catalog",
            "difficulty": "simple",
        },
        {
            "id": "property-008",
            "natural_language": "papers with table of contents",
            "expected_query": "property:toc",
            "enum_field": "property",
            "enum_value": "toc",
            "difficulty": "simple",
        },
        {
            "id": "property-009",
            "natural_language": "arxiv open access papers on black holes",
            "expected_query": 'property:eprint_openaccess abs:"black holes"',
            "enum_field": "property",
            "enum_value": "eprint_openaccess",
            "difficulty": "medium",
        },
        {
            "id": "property-010",
            "natural_language": "INSPIRE indexed papers",
            "expected_query": "property:inspire",
            "enum_field": "property",
            "enum_value": "inspire",
            "difficulty": "simple",
        },
        {
            "id": "property-011",
            "natural_language": "publisher open access articles on cosmology",
            "expected_query": 'property:pub_openaccess abs:"cosmology"',
            "enum_field": "property",
            "enum_value": "pub_openaccess",
            "difficulty": "medium",
        },
        {
            "id": "property-012",
            "natural_language": "conference papers on exoplanets",
            "expected_query": 'property:inproceedings abs:"exoplanets"',
            "enum_field": "property",
            "enum_value": "inproceedings",
            "difficulty": "simple",
        },
        {
            "id": "property-013",
            "natural_language": "papers with associated electronic sources",
            "expected_query": "property:esource",
            "enum_field": "property",
            "enum_value": "esource",
            "difficulty": "simple",
        },
        {
            "id": "property-014",
            "natural_language": "papers with OCR-processed abstracts",
            "expected_query": "property:ocr_abstract",
            "enum_field": "property",
            "enum_value": "ocr_abstract",
            "difficulty": "simple",
        },
        {
            "id": "property-015",
            "natural_language": "ADS open access papers on black holes",
            "expected_query": 'property:ads_openaccess abs:"black holes"',
            "enum_field": "property",
            "enum_value": "ads_openaccess",
            "difficulty": "medium",
        },
    ],
    "doctype": [
        {
            "id": "doctype-001",
            "natural_language": "journal articles on cosmology",
            "expected_query": 'doctype:article abs:"cosmology"',
            "enum_field": "doctype",
            "enum_value": "article",
            "difficulty": "simple",
        },
        {
            "id": "doctype-002",
            "natural_language": "PhD theses on exoplanets",
            "expected_query": 'doctype:phdthesis abs:"exoplanets"',
            "enum_field": "doctype",
            "enum_value": "phdthesis",
            "difficulty": "simple",
        },
        {
            "id": "doctype-003",
            "natural_language": "conference proceedings on gravitational waves",
            "expected_query": 'doctype:inproceedings abs:"gravitational waves"',
            "enum_field": "doctype",
            "enum_value": "inproceedings",
            "difficulty": "simple",
        },
        {
            "id": "doctype-004",
            "natural_language": "software releases in astronomy",
            "expected_query": "doctype:software database:astronomy",
            "enum_field": "doctype",
            "enum_value": "software",
            "difficulty": "medium",
        },
        {
            "id": "doctype-005",
            "natural_language": "book chapters on stellar evolution",
            "expected_query": 'doctype:inbook abs:"stellar evolution"',
            "enum_field": "doctype",
            "enum_value": "inbook",
            "difficulty": "simple",
        },
        {
            "id": "doctype-006",
            "natural_language": "conference talks about AGN",
            "expected_query": 'doctype:talk abs:"AGN"',
            "enum_field": "doctype",
            "enum_value": "talk",
            "difficulty": "simple",
        },
        {
            "id": "doctype-007",
            "natural_language": "technical reports on instrumentation",
            "expected_query": 'doctype:techreport abs:"instrumentation"',
            "enum_field": "doctype",
            "enum_value": "techreport",
            "difficulty": "simple",
        },
        {
            "id": "doctype-008",
            "natural_language": "master's theses on galaxy morphology",
            "expected_query": 'doctype:mastersthesis abs:"galaxy morphology"',
            "enum_field": "doctype",
            "enum_value": "mastersthesis",
            "difficulty": "simple",
        },
        {
            "id": "doctype-009",
            "natural_language": "press releases about JWST",
            "expected_query": "doctype:pressrelease bibgroup:JWST",
            "enum_field": "doctype",
            "enum_value": "pressrelease",
            "difficulty": "medium",
        },
        {
            "id": "doctype-010",
            "natural_language": "astronomer's telegrams about transients",
            "expected_query": 'doctype:circular abs:"transient"',
            "enum_field": "doctype",
            "enum_value": "circular",
            "difficulty": "simple",
        },
        {
            "id": "doctype-011",
            "natural_language": "erratum corrections in astrophysics",
            "expected_query": "doctype:erratum database:astronomy",
            "enum_field": "doctype",
            "enum_value": "erratum",
            "difficulty": "medium",
        },
        {
            "id": "doctype-012",
            "natural_language": "editorial articles in astronomy journals",
            "expected_query": "doctype:editorial",
            "enum_field": "doctype",
            "enum_value": "editorial",
            "difficulty": "simple",
        },
        {
            "id": "doctype-013",
            "natural_language": "book reviews on cosmology",
            "expected_query": 'doctype:bookreview abs:"cosmology"',
            "enum_field": "doctype",
            "enum_value": "bookreview",
            "difficulty": "simple",
        },
        {
            "id": "doctype-014",
            "natural_language": "astronomy proposals",
            "expected_query": "doctype:proposal database:astronomy",
            "enum_field": "doctype",
            "enum_value": "proposal",
            "difficulty": "medium",
        },
        {
            "id": "doctype-015",
            "natural_language": "arxiv preprints on dark matter",
            "expected_query": 'doctype:eprint abs:"dark matter"',
            "enum_field": "doctype",
            "enum_value": "eprint",
            "difficulty": "simple",
        },
    ],
    "bibgroup": [
        {
            "id": "bibgroup-001",
            "natural_language": "Hubble telescope papers",
            "expected_query": "bibgroup:HST",
            "enum_field": "bibgroup",
            "enum_value": "HST",
            "difficulty": "simple",
        },
        {
            "id": "bibgroup-002",
            "natural_language": "JWST observations of exoplanets",
            "expected_query": 'bibgroup:JWST abs:"exoplanets"',
            "enum_field": "bibgroup",
            "enum_value": "JWST",
            "difficulty": "medium",
        },
        {
            "id": "bibgroup-003",
            "natural_language": "ALMA observations",
            "expected_query": "bibgroup:ALMA",
            "enum_field": "bibgroup",
            "enum_value": "ALMA",
            "difficulty": "simple",
        },
        {
            "id": "bibgroup-004",
            "natural_language": "Chandra X-ray papers",
            "expected_query": "bibgroup:Chandra",
            "enum_field": "bibgroup",
            "enum_value": "Chandra",
            "difficulty": "simple",
        },
        {
            "id": "bibgroup-005",
            "natural_language": "SDSS survey data papers",
            "expected_query": "bibgroup:SDSS",
            "enum_field": "bibgroup",
            "enum_value": "SDSS",
            "difficulty": "simple",
        },
        {
            "id": "bibgroup-006",
            "natural_language": "Gaia astrometry results",
            "expected_query": "bibgroup:Gaia",
            "enum_field": "bibgroup",
            "enum_value": "Gaia",
            "difficulty": "simple",
        },
        {
            "id": "bibgroup-007",
            "natural_language": "TESS exoplanet discoveries",
            "expected_query": 'bibgroup:TESS abs:"exoplanet"',
            "enum_field": "bibgroup",
            "enum_value": "TESS",
            "difficulty": "medium",
        },
        {
            "id": "bibgroup-008",
            "natural_language": "LIGO gravitational wave detections",
            "expected_query": 'bibgroup:LIGO abs:"gravitational wave"',
            "enum_field": "bibgroup",
            "enum_value": "LIGO",
            "difficulty": "medium",
        },
        {
            "id": "bibgroup-009",
            "natural_language": "Spitzer infrared observations",
            "expected_query": "bibgroup:Spitzer",
            "enum_field": "bibgroup",
            "enum_value": "Spitzer",
            "difficulty": "simple",
        },
        {
            "id": "bibgroup-010",
            "natural_language": "Kepler planet discoveries",
            "expected_query": 'bibgroup:Kepler abs:"planet"',
            "enum_field": "bibgroup",
            "enum_value": "Kepler",
            "difficulty": "medium",
        },
        {
            "id": "bibgroup-011",
            "natural_language": "XMM Newton X-ray observations",
            "expected_query": "bibgroup:XMM",
            "enum_field": "bibgroup",
            "enum_value": "XMM",
            "difficulty": "simple",
        },
        {
            "id": "bibgroup-012",
            "natural_language": "Fermi gamma-ray papers",
            "expected_query": "bibgroup:Fermi",
            "enum_field": "bibgroup",
            "enum_value": "Fermi",
            "difficulty": "simple",
        },
        {
            "id": "bibgroup-013",
            "natural_language": "VLA radio observations",
            "expected_query": "bibgroup:VLA",
            "enum_field": "bibgroup",
            "enum_value": "VLA",
            "difficulty": "simple",
        },
        {
            "id": "bibgroup-014",
            "natural_language": "Gemini telescope papers on exoplanets",
            "expected_query": 'bibgroup:Gemini abs:"exoplanets"',
            "enum_field": "bibgroup",
            "enum_value": "Gemini",
            "difficulty": "medium",
        },
        {
            "id": "bibgroup-015",
            "natural_language": "Pan-STARRS survey data papers",
            "expected_query": "bibgroup:Pan-STARRS",
            "enum_field": "bibgroup",
            "enum_value": "Pan-STARRS",
            "difficulty": "simple",
        },
    ],
    "database": [
        {
            "id": "database-001",
            "natural_language": "astronomy papers on dark matter",
            "expected_query": 'database:astronomy abs:"dark matter"',
            "enum_field": "database",
            "enum_value": "astronomy",
            "difficulty": "simple",
        },
        {
            "id": "database-002",
            "natural_language": "physics papers on quantum gravity",
            "expected_query": 'database:physics abs:"quantum gravity"',
            "enum_field": "database",
            "enum_value": "physics",
            "difficulty": "simple",
        },
        {
            "id": "database-003",
            "natural_language": "general science papers",
            "expected_query": "database:general",
            "enum_field": "database",
            "enum_value": "general",
            "difficulty": "simple",
        },
        {
            "id": "database-004",
            "natural_language": "earth science papers on Mars geology",
            "expected_query": 'database:earthscience abs:"Mars geology"',
            "enum_field": "database",
            "enum_value": "earthscience",
            "difficulty": "simple",
        },
        {
            "id": "database-005",
            "natural_language": "astrophysics literature on supernovae",
            "expected_query": 'database:astronomy abs:"supernovae"',
            "enum_field": "database",
            "enum_value": "astronomy",
            "difficulty": "simple",
        },
        {
            "id": "database-006",
            "natural_language": "planetary science papers on Venus atmosphere",
            "expected_query": 'database:earthscience abs:"Venus atmosphere"',
            "enum_field": "database",
            "enum_value": "earthscience",
            "difficulty": "simple",
        },
        {
            "id": "database-007",
            "natural_language": "physics collection on string theory",
            "expected_query": 'database:physics abs:"string theory"',
            "enum_field": "database",
            "enum_value": "physics",
            "difficulty": "simple",
        },
        {
            "id": "database-008",
            "natural_language": "general science interdisciplinary papers",
            "expected_query": "database:general",
            "enum_field": "database",
            "enum_value": "general",
            "difficulty": "simple",
        },
        {
            "id": "database-009",
            "natural_language": "space weather papers in earth science",
            "expected_query": 'database:earthscience abs:"space weather"',
            "enum_field": "database",
            "enum_value": "earthscience",
            "difficulty": "simple",
        },
        {
            "id": "database-010",
            "natural_language": "high energy physics papers",
            "expected_query": 'database:physics abs:"high energy"',
            "enum_field": "database",
            "enum_value": "physics",
            "difficulty": "simple",
        },
    ],
}

# =============================================================================
# EDGE CASES (ambiguous operator words, complex boolean, nested operators)
# =============================================================================

EDGE_CASE_BENCHMARKS = [
    # Ambiguous operator words - should NOT trigger operators
    {
        "id": "edge-001",
        "natural_language": "citing behavior of astronomers",
        "expected_behavior": "no_operator",
        "description": "Word 'citing' used as gerund/topic, not operator trigger",
        "difficulty": "edge",
    },
    {
        "id": "edge-002",
        "natural_language": "reference materials for stellar spectra",
        "expected_behavior": "no_operator",
        "description": "Word 'reference' used as noun/topic, not operator trigger",
        "difficulty": "edge",
    },
    {
        "id": "edge-003",
        "natural_language": "papers about citation analysis techniques",
        "expected_behavior": "no_operator",
        "description": "Word 'citation' used as topic, not operator trigger",
        "difficulty": "edge",
    },
    {
        "id": "edge-004",
        "natural_language": "similar wavelengths in spectroscopy",
        "expected_behavior": "no_operator",
        "description": "Word 'similar' used as adjective, not operator trigger",
        "difficulty": "edge",
    },
    {
        "id": "edge-005",
        "natural_language": "trending towards lower masses",
        "expected_behavior": "no_operator",
        "description": "Word 'trending' used as verb, not operator trigger",
        "difficulty": "edge",
    },
    {
        "id": "edge-006",
        "natural_language": "useful techniques in data analysis",
        "expected_behavior": "no_operator",
        "description": "Word 'useful' used as adjective for 'techniques', not operator",
        "difficulty": "edge",
    },
    {
        "id": "edge-007",
        "natural_language": "reviews of telescope performance",
        "expected_behavior": "ambiguous",
        "description": "'reviews' could be operator or noun - context dependent",
        "difficulty": "edge",
    },
    # Complex boolean logic
    {
        "id": "edge-010",
        "natural_language": "papers by Einstein or Hawking on black holes",
        "expected_query": '(author:Einstein OR author:Hawking) abs:"black holes"',
        "description": "OR operator with multiple authors",
        "difficulty": "complex",
    },
    {
        "id": "edge-011",
        "natural_language": "dark matter but not axions",
        "expected_query": 'abs:"dark matter" NOT abs:axions',
        "description": "NOT/exclusion operator",
        "difficulty": "complex",
    },
    {
        "id": "edge-012",
        "natural_language": "papers on exoplanets or brown dwarfs from 2020",
        "expected_query": '(abs:"exoplanets" OR abs:"brown dwarfs") year:2020',
        "description": "OR with date filter",
        "difficulty": "complex",
    },
    {
        "id": "edge-013",
        "natural_language": "refereed papers excluding conference proceedings",
        "expected_query": "property:refereed NOT doctype:inproceedings",
        "description": "Property with doctype exclusion",
        "difficulty": "complex",
    },
    {
        "id": "edge-014",
        "natural_language": "JWST or HST papers on exoplanet atmospheres",
        "expected_query": '(bibgroup:JWST OR bibgroup:HST) abs:"exoplanet atmospheres"',
        "description": "OR across bibgroups",
        "difficulty": "complex",
    },
    {
        "id": "edge-015",
        "natural_language": "papers from Harvard or MIT but not Stanford",
        "expected_query": "(aff:Harvard OR aff:MIT) NOT aff:Stanford",
        "description": "Complex affiliation filter",
        "difficulty": "complex",
    },
    # Nested operators (should resolve to single operator or no operator)
    {
        "id": "edge-020",
        "natural_language": "citations of references to gravitational waves",
        "expected_behavior": "single_operator",
        "description": "Conflicting operators - should pick dominant one",
        "difficulty": "edge",
    },
    {
        "id": "edge-021",
        "natural_language": "trending papers that cite exoplanet papers",
        "expected_behavior": "single_operator",
        "description": "Nested trending + citations",
        "difficulty": "edge",
    },
    {
        "id": "edge-022",
        "natural_language": "reviews of papers citing LIGO",
        "expected_behavior": "single_operator",
        "description": "Nested reviews + citations",
        "difficulty": "edge",
    },
    # Parentheses edge cases
    {
        "id": "edge-030",
        "natural_language": "papers about NGC (1234)",
        "expected_behavior": "balanced_parens",
        "description": "Parentheses in object name",
        "difficulty": "edge",
    },
    {
        "id": "edge-031",
        "natural_language": "JWST (James Webb Space Telescope) observations",
        "expected_behavior": "balanced_parens",
        "description": "Parenthetical expansion in input",
        "difficulty": "edge",
    },
    {
        "id": "edge-032",
        "natural_language": "(latest) papers on cosmology",
        "expected_behavior": "balanced_parens",
        "description": "Leading parenthetical",
        "difficulty": "edge",
    },
    # Empty and edge inputs
    {
        "id": "edge-040",
        "natural_language": "",
        "expected_behavior": "valid_response",
        "description": "Empty input should not crash",
        "difficulty": "edge",
    },
    {
        "id": "edge-041",
        "natural_language": "   ",
        "expected_behavior": "valid_response",
        "description": "Whitespace-only input should not crash",
        "difficulty": "edge",
    },
    # ADS syntax passthrough
    {
        "id": "edge-050",
        "natural_language": 'author:"Hawking, S" abs:cosmology',
        "expected_behavior": "passthrough",
        "description": "Already ADS syntax - should validate and pass through",
        "difficulty": "edge",
    },
    {
        "id": "edge-051",
        "natural_language": 'citations(abs:"gravitational waves")',
        "expected_behavior": "passthrough",
        "description": "Already ADS operator syntax - pass through",
        "difficulty": "edge",
    },
    {
        "id": "edge-052",
        "natural_language": "bibstem:ApJ pubdate:[2020 TO 2023]",
        "expected_behavior": "passthrough",
        "description": "Already ADS field syntax - pass through",
        "difficulty": "edge",
    },
    # Complex combined queries
    {
        "id": "edge-060",
        "natural_language": "refereed JWST papers on exoplanet atmospheres from 2023",
        "expected_query": 'bibgroup:JWST property:refereed abs:"exoplanet atmospheres" year:2023',
        "description": "Multiple filters combined",
        "difficulty": "complex",
    },
    {
        "id": "edge-061",
        "natural_language": "highly cited refereed papers from Caltech on dark matter",
        "expected_query": 'citation_count:[100 TO *] property:refereed aff:Caltech abs:"dark matter"',
        "description": "Metric + property + affiliation + topic",
        "difficulty": "complex",
    },
    {
        "id": "edge-062",
        "natural_language": "PhD theses on cosmology from Harvard or MIT",
        "expected_query": 'doctype:phdthesis abs:"cosmology" (aff:Harvard OR aff:MIT)',
        "description": "Doctype + topic + complex affiliation",
        "difficulty": "complex",
    },
    {
        "id": "edge-063",
        "natural_language": "open access reviews on galaxy evolution",
        "expected_query": 'property:openaccess reviews(abs:"galaxy evolution")',
        "description": "Property filter with operator",
        "difficulty": "complex",
    },
    {
        "id": "edge-064",
        "natural_language": "papers on M31 galaxy",
        "expected_query": 'object:M31 OR abs:"M31"',
        "description": "Object name in query",
        "difficulty": "medium",
    },
    {
        "id": "edge-065",
        "natural_language": "papers about the Crab Nebula",
        "expected_query": 'object:"Crab Nebula" OR abs:"Crab Nebula"',
        "description": "Multi-word object name",
        "difficulty": "medium",
    },
    # Special characters and edge formatting
    {
        "id": "edge-070",
        "natural_language": "papers on H-alpha emission",
        "expected_query": 'abs:"H-alpha"',
        "description": "Hyphenated term",
        "difficulty": "simple",
    },
    {
        "id": "edge-071",
        "natural_language": "papers on CO2 in planetary atmospheres",
        "expected_query": 'abs:"CO2" abs:"planetary atmospheres"',
        "description": "Chemical formula in query",
        "difficulty": "medium",
    },
    {
        "id": "edge-072",
        "natural_language": "papers about lambda-CDM cosmology",
        "expected_query": 'abs:"lambda-CDM"',
        "description": "Technical term with special characters",
        "difficulty": "simple",
    },
    {
        "id": "edge-073",
        "natural_language": "papers on Sgr A* black hole",
        "expected_query": 'abs:"Sgr A*"',
        "description": "Object name with special character",
        "difficulty": "simple",
    },
    {
        "id": "edge-074",
        "natural_language": "papers on 21-cm cosmology",
        "expected_query": 'abs:"21-cm"',
        "description": "Numeric with hyphen",
        "difficulty": "simple",
    },
    # Proximity and phrase searches
    {
        "id": "edge-080",
        "natural_language": "papers where dark and matter appear within 3 words",
        "expected_query": 'abs:"dark matter"~3',
        "description": "Proximity search",
        "difficulty": "complex",
    },
    {
        "id": "edge-081",
        "natural_language": "exact phrase supermassive black hole in title",
        "expected_query": '=title:"supermassive black hole"',
        "description": "Exact phrase match",
        "difficulty": "medium",
    },
    # Range queries
    {
        "id": "edge-082",
        "natural_language": "papers with 2 to 5 authors",
        "expected_query": "author_count:[2 TO 5]",
        "description": "Author count range",
        "difficulty": "simple",
    },
    {
        "id": "edge-083",
        "natural_language": "papers in volume 500 to 510 of ApJ",
        "expected_query": "bibstem:ApJ volume:[500 TO 510]",
        "description": "Volume range with bibstem",
        "difficulty": "medium",
    },
    # Additional combined filters
    {
        "id": "edge-084",
        "natural_language": "recent highly cited ALMA papers",
        "expected_query": "bibgroup:ALMA citation_count:[100 TO *] year:[2020 TO *]",
        "description": "Bibgroup + metric + date",
        "difficulty": "complex",
    },
    {
        "id": "edge-085",
        "natural_language": "open access software papers",
        "expected_query": "property:openaccess doctype:software",
        "description": "Property + doctype combination",
        "difficulty": "medium",
    },
    {
        "id": "edge-086",
        "natural_language": "first author Hawking papers on black holes from the 1970s",
        "expected_query": 'first_author:Hawking abs:"black holes" year:[1970 TO 1979]',
        "description": "First author + topic + date range",
        "difficulty": "complex",
    },
    {
        "id": "edge-087",
        "natural_language": "preprints with data from MAST",
        "expected_query": "property:eprint data:MAST",
        "description": "Property + data source",
        "difficulty": "medium",
    },
    {
        "id": "edge-088",
        "natural_language": "papers in ApJ Letters about JWST from 2024",
        "expected_query": "bibstem:ApJL bibgroup:JWST year:2024",
        "description": "Bibstem + bibgroup + year",
        "difficulty": "complex",
    },
    {
        "id": "edge-089",
        "natural_language": "single-author PhD theses on cosmology",
        "expected_query": 'author_count:1 doctype:phdthesis abs:"cosmology"',
        "description": "Author count + doctype + topic",
        "difficulty": "complex",
    },
]

# =============================================================================
# REGRESSION TESTS (malformed patterns that must be rejected)
# =============================================================================

REGRESSION_BENCHMARKS = [
    # Known malformed concatenation patterns that must NEVER appear
    {
        "id": "regression-001",
        "natural_language": "papers about references in stellar spectra",
        "forbidden_patterns": ["referencesabs:", "abs:referencesabs:"],
        "description": "References as topic should not concatenate with abs:",
        "difficulty": "regression",
    },
    {
        "id": "regression-002",
        "natural_language": "citing patterns in galaxy evolution",
        "forbidden_patterns": ["citationsabs:", "citingabs:"],
        "description": "Citing as topic should not concatenate",
        "difficulty": "regression",
    },
    {
        "id": "regression-003",
        "natural_language": "useful citations in the field",
        "forbidden_patterns": ["usefulcitations(", "usefulabs:"],
        "description": "Useful + citations should not concatenate operators",
        "difficulty": "regression",
    },
    {
        "id": "regression-004",
        "natural_language": "similar references for comparison",
        "forbidden_patterns": ["similarreferences(", "similarabs:"],
        "description": "Similar + references should not concatenate",
        "difficulty": "regression",
    },
    {
        "id": "regression-005",
        "natural_language": "trending citations in cosmology",
        "forbidden_patterns": ["trendingcitations(", "trendingabs:"],
        "description": "Trending + citations should not concatenate",
        "difficulty": "regression",
    },
    {
        "id": "regression-006",
        "natural_language": "reference analysis techniques",
        "forbidden_patterns": ["referencesabs:", "references(abs:references"],
        "description": "Reference as noun should not trigger operator",
        "difficulty": "regression",
    },
    {
        "id": "regression-007",
        "natural_language": "citation metrics and impact",
        "forbidden_patterns": ["citationsabs:", "citations(abs:citations"],
        "description": "Citation as noun should not trigger operator",
        "difficulty": "regression",
    },
    {
        "id": "regression-008",
        "natural_language": "papers mentioning citing practices",
        "forbidden_patterns": ["citationsabs:", "abs:citationsabs:"],
        "description": "Citing as gerund should not trigger operator",
        "difficulty": "regression",
    },
    {
        "id": "regression-009",
        "natural_language": "studies about referencing methods",
        "forbidden_patterns": ["referencesabs:", "references(abs:references"],
        "description": "Referencing as gerund should not trigger operator",
        "difficulty": "regression",
    },
    {
        "id": "regression-010",
        "natural_language": "overview of citation networks",
        "forbidden_patterns": ["citationsabs:", "abs:citationsabs:"],
        "description": "Citation networks as topic should not concatenate",
        "difficulty": "regression",
    },
    # Known failure strings that should never appear
    {
        "id": "regression-011",
        "natural_language": "papers with citations about cosmology",
        "forbidden_patterns": [
            "citations(abs:citations",
            "citationsabs:",
            "abs:citationsabs:",
        ],
        "description": "Should not produce nested citations",
        "difficulty": "regression",
    },
    {
        "id": "regression-012",
        "natural_language": "papers about useful citations",
        "forbidden_patterns": ["usefulcitations(", "useful(citations"],
        "description": "Should not concatenate useful and citations",
        "difficulty": "regression",
    },
    {
        "id": "regression-013",
        "natural_language": "similar reference works",
        "forbidden_patterns": ["similarreferences(", "similar(references"],
        "description": "Should not concatenate similar and references",
        "difficulty": "regression",
    },
    {
        "id": "regression-014",
        "natural_language": "trending citation metrics",
        "forbidden_patterns": ["trendingcitations(", "trending(citations"],
        "description": "Should not concatenate trending and citations",
        "difficulty": "regression",
    },
    {
        "id": "regression-015",
        "natural_language": "review of references",
        "forbidden_patterns": ["reviewsreferences(", "reviews(references"],
        "description": "Should not concatenate reviews and references",
        "difficulty": "regression",
    },
    # Additional regression tests for balanced parentheses
    {
        "id": "regression-016",
        "natural_language": "papers on exoplanets (hot Jupiters)",
        "expected_behavior": "balanced_parens",
        "forbidden_patterns": [],
        "description": "Input with parentheses should produce balanced output",
        "difficulty": "regression",
    },
    {
        "id": "regression-017",
        "natural_language": "recent (2020+) papers on gravitational waves",
        "expected_behavior": "balanced_parens",
        "forbidden_patterns": [],
        "description": "Input with date in parentheses",
        "difficulty": "regression",
    },
    # Regression tests for field name integrity
    {
        "id": "regression-018",
        "natural_language": "absorbtion line studies",
        "forbidden_patterns": ["absabs:", "abs:abs:"],
        "description": "Should not duplicate abs field",
        "difficulty": "regression",
    },
    {
        "id": "regression-019",
        "natural_language": "author attribution patterns",
        "forbidden_patterns": ["authorauthor:", "author:author:"],
        "description": "Should not duplicate author field",
        "difficulty": "regression",
    },
    {
        "id": "regression-020",
        "natural_language": "year-by-year analysis",
        "forbidden_patterns": ["yearyear:", "year:year:"],
        "description": "Should not duplicate year field",
        "difficulty": "regression",
    },
    # More operator conflation tests
    {
        "id": "regression-021",
        "natural_language": "topical references in astronomy",
        "forbidden_patterns": ["topnreferences(", "topn(references"],
        "description": "Should not conflate topn with references",
        "difficulty": "regression",
    },
    {
        "id": "regression-022",
        "natural_language": "trends similar to previous work",
        "forbidden_patterns": ["trendingsimilar(", "trending(similar"],
        "description": "Should not conflate trending with similar",
        "difficulty": "regression",
    },
    {
        "id": "regression-023",
        "natural_language": "review of similar papers",
        "forbidden_patterns": ["reviewssimilar(", "reviews(similar"],
        "description": "Should not conflate reviews with similar",
        "difficulty": "regression",
    },
    {
        "id": "regression-024",
        "natural_language": "useful reviews on cosmology",
        "forbidden_patterns": ["usefulreviews(", "useful(reviews"],
        "description": "Should not conflate useful with reviews",
        "difficulty": "regression",
    },
    {
        "id": "regression-025",
        "natural_language": "top citing papers",
        "forbidden_patterns": ["topncitations(", "topn(citations"],
        "description": "Should not conflate topn with citations",
        "difficulty": "regression",
    },
    # Enum value regression tests
    {
        "id": "regression-026",
        "natural_language": "refereed preprints on black holes",
        "forbidden_patterns": ["refereedpreprints", "property:refereedpreprints"],
        "description": "Should not concatenate property values",
        "difficulty": "regression",
    },
    {
        "id": "regression-027",
        "natural_language": "article software on astronomy",
        "forbidden_patterns": ["articlesoftware", "doctype:articlesoftware"],
        "description": "Should not concatenate doctype values",
        "difficulty": "regression",
    },
    {
        "id": "regression-028",
        "natural_language": "astronomy physics papers",
        "forbidden_patterns": ["astronomyphysics", "database:astronomyphysics"],
        "description": "Should not concatenate database values",
        "difficulty": "regression",
    },
    # Invalid field value tests
    {
        "id": "regression-029",
        "natural_language": "papers in the invalid_database collection",
        "forbidden_patterns": ["database:invalid_database"],
        "expected_behavior": "valid_database_only",
        "description": "Should only use valid database enum values",
        "difficulty": "regression",
    },
    {
        "id": "regression-030",
        "natural_language": "papers with invalid_property flag",
        "forbidden_patterns": ["property:invalid_property"],
        "expected_behavior": "valid_property_only",
        "description": "Should only use valid property enum values",
        "difficulty": "regression",
    },
]


def generate_benchmark() -> dict:
    """Generate the complete benchmark evaluation set."""
    benchmark = {
        "metadata": {
            "version": "1.0.0",
            "description": "Comprehensive benchmark evaluation set for NL-to-ADS query translation",
            "created": "2026-01-22",
            "target_count": 300,
            "categories": {
                "field_types": list(FIELD_TYPE_BENCHMARKS.keys()),
                "operators": list(OPERATOR_BENCHMARKS.keys()),
                "enum_fields": list(ENUM_BENCHMARKS.keys()),
                "edge_cases": True,
                "regression_tests": True,
            },
        },
        "field_types": {},
        "operators": {},
        "enum_fields": {},
        "edge_cases": EDGE_CASE_BENCHMARKS,
        "regression_tests": REGRESSION_BENCHMARKS,
    }

    # Add field type benchmarks
    total_count = 0
    for field_type, examples in FIELD_TYPE_BENCHMARKS.items():
        benchmark["field_types"][field_type] = examples
        total_count += len(examples)

    # Add operator benchmarks
    for operator, examples in OPERATOR_BENCHMARKS.items():
        benchmark["operators"][operator] = examples
        total_count += len(examples)

    # Add enum field benchmarks
    for enum_field, examples in ENUM_BENCHMARKS.items():
        benchmark["enum_fields"][enum_field] = examples
        total_count += len(examples)

    # Add edge cases and regression tests
    total_count += len(EDGE_CASE_BENCHMARKS)
    total_count += len(REGRESSION_BENCHMARKS)

    benchmark["metadata"]["actual_count"] = total_count

    return benchmark


def print_summary(benchmark: dict) -> None:
    """Print benchmark summary."""
    print("Benchmark Evaluation Set Summary")
    print("=" * 50)

    # Field types
    print("\nField Type Benchmarks:")
    for field_type, examples in benchmark["field_types"].items():
        status = "✓" if len(examples) >= 10 else "⚠"
        print(f"  {status} {field_type}: {len(examples)}")

    # Operators
    print("\nOperator Benchmarks:")
    for operator, examples in benchmark["operators"].items():
        status = "✓" if len(examples) >= 10 else "⚠"
        print(f"  {status} {operator}: {len(examples)}")

    # Enum fields
    print("\nEnum Field Benchmarks:")
    for enum_field, examples in benchmark["enum_fields"].items():
        status = "✓" if len(examples) >= 5 else "⚠"
        print(f"  {status} {enum_field}: {len(examples)}")

    # Edge cases and regression
    print(f"\nEdge Cases: {len(benchmark['edge_cases'])}")
    print(f"Regression Tests: {len(benchmark['regression_tests'])}")

    # Total
    total = benchmark["metadata"]["actual_count"]
    target = benchmark["metadata"]["target_count"]
    status = "✓" if total >= target else "⚠"
    print(f"\n{status} Total: {total} (target: {target}+)")


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive benchmark evaluation set"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/benchmark/benchmark_queries.json"),
        help="Output JSON file path",
    )
    args = parser.parse_args()

    # Generate benchmark
    benchmark = generate_benchmark()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(args.output, "w") as f:
        json.dump(benchmark, f, indent=2)

    # Print summary
    print_summary(benchmark)

    print(f"\nOutput written to: {args.output}")

    # Check if we hit target
    if benchmark["metadata"]["actual_count"] >= 300:
        print("\n✓ Target of 300+ test cases achieved!")
        return 0
    else:
        print(f"\n⚠ Warning: Only {benchmark['metadata']['actual_count']} test cases (target: 300+)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
