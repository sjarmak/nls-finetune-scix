"""Prompts for SciX/ADS query generation and NL generation."""

# System prompt for the fine-tuned model
SYSTEM_PROMPT = 'Convert natural language to ADS/SciX search query. Output JSON: {"query": "..."}'

# Prompt for generating natural language from ADS queries (synthetic data)
NL_GENERATION_PROMPT = """Generate what a researcher would actually type into a scientific literature search box. NOT a formal instruction.

Query to match:
{query}

BANNED WORDS (never use): "Find", "Search for", "Locate", "Retrieve", "Show me", "List"

BANNED SYNTAX (never include ADS field prefixes):
- author:, abs:, title:, pubdate:, bibstem:, object:, keyword:, doi:, arXiv:
- Do NOT use brackets like [2020 TO 2023]
- Do NOT use ^ for first author

Examples of GOOD natural queries for scientific literature:
- "papers by Hawking on black holes"
- "dark matter review articles"
- "exoplanet atmospheres in the astrophysical journal"
- "gravitational waves 2015-2017"
- "papers citing the original LIGO detection"
- "cosmology papers from harvard"
- "highly cited machine learning astronomy papers"
- "recent james webb space telescope results"
- "M31 star formation rates"
- "phd theses on galaxy evolution"

Examples of BAD queries (too formal or leaking syntax):
- "Find papers by author:Einstein" ❌
- "Search for abs:dark matter" ❌
- "pubdate:[2020 TO 2023] papers" ❌
- "Show me the bibliography" ❌

Pick ONE style randomly:
1. Simple topic: "dark matter halos"
2. Author + topic: "papers by turing on computability"
3. Topic + time: "gravitational waves 2015 to 2017"
4. Topic + journal: "exoplanets in nature"
5. Topic + institution: "cosmology papers from MIT"
6. Question style: "what are the latest JWST findings?"
7. Object-focused: "papers about M87 black hole"
8. Looking for: "looking for reviews on galaxy formation"

DO NOT use ADS syntax (author:, abs:, bibstem:, pubdate:, etc.)

Respond with ONLY the query text. Nothing else."""

# Prompt for query complexity classification
QUERY_COMPLEXITY_PROMPT = """Classify this ADS query by complexity:

Query: {query}

Categories:
- simple: Single field search (e.g., author:"Name" or abs:"topic")
- compound: Multiple fields with AND/OR (e.g., author:"X" AND abs:"Y")
- advanced: Ranges, negations, nested expressions (e.g., pubdate:[X TO Y] NOT author:"Z")

Respond with just the category name."""

# Few-shot examples for training format
TRAINING_EXAMPLES = [
    {
        "nl": "papers by Stephen Hawking on black hole radiation",
        "query": 'author:"Hawking, S" abs:"black hole radiation"',
    },
    {
        "nl": "dark matter review articles from 2020",
        "query": 'abs:"dark matter" doctype:article property:refereed pubdate:2020',
    },
    {
        "nl": "highly cited gravitational wave papers",
        "query": 'abs:"gravitational waves" citation_count:[100 TO *]',
    },
    {
        "nl": "exoplanet atmospheres in the astrophysical journal",
        "query": 'abs:"exoplanet atmosphere" bibstem:ApJ',
    },
    {
        "nl": "papers about the Andromeda galaxy from Harvard",
        "query": 'object:M31 aff:"Harvard"',
    },
    {
        "nl": "machine learning in astronomy last 5 years",
        "query": 'abs:"machine learning" database:astronomy pubdate:[2021 TO 2026]',
    },
    {
        "nl": "first author papers by Jane Smith on cosmology",
        "query": '^author:"Smith, Jane" abs:cosmology',
    },
    {
        "nl": "JWST early release science papers",
        "query": 'full:"JWST" OR full:"James Webb" abs:"early release science"',
    },
]
