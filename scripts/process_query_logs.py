#!/usr/bin/env python3
"""
Process query logs from ADS and generate NL-query training pairs.

Extracts usable queries from the raw CSV, categorizes them, filters out noise,
and prepares them for NL generation.

Usage:
    python scripts/process_query_logs.py \
        --input data/datasets/queries/queries-filtered-unique*.csv \
        --output data/datasets/raw/extracted_queries.json \
        --limit 5000
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Add packages/finetune/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))

from finetune.domains.scix.validate import lint_query


# Patterns to SKIP (not useful for training)
SKIP_PATTERNS = [
    r"^citations\(",         # Citation searches
    r"^references\(",        # Reference searches
    r"^docs\(library/",      # Library document lookups
    r"pubnote:\(",           # Publication notes with DOIs
    r"^#ERROR!$",            # Errors
    r"^bibcode:\d{4}",       # Direct bibcode lookups
    r"identifier:\(",        # Bulk identifier lookups
    r"^similar\(",           # Similar article searches
    r"^trending\(",          # Trending searches
    r"^useful\(",            # Useful searches
    r"^reviews\(",           # Review searches
    r"collection:",          # Old collection syntax (deprecated)
    r"full:\"\d+\"",         # Numeric full-text searches (grant numbers)
    r"^\*:\*$",              # Match all
]

# Patterns for UNFIELDED queries (natural language-like, should output as-is)
UNFIELDED_PATTERNS = [
    r"^[A-Za-z][A-Za-z\s\-\']+$",  # Simple text (e.g., "dark matter halos")
    r"^[A-Z][A-Za-z\s]+\d{4}",     # Text with year (e.g., "galaxy formation 2020")
]


def should_skip(query: str) -> bool:
    """Check if query should be skipped."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    
    # Skip overly long queries (likely complex personal searches)
    if len(query) > 300:
        return True
    
    # Skip queries that look like non-scientific searches
    non_scientific = [
        "diary", "murder", "pirate", "lumberman", "negroes", 
        "biography", "memoir", "trial", "execution", "political",
        "newspaper", "magazine", "wiki", "blog", "official site",
        "san francisco", "masons", "register", "california"
    ]
    query_lower = query.lower()
    if any(term in query_lower for term in non_scientific):
        return True
    
    # Skip queries with many OR clauses (personal author variant searches)
    if query.count(" OR ") > 3:
        return True
    
    # Skip invalid year formats
    if re.search(r"year:\d{1,2}(?!\d)", query):  # year:16 but not year:2016
        return True
    
    # Skip 9999 date ranges (meaningless future dates)
    if "9999" in query:
        return True
    
    # Skip 0000 dates (meaningless past dates)
    if "0000" in query:
        return True
    
    return False


def is_unfielded(query: str) -> bool:
    """Check if query is unfielded (no field prefixes)."""
    field_pattern = r"\b(author|abs|abstract|title|pubdate|year|bibstem|object|keyword|doi|arXiv|orcid|aff|inst|citation_count|read_count|property|database|doctype|full|body|bibcode|identifier|bibgroup|arxiv_class):"
    return not re.search(field_pattern, query, re.IGNORECASE)


def categorize_query(query: str) -> str:
    """Categorize query by its primary field type."""
    query_lower = query.lower()
    
    if "^author:" in query_lower or ("author:" in query_lower and "^" in query):
        return "first_author"
    if "author:" in query_lower:
        return "author"
    if "object:" in query_lower:
        return "astronomy"
    if "bibstem:" in query_lower or "bibcode:" in query_lower:
        return "publication"
    if "citation_count:" in query_lower or "read_count:" in query_lower:
        return "metrics"
    if "aff:" in query_lower or "inst:" in query_lower:
        return "affiliation"
    if "arxiv:" in query_lower or "doi:" in query_lower or "arxiv_class:" in query_lower:
        return "identifiers"
    if "property:" in query_lower or "database:" in query_lower or "doctype:" in query_lower:
        return "properties"
    if "pubdate:" in query_lower or "year:" in query_lower:
        if "abs:" in query_lower or "title:" in query_lower:
            return "compound"
        return "publication"
    if "abs:" in query_lower or "title:" in query_lower or "full:" in query_lower:
        return "content"
    if is_unfielded(query):
        return "unfielded"
    
    return "compound"


def clean_query(query: str) -> str:
    """Clean up query formatting."""
    # Remove extra whitespace
    query = re.sub(r"\s+", " ", query).strip()
    
    # Normalize quotes (some have weird Unicode quotes)
    query = query.replace(""", '"').replace(""", '"')
    query = query.replace("''", '"')
    
    # Fix year format issues (year_2020 -> year:2020)
    query = re.sub(r"year_(\d{4})", r"year:\1", query)
    
    return query


def extract_queries(input_path: Path, limit: int = 0) -> list[dict]:
    """Extract and categorize queries from CSV."""
    queries = []
    seen = set()
    
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 2:
                continue
            
            query = row[1].strip()
            
            # Skip if too short or already seen
            if len(query) < 3 or query in seen:
                continue
            
            seen.add(query)
            
            # Skip unwanted patterns
            if should_skip(query):
                continue
            
            # Clean the query
            query = clean_query(query)
            
            # Validate with linter
            lint_result = lint_query(query)
            if not lint_result.valid:
                continue
            
            # Categorize
            category = categorize_query(query)
            
            queries.append({
                "ads_query": query,
                "category": category,
            })
            
            if limit > 0 and len(queries) >= limit:
                break
    
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Process query logs for training")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV file with query logs"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/datasets/raw/extracted_queries.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="Limit number of queries to extract (0 = all)"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing: {input_path}")
    queries = extract_queries(input_path, args.limit)
    
    # Report category distribution
    print(f"\nExtracted {len(queries)} valid queries")
    print("\nCategory distribution:")
    
    from collections import Counter
    cats = Counter(q["category"] for q in queries)
    for cat, count in cats.most_common():
        pct = 100 * count / len(queries)
        print(f"  {cat:20} {count:5} ({pct:5.1f}%)")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(queries, f, indent=2)
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
