"""Sample diverse ADS abstracts for human annotation of NER spans.

Queries the NASA ADS API for 100 diverse scientific abstracts across four
domains: astronomy, earth science, planetary science, and multidisciplinary.
Outputs JSONL records with bibcode, title, abstract, database, and keywords.

Requires the ADS_API_KEY environment variable to be set.

Usage:
    python scripts/sample_ads_abstracts.py \
        --output-file data/evaluation/ads_sample_raw.jsonl

    # Customize per-domain count:
    python scripts/sample_ads_abstracts.py \
        --output-file data/evaluation/ads_sample_raw.jsonl \
        --per-domain 30

    # Dry run (print queries without executing):
    python scripts/sample_ads_abstracts.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADS_API_BASE = "https://api.adsabs.harvard.edu/v1/search/query"

# Fields to request from the ADS API
ADS_FIELDS = [
    "bibcode",
    "title",
    "abstract",
    "database",
    "keyword",
    "keyword_schema",
    "year",
    "doctype",
    "citation_count",
]

# Domain query configurations.
# Each domain uses ADS search syntax to select representative papers.
DOMAIN_QUERIES: dict[str, dict[str, str]] = {
    "astronomy": {
        "query": 'database:astronomy doctype:article abs:("dark matter" OR "galaxy" OR "stellar" OR "supernova" OR "black hole" OR "neutron star" OR "exoplanet" OR "quasar" OR "cosmic" OR "gravitational")',
        "description": "Astronomy and astrophysics papers",
    },
    "earth_science": {
        "query": 'database:earthscience doctype:article abs:("climate" OR "ocean" OR "atmospheric" OR "precipitation" OR "aerosol" OR "geophysical" OR "seismic" OR "hydrological" OR "cryosphere" OR "remote sensing")',
        "description": "Earth science and geophysics papers",
    },
    "planetary_science": {
        "query": 'database:astronomy doctype:article abs:("Mars" OR "lunar" OR "planetary" OR "asteroid" OR "comet" OR "crater" OR "Titan" OR "Jupiter" OR "Venus" OR "Mercury")',
        "description": "Planetary science papers",
    },
    "multidisciplinary": {
        "query": 'doctype:article abs:("interdisciplinary" OR "multi-wavelength" OR "astrochemistry" OR "astrobiology" OR "space weather" OR "heliophysics" OR "magnetosphere" OR "biomarker" OR "spectroscopy")',
        "description": "Multidisciplinary or cross-domain papers",
    },
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ADSRecord:
    """A single ADS abstract record for annotation."""

    bibcode: str
    title: str
    abstract: str
    database: list[str]
    keywords: list[str]
    year: int
    doctype: str
    citation_count: int
    domain_category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SamplingStats:
    """Statistics about the sampling process."""

    total_fetched: int = 0
    per_domain: dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    no_abstract_skipped: int = 0
    short_abstract_skipped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_fetched": self.total_fetched,
            "per_domain": dict(sorted(self.per_domain.items())),
            "duplicates_removed": self.duplicates_removed,
            "no_abstract_skipped": self.no_abstract_skipped,
            "short_abstract_skipped": self.short_abstract_skipped,
        }


# ---------------------------------------------------------------------------
# ADS API interaction
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Retrieve the ADS API key from the environment.

    Raises:
        SystemExit: If ADS_API_KEY is not set.
    """
    api_key = os.environ.get("ADS_API_KEY", "")
    if not api_key:
        print(
            "Error: ADS_API_KEY environment variable is not set.\n"
            "Get your API key from https://ui.adsabs.harvard.edu/user/settings/token\n"
            "Then: export ADS_API_KEY=your_key_here"
        )
        sys.exit(1)
    return api_key


def query_ads(
    query: str,
    api_key: str,
    rows: int = 50,
    start: int = 0,
    sort: str = "citation_count desc",
) -> list[dict[str, Any]]:
    """Query the ADS API and return raw doc records.

    Args:
        query: ADS search query string.
        api_key: ADS API bearer token.
        rows: Number of results to return.
        start: Starting offset for pagination.
        sort: Sort order for results.

    Returns:
        List of document dicts from the ADS API response.

    Raises:
        RuntimeError: If the API request fails.
    """
    import httpx

    params: dict[str, Any] = {
        "q": query,
        "fl": ",".join(ADS_FIELDS),
        "rows": rows,
        "start": start,
        "sort": sort,
    }

    response = httpx.get(
        ADS_API_BASE,
        params=params,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"ADS API returned status {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    return data.get("response", {}).get("docs", [])


def parse_ads_doc(doc: dict[str, Any], domain_category: str) -> ADSRecord | None:
    """Parse a raw ADS API document into an ADSRecord.

    Returns None if the document lacks a usable abstract.
    """
    abstract = doc.get("abstract", "")
    if not abstract or abstract == "Not Available":
        return None

    title_list = doc.get("title", [])
    title = title_list[0] if title_list else ""

    keywords = doc.get("keyword", [])

    return ADSRecord(
        bibcode=doc.get("bibcode", ""),
        title=title,
        abstract=abstract,
        database=doc.get("database", []),
        keywords=keywords,
        year=doc.get("year", 0),
        doctype=doc.get("doctype", ""),
        citation_count=doc.get("citation_count", 0),
        domain_category=domain_category,
    )


# ---------------------------------------------------------------------------
# Sampling logic
# ---------------------------------------------------------------------------


MIN_ABSTRACT_LENGTH = 100  # characters


def sample_domain(
    domain: str,
    config: dict[str, str],
    api_key: str,
    target_count: int,
    seen_bibcodes: set[str],
    stats: SamplingStats,
) -> list[ADSRecord]:
    """Sample abstracts for a single domain.

    Fetches more than needed to account for filtering, deduplication,
    and missing abstracts.

    Args:
        domain: Domain name (astronomy, earth_science, etc.).
        config: Domain query config with 'query' and 'description' keys.
        api_key: ADS API key.
        target_count: Number of abstracts to collect for this domain.
        seen_bibcodes: Set of already-collected bibcodes (for dedup across domains).
        stats: Mutable statistics accumulator.

    Returns:
        List of ADSRecord objects (up to target_count).
    """
    records: list[ADSRecord] = []
    fetch_size = target_count * 3  # over-fetch to account for filtering
    start = 0
    max_pages = 3

    print(f"\n  [{domain}] {config['description']}")
    print(f"    Query: {config['query'][:80]}...")

    for page in range(max_pages):
        if len(records) >= target_count:
            break

        docs = query_ads(
            query=config["query"],
            api_key=api_key,
            rows=fetch_size,
            start=start,
            sort="citation_count desc",
        )

        if not docs:
            print(f"    Page {page + 1}: no results returned")
            break

        print(f"    Page {page + 1}: fetched {len(docs)} documents")

        for doc in docs:
            if len(records) >= target_count:
                break

            parsed = parse_ads_doc(doc, domain)
            if parsed is None:
                stats.no_abstract_skipped += 1
                continue

            if len(parsed.abstract) < MIN_ABSTRACT_LENGTH:
                stats.short_abstract_skipped += 1
                continue

            if parsed.bibcode in seen_bibcodes:
                stats.duplicates_removed += 1
                continue

            seen_bibcodes.add(parsed.bibcode)
            records.append(parsed)

        start += fetch_size

        # Rate limit: ADS allows 5,000 requests/day, be polite
        time.sleep(0.5)

    stats.per_domain[domain] = len(records)
    stats.total_fetched += len(records)
    print(f"    Collected: {len(records)}/{target_count}")

    return records


def sample_all_domains(
    api_key: str,
    per_domain: int = 25,
) -> tuple[list[ADSRecord], SamplingStats]:
    """Sample abstracts from all domains.

    Args:
        api_key: ADS API bearer token.
        per_domain: Number of abstracts per domain.

    Returns:
        (records, stats) tuple.
    """
    stats = SamplingStats()
    all_records: list[ADSRecord] = []
    seen_bibcodes: set[str] = set()

    print(f"Sampling {per_domain} abstracts per domain ({len(DOMAIN_QUERIES)} domains)...")

    for domain, config in DOMAIN_QUERIES.items():
        domain_records = sample_domain(
            domain=domain,
            config=config,
            api_key=api_key,
            target_count=per_domain,
            seen_bibcodes=seen_bibcodes,
            stats=stats,
        )
        all_records.extend(domain_records)

    return all_records, stats


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_records(records: list[ADSRecord], path: Path) -> None:
    """Write ADSRecord list as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def write_stats(stats: SamplingStats, path: Path) -> None:
    """Write sampling statistics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample diverse ADS abstracts for NER annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/sample_ads_abstracts.py \\
                --output-file data/evaluation/ads_sample_raw.jsonl

              python scripts/sample_ads_abstracts.py \\
                --output-file data/evaluation/ads_sample_raw.jsonl \\
                --per-domain 30

              python scripts/sample_ads_abstracts.py --dry-run
        """),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/evaluation/ads_sample_raw.jsonl"),
        help="Path to write output JSONL (default: data/evaluation/ads_sample_raw.jsonl)",
    )
    parser.add_argument(
        "--per-domain",
        type=int,
        default=25,
        help="Number of abstracts per domain (default: 25, 4 domains = 100 total)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print queries without executing them",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — Queries that would be executed:\n")
        for domain, config in DOMAIN_QUERIES.items():
            print(f"  [{domain}] {config['description']}")
            print(f"    {config['query']}")
            print()
        print(f"Target: {args.per_domain} abstracts per domain, "
              f"{args.per_domain * len(DOMAIN_QUERIES)} total")
        return 0

    api_key = _get_api_key()

    records, stats = sample_all_domains(api_key, per_domain=args.per_domain)

    # Summary
    print(f"\n{'=' * 60}")
    print("Sampling Summary")
    print(f"{'=' * 60}")
    print(f"  Total abstracts: {len(records)}")
    for domain, count in sorted(stats.per_domain.items()):
        print(f"    {domain:25s}: {count}")
    print(f"  Duplicates removed:      {stats.duplicates_removed}")
    print(f"  No-abstract skipped:     {stats.no_abstract_skipped}")
    print(f"  Short-abstract skipped:  {stats.short_abstract_skipped}")
    print(f"{'=' * 60}")

    # Write output
    write_records(records, args.output_file)
    print(f"\nRecords written to {args.output_file}")

    # Write stats alongside output
    stats_path = args.output_file.with_suffix(".stats.json")
    write_stats(stats, stats_path)
    print(f"Statistics written to {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
