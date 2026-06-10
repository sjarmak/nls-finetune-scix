"""ADS query evaluation.

Validates generated queries with the offline linter and measures semantic
agreement as ADS result-set overlap (Jaccard over the top-N bibcodes returned
by the ADS search API for each query).
"""

import os
from dataclasses import dataclass

from finetune.domains.scix.eval import compute_overlap_metrics, fetch_bibcodes
from finetune.domains.scix.validate import lint_query

# Two queries are a semantic match when their top-N result sets overlap at
# least this much (Jaccard). Calibrated mechanical threshold, not a judgment.
SEMANTIC_MATCH_THRESHOLD = 0.5


@dataclass
class QueryEvalResult:
    """Result from evaluating a query against expected."""

    valid: bool
    match: bool
    overlap: float


def _normalize(query: str) -> str:
    """Whitespace- and case-normalize a query for mechanical equality."""
    return " ".join(query.split()).lower()


def evaluate_query(
    expected: str,
    actual: str,
    n: int = 50,
    api_key: str | None = None,
) -> QueryEvalResult:
    """Evaluate a generated ADS query against the expected query.

    Checks:
    - Syntax validity via the offline linter (no API call)
    - Semantic match via ADS result-set overlap: fetch the top-N bibcodes for
      both queries and compute Jaccard overlap. Identical (normalized) queries
      short-circuit to a perfect match without API calls.

    Args:
        expected: The ground truth ADS query
        actual: The generated ADS query to evaluate
        n: Number of results per query to compare
        api_key: ADS API key (defaults to ADS_API_KEY env var)

    Returns:
        QueryEvalResult with valid, match, and overlap fields

    Raises:
        RuntimeError: If semantic comparison is needed but no ADS API key
            is configured.
    """
    if not lint_query(actual).valid:
        return QueryEvalResult(valid=False, match=False, overlap=0.0)

    if _normalize(expected) == _normalize(actual):
        return QueryEvalResult(valid=True, match=True, overlap=1.0)

    api_key = api_key or os.environ.get("ADS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ADS_API_KEY is required for semantic evaluation: queries differ "
            "and result-set overlap must be computed via the ADS search API."
        )

    expected_bibcodes = fetch_bibcodes(expected, n=n, api_key=api_key)
    actual_bibcodes = fetch_bibcodes(actual, n=n, api_key=api_key)

    jaccard, _precision, _recall = compute_overlap_metrics(expected_bibcodes, actual_bibcodes)

    return QueryEvalResult(
        valid=True,
        match=jaccard >= SEMANTIC_MATCH_THRESHOLD,
        overlap=jaccard,
    )
