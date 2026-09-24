"""Evaluation module for SciX/ADS query generation.

Provides syntactic validation and semantic evaluation via result-set overlap.
"""

import os
from dataclasses import dataclass

import httpx

from finetune.domains.scix.validate import lint_query, validate_query


class BibcodeFetchError(RuntimeError):
    """ADS could not be queried, so the result set is unknown (not empty)."""


GOLD_EMPTY = "gold query returned no results"


@dataclass
class EvalResult:
    """Result of evaluating a single query pair.

    unscorable_reason is set when overlap cannot be measured: the gold query
    returned nothing, or ADS failed for either query. Such items are excluded
    from every rate and mean rather than counted as a match or a miss.
    """

    nl: str
    expected_query: str
    generated_query: str
    syntactically_valid: bool
    syntax_errors: list[str]
    expected_bibcodes: list[str]
    generated_bibcodes: list[str]
    jaccard_overlap: float
    precision_at_n: float
    recall_at_n: float
    category: str | None = None
    unscorable_reason: str | None = None


@dataclass
class EvalSummary:
    """Summary of evaluation across multiple examples."""

    total: int
    syntactically_valid: int
    syntactic_validity_rate: float
    unscorable: int
    mean_jaccard: float
    mean_precision: float
    mean_recall: float
    by_category: dict[str, dict]


def fetch_bibcodes(
    query: str,
    n: int = 50,
    api_key: str | None = None,
    api_url: str = "https://api.adsabs.harvard.edu/v1/search/query",
) -> list[str]:
    """Fetch top N bibcodes for a query from ADS.

    Args:
        query: ADS query string
        n: Number of results to fetch
        api_key: ADS API key (defaults to env var)
        api_url: ADS API endpoint

    Returns:
        List of bibcodes; empty only when ADS answered with no documents.

    Raises:
        BibcodeFetchError: No API key, a transport error, a non-200 status or
            a malformed body. A failure is never reported as an empty result.
    """
    api_key = api_key or os.environ.get("ADS_API_KEY")
    if not api_key:
        raise BibcodeFetchError("ADS_API_KEY is not set")

    try:
        response = httpx.get(
            api_url,
            params={
                "q": query,
                "rows": n,
                "fl": "bibcode",
                "sort": "score desc",
            },
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=15.0,
        )
    except httpx.HTTPError as e:
        raise BibcodeFetchError(f"ADS request failed: {e}") from e

    if response.status_code != 200:
        raise BibcodeFetchError(f"ADS returned HTTP {response.status_code}")

    try:
        docs = response.json()["response"]["docs"]
    except (ValueError, KeyError, TypeError) as e:
        raise BibcodeFetchError(f"ADS returned a malformed body: {e}") from e
    return [doc["bibcode"] for doc in docs if "bibcode" in doc]


def compute_syntax_validity(queries: list[str]) -> float:
    """Compute what percentage of queries pass offline syntax linting.

    Args:
        queries: List of ADS query strings to validate

    Returns:
        Validity rate from 0.0 to 1.0
    """
    if not queries:
        return 0.0

    valid_count = sum(1 for q in queries if lint_query(q).valid)
    return valid_count / len(queries)


def compute_overlap_metrics(
    expected: list[str], generated: list[str]
) -> tuple[float, float, float]:
    """Compute Jaccard, precision, and recall for result sets.

    Two empty sets share no documents, so they score zero like any other
    empty side. Callers that can tell an empty gold set apart (evaluate_pair)
    exclude those items instead of scoring them.

    Returns:
        Tuple of (jaccard, precision, recall)
    """
    if not expected or not generated:
        return 0.0, 0.0, 0.0

    expected_set = set(expected)
    generated_set = set(generated)

    intersection = expected_set & generated_set
    union = expected_set | generated_set

    jaccard = len(intersection) / len(union)
    precision = len(intersection) / len(generated_set)
    recall = len(intersection) / len(expected_set)

    return jaccard, precision, recall


def evaluate_pair(
    nl: str,
    expected_query: str,
    generated_query: str,
    n: int = 50,
    api_key: str | None = None,
    category: str | None = None,
) -> EvalResult:
    """Evaluate a single NL → query pair.

    The gold query is fetched first, even for an invalid generated query, so
    every arm run on the same items excludes the same gold-empty items.

    Args:
        nl: Natural language input
        expected_query: Ground truth ADS query
        generated_query: Model-generated ADS query
        n: Number of results to compare
        api_key: ADS API key
        category: Optional category for sliced analysis

    Returns:
        EvalResult with all metrics
    """
    validation = validate_query(generated_query, api_key=api_key)
    base = dict(
        nl=nl,
        expected_query=expected_query,
        generated_query=generated_query,
        syntactically_valid=validation.valid,
        syntax_errors=[] if validation.valid else validation.errors,
        category=category,
    )
    unscored = dict(
        base, generated_bibcodes=[], jaccard_overlap=0.0, precision_at_n=0.0, recall_at_n=0.0
    )

    try:
        expected_bibcodes = fetch_bibcodes(expected_query, n=n, api_key=api_key)
    except BibcodeFetchError as e:
        return EvalResult(**unscored, expected_bibcodes=[], unscorable_reason=f"gold: {e}")
    if not expected_bibcodes:
        return EvalResult(**unscored, expected_bibcodes=[], unscorable_reason=GOLD_EMPTY)
    if not validation.valid:
        return EvalResult(**unscored, expected_bibcodes=expected_bibcodes)

    try:
        generated_bibcodes = fetch_bibcodes(generated_query, n=n, api_key=api_key)
    except BibcodeFetchError as e:
        return EvalResult(
            **unscored, expected_bibcodes=expected_bibcodes, unscorable_reason=f"generated: {e}"
        )

    jaccard, precision, recall = compute_overlap_metrics(expected_bibcodes, generated_bibcodes)
    return EvalResult(
        **base,
        expected_bibcodes=expected_bibcodes,
        generated_bibcodes=generated_bibcodes,
        jaccard_overlap=jaccard,
        precision_at_n=precision,
        recall_at_n=recall,
    )


def _means(scored: list[EvalResult]) -> tuple[float, float, float]:
    """Mean Jaccard, precision and recall over syntactically valid scored items."""
    valid = [r for r in scored if r.syntactically_valid]
    if not valid:
        return 0.0, 0.0, 0.0
    return (
        sum(r.jaccard_overlap for r in valid) / len(valid),
        sum(r.precision_at_n for r in valid) / len(valid),
        sum(r.recall_at_n for r in valid) / len(valid),
    )


def evaluate_by_category(
    results: list[EvalResult],
) -> dict[str, dict[str, float]]:
    """Evaluate queries by category (author, pubdate, bibstem, object) separately.

    Args:
        results: List of EvalResult objects with category set

    Returns:
        Dict mapping category name to metrics dict with keys:
        - total: number of examples in category
        - valid: number syntactically valid
        - validity_rate: percentage valid (0.0-1.0)
        - unscorable: number excluded because overlap could not be measured
        - mean_jaccard / mean_precision / mean_recall: averages over valid,
          scorable examples
    """
    grouped: dict[str, list[EvalResult]] = {}
    for r in results:
        grouped.setdefault(r.category or "unknown", []).append(r)

    by_category: dict[str, dict[str, float]] = {}
    for cat, items in grouped.items():
        valid_count = sum(1 for r in items if r.syntactically_valid)
        scored = [r for r in items if r.unscorable_reason is None]
        jaccard, precision, recall = _means(scored)
        by_category[cat] = {
            "total": len(items),
            "valid": valid_count,
            "validity_rate": valid_count / len(items),
            "unscorable": len(items) - len(scored),
            "mean_jaccard": jaccard,
            "mean_precision": precision,
            "mean_recall": recall,
        }
    return by_category


def summarize_results(results: list[EvalResult]) -> EvalSummary:
    """Summarize evaluation results across multiple examples.

    Syntactic validity counts every item. Overlap means cover only valid items
    whose overlap could be measured (unscorable_reason is None).

    Args:
        results: List of individual EvalResult objects

    Returns:
        EvalSummary with aggregated metrics
    """
    total = len(results)
    syntactically_valid = sum(1 for r in results if r.syntactically_valid)
    scored = [r for r in results if r.unscorable_reason is None]
    mean_jaccard, mean_precision, mean_recall = _means(scored)

    return EvalSummary(
        total=total,
        syntactically_valid=syntactically_valid,
        syntactic_validity_rate=syntactically_valid / total if total else 0.0,
        unscorable=total - len(scored),
        mean_jaccard=mean_jaccard,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
        by_category=evaluate_by_category(results),
    )
