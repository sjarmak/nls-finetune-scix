"""Evaluation module for SciX/ADS query generation.

Provides syntactic validation and semantic evaluation via result-set overlap.
"""

import os
from dataclasses import dataclass

import httpx

from finetune.domains.scix.validate import lint_query, validate_query


@dataclass
class EvalResult:
    """Result of evaluating a single query pair."""

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


@dataclass
class EvalSummary:
    """Summary of evaluation across multiple examples."""

    total: int
    syntactically_valid: int
    syntactic_validity_rate: float
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
        List of bibcodes (empty if query fails)
    """
    api_key = api_key or os.environ.get("ADS_API_KEY")
    if not api_key:
        return []

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

        if response.status_code == 200:
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            return [doc["bibcode"] for doc in docs if "bibcode" in doc]

        return []

    except Exception:
        return []


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

    Returns:
        Tuple of (jaccard, precision, recall)
    """
    if not expected and not generated:
        return 1.0, 1.0, 1.0  # Both empty = perfect match

    if not expected or not generated:
        return 0.0, 0.0, 0.0  # One empty = no overlap

    expected_set = set(expected)
    generated_set = set(generated)

    intersection = expected_set & generated_set
    union = expected_set | generated_set

    jaccard = len(intersection) / len(union) if union else 0.0
    precision = len(intersection) / len(generated_set) if generated_set else 0.0
    recall = len(intersection) / len(expected_set) if expected_set else 0.0

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
    # Validate generated query syntax
    validation = validate_query(generated_query, api_key=api_key)

    if not validation.valid:
        return EvalResult(
            nl=nl,
            expected_query=expected_query,
            generated_query=generated_query,
            syntactically_valid=False,
            syntax_errors=validation.errors,
            expected_bibcodes=[],
            generated_bibcodes=[],
            jaccard_overlap=0.0,
            precision_at_n=0.0,
            recall_at_n=0.0,
            category=category,
        )

    # Fetch results for both queries
    expected_bibcodes = fetch_bibcodes(expected_query, n=n, api_key=api_key)
    generated_bibcodes = fetch_bibcodes(generated_query, n=n, api_key=api_key)

    # Compute overlap metrics
    jaccard, precision, recall = compute_overlap_metrics(expected_bibcodes, generated_bibcodes)

    return EvalResult(
        nl=nl,
        expected_query=expected_query,
        generated_query=generated_query,
        syntactically_valid=True,
        syntax_errors=[],
        expected_bibcodes=expected_bibcodes,
        generated_bibcodes=generated_bibcodes,
        jaccard_overlap=jaccard,
        precision_at_n=precision,
        recall_at_n=recall,
        category=category,
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
        - mean_jaccard: average Jaccard similarity
        - mean_precision: average precision
        - mean_recall: average recall
    """
    if not results:
        return {}

    by_category: dict[str, dict[str, float]] = {}

    for r in results:
        cat = r.category or "unknown"
        if cat not in by_category:
            by_category[cat] = {
                "total": 0.0,
                "valid": 0.0,
                "jaccard_sum": 0.0,
                "precision_sum": 0.0,
                "recall_sum": 0.0,
            }
        by_category[cat]["total"] += 1
        if r.syntactically_valid:
            by_category[cat]["valid"] += 1
            by_category[cat]["jaccard_sum"] += r.jaccard_overlap
            by_category[cat]["precision_sum"] += r.precision_at_n
            by_category[cat]["recall_sum"] += r.recall_at_n

    for cat, stats in by_category.items():
        valid_count = stats["valid"]
        total_count = stats["total"]
        by_category[cat] = {
            "total": total_count,
            "valid": valid_count,
            "validity_rate": valid_count / total_count if total_count else 0.0,
            "mean_jaccard": stats["jaccard_sum"] / valid_count if valid_count else 0.0,
            "mean_precision": stats["precision_sum"] / valid_count if valid_count else 0.0,
            "mean_recall": stats["recall_sum"] / valid_count if valid_count else 0.0,
        }

    return by_category


def summarize_results(results: list[EvalResult]) -> EvalSummary:
    """Summarize evaluation results across multiple examples.

    Args:
        results: List of individual EvalResult objects

    Returns:
        EvalSummary with aggregated metrics
    """
    if not results:
        return EvalSummary(
            total=0,
            syntactically_valid=0,
            syntactic_validity_rate=0.0,
            mean_jaccard=0.0,
            mean_precision=0.0,
            mean_recall=0.0,
            by_category={},
        )

    total = len(results)
    valid_results = [r for r in results if r.syntactically_valid]
    syntactically_valid = len(valid_results)

    # Compute means (only over valid results for semantic metrics)
    mean_jaccard = sum(r.jaccard_overlap for r in valid_results) / len(valid_results) if valid_results else 0.0
    mean_precision = sum(r.precision_at_n for r in valid_results) / len(valid_results) if valid_results else 0.0
    mean_recall = sum(r.recall_at_n for r in valid_results) / len(valid_results) if valid_results else 0.0

    # Group by category
    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.category or "unknown"
        if cat not in by_category:
            by_category[cat] = {
                "total": 0,
                "valid": 0,
                "jaccard_sum": 0.0,
                "precision_sum": 0.0,
                "recall_sum": 0.0,
            }
        by_category[cat]["total"] += 1
        if r.syntactically_valid:
            by_category[cat]["valid"] += 1
            by_category[cat]["jaccard_sum"] += r.jaccard_overlap
            by_category[cat]["precision_sum"] += r.precision_at_n
            by_category[cat]["recall_sum"] += r.recall_at_n

    # Compute category averages
    for cat, stats in by_category.items():
        valid_count = stats["valid"]
        by_category[cat] = {
            "total": stats["total"],
            "valid": valid_count,
            "validity_rate": valid_count / stats["total"] if stats["total"] else 0.0,
            "mean_jaccard": stats["jaccard_sum"] / valid_count if valid_count else 0.0,
            "mean_precision": stats["precision_sum"] / valid_count if valid_count else 0.0,
            "mean_recall": stats["recall_sum"] / valid_count if valid_count else 0.0,
        }

    return EvalSummary(
        total=total,
        syntactically_valid=syntactically_valid,
        syntactic_validity_rate=syntactically_valid / total,
        mean_jaccard=mean_jaccard,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
        by_category=by_category,
    )
