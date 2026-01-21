"""Query evaluation using the nls_query_eval CLI.

Shells out to the Sourcegraph query evaluator for proper syntax validation
and semantic comparison.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QueryEvalResult:
    """Result from evaluating a query against expected."""

    valid: bool
    match: bool
    overlap: float


def find_eval_binary() -> Path:
    """Find the nls_query_eval binary.

    Searches in order:
    1. Repository root (development)
    2. PATH (installed)
    """
    repo_root = Path(__file__).parents[5]
    local_binary = repo_root / "nls_query_eval"
    if local_binary.exists():
        return local_binary

    import shutil

    path_binary = shutil.which("nls_query_eval")
    if path_binary:
        return Path(path_binary)

    raise FileNotFoundError(
        "nls_query_eval binary not found. Expected at repository root or in PATH."
    )


def evaluate_query(expected: str, actual: str) -> QueryEvalResult:
    """Evaluate a generated query against the expected query.

    Uses the nls_query_eval CLI which provides:
    - Syntax validation using Sourcegraph's actual parser
    - Semantic comparison with proper operator value matching

    Args:
        expected: The ground truth query
        actual: The generated query to evaluate

    Returns:
        QueryEvalResult with valid, match, and overlap fields

    Raises:
        RuntimeError: If the CLI fails to execute
    """
    binary = find_eval_binary()

    try:
        result = subprocess.run(
            [str(binary), "--expected", expected, "--actual", actual],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            raise RuntimeError(f"nls_query_eval failed: {result.stderr}")

        data = json.loads(result.stdout)
        return QueryEvalResult(
            valid=data["valid"],
            match=data["match"],
            overlap=data["overlap"],
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError("nls_query_eval timed out")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse nls_query_eval output: {e}")
