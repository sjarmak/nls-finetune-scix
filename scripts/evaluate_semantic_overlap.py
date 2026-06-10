#!/usr/bin/env python3
"""Semantic evaluation via ADS result-set overlap (PRD Phase 4).

Runs benchmark queries through a generation mode and compares the ADS result
sets of generated vs expected queries (Jaccard, Precision@N, Recall@N), plus
syntax validity — sliced by benchmark category. This is the metric that the
PRD targets (semantic match >= 70%, syntax validity >= 95%) are defined on.

Modes:
    pipeline  - run the hybrid NER pipeline in-process (no server needed)
    server    - call a running NLS server's /v1/chat/completions endpoint;
                point it at a hybrid, pipeline-only, or model-only server
                (ROUTING_MODE env var on the server) to compare routing modes.

Usage:
    # Deterministic pipeline, in-process (requires ADS_API_KEY)
    python scripts/evaluate_semantic_overlap.py --mode pipeline

    # Against a running server (hybrid or model-only)
    python scripts/evaluate_semantic_overlap.py --mode server \\
        --endpoint http://localhost:8001 --label hybrid

    # Quick run on a sample
    python scripts/evaluate_semantic_overlap.py --mode pipeline --limit 20

Results are written to data/datasets/evaluations/semantic_overlap_<label>_<date>.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages/finetune/src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_benchmark import flatten_tests, load_benchmark  # noqa: E402

from finetune.domains.scix.eval import evaluate_pair, summarize_results  # noqa: E402

DEFAULT_BENCHMARK = REPO_ROOT / "data/datasets/benchmark/benchmark_queries.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/datasets/evaluations"
SEMANTIC_MATCH_THRESHOLD = 0.5
TARGET_SEMANTIC_MATCH_RATE = 0.70
TARGET_SYNTAX_VALIDITY_RATE = 0.95
SYSTEM_PROMPT = 'Convert natural language to ADS search query. Output JSON: {"query": "..."}'


def generate_via_pipeline(nl_query: str) -> str:
    """Generate a query with the in-process hybrid pipeline."""
    from finetune.domains.scix.pipeline import process_query

    return process_query(nl_query).final_query


def generate_via_server(nl_query: str, endpoint: str, client: httpx.Client) -> str:
    """Generate a query by calling a running NLS server."""
    url = endpoint.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = f"{url}/v1/chat/completions"

    response = client.post(
        url,
        json={
            "model": "llm",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Query: {nl_query}\nDate: {datetime.now(UTC).date().isoformat()}",
                },
            ],
            "max_tokens": 128,
            "temperature": 0,
        },
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()

    # Servers may return the bare query or {"query": "..."} JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "query" in parsed:
            return parsed["query"]
    except json.JSONDecodeError:
        pass
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pipeline", "server"], default="pipeline")
    parser.add_argument(
        "--endpoint", default="http://localhost:8001", help="Server URL (server mode)"
    )
    parser.add_argument(
        "--label", default=None, help="Label for the output artifact (default: mode)"
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases")
    parser.add_argument("--rows", type=int, default=50, help="Result-set size N for overlap")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds between ADS API calls")
    args = parser.parse_args()

    if not os.environ.get("ADS_API_KEY"):
        print("ERROR: ADS_API_KEY must be set for result-set evaluation", file=sys.stderr)
        return 1

    label = args.label or args.mode
    benchmark = load_benchmark(args.benchmark)
    cases = [
        (test, category, subcategory)
        for test, category, subcategory in flatten_tests(benchmark)
        if test.get("expected_query")
    ]
    if args.limit:
        cases = cases[: args.limit]

    print(f"Evaluating {len(cases)} benchmark cases (mode={args.mode}, label={label})")

    client = httpx.Client(timeout=120.0) if args.mode == "server" else None
    results = []
    generation_errors = []

    for i, (test, category, subcategory) in enumerate(cases):
        nl = test["natural_language"]
        expected = test["expected_query"]

        try:
            if args.mode == "pipeline":
                generated = generate_via_pipeline(nl)
            else:
                generated = generate_via_server(nl, args.endpoint, client)
        except Exception as e:
            generation_errors.append({"id": test.get("id"), "nl": nl, "error": str(e)})
            print(f"  [{i + 1}/{len(cases)}] GENERATION ERROR for {test.get('id')}: {e}")
            continue

        result = evaluate_pair(
            nl=nl,
            expected_query=expected,
            generated_query=generated,
            n=args.rows,
            category=f"{category}/{subcategory}",
        )
        results.append(result)
        print(
            f"  [{i + 1}/{len(cases)}] {test.get('id')}: "
            f"valid={result.syntactically_valid} jaccard={result.jaccard_overlap:.2f}"
        )
        time.sleep(args.sleep)

    if client:
        client.close()

    summary = summarize_results(results)
    semantic_matches = sum(
        1
        for r in results
        if r.syntactically_valid and r.jaccard_overlap >= SEMANTIC_MATCH_THRESHOLD
    )
    semantic_match_rate = semantic_matches / len(results) if results else 0.0

    artifact = {
        "metadata": {
            "mode": args.mode,
            "label": label,
            "endpoint": args.endpoint if args.mode == "server" else None,
            "benchmark": str(args.benchmark),
            "cases_evaluated": len(results),
            "generation_errors": len(generation_errors),
            "rows": args.rows,
            "date": datetime.now(UTC).isoformat(),
            "semantic_match_threshold": SEMANTIC_MATCH_THRESHOLD,
        },
        "summary": {
            "syntax_validity_rate": summary.syntactic_validity_rate,
            "semantic_match_rate": semantic_match_rate,
            "mean_jaccard": summary.mean_jaccard,
            "mean_precision": summary.mean_precision,
            "mean_recall": summary.mean_recall,
            "targets": {
                "syntax_validity": {
                    "target": TARGET_SYNTAX_VALIDITY_RATE,
                    "actual": summary.syntactic_validity_rate,
                    "pass": summary.syntactic_validity_rate >= TARGET_SYNTAX_VALIDITY_RATE,
                },
                "semantic_match": {
                    "target": TARGET_SEMANTIC_MATCH_RATE,
                    "actual": semantic_match_rate,
                    "pass": semantic_match_rate >= TARGET_SEMANTIC_MATCH_RATE,
                },
            },
        },
        "by_category": summary.by_category,
        "results": [asdict(r) for r in results],
        "generation_errors": generation_errors,
    }

    output = args.output or (
        DEFAULT_OUTPUT_DIR / f"semantic_overlap_{label}_{datetime.now(UTC).date().isoformat()}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nSyntax validity: {summary.syntactic_validity_rate:.1%} (target ≥95%)")
    print(
        f"Semantic match (Jaccard ≥ {SEMANTIC_MATCH_THRESHOLD}): "
        f"{semantic_match_rate:.1%} (target ≥70%)"
    )
    print(f"Mean Jaccard: {summary.mean_jaccard:.3f}")
    print(f"Artifact: {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
