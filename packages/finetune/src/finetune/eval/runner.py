"""Evaluation runner for fine-tuned NLS Query model.

Loads validation examples, runs inference, and computes metrics.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from finetune.eval.eval import evaluate_query


@dataclass
class EvalExample:
    """A single evaluation example."""

    id: str
    input: str  # Natural language query
    expected: str  # Expected Sourcegraph query
    raw_messages: list[dict] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single example."""

    id: str
    input: str
    expected: str
    output: str
    syntax_valid: bool
    semantic_match: bool
    overlap: float
    latency_ms: float
    # Server-side token usage (vLLM only)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class EvalRunner:
    """Runner for evaluating fine-tuned models."""

    DEFAULT_VAL_PATH = Path("data/datasets/processed/val.jsonl")

    def __init__(self, val_path: Path | None = None):
        """Initialize the evaluation runner.

        Args:
            val_path: Path to validation JSONL file. Defaults to val.jsonl.
        """
        self.val_path = val_path or self.DEFAULT_VAL_PATH
        # Shared HTTP client for connection pooling (2.3x faster)
        # Lazy-initialized on first use
        self._client: httpx.Client | None = None

    def _get_client(self, timeout: float = 30.0) -> httpx.Client:
        """Get or create shared HTTP client with connection pooling.

        Connection pooling eliminates TCP+SSL handshake overhead (~600ms)
        for subsequent requests, providing 2.3x speedup.
        """
        if self._client is None:
            self._client = httpx.Client(
                http2=True,  # HTTP/2 for better connection reuse
                timeout=timeout,
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "EvalRunner":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit - close client."""
        self.close()

    def load_examples(self) -> list[EvalExample]:
        """Load evaluation examples from validation file.

        Returns:
            List of EvalExample objects parsed from val.jsonl
        """
        examples = []
        with open(self.val_path) as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())
                messages = data.get("messages", [])

                # Extract user message (input)
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

                # Parse query from user message
                query_match = re.search(r"Query:\s*(.+?)(?:\n|$)", user_msg)
                input_query = query_match.group(1).strip() if query_match else user_msg

                # Extract assistant message (expected output)
                assistant_msg = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), ""
                )

                # Parse expected query from JSON response
                expected_query = assistant_msg
                try:
                    parsed = json.loads(assistant_msg)
                    expected_query = parsed.get("query", assistant_msg)
                except json.JSONDecodeError:
                    pass

                examples.append(
                    EvalExample(
                        id=f"ex-{idx:03d}",
                        input=input_query,
                        expected=expected_query,
                        raw_messages=messages,
                    )
                )

        return examples

    def evaluate_example(
        self,
        example: EvalExample,
        model_endpoint: str,
        timeout: float = 30.0,
        endpoint_type: str = "legacy",
    ) -> EvalResult:
        """Evaluate a single example against the model.

        Args:
            example: The example to evaluate
            model_endpoint: URL of the model inference endpoint
            timeout: Request timeout in seconds
            endpoint_type: "legacy" for transformers endpoint, "vllm" for vLLM OpenAI-compatible

        Returns:
            EvalResult with metrics
        """
        # Call the model endpoint
        start_time = time.time()
        prompt_tokens = None
        completion_tokens = None
        try:
            # Use shared client for connection pooling (2.3x faster)
            client = self._get_client(timeout)
            if endpoint_type == "vllm":
                # vLLM OpenAI-compatible format - append path if not present
                url = model_endpoint.rstrip("/")
                if not url.endswith("/v1/chat/completions"):
                    # Handle case where URL ends with /v1 already
                    if url.endswith("/v1"):
                        url = f"{url}/chat/completions"
                    else:
                        url = f"{url}/v1/chat/completions"
                system_prompt = (
                    "You are a Sourcegraph query generator. Convert natural language "
                    "search requests into Sourcegraph query syntax. Return ONLY the "
                    "Sourcegraph query, no explanation."
                )
                user_content = f"Query: {example.input}"

                # Build request body with Qwen3 settings
                # enable_thinking=False disables Qwen3 thinking mode for lower latency
                request_body = {
                    "model": "llm",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 64,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                }

                response = client.post(url, json=request_body)
                response.raise_for_status()
                result = response.json()
                output = result["choices"][0]["message"]["content"]
                # Strip Qwen3 <think> tags if present
                output = re.sub(r"<think>.*?</think>\s*", "", output, flags=re.DOTALL)
                output = output.strip()
                # Capture server-side token usage
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
            else:
                # Legacy transformers endpoint format
                response = client.post(
                    model_endpoint,
                    json={
                        "query": example.input,
                    },
                )
                response.raise_for_status()
                result = response.json()
                output = result.get("sourcegraph_query", "")
        except Exception as e:
            output = f"ERROR: {e}"

        latency_ms = (time.time() - start_time) * 1000

        # Evaluate using Sourcegraph's query parser
        eval_result = evaluate_query(example.expected, output)

        return EvalResult(
            id=example.id,
            input=example.input,
            expected=example.expected,
            output=output,
            syntax_valid=eval_result.valid,
            semantic_match=eval_result.match,
            overlap=eval_result.overlap,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def run_evaluation(
        self,
        model_endpoint: str,
        sample_size: int | None = None,
        baseline_results: dict | None = None,
    ) -> dict[str, Any]:
        """Run full evaluation on validation set.

        Args:
            model_endpoint: URL of the model inference endpoint
            sample_size: Number of examples to evaluate (None = all)
            baseline_results: Optional baseline results for comparison

        Returns:
            Evaluation results dict with summary and per-example results
        """
        examples = self.load_examples()
        if sample_size:
            examples = examples[:sample_size]

        results = []
        total_syntax_valid = 0
        total_semantic_match = 0
        total_latency = 0.0

        for example in examples:
            result = self.evaluate_example(example, model_endpoint)
            results.append(result)

            if result.syntax_valid:
                total_syntax_valid += 1
            if result.semantic_match:
                total_semantic_match += 1
            total_latency += result.latency_ms

        n = len(results)
        eval_id = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Build results dict
        eval_results = {
            "id": eval_id,
            "timestamp": datetime.now().isoformat(),
            "models": {
                "fine_tuned": model_endpoint.split("/")[-1]
                if "/" in model_endpoint
                else "fine-tuned",
            },
            "summary": {
                "total": n,
                "fine_tuned": {
                    "syntax_valid": total_syntax_valid,
                    "semantic_match": total_semantic_match,
                    "avg_latency_ms": round(total_latency / n, 1) if n > 0 else 0,
                },
            },
            "results": [
                {
                    "id": r.id,
                    "input": r.input,
                    "expected": r.expected,
                    "fine_tuned": {
                        "output": r.output,
                        "syntax_valid": r.syntax_valid,
                        "semantic_match": r.semantic_match,
                        "overlap": r.overlap,
                        "latency_ms": round(r.latency_ms, 1),
                    },
                    "verdict": self._compute_verdict(
                        r, baseline_results, r.id if baseline_results else None
                    ),
                }
                for r in results
            ],
        }

        # Add baseline to models if provided
        if baseline_results:
            eval_results["models"]["baseline"] = baseline_results.get("model", "gpt-4o-mini")
            # Add baseline summary
            baseline_summary = baseline_results.get("summary", {})
            eval_results["summary"]["baseline"] = baseline_summary
            # Add baseline outputs to results
            baseline_by_id = {r["id"]: r for r in baseline_results.get("results", [])}
            for result in eval_results["results"]:
                baseline = baseline_by_id.get(result["id"], {})
                result["baseline"] = baseline.get("baseline", baseline)

        return eval_results

    def _compute_verdict(
        self,
        result: EvalResult,
        baseline_results: dict | None,
        example_id: str | None,
    ) -> str:
        """Compute verdict comparing fine-tuned to baseline.

        Returns one of: fine_tuned_better, baseline_better, tie, both_wrong
        """
        ft_good = result.syntax_valid and result.semantic_match

        if not baseline_results or not example_id:
            return "tie" if ft_good else "both_wrong"

        # Find baseline result for this example
        baseline_by_id = {r["id"]: r for r in baseline_results.get("results", [])}
        baseline = baseline_by_id.get(example_id, {})
        baseline_valid = baseline.get("syntax_valid", True)
        baseline_semantic = baseline.get("semantic_match", True)
        baseline_good = baseline_valid and baseline_semantic

        if ft_good and baseline_good:
            return "tie"
        elif ft_good and not baseline_good:
            return "fine_tuned_better"
        elif not ft_good and baseline_good:
            return "baseline_better"
        else:
            return "both_wrong"
