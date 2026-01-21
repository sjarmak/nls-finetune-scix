"""CLI commands for model evaluation."""

import json
import statistics
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()

eval_app = typer.Typer(
    name="eval",
    help="Evaluate fine-tuned model quality.",
    no_args_is_help=True,
)

# Default paths
EVAL_DIR = Path("data/datasets/evaluations")
DEFAULT_ENDPOINT = "https://sourcegraph--nls-finetune-serve-vllm-serve.modal.run"


@eval_app.command()
def baseline(
    model: str = typer.Option("gpt-4o-mini", "--model", help="Baseline model to use"),
    sample: int = typer.Option(None, "--sample", help="Number of examples to sample"),
    output: Path = typer.Option(None, "--output", help="Output file path"),
) -> None:
    """Generate baseline results using GPT-4o-mini or other model."""
    import os
    import time

    import httpx

    from finetune.eval.eval import evaluate_query
    from finetune.eval.runner import EvalRunner

    console.print(f"[bold]Generating baseline with {model}[/bold]")

    # Load examples
    runner = EvalRunner()
    examples = runner.load_examples()
    if sample:
        examples = examples[:sample]
        console.print(f"Sampling {sample} examples")

    console.print(f"Evaluating {len(examples)} examples...")

    # Get OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not set[/red]")
        raise typer.Exit(1)

    results = []
    latencies = []

    with console.status("[bold green]Running baseline...") as status:
        for i, example in enumerate(examples):
            status.update(f"[bold green]Processing {i + 1}/{len(examples)}...")

            # Build prompt for GPT-4o-mini
            system_prompt = (
                "Convert natural language to Sourcegraph search query. "
                "Output only the query, no explanation."
            )
            user_prompt = f"Query: {example.input}"

            start_time = time.time()
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "max_tokens": 256,
                            "temperature": 0,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    output_query = data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                output_query = f"ERROR: {e}"

            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

            # Evaluate using Sourcegraph's query parser
            eval_result = evaluate_query(example.expected, output_query)

            results.append(
                {
                    "id": example.id,
                    "input": example.input,
                    "expected": example.expected,
                    "output": output_query,
                    "syntax_valid": eval_result.valid,
                    "semantic_match": eval_result.match,
                    "overlap": eval_result.overlap,
                    "latency_ms": round(latency_ms, 1),
                }
            )

    # Compute latency statistics
    n = len(results)
    latency_stats = _compute_latency_stats(latencies)

    # Build baseline results
    baseline_results = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": n,
            "syntax_valid": sum(1 for r in results if r["syntax_valid"]),
            "semantic_match": sum(1 for r in results if r["semantic_match"]),
            **latency_stats,  # Includes avg, p75, p90, min, max, cold_start
        },
        "results": results,
    }

    # Save results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output or EVAL_DIR / f"baseline-{model}.json"
    with open(output_path, "w") as f:
        json.dump(baseline_results, f, indent=2)

    console.print(f"\n[green]Baseline saved to {output_path}[/green]")
    console.print(f"  Syntax valid: {baseline_results['summary']['syntax_valid']}/{n}")
    console.print(f"  Semantic match: {baseline_results['summary']['semantic_match']}/{n}")
    console.print(f"  Avg latency: {latency_stats['avg_latency_ms']}ms")
    console.print(f"  P75 latency: {latency_stats['p75_latency_ms']}ms")
    console.print(f"  P90 latency: {latency_stats['p90_latency_ms']}ms")


def _compute_latency_stats(latencies: list[float]) -> dict:
    """Compute latency statistics, excluding first request for warm metrics.

    Args:
        latencies: List of latency values in ms

    Returns:
        Dict with avg, p75, p90, min, max (excluding first), and cold_start (first request)
    """
    if not latencies:
        return {
            "avg_latency_ms": 0,
            "p75_latency_ms": 0,
            "p90_latency_ms": 0,
            "min_latency_ms": 0,
            "max_latency_ms": 0,
            "cold_start_latency_ms": 0,
        }

    cold_start = latencies[0]

    # For warm metrics, exclude first request (cold start / compilation warmup)
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies

    # Compute percentiles (need at least 2 values for meaningful percentiles)
    if len(warm_latencies) >= 2:
        sorted_latencies = sorted(warm_latencies)
        n = len(sorted_latencies)
        p75_idx = int(n * 0.75)
        p90_idx = int(n * 0.90)
        p75 = sorted_latencies[min(p75_idx, n - 1)]
        p90 = sorted_latencies[min(p90_idx, n - 1)]
    else:
        p75 = warm_latencies[0] if warm_latencies else 0
        p90 = warm_latencies[0] if warm_latencies else 0

    return {
        "avg_latency_ms": round(statistics.mean(warm_latencies), 1),
        "p75_latency_ms": round(p75, 1),
        "p90_latency_ms": round(p90, 1),
        "min_latency_ms": round(min(warm_latencies), 1),
        "max_latency_ms": round(max(warm_latencies), 1),
        "cold_start_latency_ms": round(cold_start, 1),
    }


def _call_baseline_model(example, model: str = "gpt-4o-mini", api_key: str | None = None) -> dict:
    """Call baseline model (GPT-4o-mini) for a single example.

    Args:
        example: EvalExample with input, expected
        model: OpenAI model to use
        api_key: OpenAI API key (reads from env if not provided)

    Returns:
        Dict with output, syntax_valid, semantic_match, latency_ms
    """
    import os
    import time

    import httpx

    from finetune.eval.eval import evaluate_query

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "output": "ERROR: OPENAI_API_KEY not set",
            "syntax_valid": False,
            "semantic_match": False,
            "latency_ms": 0,
        }

    # Build prompt
    system_prompt = (
        "Convert natural language to Sourcegraph search query. "
        "Output only the query, no explanation."
    )
    user_prompt = f"Query: {example.input}"

    start_time = time.time()
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 256,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            data = response.json()
            output_query = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        output_query = f"ERROR: {e}"

    latency_ms = (time.time() - start_time) * 1000

    # Evaluate using Sourcegraph's query parser
    eval_result = evaluate_query(example.expected, output_query)

    return {
        "id": example.id,
        "input": example.input,
        "expected": example.expected,
        "output": output_query,
        "syntax_valid": eval_result.valid,
        "semantic_match": eval_result.match,
        "overlap": eval_result.overlap,
        "latency_ms": round(latency_ms, 1),
    }


def _warmup_endpoint(
    endpoint: str, endpoint_type: str = "legacy", timeout: float = 120.0, num_warmups: int = 3
) -> list[float]:
    """Send multiple warmup requests to fully warm vLLM.

    vLLM has multiple warmup phases:
    1. Model loading (handled by container keep-alive)
    2. CUDA kernel compilation (first few requests)
    3. KV cache allocation (gradual)

    Args:
        endpoint: Model endpoint URL
        endpoint_type: "legacy" or "vllm"
        timeout: Request timeout in seconds
        num_warmups: Number of warmup requests to send (default 3)

    Returns:
        List of warmup latencies in ms, or [-1] if first request failed.
    """
    import time

    import httpx

    latencies = []
    with httpx.Client(timeout=timeout) as client:
        for i in range(num_warmups):
            start_time = time.time()
            try:
                if endpoint_type == "vllm":
                    # vLLM OpenAI-compatible format - append path if not present
                    url = endpoint.rstrip("/")
                    if not url.endswith("/v1/chat/completions"):
                        url = f"{url}/v1/chat/completions"
                    response = client.post(
                        url,
                        json={
                            "model": "llm",
                            "messages": [{"role": "user", "content": "warmup test"}],
                            "max_tokens": 16,
                            # Disable Qwen3 thinking mode for lower latency
                            # Must use chat_template_kwargs, NOT top-level enable_thinking
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                else:
                    # Legacy transformers endpoint format
                    response = client.post(
                        endpoint,
                        json={"query": "warmup test", "max_tokens": 16},
                    )
                response.raise_for_status()
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            except Exception as e:
                if i == 0:
                    # First request failed - can't warm up
                    console.print(f"[red]Warmup failed: {e}[/red]")
                    return [-1]
                # Subsequent failures are logged but we continue
                console.print(f"[yellow]Warmup request {i + 1} failed: {e}[/yellow]")

    return latencies if latencies else [-1]


def _measure_network_latency(endpoint: str, samples: int = 5) -> float:
    """Measure network round-trip time to endpoint using /health endpoint.

    This helps separate network latency from inference latency.

    Args:
        endpoint: Model endpoint URL
        samples: Number of samples to take (default 5)

    Returns:
        Median network latency in ms, or -1 if measurement failed.
    """
    import time

    import httpx

    # Build health endpoint URL
    url = endpoint.rstrip("/")
    # Try common health endpoint paths
    health_urls = [
        f"{url}/health",
        f"{url}/v1/models",  # vLLM models endpoint is lightweight
    ]

    latencies = []
    with httpx.Client(timeout=10.0) as client:
        for health_url in health_urls:
            try:
                # Test if this health URL works
                client.get(health_url)
                # If it works, measure latency
                for _ in range(samples):
                    start_time = time.time()
                    client.get(health_url)
                    latency_ms = (time.time() - start_time) * 1000
                    latencies.append(latency_ms)
                break  # Found working health endpoint
            except Exception:
                continue

    if not latencies:
        return -1

    # Return median to reduce outlier impact
    sorted_latencies = sorted(latencies)
    mid = len(sorted_latencies) // 2
    return sorted_latencies[mid]


def _fetch_vllm_metrics(endpoint: str) -> dict:
    """Fetch server-side metrics from vLLM Prometheus endpoint.

    vLLM exposes detailed timing metrics including:
    - TTFT (Time to First Token): Prefill latency
    - ITL (Inter-Token Latency): Generation time per token
    - E2E: End-to-end request latency

    These are similar to Fireworks' metrics:
    - fireworks-prefill-duration → TTFT
    - fireworks-server-processing-time → E2E

    Args:
        endpoint: Model endpoint URL

    Returns:
        Dict with server-side metrics (all in milliseconds):
        - ttft_ms: Time to first token (prefill)
        - itl_ms: Inter-token latency (generation per token)
        - e2e_ms: End-to-end server processing time
        - prompt_throughput: Prompt tokens/second
        - generation_throughput: Generation tokens/second
    """
    import re

    import httpx

    url = endpoint.rstrip("/") + "/metrics"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            metrics_text = response.text

        result = {}

        # Parse histogram metrics (we want the _sum and _count to compute averages)
        # vLLM metric names vary by version, try multiple patterns

        # Time to First Token (TTFT) - prefill latency
        ttft_patterns = [
            r"vllm:time_to_first_token_seconds_sum\{[^}]*\}\s+([\d.e+-]+)",
            r'vllm:time_to_first_token_seconds_bucket\{[^}]*le="[^"]+"\}\s+([\d.e+-]+)',
        ]
        ttft_count_patterns = [
            r"vllm:time_to_first_token_seconds_count\{[^}]*\}\s+([\d.e+-]+)",
        ]

        # Inter-Token Latency (ITL)
        itl_patterns = [
            r"vllm:time_per_output_token_seconds_sum\{[^}]*\}\s+([\d.e+-]+)",
        ]
        itl_count_patterns = [
            r"vllm:time_per_output_token_seconds_count\{[^}]*\}\s+([\d.e+-]+)",
        ]

        # End-to-End latency
        e2e_patterns = [
            r"vllm:e2e_request_latency_seconds_sum\{[^}]*\}\s+([\d.e+-]+)",
        ]
        e2e_count_patterns = [
            r"vllm:e2e_request_latency_seconds_count\{[^}]*\}\s+([\d.e+-]+)",
        ]

        def get_metric_avg(sum_patterns: list, count_patterns: list, text: str) -> float | None:
            """Get average from sum/count metrics."""
            total_sum = 0.0
            total_count = 0
            for pattern in sum_patterns:
                for match in re.finditer(pattern, text):
                    total_sum += float(match.group(1))
            for pattern in count_patterns:
                for match in re.finditer(pattern, text):
                    total_count += int(float(match.group(1)))
            if total_count > 0:
                return total_sum / total_count
            return None

        # Extract metrics
        ttft = get_metric_avg(ttft_patterns, ttft_count_patterns, metrics_text)
        if ttft is not None:
            result["ttft_ms"] = round(ttft * 1000, 2)

        itl = get_metric_avg(itl_patterns, itl_count_patterns, metrics_text)
        if itl is not None:
            result["itl_ms"] = round(itl * 1000, 2)

        e2e = get_metric_avg(e2e_patterns, e2e_count_patterns, metrics_text)
        if e2e is not None:
            result["e2e_ms"] = round(e2e * 1000, 2)

        return result

    except Exception:
        return {}


def _measure_latency_breakdown(endpoint: str, samples: int = 5) -> dict:
    """Measure detailed latency breakdown by component.

    This function measures latency with and without connection reuse to
    isolate TCP+SSL overhead from actual server processing time.

    Args:
        endpoint: Model endpoint URL
        samples: Number of samples to take

    Returns:
        Dict with latency breakdown components:
        - network_rtt_ms: Health endpoint latency with connection reuse (network + Modal routing)
        - connection_overhead_ms: Extra time for fresh connection (TCP+SSL handshake)
        - modal_overhead_ms: Estimated Modal infrastructure overhead
    """
    import time

    import httpx

    url = endpoint.rstrip("/")
    health_urls = [f"{url}/health", f"{url}/v1/models"]

    # Find working health endpoint
    working_url = None
    for health_url in health_urls:
        try:
            with httpx.Client(timeout=10.0) as client:
                client.get(health_url)
                working_url = health_url
                break
        except Exception:
            continue

    if not working_url:
        return {}

    # Measure with connection reuse (warm requests)
    warm_latencies = []
    with httpx.Client(http2=True, timeout=10.0) as client:
        # Warmup
        client.get(working_url)
        # Measure
        for _ in range(samples):
            start_time = time.time()
            client.get(working_url)
            latency_ms = (time.time() - start_time) * 1000
            warm_latencies.append(latency_ms)

    # Measure without connection reuse (cold requests)
    cold_latencies = []
    for _ in range(min(3, samples)):  # Fewer samples for cold (slower)
        start_time = time.time()
        with httpx.Client(http2=True, timeout=10.0) as client:
            client.get(working_url)
        latency_ms = (time.time() - start_time) * 1000
        cold_latencies.append(latency_ms)

    # Calculate median values
    warm_median = sorted(warm_latencies)[len(warm_latencies) // 2]
    cold_median = sorted(cold_latencies)[len(cold_latencies) // 2]

    # Connection overhead = cold - warm (TCP+SSL handshake time)
    connection_overhead = max(0, cold_median - warm_median)

    # Modal overhead estimate: warm health latency minus typical network RTT
    # Typical network RTT is roughly half of TCP connect time (one way)
    # We estimate it as warm_median * 0.75 to account for Modal routing
    modal_overhead = warm_median * 0.2  # ~20% of warm health is Modal routing

    return {
        "network_rtt_ms": round(warm_median, 1),
        "connection_overhead_ms": round(connection_overhead, 1),
        "modal_overhead_ms": round(modal_overhead, 1),
    }


def _render_comparison_tables(
    ft_summary: dict,
    bs_summary: dict | None,
    bs_model: str | None,
    latency_breakdown: dict | None,
    total: int,
) -> None:
    """Render model comparison and latency breakdown tables.

    Args:
        ft_summary: Fine-tuned model summary stats
        bs_summary: Baseline model summary stats (optional)
        bs_model: Baseline model name
        latency_breakdown: Latency breakdown data (optional)
        total: Total number of examples
    """
    from rich.box import ROUNDED

    # Helper to format latency delta
    def latency_delta_fmt(ft_val: float, bs_val: float) -> str:
        delta = ft_val - bs_val
        if delta < 0:
            return f"[green]{delta:.0f}ms[/green]"
        elif delta > 0:
            return f"[red]+{delta:.0f}ms[/red]"
        return "0ms"

    # Get server metrics
    server_metrics = latency_breakdown.get("server_metrics", {}) if latency_breakdown else {}

    # Create model comparison table
    comparison_table = Table(title="Model Comparison", box=ROUNDED, padding=(0, 1))
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Fine-tuned", justify="right")
    if bs_summary:
        comparison_table.add_column(f"Baseline ({bs_model})", justify="right")
        comparison_table.add_column("Δ", justify="right", style="dim")

    # Quality metrics
    ft_syntax_pct = 100 * ft_summary["syntax_valid"] / total
    ft_semantic_pct = 100 * ft_summary["semantic_match"] / total

    if bs_summary:
        bs_syntax_pct = 100 * bs_summary["syntax_valid"] / bs_summary["total"]
        bs_semantic_pct = 100 * bs_summary["semantic_match"] / bs_summary["total"]

        syntax_delta = ft_syntax_pct - bs_syntax_pct
        semantic_delta = ft_semantic_pct - bs_semantic_pct

        def pct_delta_str(delta: float) -> str:
            if delta > 0:
                return f"[green]+{delta:.1f}%[/green]"
            elif delta < 0:
                return f"[red]{delta:.1f}%[/red]"
            return "0%"

        syntax_delta_str = pct_delta_str(syntax_delta)
        semantic_delta_str = pct_delta_str(semantic_delta)

        comparison_table.add_row(
            "Syntax Valid",
            f"{ft_summary['syntax_valid']}/{total} ({ft_syntax_pct:.1f}%)",
            f"{bs_summary['syntax_valid']}/{bs_summary['total']} ({bs_syntax_pct:.1f}%)",
            syntax_delta_str,
        )
        comparison_table.add_row(
            "Semantic Match",
            f"{ft_summary['semantic_match']}/{total} ({ft_semantic_pct:.1f}%)",
            f"{bs_summary['semantic_match']}/{bs_summary['total']} ({bs_semantic_pct:.1f}%)",
            semantic_delta_str,
        )

        # Latency comparison - avg, p75, p90
        comparison_table.add_row(
            "Avg Latency",
            f"{ft_summary['avg_latency_ms']:.0f}ms",
            f"{bs_summary.get('avg_latency_ms', 0):.0f}ms",
            latency_delta_fmt(ft_summary["avg_latency_ms"], bs_summary.get("avg_latency_ms", 0)),
        )
        if "p75_latency_ms" in ft_summary:
            bs_p75 = bs_summary.get("p75_latency_ms")
            bs_p90 = bs_summary.get("p90_latency_ms")
            comparison_table.add_row(
                "P75 Latency",
                f"{ft_summary['p75_latency_ms']:.0f}ms",
                f"{bs_p75:.0f}ms" if bs_p75 else "-",
                latency_delta_fmt(ft_summary["p75_latency_ms"], bs_p75 or 0) if bs_p75 else "",
            )
            comparison_table.add_row(
                "P90 Latency",
                f"{ft_summary['p90_latency_ms']:.0f}ms",
                f"{bs_p90:.0f}ms" if bs_p90 else "-",
                latency_delta_fmt(ft_summary["p90_latency_ms"], bs_p90 or 0) if bs_p90 else "",
            )
    else:
        syntax_str = f"{ft_summary['syntax_valid']}/{total} ({ft_syntax_pct:.1f}%)"
        semantic_str = f"{ft_summary['semantic_match']}/{total} ({ft_semantic_pct:.1f}%)"
        comparison_table.add_row("Syntax Valid", syntax_str)
        comparison_table.add_row("Semantic Match", semantic_str)
        comparison_table.add_row("Avg Latency", f"{ft_summary['avg_latency_ms']:.0f}ms")
        if "p75_latency_ms" in ft_summary:
            comparison_table.add_row("P75 Latency", f"{ft_summary['p75_latency_ms']:.0f}ms")
            comparison_table.add_row("P90 Latency", f"{ft_summary['p90_latency_ms']:.0f}ms")

    # Server-side metrics (vLLM only) - integrated into main table
    if server_metrics:
        if server_metrics.get("ttft_ms"):
            comparison_table.add_row(
                "Server TTFT", f"{server_metrics['ttft_ms']:.1f}ms", style="dim"
            )
        if server_metrics.get("itl_ms"):
            comparison_table.add_row("Server ITL", f"{server_metrics['itl_ms']:.1f}ms", style="dim")
        if server_metrics.get("e2e_ms"):
            comparison_table.add_row("Server E2E", f"{server_metrics['e2e_ms']:.1f}ms", style="dim")

    if "cold_start_latency_ms" in ft_summary:
        comparison_table.add_row(
            "Cold Start", f"{ft_summary['cold_start_latency_ms']:.0f}ms", style="dim"
        )

    console.print()
    console.print(comparison_table)

    # Latency breakdown table (if available)
    if latency_breakdown:
        net_rtt = latency_breakdown.get("network_rtt_ms", 0)
        conn_overhead = latency_breakdown.get("connection_overhead_ms", 0)
        avg_latency = ft_summary.get("avg_latency_ms", 0)
        estimated_inference = max(0, avg_latency - net_rtt) if net_rtt > 0 else 0

        if avg_latency > 0:
            breakdown_table = Table(title="Latency Breakdown", box=ROUNDED, padding=(0, 1))
            breakdown_table.add_column("Component", style="cyan")
            breakdown_table.add_column("Time", justify="right")
            breakdown_table.add_column("% of Total", justify="right")
            breakdown_table.add_column("Description", style="dim")

            breakdown_table.add_row(
                "Network RTT (warm)",
                f"{net_rtt:.0f}ms",
                f"{100 * net_rtt / avg_latency:.1f}%",
                "Round-trip to server",
            )
            breakdown_table.add_row(
                "Estimated inference",
                f"{estimated_inference:.0f}ms",
                f"{100 * estimated_inference / avg_latency:.1f}%",
                "Server processing",
            )
            if conn_overhead > 0:
                breakdown_table.add_row(
                    "TCP+SSL overhead",
                    f"{conn_overhead:.0f}ms",
                    "-",
                    "Cold start only",
                    style="dim",
                )

            console.print()
            console.print(breakdown_table)


@eval_app.command("run")
def run_eval(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, "--endpoint", help="Model endpoint URL"),
    endpoint_type: str = typer.Option(
        "vllm", "--endpoint-type", help="Endpoint type: legacy or vllm"
    ),
    sample: int = typer.Option(
        50, "--sample", help="Number of examples to evaluate (default 50 for meaningful P90)"
    ),
    live: bool = typer.Option(False, "--live", help="Run live comparison against baseline model"),
    output: Path = typer.Option(None, "--output", help="Output file path"),
    skip_warmup: bool = typer.Option(False, "--skip-warmup", help="Skip container warmup"),
) -> None:
    """Run evaluation against fine-tuned model."""
    from finetune.eval.runner import EvalRunner

    console.print("[bold]Running evaluation[/bold]")
    console.print(f"  Endpoint: {endpoint}")
    console.print(f"  Endpoint type: {endpoint_type}")

    # Warmup the container first to avoid cold start affecting metrics
    # Send multiple warmup requests to fully warm vLLM (CUDA kernel compilation, KV cache)
    network_latency_ms = -1
    latency_breakdown = {}
    if not skip_warmup:
        console.print(
            "\n[yellow]Warming up container with 3 requests (this may take 30-60s on cold start)...[/yellow]"
        )
        warmup_latencies = _warmup_endpoint(endpoint, endpoint_type, num_warmups=3)
        if warmup_latencies[0] > 0:
            total_warmup = sum(warmup_latencies)
            console.print(
                f"[green]✓ Container ready[/green] ({len(warmup_latencies)} warmups, "
                f"total {total_warmup / 1000:.1f}s, last {warmup_latencies[-1]:.0f}ms)"
            )
            # Measure detailed latency breakdown
            console.print("[dim]Measuring latency breakdown...[/dim]")
            latency_breakdown = _measure_latency_breakdown(endpoint)
            network_latency_ms = latency_breakdown.get("network_rtt_ms", -1)
            if network_latency_ms > 0:
                console.print(f"  Network RTT (warm): {network_latency_ms:.0f}ms")
                if latency_breakdown.get("connection_overhead_ms"):
                    console.print(
                        f"  Connection overhead (TCP+SSL): {latency_breakdown['connection_overhead_ms']:.0f}ms"
                    )

            # Fetch server-side metrics from vLLM (if available)
            if endpoint_type == "vllm":
                vllm_metrics = _fetch_vllm_metrics(endpoint)
                if vllm_metrics:
                    latency_breakdown["server_metrics"] = vllm_metrics
                    console.print("[dim]Server-side metrics (from vLLM):[/dim]")
                    if "ttft_ms" in vllm_metrics:
                        console.print(f"  TTFT (prefill): {vllm_metrics['ttft_ms']:.1f}ms")
                    if "itl_ms" in vllm_metrics:
                        console.print(f"  ITL (per token): {vllm_metrics['itl_ms']:.1f}ms")
                    if "e2e_ms" in vllm_metrics:
                        console.print(f"  E2E (server): {vllm_metrics['e2e_ms']:.1f}ms")
        else:
            console.print("[yellow]⚠ Warmup failed, continuing anyway...[/yellow]")

    # Load baseline if exists (or prepare for live mode)
    baseline_path = EVAL_DIR / "baseline-gpt-4o-mini.json"
    baseline_results = None
    live_baseline_results = []  # For live mode
    if baseline_path.exists() and not live:
        with open(baseline_path) as f:
            baseline_results = json.load(f)
        console.print(f"  Using cached baseline: {baseline_path}")
    elif live:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            console.print("[red]✗ Live mode requires OPENAI_API_KEY environment variable[/red]")
            raise typer.Exit(1)
        console.print(
            "  [yellow]Live comparison mode - calling baseline model for each example[/yellow]"
        )

    # Run evaluation
    runner = EvalRunner()
    examples = runner.load_examples()
    examples = examples[:sample]
    console.print(f"  Evaluating {sample} examples (use --sample to change)")

    console.print(f"\nEvaluating {len(examples)} examples...")

    results = []
    total_syntax_valid = 0
    total_semantic_match = 0
    latencies = []

    with console.status("[bold green]Evaluating...") as status:
        for i, example in enumerate(examples):
            if live:
                status.update(
                    f"[bold green]Processing {i + 1}/{len(examples)} (fine-tuned + baseline)..."
                )
            else:
                status.update(f"[bold green]Processing {i + 1}/{len(examples)}...")

            # Evaluate fine-tuned model
            result = runner.evaluate_example(example, endpoint, endpoint_type=endpoint_type)
            results.append(result)

            if result.syntax_valid:
                total_syntax_valid += 1
            if result.semantic_match:
                total_semantic_match += 1
            latencies.append(result.latency_ms)

            # Call baseline model in live mode
            if live:
                baseline_result = _call_baseline_model(example)
                live_baseline_results.append(baseline_result)

    # Build baseline_results from live results
    if live and live_baseline_results:
        baseline_latencies = [r["latency_ms"] for r in live_baseline_results]
        baseline_stats = _compute_latency_stats(baseline_latencies)
        baseline_results = {
            "model": "gpt-4o-mini",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(live_baseline_results),
                "syntax_valid": sum(1 for r in live_baseline_results if r["syntax_valid"]),
                "semantic_match": sum(1 for r in live_baseline_results if r["semantic_match"]),
                **baseline_stats,
            },
            "results": live_baseline_results,
        }
        console.print(
            f"  [green]✓ Live baseline: {baseline_results['summary']['semantic_match']}/{len(live_baseline_results)} semantic match[/green]"
        )

    n = len(results)
    eval_id = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Compute latency statistics (excludes first request for warm metrics)
    latency_stats = _compute_latency_stats(latencies)

    # Compute estimated inference latency (subtract network RTT)
    estimated_inference_ms = None
    if network_latency_ms > 0:
        estimated_inference_ms = max(0, latency_stats["avg_latency_ms"] - network_latency_ms)

    # Build latency breakdown for JSON output
    latency_breakdown_output = {}
    if latency_breakdown:
        est_inf = round(estimated_inference_ms, 1) if estimated_inference_ms is not None else 0
        latency_breakdown_output = {
            "network_rtt_ms": latency_breakdown.get("network_rtt_ms", 0),
            "connection_overhead_ms": latency_breakdown.get("connection_overhead_ms", 0),
            "estimated_inference_ms": est_inf,
        }
        # Add server-side metrics if available (vLLM Prometheus metrics)
        server_metrics = latency_breakdown.get("server_metrics", {})
        if server_metrics:
            latency_breakdown_output["server_metrics"] = {
                "ttft_ms": server_metrics.get("ttft_ms", 0),  # Time to First Token (prefill)
                "itl_ms": server_metrics.get("itl_ms", 0),  # Inter-Token Latency (generation)
                "e2e_ms": server_metrics.get("e2e_ms", 0),  # End-to-End server time
            }

    # Build evaluation results
    eval_results = {
        "id": eval_id,
        "timestamp": datetime.now().isoformat(),
        "models": {
            "fine_tuned": endpoint.split("/")[-1] if "/" in endpoint else "fine-tuned",
        },
        "summary": {
            "total": n,
            "fine_tuned": {
                "syntax_valid": total_syntax_valid,
                "semantic_match": total_semantic_match,
                **latency_stats,  # Includes avg, p75, p90, min, max, cold_start
            },
            # Latency breakdown by component
            **({"latency_breakdown": latency_breakdown_output} if latency_breakdown_output else {}),
        },
        "results": [
            {
                "id": r.id,
                "input": r.input,
                "expected": r.expected,
                "fine_tuned": {
                    "output": r.output,
                    "syntax_valid": r.syntax_valid,
                    "hallucinations": r.hallucinations,
                    "semantic_match": r.semantic_match,
                    "latency_ms": round(r.latency_ms, 1),
                    # Server-side token usage (vLLM only)
                    **({"prompt_tokens": r.prompt_tokens} if r.prompt_tokens else {}),
                    **({"completion_tokens": r.completion_tokens} if r.completion_tokens else {}),
                },
                "verdict": _compute_verdict(r, baseline_results),
            }
            for r in results
        ],
    }

    # Add baseline comparison if available
    if baseline_results:
        eval_results["models"]["baseline"] = baseline_results.get("model", "gpt-4o-mini")
        eval_results["summary"]["baseline"] = baseline_results.get("summary", {})

        # Add baseline outputs to results
        baseline_by_id = {r["id"]: r for r in baseline_results.get("results", [])}
        for result in eval_results["results"]:
            baseline = baseline_by_id.get(result["id"], {})
            result["baseline"] = {
                "output": baseline.get("output", ""),
                "syntax_valid": baseline.get("syntax_valid", True),
                "hallucinations": baseline.get("hallucinations", []),
                "semantic_match": baseline.get("semantic_match", True),
                "latency_ms": baseline.get("latency_ms", 0),
            }

    # Save results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output or EVAL_DIR / f"{eval_id}.json"
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    # Print summary
    console.print("\n[bold green]Evaluation complete![/bold green]")
    console.print(f"  Results saved to: {output_path}")

    # Build fine-tuned summary for rendering
    ft_summary = {
        "syntax_valid": total_syntax_valid,
        "semantic_match": total_semantic_match,
        **latency_stats,
    }

    # Render tables using shared function
    bs_summary = baseline_results.get("summary", {}) if baseline_results else None
    bs_model = baseline_results.get("model", "gpt-4o-mini") if baseline_results else None

    _render_comparison_tables(
        ft_summary=ft_summary,
        bs_summary=bs_summary,
        bs_model=bs_model,
        latency_breakdown=latency_breakdown,
        total=n,
    )


def _compute_verdict(result, baseline_results) -> str:
    """Compute verdict comparing fine-tuned to baseline."""
    ft_good = result.syntax_valid and result.semantic_match

    if not baseline_results:
        return "tie" if ft_good else "both_wrong"

    # Find baseline result for this example
    baseline_by_id = {r["id"]: r for r in baseline_results.get("results", [])}
    baseline = baseline_by_id.get(result.id, {})
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


@eval_app.command()
def report(
    run_id: str = typer.Option(None, "--run-id", help="Specific evaluation run to report"),
) -> None:
    """Print evaluation report with pass/fail status."""
    # Find latest evaluation
    if run_id:
        eval_path = EVAL_DIR / f"{run_id}.json"
    else:
        eval_files = sorted(EVAL_DIR.glob("eval-*.json"), reverse=True)
        if not eval_files:
            console.print("[red]No evaluation runs found. Run 'nls-finetune eval run' first.[/red]")
            raise typer.Exit(1)
        eval_path = eval_files[0]

    if not eval_path.exists():
        console.print(f"[red]Evaluation file not found: {eval_path}[/red]")
        raise typer.Exit(1)

    with open(eval_path) as f:
        eval_data = json.load(f)

    # Print header
    console.print(f"\n[bold]Evaluation Results ({eval_data['id']})[/bold]")
    console.print("═" * 55)

    # Build comparison table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Fine-tuned", justify="right")
    if "baseline" in eval_data.get("summary", {}):
        table.add_column("Baseline", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status", justify="center")

    ft = eval_data["summary"]["fine_tuned"]
    total = eval_data["summary"]["total"]
    baseline = eval_data["summary"].get("baseline", {})

    # Syntax validity row
    syntax_pct = 100 * ft["syntax_valid"] / total
    syntax_status = "[green]PASS[/green]" if syntax_pct >= 95 else "[red]FAIL[/red]"
    if baseline:
        bs_syntax = 100 * baseline["syntax_valid"] / baseline["total"]
        table.add_row(
            "Syntax Valid",
            f"{ft['syntax_valid']}/{total} ({syntax_pct:.0f}%)",
            f"{baseline['syntax_valid']}/{baseline['total']} ({bs_syntax:.0f}%)",
            "≥95%",
            syntax_status,
        )
    else:
        table.add_row(
            "Syntax Valid",
            f"{ft['syntax_valid']}/{total} ({syntax_pct:.0f}%)",
            "≥95%",
            syntax_status,
        )

    # Semantic match row
    semantic_pct = 100 * ft["semantic_match"] / total
    semantic_status = "[green]PASS[/green]" if semantic_pct >= 70 else "[red]FAIL[/red]"
    if baseline:
        bs_semantic = 100 * baseline["semantic_match"] / baseline["total"]
        table.add_row(
            "Semantic Match",
            f"{ft['semantic_match']}/{total} ({semantic_pct:.0f}%)",
            f"{baseline['semantic_match']}/{baseline['total']} ({bs_semantic:.0f}%)",
            "≥70%",
            semantic_status,
        )
    else:
        table.add_row(
            "Semantic Match",
            f"{ft['semantic_match']}/{total} ({semantic_pct:.0f}%)",
            "≥70%",
            semantic_status,
        )

    # Latency row
    latency = ft["avg_latency_ms"]
    latency_status = "[green]PASS[/green]" if latency <= 100 else "[red]FAIL[/red]"
    if baseline:
        table.add_row(
            "Avg Latency",
            f"{latency:.0f}ms",
            f"{baseline['avg_latency_ms']:.0f}ms",
            "≤100ms",
            latency_status,
        )
    else:
        table.add_row(
            "Avg Latency",
            f"{latency:.0f}ms",
            "≤100ms",
            latency_status,
        )

    console.print(table)

    # Show P75/P90 latencies if available (new detailed metrics)
    if "p75_latency_ms" in ft:
        console.print("\n[bold]Latency Breakdown (excluding first request):[/bold]")
        console.print(f"  avg: {ft['avg_latency_ms']}ms")
        console.print(f"  p75: {ft['p75_latency_ms']}ms")
        console.print(f"  p90: {ft['p90_latency_ms']}ms")
        if "min_latency_ms" in ft:
            console.print(f"  min: {ft['min_latency_ms']}ms")
            console.print(f"  max: {ft['max_latency_ms']}ms")
        if "cold_start_latency_ms" in ft:
            console.print(f"  cold_start: {ft['cold_start_latency_ms']}ms (first request)")

    # Collect hallucinations
    hallucinations = []
    for result in eval_data["results"]:
        ft_result = result.get("fine_tuned", {})
        for h in ft_result.get("hallucinations", []):
            hallucinations.append(h)

    if hallucinations:
        console.print(f"\n[bold]Hallucinations Found:[/bold] {len(hallucinations)}")
        from collections import Counter

        for h, count in Counter(hallucinations).most_common(5):
            console.print(f'  - "{h}:" ({count}x)')

    # Overall status
    all_pass = syntax_pct >= 95 and semantic_pct >= 70 and latency <= 100
    if all_pass:
        console.print("\n[bold green]Status: ✓ READY FOR PRODUCTION[/bold green]")
    else:
        console.print("\n[bold yellow]Status: ⚠ NEEDS IMPROVEMENT[/bold yellow]")
        if syntax_pct < 95:
            console.print("  - Syntax validity below 95% target")
        if semantic_pct < 70:
            console.print("  - Semantic match below 70% target")
        if latency > 100:
            console.print("  - Latency above 100ms target")


@eval_app.command("show")
def show_results(
    file: Path = typer.Argument(None, help="JSON file to display (default: latest eval)"),
    full: bool = typer.Option(False, "--full", "-f", help="Show full results table"),
    verdicts: str = typer.Option(
        None,
        "--verdicts",
        "-v",
        help="Filter by verdict (comma-separated: fine_tuned_better,tie,both_wrong)",
    ),
) -> None:
    """Display results from a saved evaluation JSON file.

    Examples:
        nls-finetune eval show                    # Show latest eval summary
        nls-finetune eval show --full             # Show all examples
        nls-finetune eval show -v fine_tuned_better  # Show only wins
        nls-finetune eval show data/datasets/evaluations/eval-full-108.json
    """
    from rich.box import ROUNDED

    # Find file to display
    if file:
        eval_path = file
    else:
        eval_files = sorted(EVAL_DIR.glob("eval-*.json"), reverse=True)
        if not eval_files:
            console.print("[red]No evaluation files found.[/red]")
            raise typer.Exit(1)
        eval_path = eval_files[0]

    if not eval_path.exists():
        console.print(f"[red]File not found: {eval_path}[/red]")
        raise typer.Exit(1)

    with open(eval_path) as f:
        data = json.load(f)

    # Print header
    console.print(f"\n[bold]Evaluation: {data.get('id', eval_path.name)}[/bold]")
    console.print(f"[dim]File: {eval_path}[/dim]")
    console.print(f"[dim]Timestamp: {data.get('timestamp', 'N/A')}[/dim]")

    # Summary table using shared function
    ft = data["summary"]["fine_tuned"]
    total = data["summary"]["total"]
    bs = data["summary"].get("baseline", {})
    bs_model = data.get("models", {}).get("baseline", "gpt-4o-mini")
    latency_breakdown = data["summary"].get("latency_breakdown", {})

    _render_comparison_tables(
        ft_summary=ft,
        bs_summary=bs if bs else None,
        bs_model=bs_model,
        latency_breakdown=latency_breakdown if latency_breakdown else None,
        total=total,
    )

    # Verdict distribution
    results = data.get("results", [])
    verdict_counts = {"fine_tuned_better": 0, "tie": 0, "baseline_better": 0, "both_wrong": 0}
    for r in results:
        v = r.get("verdict", "both_wrong")
        if v in verdict_counts:
            verdict_counts[v] += 1

    verdict_table = Table(title="Verdict Distribution", box=ROUNDED, padding=(0, 1))
    verdict_table.add_column("Verdict", style="cyan")
    verdict_table.add_column("Count", justify="right")
    verdict_table.add_column("Pct", justify="right")

    def verdict_style(v: str) -> str:
        styles = {
            "fine_tuned_better": "green",
            "baseline_better": "red",
            "both_wrong": "dim",
        }
        return styles.get(v, "cyan")

    for v, count in verdict_counts.items():
        pct = 100 * count / len(results) if results else 0
        style = verdict_style(v)
        verdict_table.add_row(v.replace("_", " ").title(), str(count), f"{pct:.1f}%", style=style)

    console.print()
    console.print(verdict_table)

    # Full results table
    if full or verdicts:
        verdict_filter = None
        if verdicts:
            verdict_filter = set(v.strip() for v in verdicts.split(","))

        results_table = Table(title="Results", box=ROUNDED, padding=(0, 1), show_lines=True)
        results_table.add_column("ID", style="dim", width=8)
        results_table.add_column("Input", width=40, overflow="fold")
        results_table.add_column("Verdict", width=18)
        results_table.add_column("FT", justify="center", width=4)
        results_table.add_column("BL", justify="center", width=4)
        results_table.add_column("Latency", justify="right", width=8)

        for r in results:
            v = r.get("verdict", "both_wrong")
            if verdict_filter and v not in verdict_filter:
                continue

            ft_result = r.get("fine_tuned", {})
            bl_result = r.get("baseline", {})

            ft_ok = ft_result.get("syntax_valid", False) and ft_result.get("semantic_match", False)
            bl_ok = bl_result.get("syntax_valid", False) and bl_result.get("semantic_match", False)

            v_style = verdict_style(v)
            verdict_text = v.replace("_", " ").title()

            results_table.add_row(
                r.get("id", ""),
                r.get("input", "")[:80],
                f"[{v_style}]{verdict_text}[/{v_style}]",
                "[green]✓[/green]" if ft_ok else "[red]✗[/red]",
                "[green]✓[/green]" if bl_ok else "[red]✗[/red]" if bl_result else "-",
                f"{ft_result.get('latency_ms', 0):.0f}ms",
            )

        console.print()
        console.print(results_table)


@eval_app.command("load")
def load_test(
    endpoint: str = typer.Option(
        "https://sourcegraph--nls-finetune-serve-vllm-serve.modal.run",
        "--endpoint",
        help="Model endpoint URL",
    ),
    concurrent: int = typer.Option(5, "--concurrent", "-c", help="Number of concurrent requests"),
    total: int = typer.Option(50, "--total", "-n", help="Total number of requests"),
    warmup: bool = typer.Option(True, "--warmup/--no-warmup", help="Warmup endpoint first"),
) -> None:
    """Run concurrent load test to measure throughput and latency under load.

    Tests the model endpoint with concurrent requests to measure:
    - Throughput (requests per second)
    - Latency under load (avg, P90, P99)
    - Error rate
    """
    import asyncio
    import time

    import httpx

    console.print("[bold]Running load test[/bold]")
    console.print(f"  Endpoint: {endpoint}")
    console.print(f"  Concurrent requests: {concurrent}")
    console.print(f"  Total requests: {total}")

    # Warmup
    if warmup:
        console.print("\n[yellow]Warming up endpoint...[/yellow]")
        warmup_latencies = _warmup_endpoint(endpoint, "vllm", num_warmups=3)
        if warmup_latencies[0] > 0:
            console.print("[green]✓ Endpoint ready[/green]")
        else:
            console.print("[yellow]⚠ Warmup failed, continuing anyway...[/yellow]")

    # Build request payload
    url = endpoint.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = f"{url}/v1/chat/completions"

    payload = {
        "model": "llm",
        "messages": [
            {"role": "system", "content": "You are a Sourcegraph query generator."},
            {"role": "user", "content": "Query: find python async functions"},
        ],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async def make_request(client: httpx.AsyncClient, request_id: int) -> dict:
        """Make a single request and return timing info."""
        start = time.time()
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            latency_ms = (time.time() - start) * 1000
            return {"id": request_id, "latency_ms": latency_ms, "success": True}
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return {"id": request_id, "latency_ms": latency_ms, "success": False, "error": str(e)}

    async def run_load_test():
        """Run the load test with concurrent requests."""
        semaphore = asyncio.Semaphore(concurrent)
        results = []

        async def bounded_request(client: httpx.AsyncClient, request_id: int):
            async with semaphore:
                return await make_request(client, request_id)

        async with httpx.AsyncClient(timeout=60.0) as client:
            start_time = time.time()
            with console.status(
                f"[bold green]Running {total} requests ({concurrent} concurrent)..."
            ):
                tasks = [bounded_request(client, i) for i in range(total)]
                results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

        return results, total_time

    # Run the test
    console.print("\n[bold]Starting load test...[/bold]")
    results, total_time = asyncio.run(run_load_test())

    # Compute statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["latency_ms"] for r in successful]

    if latencies:
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        avg_latency = sum(latencies) / n
        p50_idx = int(n * 0.50)
        p90_idx = int(n * 0.90)
        p99_idx = int(n * 0.99)
        p50 = sorted_latencies[min(p50_idx, n - 1)]
        p90 = sorted_latencies[min(p90_idx, n - 1)]
        p99 = sorted_latencies[min(p99_idx, n - 1)]
        min_latency = sorted_latencies[0]
        max_latency = sorted_latencies[-1]
    else:
        avg_latency = p50 = p90 = p99 = min_latency = max_latency = 0

    throughput = len(successful) / total_time if total_time > 0 else 0
    error_rate = len(failed) / total * 100 if total > 0 else 0

    # Print results
    console.print("\n[bold green]Load Test Complete![/bold green]")
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total requests: {total}")
    console.print(f"  Successful: {len(successful)}")
    console.print(f"  Failed: {len(failed)} ({error_rate:.1f}%)")
    console.print(f"  Total time: {total_time:.1f}s")
    console.print(f"  Throughput: {throughput:.1f} req/s")

    console.print("\n[bold]Latency (successful requests):[/bold]")
    console.print(f"  avg: {avg_latency:.0f}ms")
    console.print(f"  p50: {p50:.0f}ms")
    console.print(f"  p90: {p90:.0f}ms")
    console.print(f"  p99: {p99:.0f}ms")
    console.print(f"  min: {min_latency:.0f}ms")
    console.print(f"  max: {max_latency:.0f}ms")

    if failed:
        console.print("\n[bold red]Errors:[/bold red]")
        error_types: dict[str, int] = {}
        for r in failed:
            err = r.get("error", "Unknown")
            error_types[err] = error_types.get(err, 0) + 1
        for err, count in sorted(error_types.items(), key=lambda x: -x[1])[:5]:
            console.print(f"  {count}x: {err[:80]}")
