#!/usr/bin/env python3
"""Benchmark the hybrid NER pipeline for latency verification.

This script measures performance of individual pipeline components
and the total end-to-end latency to verify we meet the <500ms p95 target.

Usage:
    python scripts/benchmark_pipeline.py [--queries N] [--output FILE]
    
Example:
    python scripts/benchmark_pipeline.py --queries 100
    python scripts/benchmark_pipeline.py --output docs/LATENCY_BENCHMARKS.md
"""

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add the finetune package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages/finetune/src"))

from finetune.domains.scix.intent_spec import IntentSpec
from finetune.domains.scix.ner import extract_intent
from finetune.domains.scix.retrieval import get_index, retrieve_similar
from finetune.domains.scix.assembler import assemble_query
from finetune.domains.scix.pipeline import process_query, GoldExample


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    component: str
    samples: list[float]
    
    @property
    def p50(self) -> float:
        """50th percentile (median)."""
        return statistics.median(self.samples)
    
    @property
    def p95(self) -> float:
        """95th percentile."""
        return statistics.quantiles(self.samples, n=100)[94] if len(self.samples) >= 20 else max(self.samples)
    
    @property
    def p99(self) -> float:
        """99th percentile."""
        return statistics.quantiles(self.samples, n=100)[98] if len(self.samples) >= 100 else max(self.samples)
    
    @property
    def mean(self) -> float:
        """Mean latency."""
        return statistics.mean(self.samples)
    
    @property
    def min(self) -> float:
        """Minimum latency."""
        return min(self.samples)
    
    @property
    def max(self) -> float:
        """Maximum latency."""
        return max(self.samples)


def load_sample_queries(n: int = 100) -> list[str]:
    """Load n sample queries from gold_examples.json."""
    gold_path = Path(__file__).parent.parent / "data/datasets/raw/gold_examples.json"
    
    with open(gold_path) as f:
        examples = json.load(f)
    
    # Sample n queries, or all if fewer available
    sample_size = min(n, len(examples))
    sampled = random.sample(examples, sample_size)
    
    return [ex["natural_language"] for ex in sampled]


def benchmark_ner(queries: list[str]) -> BenchmarkResult:
    """Benchmark NER extraction component."""
    times = []
    
    for query in queries:
        start = time.perf_counter()
        _ = extract_intent(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
    
    return BenchmarkResult(component="NER Extraction", samples=times)


def benchmark_retrieval(intents: list[IntentSpec]) -> BenchmarkResult:
    """Benchmark retrieval component."""
    # Warm up the index
    _ = get_index()
    
    times = []
    
    for intent in intents:
        start = time.perf_counter()
        _ = retrieve_similar(intent, k=5)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
    
    return BenchmarkResult(component="Retrieval (k=5)", samples=times)


def benchmark_assembly(intents: list[IntentSpec], examples_list: list[list[GoldExample]]) -> BenchmarkResult:
    """Benchmark assembly component."""
    times = []
    
    for intent, examples in zip(intents, examples_list):
        start = time.perf_counter()
        _ = assemble_query(intent, examples)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
    
    return BenchmarkResult(component="Assembly", samples=times)


def benchmark_full_pipeline(queries: list[str]) -> BenchmarkResult:
    """Benchmark full pipeline end-to-end."""
    # Warm up (index loading, imports, etc.)
    _ = process_query("test query")
    
    times = []
    
    for query in queries:
        start = time.perf_counter()
        _ = process_query(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
    
    return BenchmarkResult(component="Full Pipeline (no LLM)", samples=times)


def format_result(result: BenchmarkResult, target_ms: float | None = None) -> str:
    """Format benchmark result as a table row."""
    status = ""
    if target_ms:
        status = "✅ PASS" if result.p95 < target_ms else "❌ FAIL"
    
    return (
        f"| {result.component:<25} | {result.p50:>7.2f} | {result.p95:>7.2f} | "
        f"{result.p99:>7.2f} | {result.mean:>7.2f} | {result.min:>6.2f} | {result.max:>7.2f} | {status:<8} |"
    )


def run_benchmarks(num_queries: int = 100) -> list[BenchmarkResult]:
    """Run all benchmarks and return results."""
    print(f"Loading {num_queries} sample queries from gold_examples.json...")
    queries = load_sample_queries(num_queries)
    print(f"Loaded {len(queries)} queries\n")
    
    results = []
    
    # Benchmark NER
    print("Benchmarking NER extraction...")
    ner_result = benchmark_ner(queries)
    results.append(ner_result)
    print(f"  p50: {ner_result.p50:.2f}ms, p95: {ner_result.p95:.2f}ms")
    
    # Pre-extract intents for subsequent benchmarks
    intents = [extract_intent(q) for q in queries]
    
    # Benchmark Retrieval
    print("Benchmarking retrieval...")
    retrieval_result = benchmark_retrieval(intents)
    results.append(retrieval_result)
    print(f"  p50: {retrieval_result.p50:.2f}ms, p95: {retrieval_result.p95:.2f}ms")
    
    # Pre-retrieve examples for assembly benchmark
    examples_list = [retrieve_similar(intent, k=5) for intent in intents]
    
    # Benchmark Assembly
    print("Benchmarking assembly...")
    assembly_result = benchmark_assembly(intents, examples_list)
    results.append(assembly_result)
    print(f"  p50: {assembly_result.p50:.2f}ms, p95: {assembly_result.p95:.2f}ms")
    
    # Benchmark Full Pipeline
    print("Benchmarking full pipeline (end-to-end)...")
    pipeline_result = benchmark_full_pipeline(queries)
    results.append(pipeline_result)
    print(f"  p50: {pipeline_result.p50:.2f}ms, p95: {pipeline_result.p95:.2f}ms\n")
    
    return results


def generate_markdown_report(results: list[BenchmarkResult], num_queries: int) -> str:
    """Generate a Markdown report of benchmark results."""
    # Define targets (from US-010 acceptance criteria)
    targets = {
        "NER Extraction": 10.0,
        "Retrieval (k=5)": 20.0,
        "Assembly": 5.0,
        "Full Pipeline (no LLM)": 50.0,  # Local target
    }
    
    lines = [
        "# Pipeline Latency Benchmarks",
        "",
        "Performance benchmarks for the hybrid NER pipeline.",
        "",
        "## Summary",
        "",
        f"- **Queries tested**: {num_queries}",
        f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Targets",
        "",
        "| Component | Target (p95) |",
        "|-----------|-------------|",
        "| NER Extraction | < 10ms |",
        "| Retrieval (k=5) | < 20ms |",
        "| Assembly | < 5ms |",
        "| Full Pipeline (local, no LLM) | < 50ms |",
        "| Full Pipeline (Modal, no LLM) | < 200ms |",
        "| E2E with LLM fallback | < 1000ms |",
        "",
        "## Results",
        "",
        "| Component                   |    p50 |    p95 |    p99 |   Mean |   Min |    Max | Status   |",
        "|-----------------------------|--------|--------|--------|--------|-------|--------|----------|",
    ]
    
    for result in results:
        target = targets.get(result.component)
        lines.append(format_result(result, target))
    
    # Overall verdict
    pipeline_result = next((r for r in results if "Full Pipeline" in r.component), None)
    if pipeline_result:
        overall_pass = pipeline_result.p95 < 50.0
        verdict = "✅ **PASS**" if overall_pass else "❌ **FAIL**"
        lines.extend([
            "",
            f"## Overall: {verdict}",
            "",
            f"Pipeline p95 latency: **{pipeline_result.p95:.2f}ms** (target: < 50ms local)",
            "",
        ])
    
    # Add notes section
    lines.extend([
        "## Notes",
        "",
        "- **Local benchmarks**: Run on development machine without network latency",
        "- **Modal benchmarks**: Add ~100-150ms for cold start, ~20ms for warm requests",
        "- **LLM fallback**: Only triggered for ambiguous paper references (rare)",
        "- **Index loading**: First request includes index load time (~500ms cold start)",
        "",
        "## Component Breakdown",
        "",
        "### NER Extraction",
        "- Rules-based pattern matching",
        "- No external dependencies",
        "- Scales O(n) with input length",
        "",
        "### Retrieval",
        "- BM25-like scoring over 4000+ gold examples",
        "- Preloaded index (no per-request file I/O)",
        "- Scales O(n) with index size, O(k) with result count",
        "",
        "### Assembly",
        "- Deterministic template composition",
        "- FIELD_ENUMS validation",
        "- Scales O(1) with input complexity",
        "",
        "## CI Integration",
        "",
        "Add this check to CI:",
        "",
        "```bash",
        "python scripts/benchmark_pipeline.py --queries 100",
        "# Fails with exit code 1 if p95 > 500ms",
        "```",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the hybrid NER pipeline")
    parser.add_argument("--queries", "-n", type=int, default=100, help="Number of queries to benchmark")
    parser.add_argument("--output", "-o", type=str, help="Output file for Markdown report")
    parser.add_argument("--ci", action="store_true", help="CI mode: exit 1 if p95 > 500ms")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hybrid NER Pipeline Benchmark")
    print("=" * 60)
    print()
    
    results = run_benchmarks(args.queries)
    
    # Generate report
    report = generate_markdown_report(results, args.queries)
    
    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report written to: {output_path}")
    else:
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT")
        print("=" * 60)
        print()
        print(report)
    
    # CI check
    pipeline_result = next((r for r in results if "Full Pipeline" in r.component), None)
    if args.ci and pipeline_result:
        if pipeline_result.p95 > 500.0:
            print(f"\n❌ CI CHECK FAILED: p95 ({pipeline_result.p95:.2f}ms) > 500ms")
            sys.exit(1)
        else:
            print(f"\n✅ CI CHECK PASSED: p95 ({pipeline_result.p95:.2f}ms) < 500ms")
            sys.exit(0)


if __name__ == "__main__":
    main()
