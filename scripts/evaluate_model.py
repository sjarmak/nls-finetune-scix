#!/usr/bin/env python3
"""Evaluate fine-tuned model quality directly (without vLLM server).

This script evaluates the model by loading it directly with transformers,
which is simpler than setting up a vLLM server for quick local testing.

Usage:
    # Evaluate from HuggingFace:
    python scripts/evaluate_model.py

    # Evaluate local model:
    python scripts/evaluate_model.py --model-path ./output/merged

    # Limit sample size for quick testing:
    python scripts/evaluate_model.py --sample 20

    # Compare with baseline (requires OPENAI_API_KEY):
    python scripts/evaluate_model.py --baseline

Requirements:
    pip install torch transformers accelerate
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add the finetune package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages/finetune/src"))


@dataclass
class EvalResult:
    """Result for a single evaluation example."""
    id: str
    input: str
    expected: str
    output: str
    syntax_valid: bool
    semantic_match: bool
    exact_match: bool
    latency_ms: float


def load_model(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    print(f"Model loaded on {device}")
    return model, tokenizer


def generate_query(model, tokenizer, natural_language: str, date: str = None) -> tuple[str, float]:
    """Generate ADS query from natural language input.

    Returns:
        Tuple of (query_string, latency_ms)
    """
    import torch

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = '''Convert natural language to ADS search query. Output JSON: {"query": "..."}'''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {natural_language}\nDate: {date}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - start_time) * 1000

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Parse response
    response = response.strip()

    # Handle thinking mode output
    if "<think>" in response:
        parts = response.split("</think>")
        if len(parts) > 1:
            response = parts[-1].strip()

    # Extract JSON
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            return data.get("query", response), latency_ms
    except json.JSONDecodeError:
        pass

    return response, latency_ms


def normalize_query(query: str) -> str:
    """Normalize query for comparison."""
    normalized = query.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.replace('"', '"').replace('"', '"')
    return normalized


def check_syntax(query: str) -> bool:
    """Check if query has valid ADS syntax (basic check)."""
    # Basic checks
    if not query or query.startswith("ERROR"):
        return False

    # Check balanced parentheses
    if query.count("(") != query.count(")"):
        return False

    # Check for common malformed patterns
    bad_patterns = [
        r"citationsabs:",
        r"referencesabs:",
        r"citations\s*\(\s*citations",
        r"references\s*\(\s*references",
    ]
    for pattern in bad_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False

    return True


def check_semantic_match(expected: str, output: str) -> bool:
    """Check if output semantically matches expected query."""
    # Extract key components
    def extract_components(query: str) -> set:
        components = set()

        # Extract field:value pairs
        field_pattern = r"(\w+):(?:\"([^\"]+)\"|(\S+))"
        for match in re.finditer(field_pattern, query, re.IGNORECASE):
            field = match.group(1).lower()
            value = match.group(2) or match.group(3)
            components.add(f"{field}:{value.lower()}")

        # Extract operators
        op_pattern = r"(citations|references|trending|similar|useful|reviews|topn)\s*\("
        for match in re.finditer(op_pattern, query, re.IGNORECASE):
            components.add(f"op:{match.group(1).lower()}")

        return components

    expected_components = extract_components(expected)
    output_components = extract_components(output)

    if not expected_components:
        return True  # No specific components to check

    # Check overlap
    overlap = len(expected_components & output_components)
    total = len(expected_components)

    return overlap / total >= 0.5 if total > 0 else True


def load_validation_set(path: Path) -> list[dict]:
    """Load validation examples."""
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            # Extract NL and expected query from messages
            messages = data.get("messages", [])
            nl_input = None
            expected = None

            for msg in messages:
                if msg["role"] == "user":
                    # Extract query from user message
                    content = msg["content"]
                    if "Query:" in content:
                        nl_input = content.split("Query:")[1].split("\n")[0].strip()
                elif msg["role"] == "assistant":
                    # Extract query from assistant response
                    content = msg["content"]
                    try:
                        json_data = json.loads(content)
                        expected = json_data.get("query", "")
                    except json.JSONDecodeError:
                        expected = content

            if nl_input and expected:
                examples.append({
                    "id": f"val_{len(examples)}",
                    "input": nl_input,
                    "expected": expected,
                })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="adsabs/scix-nls-translator",
        help="Model path (HuggingFace repo or local path)",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default="data/datasets/processed/val.jsonl",
        help="Path to validation set",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run on",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run baseline comparison (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each example result",
    )
    args = parser.parse_args()

    # Check device
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    # Load validation set
    val_path = Path(args.val_path)
    if not val_path.exists():
        print(f"Error: Validation set not found: {val_path}")
        return 1

    examples = load_validation_set(val_path)
    if args.sample and args.sample < len(examples):
        examples = examples[:args.sample]

    print(f"Loaded {len(examples)} validation examples")

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    # Warmup
    print("Warming up model...")
    _ = generate_query(model, tokenizer, "warmup test")

    # Evaluate
    print(f"\nEvaluating {len(examples)} examples...")
    results = []
    total_syntax_valid = 0
    total_semantic_match = 0
    total_exact_match = 0
    latencies = []

    for i, example in enumerate(examples):
        output, latency = generate_query(model, tokenizer, example["input"])

        syntax_valid = check_syntax(output)
        semantic_match = check_semantic_match(example["expected"], output)
        exact_match = normalize_query(example["expected"]) == normalize_query(output)

        result = EvalResult(
            id=example["id"],
            input=example["input"],
            expected=example["expected"],
            output=output,
            syntax_valid=syntax_valid,
            semantic_match=semantic_match,
            exact_match=exact_match,
            latency_ms=latency,
        )
        results.append(result)

        if syntax_valid:
            total_syntax_valid += 1
        if semantic_match:
            total_semantic_match += 1
        if exact_match:
            total_exact_match += 1
        latencies.append(latency)

        if args.verbose:
            status = "✓" if syntax_valid and semantic_match else "✗"
            print(f"[{i+1}/{len(examples)}] {status} {example['input'][:50]}...")
            print(f"  Expected: {example['expected'][:60]}...")
            print(f"  Output:   {output[:60]}...")
            print(f"  Latency:  {latency:.0f}ms\n")
        elif (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(examples)}...")

    # Compute statistics
    n = len(results)

    # Latency stats (exclude first for warm metrics)
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    if warm_latencies:
        sorted_lat = sorted(warm_latencies)
        avg_latency = sum(warm_latencies) / len(warm_latencies)
        p75_idx = int(len(sorted_lat) * 0.75)
        p90_idx = int(len(sorted_lat) * 0.90)
        p75_latency = sorted_lat[min(p75_idx, len(sorted_lat) - 1)]
        p90_latency = sorted_lat[min(p90_idx, len(sorted_lat) - 1)]
        min_latency = sorted_lat[0]
        max_latency = sorted_lat[-1]
    else:
        avg_latency = p75_latency = p90_latency = min_latency = max_latency = 0

    cold_start_latency = latencies[0] if latencies else 0

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nModel: {args.model_path}")
    print(f"Examples: {n}")
    print(f"\nQuality Metrics:")
    print(f"  Syntax Valid:    {total_syntax_valid}/{n} ({100*total_syntax_valid/n:.1f}%)")
    print(f"  Semantic Match:  {total_semantic_match}/{n} ({100*total_semantic_match/n:.1f}%)")
    print(f"  Exact Match:     {total_exact_match}/{n} ({100*total_exact_match/n:.1f}%)")
    print(f"\nLatency (excluding first request):")
    print(f"  Average: {avg_latency:.0f}ms")
    print(f"  P75:     {p75_latency:.0f}ms")
    print(f"  P90:     {p90_latency:.0f}ms")
    print(f"  Min:     {min_latency:.0f}ms")
    print(f"  Max:     {max_latency:.0f}ms")
    print(f"  Cold Start: {cold_start_latency:.0f}ms")

    # Check targets
    print("\n" + "-" * 40)
    syntax_pass = total_syntax_valid / n >= 0.95
    semantic_pass = total_semantic_match / n >= 0.70
    latency_pass = avg_latency <= 200  # More lenient for local CPU/GPU

    print(f"Syntax Valid ≥95%:    {'✓ PASS' if syntax_pass else '✗ FAIL'}")
    print(f"Semantic Match ≥70%:  {'✓ PASS' if semantic_pass else '✗ FAIL'}")
    print(f"Avg Latency ≤200ms:   {'✓ PASS' if latency_pass else '✗ FAIL'}")

    if syntax_pass and semantic_pass and latency_pass:
        print("\n[✓] Model PASSES all quality targets")
    else:
        print("\n[!] Model needs improvement")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        eval_dir = Path("data/datasets/evaluations")
        eval_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = eval_dir / f"eval-{timestamp}.json"

    eval_results = {
        "id": f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "device": args.device,
        "summary": {
            "total": n,
            "syntax_valid": total_syntax_valid,
            "semantic_match": total_semantic_match,
            "exact_match": total_exact_match,
            "avg_latency_ms": round(avg_latency, 1),
            "p75_latency_ms": round(p75_latency, 1),
            "p90_latency_ms": round(p90_latency, 1),
            "min_latency_ms": round(min_latency, 1),
            "max_latency_ms": round(max_latency, 1),
            "cold_start_latency_ms": round(cold_start_latency, 1),
        },
        "results": [
            {
                "id": r.id,
                "input": r.input,
                "expected": r.expected,
                "output": r.output,
                "syntax_valid": r.syntax_valid,
                "semantic_match": r.semantic_match,
                "exact_match": r.exact_match,
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if (syntax_pass and semantic_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
