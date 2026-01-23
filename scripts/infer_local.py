#!/usr/bin/env python3
"""Local inference without vLLM for quick testing.

This script loads the fine-tuned model directly with transformers for
testing without setting up a full vLLM server.

Usage:
    # Interactive mode:
    python scripts/infer_local.py

    # Single query:
    python scripts/infer_local.py --query "papers about exoplanets from 2023"

    # Use local model path:
    python scripts/infer_local.py --model-path ./output/merged

Requirements:
    pip install torch transformers accelerate
"""

import argparse
import json
import sys
from datetime import datetime


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


def generate_query(model, tokenizer, natural_language: str, date: str = None) -> str:
    """Generate ADS query from natural language input."""
    import torch

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # System prompt for ADS query generation
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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Try to parse JSON from response
    response = response.strip()

    # Handle thinking mode output (Qwen3)
    if "<think>" in response:
        # Extract content after </think>
        parts = response.split("</think>")
        if len(parts) > 1:
            response = parts[-1].strip()

    # Try to extract JSON
    try:
        # Find JSON in response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            return data.get("query", response)
    except json.JSONDecodeError:
        pass

    return response


def interactive_mode(model, tokenizer):
    """Run interactive query generation."""
    print("\n" + "=" * 60)
    print("Interactive Mode - Enter natural language queries")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("NL Query: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            query = generate_query(model, tokenizer, user_input)
            print(f"ADS Query: {query}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Local inference for ADS query generation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="adsabs/scix-nls-translator",
        help="Model path (HuggingFace repo or local path)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Single query to process (otherwise runs interactive mode)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Reference date for query (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run on",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark with sample queries",
    )
    args = parser.parse_args()

    # Check device availability
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    if args.benchmark:
        # Run benchmark queries
        benchmark_queries = [
            "papers about exoplanets published in 2023",
            "articles by John Smith on machine learning",
            "highly cited papers about dark matter",
            "recent gravitational wave papers",
            "papers citing 2020ApJ...123..456A",
            "reviews about stellar evolution",
        ]

        print("\n" + "=" * 60)
        print("Running benchmark queries...")
        print("=" * 60 + "\n")

        import time
        total_time = 0
        for nl_query in benchmark_queries:
            start = time.time()
            ads_query = generate_query(model, tokenizer, nl_query, args.date)
            elapsed = time.time() - start
            total_time += elapsed
            print(f"Input:  {nl_query}")
            print(f"Output: {ads_query}")
            print(f"Time:   {elapsed*1000:.0f}ms\n")

        avg_time = total_time / len(benchmark_queries)
        print(f"Average latency: {avg_time*1000:.0f}ms")

    elif args.query:
        # Single query mode
        query = generate_query(model, tokenizer, args.query, args.date)
        print(f"ADS Query: {query}")

    else:
        # Interactive mode
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
