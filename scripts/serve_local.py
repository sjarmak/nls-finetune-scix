#!/usr/bin/env python3
"""Local vLLM server for testing without Modal.

This script starts a local vLLM server using the fine-tuned model from HuggingFace
or a local path. Compatible with the evaluation CLI.

Usage:
    # Serve from HuggingFace (requires internet + HF token):
    python scripts/serve_local.py

    # Serve from local merged model:
    python scripts/serve_local.py --model-path ./output/merged

    # Custom port:
    python scripts/serve_local.py --port 8080

Requirements:
    pip install vllm

The server exposes OpenAI-compatible endpoints at:
    - http://localhost:8000/v1/chat/completions
    - http://localhost:8000/v1/models
    - http://localhost:8000/health
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start local vLLM server")
    parser.add_argument(
        "--model-path",
        type=str,
        default="adsabs/scix-nls-translator",
        help="Model path (HuggingFace repo or local path)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--max-model-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    args = parser.parse_args()

    print(f"Starting vLLM server...")
    print(f"  Model: {args.model_path}")
    print(f"  Port: {args.port}")
    print(f"  Max sequence length: {args.max_model_len}")
    print()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--served-model-name", "llm",
        "--enable-prefix-caching",
        "--max-num-batched-tokens", "512",
        "--disable-log-requests",
    ]

    print(f"Command: {' '.join(cmd)}")
    print()
    print("=" * 60)
    print("Server starting... (this may take 30-60s for model loading)")
    print("=" * 60)
    print()
    print("Once ready, run evaluation with:")
    print(f"  python -m finetune.cli.main eval run --endpoint http://localhost:{args.port}")
    print()
    print("Or test directly with curl:")
    print(f'''  curl -X POST http://localhost:{args.port}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{{"model": "llm", "messages": [{{"role": "user", "content": "papers about exoplanets"}}], "max_tokens": 128}}'
''')
    print("=" * 60)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
