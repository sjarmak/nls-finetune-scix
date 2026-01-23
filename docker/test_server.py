#!/usr/bin/env python3
"""Test the NLS server endpoints."""

import subprocess
import sys
import time
import requests

def main():
    print("Testing NLS server...")

    # Test health endpoint
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        print(f"\n/health: {r.status_code}")
        print(r.json())
    except Exception as e:
        print(f"Health check failed: {e}")
        return 1

    # Test vLLM endpoint
    try:
        r = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "llm",
                "messages": [
                    {"role": "user", "content": "Query: papers about exoplanets\nDate: 2026-01-23"}
                ],
                "max_tokens": 128
            },
            timeout=60
        )
        print(f"\n/v1/chat/completions: {r.status_code}")
        data = r.json()
        if "choices" in data and len(data["choices"]) > 0:
            print(f"Generated: {data['choices'][0]['message']['content']}")
        else:
            print(data)
    except Exception as e:
        print(f"vLLM endpoint failed: {e}")
        return 1

    # Test pipeline endpoint
    try:
        r = requests.post(
            "http://localhost:8000/pipeline",
            json={
                "model": "pipeline",
                "messages": [
                    {"role": "system", "content": "Convert natural language to ADS query."},
                    {"role": "user", "content": "Query: highly cited dark matter papers\nDate: 2026-01-23"}
                ]
            },
            timeout=60
        )
        print(f"\n/pipeline: {r.status_code}")
        data = r.json()
        if "choices" in data and len(data["choices"]) > 0:
            print(f"Generated: {data['choices'][0]['message']['content']}")
        if "pipeline_result" in data:
            print(f"Pipeline timing: {data['pipeline_result'].get('debug_info', {}).get('total_time_ms', 0):.0f}ms")
    except Exception as e:
        print(f"Pipeline endpoint failed: {e}")
        return 1

    print("\n✓ All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
