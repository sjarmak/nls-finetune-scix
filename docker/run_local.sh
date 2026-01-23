#!/bin/bash
# Run NLS server locally without Docker
# Works with MPS (Apple Silicon), CUDA, or CPU
#
# Usage:
#   ./docker/run_local.sh          # Auto-detect device
#   ./docker/run_local.sh cpu      # Force CPU
#   ./docker/run_local.sh mps      # Force MPS (Apple Silicon)
#   ./docker/run_local.sh cuda     # Force CUDA

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Determine device
if [ -n "$1" ]; then
    DEVICE="$1"
else
    # Auto-detect
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE="cuda"
    elif python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        DEVICE="mps"
    else
        DEVICE="cpu"
    fi
fi

echo "=========================================="
echo "NLS Server - Local Development"
echo "=========================================="
echo "Device: $DEVICE"
echo "Model: adsabs/scix-nls-translator"
echo "Port: 8000"
echo ""
echo "Endpoints:"
echo "  Pipeline: http://localhost:8000/"
echo "  vLLM:     http://localhost:8000/v1/chat/completions"
echo "  Health:   http://localhost:8000/health"
echo ""
echo "Configure nectar .env.local:"
echo "  NL_SEARCH_PIPELINE_ENDPOINT=http://localhost:8000"
echo "  NL_SEARCH_VLLM_ENDPOINT=http://localhost:8000/v1/chat/completions"
echo "  NEXT_PUBLIC_NL_SEARCH=enabled"
echo "=========================================="
echo ""

# Set environment and run
export MODEL_NAME="adsabs/scix-nls-translator"
export DEVICE="$DEVICE"
export PORT=8000
export PYTHONPATH="$PROJECT_DIR/packages/finetune/src:$PYTHONPATH"

python docker/server.py
