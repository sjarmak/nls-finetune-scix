#!/bin/bash
# Verification script for feedback loops
# Run after every code change to ensure nothing is broken

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo "=========================================="
echo "NLS Query Fine-tune Verification"
echo "=========================================="
echo ""

# Track failures
FAILURES=0

# Helper function - runs command in subshell from ROOT_DIR
check() {
	local name="$1"
	shift
	echo -n "Checking $name... "
	if (cd "$ROOT_DIR" && "$@") >/dev/null 2>&1; then
		echo "✓ PASS"
		return 0
	else
		echo "✗ FAIL"
		FAILURES=$((FAILURES + 1))
		return 0 # Don't exit on failure, continue checking
	fi
}

# 1. Python lint
echo "--- Python Checks ---"
check "Python lint (ruff)" bash -c "cd packages/api && uv run ruff check . --quiet"
check "Python format" bash -c "cd packages/api && uv run ruff format --check . --quiet"

# 2. TypeScript checks
echo ""
echo "--- TypeScript Checks ---"
check "TypeScript types" bash -c "cd packages/web && bun run lint"

# 3. JSON validation
echo ""
echo "--- Data Validation ---"
check "Gold examples JSON" python3 -c "import json; json.load(open('data/datasets/raw/gold_examples.json'))"
check "Model configs JSON" python3 -c "import json; json.load(open('data/models/model_configs.json'))"
check "Features JSON" python3 -c "import json; json.load(open('features.json'))"

# 4. Modal scripts syntax (packages/finetune/src/finetune/modal/)
echo ""
echo "--- Modal Scripts ---"
MODAL_DIR="packages/finetune/src/finetune/modal"
check "modal/train.py syntax" python3 -m py_compile "$MODAL_DIR/train.py"
check "modal/dry_run.py syntax" python3 -m py_compile "$MODAL_DIR/dry_run.py"
check "modal/serve_vllm.py syntax" python3 -m py_compile "$MODAL_DIR/serve_vllm.py"
check "modal/merge.py syntax" python3 -m py_compile "$MODAL_DIR/merge.py"

# 5. Build check (optional, slower)
if [ "$1" = "--full" ]; then
	echo ""
	echo "--- Build Checks (--full mode) ---"
	check "Frontend build" bash -c "cd packages/web && bun run build"
fi

echo ""
echo "=========================================="
if [ $FAILURES -eq 0 ]; then
	echo "All checks passed! ✓"
	exit 0
else
	echo "$FAILURES check(s) failed! ✗"
	exit 1
fi
