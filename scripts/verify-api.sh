#!/bin/bash
# API verification script - checks if API is running and endpoints work
# Assumes API is running on localhost:8000

set -e

API_BASE="${API_BASE:-http://localhost:8000}"

echo "=========================================="
echo "API Verification (${API_BASE})"
echo "=========================================="
echo ""

FAILURES=0

check_endpoint() {
    local name="$1"
    local method="$2"
    local path="$3"
    local expected="$4"
    local data="$5"

    echo -n "Checking $name... "

    if [ "$method" = "GET" ]; then
        response=$(curl -sL "${API_BASE}${path}" 2>/dev/null || echo "CURL_FAILED")
    else
        response=$(curl -sL -X POST "${API_BASE}${path}" -H "Content-Type: application/json" -d "$data" 2>/dev/null || echo "CURL_FAILED")
    fi

    if echo "$response" | grep -q "$expected"; then
        echo "✓ PASS"
        return 0
    else
        echo "✗ FAIL"
        echo "  Expected: $expected"
        echo "  Got: ${response:0:100}..."
        FAILURES=$((FAILURES + 1))
        return 1
    fi
}

# Health check
check_endpoint "Health" "GET" "/api/health" "ok"

# Models endpoint
check_endpoint "Models list" "GET" "/api/models" "fine-tuned"

# Dataset stats
check_endpoint "Dataset stats" "GET" "/api/datasets/stats" "total_examples"

# Dataset examples
check_endpoint "Dataset examples" "GET" "/api/datasets/examples" "gold"

# Inference generate (placeholder response is fine)
check_endpoint "Inference generate" "POST" "/api/inference/generate" "sourcegraph_query" '{"query":"test","candidates":[],"model_id":"fine-tuned"}'

# Inference compare
check_endpoint "Inference compare" "POST" "/api/inference/compare" "results" '{"query":"test","candidates":[],"model_ids":["fine-tuned"]}'

echo ""
echo "=========================================="
if [ $FAILURES -eq 0 ]; then
    echo "All API checks passed! ✓"
    exit 0
else
    echo "$FAILURES API check(s) failed! ✗"
    exit 1
fi
