#!/bin/bash
# Initialization script for new development sessions
# Run this at the start of each coding session

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo "=========================================="
echo "NLS Query Fine-tune - Session Init"
echo "=========================================="
echo ""

# 1. Check mise is available
echo "--- Environment Setup ---"
if command -v mise &> /dev/null; then
    echo "✓ mise available"
    mise install
else
    echo "✗ mise not found - install with: brew install mise"
    exit 1
fi

# 2. Install dependencies if needed
echo ""
echo "--- Dependencies ---"
if [ ! -d "packages/api/.venv" ]; then
    echo "Installing Python dependencies..."
    cd packages/api && uv sync && cd ../..
else
    echo "✓ Python dependencies installed"
fi

if [ ! -d "packages/web/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd packages/web && bun install && cd ../..
else
    echo "✓ Frontend dependencies installed"
fi

# 3. Check .env file
echo ""
echo "--- Configuration ---"
if [ -f ".env" ]; then
    echo "✓ .env file exists"
else
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠ Edit .env to add your API keys"
fi

# 4. Show progress status
echo ""
echo "--- Progress Status ---"
if [ -f "features.json" ]; then
    passing=$(grep -c '"status": "passing"' features.json || echo 0)
    failing=$(grep -c '"status": "failing"' features.json || echo 0)
    echo "Features: $passing passing, $failing failing"
else
    echo "No features.json found"
fi

# 5. Show git status
echo ""
echo "--- Git Status ---"
git status --short || echo "Not a git repository"

# 6. Show recent commits
echo ""
echo "--- Recent Commits ---"
git log --oneline -5 2>/dev/null || echo "No commits yet"

# 7. Run verification
echo ""
echo "--- Running Verification ---"
./scripts/verify.sh || echo "⚠ Some checks failed - review above"

echo ""
echo "=========================================="
echo "Session initialized. Next steps:"
echo "1. Check features.json for failing features"
echo "2. Work on ONE feature at a time"
echo "3. Run 'mise run verify' after each change"
echo "4. Commit when feature passes"
echo "=========================================="
