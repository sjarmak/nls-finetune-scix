# Development Guide

## Prerequisites

- **macOS** (Apple Silicon recommended)
- **[mise](https://mise.jdx.dev/)** - Runtime manager
- **Modal account** - For fine-tuning and inference ([modal.com](https://modal.com))
- **API keys** - See README.md for which keys you need

## Setup

```bash
# 1. Install mise
brew install mise

# 2. Clone and install
git clone <repo-url>
cd nls-query-finetune
mise run install

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start development
mise run dev
```

## Architecture

### Frontend (`packages/web/`)

React application with three main pages:

| Page | Purpose |
|------|---------|
| **Playground** | Test query generation with side-by-side model comparison |
| **Dataset Browser** | View and filter training examples (gold + generated) |
| **Evaluation** | Compare model performance against baseline |

**Tech**: React 19, TypeScript, Vite, TanStack Query, Tailwind CSS, shadcn/ui

### Backend (`packages/api/`)

FastAPI server providing:

| Endpoint | Purpose |
|----------|---------|
| `/api/inference/*` | Query generation (routes to Modal or OpenAI) |
| `/api/datasets/*` | Dataset browsing and stats |
| `/api/models/*` | Model configuration |
| `/api/evaluation/*` | Evaluation results |

**Tech**: FastAPI, Pydantic, Python 3.12, uv

### Fine-tuning (`packages/finetune/`)

Training pipeline and CLI (uses src layout):

| Directory | Purpose |
|-----------|---------|
| `src/finetune/cli/` | `nls-finetune` command implementations |
| `src/finetune/modal/` | Modal training, inference, and deployment code |
| `src/finetune/eval/` | Evaluation runner and syntax validation |

**Tech**: Modal (H100 GPUs), TRL, LoRA, Qwen3-1.7B

## Development Workflow

### Daily Development

```bash
# Start servers
mise run dev

# After code changes
mise run verify

# Format code
mise run format
```

### Adding Training Data

1. Add examples to `data/datasets/raw/gold_examples.json`
2. Run `mise run validate-data` to regenerate train/val split

### Fine-tuning a Model

Use the `nls-finetune` CLI:

```bash
# Install CLI (via uv workspace - included in mise run install)
mise run install

# Verify setup
nls-finetune verify env

# Test pipeline (3 steps, quick)
nls-finetune dry-run train

# Full training (~12 min, ~$1.50)
nls-finetune train --run-name "my-run"

# Merge LoRA adapter
nls-finetune merge --run-name "my-run"

# Deploy inference endpoint
nls-finetune deploy --run-name "my-run"
```

See [docs/fine-tuning-cli.md](docs/fine-tuning-cli.md) for complete documentation.

### Running Evaluation

```bash
# Generate baseline (GPT-4o-mini)
nls-finetune eval baseline --sample 50

# Evaluate fine-tuned model
nls-finetune eval run --sample 50

# Compare results
nls-finetune eval report
```

Or use the web UI: http://localhost:5173 → Evaluation page

## Available Commands

Run `mise tasks` to see all commands:

```
dev              Start API and web servers (parallel)
dev-api          Start API server only (port 8000)
dev-web          Start web dev server only (port 5173)
verify           Run all checks (lint, types, data validation)
verify-full      Run all checks including frontend build
lint             Run linters (Python + TypeScript)
format           Format Python code
test             Run all tests
generate-data    Full data pipeline: extract → generate NL → validate
validate-data    Validate pairs and create train/val split
modal-deploy     Deploy inference endpoint to Modal
modal-train      Run training job on Modal
clean            Remove build artifacts and dependencies
```

## Troubleshooting

### Port already in use

```bash
lsof -i :8000  # Find process
kill -9 <PID>  # Kill it
```

### Module not found

```bash
cd packages/api && uv sync  # Reinstall Python deps
cd packages/web && bun install  # Reinstall Node deps
```

### Modal authentication

```bash
modal token set  # Re-authenticate
nls-finetune verify env  # Check setup
```

## Chrome DevTools MCP

For AI-assisted development with Claude Code, the Chrome DevTools MCP server enables automated browser testing and visual verification.

### Installation

```bash
claude mcp add chrome-devtools -- npx chrome-devtools-mcp@latest
```

That's it. Claude Code can now take screenshots, inspect pages, and interact with the browser.

### What It Enables

- `take_snapshot` - Get page content as accessibility tree
- `take_screenshot` - Capture visual state
- `click`, `fill`, `navigate_page` - Interact with the page

This is referenced in `features.json` verification steps for UI features.

### Manual Alternative

If not using Claude Code, verify UI changes manually in your browser.
