# Development Guide

## Prerequisites

- **macOS** (Apple Silicon recommended)
- **[mise](https://mise.jdx.dev/)** - Runtime manager
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
| `/api/inference/*` | Query generation (routes to OpenAI or local endpoint) |
| `/api/datasets/*` | Dataset browsing and stats |
| `/api/models/*` | Model configuration |
| `/api/evaluation/*` | Evaluation results |

**Tech**: FastAPI, Pydantic, Python 3.12, uv

### Fine-tuning (`packages/finetune/`)

Training pipeline and CLI (uses src layout):

| Directory | Purpose |
|-----------|---------|
| `src/finetune/cli/` | `scix-finetune` command implementations |
| `src/finetune/eval/` | Evaluation runner and syntax validation |
| `src/finetune/domains/scix/` | ADS/SciX-specific pipeline logic |
| `src/finetune/dataset_agent/` | Dataset generation agent |

**Training**: Google Colab notebook (`scripts/train_colab.ipynb`) with Unsloth, TRL, LoRA, Qwen3-1.7B

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

Training is done via the Colab notebook:

1. Open `scripts/train_colab.ipynb` in Google Colab
2. Select an A100 GPU runtime (Colab Pro, ~90 min for 50-80k pairs)
3. Upload `data/datasets/processed/train.jsonl`
4. Run all cells — trains, merges LoRA, uploads to HuggingFace

The trained model is hosted on HuggingFace at `adsabs/scix-nls-translator` and can be served with vLLM or any compatible inference server.

See [docs/fine-tuning-cli.md](docs/fine-tuning-cli.md) for complete documentation.

### Running Evaluation

```bash
# Generate baseline (GPT-4o-mini)
scix-finetune eval baseline --sample 50

# Evaluate fine-tuned model
scix-finetune eval run --endpoint <your-endpoint-url> --sample 50

# Compare results
scix-finetune eval report
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
validate-data    Validate pairs and create train/val JSONL
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
