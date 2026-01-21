# NLS Fine-tune SciX

Fine-tuning infrastructure for converting natural language to ADS/SciX scientific literature search queries.

**Example:** "papers by Hawking on black hole radiation from the 1970s" → `author:"Hawking, S" abs:"black hole radiation" pubdate:[1970 TO 1979]`

**Target:** Complementary search feature for [SciXplorer.org](https://scixplorer.org/)

## Prerequisites

- **macOS** (Apple Silicon recommended) - Linux/Windows not currently supported
- **[mise](https://mise.jdx.dev/)** - Runtime manager for Python and Bun
- **ADS API Key** - For query validation and evaluation

```bash
brew install mise
```

### ADS API Key

Get your API key from [ADS User Settings](https://ui.adsabs.harvard.edu/user/settings/token) and add it to your `.env` file.

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd nls-finetune-scix
mise run install

# Configure environment
cp .env.example .env
# Edit .env with your API keys (see below)

# Start development
mise run dev
```

- **Web UI**: http://localhost:5173
- **API**: http://localhost:8000

## API Keys

Configure in `.env`. Not all keys are required depending on what you're doing:

| Key | Required For | Where to Get |
|-----|-------------|--------------|
| `ADS_API_KEY` | Query validation, evaluation | [ADS User Settings](https://ui.adsabs.harvard.edu/user/settings/token) |
| `MODAL_TOKEN_ID` | Training, deployment | [modal.com/settings](https://modal.com/settings) |
| `MODAL_TOKEN_SECRET` | Training, deployment | [modal.com/settings](https://modal.com/settings) |
| `OPENAI_API_KEY` | GPT-4o-mini comparison | [platform.openai.com](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Dataset generation | [console.anthropic.com](https://console.anthropic.com/) |
| `MODAL_INFERENCE_ENDPOINT` | Using deployed model | Output from `scix-finetune deploy` |

### Setup by Role

| I want to... | Keys needed |
|--------------|-------------|
| **Develop the web UI** | None (uses mock data) |
| **Run the full stack locally** | `ADS_API_KEY` + `MODAL_INFERENCE_ENDPOINT` |
| **Train a new model** | Modal + ADS keys |
| **Generate new training data** | Anthropic + ADS keys |

## Commands

All commands use `mise run <task>`. Run `mise tasks` to see all available tasks.

### Development

```bash
mise run dev          # Start API + web servers (parallel)
mise run dev-api      # Start API only (port 8000)
mise run dev-web      # Start web only (port 5173)
```

### Verification

```bash
mise run verify       # Run all checks (lint, types, JSON validation)
mise run verify-full  # All checks + frontend build
mise run lint         # Linters only
mise run format       # Auto-format Python code
mise run test         # Run tests
```

### Fine-Tuning (requires Modal keys)

For full training workflow, use the `scix-finetune` CLI:

```bash
mise run install              # Install CLI (via uv workspace)
scix-finetune --help          # Show all commands
scix-finetune verify env      # Check Modal setup
scix-finetune dry-run train   # Test pipeline (3 steps, ~$0.10)
scix-finetune train           # Full training (~12 min, ~$1.50)
```

See [docs/fine-tuning-cli.md](docs/fine-tuning-cli.md) for detailed training documentation.

### Dataset Management

```bash
mise run generate-data   # Full pipeline: curate → generate NL → validate
mise run validate-data   # Validate and create train/val JSONL
```

## Project Structure

```
├── packages/
│   ├── web/            # React frontend (Vite + TanStack Query)
│   ├── api/            # FastAPI backend
│   └── finetune/       # Fine-tuning package (src layout)
│       └── src/finetune/
│           ├── cli/    # scix-finetune CLI commands
│           ├── modal/  # Modal training & inference code
│           ├── eval/   # Evaluation modules
│           └── domains/
│               └── scix/  # ADS/SciX-specific logic
│                   ├── fields.py    # ADS field definitions
│                   ├── validate.py  # Query validation
│                   └── eval.py      # Result-set evaluation
├── data/
│   ├── datasets/raw/   # Source data (curated examples, synthetic queries)
│   ├── datasets/processed/  # Training data (train.jsonl, val.jsonl)
│   └── datasets/evaluations/  # Evaluation results
├── scripts/            # Data processing scripts
├── docs/               # Documentation
└── .mise.toml          # Task runner configuration
```

## ADS Search Syntax Reference

The model learns to generate queries using [ADS Search Syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax):

| Field | Example | Description |
|-------|---------|-------------|
| `author:` | `author:"Einstein, A"` | Author name |
| `^author:` | `^author:"Hawking"` | First author only |
| `abs:` | `abs:"dark matter"` | Abstract, title, keywords |
| `title:` | `title:"exoplanet"` | Title only |
| `pubdate:` | `pubdate:[2020 TO 2023]` | Publication date range |
| `bibstem:` | `bibstem:ApJ` | Journal abbreviation |
| `object:` | `object:M31` | Astronomical object |
| `keyword:` | `keyword:"galaxies"` | Keywords |
| `aff:` | `aff:"Harvard"` | Affiliation |
| `citation_count:` | `citation_count:[100 TO *]` | Citation count range |

## Key Files

| File | Purpose |
|------|---------|
| `.mise.toml` | All available commands (`mise tasks` to list) |
| `features.json` | Feature tracking - find failing features to work on |
| `.env.example` | Environment variables template |

## Tech Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, TanStack Query
- **Backend**: FastAPI, Pydantic, Python 3.12
- **Training**: Modal (H100 GPUs), TRL, LoRA, Qwen3-1.7B
- **Validation**: ADS Search API
- **Tools**: mise (runtimes), uv (Python), Bun (Node)

## Documentation

- [Development Guide](DEVELOPMENT.md) - Architecture, workflows
- [Fine-Tuning CLI](docs/fine-tuning-cli.md) - Training pipeline documentation
- [ADS Search Syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax) - Official ADS docs
