# NLS Fine-tune SciX

Fine-tuning infrastructure for converting natural language to ADS/SciX scientific literature search queries.

**Example:** "papers by Hawking on black hole radiation from the 1970s" → `author:"Hawking, S" abs:"black hole radiation" pubdate:[1970 TO 1979]`

**Target:** Complementary search feature for [SciXplorer.org](https://scixplorer.org/)

## Prerequisites

- **macOS** (Apple Silicon recommended) - Linux/Windows not currently supported
- **[mise](https://mise.jdx.dev/)** - Runtime manager for Python and Bun
- **ADS API Key** - For query validation and evaluation
- **Google Colab** (optional) - For model training (A100 GPU via Colab Pro)

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
| `ANTHROPIC_API_KEY` | Dataset generation | [console.anthropic.com](https://console.anthropic.com/) |

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

### Fine-Tuning (via Google Colab)

Training is done via the provided Colab notebook:

1. Open `scripts/train_colab.ipynb` in Google Colab
2. Select an A100 GPU runtime (Colab Pro)
3. Upload `data/datasets/processed/train.jsonl`
4. Run all cells (~90 minutes for 50-80k pairs)
5. Model is uploaded to HuggingFace (`adsabs/scix-nls-translator`)

See [docs/fine-tuning-cli.md](docs/fine-tuning-cli.md) for detailed training and deployment documentation.

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

## How the System Works

A **fine-tuned Qwen3-1.7B model** ([adsabs/scix-nls-translator](https://huggingface.co/adsabs/scix-nls-translator)) converts natural language to ADS queries end-to-end. The model is served locally via an OpenAI-compatible endpoint and integrated into the [nectar](https://github.com/adsabs/nectar) frontend.

```
User NL query → Nectar (:8000) → Model Server (:8001) → ADS query
```

### Model

The translator is a **Qwen3-1.7B** base model fine-tuned with **LoRA** (r=16, alpha=32) using [Unsloth](https://github.com/unslothai/unsloth) + TRL on Google Colab (A100 GPU, ~90 minutes).

- **Model**: [adsabs/scix-nls-translator](https://huggingface.co/adsabs/scix-nls-translator) on HuggingFace
- **Input format**: `Query: <natural language>\nDate: <YYYY-MM-DD>`
- **Output format**: `{"query": "<ADS query>"}`
- **Training notebook**: `scripts/train_colab.ipynb`

### Training Dataset

**Dataset**: [adsabs/nls-query-training-data](https://huggingface.co/datasets/adsabs/nls-query-training-data) on HuggingFace

The training data (61,652 pairs — 55.4k train / 6.2k val) is assembled from three sources:

| Source | Description |
|--------|-------------|
| **NL pairs from query logs** | Real ADS queries from search logs, paired with natural language descriptions generated by Claude Sonnet 4. Validated to prevent query syntax leaking into NL text. |
| **Gold examples** | Hand-curated NL-to-query pairs covering diverse query patterns (topics, authors, operators, date ranges, compound queries). |
| **Synthetic pairs** | Programmatically generated from templates using seed lists of topics, astronomers, objects, journals, and institutions. Covers edge cases and common patterns. |

The combined dataset is validated, deduplicated by both NL text and query text, and split 90/10 into train/val sets with a fixed seed for reproducibility.

Each example uses standard chat completion format (system/user/assistant messages):

```jsonl
{"messages": [
  {"role": "system", "content": "Convert natural language to ADS search query. Output JSON: {\"query\": \"...\"}"},
  {"role": "user", "content": "Query: papers by Hawking on black holes from the 1970s\nDate: 2026-01-15"},
  {"role": "assistant", "content": "{\"query\": \"author:\\\"Hawking, S\\\" abs:\\\"black holes\\\" pubdate:[1970 TO 1979]\"}"}
]}
```

### Serving

The model server (`docker/server.py`) provides an OpenAI-compatible `/v1/chat/completions` endpoint. Nectar's `NL_SEARCH_VLLM_ENDPOINT` points to it.

```bash
# Start the model server (port 8001, auto-detects MPS/CUDA/CPU)
mise run dev-model

# In another terminal, start nectar
cd ~/ads-dev/nectar && pnpm dev
```

The model server loads from HuggingFace on first startup (~30s, cached after).

```bash
# Or run directly
MODEL_NAME=adsabs/scix-nls-translator PORT=8001 python docker/server.py

# Or with Docker (GPU)
docker run --gpus all -p 8001:8000 nls-server
```

Nectar `.env.local` configuration:
```bash
NEXT_PUBLIC_NL_SEARCH=enabled
NL_SEARCH_VLLM_ENDPOINT=http://localhost:8001/v1/chat/completions
```

See [docker/README.md](docker/README.md) for deployment options.

### Training a New Model

1. Open `scripts/train_colab.ipynb` in Google Colab
2. Select an A100 GPU runtime (Colab Pro)
3. Upload `data/datasets/processed/train.jsonl`
4. Run all cells (~90 minutes)
5. Model is uploaded to HuggingFace

See [docs/fine-tuning-cli.md](docs/fine-tuning-cli.md) for detailed training documentation.

## Tech Stack

- **Model**: Qwen3-1.7B + LoRA, trained via Unsloth/TRL on Google Colab
- **Serving**: FastAPI, OpenAI-compatible API, Docker
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, TanStack Query
- **Backend**: FastAPI, Pydantic, Python 3.12
- **Validation**: ADS Search API
- **Tools**: mise (runtimes), uv (Python), Bun (Node)

## Documentation

- [Docker Deployment](docker/README.md) - Local and Docker deployment
- [Fine-Tuning & Training Guide](docs/fine-tuning-cli.md) - Model training and deployment
- [Hybrid Pipeline Architecture](docs/HYBRID_PIPELINE.md) - Deterministic NER pipeline (alternative to model)
- [ADS Search Syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax) - Official ADS docs
