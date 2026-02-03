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
| `OPENAI_API_KEY` | GPT-4o-mini comparison | [platform.openai.com](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Dataset generation | [console.anthropic.com](https://console.anthropic.com/) |

### Setup by Role

| I want to... | Keys needed |
|--------------|-------------|
| **Develop the web UI** | None (uses mock data) |
| **Run the full stack locally** | `ADS_API_KEY` |
| **Train a new model** | Google Colab + HuggingFace account |
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

The project uses a **hybrid NER + template assembly pipeline** to convert natural language to ADS queries. This replaces end-to-end generation, which produced malformed operator syntax.

```
User NL → [NER Extraction] → IntentSpec → [Retrieval] → [Assembler] → Valid Query
                                               ↑
                                     gold_examples.json (4,557 examples)
```

### Pipeline Stages

1. **NER Extraction** (`ner.py`) — Rules-based extraction using regex patterns and synonym maps. Extracts authors, years, topics, constrained fields (doctype, property, bibgroup), and operators (citations, references, trending, etc.) into a structured `IntentSpec`. No ML model — pure pattern matching with strict operator gating.

2. **Few-shot Retrieval** (`retrieval.py`) — BM25-style token overlap scoring against `data/datasets/raw/gold_examples.json` (4,557 curated examples). Boosts matches on operators, doctypes, and bibgroups. Returns top-5 similar examples for assembly guidance. Runs in <20ms.

3. **Template Assembly** (`assembler.py`) — Deterministically builds ADS queries from validated `IntentSpec` fields. All enum values validated against `field_constraints.py`. Wraps with operators if detected. Runs constraint filter as final safety net.

4. **Optional LLM Resolution** (`resolver.py`) — Only triggered for ambiguous paper references with operators (e.g., "papers citing the famous black hole paper"). Searches ADS first, falls back to GPT-4o-mini if needed. Most queries skip this entirely.

### Data Requirements

| File | Purpose | Size |
|------|---------|------|
| `data/datasets/raw/gold_examples.json` | Retrieval index for few-shot guidance | 4,557 examples |
| `data/datasets/processed/train.jsonl` | Training data for fine-tuned model | 50-80k pairs |
| `packages/finetune/src/finetune/domains/scix/field_constraints.py` | Valid ADS field enum values | In code |

### Models

| Component | Model | Source | Required at Runtime |
|-----------|-------|--------|-------------------|
| NER extraction | Rules-based (no model) | — | No model needed |
| Retrieval scoring | BM25-style (no model) | — | No model needed |
| Enrichment NER | SciBERT (`allenai/scibert_scivocab_uncased`) | Trained via `notebooks/train_enrichment_model.ipynb` | Optional |
| LLM translator | Qwen3-1.7B fine-tuned | Trained via `scripts/train_colab.ipynb`, hosted on HuggingFace | For model-based fallback |
| LLM resolver | GPT-4o-mini | OpenAI API | Only for ambiguous references |

### Serving

The pipeline is served via `docker/server.py` (Docker/GPU) or `scripts/serve_local_pipeline.py` (local development):

```bash
# Pipeline-only (no GPU needed, <50ms latency)
uv run python scripts/serve_local_pipeline.py

# Full server with model fallback (GPU)
docker run --gpus all -p 8000:8000 nls-server
```

Endpoints: `POST /v1/chat/completions` (OpenAI-compatible), `POST /pipeline` (raw pipeline), `GET /health`

See [docker/README.md](docker/README.md) for deployment options and [docs/HYBRID_PIPELINE.md](docs/HYBRID_PIPELINE.md) for detailed architecture.

### Training

Two models can be trained, both via Google Colab notebooks:

| Model | Notebook | GPU | Data |
|-------|----------|-----|------|
| **NL Query Translator** (Qwen3-1.7B + LoRA) | `scripts/train_colab.ipynb` | A100 (~90 min) | `train.jsonl` (50-80k pairs) |
| **Enrichment NER** (SciBERT token classification) | `notebooks/train_enrichment_model.ipynb` | T4 or A100 | `enrichment_train.jsonl` |

See [docs/fine-tuning-cli.md](docs/fine-tuning-cli.md) for training details and deployment options.

## Tech Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, TanStack Query
- **Backend**: FastAPI, Pydantic, Python 3.12
- **NL Pipeline**: Rules-based NER, BM25 retrieval, deterministic assembly
- **Training**: Google Colab (A100 GPU), Unsloth, TRL, LoRA, Qwen3-1.7B
- **Enrichment NER**: SciBERT fine-tuned for BIO token classification
- **Validation**: ADS Search API
- **Serving**: Docker, vLLM, FastAPI
- **Tools**: mise (runtimes), uv (Python), Bun (Node)

## Documentation

- [Development Guide](DEVELOPMENT.md) - Architecture, workflows
- [Hybrid Pipeline Architecture](docs/HYBRID_PIPELINE.md) - NER + retrieval + assembly pipeline
- [Fine-Tuning & Training Guide](docs/fine-tuning-cli.md) - Model training and deployment
- [Docker Deployment](docker/README.md) - Local and beta deployment
- [Latency Benchmarks](docs/LATENCY_BENCHMARKS.md) - Pipeline performance metrics
- [ADS Search Syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax) - Official ADS docs
