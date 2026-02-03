# Training & Deployment Guide

How the NLS system is trained, what data it uses, and how it's hosted.

## System Overview

The NLS pipeline converts natural language to ADS search queries using a hybrid approach:

1. **Rules-based NER** — regex patterns extract structured intent (no ML model)
2. **BM25 retrieval** — finds similar gold examples from a curated index
3. **Deterministic assembly** — builds valid ADS queries from validated fields
4. **Optional LLM fallback** — resolves ambiguous paper references (GPT-4o-mini)
5. **Fine-tuned model fallback** — Qwen3-1.7B for end-to-end generation when pipeline can't handle a query

The core pipeline (stages 1-3) needs no GPU and runs in <50ms. The fine-tuned models are used as fallbacks and for evaluation.

## Data

### Gold Examples (runtime requirement)

**File:** `data/datasets/raw/gold_examples.json` (4,557 examples)

Used by the retrieval stage at inference time. Each example is a curated NL-to-query pair:

```json
{
  "natural_language": "terahertz josephson echo spectroscopy cuprate superconductors",
  "ads_query": "abs:(terahertz AND Josephson AND echo AND spectroscopy AND cuprate AND superconductor*)",
  "category": "topic"
}
```

### Training Data (for model training)

**File:** `data/datasets/processed/train.jsonl` (50-80k pairs)

OpenAI chat messages format used to fine-tune the Qwen3-1.7B translator:

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Generated via the dataset agent (`scix-finetune dataset-agent run`) which uses Claude to produce NL descriptions for curated ADS queries.

### Enrichment Data (for NER model training)

**Files:** `enrichment_train.jsonl`, `enrichment_val.jsonl`

Character-level span annotations for token classification:

```json
{
  "text": "papers on exoplanets by Seager since 2020",
  "spans": [
    {"start": 10, "end": 21, "type": "topic"},
    {"start": 25, "end": 31, "type": "author"},
    {"start": 38, "end": 42, "type": "date_range"}
  ]
}
```

### Field Constraints (in code)

**File:** `packages/finetune/src/finetune/domains/scix/field_constraints.py`

Defines valid values for constrained ADS fields (doctype, property, bibgroup, collection, esources, data). Used at inference time to validate and filter extracted values.

## Training: NL Query Translator (Qwen3-1.7B)

### Requirements

- Google Colab with A100 GPU runtime (Colab Pro)
- ~90 minutes for training (50-80k pairs)
- HuggingFace account for model upload

### Steps

1. Open `scripts/train_colab.ipynb` in Google Colab
2. Select an A100 GPU runtime
3. Upload `data/datasets/processed/train.jsonl`
4. Run all cells — the notebook handles:
   - Environment setup (Unsloth, TRL, PEFT)
   - Model loading (Qwen/Qwen3-1.7B with 4-bit quantization)
   - LoRA adapter training (r=16, alpha=32, all linear layers)
   - Merging LoRA into base model
   - Testing with sample queries
   - Uploading to HuggingFace (`adsabs/scix-nls-translator`)

### Configuration

| Setting | Value |
|---------|-------|
| Base model | Qwen/Qwen3-1.7B |
| Adapter | LoRA (r=16, alpha=32) |
| Target modules | q, k, v, o, gate, up, down projections |
| Precision | bf16 (A100) |
| Epochs | 3 |
| Batch size | 8, gradient accumulation 2 |
| Learning rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Max sequence length | 512 |

## Training: Enrichment NER (SciBERT)

### Requirements

- Google Colab with GPU (T4 or A100)
- Enrichment dataset (`enrichment_train.jsonl`, `enrichment_val.jsonl`)

### Steps

1. Open `notebooks/train_enrichment_model.ipynb` in Google Colab
2. Upload enrichment data (via Google Drive or direct upload)
3. Run all cells — trains SciBERT for BIO token classification

### Configuration

| Setting | Value |
|---------|-------|
| Base model | `allenai/scibert_scivocab_uncased` |
| Task | Token classification (BIO tagging) |
| Entity types | topic, institution, author, date_range |
| Labels | 9 BIO tags (O + B/I for each type) |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 10 (with early stopping, patience=3) |
| Max sequence length | 256 |

## Hosting & Deployment

After training, models are uploaded to HuggingFace and served via the inference server.

### Pipeline-only (no GPU needed)

For the rules-based pipeline without model fallback:

```bash
uv run python scripts/serve_local_pipeline.py
```

Runs on port 8001, <50ms latency, no GPU required.

### Full server with model fallback

```bash
# Docker with GPU
docker run --gpus all -p 8000:8000 nls-server

# Or direct Python
MODEL_NAME=adsabs/scix-nls-translator python docker/server.py
```

### vLLM (production)

```bash
pip install vllm
vllm serve adsabs/scix-nls-translator --max-model-len 512
```

### Text Generation Inference

```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id adsabs/scix-nls-translator
```

### AWS SageMaker

```python
from sagemaker.huggingface import HuggingFaceModel
model = HuggingFaceModel(
    model_data="adsabs/scix-nls-translator",
    role=role,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
)
predictor = model.deploy(instance_type="ml.g5.xlarge", initial_instance_count=1)
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List models (OpenAI-compatible) |
| `/v1/chat/completions` | POST | vLLM-compatible chat endpoint |
| `/pipeline` | POST | Raw hybrid pipeline endpoint |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `adsabs/scix-nls-translator` | HuggingFace model to load |
| `DEVICE` | auto-detect | `cuda`, `mps`, or `cpu` |
| `PORT` | `8000` | Server port |
| `GOLD_EXAMPLES_PATH` | auto-detect | Path to `gold_examples.json` |

## CLI Commands

The `scix-finetune` CLI provides verification and evaluation:

```bash
scix-finetune --help

# Verification
scix-finetune verify data         # Validate training data format
scix-finetune verify all          # Run all checks

# Evaluation
scix-finetune eval baseline       # Generate GPT-4o-mini baseline
scix-finetune eval run --endpoint <url>  # Evaluate deployed model
scix-finetune eval report         # Print comparison report

# Dataset generation
scix-finetune dataset-agent run   # Generate training pairs
```

## Evaluation

### Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Syntax Valid | >=95% | Generated query uses only valid ADS operators |
| Semantic Match | >=70% | Query captures same intent as expected |
| Latency | <=100ms | End-to-end request time |

### Running Evaluation

```bash
# Generate GPT-4o-mini baseline (requires OPENAI_API_KEY)
scix-finetune eval baseline --sample 50

# Evaluate against a deployed endpoint
scix-finetune eval run --endpoint <your-endpoint-url> --sample 50

# Print comparison report
scix-finetune eval report
```
