# Fine-Tuning CLI

CLI for fine-tuning Qwen3-1.7B on Modal with H100 GPU.

## Installation

```bash
# Install all dependencies including CLI (from repo root)
mise run install

# Verify installation
nls-finetune --help
```

## Environment Setup

The CLI automatically loads environment variables from `.env` in your current directory or parent directories.

### Required Environment Variables

Copy `.env.example` to `.env` and configure the required variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Variables by Command

| Command | Required Variables | Notes |
|---------|-------------------|-------|
| `verify env` | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Checks Modal authentication |
| `upload-data` | Modal tokens | Uploads to Modal volume |
| `dry-run` | Modal tokens + HuggingFace secret in Modal | Tests training pipeline |
| `train` | Modal tokens + HuggingFace secret in Modal | Runs full training |
| `merge` | Modal tokens | Merges LoRA adapter |
| `deploy` | Modal tokens | Deploys inference endpoint |
| `eval baseline` | `OPENAI_API_KEY` | Calls GPT-4o-mini API |
| `eval run` | None | Uses deployed Modal endpoint |
| `eval report` | None | Reads local evaluation files |

### Modal Secrets

Some variables must be configured as Modal secrets (not in `.env`):

```bash
# HuggingFace token (for model downloads)
modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxxxx

# Optional: Weights & Biases
modal secret create wandb-secret WANDB_API_KEY=xxxxx
```

Verify secrets are configured:
```bash
modal secret list
```

## Commands

```bash
# Help
nls-finetune --help

# Verification
nls-finetune verify env          # Check Modal setup
nls-finetune verify data         # Validate training data
nls-finetune verify config       # Check config files
nls-finetune verify all          # Run all checks

# Training
nls-finetune upload-data         # Upload data to Modal volume
nls-finetune dry-run train       # Test pipeline (3 steps)
nls-finetune train --run-name X  # Full training (~12 min)
nls-finetune merge --run-name X  # Merge LoRA adapter (required)
nls-finetune deploy --run-name X # Deploy inference endpoint

# Evaluation
nls-finetune eval baseline       # Generate GPT-4o-mini baseline
nls-finetune eval run            # Evaluate fine-tuned model
nls-finetune eval report         # Print comparison report

# Status
nls-finetune list-runs           # List training runs
nls-finetune status              # Show training status
```

## Live Endpoint

After deployment, the vLLM-based model is available at:
```
https://sourcegraph--nls-finetune-serve-vllm-serve.modal.run
```

Test with (OpenAI-compatible API):
```bash
curl -X POST https://sourcegraph--nls-finetune-serve-vllm-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [
      {"role": "system", "content": "Convert natural language to Sourcegraph query."},
      {"role": "user", "content": "find Python files with async functions"}
    ],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Phased Implementation

Work through phases sequentially. Each phase has verification before proceeding.

### Phase 1: Environment (`verify env`)

Checks:
- Modal CLI installed and authenticated
- Modal workspace accessible
- HuggingFace secret exists
- W&B secret exists (optional)

### Phase 2: Data (`verify data`)

Checks:
- train.jsonl valid JSONL with correct schema
- val.jsonl valid JSONL with correct schema
- Messages have role/content fields

### Phase 3: Config (`verify config`)

Checks:
- TRL SFTTrainer configuration valid
- LoRA parameters correct (r=16, alpha=32)

### Phase 4: Volumes (`verify volumes`, `upload-data`)

Checks:
- Modal volumes accessible
- Data uploads successfully

### Phase 5: Model Loading (`dry-run model`)

Checks:
- GPU available (H100)
- Model downloads and loads
- Forward pass succeeds

### Phase 6: Training (`dry-run train`)

Checks:
- Full pipeline works for 3 steps
- Loss decreases
- Checkpoint saves

### Phase 7: Full Training (`train`)

Runs complete fine-tuning (~12 min, ~$1.50).

### Phase 8: Deploy (`merge`, `deploy`)

Merge LoRA adapter and deploy inference endpoint.

## Configuration

### Training Stack: TRL SFTTrainer + PEFT LoRA

Key settings:
- **Base model**: Qwen/Qwen3-1.7B
- **Adapter**: LoRA (r=16, alpha=32, target_modules=all-linear)
- **Training**: 3 epochs, batch_size=4, gradient_accumulation=2
- **Learning rate**: 2e-4 with cosine scheduler
- **Precision**: bfloat16
- **GPU**: Single H100 (80GB) - more than sufficient for 1.7B model with LoRA

### Why TRL Instead of Axolotl

We switched from Axolotl to TRL SFTTrainer because:
1. **Simpler setup**: No complex YAML configuration needed
2. **Better compatibility**: Works reliably with Qwen3 models
3. **Lighter weight**: Fewer dependencies, faster container builds
4. **Sufficient for our use case**: LoRA fine-tuning doesn't need Axolotl's advanced features

## Data Format

Training data must be OpenAI messages format:

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Find Python files"}, {"role": "assistant", "content": "lang:python"}]}
```

## Architecture Decisions

| Choice | Rationale |
|--------|-----------|
| TRL SFTTrainer | Simpler than Axolotl, reliable Qwen3 support |
| LoRA (not QLoRA) | H100 has enough VRAM; faster, higher quality |
| Single H100 | 80GB VRAM is plenty for 1.7B model with LoRA |
| r=16, alpha=32 | Standard LoRA config, good balance of quality/efficiency |
| bfloat16 | Native H100 support, no precision loss |

## Estimated Resources

| Metric | Value |
|--------|-------|
| Training time | ~12 minutes |
| GPU usage | ~10-15 GB / 80 GB |
| Cost | ~$1.50 |

## Troubleshooting

### OOM Errors
- Reduce `per_device_train_batch_size` in train.py
- Enable gradient checkpointing (add to model config)

### Slow Training
- Check GPU utilization with Modal dashboard
- Verify bfloat16 is enabled

### Modal Issues
- Run `modal profile list` to verify auth
- Check secrets with `modal secret list`
- View logs: `modal container list --json` then `modal container logs <id>`

### Model Not Loading
- Check Modal volume has merged model: `modal volume ls nls-query-runs`
- Verify HuggingFace secret is set: `modal secret list`

## Training Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  upload-data │ ──▶ │    train    │ ──▶ │    merge    │
│  (to Modal)  │     │ (TRL+LoRA)  │     │ (adapter)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   deploy    │
                                        │ (endpoint)  │
                                        └─────────────┘
```

## Evaluation

### Running Evaluation

```bash
# Generate GPT-4o-mini baseline (requires OPENAI_API_KEY)
nls-finetune eval baseline --sample 50

# Evaluate fine-tuned model against baseline
nls-finetune eval run --sample 50

# Print comparison report
nls-finetune eval report
```

### Metrics Explained

The evaluation measures three key metrics:

#### 1. Syntax Validity (Target: ≥95%)

**What it measures:** Whether the generated query uses only valid Sourcegraph search operators.

**How it works:**
- Extracts all `operator:` patterns from the query (e.g., `repo:`, `lang:`, `type:`)
- Compares against the list of valid Sourcegraph operators
- Ignores quoted strings and URLs to avoid false positives

**Valid operators:** `repo`, `file`, `lang`, `language`, `l`, `content`, `type`, `case`, `rev`, `fork`, `archived`, `visibility`, `context`, `author`, `committer`, `before`, `after`, `message`, `count`, `timeout`, `patterntype`, `select`

**Hallucinations:** Invalid operators the model might generate:
- `path:` → Use `file:` instead
- `branch:` → Use `rev:` or `@branch` syntax
- `extension:` / `ext:` → Use `file:\.ext$`
- `in:`, `is:`, `sort:`, `stars:` → GitHub syntax, not supported

#### 2. Semantic Match (Target: ≥70%)

**What it measures:** Whether the generated query captures the same intent as the expected query.

**How it works:**
1. Extracts operators from both expected and actual queries
2. Computes operator overlap: `|expected ∩ actual| / |expected|`
3. Query is considered semantically matching if ≥70% of expected operators are present

**Example:**
```
Expected: repo:sourcegraph/sourcegraph lang:go type:symbol
Actual:   repo:sourcegraph/sourcegraph lang:go content:"func"

Expected operators: {repo, lang, type}
Actual operators:   {repo, lang, content}
Overlap: {repo, lang} = 2/3 = 67% → FAIL (below 70%)
```

**Edge cases:**
- If no operators expected, falls back to keyword overlap check
- Additional operators in actual query are allowed (doesn't penalize)

#### 3. Latency (Target: ≤100ms)

**What it measures:** End-to-end time from API request to response.

**Includes:**
- Network latency to Modal endpoint
- Model inference time
- JSON serialization/deserialization

**Notes:**
- First request may be slow (cold start: 10-30 seconds)
- Warm requests should be measured for accurate comparison
- 100ms target is aspirational; current models typically 1-2 seconds

### Evaluation Report

The `eval report` command shows a comparison table:

```
┌────────────────┬─────────────┬─────────────┬────────┬────────┐
│ Metric         │  Fine-tuned │    Baseline │ Target │ Status │
├────────────────┼─────────────┼─────────────┼────────┼────────┤
│ Syntax Valid   │ 49/50 (98%) │ 41/50 (82%) │   ≥95% │  PASS  │
│ Semantic Match │ 34/50 (68%) │ 22/50 (44%) │   ≥70% │  FAIL  │
│ Avg Latency    │      2167ms │       999ms │ ≤100ms │  FAIL  │
└────────────────┴─────────────┴─────────────┴────────┴────────┘
```

### Verdict Categories

For each example, the evaluation computes a verdict:

| Verdict | Meaning |
|---------|---------|
| `tie` | Both models got it right (syntax valid + semantic match) |
| `fine_tuned_better` | Fine-tuned correct, baseline wrong |
| `baseline_better` | Baseline correct, fine-tuned wrong |
| `both_wrong` | Both models failed |

### Evaluation Data

- **Source:** `data/datasets/processed/val.jsonl` (validation set)
- **Results:** `data/datasets/evaluations/eval-YYYYMMDD-HHMMSS.json`
- **Baseline:** `data/datasets/evaluations/baseline-gpt-4o-mini.json`
