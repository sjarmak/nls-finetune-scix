# Fine-Tuning Guide

Training Qwen3-1.7B on Google Colab with Unsloth + LoRA.

## Training via Colab

The recommended way to train the model is using the provided Colab notebook:

**Notebook:** `scripts/train_colab.ipynb`

### Requirements

- Google Colab with GPU runtime (T4 is sufficient)
- ~30 minutes for training
- HuggingFace account for model upload

### Steps

1. Open `scripts/train_colab.ipynb` in Google Colab
2. Select a GPU runtime (T4 or better)
3. Upload `data/datasets/processed/train.jsonl` when prompted
4. Run all cells — the notebook handles:
   - Environment setup (Unsloth, TRL, PEFT)
   - Model loading (Qwen/Qwen3-1.7B with 4-bit quantization)
   - LoRA adapter training (r=16, alpha=32, all linear layers)
   - Merging LoRA into base model
   - Testing with sample queries
   - Uploading to HuggingFace (`adsabs/scix-nls-translator`)

### Training Configuration

| Setting | Value |
|---------|-------|
| Base model | Qwen/Qwen3-1.7B |
| Adapter | LoRA (r=16, alpha=32) |
| Target modules | q, k, v, o, gate, up, down projections |
| Precision | fp16 (T4 compatible) |
| Epochs | 3 |
| Batch size | 8, gradient accumulation 2 |
| Learning rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Max sequence length | 512 |

## Hosting the Model

After training, the merged model is uploaded to HuggingFace at `adsabs/scix-nls-translator`.

### vLLM (recommended)

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

## CLI Commands

The `scix-finetune` CLI provides verification and evaluation commands:

```bash
# Help
scix-finetune --help

# Verification
scix-finetune verify data         # Validate training data format
scix-finetune verify all          # Run all checks

# Evaluation
scix-finetune eval baseline       # Generate GPT-4o-mini baseline
scix-finetune eval run --endpoint <url>  # Evaluate deployed model
scix-finetune eval report         # Print comparison report
```

## Data Format

Training data must be OpenAI chat messages format:

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Find Python files"}, {"role": "assistant", "content": "lang:python"}]}
```

## Evaluation

### Running Evaluation

```bash
# Generate GPT-4o-mini baseline (requires OPENAI_API_KEY)
scix-finetune eval baseline --sample 50

# Evaluate fine-tuned model against baseline
scix-finetune eval run --endpoint <your-endpoint-url> --sample 50

# Print comparison report
scix-finetune eval report
```

### Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Syntax Valid | >=95% | Generated query uses only valid operators |
| Semantic Match | >=70% | Query captures same intent as expected |
| Latency | <=100ms | End-to-end request time |

## Architecture Decisions

| Choice | Rationale |
|--------|-----------|
| Unsloth + LoRA | 2x faster training, 70% less VRAM on free Colab T4 |
| fp16 precision | T4 GPU compatible (bf16 requires Ampere+) |
| r=16, alpha=32 | Standard LoRA config, good quality/efficiency balance |
| HuggingFace hosting | Model accessible for any deployment target |
| vLLM serving | High-throughput OpenAI-compatible inference |
