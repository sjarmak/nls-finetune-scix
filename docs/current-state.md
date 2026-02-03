# NLS Query Fine-tuning: Current State

*Last updated: 2025-12-17*

## Overview

This document summarizes the current state of the NLS Query fine-tuning project after a complete end-to-end test.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | Qwen/Qwen3-1.7B |
| Adapter | LoRA (r=16, alpha=32) |
| Training Run | e2e-test-20251217-131656 |
| Training Data | 50-80k training pairs |
| Final Loss | 0.77 |
| Training Time | ~90 minutes (50-80k pairs) |
| Training Cost | Colab Pro (A100 GPU) |
| GPU | NVIDIA A100 (bf16) |

## Model Hosting

The trained model is uploaded to HuggingFace at `adsabs/scix-nls-translator` and can be served with vLLM:

```bash
vllm serve adsabs/scix-nls-translator --max-model-len 512
```

See [docs/fine-tuning-cli.md](fine-tuning-cli.md) for other deployment options (TGI, SageMaker).

## Evaluation Results

### Summary (50 samples)

| Metric | Fine-tuned | GPT-4o-mini | Target | Status |
|--------|------------|-------------|--------|--------|
| Syntax Valid | 98% (49/50) | 82% (41/50) | ≥95% | **PASS** |
| Semantic Match | 68% (34/50) | 44% (22/50) | ≥70% | **FAIL** (close) |
| Avg Latency | 2167ms | 999ms | ≤100ms | **FAIL** |

### Key Findings

1. **Syntax validity is excellent**: The fine-tuned model produces valid Sourcegraph queries 98% of the time, significantly better than GPT-4o-mini (82%).

2. **Semantic matching improved but below target**: At 68%, the model falls just short of the 70% target. It still outperforms GPT-4o-mini's 44%.

3. **Latency is problematic**: The model averages ~2.2 seconds per query, far above the 100ms target. Latency depends on the inference hosting setup.

### Model Behavior Observations

**Strengths:**
- Produces syntactically valid queries consistently
- Good understanding of `lang:`, `file:`, `type:` filters
- Handles commit/diff search patterns well

**Weaknesses:**
- Often adds unnecessary repo filters (defaults to `repo:github.com/sourcegraph/sourcegraph`)
- Doesn't reliably use provided candidate repositories
- Sometimes hallucinates non-existent filters (e.g., `async:yes`, `name:`)
- Uses escaped quotes in strings which may not be intended

### Sample Query Analysis

| Input | Expected | Fine-tuned Output | Correct? |
|-------|----------|-------------------|----------|
| "find Python files with async functions" | `lang:python async` | `repo:github.com/sourcegraph/ file:(test\|spec) lang:python type:symbol async:yes` | Partial |
| "commits by erik in sourcegraph last week" | `repo:^github.com/sourcegraph/sourcegraph$ type:commit after:DATE author:erik` | `repo:sourcegraph lang:javascript type:commit author:erik after:\"1 week ago\"` | Partial |
| "find config files" | `file:config*` | `file:config* case:yes` | Close |

## Recommendations

### Short-term Fixes

1. **Optimize inference**: Investigate why latency is 2x worse than GPT-4o-mini
   - Check if there's unnecessary processing
   - Consider model quantization (INT8/INT4)
   - Investigate vLLM or other fast inference backends

2. **Improve repo inference**: Training data should include examples where repo is inferred from query context

3. **Data quality**: Fix the escaped dots issue in data regeneration pipeline

### Medium-term Improvements

1. **More training data**: Current 969 examples may be insufficient. Target 2000+ diverse examples.

2. **Better evaluation set**: The semantic matching criteria may need refinement.

3. **Hyperparameter tuning**: Experiment with LoRA rank, learning rate, and epochs.

## Cost Analysis

| Operation | Cost |
|-----------|------|
| Training (Colab A100, ~90min) | Colab Pro subscription |
| Inference cost | Depends on hosting (vLLM, SageMaker, etc.) |
| GPT-4o-mini (1M queries) | ~$150 |

## Next Steps

1. Investigate latency issues (blocking)
2. Fix data regeneration pipeline (escaped dots)
3. Expand training dataset
4. A/B test against production GPT-4o-mini
