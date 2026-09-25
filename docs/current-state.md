# NLS Query Translation: Current State

*Last updated: 2026-09-24*

## Overview

The system translates natural language into ADS/SciX search queries using a
**hybrid architecture**: a deterministic NER + retrieval + template-assembly
pipeline serves queries first; a fine-tuned model handles the low-confidence
tail. See [HYBRID_PIPELINE.md](HYBRID_PIPELINE.md) for the design rationale
(the earlier end-to-end model conflated NL words like "citing" with ADS
operator syntax).

## Serving Architecture

`docker/server.py` exposes an OpenAI-compatible endpoint consumed by nectar.
Requests route through the hybrid pipeline first; the fine-tuned model is the
fallback when pipeline confidence falls below the threshold.

The pipeline's intent stage has three backends. `regex` (the default) is the
rules-based `ner.py`. `jev` and `jev_gated` use Jev typed classifiers
(TypeSafe System One, `jev-1.13.0`) for the operator, enum fields, recency,
highly cited, first author, topic and which candidate words are authors; code
proposes the candidates and parses explicit years. `jev_gated` calls Jev only
when the regex finds no operator or its structural confidence is below 0.5.
`jev` (Jev on every request) is the backend to roll out; the evaluation is
[reports/jev-intent-classifier-eval.md](../reports/jev-intent-classifier-eval.md).

| Setting | Default | Meaning |
|---------|---------|---------|
| `MODEL_NAME` | `adsabs/scix-nls-translator` | Fallback model |
| `DEVICE` | auto-detect | `cuda` / `mps` / `cpu` |
| `PORT` | `8000` | Server port |
| `ROUTING_MODE` | `hybrid` | `hybrid` / `pipeline` / `model` |
| `PIPELINE_CONFIDENCE_THRESHOLD` | `0.5` | Below this routing confidence, fall back to model |
| `TELEMETRY_LOG` | unset | JSONL log: `request` rows and `intent_shadow` rows |
| `INTENT_BACKEND` | `regex` | `regex` / `jev` / `jev_gated` |
| `TYPESAFE_API_KEY` | unset | Needed by the Jev backends and shadow mode |
| `JEV_TIMEOUT_S` | `2.0` | Wall-clock bound on each System One call, in seconds, including any wait for a free connection slot |
| `JEV_CACHE_PATH` | `data/cache/jev_systemone.jsonl`; empty in Docker | Response cache file; empty disables it |
| `SHADOW_INTENT_BACKEND` | unset | `jev` / `jev_gated`, logged next to the served regex intent |
| `GOLD_EXAMPLES_PATH` | repo `data/datasets/raw/gold_examples.json` | Retrieval examples |

- **Fallback.** A Jev error or timeout serves the regex intent; the reason is
  in `debug_info.classifier_error` and the telemetry `classifier_error` field.
- **Routing confidence.** The structural score, or `min(structural, Jev
  operator confidence)` when Jev answered.
- **Rollout.** Shadow first (`INTENT_BACKEND=regex`,
  `SHADOW_INTENT_BACKEND=jev_gated`, `TELEMETRY_LOG` set), review the log with
  `scripts/summarize_intent_shadow.py`, then serve `INTENT_BACKEND=jev_gated`.
  Commands and the telemetry row fields are in
  [docker/README.md](../docker/README.md).

## Model Details

| Property | Value |
|----------|-------|
| Base Model | Qwen/Qwen3-1.7B |
| Adapter | LoRA (r=16, alpha=32) |
| Training | Unsloth + TRL on Google Colab (A100, ~90 min) |
| Training Data | ~62k pairs ([adsabs/nls-query-training-data](https://huggingface.co/datasets/adsabs/nls-query-training-data)) |
| Hosting | [adsabs/scix-nls-translator](https://huggingface.co/adsabs/scix-nls-translator) on HuggingFace |

The previous Modal-based training/serving path (H100 + vLLM endpoint) is
superseded — see the notes in
[PRD-scix-finetune-query.md](../PRD-scix-finetune-query.md).

## Pipeline Performance

From [LATENCY_BENCHMARKS.md](LATENCY_BENCHMARKS.md) (100 queries, 2026-01-21):

| Component | p50 | p95 | Target | Status |
|-----------|-----|-----|--------|--------|
| NER Extraction | 0.08ms | 0.10ms | <10ms | PASS |
| Retrieval (k=5) | 3.90ms | 6.10ms | <20ms | PASS |
| Assembly | 0.03ms | 0.04ms | <5ms | PASS |
| Full Pipeline (no LLM) | 3.87ms | 5.47ms | <50ms | PASS |

Model fallback latency depends on hosting: ~50ms (GPU), ~500ms (Apple MPS),
~2s (CPU).

## Evaluation

Two evaluation harnesses exist:

1. **Benchmark evaluation** (`mise run eval:benchmark`) — exact match, field
   assignment, operator accuracy, and syntax validity against
   `data/datasets/benchmark/benchmark_queries.json`, sliced by category.
2. **Semantic overlap evaluation** (`mise run eval:semantic`) — result-set
   overlap (Jaccard, Precision@N, Recall@N) via the ADS API; this is the
   metric the PRD targets are defined on (semantic match ≥70%, syntax
   validity ≥95%). Use `--mode server` with different `ROUTING_MODE` servers
   to compare pipeline-only vs model-only vs hybrid routing.

Evaluation artifacts are written to `data/datasets/evaluations/`. Enrichment
model evaluations live in `reports/`.

**Status:** semantic-overlap numbers for the current hybrid routing have not
been published yet — run `mise run eval:semantic` (needs `ADS_API_KEY`) and
record the results here.

## Known Weaknesses / Next Steps

1. **Publish semantic-overlap results** for pipeline vs model vs hybrid to
   validate the routing threshold (currently 0.5).
2. **Constrained decoding** for the model fallback (JSON-schema / grammar
   constrained generation) would eliminate malformed model output
   structurally instead of post-hoc filtering in `constrain.py`.
3. **Serving economics**: serve the fallback model with vLLM on GPU, or a
   GGUF int4 quantization via llama.cpp for CPU deployments.
4. **Data flywheel**: enable `TELEMETRY_LOG` in deployments; fallback-path
   and user-edited queries are the highest-value additions to training data
   and to the NER pattern set.
5. **Retrieval**: the few-shot retriever uses token overlap with hand-tuned
   boosts; evaluate an embedding index against it once semantic-overlap
   metrics are in place.
