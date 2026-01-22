# Pipeline Latency Benchmarks

Performance benchmarks for the hybrid NER pipeline.

## Summary

- **Queries tested**: 100
- **Date**: 2026-01-21 22:07:38

## Targets

| Component | Target (p95) |
|-----------|-------------|
| NER Extraction | < 10ms |
| Retrieval (k=5) | < 20ms |
| Assembly | < 5ms |
| Full Pipeline (local, no LLM) | < 50ms |
| Full Pipeline (Modal, no LLM) | < 200ms |
| E2E with LLM fallback | < 1000ms |

## Results

| Component                   |    p50 |    p95 |    p99 |   Mean |   Min |    Max | Status   |
|-----------------------------|--------|--------|--------|--------|-------|--------|----------|
| NER Extraction            |    0.08 |    0.10 |    0.70 |    0.09 |   0.06 |    0.71 | ✅ PASS   |
| Retrieval (k=5)           |    3.90 |    6.10 |    6.50 |    4.04 |   1.00 |    6.50 | ✅ PASS   |
| Assembly                  |    0.03 |    0.04 |    1.02 |    0.04 |   0.02 |    1.03 | ✅ PASS   |
| Full Pipeline (no LLM)    |    3.87 |    5.47 |    6.48 |    3.90 |   1.34 |    6.49 | ✅ PASS   |

## Overall: ✅ **PASS**

Pipeline p95 latency: **5.47ms** (target: < 50ms local)

## Notes

- **Local benchmarks**: Run on development machine without network latency
- **Modal benchmarks**: Add ~100-150ms for cold start, ~20ms for warm requests
- **LLM fallback**: Only triggered for ambiguous paper references (rare)
- **Index loading**: First request includes index load time (~500ms cold start)

## Component Breakdown

### NER Extraction
- Rules-based pattern matching
- No external dependencies
- Scales O(n) with input length

### Retrieval
- BM25-like scoring over 4000+ gold examples
- Preloaded index (no per-request file I/O)
- Scales O(n) with index size, O(k) with result count

### Assembly
- Deterministic template composition
- FIELD_ENUMS validation
- Scales O(1) with input complexity

## CI Integration

Add this check to CI:

```bash
python scripts/benchmark_pipeline.py --queries 100
# Fails with exit code 1 if p95 > 500ms
```