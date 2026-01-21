# Research: Query Expansion via Embeddings + Database Constraints

## Executive Summary

This document investigates two approaches for improving natural language to ADS query translation:
1. **Semantic embeddings** for synonym expansion
2. **Database constraints** for author/title validation

**Recommendation:** Use ADS autocomplete API for author validation; defer embedding-based expansion until latency requirements are met.

## Current State

The fine-tuned Qwen3-1.7B model achieves:
- 93% token accuracy
- ~0.4s inference latency (warm)
- Post-processing handles common issues (author initials, field normalization, abbreviations)

### Remaining Issues
- Model sometimes invents author initials not in input
- Synonym coverage limited to training data
- No validation that authors/objects exist in ADS

## Approach 1: Semantic Embeddings for Synonym Expansion

### Concept
Use sentence embeddings to find semantically similar terms and expand queries.

Example:
- User: "papers about planet formation"
- Expansion: `abs:(planet formation OR protoplanetary disk OR accretion)`

### Implementation Options

| Method | Latency | Quality | Complexity |
|--------|---------|---------|------------|
| Sentence-BERT local | ~50ms | High | Medium |
| OpenAI embeddings API | ~100-200ms | Very High | Low |
| Pre-computed synonym table | ~1ms | Medium | Low |

### Latency Analysis

Current pipeline: User → NLSearch → Modal (~400ms) → Result count (~100ms)

Adding embedding lookup:
- **Best case** (pre-computed table): +1-5ms — **Acceptable**
- **Sentence-BERT**: +50ms — **Acceptable**
- **External API**: +100-200ms — **Borderline**

### Recommendation
Start with pre-computed synonym table for common astronomy terms:
```json
{
  "black hole": ["black hole", "BH", "singularity", "event horizon"],
  "planet formation": ["planet formation", "protoplanetary disk", "accretion"],
  "supernova": ["supernova", "SN", "stellar explosion", "type Ia"]
}
```

Expand to Sentence-BERT if synonym table proves insufficient.

## Approach 2: Database Constraints for Validation

### Concept
Validate that authors, objects, and journals exist in ADS before returning query.

### ADS Autocomplete API

ADS provides an autocomplete endpoint that could validate/correct entities:

```
GET https://api.adsabs.harvard.edu/v1/autocomplete?term=hawking&field=author
```

Returns:
```json
{
  "suggestions": [
    "Hawking, Stephen",
    "Hawking, S W",
    "Hawking, Thomas"
  ]
}
```

### Latency Impact

| Validation | Latency | Usefulness |
|------------|---------|------------|
| Author autocomplete | ~50-100ms | High — prevents invalid authors |
| Object autocomplete | ~50-100ms | Medium — SIMBAD names vary |
| Journal autocomplete | ~50-100ms | Low — users rarely specify journals |

### Recommendation
**Author validation only** via ADS autocomplete:
1. Extract author names from generated query
2. Validate each against ADS autocomplete
3. If no match, try fuzzy match from suggestions
4. If still no match, use original (let user see 0 results)

### Implementation Sketch

```typescript
// In /api/nl-search.ts
async function validateAuthors(query: string): Promise<string> {
  const authorMatch = query.match(/author:"([^"]+)"/g);
  if (!authorMatch) return query;
  
  for (const match of authorMatch) {
    const name = match.slice(8, -1); // Extract name from author:"name"
    const suggestions = await fetchAutocomplete(name, 'author');
    if (suggestions.length && !suggestions.includes(name)) {
      // Replace with best match
      query = query.replace(match, `author:"${suggestions[0]}"`);
    }
  }
  return query;
}
```

## Concerns: Do DB Constraints Overwhelm the Model?

### Risk
Adding real-time database validation could:
1. Increase latency beyond acceptable threshold
2. Create race conditions with multiple parallel validations
3. Add failure modes (API timeouts, rate limits)

### Mitigation
1. **Timeout**: Set 200ms hard limit for validation; return unvalidated if exceeded
2. **Cache**: Cache autocomplete results (author names don't change often)
3. **Optional**: Make validation optional via query param `?validate=true`

## Alternative Approaches Considered

### 1. Fine-tune with Entity Lists
Include author lists in system prompt:
```
Known authors: Hawking, S; Einstein, A; Penrose, R; ...
```

**Rejected:** Context length limits, model still hallucinates.

### 2. RAG for Query Examples
Retrieve similar NL→query examples at inference time.

**Rejected:** Adds 100-200ms latency, marginal quality improvement.

### 3. Constrained Decoding
Force model to only output valid field names using grammar-constrained decoding.

**Deferred:** Requires vLLM config changes; current post-processing handles most issues.

## Recommended Next Steps

### Phase 1: Low-Hanging Fruit (1-2 days)
1. [ ] Add pre-computed synonym table for top 50 astronomy terms
2. [ ] Integrate ADS author autocomplete with 200ms timeout
3. [ ] Cache autocomplete results for 1 hour

### Phase 2: If Needed (1 week)
4. [ ] Replace synonym table with Sentence-BERT for dynamic expansion
5. [ ] Add object name validation via SIMBAD
6. [ ] Measure impact on result-set overlap metrics

### Phase 3: Long-term
7. [ ] Consider constrained decoding for field names
8. [ ] Explore RAG if quality still insufficient

## Metrics to Track

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Author query validity | ~85% | 95% | Autocomplete validation |
| Mean result-set overlap | TBD | >0.5 Jaccard | A/B test |
| E2E latency (p95) | ~500ms | <1s | Add monitoring |

## Conclusion

For immediate improvements, **ADS autocomplete API** for author validation is the best ROI:
- Low latency impact (~50ms)
- High practical value (prevents invalid author queries)
- No model changes required

Embedding-based synonym expansion should wait until core issues are resolved and user feedback identifies synonym coverage as a priority.
