# Testing v4-fixed-operators Model

## Status
✅ **LIVE AND TESTABLE**

Model: `v4-fixed-operators`
Endpoint: `https://sjarmak--nls-finetune-serve-vllm-serve.modal.run`
Status: Deployed on Modal H100

## Quick Test (Command Line)

```bash
# Test operator syntax via Modal endpoint directly
curl -s -X POST https://sjarmak--nls-finetune-serve-vllm-serve.modal.run/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llm",
    "messages": [
      {"role": "system", "content": "Convert to ADS query. Output ONLY query."},
      {"role": "user", "content": "papers citing black holes"}
    ],
    "max_tokens": 64,
    "temperature": 0.5
  }' | jq '.choices[0].message.content'
```

Expected: `citations(abs:"black hole")` or `citations(abs:black hole)`
NOT: `citationsauthor:` ❌ (this was the bug we fixed)

## Test Suite

### 1. Citations Operator
```
Input:  "papers citing work on black holes"
Output: citations(abs:black hole) ✓
        citations(abs:"black hole") ✓
NOT:    citationsauthor: ❌
```

### 2. Trending Operator
```
Input:  "trending papers on exoplanets"
Output: trending(abs:exoplanet) ✓
NOT:    trendingabs: ❌
```

### 3. Similar Operator
```
Input:  "papers similar to JWST observations"
Output: similar(abs:JWST abs:observation) ✓
NOT:    similarabs: ❌
```

### 4. Useful Operator
```
Input:  "useful references for dark matter"
Output: useful(abs:matter) ✓
NOT:    usefulabs: ❌
```

### 5. References Operator
```
Input:  "papers referenced by gravitational wave work"
Output: references(abs:"gravitational wave") ✓
NOT:    referencesabs: ❌
```

## Browser Testing (if nectar UI is running)

1. **Start UI**:
   ```bash
   cd ~/ads-dev/nectar
   npm run dev
   ```

2. **Open browser**: http://localhost:3000

3. **Find NL Search Component**: Look for "Natural Language Search" or similar

4. **Type test queries**:
   - "papers citing black holes" → Should generate `citations(...)`
   - "trending exoplanets" → Should generate `trending(...)`
   - "papers by Einstein" → Should generate author query

5. **Verify**: 
   - Operators use parentheses `(...)`
   - No malformed syntax like `operatorfield:`

## API Testing (nectar backend)

If UI API is running at `http://localhost:8000`:

```bash
curl -X POST http://localhost:8000/api/nl-search \
  -H 'Content-Type: application/json' \
  -d '{"query":"papers citing gravitational waves"}' | jq '.queries'
```

Expected: Array of query suggestions, each with proper operator syntax

## What Was Fixed

**Root Cause**: Training data had 28 operator calls with multiple fields but no separator (AND/OR), which taught the model that concatenating operator names with field names was valid.

**Training Data Fix**: Added 34 examples where multiple fields in operators are now properly separated with AND.

**Code Fix**: Added `_fix_malformed_operators()` post-processing in `constrain.py` to reconstruct any remaining malformed operators as a safety net.

## Files Involved

- **Model**: Lives in Modal `/runs/v4-fixed-operators/merged`
- **Training Data**: 
  - `data/datasets/processed/all_pairs.json` (17 examples fixed)
  - `data/datasets/raw/gold_examples.json` (17 examples fixed)
- **Post-processing Filter**: 
  - `packages/finetune/src/finetune/domains/scix/constrain.py` (new `_fix_malformed_operators()`)
- **UI Integration** (if available):
  - `~/ads-dev/nectar/src/pages/api/nl-search.ts`
  - `~/ads-dev/nectar/src/lib/field-constraints.ts`

## Monitoring

Watch Modal dashboard for inference logs:
- https://modal.com/apps/sjarmak/main/deployed/nls-finetune-serve-vllm

Check request latency:
```bash
# Warm inference should be ~0.3-0.5 seconds
time curl -s https://sjarmak--nls-finetune-serve-vllm-serve.modal.run/v1/models | jq -r '.[0].id'
```

## Known Limitations

1. **Multiple fields without AND**: Some model outputs like `similar(abs:field1 abs:field2)` don't have AND between them. This may be valid ADS syntax depending on the operator, but is suboptimal for clarity.

2. **Field value truncation**: Some generated values are truncated (e.g., `abs:bl` instead of `abs:black`). This is a tokenization/generation issue that would need more training examples to fix.

3. **Prose mode**: When not given a system prompt, model reverts to prose explanations instead of queries. This is expected behavior for a language model.

## Next Steps

If issues are found:
1. Add more training examples for edge cases
2. Increase operator coverage from 4.4% to 10%+
3. Consider using constrain.py filter in Python API as well
4. Retrain with constrain-aware loss weighting
