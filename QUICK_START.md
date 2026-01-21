# Quick Start: Extended Ralph Workflow

## TL;DR - The Commands You Need

### Start the Ralph loop (everything automatic + manual steps)

```bash
cd /Users/sjarmak/nls-finetune-scix
./ralph.sh --tool amp 15
```

Ralph will guide you through all 8 stories (US-008 to US-015).

**Estimated time:** 3-4 hours total

---

## What Ralph Does

### Automatic (Ralph handles these)

✅ **US-008**: Fix operator syntax in training data (~30 min)
- Audits for bad operator patterns
- Fixes citations(), trending(), useful(), etc.
- Verifies no more bad patterns exist

### Manual (You do these, Ralph coordinates)

⏳ **US-009**: Retrain model (~45 min, mostly waiting)
- Run: `scix-finetune train --run-name v3-operators`
- Ralph will wait and ask when done

⏳ **US-010**: Deploy to Modal (~10 min)
- Run: `modal deploy serve_vllm.py`
- Ralph will verify endpoint is live

⏳ **US-011-013**: Test in UI (~70 min)
- Test 15 queries in nectar: 5 operators + 5 constraints + 5 regressions
- All must pass with valid syntax and results

⏳ **US-014**: Verify metrics (~30 min)
- Measure latency, syntax validity, correction rate
- Compare v2-4k-pairs vs v3-operators

⏳ **US-015**: Iterate if needed (~0-60 min)
- If tests fail, identify root cause
- Fix and loop back

---

## The 3 Test Phases

### Phase 1: Operator Queries (US-011)

Test these 5 in nectar NL search (http://localhost:3000):

```
1. "papers similar to 2019ApJ...887L...1K"
   Expected: similar(bibcode:"2019ApJ...887L...1K")

2. "trending papers on exoplanets"
   Expected: trending(abs:"exoplanets")

3. "papers citing gravitational waves"
   Expected: citations(abs:"gravitational waves")

4. "useful papers on dark matter"
   Expected: useful(abs:"dark matter")

5. "reviews of cosmology"
   Expected: reviews(abs:"cosmology")
```

**Pass:** All 5 return valid syntax + results > 0

### Phase 2: Constraint Validation (US-012)

Test these 5 to verify post-processing:

```
1. "ADS papers" → invalid database removed ✓
2. "refereed articles" → property:refereed kept ✓
3. "papers by Hubble" → invalid bibgroup handled ✓
4. "PhD theses" → doctype:phdthesis kept ✓
5. "data papers with open access" → both properties kept ✓
```

**Pass:** Invalid fields removed, valid ones kept, results not empty

### Phase 3: Regression Tests (US-013)

Test these 5 to confirm original bugs are fixed:

```
1. "papers by jarmak" → author:"jarmak" (not bare field) ✓
2. "papers by kelbert" → author:"kelbert" (no hallucinated initials) ✓
3. "citations from gravitational wave papers" → citations(abs:"gravitational waves") ✓
4. "trending papers on cosmology" → trending(abs:"cosmology") (quoted values) ✓
5. "papers similar to famous paper" → similar(bibcode:"...") (balanced parens) ✓
```

**Pass:** All 5 generate valid queries + return results

---

## What Gets Delivered

After Ralph finishes US-008-015:

✅ Fixed operator syntax (37 examples)
✅ Retrained model (v3-operators)
✅ Deployed to Modal
✅ Tested in UI (15 test cases passing)
✅ Performance metrics (latency, accuracy, correction rate)
✅ Before/after comparison showing improvements
✅ Iteration logic documented

---

## If Something Goes Wrong

### Test fails in US-011-014

Ralph will ask what went wrong. Two options:

**Option A: Data issue** (missing training examples, wrong format)
- Fix examples in `data/datasets/processed/all_pairs.json`
- Go back to US-008
- Ralph reruns from there

**Option B: Model issue** (wrong hyperparameters)
- Adjust `learning_rate`, `num_train_epochs`, `per_device_train_batch_size` in `train.py`
- Go back to US-009 to retrain with new params
- Ralph reruns from there

---

## Monitoring Progress

### Check what story Ralph is on

```bash
jq '.userStories[] | select(.passes == false) | {id, title}' prd.json | head -1
```

### See test results in progress.txt

```bash
tail -50 progress.txt
```

### Check Modal deployment

```bash
# Is endpoint live?
curl -s https://sjarmak--nls-finetune-serve-vllm-serve.modal.run/v1/models | jq .

# Test an inference
curl -s -X POST https://sjarmak--nls-finetune-serve-vllm-serve.modal.run/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llm","messages":[{"role":"user","content":"Query: trending papers on black holes"}],"max_tokens":64}'
```

### Check training logs

```bash
# Monitor training (if running)
modal logs scix-finetune-train
```

---

## Timeline

| Phase | Time | Status |
|-------|------|--------|
| US-008 (data fixes) | 30 min | Automatic |
| US-009 (retrain) | 45 min | Manual (mostly waiting) |
| US-010 (deploy) | 10 min | Manual |
| US-011 (test operators) | 30 min | Manual |
| US-012 (test constraints) | 20 min | Manual |
| US-013 (regression tests) | 20 min | Manual |
| US-014 (verify metrics) | 30 min | Manual |
| US-015 (iterate if needed) | 0-60 min | Manual |
| **TOTAL** | **3-4 hours** | Mixed |

---

## Files You'll Touch

| File | Action | Purpose |
|------|--------|---------|
| `all_pairs.json` | Fix (US-008) | Training data - fix operator syntax |
| `gold_examples.json` | Fix (US-008) | Reference examples - fix operator syntax |
| `train.py` | Maybe adjust (US-009) | If model params need tuning |
| `progress.txt` | Review | Track what was done |
| `prd.json` | Auto-update | Ralph marks stories as complete |
| `OPERATOR_METRICS.md` | Create (US-014) | Final metrics report |

---

## Key Infrastructure Already Ready

✅ Field constraints module (field_constraints.py)
✅ Post-processing filter (constrain_query_output)
✅ API integration (nl-search.ts)
✅ Training data fixes (536 bare fields fixed)
✅ Modal infrastructure (vLLM, GPU training)
✅ NL search UI (nectar)
✅ Evaluation harness

Everything is in place. Ralph just orchestrates the final fixes and verification.

---

## References

- **Full overview:** `WORKFLOW_SUMMARY.md`
- **Metadata filtering:** `METADATA_FILTERING_EXPLAINED.md`
- **Setup & details:** `RALPH_SETUP.md`, `RALPH_INTEGRATION_GUIDE.md`
- **PRD with acceptance criteria:** `prd.json`
- **Ralph instructions:** `prompt.md`

---

**Ready?**

```bash
./ralph.sh --tool amp 15
```

Let Ralph handle the rest! 🚀
