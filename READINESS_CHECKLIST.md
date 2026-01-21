# Ralph Extended Workflow - Readiness Checklist

## ✅ Foundation Phase Complete (US-001 to US-007)

- [x] Field constraints module created (field_constraints.py)
  - [x] 22 DOCTYPES defined
  - [x] 21 PROPERTIES defined
  - [x] 3 DATABASES defined
  - [x] 53 BIBGROUPS defined
  - [x] 8 ESOURCES defined
  - [x] 24 DATA_SOURCES defined

- [x] Query validation function implemented (validate.py)
  - [x] validate_field_constraints() checks all constrained fields
  - [x] Returns validation errors with suggestions
  - [x] 24 unit tests passing

- [x] Post-processing filter implemented (constrain.py)
  - [x] constrain_query_output() removes invalid field values
  - [x] Logs all corrections
  - [x] Handles edge cases (quoted values, OR lists, parentheses)
  - [x] 42 unit tests passing

- [x] Training data fixed (US-004)
  - [x] 536 bare bibstem fields quoted
  - [x] Other bare fields (author, abs, title, keyword) fixed
  - [x] All queries pass lint_query validation

- [x] API integration complete (US-006)
  - [x] Field constraints imported in nl-search.ts
  - [x] Validation applied after model inference
  - [x] Post-processing before ADS API call
  - [x] 856 nectar tests passing

- [x] Documentation complete (US-007)
  - [x] AGENTS.md updated with field constraint section
  - [x] Field enums reference documented
  - [x] Good vs bad training examples shown
  - [x] Common hallucinations and fixes listed

## ✅ Extended Workflow Documents Prepared (New)

- [x] prd.json extended with US-008 to US-015
  - [x] Each story has detailed acceptance criteria
  - [x] Priority levels set appropriately
  - [x] Pass/fail tracking ready

- [x] prompt.md updated
  - [x] Workflow diagram added
  - [x] Iteration gate explained

- [x] RALPH_SETUP.md expanded
  - [x] 6-phase workflow flowchart
  - [x] Timeline estimate: 3-4 hours
  - [x] Manual vs automatic steps clear

- [x] RALPH_INTEGRATION_GUIDE.md enhanced
  - [x] US-008 to US-015 specifications
  - [x] Test cases documented
  - [x] Iteration decision tree included

- [x] New documentation created
  - [x] WORKFLOW_SUMMARY.md (complete overview)
  - [x] METADATA_FILTERING_EXPLAINED.md (answers metadata question)
  - [x] QUICK_START.md (TL;DR for running Ralph)
  - [x] YOUR_QUESTIONS_ANSWERED.md (Q&A on all three topics)
  - [x] READINESS_CHECKLIST.md (this file)

## ✅ Model & Infrastructure Ready

- [x] Training data ready
  - [x] all_pairs.json exists (3025 pairs)
  - [x] gold_examples.json exists (reference examples)
  - [x] All training data valid ADS syntax

- [x] Modal infrastructure ready
  - [x] train.py configured for H100
  - [x] serve_vllm.py configured for inference
  - [x] Training volume exists (scix-finetune-runs)
  - [x] vLLM cache volume exists

- [x] Current model deployed
  - [x] v2-4k-pairs model endpoint live
  - [x] Warm latency ~0.4s
  - [x] NL search in nectar UI functional

- [x] Evaluation harness ready
  - [x] Syntax validity metrics implemented
  - [x] Result-set overlap metrics implemented
  - [x] Feature-sliced reporting available

## ✅ Test Infrastructure Ready

- [x] nectar UI ready for testing
  - [x] NL search component deployed
  - [x] Can submit natural language queries
  - [x] Can view suggested ADS queries
  - [x] Can view result counts

- [x] Modal endpoint accessible
  - [x] Can call vLLM inference endpoint
  - [x] Can query model with test inputs
  - [x] Deployment tested and working

- [x] ADS API accessible
  - [x] ADS_API_KEY configured
  - [x] Query testing available
  - [x] Result validation possible

## ✅ Operator Syntax Fixes Ready

- [x] audit_operators.py script exists
  - [x] Can find malformed operator patterns
  - [x] Can identify missing quotes
  - [x] Can identify extra parentheses

- [x] 37 bad operator examples identified
  - [x] Citations() missing quotes
  - [x] Trending() missing quotes
  - [x] Useful() missing quotes
  - [x] Similar() malformed parens
  - [x] Reviews() issues identified
  - [x] References() issues identified

- [x] Training data files accessible
  - [x] all_pairs.json can be edited
  - [x] gold_examples.json can be edited
  - [x] Changes can be validated

## 🔄 Ready to Execute

### Start Command

```bash
cd /Users/sjarmak/nls-finetune-scix
./ralph.sh --tool amp 15
```

### What Will Happen

1. **US-008** (~30 min) - Ralph automatically:
   - Audits operator syntax
   - Fixes malformed examples
   - Verifies no bad patterns remain

2. **US-009** (~45 min) - You'll run:
   - `scix-finetune train --run-name v3-operators`
   - Wait for training to complete
   - Ralph will monitor for completion

3. **US-010** (~10 min) - You'll run:
   - `modal deploy serve_vllm.py`
   - Ralph verifies endpoint live

4. **US-011 to US-013** (~70 min) - You'll test:
   - 5 operator queries
   - 5 constraint validation cases
   - 5 regression tests
   - Ralph collects results

5. **US-014** (~30 min) - You'll measure:
   - Latency metrics
   - Syntax validity %
   - Correction rate %
   - Generate metrics report

6. **US-015** (0-60 min) - Ralph will:
   - Check if all tests passed
   - If yes: mark complete, merge branch
   - If no: help identify root cause, loop back

### Success Criteria (All Must Pass)

#### US-011: Operator Query Tests
- [ ] "papers similar to..." returns `similar(bibcode:"...")`
- [ ] "trending papers..." returns `trending(abs:"...")`
- [ ] "papers citing..." returns `citations(abs:"...")`
- [ ] "useful papers..." returns `useful(abs:"...")`
- [ ] "reviews of..." returns `reviews(abs:"...")`
- [ ] All 5 return valid queries with results > 0

#### US-012: Constraint Validation Tests
- [ ] Invalid fields are removed gracefully
- [ ] Valid fields are preserved
- [ ] Results not empty from over-correction
- [ ] No regression vs v2-4k-pairs baseline

#### US-013: Regression Tests
- [ ] "papers by jarmak" → `author:"jarmak"` (not bare)
- [ ] "papers by kelbert" → no hallucinated initials
- [ ] "citations..." → correct operator syntax
- [ ] "trending..." → quoted values inside operator
- [ ] "similar..." → balanced parentheses
- [ ] All 5 return valid queries with results > 0

#### US-014: Performance Verification
- [ ] Warm latency < 0.5s
- [ ] Cold start < 1.5s
- [ ] Syntax validity > 95%
- [ ] Correction rate < 5%
- [ ] Error rate (empty results) < 5%

## 🚀 Go/No-Go Decision

### Go Criteria (Proceed with Ralph)

- [x] All foundation stories (US-001-007) complete
- [x] All documents updated and ready
- [x] Current model endpoint live
- [x] Training infrastructure verified
- [x] Testing infrastructure ready
- [x] At least 30 min uninterrupted time for US-008
- [x] Access to run manual commands (US-009-015)
- [x] Ready to monitor training for ~40 min
- [x] Ready to run tests in UI for ~70 min

### Current Status: **✅ GO**

Everything is ready. Ralph can execute the full workflow.

---

## How to Monitor Progress

### During Execution

```bash
# Check which story Ralph is on
jq '.userStories[] | select(.passes == false) | .id' prd.json | head -1

# See progress log
tail -20 progress.txt

# Check training status (while US-009 running)
modal logs scix-finetune-train

# Test endpoint manually
curl -s https://sjarmak--nls-finetune-serve-vllm-serve.modal.run/v1/models
```

### After Completion

```bash
# All stories should be complete
jq '.userStories | map(select(.passes == false)) | length' prd.json
# Should output: 0

# See final summary
tail -50 progress.txt

# Check generated metrics
cat data/datasets/OPERATOR_METRICS.md
```

---

## Next Step

When you're ready to start:

```bash
./ralph.sh --tool amp 15
```

Ralph will guide you through the rest. The workflow is fully automated for data fixes (US-008) and coordinated for all manual steps (US-009-015).

**Estimated completion time: 3-4 hours**

Good luck! 🚀
