# E2E Testing Notes

*Date: 2025-12-17*

This document captures issues, bugs, and observations from end-to-end testing of the fine-tuning pipeline.

## Critical Issues

### 1. Data Regeneration Pipeline Broken

**Severity:** High

**Description:** Running `mise run generate-data` produces only 118 valid examples instead of the expected ~1000+. The validation rejects 432/550 queries due to "escaped dots in github.com".

**Root Cause:** The extraction script (`scripts/extract_queries.py`) URL-decodes queries from BigQuery that contain regex escapes (`github\.com`). The validation script (`scripts/validate_dataset.py`) rejects these as invalid on line 70-71:

```python
if "github\\.com" in query:
    issues.append("has escaped dots in github.com (should be unescaped in domain)")
```

**Problem:** Both forms are valid in Sourcegraph:
- `repo:^github.com/org/repo$` (unescaped - preferred)
- `repo:^github\.com/org/repo$` (escaped - technically correct regex)

The gold examples use unescaped dots, but extracted real-world queries have escaped dots.

**Fix Options:**
1. Normalize queries in extraction to remove unnecessary escapes
2. Update validation to accept both forms
3. Add a preprocessing step to clean up queries

**Workaround:** Use the existing processed data (restored from git).

---

### 2. Modal Web Endpoint Limit

**Severity:** Low (documentation issue)

**Description:** Deployment can fail with "reached limit of 8 web endpoints" on free/starter Modal plans.

**Error:**
```
Deployment failed: reached limit of 8 web endpoints (# already deployed => 8, # in this app => 1).
```

**Workaround:** Stop old Modal apps to free up endpoint quota:
```bash
modal app list  # Find old apps
modal app stop <app-id>
```

**Documentation:** Add this limitation to `docs/fine-tuning-cli.md`.

---

## Model Quality Issues

### 3. Model Doesn't Infer Repository from Context

**Severity:** Medium

**Description:** When a repository is implied in the query (e.g., "javascript files in react-router"), the model often uses a wrong repository (usually `github.com/sourcegraph/sourcegraph`).

**Example:**
```json
// Input
{"query": "javascript files in react-router"}

// Output
{"sourcegraph_query": "repo:github.com/facebook/react type:symbol file:*.js ..."}
```

The model used `facebook/react` instead of inferring `remix-run/react-router` from the query.

**Possible Cause:** Training data may not have enough examples of inferring repos from query context.

---

### 4. Model Hallucinates Invalid Filters

**Severity:** Medium

**Description:** The model sometimes generates non-existent Sourcegraph filters.

**Examples:**
- `async:yes` - doesn't exist
- `content:"..."` - not a filter (should just be text)
- `name:"..."` - incorrect usage

---

### 5. Latency Far Above Target

**Severity:** High

**Description:** Average latency is ~2167ms, far exceeding the 100ms target. Even GPT-4o-mini is faster at ~1000ms.

**Breakdown:**
- Cold start: ~24 seconds
- Warm queries: ~1.4-2.0 seconds

**Possible Causes:**
1. Modal function cold starts
2. Model loading from volume each request
3. No optimization (INT8 quantization, etc.)
4. Network latency to Modal

**Investigation Needed:**
- Profile where time is spent
- Compare against local inference
- Test with quantized model

---

## Documentation Gaps

### 6. No Error Recovery Documentation

**Location:** `docs/fine-tuning-cli.md`

**Issue:** No guidance on what to do when things fail (endpoint limits, training OOM, etc.)

---

## Minor Issues / Tech Debt

### 7. Progress Output Mixed with curl Stats

When running curl commands, the transfer stats are mixed into the output, making it hard to parse responses.

### 8. Axolotl Config Still Referenced

The `finetune/config/axolotl.yaml` file exists but the codebase uses TRL SFTTrainer. This could confuse developers.

### 9. No Training Run Cleanup

Old training runs accumulate in Modal volumes. There's no cleanup command.

```bash
# Current runs (some have 0 checkpoints)
run-20251215-161757 (0 checkpoints)
run-20251215-162509 (0 checkpoints)
run-20251216-130527 (0 checkpoints)
...
```

---

## Recommendations by Priority

### P0 (Blocking)
1. Fix latency - currently unusable for production
2. Fix data regeneration pipeline

### P1 (Important)
3. Document endpoint limits
4. Improve candidate utilization in training

### P2 (Nice to Have)
5. Add training run cleanup command
6. Remove/update axolotl.yaml
7. Improve error messages with recovery steps

---

## Fixed Issues

The following issues were identified during E2E testing and have been resolved:

- ~~Environment Variable Loading for CLI~~ - Fixed: Added python-dotenv auto-loading
- ~~Missing .env Setup Instructions~~ - Fixed: Added Environment Setup section to docs
- ~~Evaluation Metrics Not Explained~~ - Fixed: Added comprehensive Evaluation section to docs
