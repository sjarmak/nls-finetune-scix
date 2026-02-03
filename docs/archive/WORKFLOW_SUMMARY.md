# Ralph End-to-End Workflow Summary

## Status Overview

This document summarizes the complete Ralph workflow for fixing training data issues, retraining the model, deploying it, and testing it end-to-end in the SciX UI.

### ✅ COMPLETE: Foundation (US-001 to US-007)

Ralph successfully created the entire field constraints and validation infrastructure:

| Story | Title | Status |
|-------|-------|--------|
| US-001 | Create ADS field constraints module | ✅ DONE |
| US-002 | Add query constraint validation function | ✅ DONE |
| US-003 | Implement post-processing filter for model output | ✅ DONE |
| US-004 | Audit and fix remaining bare fields in training data | ✅ DONE |
| US-005 | Create training data quality report | ✅ DONE |
| US-006 | Integrate field constraints into API validation layer | ✅ DONE |
| US-007 | Document field constraint patterns in AGENTS.md | ✅ DONE |

**What was implemented:**
- `field_constraints.py`: 22 doctypes, 21 properties, 3 databases, 53 bibgroups, 8 esources, 24 data sources
- `constrain_query_output()`: Post-processing filter that removes invalid field values before ADS API calls
- API integration: Validation applied in `~/ads-dev/nectar/src/pages/api/nl-search.ts`
- 536 bare bibstem fields fixed across training data
- Quality report with before/after metrics

### 🔄 IN PROGRESS: Extended Workflow (US-008 to US-015)

Ralph now continues with the operator syntax fixes and full end-to-end testing:

| Story | Title | Status | Est. Time |
|-------|-------|--------|-----------|
| US-008 | Fix operator syntax in training data | ⏳ PENDING | 30 min |
| US-009 | Retrain model with fixed operator data | ⏳ PENDING | 45 min |
| US-010 | Deploy retrained model to Modal | ⏳ PENDING | 10 min |
| US-011 | Test UI with operator queries via nectar | ⏳ PENDING | 30 min |
| US-012 | Test edge cases and verify constraint validation | ⏳ PENDING | 20 min |
| US-013 | Regression test: verify original issues are fixed | ⏳ PENDING | 20 min |
| US-014 | Performance verification and final sign-off | ⏳ PENDING | 30 min |
| US-015 | Iteration control: if tests fail, return to data fixes | ⏳ PENDING | 0-60 min |

## What Was Already Done (US-001 to US-007)

### 1. Field Constraints Module (`field_constraints.py`)

Created enumerated valid values for constrained ADS fields:

```python
DOCTYPES = {
    "article", "eprint", "book", "phdthesis", "proposal", "software",
    "inproceedings", "inbook", "abstract", "bookreview", "catalog",
    "circular", "editorial", "erratum", "mastersthesis", "newsletter",
    "obituary", "pressrelease", "proceedings", "talk", "techreport", "misc"
}

PROPERTIES = {
    "refereed", "notrefereed", "openaccess", "ads_openaccess",
    "author_openaccess", "pub_openaccess", "eprint_openaccess",
    "article", "nonarticle", "eprint", "inproceedings", "software",
    "catalog", "associated", "data", "esource", "inspire",
    "library_catalog", "presentation", "toc", "ocr_abstract"
}

DATABASES = {"astronomy", "physics", "general"}

BIBGROUPS = {
    "HST", "JWST", "Spitzer", "Chandra", "ALMA", "VLT", "Keck", 
    "Gemini", "Pan-STARRS", "SDSS", "2MASS", ... (53 total)
}

ESOURCES = {
    "PUB_PDF", "PUB_HTML", "EPRINT_PDF", "EPRINT_HTML",
    "AUTHOR_PDF", "AUTHOR_HTML", "ADS_PDF", "ADS_SCAN"
}

DATA_SOURCES = {
    "MAST", "IRSA", "NED", "SIMBAD", "VizieR", "Herschel",
    "Chandra", "WISE", ... (24 total)
}
```

### 2. Query Validation Function (`validate_field_constraints()`)

Validates model-generated queries against field enumerations:

```python
def validate_field_constraints(query: str) -> ConstraintValidationResult:
    """
    Check all doctype, database, property, bibgroup values against FIELD_ENUMS.
    Returns list of invalid fields with suggestions for corrections.
    """
```

**Features:**
- Case-insensitive matching
- Support for OR lists: `property:(refereed OR openaccess)`
- Quoted value extraction: `doctype:"article"`
- Suggestions for invalid values
- Detailed error messages

### 3. Post-Processing Filter (`constrain_query_output()`)

Safety net to remove invalid field values before they reach the ADS API:

```python
def constrain_query_output(query: str) -> str:
    """
    Remove any doctype/database/property/bibgroup/esources/data values
    not in FIELD_ENUMS. Log warnings for each removed field.
    """
```

**Handles:**
- Invalid enum values (e.g., `bibstem:phdthesis` → removed)
- Common model hallucinations:
  - `doctype:journal` → removed (should be `article`)
  - `property:peerreviewed` → removed (should be `refereed`)
  - `database:astrophysics` → removed (should be `astronomy`)
  - `bibgroup:Hubble` → removed (should be `HST`)
- Preserves valid combinations (e.g., `property:refereed AND property:openaccess`)
- Cleans up orphaned operators and malformed parentheses

### 4. Training Data Fixes (US-004)

Fixed 536 bare bibstem fields:

**Before:**
```json
{"nl": "ApJ papers", "query": "bibstem:ApJ"}  // Missing quotes
```

**After:**
```json
{"nl": "ApJ papers", "query": "bibstem:\"ApJ\""}  // Quoted
```

Also fixed bare `author:`, `abs:`, `title:`, `keyword:` fields.

### 5. API Integration (US-006)

Integrated into `~/ads-dev/nectar/src/pages/api/nl-search.ts`:

```typescript
// After model inference, before ADS API call
const validation = validate_field_constraints(query);
if (!validation.valid) {
    // Log violations
    query = constrain_query_output(query);  // Auto-correct
}

// Only corrected queries sent to ADS API
const results = await adsApi.search(query);
```

Returns detailed validation info to frontend:
```typescript
{
    query: string,
    constraintViolations: string[],
    corrections: { field: string, removed: string }[],
    rawQuery: string  // Original before correction
}
```

## What Needs to Happen Next (US-008 to US-015)

### Phase 1: Fix Operator Syntax (US-008)

**Problem:** 37 malformed operator examples in training data:

```
❌ Bad examples found:
  - citations(abs:topic) → missing quotes around topic
  - trending(abs:(grains)) → extra/malformed parentheses
  - useful(author:smith) → bare field value
```

**Solution:** Fix all examples to correct syntax:

```
✅ Correct syntax:
  - citations(abs:"topic")
  - trending(abs:"grains")
  - useful(author:"smith")
```

**Ralph steps:**
1. Run `python scripts/audit_operators.py` to find bad patterns
2. Fix in `all_pairs.json` and `gold_examples.json`
3. Verify with `audit_operators.py` again (should show 0 bad patterns)
4. Commit with message: `[US-008] Fix operator syntax in training data`

### Phase 2: Retrain Model (US-009)

**Command:**
```bash
scix-finetune train --run-name v3-operators
```

**What happens:**
1. Uses fixed training data from US-008
2. Trains on H100 GPU for ~40 minutes
3. Creates checkpoint at `/runs/v3-operators/checkpoint-700`
4. Merge LoRA weights: `scix-finetune merge`
5. Result: `/runs/v3-operators/merged` (vLLM will pick this automatically)

**Expected improvement:**
- Model learns correct operator syntax natively
- No more bare fields inside operators
- Balanced parentheses in all operator calls

### Phase 3: Deploy New Model (US-010)

**Command:**
```bash
modal deploy serve_vllm.py
```

**What happens:**
1. vLLM picks latest merged model by modification time (mtime)
2. Model loaded from `/runs/v3-operators/merged`
3. Endpoint: https://sjarmak--nls-finetune-serve-vllm-serve.modal.run
4. Warm latency should be ~0.4-0.5s
5. Old v2-4k-pairs model automatically replaced

### Phase 4: Test Operator Queries (US-011)

Test in nectar UI (http://localhost:3000):

1. "papers similar to 2019ApJ...887L...1K" 
   → Expected: `similar(bibcode:"2019ApJ...887L...1K")` ✓

2. "trending papers on exoplanets"
   → Expected: `trending(abs:"exoplanets")` ✓

3. "papers citing gravitational waves"
   → Expected: `citations(abs:"gravitational waves")` ✓

4. "useful papers on dark matter"
   → Expected: `useful(abs:"dark matter")` ✓

5. "reviews of cosmology"
   → Expected: `reviews(abs:"cosmology")` ✓

**Pass criteria:**
- All 5 queries have valid ADS syntax (quoted values, balanced parens) ✓
- All 5 return results (count > 0) ✓
- No syntax errors in browser console ✓

### Phase 5: Test Constraint Validation (US-012)

Verify the post-processing filter works correctly:

1. "ADS papers" 
   → Model may output invalid database → post-processing removes it ✓

2. "refereed articles"
   → `property:refereed` (valid) → kept ✓

3. "papers by Hubble"
   → May output `bibgroup:Hubble` → post-processing handles it ✓

4. "PhD theses"
   → `doctype:phdthesis` (valid) → kept ✓

5. "data papers with open access"
   → `property:openaccess AND property:data` (valid) → kept ✓

**Pass criteria:**
- Invalid fields are removed gracefully ✓
- Results not empty from over-correction ✓
- Compare with v2-4k-pairs baseline - no regression ✓

### Phase 6: Regression Test Original Issues (US-013)

Verify fixes for original problems:

1. "papers by jarmak"
   → Old: bare `author:jarmak`
   → New: quoted `author:"jarmak"` ✓

2. "papers by kelbert"
   → Old: hallucinated initial `author:"kelbert, M"`
   → New: correct `author:"kelbert"` ✓

3. "citations from gravitational wave papers"
   → Old: `citationsjarmak` (no operator syntax)
   → New: `citations(abs:"gravitational waves")` ✓

4. "trending papers on cosmology"
   → Old: `trendingabs:cosmology` (missing quotes & operator parens)
   → New: `trending(abs:"cosmology")` ✓

5. "papers similar to famous paper"
   → Old: malformed parentheses `similar(bibcode:(2019ApJ...))`
   → New: correct `similar(bibcode:"2019ApJ...")` ✓

**Pass criteria:**
- All 5 generate valid ADS syntax ✓
- All 5 return results ✓
- Before/after comparison shows improvement ✓

### Phase 7: Performance Verification (US-014)

Measure key metrics:

**Latency:**
- Warm request: < 0.5s ✓
- Cold start: < 1.5s ✓ (with `min_containers=1`)

**Accuracy:**
- Syntax validity: > 95% of generated queries valid ✓
- Correction rate: < 5% of queries modified by post-processing ✓
- Error rate: < 5% queries return empty ADS results ✓

**Comparison (v2-4k-pairs vs v3-operators):**
- Operator accuracy improvement: measure via eval script
- Bare field elimination: should be 100%
- Constraint violation reduction: track via logs

**Output:** `data/datasets/OPERATOR_METRICS.md` with before/after comparison

### Phase 8: Iteration if Needed (US-015)

If any test fails:

**Decision tree:**
```
Test failure detected?
├─ Data quality issue? (missing examples, wrong format)
│   └─ Add/fix examples in all_pairs.json
│   └─ Go back to US-008 (audit & fix)
│   └─ Re-run US-009 (retrain)
│   └─ Re-run US-010 (deploy)
│   └─ Re-run US-011-014 (test)
│
└─ Model hyperparameter issue? (wrong learning rate, epochs, etc.)
    └─ Adjust train.py hyperparameters
    └─ Go back to US-009 (retrain with new params)
    └─ Re-run US-010 (deploy)
    └─ Re-run US-011-014 (test)
```

**Iteration limits:** Max 3 loops. If still failing after 3, escalate.

## Quick Start: Running Ralph

```bash
cd /Users/sjarmak/nls-finetune-scix

# Start the loop (will run through all pending stories)
./ralph.sh --tool amp 15
```

Ralph will:
1. Find next incomplete story (starting with US-008)
2. Run data fixes automatically (US-008)
3. For manual steps (US-009-014), provide instructions
4. Move to next story when you mark it complete
5. Iterate until all stories pass

**Expected timeline:** 3-4 hours total

## Key Infrastructure Already in Place

✅ **Field constraints module** (`field_constraints.py`)
✅ **Validation functions** (`validate.py`, `constrain.py`)
✅ **Post-processing filter** deployed in API (`nl-search.ts`)
✅ **Training data fixes** (536 bare fields fixed in US-004)
✅ **Modal infrastructure** (vLLM endpoint, GPU training)
✅ **NL search UI** (nectar with test capability)
✅ **Evaluation harness** (syntax validity metrics)

## What Gets Delivered

After Ralph completes US-008 to US-015:

✅ Fixed operator syntax in training data (37 examples corrected)
✅ Retrained model (v3-operators with correct operator patterns)
✅ Deployed to production Modal endpoint
✅ Verified in UI with 15 test cases (operators, constraints, regressions)
✅ Performance metrics documented
✅ Before/after comparison showing improvements
✅ Iteration cycle documented for future use
✅ AGENTS.md updated with operator syntax rules

## How Query Filtering Based on Metadata Works

This was already implemented in **US-003 and US-006**:

The `constrain_query_output()` function in `constrain.py`:

1. **Validates each field** against `FIELD_ENUMS` from `field_constraints.py`
2. **Removes invalid values** (e.g., invalid doctypes, databases, properties)
3. **Preserves valid field combinations** (e.g., `property:refereed AND property:openaccess`)
4. **Logs corrections** for debugging and monitoring
5. **Handles edge cases** (OR lists, quoted values, malformed parentheses)

**Example:**

```python
# Model outputs (with hallucinations):
query = 'doctype:journal property:peerreviewed AND bibstem:"ApJ" OR database:astrophysics'

# constrain_query_output() cleans it:
cleaned = 'bibstem:"ApJ"'  # Removed: journal, peerreviewed, astrophysics

# Only valid queries reach ADS API
results = adsApi.search(cleaned)
```

This is applied **after model inference** but **before ADS API call**, providing a safety net for model hallucinations.

## References

- **prd.json**: User stories with acceptance criteria and pass/fail status
- **prompt.md**: Ralph agent instructions
- **ralph.sh**: Main automation loop script
- **RALPH_SETUP.md**: Setup and execution guide
- **RALPH_INTEGRATION_GUIDE.md**: Technical integration details
- **field_constraints.py**: Field enumeration definitions
- **constrain.py**: Post-processing filter implementation
- **validate.py**: Field constraint validation functions
- **nl-search.ts**: API integration (~/ads-dev/nectar)
- **AGENTS.md**: Documented patterns and gotchas

---

**Status**: Ready to run Ralph for US-008 through US-015

**Next command:**
```bash
./ralph.sh --tool amp 15
```

Ralph will handle the rest! 🚀
