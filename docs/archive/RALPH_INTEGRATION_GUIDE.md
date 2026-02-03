# Ralph Integration Points

This document maps where Ralph will make changes and how they integrate with existing code.

## 1. Field Constraints Module (US-001)

**New file**: `packages/finetune/src/finetune/domains/scix/field_constraints.py`

```python
# Example structure
FIELD_ENUMS = {
    "database": ["ASTRONOMY", "PHYSICS", "GENERAL"],
    "doctype": [
        "article", "eprint", "book", "phdthesis", "proposal", "software",
        "inproceedings", "inbook", "abstract", "bookreview", "catalog",
        "circular", "erratum", "mastersthesis", "newsletter", "obituary",
        "pressrelease", "techreport", "talk", "misc"
    ],
    "property": [
        "refereed", "notrefereed", "openaccess", "ads_openaccess",
        "author_openaccess", "pub_openaccess", "eprint_openaccess",
        "data", "esource", "library_catalog", "inspire", "associated",
        "nonarticle", "article", "toc", "presentation", "ocr_abstract",
        "Catalog", "Nonarticle"  # Aliases
    ],
    "bibgroup": ["CfA", "Gemini", "Keck", "HST", "JWST", "SDO", ...],
    "esources": ["PUB_PDF", "PUB_HTML", "EPRINT_PDF", "AUTHOR_PDF", ...],
}
```

**Integration**: Used by validation functions and post-processing filter

## 2. Enhanced Validation (US-002)

**Update file**: `packages/finetune/src/finetune/domains/scix/validate.py`

Add new function:
```python
def validate_field_constraints(query: str) -> ValidationResult:
    """Check that all field values are in FIELD_ENUMS."""
    # Extract field:value pairs
    # Validate each against FIELD_ENUMS
    # Return ValidationResult with errors for invalid values
```

**Tests**: `packages/finetune/src/finetune/domains/scix/test_validate.py`
- 10+ test cases for valid/invalid field combinations
- Edge cases: empty fields, malformed syntax, etc.

**Integration**: Called during data validation and training

## 3. Post-Processing Filter (US-003)

**Update file**: `~/ads-dev/nectar/src/pages/api/nl-search.ts` (in ads-dev project)

OR create new file in this repo and import it:
`packages/finetune/src/finetune/domains/scix/post_process.py`

```python
def constrain_query_output(query: str) -> str:
    """Clean up model-generated queries by enforcing field constraints.
    
    Removes invalid field values, logs corrections.
    """
    import re
    
    # Remove invalid doctype/database/property values
    for field in ["doctype", "database", "property", "bibgroup"]:
        # Extract field values and validate
        # Remove invalid ones
    
    return query.strip()
```

**Integration in API**:
```typescript
// In ~/ads-dev/nectar/src/pages/api/nl-search.ts
const constrainQueryOutput = require("@/lib/query-constraints").constrainQueryOutput;

// After model inference:
let query = await model.generate(naturalLanguage);
query = constrainQueryOutput(query);  // Clean up before ADS API call

// Send to ADS
const results = await adsApi.search(query);
```

## 4. Training Data Fixes (US-004)

**Files modified**:
- `data/datasets/processed/all_pairs.json` (3025 pairs)
- `data/datasets/raw/gold_examples.json` (reference data)

**Changes**:
- Quote all bare `author:` values → `author:"name"`
- Quote all bare `abs:`, `title:`, `full:` values
- Verify all queries pass `validate.py` linting

**Audit script**: `scripts/audit_bare_fields.py`

Run before/after:
```bash
python scripts/audit_bare_fields.py
# Before: 45 bare fields found
# After: 0 bare fields found (target)
```

## 5. Quality Report (US-005)

**New file**: `data/datasets/QUALITY_REPORT.md`

```markdown
# Training Data Quality Report

## Metrics
- Total pairs: 3025
- Validation pass rate: 99.5%
- Invalid fields found: 0
- Before/after improvements: [list]

## Category Distribution
- first_author: 853 examples (28.2%)
- unfielded: 557 examples (18.4%)
- author: 448 examples (14.8%)
... etc

## Recommendations
1. Expand unfielded category with real ADS queries
2. Add more affiliation-based examples
3. Generate examples from CanonicalAffiliations data
```

## 6. API Integration (US-006)

**Files to update** in ~/ads-dev:

### `nectar/src/pages/api/nl-search.ts`

```typescript
// Remove these functions (no longer needed):
// - validateAuthors()
// - resolveObjectNames() 
// - expandSynonyms()

// Add this (NEW):
import { constrain_query_output } from "@/lib/query-constraints";

// After model generation:
let query = await model.generate(nl);

// Apply constraint validation
const validation = validate_field_constraints(query);
if (!validation.valid) {
    // Either:
    // 1. Log and auto-correct
    query = constrain_query_output(query);
    // 2. OR return error to user
    return { error: validation.errors };
}

// Only generateQueryVariations remains (for UX)
const variations = generateQueryVariations(query);

// Call ADS API
const results = await adsApi.search(query);
```

### `nectar/lib/query-constraints.ts` (NEW)

```typescript
import { FIELD_ENUMS } from "@/lib/ads-field-enums";
import { validate_field_constraints } from "@/lib/validate-constraints";

export function constrainQueryOutput(query: string): string {
    // Enforce field constraints
    // Remove invalid values
    // Return cleaned query
}

export const FIELD_ENUMS = {
    // ... copied from Python module
};
```

## 7. Documentation (US-007)

**Update file**: `AGENTS.md`

Add new section:
```markdown
## ADS Field Constraints

### Why Field Enumerations Matter

The model must learn that certain fields have limited valid values:
- `doctype: article|eprint|book|phdthesis|proposal|software` (16 values)
- `database: ASTRONOMY|PHYSICS|GENERAL` (3 values)
- `property: refereed|openaccess|data|notrefereed` (19 values)
- `bibgroup: CfA|Gemini|Keck|HST|JWST|...` (30+ values)

Using invalid values → query fails at ADS API.

### How We Teach This

1. **Training data**: Only include valid field values
   - Example ✓: `property:refereed`
   - Example ✗: `property:awesome` (not a real ADS value)

2. **Post-processing**: Catch invalid values before they reach ADS
   - If model outputs `bibstem:phdthesis`, remove it (there's no such bibstem)
   - Log the correction for debugging

3. **Testing**: Verify all training pairs against ADS schema
   ```bash
   python scripts/audit_bare_fields.py  # Check for format issues
   python -m finetune.domains.scix.validate # Lint all queries
   ```

### Common Model Hallucinations

| Input | Model Output (Wrong) | Corrected |
|-------|-----------------|----------|
| "ADS papers" | `author:"ADS"` or `bibstem:ADS` | Empty or `database:ASTRONOMY` |
| "data papers" | `bibstem:data` | `property:data` |
| "refereed articles" | `property:refereeed` (typo) | `property:refereed` |

### Resources

- ADS Schema: github.com/adsabs/montysolr (schema.xml)
- Field Constraints: packages/finetune/src/finetune/domains/scix/field_constraints.py
- Validation: packages/finetune/src/finetune/domains/scix/validate.py
```

## Data Flow After Ralph

```
User Input
    ↓
Model (Qwen3-1.7B) [trained on fixed data]
    ↓
API Route (nl-search.ts)
    ├→ validate_field_constraints() → check all fields valid
    ├→ constrain_query_output() → remove invalid fields
    └→ generateQueryVariations() → create UX-friendly versions
    ↓
ADS API
    ↓
Results
```

## Testing After Ralph Completes

### Test Cases to Verify

1. **Jarmak case** (original bug)
   ```
   Input: "papers by jarmak"
   Expected: author:"jarmak"
   Before: jarmak (bare)
   After: author:"jarmak" (fixed)
   ```

2. **Invalid field case**
   ```
   Input: "ADS papers"
   Model outputs: author:"ADS" OR bibstem:ADS
   Post-processing catches: invalid author value "ADS"
   Corrected: database:ASTRONOMY (or empty)
   ```

3. **Mixed valid/invalid**
   ```
   Input: "refereed articles from 2020"
   Model outputs: property:refereed AND year:2020 AND property:refereeed (typo)
   Post-processing removes: property:refereeed (invalid)
   Result: property:refereed AND year:2020
   ```

## Integration Checklist

- [ ] Ralph completes all 7 stories
- [ ] Review QUALITY_REPORT.md for improvements
- [ ] Test 3 cases above in staging environment
- [ ] Retrain model with fixed training data
- [ ] Deploy new model to Modal
- [ ] Update API route with post-processing filter
- [ ] Monitor logs for validation corrections
- [ ] Measure improvement in user query satisfaction

## Rollback Plan

If validation is too strict:

1. Comment out constrain_query_output() in API
2. Keep validate_field_constraints() for logging only
3. Gradually refine based on real query logs
4. Re-enable corrections once confident

## Extended Workflow: US-008 to US-015 (End-to-End Testing)

After US-001 to US-007 foundation is complete, Ralph runs the operator syntax fix through full deployment and testing:

### US-008: Fix Operator Syntax in Training Data
- Audit operator examples for missing quotes and malformed parentheses
- Fix citations(), trending(), useful(), reviews(), similar(), references()
- Run scripts/audit_operators.py to verify no remaining issues
- Update AGENTS.md with operator syntax rules
- **Time**: ~30 minutes

### US-009: Retrain Model with Fixed Data
- Run: `scix-finetune train --run-name v3-operators`
- Training on H100 GPU: ~40 minutes
- Merge LoRA weights: `scix-finetune merge`
- Result: `/runs/v3-operators/merged` (picked by vLLM via mtime)
- **Time**: ~45 minutes (mostly waiting)

### US-010: Deploy to Modal
- Run: `modal deploy serve_vllm.py`
- vLLM picks latest merged model automatically
- Verify endpoint: curl https://sjarmak--nls-finetune-serve-vllm-serve.modal.run/v1/models
- Test warm latency (should be ~0.4-0.5s)
- **Time**: ~10 minutes

### US-011: Test Operator Queries in UI
Test 5 operator types via NL search in nectar:
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
- Verify all 5 return valid syntax ✓
- Verify all 5 return results (count > 0) ✓
- **Time**: ~30 minutes

### US-012: Test Constraint Validation Edge Cases
Test invalid field handling:
```
1. "ADS papers" → model may output invalid database → removed by post-processing ✓
2. "refereed articles" → property:refereed (valid, kept) ✓
3. "papers by Hubble" → may output bibgroup:Hubble → corrected or removed ✓
4. "PhD theses" → doctype:phdthesis (valid, kept) ✓
5. "data papers with open access" → property:openaccess AND property:data (valid, kept) ✓
```
- Verify post-processing removes only invalid fields ✓
- Verify results not empty from over-correction ✓
- Compare with v2-4k-pairs baseline ✓
- **Time**: ~20 minutes

### US-013: Regression Test Original Issues
Verify fixes for original bugs from US-004 and US-008:
```
1. "papers by jarmak" → author:"jarmak" (no bare fields) ✓
2. "papers by kelbert" → author:"kelbert" (no hallucinated initials) ✓
3. "citations from gravitational wave papers" → citations(abs:"gravitational waves") (correct operator) ✓
4. "trending papers on cosmology" → trending(abs:"cosmology") (quoted values inside operator) ✓
5. "papers similar to famous paper" → similar(bibcode:"...") (balanced parentheses) ✓
```
- All 5 must generate valid ADS syntax ✓
- All 5 must return results ✓
- Compare before/after outputs ✓
- **Time**: ~20 minutes

### US-014: Performance Verification
Measure and validate metrics:
- **Latency**: Warm ~0.4s, cold <1.5s ✓
- **Syntax Validity**: > 95% of queries valid ADS syntax ✓
- **Correction Rate**: < 5% of queries modified by post-processing ✓
- **Error Rate**: < 5% queries return empty ADS results ✓
- **Before/After**: Compare v2-4k-pairs vs v3-operators
  - Operator accuracy improvement
  - Bare field elimination
  - Constraint violation reduction
- Generate metrics report at: `data/datasets/OPERATOR_METRICS.md`
- **Time**: ~30 minutes

### US-015: Iteration Control
If any test fails (US-011-014):

**Decision tree:**
```
Test failed?
├─ YES: Is it a data quality issue?
│   ├─ YES (missing examples, wrong format)
│   │   └─ Fix data in all_pairs.json/gold_examples.json
│   │   └─ Go back to US-008, audit & fix
│   │   └─ Re-run US-009 (retrain)
│   │   └─ Re-run US-010 (deploy)
│   │   └─ Re-run US-011-014 (test)
│   │
│   └─ NO (model hyperparameter issue)
│       └─ Adjust learning_rate, epochs, batch_size in train.py
│       └─ Go back to US-009 (retrain with new params)
│       └─ Re-run US-010 (deploy)
│       └─ Re-run US-011-014 (test)
│
└─ NO: All tests pass ✓
    └─ Mark US-008 through US-014 as complete
    └─ Merge improve-training-data → main
    └─ Create summary with before/after metrics
    └─ Update progress.txt with results
```

**Iteration limits:**
- Max 3 iteration loops
- If > 3 loops needed, escalate to manual review
- Document findings in progress.txt for next iteration
- **Time**: 0-60 min depending on results

## Complete Workflow Flowchart

```
Data Audit (US-008) 
    ↓ [Fix bare fields & operators]
Retrain Model (US-009) 
    ↓ [40 min on H100]
Deploy (US-010) 
    ↓ [5 min deployment]
Test Operators (US-011) 
    ↓ [5 operator queries]
Test Constraints (US-012) 
    ↓ [5 edge cases]
Regression Test (US-013) 
    ↓ [5 original bugs]
Verify Metrics (US-014) 
    ↓ [latency, accuracy, correction rate]
All Pass? (US-015)
    ├─ YES → ✅ COMPLETE (merge & summary)
    └─ NO → Fix root cause & loop back to US-008 or US-009
```

## Future Improvements

Beyond Ralph's scope:

1. **Query mining** - Extract real queries from SciX logs, add to training data
2. **Affiliation expansion** - Use CanonicalAffiliations to generate affiliation pairs
3. **Synonym augmentation** - Use ads_text_simple.synonyms for data augmentation
4. **Online learning** - Capture user corrections and retrain
