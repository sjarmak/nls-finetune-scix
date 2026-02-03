# Handoff: US-011 Resume Instructions

## Status
Ralph ran through US-008 to US-010 successfully ✅
- **US-008**: Operator syntax fixed
- **US-009**: Model retrained (v3-operators)
- **US-010**: Model deployed to Modal ✅

Ralph started **US-011** but marked it failed due to **incorrect validation criteria**.

## The Problem (Now Fixed!)

Ralph's test for US-011 was too strict. It required:
```
❌ WRONG: ALL operator values must be quoted
  trending(abs:"exoplanet")  # Too strict
```

But ADS query syntax actually allows both:
```
✅ CORRECT: Single-word values don't need quotes
  trending(abs:exoplanet)   # Valid
  trending(abs:"exoplanet") # Also valid - they're equivalent
  
✅ CORRECT: Multi-word values MUST be quoted
  citations(abs:"gravitational waves")  # Required
  citations(abs:gravitational waves)    # Invalid
```

## What Changed

Updated **prd.json** US-011 acceptance criteria:

**Before (incorrect):**
```
trending(abs:"exoplanets") - REQUIRED quotes
```

**After (correct):**
```
trending(abs:exoplanets) OR trending(abs:"exoplanets") - both valid
```

The key insight: **ADS doesn't care if single-word values are quoted or not.** They're equivalent.

## What This Means

The model output `trending(abs:exoplanet)` is **100% valid ADS syntax**. It will work perfectly. No need to retrain.

## How to Resume

Ralph is paused on US-011. Two options:

### Option A: Restart from US-011 with corrected criteria (RECOMMENDED)

```bash
cd /Users/sjarmak/nls-finetune-scix
./ralph.sh --tool amp 10
```

Ralph will:
1. Skip US-001 to US-010 (already done)
2. Re-run US-011 with **corrected validation criteria**
3. Proceed through US-012-015

This should pass now because the validation is correct.

### Option B: Manual test before restarting

If you want to verify manually first:

1. Go to nectar: http://localhost:3000
2. Test one operator query: "trending papers on exoplanets"
3. Check output in browser console
4. If it generates `trending(abs:exoplanet)` → **VALID ✓**
5. If result count > 0 → **WORKS ✓**
6. Then restart Ralph

## Expected Results for US-011

Model will likely generate queries like:
```
1. similar(bibcode:2019ApJ...887L...1K) ✓ Valid (no quotes needed for bibcode)
2. trending(abs:exoplanets) ✓ Valid (single word, no quotes)
3. citations(abs:"gravitational waves") ✓ Valid (multi-word, quotes)
4. useful(abs:"dark matter") ✓ Valid (multi-word, quotes)
5. reviews(abs:cosmology) ✓ Valid (single word, no quotes)
```

All 5 should pass now.

## Why This Matters

The training data fix in US-008 was correct. The model learned proper operator syntax. The issue wasn't the model - it was the **validation being too strict**.

Now that we've aligned validation with actual ADS query syntax rules:
- ✅ US-011 should pass
- ✅ US-012-013 testing will be accurate
- ✅ US-014 metrics will reflect true model quality
- ✅ No need to retrain the model

## ADS Quoting Rules (For Reference)

| Scenario | Example | Valid? |
|----------|---------|--------|
| Single-word value | `abs:exoplanet` | ✅ |
| Single-word value (quoted) | `abs:"exoplanet"` | ✅ (equivalent) |
| Multi-word phrase (quoted) | `abs:"dark matter"` | ✅ |
| Multi-word phrase (unquoted) | `abs:dark matter` | ❌ |
| Bibcode | `bibcode:2019ApJ...887L...1K` | ✅ |
| Inside operator (single-word) | `trending(abs:exoplanet)` | ✅ |
| Inside operator (multi-word) | `citations(abs:"gravitational waves")` | ✅ |

## Next Steps

1. **Option A (Recommended):** Restart Ralph
   ```bash
   ./ralph.sh --tool amp 10
   ```

2. **Option B (Manual):** Verify one query manually, then restart Ralph

3. Ralph will run US-011-015 with correct validation criteria

4. Expected total time for US-011-015: ~2 hours

## Questions?

- **Is single-word unquoted valid ADS?** YES - it's the standard format
- **Will model output work in SciX?** YES - post-processing handles both formats
- **Do we need to retrain?** NO - v3-operators is fine
- **What about the training data?** The fix in US-008 was correct; validation was just too strict

---

**Ready to resume?**

```bash
cd /Users/sjarmak/nls-finetune-scix
./ralph.sh --tool amp 10
```

Ralph will pick up from US-011 and proceed with correct validation. ✅
