# Ralph Session: Improve Training Data & Post-Processing Validation

This document explains how to run Ralph to autonomously implement training data improvements and post-processing validation for the NLS Fine-tune SciX project.

## What Ralph Will Do

Ralph is an autonomous AI agent loop that will:

1. **Create field constraints module** - Define valid ADS field values (doctype, database, property, etc.)
2. **Build validation functions** - Check queries against field constraints
3. **Implement post-processing filter** - Clean up model output before returning to user
4. **Fix remaining data issues** - Quote bare field values in training data
5. **Generate quality report** - Document improvements and metrics
6. **Integrate into API** - Connect validation to the nectar API layer
7. **Document patterns** - Update AGENTS.md with learnings

## Prerequisites

1. **Amp or Claude Code** installed and authenticated
   ```bash
   # For Amp (recommended)
   which amp  # Should return path
   
   # For Claude Code
   npm install -g @anthropic-ai/claude-code
   ```

2. **jq** installed
   ```bash
   brew install jq  # macOS
   apt-get install jq  # Linux
   ```

3. **Git** configured
   ```bash
   git config --global user.email "you@example.com"
   git config --global user.name "Your Name"
   ```

4. **mise** for running project commands
   ```bash
   which mise  # Should return path
   ```

## Complete Workflow: Data Fix → Retrain → Deploy → Test

The Ralph loop now implements a **full end-to-end workflow**:

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Data Fixes (US-008)                             │
│ - Fix bare fields and operator syntax in training data   │
│ - Verify with scripts/audit_*.py                        │
└─────────┬───────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Retrain (US-009)                               │
│ - Run: scix-finetune train --run-name v3-operators      │
│ - Wait ~40 min for training on H100                     │
│ - Merge LoRA: scix-finetune merge                       │
│ - Result: /runs/v3-operators/merged                     │
└─────────┬───────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Deploy (US-010)                                │
│ - Run: modal deploy serve_vllm.py                       │
│ - Wait ~5 min for deployment                            │
│ - Verify: endpoint live & serving v3-operators          │
└─────────┬───────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Test UI (US-011 to US-013)                     │
│ - Test 5 operator queries in nectar                     │
│ - Test 5 constraint edge cases                          │
│ - Test 5 regression cases (original bugs)               │
│ - All tests must pass with valid syntax & results       │
└─────────┬───────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 5: Verify & Sign-off (US-014)                     │
│ - Measure latency & accuracy metrics                    │
│ - Compare before/after improvements                     │
│ - Confirm > 95% syntax validity                         │
│ - Confirm < 5% correction rate                          │
└─────────┬───────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 6: Iterate if Needed (US-015)                     │
│ - If tests fail: identify root cause                    │
│ - Fix training data or hyperparameters                  │
│ - Loop back to US-008 or US-009                         │
│ - Repeat until all tests pass                           │
└─────────────────────────────────────────────────────────┘
```

## How to Run Ralph

### Option 1: Run Complete Workflow with Amp (Default)

```bash
cd /Users/sjarmak/nls-finetune-scix
./ralph.sh --tool amp 15
```

Ralph will:
- Create/checkout the `improve-training-data` branch
- Pick the next story where `passes: false`
- Invoke Amp to implement it
- For manual steps (US-009, US-010, US-011-014):
  - Provide instructions
  - Wait for you to complete and report results
- Mark stories as complete and advance
- Repeat until all stories pass or hit max iterations

### Expected Timeline

- **US-008** (data fixes): ~30 min
- **US-009** (retraining): ~45 min (manual, don't wait in Ralph)
- **US-010** (deployment): ~10 min (manual)
- **US-011 to US-014** (testing): ~60 min (manual)
- **US-015** (iteration): 0-60 min depending on test results

**Total**: ~3-4 hours for full workflow

### Option 2: Run with Claude Code

```bash
cd /Users/sjarmak/nls-finetune-scix
./ralph.sh --tool claude
```

### Option 3: Run with Limited Iterations

```bash
./ralph.sh --tool amp --iterations 5
```

Limit iterations to test or do partial work.

## Key Files

| File | Purpose |
|------|---------|
| `prd.json` | User stories (task list) - Updated by Ralph as stories complete |
| `progress.txt` | Learning log - Ralph appends discoveries here |
| `prompt.md` | Instructions for the AI agent |
| `ralph.sh` | The main Ralph loop script |
| `scripts/audit_bare_fields.py` | Helper script for data auditing |
| `AGENTS.md` | Updated by Ralph with patterns/gotchas |

## User Story Breakdown

**Story 1 (Priority 1)**: Create field constraints module
- Define FIELD_ENUMS dict with all valid field values from ADS schema
- Include doctypes, properties, databases, bibgroups, esources
- Takes ~30-40 minutes

**Story 2 (Priority 2)**: Add constraint validation function
- Create validate_field_constraints() to check queries
- Add unit tests
- Takes ~40-50 minutes

**Story 3 (Priority 2)**: Implement post-processing filter
- Create constrain_query_output() to clean model output
- Handle edge cases and malformed queries
- Takes ~30-40 minutes

**Story 4 (Priority 1)**: Fix remaining bare fields in training data
- Run audit script to find unquoted fields
- Fix in JSON files
- Takes ~20-30 minutes

**Story 5 (Priority 3)**: Generate quality report
- Create QUALITY_REPORT.md with metrics
- Document improvements
- Takes ~20 minutes

**Story 6 (Priority 3)**: Integrate into API
- Update ~/ads-dev/nectar API route
- Apply validation before ADS API calls
- Takes ~30-40 minutes

**Story 7 (Priority 3)**: Document patterns
- Update AGENTS.md with field constraint knowledge
- Takes ~15 minutes

**Total estimated time**: ~3-4 hours for Ralph to complete all stories

## What Happens During Each Iteration

1. Ralph reads `prd.json` and finds the next incomplete story
2. Ralph invokes your AI tool (Amp or Claude) with story details
3. The AI implements the story based on acceptance criteria
4. You verify that all acceptance criteria pass
5. If yes:
   - Ralph marks story as `passes: true`
   - Ralph commits changes with story ID
   - Ralph appends learnings to `progress.txt`
   - Ralph moves to next story
6. If no:
   - Ralph asks for feedback
   - The AI gets another chance in the next iteration

## Quality Gates

Before marking a story complete, Ralph will verify:

- [ ] All acceptance criteria implemented
- [ ] `mise run lint` passes (no style/import errors)
- [ ] `mise run test` passes (all tests green)
- [ ] `mise run verify` passes (full verification)
- [ ] Commit message includes story ID (e.g., `[US-001]`)
- [ ] AGENTS.md updated with patterns discovered
- [ ] progress.txt updated with learnings

## Monitoring Ralph's Progress

### Check Current Status

```bash
# See which story Ralph is on
jq '.userStories[] | select(.passes == false) | .id' prd.json | head -1

# See all completed stories
jq '.userStories[] | select(.passes == true) | {id, title}' prd.json

# See progress log
tail -50 progress.txt
```

### View Generated Code

Ralph commits after each story:

```bash
# See recent commits
git log --oneline -10

# Diff latest commit
git show HEAD

# See all changes on this branch
git diff main..improve-training-data
```

### Pause and Resume

If Ralph hits max iterations:

```bash
# Check how many stories are left
jq '.userStories[] | select(.passes == false) | .id' prd.json | wc -l

# Resume Ralph where it left off
./ralph.sh --tool amp --iterations 5
```

Ralph will skip completed stories and continue with incomplete ones.

## Debugging Tips

### If Ralph Can't Find Amp/Claude

```bash
# Check if amp is installed
which amp
# If not found, install via: https://ampcode.com

# Check if claude is installed  
which claude
# If not found: npm install -g @anthropic-ai/claude-code
```

### If Ralph Gets Stuck on a Story

1. Stop Ralph with `Ctrl+C`
2. Review the branch:
   ```bash
   git status
   git diff
   ```
3. Manually fix issues if needed
4. Commit your changes:
   ```bash
   git add -A
   git commit -m "[US-XXX] Manual fix for story"
   ```
5. Mark story complete in prd.json:
   ```bash
   jq '.userStories[0].passes = true' prd.json > tmp && mv tmp prd.json
   ```
6. Resume Ralph:
   ```bash
   ./ralph.sh --tool amp
   ```

### If Tests Fail

Ralph's quality gates will catch test failures. The AI will debug and fix.

To manually debug:

```bash
# Run specific test
mise run test -- --grep "query validation"

# Run lint only
mise run lint

# Run full verification
mise run verify
```

## After Ralph Completes

When Ralph outputs `<promise>COMPLETE</promise>`:

1. **Verify all stories passed**
   ```bash
   jq '.userStories | map(select(.passes == false)) | length' prd.json
   # Should return: 0
   ```

2. **Review changes**
   ```bash
   git diff main..improve-training-data --stat
   ```

3. **Merge to main**
   ```bash
   git checkout main
   git pull
   git merge improve-training-data
   git push
   ```

4. **Retrain the model** with improved data
   ```bash
   scix-finetune train --run-name v3-improved
   ```

5. **Update API** to use new post-processing layer

## Expected Outcomes

After Ralph completes:

✅ **New module**: `packages/finetune/src/finetune/domains/scix/field_constraints.py`
✅ **Enhanced validation**: Extended `validate.py` with field constraint checking
✅ **Post-processing filter**: New `constrain_query_output()` function in API route
✅ **Cleaned training data**: All bare fields quoted in all_pairs.json and gold_examples.json
✅ **Quality metrics**: `data/datasets/QUALITY_REPORT.md` documenting improvements
✅ **API integration**: `~/ads-dev/nectar` updated with validation layer
✅ **Documentation**: AGENTS.md updated with ADS field constraints knowledge

## Next Steps (Manual)

After Ralph completes:

1. **Retrain model** with fixed training data
   ```bash
   scix-finetune train --run-name v3-field-constraints
   ```

2. **Deploy** new version to Modal
   ```bash
   modal deploy serve_vllm.py
   ```

3. **Test** the improved model against known problem cases:
   - "papers by jarmak" → Should now output `author:"jarmak"`
   - "ADS papers" → Should handle gracefully (not output gibberish)
   - Other edge cases from your testing

4. **Monitor** improvements via eval metrics

## Troubleshooting

### "Error: jq is required but not installed"
```bash
brew install jq
```

### "Error: Neither amp nor claude command found"
Install one of:
- Amp: https://ampcode.com (sign up and install CLI)
- Claude Code: `npm install -g @anthropic-ai/claude-code`

### "Error: prompt.md not found"
Already created - should be in your repo root. If not:
```bash
# It's already in the repo
ls -la prompt.md
```

### Ralph gets stuck asking "Did story pass? (y/n)"
You need to manually review the code and answer. Ralph is checking if the AI's implementation actually worked. Read the acceptance criteria and verify:
- All criteria implemented
- Tests pass
- No syntax errors

## Support

For Ralph questions: https://ghuntley.com/ralph/
For project questions: See README.md and DEVELOPMENT.md
