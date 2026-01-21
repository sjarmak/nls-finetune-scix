# Ralph Task Suite - Complete Index

## Overview

This is a **Ralph task suite** - a set of user stories that an autonomous AI agent (Ralph) will implement to improve training data quality and add post-processing validation to the NLS Fine-tune SciX project.

**Ralph will run autonomously for 3-4 hours, implementing all 7 stories until they pass.**

## Quick Links

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [RALPH_QUICK_START.txt](RALPH_QUICK_START.txt) | **Start here** - 1-minute overview | 1 min |
| [RALPH_SETUP.md](RALPH_SETUP.md) | Complete setup and usage guide | 10 min |
| [RALPH_INTEGRATION_GUIDE.md](RALPH_INTEGRATION_GUIDE.md) | Technical details and code locations | 10 min |
| [prd.json](prd.json) | User stories (machine-readable) | - |
| [prompt.md](prompt.md) | Ralph's instructions | 5 min |

## To Run Ralph

```bash
./ralph.sh
```

Then answer "y" or "n" when asked if each story passed.

## What Ralph Will Build

1. **field_constraints.py** - ADS field enumerations (16 doctypes, 3 databases, 19 properties, etc.)
2. **Query validation functions** - Check fields against constraints
3. **Post-processing filter** - Clean model output before API calls
4. **Fixed training data** - Quote all bare field values (45 found)
5. **Quality report** - Document improvements and metrics
6. **API integration** - Connect validation to ~/ads-dev/nectar
7. **Documentation** - Update AGENTS.md with learned patterns

## User Stories

### Priority 1 (Foundation)
- **US-001** Create ADS field constraints module (field_constraints.py)
- **US-004** Fix remaining bare fields in training data

### Priority 2 (Core Functionality)
- **US-002** Add query constraint validation functions
- **US-003** Implement post-processing filter for model output

### Priority 3 (Polish & Documentation)
- **US-005** Generate training data quality report
- **US-006** Integrate field constraints into API validation layer
- **US-007** Document field constraint patterns in AGENTS.md

## Key Files

```
prd.json                      ← Task list (what Ralph will do)
prompt.md                     ← Ralph's instructions
ralph.sh                      ← Main automation script
RALPH_QUICK_START.txt         ← 1-minute overview
RALPH_SETUP.md                ← Complete usage guide
RALPH_INTEGRATION_GUIDE.md    ← Technical architecture
RALPH_INDEX.md                ← This file
scripts/audit_bare_fields.py  ← Helper for data auditing
progress.txt                  ← Created by Ralph with learnings
```

## The Problem Ralph Solves

**Current state:**
- Model outputs: `author:jarmak` (unquoted) ❌
- Model outputs: `bibstem:phdthesis` (invalid) ❌
- Training data has 45+ quality issues ❌
- No validation at inference time ❌

**After Ralph:**
- Model learns correct syntax: `author:"jarmak"` ✓
- Invalid fields caught and removed ✓
- 100% training data quality verified ✓
- Validation layer prevents hallucinations ✓

## How It Works

1. **Ralph reads prd.json** - Finds next incomplete story
2. **Ralph invokes Amp/Claude** - AI implements the story
3. **You verify it works** - Answer "y" if all acceptance criteria pass
4. **Ralph commits and continues** - Moves to next story
5. **Repeat until all 7 stories pass** - Usually 3-4 hours

## Getting Started

### Step 1: Prerequisites
```bash
which amp  # Make sure Amp is installed
jq --version  # Make sure jq is installed
```

### Step 2: Review (Optional)
```bash
cat RALPH_QUICK_START.txt      # 1-minute overview
cat RALPH_SETUP.md             # Full documentation
```

### Step 3: Run Ralph
```bash
./ralph.sh
```

### Step 4: Monitor Progress
```bash
# While Ralph runs:
jq '.userStories[] | {id, passes}' prd.json
tail -20 progress.txt
git log --oneline -10
```

### Step 5: When Complete
```bash
# Verify all stories passed:
jq '.userStories[] | select(.passes == false) | .id' prd.json | wc -l
# Should output: 0

# Review changes:
git diff main..improve-training-data --stat

# Retrain model with fixed data:
scix-finetune train --run-name v3-field-constraints
```

## What Happens When Ralph Asks "Did story pass?"

Ralph will:
1. Show you what the AI implemented
2. Ask if all acceptance criteria were met
3. Expect: `y` (yes, passes) or `n` (no, retry)

**You should answer "y" only if:**
- All acceptance criteria are implemented
- `mise run lint` would pass (no style errors)
- `mise run test` would pass (tests green)
- No obvious bugs or incomplete code

## Monitoring Commands

```bash
# Check current story
jq '.userStories[] | select(.passes == false) | {id, title}' prd.json | head -1

# Count completed stories
jq '.userStories[] | select(.passes == true) | .id' prd.json | wc -l

# See learnings logged by Ralph
tail -50 progress.txt

# View recent commits
git log --oneline -10

# See all changes on this branch
git diff main..improve-training-data --stat
```

## Troubleshooting

**"Error: which amp not found"**
→ Install Amp at https://ampcode.com

**"Error: jq is required"**
→ Run: `brew install jq`

**"Ralph got stuck on a story"**
→ Stop with Ctrl+C, read RALPH_SETUP.md "Debugging Tips"

**"I want to run Ralph on only 1 story"**
→ Edit prd.json, set `passes: true` for other stories, run `./ralph.sh`

## Advanced Usage

### Run with Claude Code instead of Amp
```bash
./ralph.sh --tool claude
```

### Limit iterations (for testing)
```bash
./ralph.sh --tool amp 3  # Only run 3 iterations
```

### Skip a story (for testing)
```bash
# Edit prd.json and set passes: true for that story
jq '.userStories[0].passes = true' prd.json > tmp && mv tmp prd.json
```

### Check what Ralph will work on next
```bash
jq '.userStories[] | select(.passes == false) | {priority, id, title}' prd.json | head -1
```

## Expected Timeline

| Story | Estimate | Type |
|-------|----------|------|
| US-001 | 30-40 min | Create module |
| US-002 | 40-50 min | Add validation |
| US-003 | 30-40 min | Post-processing |
| US-004 | 20-30 min | Fix data |
| US-005 | 20 min | Report |
| US-006 | 30-40 min | API integration |
| US-007 | 15 min | Documentation |
| **Total** | **3-4 hours** | **Autonomous** |

## After Ralph Completes

1. **Review results**
   ```bash
   git log --oneline -20  # See all commits
   cat data/datasets/QUALITY_REPORT.md  # See metrics
   git diff main..improve-training-data -- packages/  # See code changes
   ```

2. **Merge to main**
   ```bash
   git checkout main
   git merge improve-training-data
   ```

3. **Retrain model**
   ```bash
   scix-finetune train --run-name v3-field-constraints
   ```

4. **Deploy new model**
   ```bash
   modal deploy serve_vllm.py
   ```

5. **Test improvements**
   - "papers by jarmak" → should now output `author:"jarmak"`
   - "ADS papers" → should handle gracefully
   - Other edge cases

## Support & Documentation

- **Ralph docs**: https://ghuntley.com/ralph/
- **ADS API**: https://github.com/adsabs/adsabs-dev-api
- **This project**: README.md, DEVELOPMENT.md
- **Question about Ralph?**: Read RALPH_SETUP.md

## Key Concepts

**User Story** - A single feature/task that Ralph will implement. Each story has:
- ID (US-001, US-002, etc.)
- Title
- Description
- Acceptance Criteria (must all pass)
- Priority (1-3)
- Pass status (true/false)

**PRD** - Product Requirements Document (prd.json). Machine-readable list of all user stories.

**Iteration** - One run of Ralph. Each iteration:
1. Picks next incomplete story
2. Invokes AI (Amp or Claude)
3. Asks if it passed
4. Commits and updates progress
5. Repeats

**Quality Gate** - A check that must pass before marking a story complete (lint, test, type checking, etc.)

## Philosophy

Ralph works by:
1. **Small, focused stories** - Each completable in one session
2. **Clear acceptance criteria** - No ambiguity about "done"
3. **Automated feedback loops** - Tests catch problems immediately
4. **Persistent memory** - progress.txt and git history
5. **Fresh context each iteration** - AI starts clean each time

This approach scales to large features by breaking them into small stories.

---

**Ready?** Run: `./ralph.sh`

**Have questions?** Read: [RALPH_SETUP.md](RALPH_SETUP.md)

**Need technical details?** Read: [RALPH_INTEGRATION_GUIDE.md](RALPH_INTEGRATION_GUIDE.md)
