# PRD: NLS Fine-tune SciX Query Translation

## Overview

Infrastructure to fine-tune a model that translates natural language to ADS/SciX scientific literature search queries.

**Example:**
- Input: "papers by Hawking on black hole radiation from the 1970s"
- Output: `author:"Hawking, S" abs:"black hole radiation" pubdate:[1970 TO 1979]`

**Target:** Complementary search feature for [SciXplorer.org](https://scixplorer.org/)

---

## Agent Harness Architecture

This PRD follows the [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) pattern.

### Session Types

| Session Type | Purpose | Tools |
|--------------|---------|-------|
| **Initializer** | First session: bootstrap environment, create tracking files | `mise run install`, git |
| **Coding** | Subsequent sessions: incremental feature work | All tools, `mise run verify` |

### Persistent State Files

| File | Purpose | Agent Updates |
|------|---------|---------------|
| `features.json` | Feature tracking with pass/fail status | On feature completion |
| `.beads/` | Multi-session task tracking (beads CLI) | Via `bd` commands |
| `docs/current-state.md` | Model evaluation results, findings | After evaluations |

---

## Session Initialization Protocol

Every coding session MUST start with:

```bash
# 1. Verify working directory
pwd

# 2. Review progress state
bd ready                    # Available beads
cat features.json | jq '.features[] | select(.status == "failing") | .id' | head -1

# 3. Check recent commits
git log --oneline -5

# 4. Verify environment
mise run verify

# 5. Start servers (if needed)
mise run dev
```

---

## Feature List Specification

Features are tracked in `features.json` with this schema:

```json
{
  "id": "domain-001",
  "name": "SciX domain module created with field definitions",
  "category": "setup|domain|data|api|cli|modal|eval|web",
  "status": "passing|failing",
  "verification": "command to verify feature works"
}
```

### Critical Constraints

1. **Never delete or edit verification commands** - this could lead to missing or buggy functionality
2. **Work on ONE failing feature at a time** - prevents scope creep
3. **Mark passing ONLY after verification succeeds** - run `mise run verify` first
4. **Commit after each feature** - enables rollback if needed

---

## Phase Breakdown

### Phase 0: Bootstrap (CLOSED)
- [x] Repository structure with monorepo layout
- [x] mise configuration for Python/Bun runtimes
- [x] Basic project scaffolding

### Phase 1: ADS Query Validator + Dataset Integration (CLOSED)
- [x] `finetune/domains/scix/fields.py` - 40+ ADS field definitions
- [x] `finetune/domains/scix/validate.py` - Offline linter + API validation
- [x] `finetune/domains/scix/prompts.py` - Training/generation prompts
- [x] NL validation to detect ADS syntax leakage

### Phase 2: Data Ingestion + Synthetic NL Generation (CLOSED)
- [x] Curated gold examples (96 in `data/datasets/raw/gold_examples.json`)
- [x] `scripts/generate_nl.py` - Synthetic NL generation with ADS-specific handling
- [x] `scripts/validate_dataset.py` - Dataset validation pipeline
- [x] Training data output to `data/datasets/processed/` (2,447 unique pairs)

### Phase 3: Train + Inference with Validation Retry (CLOSED)
- [x] Modal `train.py` syntax valid
- [x] Training JSONL with 2,447 examples (exceeds 500 target)
- [x] LoRA training on Qwen3-1.7B (~12.5 min on H100, 93% token accuracy)
- [x] Inference endpoint deployed: https://sjarmak--nls-finetune-serve-vllm-serve.modal.run
- [x] vLLM serving with OpenAI-compatible API (model name: "llm")

**Results:**
- Training time: ~12.5 minutes on H100 ✓
- Training cost: ~$1.50 ✓
- Inference latency: <500ms warm ✓

**Volume Configuration:**
- Data: `scix-finetune-data`
- Runs: `scix-finetune-runs`

### Phase 4: Evaluation Harness for ADS Queries (OPEN)
**Bead:** `nls-finetune-scix-ybr`

| Feature ID | Description | Status |
|------------|-------------|--------|
| `eval-001` | Compute result-set overlap (Jaccard, Precision@N) | failing |
| `eval-002` | Syntactic validity rate metric | failing |
| `eval-003` | Feature-sliced reporting (author, pubdate, bibstem, object) | failing |
| `eval-004` | JSON artifacts to `data/datasets/evaluations/` | failing |
| `eval-005` | Comparison vs GPT-4o-mini baseline | failing |

**Acceptance Criteria:**
- Evaluation runs against ADS API
- Outputs structured JSON reports
- Semantic match rate ≥70%

### Phase 5: SciX Local Playground Integration (OPEN)
**Bead:** `nls-finetune-scix-f2w`
**Blocked by:** Phase 3 ✓

Integration with local SciX development environment at `~/ads-dev`.

**Environment:**
- Backend API: `http://localhost:5001` (adsws via Docker)
- Frontend: `http://localhost:8000` (nectar via Next.js)
- Branch: `sj/fine-tune` (checked out across all repos)
- Startup: `~/ads-dev/START_DEV.sh` + `cd ~/ads-dev/nectar && pnpm dev`

| Feature ID | Description | Status |
|------------|-------------|--------|
| `integ-001` | NL search component added to nectar UI | failing |
| `integ-002` | Modal endpoint proxy in nectar API routes | failing |
| `integ-003` | Query suggestion display with copy/apply buttons | failing |
| `integ-004` | Preview result count via ADS API | failing |
| `integ-005` | Feature flag for NL search toggle | failing |
| `integ-006` | Playwright e2e test for NL search flow | failing |

**Playwright Testing:**
- Config: `~/ads-dev/nectar/e2e/playwright.config.ts`
- Run: `cd ~/ads-dev/nectar && pnpm test:e2e`
- UI mode: `pnpm test:e2e:ui`

**Acceptance Criteria:**
- NL search works in local playground
- Playwright e2e tests pass
- <2s end-to-end latency (NL → query → preview count)

### Phase 6: Production Deployment (FUTURE)
**Blocked by:** Phase 4, Phase 5

| Feature ID | Description | Status |
|------------|-------------|--------|
| `prod-001` | Feature flag in production config | future |
| `prod-002` | Monitoring and alerting for Modal endpoint | future |
| `prod-003` | A/B testing framework integration | future |
| `prod-004` | Web UI deployed to scixplorer.org | future |

**Acceptance Criteria:**
- Web UI deployed to scixplorer.org
- Behind feature flag initially
- Syntax validity rate ≥95%

---

## Incremental Progress Protocol

For each coding session:

```
1. SELECT   → Pick first failing feature from features.json
2. IMPLEMENT → Write minimal code to pass verification
3. VERIFY   → Run `mise run verify`
4. UPDATE   → Set feature status to "passing" (only if verified)
5. COMMIT   → `git commit -m "feat(domain-001): <description>"`
6. SYNC     → `bd sync` to update beads
```

### Session End Checklist

Before ending ANY session:

```bash
[ ] 1. git status              # Check uncommitted changes
[ ] 2. git add <files>         # Stage changes
[ ] 3. bd sync                 # Commit beads changes
[ ] 4. git commit -m "..."     # Commit code
[ ] 5. bd sync                 # Commit any new beads
[ ] 6. git push                # Push to remote
```

---

## Inference Endpoint Reference

### Deployed Endpoint

```
URL: https://sjarmak--nls-finetune-serve-vllm-serve.modal.run
Model: llm
Format: OpenAI-compatible chat completions
```

### Request Format

```bash
curl -X POST https://sjarmak--nls-finetune-serve-vllm-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [
      {"role": "system", "content": "Convert natural language to ADS search query. Output JSON: {\"query\": \"...\"}"},
      {"role": "user", "content": "Query: papers by Hawking on black holes\nDate: 2025-12-15"}
    ],
    "max_tokens": 128
  }'
```

### Response Format

```json
{
  "choices": [{
    "message": {
      "content": "{\"query\": \"author:\\\"Hawking, S.\\\" abs:\\\"black holes\\\"}\"}"
    }
  }]
}
```

### Integration Notes

- System prompt: `Convert natural language to ADS search query. Output JSON: {"query": "..."}`
- User message: `Query: {natural_language}\nDate: {date}`
- Parse response as JSON to extract the `query` field
- Validate extracted query with `lint_query()` before use

---

## Testing Integration

### Verification Commands

```bash
mise run verify              # All checks (lint, types, JSON)
mise run verify-full         # All checks + frontend build
mise run test                # Unit tests
scix-finetune verify env     # Modal environment check
scix-finetune verify data    # Training data validation
```

### End-to-End Verification

Before marking any ML feature complete:

1. Run `scix-finetune dry-run train` (3 steps, ~$0.10)
2. Test inference endpoint with sample queries
3. Verify ADS API returns results for generated queries

### Local Playground Testing (ads-dev)

**Start the environment:**
```bash
# Terminal 1: Start backend services
~/ads-dev/START_DEV.sh

# Terminal 2: Start frontend
cd ~/ads-dev/nectar && pnpm dev
```

**Run Playwright e2e tests:**
```bash
cd ~/ads-dev/nectar

# Run all e2e tests
pnpm test:e2e

# Run with UI for debugging
pnpm test:e2e:ui

# Run specific test file
pnpm test:e2e -- e2e/tests/nl-search.spec.ts

# Run headed (visible browser)
pnpm test:e2e:headed
```

**Manual testing:**
1. Open http://localhost:8000
2. Enable NL search via feature flag (or toggle if UI exists)
3. Enter natural language query: "papers by Hawking on black holes"
4. Verify query suggestion appears
5. Click apply/copy and verify search results

---

## Common Failure Modes & Prevention

| Problem | Prevention |
|---------|-----------|
| Premature completion claims | Structured feature list with pass/fail tracking |
| Undocumented progress | Git commits + beads updates at session end |
| Features marked complete without testing | Mandatory `mise run verify` before status update |
| Time wasted on environment setup | `mise run install` automation |
| Lost context between sessions | beads CLI for multi-session tracking |
| Scope creep | ONE failing feature per session |

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Syntax validity rate | ≥95% | TBD (pending eval) |
| Semantic match rate | ≥70% | TBD (pending eval) |
| Inference latency (warm) | <500ms | ✅ <500ms |
| Training time | <15min | ✅ ~12.5min |
| Training cost | <$2 | ✅ ~$1.50 |
| Token accuracy | ≥90% | ✅ 93% |
| Training examples | ≥500 | ✅ 2,447 |

---

## Technical Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 19, TypeScript, Vite, Tailwind, shadcn/ui |
| **Backend** | FastAPI, Pydantic, Python 3.12 |
| **Training** | Modal (H100), TRL, LoRA, Qwen3-1.7B |
| **Validation** | ADS Search API |
| **Tools** | mise (runtimes), uv (Python), Bun (Node), beads (tasks) |

---

## API Keys Required

| Key | Required For | Source |
|-----|-------------|--------|
| `ADS_API_KEY` | Query validation, evaluation | [ADS Settings](https://ui.adsabs.harvard.edu/user/settings/token) |
| `MODAL_TOKEN_ID` | Training, deployment | [modal.com](https://modal.com/settings) |
| `MODAL_TOKEN_SECRET` | Training, deployment | [modal.com](https://modal.com/settings) |
| `ANTHROPIC_API_KEY` | Dataset generation | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | GPT-4o-mini comparison | [platform.openai.com](https://platform.openai.com/api-keys) |

---

## References

- [ADS Search Syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax)
- [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [SciXplorer.org](https://scixplorer.org/)
- [Modal Documentation](https://modal.com/docs)
