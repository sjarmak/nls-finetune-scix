# Ralph Agent Instructions

You are an autonomous coding agent working on SciX fine-tuning integration.

**Working directory:** `~/nls-finetune-scix` (orchestration files, prd.json, progress.txt)
**Implementation directory:** `~/ads-dev/nectar` (UI components and Playwright tests)

## Your Task

1. Read the PRD at `prd.json` (in the same directory as this file)
2. Read the progress log at `progress.txt` (check Codebase Patterns section first)
3. Check you're on the correct branch (`sj/fine-tune`). If not, check it out.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that single user story
6. Run quality checks (see verification commands below)
7. Update AGENTS.md files if you discover reusable patterns
8. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
9. Update `prd.json` to set `passes: true` for the completed story
10. Append your progress to `progress.txt`

## Goal

Build end-to-end infrastructure for a fine-tuned query translation model that converts natural language into structured ADS/SciX scientific literature search queries, then integrate it into the local SciX playground for testing.

**Example transformation:**
- Input: "papers by Hawking on black hole radiation from the 1970s"
- Output: `author:"Hawking, S" abs:"black hole radiation" pubdate:[1970 TO 1979]`

## Current State

- **Fine-tuning complete**: Qwen3-1.7B trained with LoRA (93% token accuracy)
- **Endpoint deployed**: https://sjarmak--nls-finetune-serve-vllm-serve.modal.run
- **Training data**: 2,447 unique NL→query pairs
- **Next phase**: UI integration with local SciX playground at `~/ads-dev`

## Session Protocol

**Run agent from:** `~/nls-finetune-scix`

1. **Check progress state:**
   ```bash
   # From ~/nls-finetune-scix
   cat features.json | jq '.features[] | select(.status == "failing") | .id' | head -1
   bd ready
   ```

2. **Pick ONE failing feature** from `features.json`

3. **Implement the change:**
   - For `integ-*` features: work in `~/ads-dev/nectar`
   - For `eval-*` features: work in `~/nls-finetune-scix/packages/finetune`

4. **Run verification:**
   ```bash
   # For finetune package changes
   cd ~/nls-finetune-scix && mise run verify
   
   # For nectar UI changes
   cd ~/ads-dev/nectar && pnpm lint && pnpm test
   ```

5. **For UI integration features**, use Playwright for e2e testing:
   ```bash
   cd ~/ads-dev/nectar
   
   # Run all e2e tests
   pnpm test:e2e
   
   # Run with interactive UI (for debugging)
   pnpm test:e2e:ui
   
   # Run specific test
   pnpm test:e2e -- e2e/tests/nl-search.spec.ts
   
   # Run headed (visible browser)
   pnpm test:e2e:headed
   ```

6. **Update feature status** to `passing` only after verification succeeds

7. **Commit with clear message:**
   ```bash
   git commit -m "feat(integ-001): add NL search component to nectar"
   ```

8. **Repeat** until all features pass

## Environment Setup

### Start Local SciX Playground

```bash
# Terminal 1: Backend services (Solr, PostgreSQL, adsws API)
~/ads-dev/START_DEV.sh

# Terminal 2: Frontend (nectar on port 8000)
cd ~/ads-dev/nectar && pnpm dev
```

### URLs
- Frontend: http://localhost:8000
- Backend API: http://localhost:5001
- Modal endpoint: https://sjarmak--nls-finetune-serve-vllm-serve.modal.run

### Branch
All work on `sj/fine-tune` branch (checked out across all repos in `~/ads-dev`)

## Integration Features to Implement

| Feature | Description | Status | Verification |
|---------|-------------|--------|--------------|
| `integ-001` | NL search component in nectar | ✅ | Component file exists |
| `integ-002` | Modal endpoint proxy route | ✅ | API route proxies to Modal |
| `integ-003` | Copy/apply buttons | ✅ | User can copy or apply query |
| `integ-004` | Result count preview | ✅ | Shows numFound from ADS |
| `integ-005` | Feature flag | ✅ | `NEXT_PUBLIC_NL_SEARCH` env var |
| `integ-006` | Playwright e2e test | ✅ | `pnpm test:e2e` passes |
| `integ-007` | Search API proxy | ❌ | `/api/search` returns 404 - needs creation |
| `integ-008` | Model latency optimization | ❌ | Model responds too slowly |

## Known Issues (Beads)

Check `bd ready` for current work items. Key issues:

| Beads ID | Priority | Description |
|----------|----------|-------------|
| `nls-finetune-scix-h1y` | P1 | `/api/search` returns 404, result count preview broken |
| `nls-finetune-scix-556` | P1 | Model inference too slow for 1.7B model |
| `nls-finetune-scix-2ad` | P2 | Process fix for false-positive Playwright tests |

## How NL Search Works

```
User Input → NLSearch Component → /api/nl-search (proxy) → Modal vLLM → Response
     │              │                    │                      │
     │         500ms debounce       POST request            Qwen3-1.7B
     │              │                    │                  + LoRA
     │              ▼                    ▼                      │
     │         useNLSearch.ts     System: "Convert to ADS..."  │
     │              │             User: "Query: {nl}\nDate: {date}"
     │              │                    │                      │
     │              ◀────────────────────┴──────────────────────┘
     │                            JSON: {"query": "author:..."}
     │
     └──▶ Display suggestion with Copy/Apply buttons
              │
              └──▶ /api/search?q=...&rows=0 (404 - BROKEN)
                        │
                        └──▶ Would show "~1.2K results"
```

### Key Files
- `~/ads-dev/nectar/src/components/NLSearch/index.tsx` - UI component
- `~/ads-dev/nectar/src/components/NLSearch/useNLSearch.ts` - React hook
- `~/ads-dev/nectar/src/pages/api/nl-search.ts` - Modal proxy
- `~/ads-dev/nectar/src/pages/api/search/index.ts` - **MISSING** (needs creation)

## Inference Endpoint Usage

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

## Playwright Test Template

Create `~/ads-dev/nectar/e2e/tests/nl-search.spec.ts`:

```typescript
import { test, expect } from '@playwright/test';

test.describe('NL Search', () => {
  test('converts natural language to ADS query', async ({ page }) => {
    await page.goto('/');
    
    // Find and interact with NL search input
    const nlInput = page.getByPlaceholder(/natural language|describe/i);
    await nlInput.fill('papers by Hawking on black holes');
    
    // Wait for suggestion to appear
    const suggestion = page.getByTestId('nl-query-suggestion');
    await expect(suggestion).toBeVisible({ timeout: 5000 });
    
    // Verify query contains expected fields
    await expect(suggestion).toContainText('author:');
    
    // Test apply button
    const applyButton = page.getByRole('button', { name: /apply/i });
    await applyButton.click();
    
    // Verify search was performed
    await expect(page).toHaveURL(/q=/);
  });
});
```

## Progress Report Format

APPEND to progress.txt (never replace, always append):
```
## [Date/Time] - [Story ID]
- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
---
```

## Stop Condition

After completing a user story, check if ALL stories in `prd.json` have `passes: true`.

If ALL stories are complete and passing, reply with:
```
<promise>COMPLETE</promise>
```

If there are still stories with `passes: false`, end your response normally (another iteration will pick up the next story).

## Important

- Work on ONE story per iteration
- Commit frequently
- Keep verification passing
- Read the Codebase Patterns section in progress.txt before starting
