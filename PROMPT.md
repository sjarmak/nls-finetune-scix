# Ralph Agent Instructions

You are an autonomous coding agent working on a software project.

## Your Task

1. Read the PRD at `prd.json` (in the same directory as this file)
2. Read the progress log at `progress.txt` (check Codebase Patterns section first)
3. Check you're on the correct branch from PRD `branchName`. If not, check it out or create from main.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that single user story
6. Run quality checks (e.g., typecheck, lint, test - use whatever your project requires)
7. Update AGENTS.md files if you discover reusable patterns (see below)
8. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
9. Update the PRD to set `passes: true` for the completed story
10. Append your progress to `progress.txt`

## Project Context

This project builds a **hybrid NER + fine-tuned model pipeline** for converting natural language to ADS/SciX scientific literature search queries.

**Problem solved:** End-to-end fine-tuned models generate malformed queries because they conflate natural language words like "citing" or "references" with ADS operator syntax, producing outputs like `citations(abs:referencesabs:...)`.

**Solution architecture:**
1. **Intent Extraction (NER)** - Parse user NL to extract structured `IntentSpec`
2. **Few-shot Retrieval** - Match against `gold_examples.json` for pattern guidance
3. **Template Assembly** - Build query deterministically from validated building blocks
4. **Fine-tuned Model** - Used ONLY as fallback for ambiguous entity resolution

## Key Files

- `prd.json` - User stories (update `passes` field when complete)
- `progress.txt` - Append learnings after each iteration
- `AGENTS.md` - Update with patterns and gotchas discovered
- `packages/finetune/src/finetune/domains/scix/` - Core domain logic
- `packages/finetune/src/finetune/domains/scix/field_constraints.py` - FIELD_ENUMS for validation
- `packages/finetune/src/finetune/domains/scix/constrain.py` - Post-processing filter
- `data/datasets/raw/gold_examples.json` - 400+ curated gold examples
- `~/ads-dev/nectar/` - Frontend/API integration

## Quality Checklist

Before marking a story `passes: true`:

- [ ] All acceptance criteria implemented
- [ ] Code passes: `mise run lint`
- [ ] Tests pass: `mise run test` (if applicable)
- [ ] No regressions: `mise run verify`
- [ ] Commit message references story ID (e.g., "US-001: Define IntentSpec")
- [ ] Update AGENTS.md with new patterns discovered
- [ ] Append summary to progress.txt

## Browser Testing (Required for UI Stories)

For any story that changes UI or integration, you MUST verify it works in the browser:

1. Start nectar dev server: `cd ~/ads-dev/nectar && pnpm dev`
2. Open Playwright or browser at `http://localhost:8000`
3. Test NL search input with test cases from acceptance criteria
4. Verify:
   - Generated query syntax is valid (no `citationsabs:` patterns)
   - Parentheses are balanced
   - Results render (API returns 200)
5. Take screenshots if helpful for documentation

A frontend story is NOT complete until browser verification passes.

## Progress Report Format

APPEND to progress.txt (never replace, always append):
```
## [Date/Time] - [Story ID]
Thread: https://ampcode.com/threads/$AMP_CURRENT_THREAD_ID
- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered (e.g., "this codebase uses X for Y")
  - Gotchas encountered (e.g., "don't forget to update Z when changing W")
  - Useful context (e.g., "the evaluation panel is in component X")
---
```

## Consolidate Patterns

If you discover a **reusable pattern** that future iterations should know, add it to the `## Codebase Patterns` section at the TOP of progress.txt (create it if it doesn't exist).

## Update AGENTS.md Files

Before committing, check if any edited files have learnings worth preserving in nearby AGENTS.md files:

1. **Identify directories with edited files**
2. **Check for existing AGENTS.md**
3. **Add valuable learnings** - API patterns, gotchas, dependencies, testing approaches

## Important Constraints

1. **Latency Budget**: < 500ms total for rule-based path (no LLM)
2. **FIELD_ENUMS Validation**: All doctype/property/database/bibgroup values must be in enums
3. **Operator Gating**: Only set operators when explicit patterns match (not generic words)
4. **No LLM for Operators**: Operators/fields become enum decisions, not generated text

## Stop Condition

After completing a user story, check if ALL stories have `passes: true`.
If ALL stories are complete and passing, reply with:
<promise>COMPLETE</promise>

If there are still stories with `passes: false`, end your response normally.

## Notes

- Work on ONE story per iteration
- Each iteration is fresh context - progress.txt and git history are your memory
- Commit frequently
- Keep CI green
- Read the Codebase Patterns section in progress.txt before starting
