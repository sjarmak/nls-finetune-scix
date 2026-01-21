# Agent Instructions

## CRITICAL: Verification-First Workflow

**YOU MUST verify every change before marking it complete:**
1. Run `mise run verify` after ANY code change
2. Check `features.json` - work on ONE failing feature at a time
3. Use Chrome DevTools MCP for visual verification of web changes
4. Never claim success without test output proving it

## Project Context

This project fine-tunes models to convert natural language to **ADS/SciX scientific literature search queries**.

**Target:** Complementary search feature for [SciXplorer.org](https://scixplorer.org/)

**Example:**
- Input: "papers by Hawking on black hole radiation from the 1970s"
- Output: `author:"Hawking, S" abs:"black hole radiation" pubdate:[1970 TO 1979]`

## Key Domain Files

- `packages/finetune/src/finetune/domains/scix/fields.py` - ADS field definitions
- `packages/finetune/src/finetune/domains/scix/validate.py` - Query validation (offline lint + API)
- `packages/finetune/src/finetune/domains/scix/prompts.py` - Prompts for training/generation
- `packages/finetune/src/finetune/domains/scix/eval.py` - Evaluation via result-set overlap

## Session Start

1. Run `mise run verify` to verify environment
2. Check `features.json` for next task
3. Check `bd ready` for available beads

## Incremental Progress

1. Find first feature with `"status": "failing"` in `features.json`
2. Implement ONLY that feature
3. Run `mise run verify`
4. Update feature status to `"passing"` only after verification succeeds
5. Commit with message referencing feature

## Code Standards

**Python (packages/api/, packages/finetune/):**
- Format: `mise run format`
- Lint: `mise run lint`
- Type hints required

**TypeScript (packages/web/):**
- Strict mode enabled
- No unused variables/imports
- Use `@/` path alias

## Fine-Tuning CLI

```bash
scix-finetune --help           # Show all commands
scix-finetune verify env       # Check Modal setup
scix-finetune verify data      # Validate training data
scix-finetune dry-run train    # Test pipeline (3 steps)
scix-finetune train            # Run full training
```

See `docs/fine-tuning-cli.md` for full documentation.

## ADS Search Syntax Reference

Key fields the model must learn:
- `author:"Last, F"` - Author search
- `^author:"Last"` - First author only
- `abs:"topic"` - Abstract, title, keywords
- `title:"exact phrase"` - Title only
- `pubdate:[2020 TO 2023]` - Date range
- `bibstem:ApJ` - Journal abbreviation
- `object:M31` - Astronomical object
- `citation_count:[100 TO *]` - Highly cited

Full syntax: https://ui.adsabs.harvard.edu/help/search/search-syntax

## References

- `README.md` - Commands, project structure, setup
- `features.json` - Feature tracking
- `docs/fine-tuning-cli.md` - Fine-tuning CLI setup and usage
