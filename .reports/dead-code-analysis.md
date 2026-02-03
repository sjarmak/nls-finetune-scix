# Dead Code Analysis Report

Generated: 2026-02-03

## Summary

| Category | Count |
|----------|-------|
| Unused imports (Python) | 42 |
| Unused variables (Python) | 5 |
| Orphaned module (no imports) | 1 |
| Root documentation bloat | 15 files (~4,100 lines) |
| Archived scripts | 11 files |

## Findings by Severity

### SAFE - Unused Imports (auto-fixable with `ruff --fix`)

These are unused `import` statements flagged by ruff F401/F841. They can be
auto-removed safely.

**In packages/ (test files):**
- `packages/finetune/tests/test_constrain.py:5` - `pytest` imported but unused
- `packages/finetune/tests/test_validate.py:3` - `pytest` imported but unused
- `packages/finetune/tests/test_validate.py:6` - `ConstraintValidationResult` imported but unused

**In scripts/ (42 total violations across scripts):**
- Various `re`, `sys`, `Path`, `Counter`, `deepcopy`, `asdict`, `field`, `json`, `os` imports
- 5 unused local variables (`field_groups`, `words`, `remaining`, `inp_lower`, `vocab_counts`, `train_result_phase1`, `train_result_phase2`)

### SAFE - Orphaned Module

- `packages/finetune/src/finetune/domains/scix/keyword_constraints.py`
  - Not imported by any file in the codebase
  - Contains UAT keyword validation functions
  - May have been superseded by `field_constraints.py`

### CAUTION - Root Documentation Bloat (15 files, ~4,100 lines)

These markdown/text files at the repo root are not referenced by any code:

| File | Lines | Purpose |
|------|-------|---------|
| activity.md | 11 | Unknown |
| CHANGES_SUMMARY.txt | 419 | Changelog notes |
| HANDOFF_US011_RESUME.md | 143 | Session handoff doc |
| METADATA_FILTERING_EXPLAINED.md | 419 | Internal explanation |
| QUICK_START.md | 232 | Quickstart guide |
| RALPH_DOCS_INDEX.md | 294 | Ralph agent index |
| RALPH_INDEX.md | 285 | Ralph agent index |
| RALPH_INTEGRATION_GUIDE.md | 449 | Ralph integration |
| RALPH_QUICK_START.txt | 59 | Ralph quickstart |
| RALPH_SETUP.md | 405 | Ralph setup guide |
| READINESS_CHECKLIST.md | 269 | Readiness checklist |
| TESTING_V4.md | 146 | Testing notes |
| TRAINING_ANALYSIS_US012.md | 99 | Training analysis |
| WORKFLOW_SUMMARY.md | 437 | Workflow summary |
| YOUR_QUESTIONS_ANSWERED.md | 330 | Q&A doc |

**Note:** These are documentation files, not code. Deletion won't break
anything but may lose useful reference material. Consider archiving instead.

### CAUTION - Archived Scripts (11 files)

`scripts/archive/` contains 11 Python scripts that are one-off fixes/tests
from previous iterations. They are not imported by any active code.

### NOT DEAD - Initially Flagged But Actually Used

| Module | Why it looked dead | Actual usage |
|--------|-------------------|--------------|
| `serve_pipeline.py` | 0 import refs | Modal deploy target |
| `serve_vllm.py` | 0 import refs | CLI deploy command |
| `serve_vllm_fp8_finetuned.py` | 0 import refs | CLI deploy command |
| `resolver.py` | 0 import refs via module name | Used by `pipeline.py`, `assembler.py` |
| `quantize_fp8.py` | Low refs | Used by CLI `training.py` |
| `@tanstack/react-router` | 0 direct imports | Used via `@tanstack` in App.tsx |

## Recommended Actions

### 1. Auto-fix unused imports (SAFE)

```bash
python -m ruff check --select F401,F841 --fix packages/ scripts/
```

### 2. Remove orphaned module (SAFE)

```bash
rm packages/finetune/src/finetune/domains/scix/keyword_constraints.py
```

### 3. Consider archiving root docs (OPTIONAL)

Move unreferenced docs to `docs/archive/` or delete if no longer needed.
