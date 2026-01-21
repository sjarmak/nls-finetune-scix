# Dataset Guide

Training data for fine-tuning Qwen3-1.7B to convert natural language → Sourcegraph queries.

## Getting Started

### 1. Check Current Quality

```bash
# View latest evaluation results
nls-finetune eval report
```

Target metrics: **syntax ≥95%**, **semantic ≥70%**, **latency ≤100ms**

### 2. See What's Failing

```bash
# Start the web UI
mise run dev

# Open http://localhost:5173/evaluation
# Click on a run to see individual failures
```

Or via CLI:
```bash
cat data/datasets/evaluations/eval-*.json | jq '.results[] | select(.semantic_match == false)'
```

### 3. Improve the Data

**Option A: Add/edit gold examples** (quick, for specific fixes)
```bash
# Edit the file
vim data/datasets/raw/gold_examples.json

# Regenerate training data
mise run validate-data
```

**Option B: Regenerate from BigQuery** (for larger changes)
```bash
# 1. Extract fresh queries from BigQuery
mise run extract-queries

# 2. Generate NL descriptions with Claude
mise run generate-nl

# 3. Validate and split into train/val
mise run validate-data
```

### 4. Train & Evaluate

```bash
# Train new model (~12 min, ~$1.50)
nls-finetune train --run-name "my-improvement"

# Merge the LoRA adapter
nls-finetune merge --run-name "my-improvement"

# Deploy and evaluate
nls-finetune deploy --run-name "my-improvement"
nls-finetune eval run
nls-finetune eval report
```

### Quick Reference

| Task | Command |
|------|---------|
| See current quality | `nls-finetune eval report` |
| Regenerate JSONL | `mise run validate-data` |
| Full data pipeline | `mise run generate-data` |
| Train model | `nls-finetune train` |
| Test training setup | `nls-finetune dry-run train` |
| Evaluate model | `nls-finetune eval run` |

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Training examples | 969 |
| Validation examples | 108 |
| Gold (hand-curated) | 12 |
| Generated (Claude) | 1,077 |

### Directory Structure

```
data/
├── datasets/
│   ├── raw/                        # Source data
│   │   ├── gold_examples.json          # Hand-curated examples
│   │   ├── extracted_queries.json      # Queries from BigQuery
│   │   └── nl_pairs.json               # Claude-generated NL
│   │
│   ├── processed/                  # Training-ready data
│   │   ├── train.jsonl                 # Training set (90%)
│   │   ├── val.jsonl                   # Validation set (10%)
│   │   └── valid_pairs.json            # Intermediate format
│   │
│   └── evaluations/                # Results
│       ├── baseline-gpt-4o-mini.json
│       └── eval-*.json
```

### Category Distribution

| Category | Count | Description |
|----------|-------|-------------|
| repo_scoped | 300 | Searches within specific repos |
| commit_search | 150 | Git history/blame queries |
| lang_filtered | 150 | Language-specific searches |
| diff_search | 149 | Added/removed code searches |
| keyword_only | 100 | Simple text searches |
| symbol_search | 100 | Function/class definitions |
| file_filtered | 50 | File pattern searches |
| dependency_search | 48 | Package version searches |
| author_search | 30 | Who wrote what |

---

## Data Formats

### Gold Examples (`raw/gold_examples.json`)

```json
{
  "user_query": "Find auth middleware in django",
  "date": "2023-10-15",
  "expected_output": {
    "query": "repo:^github.com/django/django$ auth middleware"
  }
}
```

### Training Format (`processed/train.jsonl`)

```json
{
  "messages": [
    {"role": "system", "content": "Convert natural language to Sourcegraph search query..."},
    {"role": "user", "content": "Query: find auth middleware in django\nDate: 2025-12-15"},
    {"role": "assistant", "content": "{\"query\": \"repo:^github.com/django/django$ auth middleware\"}"}
  ]
}
```

---

## BigQuery Access

See [analytics-data-access.md](../docs/analytics-data-access.md) for setup.

### SQL Query Template

```sql
SELECT DISTINCT
    REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') as encoded_query
FROM `telligentsourcegraph.dotcom_events.search_urls`
WHERE url LIKE '%/search?%'
    AND REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') IS NOT NULL
    AND LENGTH(REGEXP_EXTRACT(url, r'[?&]q=([^&]+)')) BETWEEN 10 AND 500
ORDER BY RAND()
LIMIT {target}
```

Category filters in `scripts/extract_queries.py`:
- `commit_search`: `url LIKE '%type:commit%'`
- `diff_search`: `url LIKE '%type:diff%'`
- `symbol_search`: `url LIKE '%type:symbol%'`
- `lang_filtered`: `url LIKE '%lang:%'`
- `repo_scoped`: `url LIKE '%repo:%'`

---

## Common Issues

### Bad NL descriptions
Claude sometimes generates poor descriptions:
- Too literal: "search for X"
- Contains syntax: "find repo:github.com/..." (should be natural language only)

Fix: Edit `nl_pairs.json` or regenerate with adjusted prompts.

### Category imbalance
Adjust targets in `scripts/extract_queries.py`:
```python
CATEGORY_TARGETS = {
    "repo_scoped": 300,
    "commit_search": 150,
    # ...
}
```

### Validation failures
Check `mise run validate-data` output for:
- Duplicate queries
- Invalid JSON
- NL containing Sourcegraph syntax
