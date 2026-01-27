# PRD: NER Annotation Dashboard, SWEET Curation & Model Retraining

## Problem Statement

The SciX enrichment NER model achieves F1=0.9993 on synthetic test data but only F1=0.095 on real ADS abstracts. The evaluation report identifies three root causes:

1. **Noisy gold annotations**: Catalog-based auto-annotation produces garbage labels — SWEET contributes 70.8% of gold spans, many are common English words ("present", "from", "based on", "full")
2. **Template-only training data**: The model was trained on synthetic snippets and has never seen real scientific prose
3. **HTML markup in abstracts**: ADS abstracts contain `<SUB>`, `<SUP>`, etc. that corrupt tokenizer spans

## Solution

Three workstreams that unblock the path from CONDITIONAL GO to FULL GO:

### Workstream A: NER Annotation Dashboard
Build a single-file HTML annotation tool (adapting the existing `review_live.html` pattern) for human review of NER spans on real ADS abstracts. Pre-populate with model predictions + catalog matches so the reviewer corrects rather than annotates from scratch.

### Workstream B: SWEET Vocabulary Curation
Automated filtering of SWEET's 12,986 entries to remove common English words, short terms, and stopwords. Produces a curated vocabulary that eliminates the largest source of annotation noise.

### Workstream C: HTML Preprocessing & Mixed Retraining
Strip HTML from abstracts, retrain SciBERT on mixed synthetic + human-annotated data, re-evaluate.

## User Stories

### US-001: Strip HTML from ADS abstracts
**Priority**: 1 (prerequisite for everything else)

Write an HTML preprocessing function that strips `<SUB>`, `<SUP>`, `<I>`, `<B>`, `<A>`, and other HTML tags from abstract text while preserving the plain text content. Apply to the 100 sampled abstracts.

**Acceptance Criteria:**
- Function `strip_html_tags(text: str) -> str` in a reusable module
- Handles nested tags, self-closing tags, HTML entities (`&amp;`, `&lt;`, etc.)
- Preserves whitespace and text content
- Updates `data/evaluation/ads_sample_raw.jsonl` with cleaned abstracts (add `abstract_clean` field)
- Unit tests for edge cases (nested tags, entities, empty tags, malformed HTML)
- Typecheck passes

### US-002: Curate SWEET vocabulary — automated filtering
**Priority**: 1

Filter SWEET's 12,986 entries to remove noise. Produce a curated vocabulary file.

**Acceptance Criteria:**
- Load `topic_catalog_sweet.jsonl` from the US-001 pipeline run
- Remove entries where the canonical label is:
  - Fewer than 4 characters
  - In a standard English stopword list (NLTK or similar, 500+ words)
  - In a high-frequency English word list (top 5,000 common words)
  - A single common English word (not a multi-word scientific term)
- Produce `data/vocabularies/sweet_curated.jsonl` with surviving entries
- Produce `data/vocabularies/sweet_removed.jsonl` with removed entries + removal reason
- Produce `data/vocabularies/sweet_curation_report.json` with stats (original count, removed count, kept count, breakdown by removal reason)
- Log borderline cases (4-6 char terms that might be legitimate science) to `data/vocabularies/sweet_borderline.jsonl` for optional manual review
- Typecheck passes

### US-003: Re-annotate abstracts with curated vocabulary + HTML preprocessing
**Priority**: 2 (depends on US-001, US-002)

Re-run the catalog keyword annotation on the 100 sampled abstracts using cleaned text and curated SWEET vocabulary. This produces better auto-annotations for the dashboard.

**Acceptance Criteria:**
- Use `abstract_clean` (HTML-stripped) text from US-001
- Use curated SWEET vocabulary from US-002 (instead of full SWEET)
- Keep UAT, GCMD, ROR, planetary catalogs unchanged
- Re-annotate all 100 abstracts
- Produce `data/evaluation/ads_sample_reannotated.jsonl`
- Produce comparison stats: span count before vs after curation (expect 50-70% reduction in SWEET spans)
- Typecheck passes

### US-004: Run NER model predictions on cleaned abstracts
**Priority**: 2 (depends on US-001)

Run the trained SciBERT model on the HTML-cleaned abstracts to generate model predictions that will be shown alongside auto-annotations in the dashboard.

**Acceptance Criteria:**
- Load trained model from `models/enrichment_ner/` (or wherever US-006 saved it)
- Run inference on all 100 cleaned abstracts
- Produce `data/evaluation/ads_sample_predictions.jsonl` with model-predicted spans
- Each record has: bibcode, spans (with surface, start, end, type, confidence)
- Confidence is the softmax probability of the predicted BIO label
- Typecheck passes

### US-005: Build NER annotation dashboard HTML
**Priority**: 2 (depends on US-003, US-004)

Build a single-file HTML annotation tool that displays ADS abstracts with highlighted spans from two sources (auto-annotation and model predictions), and lets the reviewer accept, reject, edit, or add spans.

**Acceptance Criteria:**
- Single HTML file at `data/evaluation/review_ner_annotations.html`
- Loads data inline (embedded JSON) from the re-annotated + model prediction files
- For each abstract, shows:
  - Bibcode, title, domain category, citation count in header
  - Abstract text with color-coded span highlights:
    - Blue: auto-annotation spans (from curated catalogs)
    - Green: model prediction spans
    - Purple: spans present in both (agreement)
    - Gold: user-added spans
  - Span list panel showing all spans with accept/reject toggles
  - Text selection creates a new span (with type dropdown: topic/entity/institution/author/date_range)
  - Click on highlighted span toggles accept/reject
  - Notes field per abstract
- Stats dashboard: total abstracts, reviewed count, pending count, span counts by type
- Filter controls: domain category, review status (pending/reviewed/skipped), has-disagreements
- localStorage persistence (key: `nls-ner-annotations-v1`)
- Export button produces JSONL matching `enrichment_labels.jsonl` schema
- Follows the same visual style as `review_live.html` (light theme, card layout, stat cards)
- Typecheck passes (for any generation script)

### US-006: Generate annotation dashboard with embedded data
**Priority**: 3 (depends on US-005)

Write a Python script that generates the annotation dashboard HTML with real data embedded.

**Acceptance Criteria:**
- Script at `scripts/generate_annotation_dashboard.py`
- Reads: `ads_sample_reannotated.jsonl`, `ads_sample_predictions.jsonl`, annotation guide
- Embeds data as inline JSON in the HTML template
- Produces `data/evaluation/review_ner_annotations.html`
- Runnable: `python scripts/generate_annotation_dashboard.py`
- Generated HTML opens in browser and is functional
- Typecheck passes

### US-007: Sample additional ADS abstracts (200 more)
**Priority**: 3

Expand the annotation corpus from 100 to 300 abstracts for better training coverage.

**Acceptance Criteria:**
- Extend `scripts/sample_ads_abstracts.py` to support `--count` and `--exclude` flags
- Sample 200 new abstracts (50 per domain) excluding existing 100 bibcodes
- Apply HTML cleaning from US-001
- Auto-annotate with curated vocabulary from US-002
- Run model predictions from US-004
- Append to existing files or produce `ads_sample_batch2_*.jsonl`
- Regenerate dashboard HTML with all 300 abstracts
- Typecheck passes

### US-008: Export human annotations to training format
**Priority**: 3

Write a script that converts exported dashboard annotations into the enrichment training format (matching `enrichment_labels.jsonl` schema) for mixed-data retraining.

**Acceptance Criteria:**
- Script at `scripts/export_annotations_to_training.py`
- Reads exported JSON from the dashboard (localStorage export)
- Converts to `EnrichmentRecord` format with spans, domain_tags, provenance
- Validates byte offsets (`text[start:end] == surface`) for all spans
- Produces `data/datasets/enrichment/human_annotated_train.jsonl` and `*_val.jsonl` (90/10 split)
- Reports: total records, spans by type, spans by vocabulary
- Typecheck passes

### US-009: Mixed-data retraining script
**Priority**: 4 (depends on US-008, needs human annotations to exist)

Extend the training script to support curriculum learning: pre-train on synthetic data, then fine-tune on human annotations.

**Acceptance Criteria:**
- Extend `scripts/train_enrichment_model.py` with `--curriculum` flag
- When `--curriculum` is set:
  1. Train on synthetic data for N epochs (configurable, default 3)
  2. Switch to human-annotated data for M epochs (configurable, default 5)
  3. Use lower learning rate for phase 2 (default 5e-6 vs 2e-5)
- Support `--human-train` and `--human-val` paths for human annotation files
- Log phase transitions and metrics per phase
- Save best model checkpoint (by val F1) from phase 2
- Typecheck passes

### US-010: Re-evaluate with mixed-trained model
**Priority**: 4 (depends on US-009)

Run the full evaluation pipeline with the mixed-trained model and produce an updated report.

**Acceptance Criteria:**
- Run evaluation on: synthetic test set, real-world annotated test set (from dashboard export)
- Compare to: keyword baseline, synthetic-only model, previous results
- Produce `reports/enrichment_eval_v2.json` and `reports/enrichment_eval_v2.md`
- Include: synthetic-to-real gap delta (should shrink), per-vocabulary breakdown, per-domain breakdown
- Go/no-go assessment with updated thresholds
- Typecheck passes

## Architecture Notes

### Annotation Dashboard Data Flow
```
ads_sample_raw.jsonl
    |
    v
[US-001: strip HTML] --> ads_sample_raw.jsonl (+ abstract_clean field)
    |
    v
[US-002: curate SWEET] --> sweet_curated.jsonl
    |
    v
[US-003: re-annotate] --> ads_sample_reannotated.jsonl
    |
[US-004: model predict] --> ads_sample_predictions.jsonl
    |
    v
[US-006: generate dashboard] --> review_ner_annotations.html
    |
    v
[Human reviews in browser] --> localStorage
    |
    v
[Export JSON] --> human_annotations.json
    |
    v
[US-008: convert] --> human_annotated_train.jsonl + human_annotated_val.jsonl
    |
    v
[US-009: curriculum train] --> models/enrichment_ner_v2/
    |
    v
[US-010: evaluate] --> enrichment_eval_v2.md
```

### Dashboard UI Architecture
- Adapt `review_live.html` card layout and event delegation pattern
- Add text selection API for span creation (window.getSelection())
- Use CSS `<mark>` elements with data attributes for span highlighting
- Color-code by source: auto-annotation (blue), model prediction (green), agreement (purple), user-added (gold)
- Span conflicts: when auto and model disagree, show both with visual indicator

### Key Constraints
- Dashboard must be a single HTML file (no build step, no server)
- All data embedded as inline JSON (generated by Python script)
- localStorage for persistence, JSON download for export
- Target: 300 abstracts reviewable in 4-6 hours of focused annotation work
- Training budget: Google Colab Pro (T4/A100 GPU)
