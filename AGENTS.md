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
- `bibstem:"ApJ"` - Journal abbreviation (MUST be quoted)
- `object:M31` - Astronomical object
- `citation_count:[100 TO *]` - Highly cited

Full syntax: https://ui.adsabs.harvard.edu/help/search/search-syntax

## ADS Operator Syntax

ADS supports function-like operators that take query expressions as arguments:

| Operator | Purpose | Example |
|----------|---------|---------|
| `citations()` | Find papers that cite the given search results | `citations(abs:"gravitational wave")` |
| `references()` | Find papers referenced by the given search results | `references(abs:"supernova")` |
| `trending()` | Find currently popular papers matching the query | `trending(abs:"exoplanet")` |
| `useful()` | Find high-utility papers matching the query | `useful(abs:"cosmology")` |
| `similar()` | Find textually similar papers | `similar(abs:"black hole merger")` |
| `reviews()` | Find review articles matching the query | `reviews(abs:"magnetar")` |

### Operator Syntax Rules

**CRITICAL:** All field values inside operators MUST be quoted:

| ❌ Bad | ✅ Good |
|--------|---------|
| `citations(abs:cosmology)` | `citations(abs:"cosmology")` |
| `trending(abs:exoplanet)` | `trending(abs:"exoplanet")` |
| `references(abs:magnetar)` | `references(abs:"magnetar")` |
| `similar(abs:JWST)` | `similar(abs:"JWST")` |
| `useful(abs:photometry)` | `useful(abs:"photometry")` |

**Avoid malformed parentheses:**

| ❌ Bad | ✅ Good |
|--------|---------|
| `trending(abs:(exoplanets))` | `trending(abs:"exoplanets")` |
| `useful(abs:(M31))` | `useful(abs:"M31")` |

**Nested operators require quoting at all levels:**

| ❌ Bad | ✅ Good |
|--------|---------|
| `useful(citations(abs:cosmology))` | `useful(citations(abs:"cosmology"))` |

### Audit Script

Run `python scripts/audit_operators.py` to find malformed operator patterns in training data.

Run `python scripts/fix_operators.py` to fix them.

## ADS Field Constraints

### Overview

ADS has six fields with **constrained vocabularies** that the model must learn. Invalid values cause query failures or unexpected results. See [`field_constraints.py`](packages/finetune/src/finetune/domains/scix/field_constraints.py) for the authoritative list.

### FIELD_ENUMS Reference

| Field | Valid Values | Description |
|-------|--------------|-------------|
| `doctype` | 22 values: `abstract`, `article`, `book`, `bookreview`, `catalog`, `circular`, `editorial`, `eprint`, `erratum`, `inbook`, `inproceedings`, `mastersthesis`, `misc`, `newsletter`, `obituary`, `phdthesis`, `pressrelease`, `proceedings`, `proposal`, `software`, `talk`, `techreport` | Document type classification |
| `property` | 21 values: `ads_openaccess`, `author_openaccess`, `eprint_openaccess`, `pub_openaccess`, `openaccess`, `article`, `nonarticle`, `refereed`, `notrefereed`, `eprint`, `inproceedings`, `software`, `catalog`, `associated`, `data`, `esource`, `inspire`, `library_catalog`, `presentation`, `toc`, `ocr_abstract` | Record properties (peer review, open access, etc.) |
| `database` | 3 values: `astronomy`, `physics`, `general` | Database collection |
| `bibgroup` | 53 values: `HST`, `JWST`, `Spitzer`, `Chandra`, `XMM`, `GALEX`, `Kepler`, `K2`, `TESS`, `FUSE`, `IUE`, `EUVE`, `Copernicus`, `IRAS`, `WISE`, `NEOWISE`, `Fermi`, `Swift`, `RXTE`, `NuSTAR`, `SOHO`, `STEREO`, `SDO`, `ESO/Telescopes`, `CFHT`, `Gemini`, `Keck`, `VLT`, `Subaru`, `NOAO`, `NOIRLab`, `CTIO`, `KPNO`, `Pan-STARRS`, `SDSS`, `2MASS`, `UKIRT`, `ALMA`, `JCMT`, `APEX`, `ARECIBO`, `VLA`, `VLBA`, `GBT`, `LOFAR`, `MeerKAT`, `SKA`, `Gaia`, `Hipparcos`, `CfA`, `NASA PubSpace`, `LISA`, `LIGO` | Telescope/observatory bibliographic groups |
| `esources` | 8 values: `PUB_PDF`, `PUB_HTML`, `EPRINT_PDF`, `EPRINT_HTML`, `AUTHOR_PDF`, `AUTHOR_HTML`, `ADS_PDF`, `ADS_SCAN` | Electronic source types |
| `data` | 24 values: `ARI`, `BICEP2`, `Chandra`, `CXO`, `ESA`, `ESO`, `GCPD`, `GTC`, `HEASARC`, `Herschel`, `INES`, `IRSA`, `ISO`, `KOA`, `MAST`, `NED`, `NExScI`, `NOAO`, `PDS`, `SIMBAD`, `Spitzer`, `TNS`, `VizieR`, `XMM` | Data archive sources |

### Why Bare Fields Are Problematic

**Problem:** Unquoted field values teach the model incorrect syntax patterns.

When training data contains bare values like `bibstem:ApJ`, the model learns to omit quotes. This works for simple cases but fails for values with special characters (spaces, slashes, colons).

**Impact:** Model generalizes "don't quote" and produces invalid queries for edge cases.

### Good vs Bad Training Examples

| Field | ❌ Bad (bare) | ✅ Good (quoted) | Why |
|-------|---------------|------------------|-----|
| `bibstem` | `bibstem:ApJ` | `bibstem:"ApJ"` | Teaches consistent quoting; needed for `bibstem:"A&A"` |
| `author` | `author:Hawking` | `author:"Hawking, S"` | Real queries need "Last, F" format |
| `title` | `title:black holes` | `title:"black holes"` | Multi-word phrases require quotes |
| `doctype` | `doctype:article` | `doctype:article` | Enum values can be bare (no special chars) |
| `property` | `property:refereed` | `property:refereed` | Enum values can be bare |
| `database` | `database:astrophysics` | `database:astronomy` | Invalid value! Use only valid enums |

### Common Model Hallucinations

The model frequently generates these invalid values. [`constrain.py`](packages/finetune/src/finetune/domains/scix/constrain.py) removes them at inference time:

| Field | Invalid Hallucination | Valid Alternative | Notes |
|-------|----------------------|-------------------|-------|
| `doctype` | `journal` | `article` | "journal" is not a doctype |
| `doctype` | `paper` | `article` | Generic term, not in enum |
| `doctype` | `publication` | `article` | Too broad |
| `doctype` | `thesis` | `phdthesis` or `mastersthesis` | Must be specific |
| `property` | `peerreviewed` | `refereed` | Common alias, but invalid |
| `property` | `peer-reviewed` | `refereed` | Hyphenated variant |
| `property` | `open_access` | `openaccess` | Underscore variant |
| `database` | `astrophysics` | `astronomy` | Common confusion |
| `database` | `astro` | `astronomy` | Abbreviated |
| `bibgroup` | `Hubble` | `HST` | Use telescope code |
| `bibgroup` | `Webb` | `JWST` | Use telescope code |
| `bibgroup` | `Sloan` | `SDSS` | Use survey code |

### How constrain_query_output() Fixes Hallucinations

The post-processing filter in `constrain.py`:

1. **Scans** query for each constrained field type
2. **Validates** each value against FIELD_ENUMS
3. **Removes** invalid field:value pairs entirely
4. **Logs** warnings for each removed field
5. **Cleans up** orphaned operators (AND, OR, NOT)

```python
# Example: Model outputs invalid doctype
query = 'doctype:journal property:refereed abs:"exoplanets"'
clean = constrain_query_output(query)
# Result: 'property:refereed abs:"exoplanets"'
# Logged: "Removed invalid doctype value: 'journal'"
```

### Future: CanonicalAffiliations

ADS maintains a `CanonicalAffiliations` dataset for institutional affiliation normalization. This could enable future affiliation-based training:

```
aff:"Harvard-Smithsonian Center for Astrophysics"
inst:"CfA"
```

The canonical forms ensure consistent matching across variations (e.g., "CfA", "Center for Astrophysics", "Harvard-Smithsonian"). Consider adding affiliation constraints when training data includes `aff:` or `inst:` fields.

### ADS Documentation Sources

- **Search Syntax**: https://ui.adsabs.harvard.edu/help/search/search-syntax
- **Document Types**: https://ui.adsabs.harvard.edu/help/search/search-syntax#document-type
- **Properties**: https://ui.adsabs.harvard.edu/help/search/search-syntax#properties
- **Bibgroups FAQ**: https://ui.adsabs.harvard.edu/help/data_faq/Bibgroups
- **Solr Schema**: https://github.com/adsabs/montysolr/blob/main/deploy/adsabs/server/solr/collection1/conf/schema.xml

## Training Iteration Decision Tree (US-015)

When tests fail in US-011-014, follow this decision tree:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RUN TEST SUITE                                  │
│               (US-011, US-012, US-013, US-014)                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   ALL TESTS PASS? │
                    └─────────┬─────────┘
                              │
           ┌──────────────────┴──────────────────┐
           │                                      │
     ┌─────▼─────┐                          ┌────▼────┐
     │    YES    │                          │   NO    │
     └─────┬─────┘                          └────┬────┘
           │                                      │
           ▼                                      ▼
   ┌───────────────┐                   ┌────────────────────┐
   │ 1. Mark done  │                   │ ANALYZE FAILURE:   │
   │ 2. Merge to   │                   │ Is it a DATA issue │
   │    main       │                   │ or MODEL issue?    │
   │ 3. Update     │                   └────────┬───────────┘
   │    progress   │                            │
   └───────────────┘               ┌────────────┴────────────┐
                                   │                          │
                            ┌──────▼──────┐            ┌──────▼──────┐
                            │ DATA ISSUE  │            │ MODEL ISSUE │
                            └──────┬──────┘            └──────┬──────┘
                                   │                          │
                                   ▼                          ▼
                        ┌─────────────────────┐   ┌──────────────────────┐
                        │ Symptoms:           │   │ Symptoms:            │
                        │ - Wrong patterns    │   │ - Overfitting        │
                        │ - Missing coverage  │   │ - Underfitting       │
                        │ - Malformed syntax  │   │ - Slow convergence   │
                        │ - Bad field values  │   │ - High loss          │
                        └─────────┬───────────┘   └──────────┬───────────┘
                                  │                          │
                                  ▼                          ▼
                        ┌─────────────────────┐   ┌──────────────────────┐
                        │ FIX:                │   │ FIX:                 │
                        │ 1. Add examples to  │   │ 1. Adjust hyperparams│
                        │    gold_examples    │   │    in train.py:      │
                        │ 2. Run data pipeline│   │    - learning_rate   │
                        │ 3. Goto US-008      │   │    - epochs          │
                        └─────────────────────┘   │    - batch_size      │
                                                  │ 2. Goto US-009       │
                                                  └──────────────────────┘
```

### Data Issue Indicators
- Model outputs bare field values (e.g., `bibstem:ApJ` instead of `bibstem:"ApJ"`)
- Model hallucinate initials (e.g., `author:"kelbert, M"`)
- Missing operator quoting (e.g., `citations(abs:cosmology)`)
- Invalid field enum values (e.g., `doctype:journal`)

### Model Issue Indicators
- Training loss doesn't decrease
- Token accuracy plateaus below 85%
- Model repeats tokens (e.g., `abs:abs:abs:abs:...`)
- All outputs look similar regardless of input

### Ralph Loop Reference
See `progress.txt` for full iteration history and metrics.

## Hybrid NER Pipeline Architecture (NEW)

### Overview

The project has transitioned from end-to-end fine-tuned generation to a **hybrid NER + template assembly pipeline**. This eliminates malformed operator syntax caused by the model conflating natural language with ADS operators.

### Why End-to-End Generation Failed

The fine-tuned Qwen3-1.7B model learned to inject operator names into field values:
- Input: "papers about references in stellar spectra"
- **Bad output**: `citations(abs:referencesabs:stellar spectra)`
- **Root cause**: Training data conflated NL words with ADS operators

### New Pipeline Architecture

```
User NL → [NER Extractor] → IntentSpec → [Retrieval] → [Assembler] → Valid Query
                                              ↓
                                     gold_examples.json
                                              ↓
                                     (optional LLM for paper resolution)
```

### IntentSpec Fields

| Field | Type | Description |
|-------|------|-------------|
| `free_text_terms` | list[str] | Topic phrases for abs:/title: |
| `authors` | list[str] | Author names |
| `year_from`, `year_to` | int | Year range |
| `doctype` | set[str] | Must be in FIELD_ENUMS |
| `property` | set[str] | Must be in FIELD_ENUMS |
| `database` | set[str] | Must be in FIELD_ENUMS |
| `bibgroup` | set[str] | Must be in FIELD_ENUMS |
| `operator` | str | None or one of: citations, references, trending, useful, similar, reviews |
| `operator_target` | str | Optional bibcode/identifier for operator |

### CRITICAL: Operator Gating Rules

**Set operator ONLY when explicit patterns match:**

| Pattern | Operator Set |
|---------|--------------|
| "papers citing X", "cited by", "who cited" | `citations` |
| "references of", "papers referenced by", "bibliography of" | `references` |
| "similar to this paper", "like <paper>" | `similar` |
| "trending papers on", "what's hot in" | `trending` |
| "most useful papers on" | `useful` |
| "reviews of", "review articles on" | `reviews` |

**Do NOT set operator for:**
- "citing" as a topic (e.g., "papers about citing practices")
- "references" as a topic (e.g., "papers about references in spectra")
- Generic use of these words without explicit operator intent

### Synonym Maps for Enum Values

| User Input | Maps To |
|------------|---------|
| "refereed", "peer reviewed" | `property:refereed` |
| "open access", "oa" | `property:openaccess` |
| "arxiv", "preprint" | `property:eprint` |
| "Hubble" | `bibgroup:HST` |
| "Webb", "James Webb" | `bibgroup:JWST` |
| "Sloan" | `bibgroup:SDSS` |

### Key Files (Hybrid Pipeline)

| File | Purpose |
|------|---------|
| `intent_spec.py` | IntentSpec dataclass definition |
| `ner.py` | Rules-based NER with operator gating |
| `retrieval.py` | Few-shot retrieval from gold_examples |
| `assembler.py` | Deterministic query assembly |
| `resolver.py` | Optional LLM for paper reference resolution |
| `pipeline.py` | Main pipeline orchestration |

### Testing Requirements

1. **Operator gating tests**: Verify "citing" as topic ≠ citations() operator
2. **Enum validation tests**: All values validated against FIELD_ENUMS
3. **Playwright E2E tests**: Browser tests on localhost:8000
4. **Regression tests**: Known failure patterns never appear in output

## References

- `README.md` - Commands, project structure, setup
- `features.json` - Feature tracking
- `docs/fine-tuning-cli.md` - Fine-tuning CLI setup and usage
- `docs/HYBRID_PIPELINE.md` - Hybrid pipeline architecture (created by US-011)
