# PRD: ADS/SciX Data Model Coverage Enhancement

## Introduction

Enhance the natural language to ADS/SciX query translation system by achieving comprehensive coverage of the ADS data model. Currently, the training data focuses heavily on common patterns (author, topic, year) but underrepresents many powerful search capabilities (collections, bibgroups, positional search, full-text, operator variations). This project audits the complete data model, generates targeted training examples for underrepresented elements, and improves interpretation flexibility for operator triggering.

## Goals

- Achieve >90% coverage of ADS data model elements in training data
- Improve operator interpretation flexibility (e.g., "papers that cite" should trigger citations(), not just "papers citing")
- Generate grounded training data using actual indexed values (journals, authors, bibgroups)
- Create benchmark evaluation set with human review for measuring quality
- Extend guardrail constraints for new field types discovered during audit

## User Stories

### Phase 1: Data Model Audit

#### US-DMA-001: Complete field inventory from ADS Solr schema
**Description:** As a data engineer, I need a complete inventory of all searchable ADS fields so I know what our training data must cover.

**Acceptance Criteria:**
- [ ] Extract all fields from ADS Solr schema (montysolr repo)
- [ ] Create `data/model/ads_field_inventory.json` with field name, type, description, example values
- [ ] Cross-reference with Deep Search results on field documentation
- [ ] Categorize fields: text (abs, title), enum (doctype, property), identifier (bibcode, doi), metric (citation_count), date (pubdate, year), special (object, aff)
- [ ] Document which fields are indexed vs stored-only
- [ ] Run: mise run lint - passes

#### US-DMA-002: Audit current training data coverage
**Description:** As a data engineer, I need to measure current training data coverage against the complete field inventory.

**Acceptance Criteria:**
- [ ] Create `scripts/audit_training_coverage.py` that analyzes gold_examples.json
- [ ] For each field in inventory: count examples using that field, list unique values
- [ ] For each operator: count examples, measure input phrase variety
- [ ] For each enum field: count how many valid values are represented vs total
- [ ] Output coverage report: `data/datasets/evaluations/coverage_audit.json`
- [ ] Identify gaps: fields with 0 examples, operators with <10 examples, enum values never used
- [ ] Run: mise run lint - passes

#### US-DMA-003: Document missing bibgroups and collections
**Description:** As a data engineer, I need to identify bibgroups and collections missing from our constraints and training data.

**Acceptance Criteria:**
- [ ] Query ADS API or Solr facets to get complete list of active bibgroups
- [ ] Compare against field_constraints.py BIBGROUPS
- [ ] Add missing bibgroups to FIELD_ENUMS (e.g., any new telescopes)
- [ ] Add `earthscience` to DATABASES (confirmed in ADS but missing from constraints)
- [ ] Document bibgroup synonyms (Hubble->HST, Webb->JWST, Sloan->SDSS)
- [ ] Create mapping file: `data/model/bibgroup_synonyms.json`
- [ ] Run: mise run verify - all tests pass

### Phase 2: Training Data Generation

#### US-TDG-001: Generate collection/database examples
**Description:** As a data curator, I need training examples for database/collection filtering since current data has 0 examples.

**Acceptance Criteria:**
- [ ] Generate 50+ examples for `database:astronomy`, `database:physics`, `database:general`
- [ ] Generate 50+ examples for `collection:earthscience` (newly added)
- [ ] Include natural language variations: "astronomy papers", "physics literature", "earth science studies"
- [ ] Mix with other fields: "astronomy papers about dark matter from 2020"
- [ ] Run through LLM curator for quality validation
- [ ] Add to `gold_examples.json` with category "collection"
- [ ] Run: mise run curate - validates successfully

#### US-TDG-002: Generate bibgroup examples for all telescopes
**Description:** As a data curator, I need training examples for underrepresented bibgroups.

**Acceptance Criteria:**
- [ ] Current coverage: 126 examples across ~10 bibgroups, need broader coverage
- [ ] Generate 10+ examples for each of 53 bibgroups in FIELD_ENUMS
- [ ] Include synonym variations: "Hubble observations" -> `bibgroup:HST`
- [ ] Mix with operators: "papers citing JWST observations"
- [ ] Mix with properties: "refereed ALMA papers"
- [ ] Run through LLM curator for quality validation
- [ ] Add to `gold_examples.json` with category "bibgroup"
- [ ] Run: mise run curate - validates successfully

#### US-TDG-003: Generate property examples for underrepresented values
**Description:** As a data curator, I need training examples for property values beyond refereed/openaccess/data.

**Acceptance Criteria:**
- [ ] Current coverage: 5 property values used, 21 total valid values
- [ ] Generate 10+ examples for each underrepresented property: notrefereed, eprint, software, catalog, inspire, toc, presentation, associated, etc.
- [ ] Include natural language variations: "preprints" -> `property:eprint`, "datasets" -> `property:data`
- [ ] Generate negation patterns: "non-refereed papers" -> `property:notrefereed`
- [ ] Run through LLM curator for quality validation
- [ ] Add to `gold_examples.json` with category "properties"
- [ ] Run: mise run curate - validates successfully

#### US-TDG-004: Generate doctype examples for underrepresented values
**Description:** As a data curator, I need training examples for doctype values beyond article/software/catalog.

**Acceptance Criteria:**
- [ ] Current coverage: 9 doctype values used, 22 total valid values
- [ ] Generate 10+ examples for each underrepresented doctype: abstract, bookreview, circular, editorial, erratum, inbook, mastersthesis, misc, newsletter, obituary, pressrelease, talk
- [ ] Include natural language variations: "conference talks" -> `doctype:talk`, "press releases" -> `doctype:pressrelease`
- [ ] Run through LLM curator for quality validation
- [ ] Add to `gold_examples.json` with category "doctype"
- [ ] Run: mise run curate - validates successfully

### Phase 3: Operator Interpretation Flexibility

#### US-OIF-001: Expand operator trigger patterns
**Description:** As a NER developer, I need more flexible operator detection that handles natural language variations.

**Acceptance Criteria:**
- [ ] Current issue: "papers citing" triggers citations() but "papers that cite" does not
- [ ] Add trigger patterns: "papers that cite", "work citing", "research citing", "studies citing"
- [ ] Add trigger patterns for references(): "papers referenced by", "bibliography of", "cited in"
- [ ] Add trigger patterns for trending(): "popular papers", "hot topics", "what's trending"
- [ ] Add trigger patterns for similar(): "related to", "papers like", "work similar to"
- [ ] Add trigger patterns for useful(): "helpful papers", "foundational work"
- [ ] Add trigger patterns for reviews(): "review articles", "survey papers", "overviews of"
- [ ] Update `guardrail/operator_gating.py` with expanded patterns
- [ ] Add 50+ regression tests for new patterns
- [ ] Run: mise run test - all tests pass

#### US-OIF-002: Generate training data for operator variations
**Description:** As a data curator, I need training examples with diverse operator trigger phrases.

**Acceptance Criteria:**
- [ ] For each operator, generate 20+ examples with varied trigger phrases
- [ ] citations(): "papers citing X", "work that cites X", "articles citing X", "research citing X", "papers that reference X"
- [ ] references(): "what does X cite", "bibliography of X", "sources cited by X", "references in X"
- [ ] trending(): "hot papers on X", "popular research on X", "what's trending in X"
- [ ] similar(): "papers like X", "related to X", "work similar to X", "studies resembling X"
- [ ] useful(): "helpful papers on X", "foundational work on X", "essential reading on X"
- [ ] reviews(): "review articles on X", "survey of X", "comprehensive reviews on X"
- [ ] Run through LLM curator for quality validation
- [ ] Add to `gold_examples.json` with category "operator"
- [ ] Run: mise run curate - validates successfully

### Phase 4: Indexed Content Integration

#### US-ICI-001: Generate author-grounded examples
**Description:** As a data curator, I need training examples with real author names from the ADS index.

**Acceptance Criteria:**
- [ ] Query ADS for top 1000 most-cited authors in astronomy
- [ ] Create `data/model/indexed_authors.json` with name, orcid, affiliation, h-index
- [ ] Generate 200+ examples using real author names with proper format
- [ ] Include first-author queries: "first-author papers by Hawking"
- [ ] Include multi-author queries: "papers by Einstein and Rosen"
- [ ] Include affiliation queries: "Harvard astronomers working on exoplanets"
- [ ] Run through LLM curator for quality validation
- [ ] Add to `gold_examples.json` with category "author"
- [ ] Run: mise run curate - validates successfully

#### US-ICI-002: Generate journal-grounded examples
**Description:** As a data curator, I need training examples with real journal bibstems from the ADS index.

**Acceptance Criteria:**
- [ ] Query ADS for all active journal bibstems
- [ ] Create `data/model/indexed_journals.json` with bibstem, full name, publisher
- [ ] Generate 100+ examples using real bibstems: "papers in ApJ", "Nature articles", "MNRAS research"
- [ ] Include journal variations: "Astrophysical Journal papers" -> `bibstem:"ApJ"`
- [ ] Include combined queries: "ApJ papers about exoplanets from 2023"
- [ ] Run through LLM curator for quality validation
- [ ] Add to `gold_examples.json` with category "journal"
- [ ] Run: mise run curate - validates successfully

#### US-ICI-003: Generate object-grounded examples
**Description:** As a data curator, I need training examples with real astronomical object names.

**Acceptance Criteria:**
- [ ] Create list of common astronomical objects: galaxies (M31, M87, NGC 4258), nebulae (Orion, Crab), stars (Proxima Centauri, Betelgeuse), clusters (Pleiades, Coma)
- [ ] Create `data/model/astronomical_objects.json` with object name, type, coordinates
- [ ] Generate 100+ examples using object: field: "papers about M31", "Crab Nebula observations"
- [ ] Include cone search patterns: "papers within 1 degree of M87"
- [ ] Include combined queries: "JWST observations of Orion Nebula"
- [ ] Run through LLM curator for quality validation
- [ ] Add to `gold_examples.json` with category "object"
- [ ] Run: mise run curate - validates successfully

### Phase 5: Benchmark and Evaluation

#### US-BE-001: Create comprehensive benchmark set
**Description:** As a QA engineer, I need a benchmark evaluation set covering all data model elements.

**Acceptance Criteria:**
- [ ] Create `data/datasets/benchmark/benchmark_queries.json` with 500+ test cases
- [ ] Include 10+ examples for each: field type, operator, property value, doctype, bibgroup
- [ ] Include edge cases: ambiguous operator words, complex boolean logic, nested operators
- [ ] Include regression tests: all known malformed patterns that must be rejected
- [ ] Flag examples requiring human review (ambiguous intent)
- [ ] Run: mise run lint - passes

#### US-BE-002: Human review of benchmark set
**Description:** As a QA engineer, I need human verification of benchmark ground truth.

**Acceptance Criteria:**
- [ ] Export benchmark to review format (CSV or spreadsheet)
- [ ] Human reviewer validates: NL->query mapping is correct, query returns expected results
- [ ] Flag and fix any incorrect ground truth
- [ ] Document review process in `docs/BENCHMARK_REVIEW.md`
- [ ] Import reviewed benchmark back to JSON format
- [ ] Create `data/datasets/benchmark/benchmark_reviewed.json` with reviewer annotations

#### US-BE-003: Automated benchmark evaluation
**Description:** As a QA engineer, I need automated evaluation against the benchmark set.

**Acceptance Criteria:**
- [ ] Create `scripts/evaluate_benchmark.py` that runs model against benchmark
- [ ] Metrics: exact match rate, field assignment accuracy, operator accuracy, syntax validity
- [ ] Breakdown by category: author queries, operator queries, property queries, etc.
- [ ] Compare before/after data model coverage enhancement
- [ ] Output evaluation report: `data/datasets/evaluations/benchmark_results.json`
- [ ] Add to CI: fail if any category drops below threshold
- [ ] Run: mise run lint - passes

### Phase 6: Retrain and Validate

#### US-RV-001: Retrain model with enhanced data
**Description:** As a system owner, I need the model retrained on data model-enhanced training set.

**Acceptance Criteria:**
- [ ] Merge all new generated examples into gold_examples.json (target: 6000+ examples)
- [ ] Run curation pipeline: mise run curate
- [ ] Train new model: scix-finetune train
- [ ] Deploy to Modal endpoint
- [ ] Run benchmark evaluation
- [ ] Document training results in progress.txt
- [ ] Verify in browser: diverse query types work correctly

#### US-RV-002: Regression testing post-enhancement
**Description:** As a QA engineer, I need to verify no regressions after data model enhancement.

**Acceptance Criteria:**
- [ ] Run all existing tests: mise run test
- [ ] Run integration tests: mise run test:integration
- [ ] Run ADS live validation: mise run eval:ads
- [ ] Compare benchmark results to pre-enhancement baseline
- [ ] No category should regress >5% from baseline
- [ ] All malformed pattern tests must pass (0 citationsabs: etc.)
- [ ] Document any regressions and fixes in progress.txt

## Functional Requirements

- FR-1: The system must support all 4 database/collection values: astronomy, physics, general, earthscience
- FR-2: The system must support all 53 bibgroup values with synonym mapping
- FR-3: The system must support all 22 doctype values with natural language variations
- FR-4: The system must support all 21 property values with natural language variations
- FR-5: The system must trigger operators for expanded phrase patterns (not just exact matches)
- FR-6: The system must use indexed author names in proper ADS format (Last, First)
- FR-7: The system must use indexed journal bibstems with full name mapping
- FR-8: The system must use indexed object names with SIMBAD/NED resolution
- FR-9: The system must pass benchmark evaluation with >85% overall accuracy
- FR-10: The system must maintain 0% malformed operator patterns in output

## Non-Goals (Out of Scope)

- Adding new operators not currently supported by ADS (e.g., custom ranking functions)
- Building a custom author name disambiguation system (use ADS's existing resolution)
- Creating a journal recommendation system
- Implementing real-time query suggestions/autocomplete
- Supporting non-English natural language input
- Building a training data GUI (continue using JSON/script workflow)

## Technical Considerations

### Existing Architecture
- LLM curator agent exists for quality validation
- Guardrail validators exist (syntax, entity, author, operator gating)
- Curation pipeline exists (mise run curate)
- Training pipeline exists (scix-finetune train)
- Evaluation scripts exist (validate_ads_queries.py, llm_judge.py)

### Data Sources for Indexed Content
- ADS API: `/search/query` with facets for top authors, journals, objects
- Solr schema: montysolr repo for field definitions
- Bibgroups: ADS help documentation + API facets
- Objects: SIMBAD/NED for astronomical object names

### Scaling Considerations
- Current: 4002 examples
- Target: 6000+ examples
- Training time on Modal H100: ~1 hour for 4k examples
- Curation with LLM judge: ~0.5s per example

### Dependencies
- ADS API key for validation queries
- Claude API key for LLM curator/judge
- Modal account for training
- Together.ai or local vLLM for inference

## Success Metrics

- **Coverage Metric**: >90% of data model elements represented in training data
  - All 4 databases
  - All 53 bibgroups
  - All 22 doctypes
  - All 21 properties
  - All 7 operators with >30 examples each

- **Benchmark Accuracy**: >85% overall on reviewed benchmark set
  - Author queries: >90%
  - Topic queries: >90%
  - Operator queries: >85%
  - Property/filter queries: >85%
  - Complex compound queries: >75%

- **Regression Prevention**: 0% malformed patterns
  - No `citationsabs:`, `referencesabs:`, etc.
  - No invalid enum values
  - No unbalanced parentheses

## Open Questions

1. Should we generate synthetic author names or only use real indexed authors?
   - Recommendation: Use real indexed authors for ground truth quality

2. How many examples per bibgroup is sufficient for reliable recognition?
   - Recommendation: Minimum 10, target 20 for high-use telescopes (HST, JWST, Chandra)

3. Should operator trigger patterns be exhaustively enumerated or use regex patterns?
   - Recommendation: Hybrid - explicit patterns for common cases, regex for variations

4. What is the budget for human benchmark review?
   - Need: ~500 examples * ~30 seconds = ~4 hours of human review time

5. Should we include negative examples (ambiguous operator words used as topics)?
   - Recommendation: Yes, critical for operator gating accuracy
