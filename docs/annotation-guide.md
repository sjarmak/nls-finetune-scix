# Enrichment Annotation Guide

Guidelines for annotating scientific text spans in ADS abstracts for NER model evaluation.

## Overview

Annotators mark spans of text that refer to **scientific topics**, **institutions**, **authors**, or **date ranges**. Each span gets a type label and, when possible, a canonical ID from a controlled vocabulary.

This guide covers:
1. What counts as each span type
2. Span boundary rules
3. Ambiguous cases and edge cases
4. Output format

## Span Types

### topic

A scientific concept, keyword, or subject area that could map to a controlled vocabulary (UAT, SWEET, GCMD, or planetary nomenclature).

**Annotate as topic:**
- Named scientific concepts: `dark matter`, `solar wind`, `aerosol optical depth`
- Astronomical objects by class: `galaxy clusters`, `neutron stars`, `exoplanets`
- Earth science phenomena: `precipitation`, `El Nino`, `sea surface temperature`
- Methods or techniques when domain-specific: `spectroscopy`, `radiative transfer`
- Named planetary features: `Gale Crater`, `Olympus Mons`, `Valles Marineris`

**Do NOT annotate as topic:**
- Generic scientific terms: `data`, `analysis`, `results`, `method`, `model`
- Verbs or actions: `observed`, `measured`, `detected`
- Units or numbers: `km/s`, `10^6`, `3.5 AU`
- Proper nouns that are instrument/mission names (these are not in the topic catalogs)

**Examples:**

| Text | Span | Start | End | Type |
|------|------|-------|-----|------|
| "We study **dark matter** halos in galaxy clusters" | dark matter | 9 | 20 | topic |
| "**Precipitation** patterns over the Amazon basin" | Precipitation | 0 | 13 | topic |
| "Observations of **Gale Crater** on Mars" | Gale Crater | 16 | 27 | topic |

### institution

A research institution, university, observatory, or organization that could map to the ROR registry.

**Annotate as institution:**
- Universities: `Harvard University`, `MIT`, `University of Tokyo`
- Research institutes: `Max Planck Institute`, `Jet Propulsion Laboratory`
- Observatories: `European Southern Observatory`, `Keck Observatory`
- Space agencies: `NASA`, `ESA`, `JAXA`
- Well-known abbreviations: `STScI`, `CERN`, `NOAA`

**Do NOT annotate as institution:**
- Instrument or telescope names (e.g., `Hubble`, `JWST`, `Chandra` are instruments, not institutions)
- Collaboration names (e.g., `LIGO Scientific Collaboration`)
- Funding agency grant numbers

**Examples:**

| Text | Span | Start | End | Type |
|------|------|-------|-----|------|
| "Researchers at **NASA Goddard**" | NASA Goddard | 15 | 26 | institution |
| "A study by the **European Southern Observatory**" | European Southern Observatory | 15 | 43 | institution |

### author

A person's name mentioned in the text body (not the metadata author list).

**Annotate as author:**
- Full names: `Albert Einstein`, `Vera Rubin`
- Surnames with context: `Einstein's theory`, `the Hubble constant` (when referring to the person, not the telescope)

**Do NOT annotate as author:**
- Names in the metadata author field (only annotate in-text mentions)
- Names used as adjectives for methods/laws without person intent: `Bayesian`, `Gaussian`
- Names that are part of proper nouns (e.g., `Hubble Space Telescope` — annotate the telescope as an instrument, not an author)

**Examples:**

| Text | Span | Start | End | Type |
|------|------|-------|-----|------|
| "Following **Salpeter** (1955), we assume..." | Salpeter | 10 | 18 | author |

### date_range

A temporal expression indicating a specific time period relevant to the data or observations.

**Annotate as date_range:**
- Year ranges: `2010-2020`, `1990 to 2005`
- Specific periods: `the 2003 season`, `between January and March 2019`
- Relative periods when anchored: `the last decade` (only if context makes it specific)

**Do NOT annotate as date_range:**
- Publication years in citations: `(Smith et al. 2020)`
- Years that are part of instrument/mission names: `Gaia DR3`
- Generic temporal words: `recently`, `previously`, `long-term`

## Span Boundary Rules

### Rule 1: Maximal meaningful span

Include all words that form a single coherent concept. Prefer longer spans when they form a recognized term.

```
GOOD: "sea surface temperature"  (single concept)
BAD:  "sea surface" + "temperature"  (split breaks the concept)
```

### Rule 2: No leading/trailing whitespace

Spans must start and end on non-whitespace characters.

```
GOOD: start=5, end=16 → "dark matter"
BAD:  start=4, end=17 → " dark matter "
```

### Rule 3: No punctuation at boundaries

Exclude leading/trailing punctuation unless it is part of the name.

```
GOOD: "dark matter" from "...dark matter, which..."
BAD:  "dark matter," (trailing comma included)
```

### Rule 4: Byte-exact offsets

`text[start:end]` must exactly reproduce the annotated `surface` string. This is the primary validation check.

```python
assert text[span["start"]:span["end"]] == span["surface"]
```

### Rule 5: Non-overlapping spans

Spans must not overlap. If two possible annotations share characters, choose the more specific or longer one.

```
Text: "Mars surface temperature"
GOOD: one span "Mars surface temperature" (topic)
ALSO OK: "Mars" (topic) + "surface temperature" (topic) if both are separate concepts
BAD: "Mars surface" (topic) overlapping with "surface temperature" (topic)
```

## Ambiguous Cases

### Case 1: Instruments vs. institutions

`Hubble` can refer to:
- The person Edwin Hubble → `author`
- The Hubble Space Telescope → do NOT annotate (instrument, not in catalogs)
- "Hubble constant" → do NOT annotate (named physical constant)

Use context to decide. When ambiguous, do not annotate.

### Case 2: Acronyms

Annotate acronyms only if they unambiguously refer to an annotatable entity:
- `NASA` → `institution` (clear)
- `UV` → do NOT annotate (too generic, not a topic in controlled vocabularies)
- `CMB` → `topic` if context is "cosmic microwave background" (maps to UAT)

### Case 3: Nested entities

When a longer span contains a shorter one, annotate only the most specific:
- "Harvard-Smithsonian Center for Astrophysics" → one `institution` span (not `Harvard` + `Smithsonian` separately)
- "Mars polar ice cap" → could be one `topic` or split into `Mars` (topic) + `polar ice cap` (topic); prefer the interpretation that best matches controlled vocabulary entries

### Case 4: Lists of topics

Annotate each item in a list separately:
- "dark matter, dark energy, and cosmic expansion" → three `topic` spans

### Case 5: Possessives and modifiers

Include the core entity but exclude possessives:
- "Mars's atmosphere" → annotate `Mars` (topic), not `Mars's`
- "Jovian magnetosphere" → annotate `Jovian magnetosphere` as one `topic` if it appears in the vocabulary; otherwise annotate individually

## Output Format

Each annotated abstract is a JSON record:

```json
{
  "bibcode": "2024ApJ...123..456A",
  "title": "Dark Matter Distribution in Galaxy Clusters",
  "abstract": "We study the dark matter distribution in nearby galaxy clusters using weak gravitational lensing data from the Subaru telescope.",
  "database": ["astronomy"],
  "keywords": ["dark matter", "galaxy clusters", "gravitational lensing"],
  "year": 2024,
  "doctype": "article",
  "citation_count": 42,
  "domain_category": "astronomy",
  "spans": [
    {
      "surface": "dark matter",
      "start": 13,
      "end": 24,
      "type": "topic",
      "canonical_id": "uat:dark_matter"
    },
    {
      "surface": "galaxy clusters",
      "start": 44,
      "end": 59,
      "type": "topic",
      "canonical_id": "uat:galaxy_clusters"
    },
    {
      "surface": "gravitational lensing",
      "start": 72,
      "end": 93,
      "type": "topic",
      "canonical_id": "uat:gravitational_lensing"
    }
  ]
}
```

### Field descriptions

| Field | Type | Description |
|-------|------|-------------|
| `surface` | string | Exact text as it appears in the abstract |
| `start` | int | Character offset of span start (0-indexed) |
| `end` | int | Character offset of span end (exclusive) |
| `type` | string | One of: `topic`, `institution`, `author`, `date_range` |
| `canonical_id` | string | ID from controlled vocabulary (e.g., `uat:dark_matter`, `ror:abc123`), or empty string if no match |

## Annotation Workflow

1. **Read the full abstract** before annotating
2. **Focus on titles + first 2-3 sentences** for the pilot (50-abstract set)
3. **Mark all recognizable spans** following the type definitions above
4. **Look up canonical IDs** in the catalogs when possible:
   - UAT: `data/datasets/agent_runs/*/normalized/topic_catalog_uat.jsonl`
   - ROR: `data/datasets/agent_runs/*/normalized/entity_catalog_ror.jsonl`
   - SWEET: `data/datasets/agent_runs/*/normalized/topic_catalog_sweet.jsonl`
   - GCMD: `data/datasets/agent_runs/*/normalized/topic_catalog_gcmd.jsonl`
   - Planetary: `data/datasets/agent_runs/*/normalized/entity_catalog_planetary.jsonl`
5. **Validate offsets**: Run `text[start:end] == surface` for every span
6. **Save as JSONL** to `data/evaluation/ads_sample_annotated.jsonl`

## Quality Checklist

Before submitting annotations:

- [ ] Every span has `type` set to one of the four allowed types
- [ ] Every span has byte-exact offsets (`text[start:end] == surface`)
- [ ] No overlapping spans
- [ ] No leading/trailing whitespace in surface strings
- [ ] No leading/trailing punctuation in surface strings (unless part of the name)
- [ ] Topics are specific enough to match a controlled vocabulary entry
- [ ] Institutions are named organizations (not instruments or collaborations)
- [ ] Authors are in-text person mentions (not metadata or citation references)
- [ ] Date ranges are specific temporal expressions (not citation years)
