# Training Data Quality Report

Generated: 2026-01-21 17:16:24

## Summary

| Metric | Value |
|--------|-------|
| Total pairs | 3,025 |
| Training set | 2,722 |
| Validation set | 303 |
| Lint pass rate | 100.0% (3,025/3,025) |
| Lint failures | 0 |

## Category Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| first_author | 853 | 28.2% |
| unfielded | 557 | 18.4% |
| author | 448 | 14.8% |
| content | 286 | 9.5% |
| publication | 219 | 7.2% |
| operator | 160 | 5.3% |
| filters | 153 | 5.1% |
| compound | 107 | 3.5% |
| conversational | 53 | 1.8% |
| affiliation | 47 | 1.6% |
| metrics | 43 | 1.4% |
| astronomy | 41 | 1.4% |
| properties | 18 | 0.6% |
| syntax | 10 | 0.3% |
| identifiers | 7 | 0.2% |
| second_order | 4 | 0.1% |
| positional | 4 | 0.1% |
| citations | 3 | 0.1% |
| identifier | 3 | 0.1% |
| data | 3 | 0.1% |
| object | 3 | 0.1% |
| property | 3 | 0.1% |

## Invalid Field Values

### Database

| Invalid Value | Count | Suggestions |
|---------------|-------|-------------|
| `astronomy,` | 1 | astronomy |

### Bibgroup

| Invalid Value | Count | Suggestions |
|---------------|-------|-------------|
| `SETI` | 12 | - |
| `ESO` | 6 | ESO/Telescopes |

## Bare (Unquoted) Field Values

### abs
Total unquoted values: 38

| Value | Count |
|-------|-------|
| `cosmology` | 7 |
| `exoplanet` | 3 |
| `JWST` | 3 |
| `quasar` | 3 |
| `magnetar` | 2 |
| `supernova` | 2 |
| `pulsar` | 2 |
| `galaxy,` | 2 |
| `supernova,` | 1 |
| `photometry` | 1 |
| ... | (12 more) |

### aff
Total unquoted values: 6

| Value | Count |
|-------|-------|
| `Harvard` | 1 |
| `LIGO` | 1 |
| `Instituto` | 1 |
| `MIT` | 1 |
| `Caltech` | 1 |
| `harvard,1` | 1 |

### author
Total unquoted values: 1

| Value | Count |
|-------|-------|
| `^hamuy` | 1 |

### object
Total unquoted values: 6

| Value | Count |
|-------|-------|
| `M31` | 2 |
| `LMC` | 1 |
| `Betelgeuse` | 1 |
| `M87` | 1 |
| `TRAPPIST-1` | 1 |

### title
Total unquoted values: 10

| Value | Count |
|-------|-------|
| `A` | 7 |
| `ARP299` | 2 |
| `2mass` | 1 |

## Examples

### Good Examples (Valid Queries)

**1. first_author**
- NL: "papers by middelberg from 2005"
- Query: `author:"^middelberg" year:2005`

**2. first_author**
- NL: "papers by bramante-elahi first author"
- Query: `author:"^bramante-elahi"`

**3. first_author**
- NL: "papers by Kumar and Goodman"
- Query: `author:"^kumar" author:"goodman"`

**4. first_author**
- NL: "papers by Sune Toft first author"
- Query: `author:("^toft, sune")`

**5. operator**
- NL: "useful references for CMB analysis"
- Query: `useful(abs:"cosmic microwave background")`

### Bad Examples (Invalid Queries)

✅ No invalid queries found!

## Recommendations

2. **Fix 19 invalid field values** - Invalid doctype/database/property/bibgroup values
3. **Quote 61 bare field values** - Fields like author, bibstem need quoting
4. **Balance category distribution** - Underrepresented: properties, syntax, identifiers, second_order, citations

## Before/After Metrics

| Metric | Before (US-004) | After (US-005) | Change |
|--------|-----------------|----------------|--------|
| Total pairs | 3,025 | 3,025 | - |
| Lint pass rate | 100% | 100.0% | - |
| Bare bibstem values | 536 | 0 | ✅ Fixed |
| Total bare fields | 536+ | 61 | ✅ Improved |
