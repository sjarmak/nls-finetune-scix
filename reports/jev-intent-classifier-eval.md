# Jev typed classifiers as the IntentSpec reasoning stage: evaluation

Run date: 2026-09-24. Plan and pre-registered criteria:
`docs/research/jev-intent-classifier-experiment.md`. Model pinned to
`jev-1.13.0` (recorded on every row). Comparison LLM: `claude-haiku-4-5`
(served as `claude-haiku-4-5-20251001`).

## Decision

**Go on the intent stage.** On the 152-item
held-out paraphrase set, reviewed and approved by Stephanie before any model
saw it, Jev scores 0.955 operator macro-F1 against 0.146 for the shipped
regex, an 81-point margin on a set built from phrasings the regex has never
seen. It makes zero operator false positives on the 40-item operator-negative
stratum, the conflation failure the hybrid pipeline exists to prevent. At 90%
coverage its operator accuracy is 100%. The gated deployable shape (arm E)
meets the latency criterion everywhere and the cost criterion on the
benchmark, missing it by 19% to 26% on the sets where the gate opens for
nearly every query. Criteria 1, 2 and 3 are met.

The Haiku comparison (arm C) closes the "exceeds arm C" half of criterion 1.
Asked the same questions with the same schema, Claude Haiku 4.5 scores 0.933
operator macro-F1 on the held-out set, 2.2 points behind Jev, and it makes
three operator false positives on the operator-negative stratum (7.5%) where
Jev makes none. All three are the word "useful" or a synonym read as the
`useful` operator ("useful yield in solar cell efficiency modelling",
"foundational models for spectral classification", "must-have calibrations
for CCD photometry"), each at Haiku's full confidence. That is the
conflation failure criterion 2 exists to catch: Haiku as the intent stage
would fail the no-go criterion outright. Haiku also has no usable abstention
signal on this task (it answered `unknown` for the operator on 5 of 152
items and never otherwise hedged), so a threshold cannot buy back precision,
whereas Jev's 0.44-confidence miss sits below any sensible cut. The win is
attributable to the typed classifier, not only to asking narrow questions,
though the narrow questions do most of the work for both: the two are
within 2 points on every set.

End to end, the assembled queries score the same with or without Jev: the
result-set overlap is set by the regex topic spans, which no arm touches, and
the differences between backends are smaller than the run-to-run noise of
ADS's second-order operators. Jev wins more paired items than it loses on
both sets, but the mean moves by under a point. Criterion 4 is therefore
"no regression shown, rise not shown".

The recommendation is to adopt arm E (regex first, Jev when the regex finds
no operator or is unsure) as the operator, enum and gate decider in the
hybrid route, keep the regex as the extractor for names, years and topics,
and treat Jev's confidence as the routing signal. The doctype prior from the
regex should not be passed to Jev (arm D shows it biases Jev toward the
regex's `doctype:article` error).

## What was run

| arm | intent source |
|-----|---------------|
| A | regex `extract_intent` as shipped |
| B | Jev decides operator, enums, gates; regex names, years, topics |
| C | Claude Haiku 4.5, same question set, structured output, `unknown` per field |
| D | as B, with the regex IntentSpec sent to Jev as extra `state` |
| E | regex first; Jev only when regex finds no operator or confidence < 0.5 |

Datasets scored: the held-out paraphrase set (152 items, its own section below), benchmark (253 items with a gold query; 15 `topn` items
have no IntentSpec operator and are excluded from operator metrics) and the
synthetic val set (478 items, labels derived mechanically from the gold ADS
query by `scripts/derive_intent_labels.py`, 5 `topn` excluded). Every item
was scored once per arm; a 100-item benchmark subset was scored three times
per model arm with the cache bypassed for the stability numbers.

Two rounds were run. Round 1 used the first draft of the operator option
text. Jev answered `none` for nearly every `useful`, `reviews` and `trending`
request ("must-read papers on black holes", "survey papers on galaxy
evolution", "hot topics in exoplanet research") and answered `citations` for
citation-count filters ("papers with more than 100 citations"). Both are
option-description problems, not model failures: the descriptions said what
the ADS operator computes, not what a person asking for it sounds like. Round
2 rewrote the seven operator descriptions and the instruction (final text in
`jev_option_text.py`) and re-ran every arm on both datasets. The rewrite was
informed by the benchmark, so the benchmark round-2 numbers are tuned; the
val-set numbers are the check that the rewrite generalised, and they did
(arm B operator macro-F1 on val moved from 0.652 to 0.830 without the val
set being looked at). All numbers below are round 2 unless labelled.

## Operator classification

Macro-F1 over the 7 classes present, accuracy, and false-positive rate on
gold-`none` items (an operator predicted where the gold has none).

| dataset | arm | macro-F1 | accuracy | FP on gold none | acc at 90% coverage | ECE |
|---|---|---|---|---|---|---|
| benchmark | A regex | 0.938 | 0.954 | 0.7% (1/147) | 0.988 (no signal, 100% cov) | 0.038 |
| benchmark | B jev | 0.980 | 0.983 | 1.4% (2/147) | 1.000 | 0.024 |
| benchmark | C haiku | 0.984 | 0.987 | 0.0% (0/147) | 0.987 (unknown rate 0.4%) | 0.013 |
| benchmark | D jev+state | **0.995** | **0.996** | 0.7% (1/147) | 1.000 | 0.023 |
| benchmark | E gated | 0.991 | 0.992 | 1.4% (2/147) | 0.996 | 0.018 |
| val | A regex | 0.814 | 0.970 | 1.1% (5/436) | 0.824 (no signal) | 0.126 |
| val | B jev | 0.830 | 0.964 | 3.2% (14/436) | 0.991 | 0.017 |
| val | C haiku | 0.855 | 0.968 | 2.8% (12/436) | 0.968 (unknown rate 0.2%) | 0.036 |
| val | D jev+state | **0.899** | **0.981** | 1.6% (7/436) | 1.000 | 0.013 |
| val | E gated | 0.795 | 0.960 | 3.9% (17/436) | 0.979 | 0.022 |

Round 1 for reference: B 0.626 / 0.652 macro-F1, D 0.963 / 0.850, E 0.946 /
0.847, C 0.816 / 0.818 on benchmark / val.

Per-class on val, where the operator classes are small (11 citations, 4
references, 3 similar, 8 trending, 4 useful, 7 reviews): D scores 1.0 on
references, similar and trending; the weak classes for every arm are
`useful` (0.75 to 0.86) and `reviews` (0.52 to 0.63). The regex fails on
"who cites JWST papers", "trending exoplanets", "reviews by first author
Hawking"; Jev fails on "helpful papers on exoplanets" and on "bibliography of
cosmology papers from 2023".

The regex's "confidence" is the constant 0.95 whenever a pattern fires and
absent otherwise, so its selective and ECE columns describe a constant, not a
signal. The regex ECE of 0.126 on val is the cost of being 95% sure on
patterns that fire wrongly 3% of the time with no way to tell which.

Arm E on val is the odd row: it calls Jev for 93% of val items (the regex
rarely finds an operator on synthetic NL) and inherits Jev's false positives
plus the regex's own, so on that set it is the worst of both. On the benchmark
it calls Jev for 68% of items and is within a point of D.

### The val false positives, itemised

Arm B's 14 gold-`none` items with a predicted operator on val:

- 6 items where the synthetic NL says "reviews" or "review articles" but the
  gold is a plain `abs:` search with `doctype:article property:refereed`
  ("black holes reviews", "supernova remnants review", "any good active
  galactic nuclei reviews?"). The NL generator put the word in; the gold does
  not carry it. Jev's answer matches the text as written.
- 3 items where "textbook reviews" or "reviews of astronomy books" means
  `doctype:bookreview`. Jev answered the `reviews` operator. A genuine
  confusion, and a fixable one: "book review" is a document type, not an
  operator, and the doctype option text can say so.
- 2 items where "HST bibliography" or "ALMA bibliography" is ADS jargon for
  `bibgroup:`; Jev answered `references`. Also fixable in the option text.
- 1 "influential radio astronomy research" (gold `citation_count:[50 TO *]`,
  Jev `useful`), 1 "refereed Icarus papers" (Jev `references` at 0.35), 1
  "research purpose operating systems survey" (a paper title; Jev `reviews`).

Arm D, which sees the regex IntentSpec in its state, avoids 7 of these 14.
Arm C makes 12 of the same kind of errors, so the pattern is the question,
not the model.

## Enum fields

Micro set-F1 (property restricted to refereed / openaccess / eprint, the only
values any arm produces; gold uses others and they are dropped from both
sides). Exact-match rate in parentheses.

| dataset | arm | property | doctype | bibgroup | collection |
|---|---|---|---|---|---|
| benchmark | A | 0.714 (0.972) | 0.080 (0.387) | 0.794 (0.945) | 0.778 (0.968) |
| benchmark | B | 0.741 (0.976) | 0.485 (0.874) | **0.915** (0.980) | 0.441 (0.870) |
| benchmark | C | 0.556 (0.949) | **0.638** (0.933) | 0.931 (0.988) | 0.321 (0.783) |
| benchmark | D | 0.741 (0.976) | 0.211 (0.506) | 0.806 (0.949) | 0.778 (0.968) |
| benchmark | E | 0.741 (0.976) | 0.400 (0.834) | 0.848 (0.960) | 0.473 (0.885) |
| val | A | 0.647 (0.950) | 0.031 (0.368) | 0.421 (0.908) | 0.556 (0.950) |
| val | B | **0.914** (0.985) | 0.391 (0.891) | **0.811** (0.971) | 0.360 (0.881) |
| val | C | 0.771 (0.960) | 0.333 (0.906) | 0.806 (0.973) | 0.224 (0.785) |
| val | D | 0.902 (0.983) | 0.120 (0.577) | 0.638 (0.929) | 0.556 (0.950) |
| val | E | 0.914 (0.985) | 0.337 (0.881) | 0.763 (0.962) | 0.364 (0.883) |

Three things stand out.

- **The regex emits `doctype:article` for the word "papers".** 140 of 238
  benchmark items disagree with gold on doctype, and almost all are
  `gold=[] pred=[article]` ("papers about dark matter halos"). The benchmark
  gold convention is that "papers" carries no doctype. This is a shipped
  regex behaviour, not a scoring artefact, and it is why the regex doctype F1
  is near zero on both sets. Jev was told the convention in its option text
  and follows it (exact match 0.87 to 0.89).
- **Arm D inherits the regex's doctype error.** With the regex IntentSpec in
  its state, Jev keeps `article` far more often (exact match drops from 0.87
  to 0.51). Structural context helps on the operator and hurts on doctype:
  Jev defers to a wrong prior. If D is the deployed shape the regex doctype
  should be left out of the state, or the regex fixed.
- **Collection is where Jev over-answers.** Gold almost never sets
  `database:`; Jev picks `astronomy` for astronomy queries at moderate
  confidence. The threshold study below applies here too: at confidence 0.9
  most of those go away. The regex only sets collection on explicit words and
  is right to.

## Calibration and abstention

Selective accuracy against coverage on the operator answer, arm B and D, val
set (the uncontaminated one):

| threshold | B coverage | B accuracy | D coverage | D accuracy |
|---|---|---|---|---|
| 0.50 | 0.98 | 0.974 | 0.99 | 0.985 |
| 0.70 | 0.96 | 0.982 | 0.95 | 0.993 |
| 0.80 | 0.94 | 0.989 | 0.94 | 0.996 |
| 0.90 | 0.92 | 0.988 | 0.93 | 1.000 |
| 0.95 | 0.89 | 0.993 | 0.92 | 1.000 |

ECE on the operator confidence is 0.013 to 0.024 for the Jev arms on both
sets, against 0.126 for the regex constant on val. The "calibrated" claim
holds on this task: the reliability bins are within a few points of the
diagonal everywhere there are enough items to say. Haiku's `unknown` option
was used on 0.2% to 0.4% of items, which confirms the softwaredoug
observation that a general LLM with an abstain option still guesses; its
errors carry confidence 1.0 by construction and there is no threshold to
tune.

## Stability

Three uncached repeats on the first 100 benchmark items, fraction of items
whose answer changed:

| arm | operator | property | doctype | bibgroup | collection | search_kind |
|---|---|---|---|---|---|---|
| B | 0.00 | 0.00 | 0.01 | 0.00 | 0.02 | 0.00 |
| C | 0.00 | 0.03 | 0.03 | 0.01 | 0.08 | 0.05 |
| D | 0.00 | 0.00 | 0.07 | 0.00 | 0.00 | 0.01 |
| E | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.01 |

Jev is deterministic on the operator across repeats and nearly so on the
enums. Haiku at temperature default drifts on collection and search_kind.

## Latency and cost

Client-side, this machine to `api.typesafe.ai`, four concurrent requests.

| arm | p50 ms | p95 ms | mean input tokens | $/query | classifier call rate |
|---|---|---|---|---|---|
| A regex | 0 | 0 | 0 | 0 | 0% |
| B jev | 147 to 151 | 205 to 208 | 2,903 | $0.000128 | 100% |
| C haiku | 1,437 to 1,599 | 1,810 to 1,852 | 3,215 in, 79 out | $0.00376 | 100% |
| D jev+state | 149 to 151 | 205 to 210 | 3,080 | $0.000135 | 100% |
| E gated | 135 to 147 | 185 to 205 | 1,962 (bench) / 2,698 (val) | $0.000086 / $0.000119 | 68% / 93% |

The plan estimated 1.2k input tokens per call; the real figure is 2.9k
because the doctype and bibgroup questions carry 22 and 55 option
descriptions. Cost is still under the $0.0001 criterion for arm E on the
benchmark and 19% over it on val, where the gate opens more often. Whole
experiment: about $0.30 of Jev calls and $5.30 of Haiku calls, all cached in
`data/cache/`.

## Held-out paraphrase set

152 items in four strata, drafted by Claude Fable 5.1, mechanically checked
(the 42 operator-positive items are asserted to fire no regex), then reviewed
and approved item by item by Stephanie on 2026-09-24 with no rejections.
No model scored an item before approval. Arm C ran through the logged-in
`claude` CLI (`--llm-transport cli`) rather than the Anthropic API, because
the API account had no credit; see Caveats for what that changes.

| arm | macro-F1 | accuracy | FP on operator-negative (40) | FP on all gold-none (106) | acc at 90% cov | ECE | p95 ms | $/query |
|---|---|---|---|---|---|---|---|---|
| A regex | 0.146 | 0.697 | 0 | 1 | 0.500 (no signal) | 0.450 | 0 | 0 |
| B jev | 0.955 | 0.974 | **0** | 3 | **1.000** (thr 0.73) | 0.061 | 223 | $0.000128 |
| C haiku (CLI) | 0.933 | 0.954 | 3 | 4 | 0.973 (cov 0.97, no threshold) | 0.039 | 7,107 | $0.00659 at list price |
| D jev+state | 0.923 | 0.954 | **0** | 2 | 0.985 | 0.038 | 215 | $0.000135 |
| E gated | **0.965** | **0.980** | **0** | 3 | 0.993 | 0.060 | 206 | $0.000126 (Jev called 98.7%) |

Operator accuracy by stratum (correct / items):

| arm | operator-positive (42) | operator-negative (40) | enum synonyms (40) | ambiguous (30) |
|---|---|---|---|---|
| A regex | 0 / 42 | 40 / 40 | 39 / 40 | 27 / 30 |
| B jev | 41 / 42 | 40 / 40 | 39 / 40 | 28 / 30 |
| C haiku | 41 / 42 | 37 / 40 | 40 / 40 | 27 / 30 |
| D jev+state | 38 / 42 | 40 / 40 | 39 / 40 | 28 / 30 |
| E gated | 42 / 42 | 40 / 40 | 39 / 40 | 28 / 30 |

The regex arm's zero on the positive stratum is by construction; the point
of the set is that Jev reads "papers that build on the Planck 2018 cosmology
results", "who has picked up on the Riess et al. 2019 measurement" and
"cornerstone results in exoplanet transit photometry" without a pattern for
any of them. Its one positive-stratum miss is "kindred studies to exoplanet
atmosphere retrievals with JWST transmission spectra" (gold `similar`,
answered `none` at 0.44, below any sensible threshold). On the negative
stratum, "citation analysis techniques in astronomy bibliometrics",
"self-citation rates", "trending towards lower masses" and the other 37 all
come back `none`.

Every gold-`none` false positive across the three Jev arms is the same word:
"book reviews of cosmology textbooks" (gold `doctype:bookreview`), "the
review" and "that big survey" (ambiguous stratum, gold `none` with
`needs_clarification`). All three are `reviews` at confidence 0.48 to 0.95.
The book-review confusion appeared on val too and is the one option-text fix
this set recommends.

Haiku's seven operator errors split differently from Jev's. It matches Jev
on the positive stratum (41 / 42; its one miss is "sources underpinning the
Bullet Cluster dark matter paper", gold `references`, answered `none`) and
beats every Jev arm on the enum-synonym items (40 / 40, no book-review
confusion). It loses on the negative stratum, where the three "useful"
false positives above are the only operator false positives any arm makes
on those 40 items, and on the ambiguous stratum (27 / 30: "latest results"
answered `trending`, "cite this" and "what does it cite" answered `none`).
Its `confidence` is 1.0 on every answered field and 0.0 on `unknown`, so
its calibration (ECE 0.039) and selective-accuracy curve are flat: 0.973
at every threshold. Jev's curve rises to 1.000 at 90% coverage because its
misses are low-confidence. On the enum fields Haiku volunteers
`collection:astronomy` on 97 of the 152 items (set F1 0.06) and changes
that answer on 43% of items across the three stability repeats; its other
enum F1s are property 0.86, doctype 0.83, bibgroup 0.64. Gate accuracy is
higher than Jev's (search_kind 0.93, needs_clarification 0.93,
refers_to_specific_paper 0.98).

Arm D is worse than B here (38 / 42 on the positive stratum). With the regex
IntentSpec in its state, and the regex saying "no operator" on every one of
these items, Jev defers to it on four ("later studies drawing on the Salpeter
initial mass function", "what did the JWST early release science team build
upon"). The structural context that helped on the benchmark, where the regex
is usually right, hurts on the set where it is always wrong. That is the
expected sign of a prior, and it is why arm E (which does not pass the regex
state) is the recommended shape.

Enum fields on the 40 enum-synonym items (set F1 B / D / E): property 0.95 /
1.00 / 0.95, doctype 0.87 / 0.47 / 0.83, bibgroup 0.58 / 0.47 / 0.58,
collection 0.32 / 0.00 / 0.32. Jev reads "nothing unrefereed",
"e-print postings", "manuscripts posted before journal publication",
"doctoral dissertations" and "anyone can read without paying" correctly.
The residual errors are the same shape as on val: `collection:astronomy`
volunteered on astronomy topics, `doctype:eprint` added alongside
`property:eprint`, and the book-review item. Arm D's doctype F1 drop is the
regex `article` prior again.

Gate fields, which only this set labels (B / D / E accuracy): search_kind
0.88 / 0.85 / 0.89; needs_clarification 0.82 / 0.84 / 0.83 with recall 1.0
and precision 0.52 (Jev flags every genuinely underspecified item and about
as many that a librarian would just run); refers_to_specific_paper 0.89 /
0.88 / 0.89 with precision 1.0 and recall 0.43 (it never invents a specific
paper, and misses more than half of the ones the labels say are there,
mostly "the first LIGO detection paper" style descriptions). As a resolver
gate the precision is what matters; the recall means the resolver would be
skipped on some items it should run on.

Stability over 3 uncached repeats of 100 items: operator changed on 2% (B),
3% (D), 1% (E) of items, all at confidence below 0.6; enums 0% to 6%.

## End to end

Assembled-query results from `scripts/evaluate_semantic_overlap.py
--intent-backend`, ADS result-set overlap against the gold query (Jaccard at
50 rows; semantic match is Jaccard at or above 0.5). Only the three deployable
backends were run end to end.

| dataset | backend | valid | ADS timeouts | empty query | semantic match | mean Jaccard |
|---|---|---|---|---|---|---|
| benchmark | regex | 91.3% | 21 | 1 | 32.4% | 0.359 |
| benchmark | jev | 89.7% | 23 | 3 | 30.4% | 0.350 |
| benchmark | jev_gated | 92.1% | 19 | 1 | 31.2% | 0.353 |
| val | regex | 97.5% | 10 | 2 | 27.0% | 0.280 |
| val | jev | 97.5% | 7 | 5 | 26.8% | 0.286 |
| val | jev_gated | 97.3% | 9 | 4 | 27.4% | 0.291 |

The headline is flat: every backend lands within a point or two of the
others, and all of them sit far below the 70% semantic-match target. Two
things explain the flatness.

- **The end-to-end number is dominated by topic extraction, which no arm
  changes.** Regex topic spans like `abs:"discussing supermassive black hole
  growth"` or `abs:"using panstarrs"` miss the gold result set whatever the
  operator and enum decisions are. The intent classifier can only move items
  where the operator or an enum field is the deciding factor.
- **The measurement is noisy at the level of the differences.** The "invalid"
  queries are almost all ADS API timeouts on the second-order operators
  (`trending()`, `similar()`, `useful()`), and which items time out varies
  from run to run (8 to 10 benchmark items time out on one side only). Among
  items where two backends produced the identical query, 5 of 54 on the
  benchmark still scored differently (`similar(abs:"penrose singularities")`
  1.00 in one run and 0.00 in the next), so ADS itself returns different
  result sets for the same second-order query minutes apart.

Paired on items where both backends produced a valid query and the queries
differ, which is the only place the intent stage can show up:

| dataset | comparison | items | jev better | regex better | mean Jaccard regex to jev |
|---|---|---|---|---|---|
| benchmark | regex vs jev | 169 | 23 | 16 | 0.324 to 0.328 |
| benchmark | regex vs jev_gated | 142 | 19 | 10 | 0.294 to 0.290 |
| val | regex vs jev | 354 | 50 | 21 | 0.263 to 0.270 |
| val | regex vs jev_gated | 341 | 51 | 18 | 0.266 to 0.279 |

Jev wins more items than it loses on both sets, by 2 to 1 on val, and the
mean moves by under a point. The wins are the two things the intent metrics
already showed: dropping the spurious `doctype:article` ("papers by
Chandrasekhar" 0.56 to 1.00) and recognising facilities as bibgroups
("studies using APEX data" 0.00 to 1.00). The losses are the same val jargon
false positives ("refereed Icarus papers" to `references(...)`, 1.00 to
0.00) and one dropped bibgroup ("papers citing Hubble deep field
observations", where Jev did not tag HST).

For criterion 4, the benchmark semantic-match rate drops 2.0 points for arm
B and 1.2 for arm E in the headline table, which is within the timeout noise
(4 and 3 points respectively), and the paired comparison shows no drop. The
rise on the held-out set is not demonstrated: val moves by less than a point
in either direction, and the paraphrase set is unscored.

## Criteria table

| # | criterion | result | status |
|---|---|---|---|
| 1 | B macro-F1 on paraphrase set ≥ A + 15 points and > C | B 0.955 vs A 0.146 (+81 points); vs C 0.933 (+2.2 points, and C makes 3 operator-negative false positives to B's 0) | met |
| 2 | B false-positive rate on operator-negative stratum ≤ 2% | 0 of 40 (B, D and E); on all gold-none items 3 of 106 (2.8%), all the word "review" | met |
| 3 | B selective accuracy ≥ 95% at 90% coverage | 1.000 held-out, 1.000 benchmark, 0.988 val | met |
| 4 | end-to-end semantic match: no drop on benchmark, rise on held-out | benchmark: headline -2.0 (B) / -1.2 (E) points, inside ADS timeout noise, paired shows no drop; val +0.4 (E), -0.2 (B); paraphrase set unscored | no drop shown; rise not shown |
| 5 | E p95 added latency ≤ 600 ms, mean cost ≤ $0.0001/query | p95 185 to 206 ms; $0.000086 benchmark, $0.000119 val, $0.000126 held-out | latency met; cost met on benchmark only |

## Caveats

- **Benchmark contamination.** The regex patterns were written against it;
  its round-2 Jev numbers are also tuned, since the operator option text was
  rewritten after reading its round-1 failures. The val set is the only
  untuned number in this report, and its NL is synthetic (generated from ADS
  queries), so it is easier than user input for every arm and its operator
  classes are tiny (3 to 11 items each).
- **Label noise in val.** At least six "reviews" items carry the word in the
  NL and no operator in the gold. They count against every arm that reads
  the text and for the regex that ignores it.
- **The held-out set was drafted by Claude Fable 5.1** and reviewed by one
  person. It is 152 items, so a single item is 0.7 points of accuracy and
  the per-class operator numbers rest on 5 to 10 items each. The drafting
  model is excluded from arm C.
- **Arm C on the held-out set went through the `claude` CLI**, not the
  Messages API, because the API account had no credit. Same model, same
  system prompt, same JSON schema, thinking disabled, no tools, settings and
  CLAUDE.md excluded; but the CLI delivers structured output through its own
  tool wrapper, so each call carried about 4,600 input tokens against
  3,200 over the API, and the latency (p50 5.6 s, p95 7.1 s at four
  concurrent calls) includes CLI process start-up and is not comparable to
  the API numbers on the benchmark and val rows. The $/query figure is
  computed at API list price from the reported tokens; the run was billed
  to the Claude subscription, not the API account. Classification accuracy
  is the quantity the criterion uses and is unaffected by the transport.
- **Property F1 is restricted** to the three values any arm produces. Gold
  uses many more (`notrefereed`, `article`, `ads_openaccess`, ...); those are
  out of scope for every arm alike.
- **The regex doctype behaviour** ("papers" → `doctype:article`) is a shipped
  bug against the gold convention, found by this experiment and not fixed
  here. It depresses A and D and should be fixed regardless of the Jev
  decision.
- **Token cost is 2.4x the plan's estimate.** The option lists are the
  reason; a shorter bibgroup list (the 10 facilities that appear in the data)
  would roughly halve it.
- **Jev confidence is a model output** and was measured calibrated on these
  two sets; that is a result, not a premise, and it was measured on
  `jev-1.13.0` only.
- **End-to-end overlap is noisy.** ADS times out on 4% to 9% of second-order
  queries per run and returns different result sets for the same
  `similar()` query minutes apart; differences under about 3 points between
  backends are not distinguishable from that.
- **Latency** is this machine to TypeSafe with four concurrent requests.
- **No real user queries** are in any dataset.

## Artifacts

- Rows and metrics: `data/datasets/evaluations/intent_classifiers_{benchmark,val}{,_v2}_2026-09-24{.jsonl,_metrics.json}`
  (round 1 without suffix, round 2 with `_v2`).
- Held-out: `data/datasets/evaluations/intent_classifiers_heldout_2026-09-24{.jsonl,_metrics.json}` (arms A, B, D, E; the arm C rows in this file are the failed API attempts) and `intent_classifiers_heldout_c_2026-09-24{.jsonl,_metrics.json}` (arm C via the CLI; `llm_transport: cli` in the metadata).
- End to end: `data/datasets/evaluations/semantic_overlap_pipeline_{regex,jev,jev_gated}_{benchmark,val}_2026-09-24.json`.
- Request caches: `data/cache/jev_systemone.jsonl`, `data/cache/llm_intent.jsonl`.
- Labels: `data/datasets/evaluations/intent_labels.jsonl`.
- Held-out paraphrases (approved): `data/datasets/benchmark/heldout_paraphrases.json`; review sheet `reports/heldout-paraphrase-review-sheet.md`; approval recorded with `scripts/approve_heldout_paraphrases.py`.
- Code: `packages/finetune/src/finetune/domains/scix/{jev_intent,jev_option_text,llm_intent}.py`,
  `scripts/{evaluate_intent_classifiers,intent_metrics,derive_intent_labels,check_heldout_paraphrases}.py`,
  `--intent-backend` on `scripts/evaluate_semantic_overlap.py`, `INTENT_BACKEND` on `docker/server.py`.
