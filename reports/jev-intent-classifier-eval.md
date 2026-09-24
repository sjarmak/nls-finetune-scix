# Jev typed classifiers as the IntentSpec reasoning stage: evaluation

Run date: 2026-09-24. Plan and pre-registered criteria:
`docs/research/jev-intent-classifier-experiment.md`. Model pinned to
`jev-1.13.0` (recorded on every row). Comparison LLM: `claude-haiku-4-5`
(served as `claude-haiku-4-5-20251001`).

## Decision

**Go on the intent stage.** The decision was made on the round-2 numbers
and re-checked in round 3 (section "Round 3: after the rollout fixes
(2026-09-24)") with the code now on main. Where the rounds differ, round 3
describes main.

On the 152-item held-out paraphrase set, reviewed and approved by Stephanie
before any model saw it, Jev scores 0.986 operator macro-F1 against 0.146
for the regex, an 84-point margin on a set built from phrasings the regex
has never seen (0.955 in round 2, before the book-review and sky-survey
option-text fix). It makes zero operator false positives on the 40-item
operator-negative stratum in both rounds, the conflation failure the hybrid
pipeline exists to prevent, and its operator accuracy at 90% coverage is
100%. The gated deployable shape (arm E) scores 0.975, meets the latency
criterion everywhere, and meets the cost criterion on the benchmark only
($0.000088 per query), missing it by 22% on val and 29% on the held-out set,
where the gate opens for nearly every query. Criteria 1, 2 and 3 are met.

The Haiku comparison (arm C) closes the "exceeds arm C" half of criterion 1.
Asked the same questions with the same schema, Claude Haiku 4.5 scores 0.922
operator macro-F1 on the held-out set in round 3 (0.933 in round 2), 6.4
points behind Jev, and it makes three operator false positives on the
operator-negative stratum (7.5%) where Jev makes none, in both rounds. All
three are the word "useful" or a synonym read as the `useful` operator
("useful yield in solar cell efficiency modelling", "foundational models for
spectral classification", "must-have calibrations for CCD photometry"), each
at Haiku's full confidence. That is the conflation failure criterion 2
exists to catch: Haiku as the intent stage would fail the no-go criterion
outright. Haiku also has no usable abstention signal on this task (it
answered `unknown` for the operator on 8 of 152 held-out items in round 3
and never otherwise hedged), so a threshold cannot buy back precision,
whereas Jev's low-confidence answers sit below any sensible cut. The win is
attributable to the typed classifier, not only to asking narrow questions,
though the narrow questions do most of the work for both.

End to end, the intent stage still does not move the assembled-query
overlap. The regex doctype fix now on main (the regex no longer turns
"papers" into `doctype:article`) raised the regex backend's benchmark
semantic match from 32.4% to 36.8%, and removed the one systematic error
that gave Jev its round-2 end-to-end wins. With it gone, the regex backend
leads the gated Jev backend by 2.8 points on the benchmark (36.8% to 34.0%)
and ties it on val (27.8% each). On the 58 benchmark items where the two
build different valid queries, the regex scores higher on 6 and Jev on 2;
on val, Jev wins 15 of 112 and loses 6. The benchmark gap is inside the
run-to-run noise of ADS (5 items time out on one side only, 2.0 points), but
it is in the wrong direction. Five of the six losses trace to a Jev enum
decision (an added `doctype:article`, `doctype:eprint` or
`collection:astronomy`, or a dropped `bibgroup:HST` or `doctype:software`);
the sixth is `useful` for "top 5 papers on gravitational waves", a `topn`
request no arm supports. Criterion 4 is therefore
"no drop not shown on the benchmark, rise not shown".

The recommendation is to adopt arm E (regex first, Jev when the regex finds
no operator or is unsure) as the operator and gate decider in the hybrid
route, keep the regex as the extractor for names, years and topics, and
route on min(structural confidence, Jev operator confidence), which main now
does. Two changes would close the remaining gaps. First, the regex fires the
`reviews` operator on "book reviews" at 0.9 confidence, which closes the
gate, so the book-review option-text fix never reaches arm E; the regex
needs the same book-review rule Jev now has. Second, Jev's enum answers
(doctype, collection) cost more end to end than they win; keeping the regex
enum fields when the regex sets them, or applying Jev's enum answers only
above a confidence threshold, is the next experiment. The regex IntentSpec
should still not be passed to Jev as state (arm D): with the regex doctype
fixed, D no longer inherits a doctype error, but it still defers to the
regex's "no operator" on the held-out positive stratum (39 of 42, against 42
of 42 for E).

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

Three rounds were run. Rounds 1 and 2 are the pre-registered experiment
and are recorded as run in the sections that follow; round 3 re-scored
every arm after the rollout fixes and has its own section. Round 1 used the first draft of the operator option
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
set being looked at). All numbers in the sections from "Operator
classification" to "End to end" are round 2 unless labelled.

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
  operator, and the doctype option text can say so (done in `66d02e2`).
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
  and follows it (exact match 0.87 to 0.89). (Fixed on main in `78ecc49`;
  round 3 has the new numbers.)
- **Arm D inherits the regex's doctype error.** With the regex IntentSpec in
  its state, Jev keeps `article` far more often (exact match drops from 0.87
  to 0.51). Structural context helps on the operator and hurts on doctype:
  Jev defers to a wrong prior. (With the regex fixed on main, arm D's
  doctype F1 is the best of any arm on benchmark and val in round 3.)
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
this set recommended. (Applied on main in `66d02e2`; round 3 has the
outcome.)

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

## Round 3: after the rollout fixes (2026-09-24)

Every arm was re-scored on all three datasets with the code on main after
these commits:

- `66d02e2`: the Jev option text separates book reviews (a document type)
  and sky surveys (a data source) from the `reviews` operator
  (`jev_option_text.py`).
- `78ecc49`: the regex no longer maps generic "papers", "articles" or
  "publications" to `doctype:article`; explicit phrasings seen in gold
  ("journal articles", "PhD theses", "conference talks", "technical
  reports", "book chapters") still map.
- `b341558`: the pipeline serves the regex intent when Jev fails or times
  out (`classifier_error`; `JEV_TIMEOUT_S`, default 2 s).
- `48de081`: routing confidence is min(structural, Jev operator confidence)
  when Jev answered.
- `336547c`: `SHADOW_INTENT_BACKEND=jev|jev_gated` serves the regex and runs
  Jev off the request path, writing `intent_shadow` telemetry rows.

The option-text change alters every Jev request fingerprint, so arms B, D
and E made fresh Jev calls. The Haiku prompt is rendered from the same
question set (`llm_intent.build_prompt`), so the change altered every arm C
fingerprint too: the first arm C attempt over the API missed the cache on
all 253 benchmark and 478 val rows and failed on the account's zero credit
balance. Arm C was therefore re-run through the `claude` CLI on all three
sets (`--llm-transport cli`), which differs from the round-2 benchmark and
val arm C runs in transport as well as prompt; see Caveats. Stability
settings match round 2: 3 repeats of the first 100 items on the benchmark
and held-out sets, none on val.

### Intent classification, all three datasets

Operator macro-F1 (round 2 to round 3), accuracy at 90% coverage, ECE,
doctype set F1 with exact match in parentheses, latency, cost and the
share of items that changed operator answer across the three repeats.

| dataset | arm | op macro-F1 | FP on gold none | acc at 90% cov | ECE | doctype F1 (exact) | p50 / p95 ms | $/query | op unstable |
|---|---|---|---|---|---|---|---|---|---|
| benchmark | A regex | 0.938 to 0.938 | 1/147 | 0.988 (no signal) | 0.038 | 0.737 (0.964) | 0 / 0 | 0 | n/a |
| benchmark | B jev | 0.980 to 0.984 | 1/147 | 1.000 | 0.027 | 0.500 (0.881) | 159 / 269 | $0.000131 | 0.00 |
| benchmark | C haiku (CLI) | 0.984 to 0.947 | 0/147 | 0.966 (no threshold) | 0.034 | 0.632 (0.917) | 4,238 / 7,392 | $0.00604 | 0.09 |
| benchmark | D jev+state | 0.995 to 0.995 | 1/147 | 1.000 | 0.029 | **0.750** (0.953) | 160 / 245 | $0.000138 | 0.00 |
| benchmark | E gated | 0.991 to 0.991 | 2/147 | 0.996 | 0.018 | 0.526 (0.905) | 149 / 269 | $0.000088 | 0.00 |
| val | A regex | 0.814 to 0.814 | 5/436 | 0.824 (no signal) | 0.126 | 0.556 (0.971) | 0 / 0 | 0 | n/a |
| val | B jev | 0.830 to 0.842 | 11/436 | 0.995 | 0.019 | 0.429 (0.900) | 157 / 223 | $0.000131 | not run |
| val | C haiku (CLI) | 0.855 to 0.868 | 11/436 | 0.972 (no threshold) | 0.030 | 0.405 (0.908) | 4,168 / 6,772 | $0.00586 | not run |
| val | D jev+state | 0.899 to 0.894 | 8/436 | 1.000 | 0.006 | **0.588** (0.958) | 157 / 228 | $0.000138 | not run |
| val | E gated | 0.795 to 0.798 | 16/436 | 0.981 | 0.024 | 0.415 (0.902) | 154 / 219 | $0.000122 | not run |
| held-out | A regex | 0.146 to 0.146 | 1/106 | 0.500 (no signal) | 0.450 | 0.000 (0.842) | 0 / 0 | 0 | n/a |
| held-out | B jev | 0.955 to **0.986** | 1/106 | **1.000** (thr 0.73) | 0.072 | **0.870** (0.961) | 150 / 203 | $0.000131 | 0.00 |
| held-out | C haiku (CLI) | 0.933 to 0.922 | 4/106 | 0.958 (no threshold) | 0.079 | 0.857 (0.961) | 5,880 / 7,884 | $0.00656 | 0.05 |
| held-out | D jev+state | 0.923 to 0.930 | 2/106 | 0.993 | 0.040 | 0.773 (0.934) | 153 / 200 | $0.000138 | 0.01 |
| held-out | E gated | 0.965 to 0.975 | 2/106 | 0.993 | 0.065 | 0.826 (0.954) | 152 / 218 | $0.000129 | 0.01 |

Arm E called Jev on 67.6% of benchmark, 92.9% of val and 98.7% of held-out
items. The mean Jev request grew from 2,903 to 3,118 input tokens with the
longer option text, which is the whole cost increase.

Held-out operator accuracy by stratum, round 2 to round 3 (correct / items):

| arm | operator-positive (42) | operator-negative (40) | enum synonyms (40) | ambiguous (30) |
|---|---|---|---|---|
| A regex | 0 to 0 | 40 to 40 | 39 to 39 | 27 to 27 |
| B jev | 41 to **42** | 40 to 40 | 39 to **40** | 28 to **29** |
| C haiku | 41 to 40 | 37 to 37 | 40 to 40 | 27 to 27 |
| D jev+state | 38 to 39 | 40 to 40 | 39 to 39 | 28 to 28 |
| E gated | 42 to 42 | 40 to 40 | 39 to 39 | 28 to **29** |

Arm B moved on three held-out items: "book reviews of cosmology textbooks"
(`reviews` at 0.53 to `none` at 0.73), "that big survey" (`reviews` at 0.48
to `none` at 0.91), and "kindred studies to exoplanet atmosphere retrievals
with JWST transmission spectra" (`none` at 0.44 to `similar` at 0.52, a
borderline item whose confidence crossed 0.5).
Its one remaining error is "the review" (gold `none` with
`needs_clarification`, answered `reviews` at 0.71). On val, B's operator
false positives on gold `none` fell from 14 to 11.

Arm C in round 3 differs from round 2 in two ways at once (prompt and, on
benchmark and val, transport), so its benchmark drop from 0.984 to 0.947 is
not attributable to either. Ten benchmark items changed: two new correct
answers and eight new errors, all eight `none` ("what does Einstein cite",
"related to pulsar timing research", "survey of AGN variability"). Its three "useful" false positives on the
held-out negative stratum are unchanged. Through the CLI its operator
answer changed across repeats on 9% of benchmark items, against 0% over the
API in round 2.

### Book-review items

Every item in the three datasets whose gold doctype is `bookreview`, round
2 to round 3 (operator / doctype):

| item | gold | A regex | B jev | E gated |
|---|---|---|---|---|
| "book reviews on cosmology" (benchmark) | none / bookreview | reviews / book | reviews to **none** / bookreview | reviews / book |
| "textbook reviews from 2020" (val) | none / bookreview | none / [] to none / bookreview | reviews to **none** / bookreview | reviews to **none** / bookreview |
| "recent reviews of astronomy books" (val) | none / bookreview | reviews / book | reviews to **none** / bookreview | reviews / book |
| "textbook reviews related to astrophysics" (val) | none / bookreview | similar / [] to similar / bookreview | reviews to **none** / bookreview | similar / [] to similar / bookreview |
| "book reviews of cosmology textbooks" (held-out) | none / bookreview | reviews / book | reviews to **none** / bookreview | reviews / book |

The option-text fix works for Jev: arm B now gets all five right. It
reaches arm E on only one of the five, because on the other four the regex
fires an operator (`reviews` on "book reviews" and "reviews of ... books",
`similar` on "related to") at 0.9 structural confidence, which closes the
gate, and E serves the regex answer with `doctype:book`. The book-review
error in the deployed shape is now a regex error; the fix belongs in
`ner.py`, which is outside this report's scope.

**Follow-up, same day.** `ner.py` now treats "book review(s)" and "reviews
of ... books" as the `bookreview` document type before the operator rules
run. The regex alone now gets four of the five items right (operator
`none`, doctype `bookreview`). "textbook reviews related to astrophysics"
still fires `similar` on "related to", now with doctype `bookreview`. The
tables above were scored before this change and were not re-run. The
held-out item "book reviews of cosmology textbooks" is now in-map for the
regex; the held-out file marks it `regex_in_map`, and the check script
lists it. Its effect on any re-score can only favour arm A.

### The regex doctype change

Arm A's doctype set F1 rose from 0.080 to 0.737 on the benchmark (exact
match 0.387 to 0.964) and from 0.031 to 0.556 on val (0.368 to 0.971). On
the held-out set, exact match rose from 0.638 to 0.842 (31 fewer spurious
`article` answers), while F1 stays at 0.000 because the regex matches none
of the paraphrased document types ("doctoral dissertations", "conference
proceedings"). The remaining benchmark disagreements are 9 items, one of
them a spurious `article`. Arm D, which sees the regex IntentSpec, gains
the most: its benchmark doctype F1 rose from 0.211 to 0.750, now the best
of any arm, and on the held-out set from 0.466 to 0.773. Arms B and E move
by a few points only, through the option-text change.

End to end, the same change is the largest movement in this report:

| dataset | backend | round | valid | ADS timeouts | empty query | semantic match | mean Jaccard |
|---|---|---|---|---|---|---|---|
| benchmark | regex | 2 | 91.3% | 21 | 1 | 32.4% | 0.359 |
| benchmark | regex | 3 | 95.3% | 11 | 1 | **36.8%** | **0.400** |
| benchmark | jev_gated | 2 | 92.1% | 19 | 1 | 31.2% | 0.353 |
| benchmark | jev_gated | 3 | 94.1% | 14 | 1 | 34.0% | 0.378 |
| val | regex | 2 | 97.5% | 10 | 2 | 27.0% | 0.280 |
| val | regex | 3 | 98.1% | 7 | 2 | 27.8% | 0.292 |
| val | jev_gated | 2 | 97.3% | 9 | 4 | 27.4% | 0.291 |
| val | jev_gated | 3 | 97.3% | 9 | 4 | 27.8% | 0.295 |

Paired on items where both runs produced a valid query and the queries
differ:

| dataset | comparison | items | second better | first better | mean Jaccard, first to second |
|---|---|---|---|---|---|
| benchmark | regex round 2 vs regex round 3 | 145 | 29 | 6 | 0.296 to 0.354 |
| val | regex round 2 vs regex round 3 | 292 | 42 | 13 | 0.264 to 0.282 |
| benchmark | regex round 3 vs jev_gated round 3 | 58 | 2 | 6 | 0.361 to 0.320 |
| val | regex round 3 vs jev_gated round 3 | 112 | 15 | 6 | 0.315 to 0.327 |

Part of the benchmark rise is ADS itself: round 3 saw 11 regex timeouts
against 21, and 3 of the 86 items whose regex query did not change between
rounds scored differently. The paired rows, which exclude both effects,
still show the regex fix winning 29 items to 6 on the benchmark and 42 to
13 on val.

The fix also removes Jev's round-2 end-to-end advantage. In round 2 the
regex and jev_gated queries differed on 142 benchmark items, mostly by the
regex's `doctype:article`; in round 3 they differ on 58, and there the
regex wins 6 to 2. Jev's two wins are facility bibgroups ("VLA radio
observations" 0.00 to 1.00). Its losses are enum decisions: `doctype:eprint`
added to "arxiv open access papers on black holes" (0.79 to 0.00),
`doctype:software` dropped from "open access software papers" (1.00 to
0.00), `bibgroup:HST` dropped from "JWST or HST papers on exoplanet
atmospheres" (1.00 to 0.49), and `collection:astronomy` added to two
cosmology queries (1.00 to 0.96); plus `useful` for "top 5 papers on
gravitational waves", a `topn` request. On val Jev still wins more than it
loses (15 to 6), again by facility bibgroups ("studies using APEX data",
"CTIO data", "X-ray bursts research with Rossi XTE", each 0.00 to 1.00),
and loses on the val jargon items ("refereed Icarus papers" to
`references(...)`, "supernova remnants review" to `reviews(...)`).

Noise is as in round 2: on the benchmark 5 items timed out in one backend's
run only (2.0 points of semantic match) and 4 of the 179 items with an
identical query in both backends scored differently; on val 4 items timed
out on one side only.

### Routing on Jev's operator confidence

The server falls back to the model when the pipeline's confidence is below
`PIPELINE_CONFIDENCE_THRESHOLD` (0.5). `pipeline.routing_confidence` makes
that confidence min(structural, Jev operator confidence) whenever Jev
answered, and the structural heuristic alone otherwise (regex intent, gate
closed, or Jev failed). Applying it to the round-3 held-out rows (the served
IntentSpec rebuilt from each row and scored with
`compute_pipeline_confidence`):

| backend | routing | routed to model | served by pipeline | operator accuracy on served items |
|---|---|---|---|---|
| regex | structural | 11 | 141 | 0.688 (97/141) |
| jev_gated | structural only | 10 | 142 | 0.986 (140/142) |
| jev_gated | min(structural, Jev) | 13 | 139 | 0.986 (137/139) |

The rule sends three more items to the model, each with a Jev operator
confidence under 0.5: "what did the JWST early release science team build
upon" (`references`, 0.45), "citation networks among gravitational wave
papers" (`none`, 0.41) and "foundational models for spectral
classification" (`none`, 0.49). All three were answered correctly in this
run, so on this draw the rule costs three model calls and changes no served
answer. It earns its place on the next draw: replaying the held-out set
through `process_query` from the Jev cache, which holds a later repeat of
each request, returns `useful` at 0.45 for "foundational models for
spectral classification", a false positive on the operator-negative
stratum, and the rule routes it to the model (12 routed, 140 served at
0.986, no fallbacks). The two errors the pipeline still serves are both
above the threshold: "book reviews of cosmology textbooks" (regex `reviews`
at 0.9, Jev never called) and "the review" (Jev `reviews` at 0.71).

The held-out set has intent labels, not gold ADS queries, so it cannot be
scored end to end. The routing table is the nearest measure of the "rise on
held-out" half of criterion 4: with the gated Jev backend the pipeline
serves 139 of 152 held-out requests with 98.6% operator accuracy, against
141 at 68.8% for the regex backend.

## Criteria table

Round 3 values (the code on main); round 2 in parentheses where it differs.

| # | criterion | result | status |
|---|---|---|---|
| 1 | B macro-F1 on paraphrase set ≥ A + 15 points and > C | B 0.986 (0.955) vs A 0.146, +84 points; vs C 0.922 (0.933), +6.4 points, and C makes 3 operator-negative false positives to B's 0 | met |
| 2 | B false-positive rate on operator-negative stratum ≤ 2% | 0 of 40 (B, D and E, both rounds); on all gold-none items 1 of 106 (3 of 106), "the review" | met |
| 3 | B selective accuracy ≥ 95% at 90% coverage | 1.000 held-out, 1.000 benchmark, 0.995 (0.988) val | met |
| 4 | end-to-end semantic match: no drop on benchmark, rise on held-out | benchmark: jev_gated 34.0% vs regex 36.8%, -2.8 points (round 2: -1.2), inside ADS noise of about 2 points of one-sided timeouts but in the wrong direction; paired regex 6 wins to Jev 2 on 58 differing items (round 2: Jev 19 to 10). Val: 27.8% each; paired Jev 15 to 6. Held-out has no gold queries; as a proxy the jev_gated pipeline serves 139 of 152 at 0.986 operator accuracy vs 141 at 0.688 for regex | no drop not shown; rise shown on operator accuracy only, not end to end |
| 5 | E p95 added latency ≤ 600 ms, mean cost ≤ $0.0001/query | p95 218 to 269 ms (185 to 206); $0.000088 benchmark, $0.000122 val, $0.000129 held-out ($0.000086 / $0.000119 / $0.000126) | latency met; cost met on benchmark only |

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
  to the Claude subscription, not the API account. In round 3 all three
  sets went through the CLI; see "Arm C changed transport between rounds"
  below.
- **Property F1 is restricted** to the three values any arm produces. Gold
  uses many more (`notrefereed`, `article`, `ads_openaccess`, ...); those are
  out of scope for every arm alike.
- **The regex doctype behaviour** ("papers" → `doctype:article`) depressed
  arms A and D in rounds 1 and 2. It is fixed on main (`78ecc49`) and round
  3 measures the effect. The regex still fires the `reviews` operator on
  "book reviews", which closes the arm E gate before Jev sees the request.
- **Arm C changed transport between rounds.** In round 3 the Haiku prompt
  changed with the option text, every request missed the cache, and all
  three sets ran through the `claude` CLI. The round-2 benchmark and val arm
  C numbers came from the Messages API. The two rounds of arm C differ in
  prompt and transport together, and the CLI runs are less stable (operator
  answer changed on 9% of benchmark items across repeats).
- **Token cost is 2.6x the plan's estimate** (3,118 input tokens per Jev
  call in round 3, 2,903 in round 2, 1,200 planned). The option lists are
  the reason; a shorter bibgroup list (the 10 facilities that appear in the
  data) would roughly halve it.
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
- Round 3 (the code on main): `data/datasets/evaluations/intent_classifiers_{benchmark,val,heldout}_v3_2026-09-24{.jsonl,_metrics.json}` (arms A, B, D, E) and `intent_classifiers_{benchmark,val,heldout}_c_v3_2026-09-24{.jsonl,_metrics.json}` (arm C via the CLI); end to end `semantic_overlap_pipeline_{regex,jev_gated}_{benchmark,val}_v3_2026-09-24.json`.
- Request caches: `data/cache/jev_systemone.jsonl`, `data/cache/llm_intent.jsonl`.
- Labels: `data/datasets/evaluations/intent_labels.jsonl`.
- Held-out paraphrases (approved): `data/datasets/benchmark/heldout_paraphrases.json`; review sheet `reports/heldout-paraphrase-review-sheet.md`; approval recorded with `scripts/approve_heldout_paraphrases.py`.
- Code: `packages/finetune/src/finetune/domains/scix/{jev_intent,jev_option_text,llm_intent}.py`,
  `scripts/{evaluate_intent_classifiers,intent_metrics,derive_intent_labels,check_heldout_paraphrases}.py`,
  `--intent-backend` on `scripts/evaluate_semantic_overlap.py`, `INTENT_BACKEND` and `SHADOW_INTENT_BACKEND` on `docker/server.py`, `scripts/summarize_intent_shadow.py`, `routing_confidence` in `pipeline.py`.
