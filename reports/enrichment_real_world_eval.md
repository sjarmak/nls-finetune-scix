# SciX Enrichment Model — Real-World Evaluation Report

## Summary


- **Test abstracts**: 100
- **Gold spans**: 4795
- **Predicted spans**: 2648
- **Overall Precision**: 0.1333
- **Overall Recall**: 0.0736
- **Overall F1 (micro)**: 0.0949
- **Macro F1 (by type)**: 0.0576


## Synthetic-to-Real Performance Gap

| Metric | Synthetic Test | Real ADS Abstracts | Delta |
|--------|---------------|-------------------|-------|
| Precision | 0.9993 | 0.1333 | -0.8660 |
| Recall | 0.9993 | 0.0736 | -0.9257 |
| F1 | 0.9993 | 0.0949 | -0.9044 |

**Performance gap**: F1 dropped by **0.9044** from synthetic to real data.


### Per-Type Gap


| Type | Synthetic F1 | Real F1 | Delta |

|------|-------------|---------|-------|

| entity | 1.0000 | 0.0131 | -0.9869 |

| topic | 0.9991 | 0.1022 | -0.8969 |




## Detailed Metrics


### By Entity Type

| Slice | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| entity | 0.0588 | 0.0074 | 0.0131 | 4 | 64 | 538 |
| topic | 0.1353 | 0.0821 | 0.1022 | 349 | 2231 | 3904 |

### By Domain

| Slice | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| astronomy | 1.0000 | 0.2178 | 0.3577 | 159 | 0 | 571 |
| earthscience | 1.0000 | 0.0539 | 0.1023 | 190 | 0 | 3333 |
| multidisciplinary | 1.0000 | 0.0090 | 0.0179 | 4 | 0 | 438 |
| planetary | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 100 |
| unknown | 0.0000 | 0.0000 | 0.0000 | 0 | 2295 | 0 |

### By Source Vocabulary

| Slice | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| gcmd | 1.0000 | 0.1163 | 0.2083 | 15 | 0 | 114 |
| planetary | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 100 |
| predicted_fp | 0.0000 | 0.0000 | 0.0000 | 0 | 2295 | 0 |
| ror | 1.0000 | 0.0090 | 0.0179 | 4 | 0 | 438 |
| sweet | 1.0000 | 0.0516 | 0.0981 | 175 | 0 | 3219 |
| uat | 1.0000 | 0.2178 | 0.3577 | 159 | 0 | 571 |


## Correct Predictions



## Error Analysis


**Example 1** (`2020A&A...641A...6P`)

> We present cosmological parameter results from the final full-mission Planck measurements of the cosmic microwave background (CMB) anisotropies, combi...

- TP=4, FP=42, FN=99

- MISSED: `present` [topic]

- MISSED: `from` [topic]

- MISSED: `Planck` [entity]

- MISSED: `cosmic microwave background` [topic]

- MISSED: `from` [topic]

- SPURIOUS: `cosmological parameter` [topic]

- SPURIOUS: `the cosmic microwave background (CMB) anisotropies` [topic]

- SPURIOUS: `the` [topic]

- SPURIOUS: `temperature` [topic]

- SPURIOUS: `polarization maps` [topic]



**Example 2** (`1998AJ....116.1009R`)

> We present spectral and photometric observations of 10 Type Ia supernovae (SNe Ia) in the redshift range 0.16 &lt;= z &lt;= 0.62. The luminosity dista...

- TP=6, FP=56, FN=100

- MISSED: `present` [topic]

- MISSED: `spectral` [topic]

- MISSED: `Type Ia supernovae` [topic]

- MISSED: `redshift` [topic]

- MISSED: `range` [topic]

- SPURIOUS: `photometric` [topic]

- SPURIOUS: `10 Type Ia supernovae (SNe Ia)` [topic]

- SPURIOUS: `luminosity distances` [topic]

- SPURIOUS: `SN Ia luminosity` [topic]

- SPURIOUS: `light curve shape` [topic]



**Example 3** (`1999ApJ...517..565P`)

> We report measurements of the mass density, Ω<SUB>M</SUB>, and cosmological-constant energy density, Ω<SUB>Λ</SUB>, of the universe based on the analy...

- TP=5, FP=27, FN=49

- MISSED: `constant` [topic]

- MISSED: `energy density` [topic]

- MISSED: `based on` [topic]

- MISSED: `type Ia supernovae` [topic]

- MISSED: `Cosmology` [topic]

- SPURIOUS: `the mass density,` [topic]

- SPURIOUS: `Ω<SUB>M</SUB>,` [topic]

- SPURIOUS: `cosmological-constant energy density,` [topic]

- SPURIOUS: `Ω<SUB>Λ</SUB>,` [topic]

- SPURIOUS: `the universe` [topic]



**Example 4** (`2020NatMe..17..261V`)

> SciPy is an open source scientific computing library for the Python programming language. SciPy 1.0 was released in late 2017, about 16 years after th...

- TP=4, FP=21, FN=33

- MISSED: `open` [topic]

- MISSED: `release` [topic]

- MISSED: `standard` [topic]

- MISSED: `algorithms` [topic]

- MISSED: `unique` [topic]

- SPURIOUS: `SciPy` [topic]

- SPURIOUS: `scientific computing` [topic]

- SPURIOUS: `SciPy` [topic]

- SPURIOUS: `SciPy` [topic]

- SPURIOUS: `scientific algorithms` [topic]



**Example 5** (`1998ApJ...500..525S`)

> We present a full-sky 100 μm map that is a reprocessed composite of the COBE/DIRBE and IRAS/ISSA maps, with the zodiacal foreground and confirmed poin...

- TP=4, FP=54, FN=75

- MISSED: `present` [topic]

- MISSED: `full` [topic]

- MISSED: `composite` [topic]

- MISSED: `ISSA` [entity]

- MISSED: `ISSA` [entity]

- SPURIOUS: `full-sky` [topic]

- SPURIOUS: `100 μm` [topic]

- SPURIOUS: `COBE` [topic]

- SPURIOUS: `DIRBE` [topic]

- SPURIOUS: `zodiacal foreground` [topic]



**Example 6** (`2016A&A...594A..13P`)

> This paper presents cosmological results based on full-mission Planck observations of temperature and polarization anisotropies of the cosmic microwav...

- TP=7, FP=47, FN=125

- MISSED: `based on` [topic]

- MISSED: `Planck` [entity]

- MISSED: `polarization` [topic]

- MISSED: `cosmic microwave background` [topic]

- MISSED: `radiation` [topic]

- SPURIOUS: `Planck` [topic]

- SPURIOUS: `temperature` [topic]

- SPURIOUS: `polarization anisotropies of` [topic]

- SPURIOUS: `the cosmic microwave background (CMB) radiation` [topic]

- SPURIOUS: `Planck nominal-mission` [topic]



**Example 7** (`1973A&A....24..337S`)

> The outward transfer of the angular momentum of the accreting matter leads to the formation of a disk around the black hole. The structure and radiati...

- TP=9, FP=67, FN=59

- MISSED: `outward` [topic]

- MISSED: `angular momentum` [topic]

- MISSED: `leads` [topic]

- MISSED: `black` [entity]

- MISSED: `radiation` [topic]

- SPURIOUS: `outward transfer` [topic]

- SPURIOUS: `the angular momentum of the accreting matter` [topic]

- SPURIOUS: `disk` [topic]

- SPURIOUS: `structure` [topic]

- SPURIOUS: `radiation spectrum` [topic]



**Example 8** (`2016PhRvL.116f1102A`)

> On September 14, 2015 at 09:50:45 UTC the two detectors of the Laser Interferometer Gravitational-Wave Observatory simultaneously observed a transient...

- TP=3, FP=9, FN=32

- MISSED: `Laser Interferometer Gravitational-Wave Observatory` [topic]

- MISSED: `observed` [topic]

- MISSED: `signal` [topic]

- MISSED: `signal` [topic]

- MISSED: `frequency` [topic]

- SPURIOUS: `ringdown of` [topic]

- SPURIOUS: `matched-filter` [topic]

- SPURIOUS: `luminosity distance` [topic]

- SPURIOUS: `41 0<SUB>-180</SUB><SUP>+160</SUP> Mpc` [topic]

- SPURIOUS: `source` [topic]



**Example 9** (`1975CMaPh..43..199H`)

> In the classical theory black holes can only absorb and not emit particles. However it is shown that quantum mechanical effects cause black holes to c...

- TP=10, FP=22, FN=25

- MISSED: `theory` [topic]

- MISSED: `quantum` [entity]

- MISSED: `create` [entity]

- MISSED: `left` [topic]

- MISSED: `odot` [entity]

- SPURIOUS: `particles` [topic]

- SPURIOUS: `hot bodies` [topic]

- SPURIOUS: `hkappa }/{2π k` [topic]

- SPURIOUS: `10^{ - 6} left(` [topic]

- SPURIOUS: `M_ odot }/M} right){}^ circ K` [topic]



**Example 10** (`2003ApJS..148..175S`)

> WMAP precision data enable accurate testing of cosmological models. We find that the emerging standard model of cosmology, a flat Λ-dominated universe...

- TP=5, FP=28, FN=54

- MISSED: `precision` [topic]

- MISSED: `accurate` [topic]

- MISSED: `cosmological models` [topic]

- MISSED: `find` [entity]

- MISSED: `emerging` [topic]

- SPURIOUS: `cosmological` [topic]

- SPURIOUS: `emerging standard model` [topic]

- SPURIOUS: `flat Λ-dominated universe seeded by` [topic]

- SPURIOUS: `nearly scale-invariant` [topic]

- SPURIOUS: `adiabatic Gaussian fluctuations` [topic]




## Performance Gap Analysis

The synthetic-to-real gap is expected because:

1. **Training data distribution**: The model was trained on template-generated snippets
   with catalog entries inserted deterministically. Real abstracts use natural language
   that may reference the same concepts with different phrasing.

2. **Text length**: Real abstracts are much longer (avg ~1,500 chars) than synthetic
   snippets (~50-200 chars). The SciBERT tokenizer truncates to 512 tokens,
   which may miss spans in later portions of long abstracts.

3. **Vocabulary mismatch**: Catalog labels are canonical forms (e.g., "dark matter")
   while abstracts may use abbreviations, acronyms, or alternative phrasings
   (e.g., "DM", "non-baryonic matter").

4. **Context effects**: In synthetic data, catalog entries appear in predictable
   template positions. In real text, the same terms appear in varied syntactic
   contexts that the model hasn't seen during training.

5. **Annotation methodology**: The "gold" annotations on real text are derived from
   catalog keyword matching, which has known limitations (false positives from
   short common terms, inability to handle abbreviations). This means some
   "errors" may actually be correct model behavior.

