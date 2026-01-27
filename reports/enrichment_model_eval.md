# SciX Enrichment Model — Evaluation Report

## Summary


- **Test records**: 1070
- **Gold spans**: 1366
- **Predicted spans**: 1366
- **Overall Precision**: 0.9993
- **Overall Recall**: 0.9993
- **Overall F1 (micro)**: 0.9993
- **Macro F1 (by type)**: 0.9995


## Baseline Comparison

| Metric | Keyword Baseline | NER Model | Delta |
|--------|-----------------|-----------|-------|
| Precision | 0.2556 | 0.9993 | +0.7437 |
| Recall | 0.9963 | 0.9993 | +0.0030 |
| F1 | 0.4069 | 0.9993 | +0.5924 |



## Detailed Metrics


### By Entity Type

| Slice | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| entity | 1.0000 | 1.0000 | 1.0000 | 293 | 0 | 0 |
| topic | 0.9991 | 0.9991 | 0.9991 | 1072 | 1 | 1 |

### By Domain

| Slice | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| astronomy | 1.0000 | 1.0000 | 1.0000 | 525 | 0 | 0 |
| earthscience | 1.0000 | 0.9982 | 0.9991 | 547 | 0 | 1 |
| multidisciplinary | 1.0000 | 1.0000 | 1.0000 | 145 | 0 | 0 |
| planetary | 1.0000 | 1.0000 | 1.0000 | 148 | 0 | 0 |
| unknown | 0.0000 | 0.0000 | 0.0000 | 0 | 1 | 0 |

### By Source Vocabulary

| Slice | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| gcmd | 1.0000 | 1.0000 | 1.0000 | 105 | 0 | 0 |
| planetary | 1.0000 | 1.0000 | 1.0000 | 148 | 0 | 0 |
| predicted_fp | 0.0000 | 0.0000 | 0.0000 | 0 | 1 | 0 |
| ror | 1.0000 | 1.0000 | 1.0000 | 145 | 0 | 0 |
| sweet | 1.0000 | 0.9977 | 0.9989 | 442 | 0 | 1 |
| uat | 1.0000 | 1.0000 | 1.0000 | 525 | 0 | 0 |


## Correct Predictions


**Example 1** (`enr_snp_f43abb82dd`)

> Mineralogical mapping around Jet Process Corporation (United States) from spectral data

- `Jet Process Corporation (United States)` [entity] — correct

**Example 2** (`enr_snp_a5a01f6bcb`)

> Topographic analysis of Israel Science Foundation using altimetry data

- `Israel Science Foundation` [entity] — correct

**Example 3** (`enr_snp_7eae4ce3ec`)

> We present new observations of droplet obtained from ground-based telescopes. Our analysis reveals significant variability in {topic} across different epochs. These findings have implications for models of FLOODING and related processes.

- `droplet` [topic] — correct
- `FLOODING` [topic] — correct

**Example 4** (`enr_snp_98b7660733`)

> Characterization of Tertiary stars in protoplanetary disks

- `Tertiary stars` [topic] — correct

**Example 5** (`enr_snp_f0c45f26f8`)

> A survey of BORA WINDS across stellar populations

- `BORA WINDS` [topic] — correct

**Example 6** (`enr_snp_8d86d53ef1`)

> On the relationship between SEVERE CYCLONIC STORMS (N. INDIAN) and stellar mass

- `SEVERE CYCLONIC STORMS (N. INDIAN)` [topic] — correct

**Example 7** (`enr_snp_6b8d6a82f4`)

> We present new observations of CHELICERATES obtained from ground-based telescopes. Our analysis reveals significant variability in {topic} across different epochs. These findings have implications for models of light freeze and related processes.

- `CHELICERATES` [topic] — correct
- `light freeze` [topic] — correct

**Example 8** (`enr_snp_20dd7b34b1`)

> Topographic analysis of Porter using altimetry data

- `Porter` [entity] — correct

**Example 9** (`enr_snp_a9dd61c4ae`)

> Research output from Galois F in the past decade

- `Galois F` [entity] — correct

**Example 10** (`enr_snp_b1845b8ac2`)

> Mapping Visual observation using multi-wavelength imaging

- `Visual observation` [topic] — correct


## Error Analysis


**Example 1** (`enr_snp_2cf7795415`)

> We present new observations of a obtained from ground-based telescopes. Our analysis reveals significant variability in {topic} across different epochs. These findings have implications for models of Silver and related processes.

- MISSED: `a` [topic]
- SPURIOUS: `a` [topic]


        ## Go / No-Go Recommendation

        **Recommendation: GO**

        The model achieves an overall span-level F1 of **0.9993** (macro F1 by type: 0.9995). This meets the proof-of-concept threshold of F1 >= 0.70 on synthetic data. The approach is viable for scaling to the full ADS corpus, with the following recommended next steps:

1. **Human annotation**: Annotate 500-1,000 real ADS abstracts to measure real-world performance.
2. **Entity linking**: Integrate the catalog-matching step to map extracted spans to canonical IDs.
3. **Scale training**: Increase synthetic dataset to 50K+ examples with more template diversity.
4. **Production inference**: Deploy via batch processing with SciBERT-optimized serving.

