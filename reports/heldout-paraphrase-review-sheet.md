# Held-out paraphrase review sheet

152 drafted items, none scored yet. For each line, ask one question: **are the labels on the right what you would want the search to do for the sentence on the left?** Nothing else needs checking; the mechanical checks (schema, enum values, regex-blindness of the positive stratum) already pass.

When done, run one command with the IDs you reject (edit text or labels directly in `data/datasets/benchmark/heldout_paraphrases.json` first if an item is fixable rather than wrong):

```
uv run python scripts/approve_heldout_paraphrases.py --reviewer Stephanie --reject <id> <id> ...
```

Label key: `op` = operator (absent means none); `kind` = search kind; `clarify` = needs_clarification; `specific-paper` = one identifiable work must be resolved first. Conventions: generic "papers" carries no doctype; a facility named as the data source is a bibgroup, a facility inside a topic is not.


## Operator positive (42)

The person wants a result-set transform (citations, references, similar, trending, useful, reviews) phrased so that no regex fires. Check: is the labelled operator what you would run in ADS for this sentence? Is `refers_to_specific_paper` right (true only when one identifiable work must be looked up first)?

| id | sentence | labels |
|---|---|---|
| hp-pos-001 | papers that build on the Planck 2018 cosmology results | op=citations · kind=paper_reference · specific-paper |
| hp-pos-002 | who has picked up on the Riess et al. 2019 Hubble tension measurement since it came out | op=citations · kind=paper_reference · specific-paper |
| hp-pos-003 | follow-up work to the first LIGO binary black hole detection paper | op=citations · kind=paper_reference · specific-paper |
| hp-pos-004 | downstream research stemming from the Event Horizon Telescope M87 image | op=citations · kind=paper_reference · specific-paper |
| hp-pos-005 | later studies drawing on the Salpeter initial mass function | op=citations · kind=topic |
| hp-pos-006 | how has the Kennicutt-Schmidt law paper been used since publication | op=citations · kind=paper_reference · specific-paper |
| hp-pos-007 | subsequent literature engaging with the NFW dark matter halo profile | op=citations · kind=paper_reference · specific-paper |
| hp-pos-008 | what came after the DESI 2024 baryon acoustic oscillation release, in terms of papers leaning on it | op=citations · kind=paper_reference · specific-paper |
| hp-pos-009 | everything that has since made use of the Gaia DR3 catalog paper | op=citations · kind=paper_reference · specific-paper |
| hp-pos-010 | research that takes the TRAPPIST-1 discovery as its starting point | op=citations · kind=paper_reference · specific-paper |
| hp-pos-011 | the works the Planck 2018 cosmology paper drew on | op=references · kind=paper_reference · specific-paper |
| hp-pos-012 | what did the JWST early release science team build upon | op=references · kind=topic |
| hp-pos-013 | sources underpinning the Bullet Cluster dark matter paper | op=references · kind=paper_reference · specific-paper |
| hp-pos-014 | the reading list behind the Hubble constant tension review by Di Valentino | op=references · kind=paper_reference · specific-paper |
| hp-pos-015 | the prior literature the LIGO GW150914 discovery paper leans on | op=references · kind=paper_reference · specific-paper |
| hp-pos-016 | the earlier results that the DESI BAO analysis rests on | op=references · kind=paper_reference · specific-paper |
| hp-pos-017 | what the authors of the first fast radio burst paper were relying on | op=references · kind=paper_reference · specific-paper |
| hp-pos-018 | the foundations the TRAPPIST-1 discovery paper was assembled from | op=references · kind=paper_reference · specific-paper |
| hp-pos-019 | papers in the same vein as the M87 black hole shadow imaging | op=similar · kind=paper_reference · specific-paper |
| hp-pos-020 | along the lines of Riess 2019 on the local Hubble constant | op=similar · kind=paper_reference · specific-paper |
| hp-pos-021 | kindred studies to exoplanet atmosphere retrievals with JWST transmission spectra | op=similar · kind=topic |
| hp-pos-022 | anything akin to the Bullet Cluster lensing analysis | op=similar · kind=paper_reference · specific-paper |
| hp-pos-023 | more of the same as fast radio burst dispersion measure cosmology | op=similar · kind=topic |
| hp-pos-024 | research adjacent to the TRAPPIST-1 habitability assessments | op=similar · kind=topic |
| hp-pos-025 | papers in the spirit of the NFW halo profile derivation | op=similar · kind=paper_reference · specific-paper |
| hp-pos-026 | what everybody is reading on exoplanet atmospheres right now | op=trending · kind=topic |
| hp-pos-027 | the buzz in gravitational wave astronomy this month | op=trending · kind=topic |
| hp-pos-028 | what is getting attention lately in fast radio bursts | op=trending · kind=topic |
| hp-pos-029 | the current favourites among galaxy formation papers | op=trending · kind=topic |
| hp-pos-030 | where the crowd is heading in dark energy research at the moment | op=trending · kind=topic |
| hp-pos-031 | the papers people are flocking to on JWST high-redshift galaxies | op=trending · kind=topic |
| hp-pos-032 | the bedrock literature on stellar nucleosynthesis | op=useful · kind=topic |
| hp-pos-033 | what should I read first on magnetic reconnection | op=useful · kind=topic |
| hp-pos-034 | the classics of galactic dynamics | op=useful · kind=topic |
| hp-pos-035 | the go-to papers for cosmic microwave background anisotropies | op=useful · kind=topic |
| hp-pos-036 | core reading for accretion disk theory | op=useful · kind=topic |
| hp-pos-037 | cornerstone results in exoplanet transit photometry | op=useful · kind=topic |
| hp-pos-038 | a synthesis of the fast radio burst literature | op=reviews · kind=topic |
| hp-pos-039 | an overall summary of where dark energy research stands | op=reviews · kind=topic |
| hp-pos-040 | a primer on gravitational lensing | op=reviews · kind=topic |
| hp-pos-041 | a roundup of galaxy cluster mass estimation methods | op=reviews · kind=topic |
| hp-pos-042 | big-picture treatments of star formation efficiency | op=reviews · kind=topic |

## Operator negative (40)

Operator-like words used as the topic. The label is always operator `none`. Check: would a librarian really treat this as a plain topic search? If you read it as an instruction, reject or edit it.

| id | sentence | labels |
|---|---|---|
| hp-neg-001 | citation analysis techniques in astronomy bibliometrics | kind=topic |
| hp-neg-002 | citation networks among gravitational wave papers | kind=topic |
| hp-neg-003 | self-citation rates in astrophysics journals | kind=topic |
| hp-neg-004 | citation counts as a predictor of telescope time allocation outcomes | kind=topic |
| hp-neg-005 | reference frame realization with VLBI | kind=topic |
| hp-neg-006 | the International Celestial Reference Frame | kind=topic |
| hp-neg-007 | reference stars for adaptive optics wavefront sensing | kind=topic |
| hp-neg-008 | Gaia reference frame alignment with quasars | kind=topic |
| hp-neg-009 | similarity metrics for stellar spectra classification | kind=topic |
| hp-neg-010 | self-similar solutions for supernova blast waves | kind=topic |
| hp-neg-011 | morphological similarity of spiral galaxies | kind=topic |
| hp-neg-012 | similarity-based photometric redshift estimation | kind=topic |
| hp-neg-013 | trend analysis of solar cycle sunspot numbers | kind=topic |
| hp-neg-014 | the stellar initial mass function trending toward lower masses in dense clusters | kind=topic |
| hp-neg-015 | secular trends in Earth's rotation rate | kind=topic |
| hp-neg-016 | long-term trends in exoplanet detection rates | kind=topic |
| hp-neg-017 | hot Jupiters | kind=topic |
| hp-neg-018 | hot gas in galaxy clusters | kind=topic |
| hp-neg-019 | hot subdwarf stars | kind=topic |
| hp-neg-020 | popular science communication of black hole research | kind=topic |
| hp-neg-021 | useful yield in solar cell efficiency modelling | kind=topic |
| hp-neg-022 | utility functions for observation scheduling | kind=topic |
| hp-neg-023 | helpful diagnostics for stellar age estimation | kind=topic |
| hp-neg-024 | essential amino acids found in meteorites | kind=topic |
| hp-neg-025 | key exchange protocols in quantum cryptography | kind=topic |
| hp-neg-026 | landmark detection for planetary rover navigation | kind=topic |
| hp-neg-027 | survey completeness corrections in galaxy surveys | kind=topic |
| hp-neg-028 | the Sloan Digital Sky Survey spectroscopic pipeline | kind=topic |
| hp-neg-029 | all-sky survey strategies for transient detection | kind=topic |
| hp-neg-030 | literature growth rates in astrophysics | kind=topic |
| hp-neg-031 | reviewer bias in telescope time allocation | kind=topic |
| hp-neg-032 | the peer review process in astronomy journals | kind=topic |
| hp-neg-033 | systematic errors in the distance ladder | kind=topic |
| hp-neg-034 | comprehensive models of the solar dynamo | kind=topic |
| hp-neg-035 | comparable-mass binary black hole mergers | kind=topic |
| hp-neg-036 | related-rates problems in orbital mechanics education | kind=topic |
| hp-neg-037 | seminal vesicle imaging | kind=topic |
| hp-neg-038 | foundational models for spectral classification | kind=topic |
| hp-neg-039 | must-have calibrations for CCD photometry | kind=topic |
| hp-neg-040 | introduction rates of invasive species tracked by satellite imagery | kind=topic |

## Enum synonyms (40)

Everyday wording for a property, doctype, bibgroup or collection value that the synonym maps do not know. Check: is the labelled field value the one an ADS user would want, and is it the only one?

| id | sentence | labels |
|---|---|---|
| hp-enum-001 | cosmic ray papers anyone can read without paying | property=openaccess · kind=topic |
| hp-enum-002 | cosmic microwave background papers that are not behind a paywall | property=openaccess · kind=topic |
| hp-enum-003 | publicly readable studies of the solar wind | property=openaccess · kind=topic |
| hp-enum-004 | no-subscription-needed papers on stellar winds | property=openaccess · kind=topic |
| hp-enum-005 | papers vetted by referees on pulsar timing arrays | property=refereed · kind=topic |
| hp-enum-006 | work on exoplanet atmospheres that passed referee scrutiny | property=refereed · kind=topic |
| hp-enum-007 | journal-accepted studies of galaxy mergers, nothing unrefereed | property=refereed · kind=topic |
| hp-enum-008 | e-print postings on fast radio bursts | property=eprint · kind=topic |
| hp-enum-009 | manuscripts posted before journal publication on dark matter direct detection | property=eprint · kind=topic |
| hp-enum-010 | doctoral dissertations on star formation | doctype=phdthesis · kind=topic |
| hp-enum-011 | doctorates written on galactic archaeology | doctype=phdthesis · kind=topic |
| hp-enum-012 | MSc theses on exoplanet detection | doctype=mastersthesis · kind=topic |
| hp-enum-013 | symposium contributions on galactic archaeology | doctype=inproceedings · kind=topic |
| hp-enum-014 | contributions to the IAU symposia on stellar populations | doctype=inproceedings · kind=topic |
| hp-enum-015 | textbooks on radiative transfer | doctype=book · kind=topic |
| hp-enum-016 | full-length volumes on stellar structure | doctype=book · kind=topic |
| hp-enum-017 | catalogues of variable stars | doctype=catalog · kind=topic |
| hp-enum-018 | data tables of open cluster members | doctype=catalog · kind=topic |
| hp-enum-019 | Python packages for spectral fitting | doctype=software · kind=topic |
| hp-enum-020 | analysis pipelines released as installable tools for time-series photometry | doctype=software · kind=topic |
| hp-enum-021 | technical memos on detector calibration | doctype=techreport · kind=topic |
| hp-enum-022 | observing proposals for transient follow-up | doctype=proposal · kind=topic |
| hp-enum-023 | press releases about exoplanet discoveries | doctype=pressrelease · kind=topic |
| hp-enum-024 | corrections to published results on the Hubble constant | doctype=erratum · kind=topic |
| hp-enum-025 | AAS meeting abstracts on brown dwarfs | doctype=abstract · kind=topic |
| hp-enum-026 | obituaries of radio astronomers | doctype=obituary · kind=topic |
| hp-enum-027 | GCN circulars on gamma-ray bursts | doctype=circular · kind=topic |
| hp-enum-028 | recorded seminar talks on cosmology | doctype=talk · kind=topic |
| hp-enum-029 | book reviews of cosmology textbooks | doctype=bookreview · kind=topic |
| hp-enum-030 | Green Bank Telescope pulsar papers | bibgroup=GBT · kind=topic |
| hp-enum-031 | Arecibo observations of the Crab pulsar | bibgroup=ARECIBO · kind=topic |
| hp-enum-032 | LOFAR low-frequency transient studies | bibgroup=LOFAR · kind=topic |
| hp-enum-033 | MeerKAT observations of galaxy clusters | bibgroup=MeerKAT · kind=topic |
| hp-enum-034 | Pan-STARRS supernova discoveries | bibgroup=Pan-STARRS · kind=topic |
| hp-enum-035 | NuSTAR hard X-ray studies of active galactic nuclei | bibgroup=NuSTAR · kind=topic |
| hp-enum-036 | Hipparcos parallax papers | bibgroup=Hipparcos · kind=topic |
| hp-enum-037 | geophysics papers on mantle convection | collection=earthscience · kind=topic |
| hp-enum-038 | climate science studies of aerosol forcing | collection=earthscience · kind=topic |
| hp-enum-039 | condensed matter work on topological insulators | collection=physics · kind=topic |
| hp-enum-040 | biology papers on extremophiles | collection=general · kind=topic |

## Ambiguous (30)

Underspecified requests. Check: would you ask a follow-up question before searching (`needs_clarification`)? If the item is fine to run as a search, reject it or clear the flag.

| id | sentence | labels |
|---|---|---|
| hp-amb-001 | recent stuff | kind=topic · clarify |
| hp-amb-002 | that paper from the conference | kind=paper_reference · clarify · specific-paper |
| hp-amb-003 | the usual | kind=topic · clarify |
| hp-amb-004 | papers | kind=topic · clarify |
| hp-amb-005 | find it | kind=topic · clarify |
| hp-amb-006 | Smith | kind=mixed · clarify |
| hp-amb-007 | Mercury | kind=mixed · clarify |
| hp-amb-008 | Kepler | kind=mixed · clarify |
| hp-amb-009 | the Nature paper on exoplanets | kind=paper_reference · clarify · specific-paper |
| hp-amb-010 | latest results | kind=topic · clarify |
| hp-amb-011 | something on stars, maybe galaxies | kind=topic · clarify |
| hp-amb-012 | the Hubble paper | kind=paper_reference · clarify · specific-paper |
| hp-amb-013 | what my advisor mentioned last week | kind=topic · clarify |
| hp-amb-014 | the one with the big table | kind=paper_reference · clarify · specific-paper |
| hp-amb-015 | stuff by that group in Heidelberg | kind=author · clarify |
| hp-amb-016 | cite this | op=citations · kind=paper_reference · clarify · specific-paper |
| hp-amb-017 | what does it cite | op=references · kind=paper_reference · clarify · specific-paper |
| hp-amb-018 | more like this | op=similar · kind=paper_reference · clarify · specific-paper |
| hp-amb-019 | Andromeda or the other one | kind=object · clarify |
| hp-amb-020 | the 2020 paper | kind=paper_reference · clarify · specific-paper |
| hp-amb-021 | physics | kind=topic · clarify |
| hp-amb-022 | Jupiter Saturn | kind=object · clarify |
| hp-amb-023 | things trending | op=trending · kind=topic · clarify |
| hp-amb-024 | good papers | kind=topic · clarify |
| hp-amb-025 | new | kind=topic · clarify |
| hp-amb-026 | the review | kind=paper_reference · clarify · specific-paper |
| hp-amb-027 | that big survey | kind=paper_reference · clarify · specific-paper |
| hp-amb-028 | papers by Li | kind=author · clarify |
| hp-amb-029 | Chandra | kind=mixed · clarify |
| hp-amb-030 | show me everything | kind=topic · clarify |
