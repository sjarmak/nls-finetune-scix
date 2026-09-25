# Keyword query review sheet

150 drafted keyword queries in `data/datasets/benchmark/keyword_queries.json`, 15 per stratum. For each line, ask one question: **is the gold query what the search should run for the words on the left?** Every gold query was run against ADS on 2026-09-25; `hits` is its numFound.

To reject or fix an item, edit it in the JSON (or send the list of IDs to reject). `amb` marks items where a second reading is also reasonable; the note after the table says which.

One item deliberately returns 0: `kw-st-001` ("accomazzi europa"), the motivating example. Accomazzi has no abstract mentioning Europa; `author:"Accomazzi" abs:europe` finds 7.

Conventions: surname-only authors are `"Last"`, full names `"Last, First"`; a facility typed as a filter is a bibgroup, a mission paired with a surname is the topic; objects are abs: terms; operator is none and first_author false throughout.


## Surname + topic (15)

A bare surname followed by a topic. Check: is the surname a real author with papers on that topic, and is the topic grouped the way you would quote it?

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-st-001 | accomazzi europa | Papers by Accomazzi about Europa. | `author:"Accomazzi" abs:europa` | 0 |  |
| kw-st-002 | casey dusty star forming galaxies | Papers by Casey on dusty star-forming galaxies. | `author:"Casey" abs:"dusty star forming galaxies"` | 63 |  |
| kw-st-003 | seager biosignature gases | Papers by Seager on biosignature gases. | `author:"Seager" abs:"biosignature gases"` | 50 |  |
| kw-st-004 | kurtz bibliometrics | Papers by Kurtz on bibliometrics. | `author:"Kurtz" abs:bibliometrics` | 9 |  |
| kw-st-005 | Riess type ia supernovae | Papers by Riess on Type Ia supernovae. | `author:"Riess" abs:"type ia supernovae"` | 391 |  |
| kw-st-006 | madau star formation history | Papers by Madau on the cosmic star formation history. | `author:"Madau" abs:"star formation history"` | 24 |  |
| kw-st-007 | bullock dark matter substructure | Papers by Bullock on dark matter substructure. | `author:"Bullock" abs:"dark matter substructure"` | 9 |  |
| kw-st-008 | heckman agn feedback | Papers by Heckman on AGN feedback. | `author:"Heckman" abs:"agn feedback"` | 20 |  |
| kw-st-009 | Kormendy supermassive black holes | Papers by Kormendy on supermassive black holes. | `author:"Kormendy" abs:"supermassive black holes"` | 56 |  |
| kw-st-010 | loeb first stars | Papers by Loeb on the first stars. | `author:"Loeb" abs:"first stars"` | 100 |  |
| kw-st-011 | batygin planet nine | Papers by Batygin on Planet Nine. | `author:"Batygin" abs:"planet nine"` | 27 |  |
| kw-st-012 | charbonneau transiting planets | Papers by Charbonneau on transiting planets. | `author:"Charbonneau" abs:"transiting planets"` | 188 |  |
| kw-st-013 | rigby lensed galaxies | Papers by Rigby on gravitationally lensed galaxies. | `author:"Rigby" abs:"lensed galaxies"` | 147 |  |
| kw-st-014 | Tremaine secular dynamics | Papers by Tremaine on secular dynamics. | `author:"Tremaine" abs:"secular dynamics"` | 3 |  |
| kw-st-015 | shen quasar black hole masses | Papers by Shen on quasar black hole masses. | `author:"Shen" abs:"quasar black hole masses"` | 14 |  |

Notes:

- kw-st-001: Kept as the motivating example: the intended reading returns 0 in ADS (Accomazzi has no abstract mentioning Europa; author:"Accomazzi" abs:europe finds 7, which may be what was meant).

## Surname + eponymous mission (15)

A surname followed by a mission or telescope named after a person. The mission is the topic (abs:), not a second author and not a bibgroup filter.

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-sm-001 | jarmak cassini | Papers by Jarmak about the Cassini mission or its data. | `author:"Jarmak" abs:cassini` | 6 |  |
| kw-sm-002 | spilker cassini | Papers by Spilker about the Cassini mission. | `author:"Spilker" abs:cassini` | 302 |  |
| kw-sm-003 | bolton juno | Papers by Bolton about the Juno mission. | `author:"Bolton" abs:juno` | 1094 |  |
| kw-sm-004 | stern new horizons | Papers by Stern about the New Horizons mission. | `author:"Stern" abs:"new horizons"` | 862 |  |
| kw-sm-005 | brown gaia | Papers by Brown about the Gaia mission. | `author:"Brown" abs:gaia` | 329 |  |
| kw-sm-006 | prusti gaia | Papers by Prusti about the Gaia mission. | `author:"Prusti" abs:gaia` | 117 |  |
| kw-sm-007 | Tauber planck | Papers by Tauber about the Planck mission. | `author:"Tauber" abs:planck` | 230 |  |
| kw-sm-008 | laureijs euclid | Papers by Laureijs about the Euclid mission. | `author:"Laureijs" abs:euclid` | 112 |  |
| kw-sm-009 | spergel roman | Papers by Spergel about the Roman Space Telescope. | `author:"Spergel" abs:roman` | 13 | yes |
| kw-sm-010 | rigby jwst | Papers by Rigby about JWST. | `author:"Rigby" abs:jwst` | 109 | yes |
| kw-sm-011 | Borucki kepler | Papers by Borucki about the Kepler mission. | `author:"Borucki" abs:kepler` | 304 | yes |
| kw-sm-012 | pilbratt herschel | Papers by Pilbratt about the Herschel Space Observatory. | `author:"Pilbratt" abs:herschel` | 87 | yes |
| kw-sm-013 | weisskopf chandra | Papers by Weisskopf about the Chandra X-ray Observatory. | `author:"Weisskopf" abs:chandra` | 192 | yes |
| kw-sm-014 | ricker tess | Papers by Ricker about the TESS mission. | `author:"Ricker" abs:tess` | 432 | yes |
| kw-sm-015 | glassmeier rosetta | Papers by Glassmeier about the Rosetta mission. | `author:"Glassmeier" abs:rosetta` | 161 |  |

Notes:

- kw-sm-009: Many of Spergel's Roman papers predate the name and say WFIRST; abs:roman misses them.
- kw-sm-010: bibgroup:JWST (papers using JWST data) is also reasonable; gold treats the mission as the topic, as in the rest of this stratum.
- kw-sm-011: bibgroup:Kepler is also reasonable; gold treats the mission as the topic.
- kw-sm-012: bibgroup:Herschel is also reasonable; gold treats the mission as the topic.
- kw-sm-013: bibgroup:Chandra is also reasonable; gold treats the mission as the topic.
- kw-sm-014: bibgroup:TESS is also reasonable; gold treats the mission as the topic.

## Eponymous scientific term (15)

A term named after a person (Chandrasekhar limit, Roche lobe). No author should be read out of it.

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-ep-001 | chandrasekhar limit white dwarfs | Papers on the Chandrasekhar limit for white dwarfs; Chandrasekhar is not an author here. | `abs:"chandrasekhar limit" abs:"white dwarfs"` | 530 |  |
| kw-ep-002 | einstein ring strong lensing | Papers on Einstein rings in strong gravitational lensing. | `abs:"einstein ring" abs:"strong lensing"` | 244 |  |
| kw-ep-003 | hawking radiation primordial black holes | Papers on Hawking radiation from primordial black holes. | `abs:"hawking radiation" abs:"primordial black holes"` | 321 |  |
| kw-ep-004 | roche lobe overflow | Papers on Roche lobe overflow in binaries. | `abs:"roche lobe overflow"` | 1513 |  |
| kw-ep-005 | jeans mass molecular clouds | Papers on the Jeans mass in molecular clouds. | `abs:"jeans mass" abs:"molecular clouds"` | 184 |  |
| kw-ep-006 | kozai-lidov hot jupiters | Papers on Kozai-Lidov migration of hot Jupiters. | `abs:"kozai-lidov" abs:"hot jupiters"` | 70 |  |
| kw-ep-007 | Tully-Fisher relation | Papers on the Tully-Fisher relation. | `abs:"tully-fisher relation"` | 3427 |  |
| kw-ep-008 | faber-jackson relation elliptical galaxies | Papers on the Faber-Jackson relation of elliptical galaxies. | `abs:"faber-jackson relation" abs:"elliptical galaxies"` | 66 |  |
| kw-ep-009 | bondi accretion | Papers on Bondi accretion. | `abs:"bondi accretion"` | 414 |  |
| kw-ep-010 | eddington ratio agn | Papers on Eddington ratios of AGN. | `abs:"eddington ratio" abs:agn` | 1589 |  |
| kw-ep-011 | yarkovsky effect asteroids | Papers on the Yarkovsky effect on asteroids. | `abs:"yarkovsky effect" abs:asteroids` | 646 |  |
| kw-ep-012 | Kelvin-Helmholtz instability | Papers on the Kelvin-Helmholtz instability. | `abs:"kelvin-helmholtz instability"` | 7214 |  |
| kw-ep-013 | alfven waves solar wind | Papers on Alfven waves in the solar wind. | `abs:"alfven waves" abs:"solar wind"` | 1505 |  |
| kw-ep-014 | press-schechter mass function | Papers on the Press-Schechter mass function. | `abs:"press-schechter mass function"` | 51 |  |
| kw-ep-015 | lyman alpha forest | Papers on the Lyman-alpha forest. | `abs:"lyman alpha forest"` | 15972 |  |

## Two or three surnames (15)

Co-author searches, some with a topic.

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-ms-001 | kurtz accomazzi | Papers co-authored by Kurtz and Accomazzi. | `author:"Kurtz" author:"Accomazzi"` | 188 |  |
| kw-ms-002 | riess scolnic hubble constant | Papers by Riess and Scolnic on the Hubble constant. | `author:"Riess" author:"Scolnic" abs:"hubble constant"` | 55 |  |
| kw-ms-003 | seager deming exoplanet atmospheres | Papers by Seager and Deming on exoplanet atmospheres. | `author:"Seager" author:"Deming" abs:"exoplanet atmospheres"` | 16 |  |
| kw-ms-004 | casey narayanan | Papers co-authored by Casey and Narayanan. | `author:"Casey" author:"Narayanan"` | 107 |  |
| kw-ms-005 | genzel gillessen galactic center | Papers by Genzel and Gillessen on the Galactic Center. | `author:"Genzel" author:"Gillessen" abs:"galactic center"` | 144 |  |
| kw-ms-006 | madau dickinson | Papers co-authored by Madau and Dickinson (e.g. the 2014 star formation history review). | `author:"Madau" author:"Dickinson"` | 27 |  |
| kw-ms-007 | navarro frenk white | Papers by Navarro, Frenk and White (the NFW profile papers). | `author:"Navarro" author:"Frenk" author:"White"` | 51 | yes |
| kw-ms-008 | blandford znajek | Papers by Blandford and Znajek. | `author:"Blandford" author:"Znajek"` | 1 | yes |
| kw-ms-009 | kormendy ho | Papers co-authored by Kormendy and Ho. | `author:"Kormendy" author:"Ho"` | 22 |  |
| kw-ms-010 | Springel Hernquist | Papers co-authored by Springel and Hernquist. | `author:"Springel" author:"Hernquist"` | 252 |  |
| kw-ms-011 | vogelsberger genel illustris | Papers by Vogelsberger and Genel on the Illustris simulations. | `author:"Vogelsberger" author:"Genel" abs:illustris` | 37 |  |
| kw-ms-012 | rigby gladders lensed galaxies | Papers by Rigby and Gladders on lensed galaxies. | `author:"Rigby" author:"Gladders" abs:"lensed galaxies"` | 80 |  |
| kw-ms-013 | el-badry rix wide binaries | Papers by El-Badry and Rix on wide binaries. | `author:"El-Badry" author:"Rix" abs:"wide binaries"` | 10 |  |
| kw-ms-014 | jarmak colwell saturn rings | Papers by Jarmak and Colwell on Saturn's rings. | `author:"Jarmak" author:"Colwell" abs:"saturn rings"` | 4 |  |
| kw-ms-015 | binney tremaine | Works co-authored by Binney and Tremaine (Galactic Dynamics). | `author:"Binney" author:"Tremaine"` | 8 |  |

Notes:

- kw-ms-007: Could also mean the NFW profile as a topic (abs:"navarro frenk white" or abs:nfw); gold reads three surnames.
- kw-ms-008: Could also mean the Blandford-Znajek mechanism as a topic; gold reads two surnames because the query has no hyphen.

## Full name + topic (15)

First and last name, then a topic. Gold writes 'Last, First'.

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-fn-001 | sara seager exoplanet atmospheres | Papers by Sara Seager on exoplanet atmospheres. | `author:"Seager, Sara" abs:"exoplanet atmospheres"` | 148 |  |
| kw-fn-002 | Caitlin Casey submillimeter galaxies | Papers by Caitlin Casey on submillimeter galaxies. | `author:"Casey, Caitlin" abs:"submillimeter galaxies"` | 56 |  |
| kw-fn-003 | jane rigby gravitational lensing | Papers by Jane Rigby on gravitational lensing. | `author:"Rigby, Jane" abs:"gravitational lensing"` | 146 |  |
| kw-fn-004 | adam riess cepheids | Papers by Adam Riess on Cepheids. | `author:"Riess, Adam" abs:cepheids` | 167 |  |
| kw-fn-005 | Kareem El-Badry wide binaries | Papers by Kareem El-Badry on wide binaries. | `author:"El-Badry, Kareem" abs:"wide binaries"` | 26 |  |
| kw-fn-006 | alberto accomazzi digital libraries | Papers by Alberto Accomazzi on digital libraries. | `author:"Accomazzi, Alberto" abs:"digital libraries"` | 63 |  |
| kw-fn-007 | michael kurtz bibliometrics | Papers by Michael Kurtz on bibliometrics. | `author:"Kurtz, Michael" abs:bibliometrics` | 9 |  |
| kw-fn-008 | stephanie jarmak saturn rings | Papers by Stephanie Jarmak on Saturn's rings. | `author:"Jarmak, Stephanie" abs:"saturn rings"` | 5 |  |
| kw-fn-009 | Andrea Ghez galactic center | Papers by Andrea Ghez on the Galactic Center. | `author:"Ghez, Andrea" abs:"galactic center"` | 307 |  |
| kw-fn-010 | neta bahcall galaxy clusters | Papers by Neta Bahcall on galaxy clusters. | `author:"Bahcall, Neta" abs:"galaxy clusters"` | 154 |  |
| kw-fn-011 | wendy freedman hubble constant | Papers by Wendy Freedman on the Hubble constant. | `author:"Freedman, Wendy" abs:"hubble constant"` | 148 |  |
| kw-fn-012 | heidi hammel neptune | Papers by Heidi Hammel on Neptune. | `author:"Hammel, Heidi" abs:neptune` | 190 |  |
| kw-fn-013 | Chris Lintott galaxy zoo | Papers by Chris Lintott on Galaxy Zoo. | `author:"Lintott, Chris" abs:"galaxy zoo"` | 177 |  |
| kw-fn-014 | priyamvada natarajan black hole seeds | Papers by Priyamvada Natarajan on black hole seeds. | `author:"Natarajan, Priyamvada" abs:"black hole seeds"` | 33 |  |
| kw-fn-015 | sandra faber galaxy evolution | Papers by Sandra Faber on galaxy evolution. | `author:"Faber, Sandra" abs:"galaxy evolution"` | 254 |  |

## Facility filter + topic (15)

A facility typed in front of a topic is read as its bibgroup (papers using that facility's data).

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-fa-001 | jwst brown dwarfs | Papers on brown dwarfs that use JWST data. | `abs:"brown dwarfs" bibgroup:JWST` | 143 |  |
| kw-fa-002 | hst m31 cepheids | Papers on Cepheids in M31 that use Hubble data. | `abs:m31 abs:cepheids bibgroup:HST` | 37 |  |
| kw-fa-003 | Chandra galaxy clusters | Papers on galaxy clusters that use Chandra data. | `abs:"galaxy clusters" bibgroup:Chandra` | 2643 |  |
| kw-fa-004 | ALMA protoplanetary disks | Papers on protoplanetary disks that use ALMA data. | `abs:"protoplanetary disks" bibgroup:ALMA` | 904 |  |
| kw-fa-005 | tess hot jupiters | Papers on hot Jupiters that use TESS data. | `abs:"hot jupiters" bibgroup:TESS` | 133 |  |
| kw-fa-006 | kepler eclipsing binaries | Papers on eclipsing binaries that use Kepler data. | `abs:"eclipsing binaries" bibgroup:Kepler` | 286 |  |
| kw-fa-007 | spitzer phase curves | Papers on exoplanet phase curves that use Spitzer data. | `abs:"phase curves" bibgroup:Spitzer` | 122 |  |
| kw-fa-008 | herschel cold dust | Papers on cold dust that use Herschel data. | `abs:"cold dust" bibgroup:Herschel` | 208 |  |
| kw-fa-009 | xmm-newton ultraluminous x-ray sources | Papers on ultraluminous X-ray sources that use XMM-Newton data; Newton is not an author. | `abs:"ultraluminous x-ray sources" bibgroup:XMM` | 534 |  |
| kw-fa-010 | keck quasar absorption lines | Papers on quasar absorption lines that use Keck data. | `abs:"quasar absorption lines" bibgroup:Keck` | 529 |  |
| kw-fa-011 | gemini direct imaging exoplanets | Papers on directly imaged exoplanets that use Gemini data. | `abs:"direct imaging" abs:exoplanets bibgroup:Gemini` | 78 |  |
| kw-fa-012 | swift gamma-ray bursts | Papers on gamma-ray bursts that use Swift data. | `abs:"gamma-ray bursts" bibgroup:Swift` | 818 |  |
| kw-fa-013 | galex star formation | Papers on star formation that use GALEX data. | `abs:"star formation" bibgroup:GALEX` | 1707 |  |
| kw-fa-014 | sdo solar flares | Papers on solar flares that use SDO data. | `abs:"solar flares" bibgroup:"Solar Dynamics Observatory"` | 3007 |  |
| kw-fa-015 | soho coronal mass ejections | Papers on coronal mass ejections that use SOHO data. | `abs:"coronal mass ejections" bibgroup:SOHO` | 2983 |  |

## Object + topic (15)

An astronomical object and a topic. Objects are gold abs: terms: the ADS search API rejects object: ('undefined field object').

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-ob-001 | M87 black hole shadow | Papers on the black hole shadow of M87. | `abs:m87 abs:"black hole shadow"` | 335 |  |
| kw-ob-002 | NGC 1275 filaments | Papers on the filaments around NGC 1275. | `abs:"ngc 1275" abs:filaments` | 129 |  |
| kw-ob-003 | trappist-1 atmospheres | Papers on the atmospheres of the TRAPPIST-1 planets. | `abs:"trappist-1" abs:atmospheres` | 614 |  |
| kw-ob-004 | sgr a* flares | Papers on flares from Sgr A*. | `abs:"sgr a\*" abs:flares` | 1372 |  |
| kw-ob-005 | m31 globular clusters | Papers on globular clusters in M31. | `abs:m31 abs:"globular clusters"` | 1501 |  |
| kw-ob-006 | crab nebula gamma rays | Papers on gamma rays from the Crab Nebula. | `abs:"crab nebula" abs:"gamma rays"` | 3247 |  |
| kw-ob-007 | betelgeuse dimming | Papers on the dimming of Betelgeuse. | `abs:betelgeuse abs:dimming` | 121 |  |
| kw-ob-008 | proxima centauri flares | Papers on flares from Proxima Centauri. | `abs:"proxima centauri" abs:flares` | 205 |  |
| kw-ob-009 | SN 1987A neutrinos | Papers on neutrinos from SN 1987A. | `abs:"sn 1987a" abs:neutrinos` | 588 |  |
| kw-ob-010 | eta carinae great eruption | Papers on the Great Eruption of Eta Carinae. | `abs:"eta carinae" abs:"great eruption"` | 160 |  |
| kw-ob-011 | 3c 273 jet | Papers on the jet of 3C 273. | `abs:"3c 273" abs:jet` | 683 |  |
| kw-ob-012 | omega centauri multiple populations | Papers on multiple stellar populations in Omega Centauri. | `abs:"omega centauri" abs:"multiple populations"` | 56 |  |
| kw-ob-013 | lmc cepheids | Papers on Cepheids in the Large Magellanic Cloud. | `abs:lmc abs:cepheids` | 1289 |  |
| kw-ob-014 | vega debris disk | Papers on the Vega debris disk. | `abs:vega abs:"debris disk"` | 215 |  |
| kw-ob-015 | GW170817 kilonova | Papers on the kilonova from GW170817. | `abs:gw170817 abs:kilonova` | 539 |  |

## Topic + year or range (15)

Years and ranges, a few with an author or facility.

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-ty-001 | exoplanet atmospheres 2020 | Papers on exoplanet atmospheres published in 2020. | `abs:"exoplanet atmospheres" pubdate:[2020 TO 2020]` | 519 |  |
| kw-ty-002 | fast radio bursts 2018-2022 | Papers on fast radio bursts published 2018 to 2022. | `abs:"fast radio bursts" pubdate:[2018 TO 2022]` | 2120 |  |
| kw-ty-003 | gravitational waves 2016 | Papers on gravitational waves published in 2016. | `abs:"gravitational waves" pubdate:[2016 TO 2016]` | 2654 |  |
| kw-ty-004 | kilonovae 2017-2019 | Papers on kilonovae published 2017 to 2019. | `abs:kilonovae pubdate:[2017 TO 2019]` | 172 |  |
| kw-ty-005 | weak lensing 2019 | Papers on weak lensing published in 2019. | `abs:"weak lensing" pubdate:[2019 TO 2019]` | 375 |  |
| kw-ty-006 | tidal disruption events 2015-2020 | Papers on tidal disruption events published 2015 to 2020. | `abs:"tidal disruption events" pubdate:[2015 TO 2020]` | 874 |  |
| kw-ty-007 | hot jupiters 2010 | Papers on hot Jupiters published in 2010. | `abs:"hot jupiters" pubdate:[2010 TO 2010]` | 193 |  |
| kw-ty-008 | magnetars 2020-2023 | Papers on magnetars published 2020 to 2023. | `abs:magnetars pubdate:[2020 TO 2023]` | 1170 |  |
| kw-ty-009 | exomoons since 2018 | Papers on exomoons published from 2018 on. | `abs:exomoons pubdate:[2018 TO *]` | 315 |  |
| kw-ty-010 | cosmic rays before 2000 | Papers on cosmic rays published before 2000. | `abs:"cosmic rays" pubdate:[* TO 1999]` | 34490 | yes |
| kw-ty-011 | cmb anisotropies 1990s | Papers on CMB anisotropies published in the 1990s. | `abs:"cmb anisotropies" pubdate:[1990 TO 1999]` | 432 |  |
| kw-ty-012 | riess 2016 hubble constant | Papers by Riess on the Hubble constant published in 2016. | `author:"Riess" abs:"hubble constant" pubdate:[2016 TO 2016]` | 5 |  |
| kw-ty-013 | seager 2013 biosignatures | Papers by Seager on biosignatures published in 2013. | `author:"Seager" abs:biosignatures pubdate:[2013 TO 2013]` | 1 |  |
| kw-ty-014 | 1998 supernova cosmology | Papers on supernova cosmology published in 1998. | `abs:"supernova cosmology" pubdate:[1998 TO 1998]` | 25 |  |
| kw-ty-015 | JWST 2023 high redshift galaxies | Papers on high-redshift galaxies from JWST data published in 2023. | `abs:"high redshift galaxies" pubdate:[2023 TO 2023] bibgroup:JWST` | 111 |  |

Notes:

- kw-ty-010: 'before 2000' read as ending 1999; ending 2000 is also defensible.

## Pure multi-word topic (15)

No names at all. Check the phrase grouping.

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-pt-001 | dark matter halo profiles | Papers on density profiles of dark matter halos. | `abs:"dark matter halo profiles"` | 497 |  |
| kw-pt-002 | tidal disruption events x-ray | Papers on the X-ray emission of tidal disruption events. | `abs:"tidal disruption events" abs:"x-ray"` | 1243 |  |
| kw-pt-003 | hot jupiter inflated radii | Papers on the inflated radii of hot Jupiters. | `abs:"hot jupiter" abs:"inflated radii"` | 108 |  |
| kw-pt-004 | cosmic ray acceleration supernova remnants | Papers on cosmic-ray acceleration in supernova remnants. | `abs:"cosmic ray acceleration" abs:"supernova remnants"` | 824 |  |
| kw-pt-005 | black hole spin x-ray reflection | Papers measuring black hole spin with X-ray reflection. | `abs:"black hole spin" abs:"x-ray reflection"` | 139 |  |
| kw-pt-006 | galaxy quenching environment | Papers on environmental quenching of galaxies. | `abs:"galaxy quenching" abs:environment` | 359 |  |
| kw-pt-007 | stellar streams milky way halo | Papers on stellar streams in the Milky Way halo. | `abs:"stellar streams" abs:"milky way halo"` | 316 |  |
| kw-pt-008 | primordial black holes dark matter | Papers on primordial black holes as dark matter. | `abs:"primordial black holes" abs:"dark matter"` | 1972 |  |
| kw-pt-009 | solar wind turbulence | Papers on turbulence in the solar wind. | `abs:"solar wind turbulence"` | 1866 |  |
| kw-pt-010 | magnetic reconnection solar flares | Papers on magnetic reconnection in solar flares. | `abs:"magnetic reconnection" abs:"solar flares"` | 3399 |  |
| kw-pt-011 | baryon acoustic oscillations | Papers on baryon acoustic oscillations. | `abs:"baryon acoustic oscillations"` | 6828 |  |
| kw-pt-012 | radius valley super earths | Papers on the radius valley of super-Earths. | `abs:"radius valley" abs:"super earths"` | 133 |  |
| kw-pt-013 | agn feedback cooling flows | Papers on AGN feedback in cooling flows. | `abs:"agn feedback" abs:"cooling flows"` | 238 |  |
| kw-pt-014 | ice giants interior structure | Papers on the interior structure of the ice giants. | `abs:"ice giants" abs:"interior structure"` | 77 |  |
| kw-pt-015 | planet formation pebble accretion | Papers on pebble accretion in planet formation. | `abs:"planet formation" abs:"pebble accretion"` | 363 |  |

## Tricky (15)

Surnames that are also common words or missions (Webb, Newton, White, Brown, Ho), 'hubble' as person vs constant vs telescope, hyphenated and two-word surnames, acronyms.

| id | query | intended meaning | gold query | hits | amb |
|---|---|---|---|---|---|
| kw-tr-001 | hubble constant tension | Papers on the Hubble tension; Hubble is neither an author nor the HST bibgroup. | `abs:"hubble constant" abs:tension` | 1318 | yes |
| kw-tr-002 | hubble deep field | Papers on the Hubble Deep Field. | `abs:"hubble deep field"` | 3845 | yes |
| kw-tr-003 | hubble 1929 | Edwin Hubble's 1929 papers (the velocity-distance relation). | `author:"Hubble" pubdate:[1929 TO 1929]` | 5 |  |
| kw-tr-004 | webb galaxy clusters | Papers by an astronomer named Webb (e.g. Tracy Webb) on galaxy clusters. | `author:"Webb" abs:"galaxy clusters"` | 89 | yes |
| kw-tr-005 | newton m dwarfs rotation | Papers by Newton (e.g. Elisabeth Newton) on M dwarf rotation. | `author:"Newton" abs:"m dwarfs" abs:rotation` | 55 |  |
| kw-tr-006 | planck cmb lensing | Papers on CMB lensing from the Planck mission; Planck is not an author. | `abs:planck abs:"cmb lensing"` | 683 |  |
| kw-tr-007 | el-badry wide binaries | Papers by El-Badry on wide binaries (hyphenated surname). | `author:"El-Badry" abs:"wide binaries"` | 26 |  |
| kw-tr-008 | bullock boylan-kolchin too big to fail | Papers by Bullock and Boylan-Kolchin on the too-big-to-fail problem. | `author:"Bullock" author:"Boylan-Kolchin" abs:"too big to fail"` | 12 |  |
| kw-tr-009 | white galaxy formation | Papers by White (e.g. Simon White) on galaxy formation. | `author:"White" abs:"galaxy formation"` | 314 | yes |
| kw-tr-010 | ho low luminosity agn | Papers by Ho (e.g. Luis Ho) on low-luminosity AGN. | `author:"Ho" abs:"low luminosity agn"` | 80 |  |
| kw-tr-011 | vera rubin rotation curves | Papers by Vera Rubin on galaxy rotation curves, not the Rubin Observatory. | `author:"Rubin, Vera" abs:"rotation curves"` | 60 |  |
| kw-tr-012 | brown kuiper belt | Papers by Brown (e.g. Mike Brown) on the Kuiper belt. | `author:"Brown" abs:"kuiper belt"` | 280 |  |
| kw-tr-013 | jocelyn bell burnell pulsars | Papers by Jocelyn Bell Burnell on pulsars (two-word surname). | `author:"Bell Burnell, Jocelyn" abs:pulsars` | 9 |  |
| kw-tr-014 | chandra deep field south | Papers on the Chandra Deep Field South survey field. | `abs:"chandra deep field south"` | 2332 | yes |
| kw-tr-015 | frb host galaxies | Papers on the host galaxies of fast radio bursts (FRB acronym). | `abs:frb abs:"host galaxies"` | 858 |  |

Notes:

- kw-tr-001: Grouping as one phrase "hubble constant tension" is too narrow; "hubble tension" is the field's name for it. Gold keeps the user's words split into constant + tension.
- kw-tr-002: bibgroup:HST is defensible (the HDF is HST data); gold keeps the name as a topic phrase.
- kw-tr-004: Could mean JWST observations of galaxy clusters (bibgroup:JWST or abs:webb); gold reads a surname because 'webb' alone is rarely used for the telescope.
- kw-tr-009: 'white' is also a common word; a pure topic reading is possible but unlikely.
- kw-tr-014: bibgroup:Chandra is defensible; gold keeps the field name as a topic phrase.
