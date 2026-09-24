"""Option descriptions shown to Jev for each choice question.

Keys are exactly the legal enum values from ``field_constraints`` and
``intent_spec`` plus ``none``; ``jev_intent.build_questions`` asserts the
sets match so a new enum value cannot silently go unasked.
"""

OPERATOR_DESCRIPTIONS: dict[str, str] = {
    "none": (
        "A plain search with no result-set operator. Also the answer when a citation or "
        "read count is used as a numeric filter (highly cited, more than 100 citations, "
        "most read) and when operator-like words are the subject being studied."
    ),
    "citations": (
        "The papers that cite a given paper, author or result set: forward citations, "
        "work that builds on or responds to it. Not a filter on citation counts."
    ),
    "references": (
        "The papers a given work cites: its bibliography or reference list, the sources "
        "it draws on."
    ),
    "similar": (
        "Papers textually similar to a given paper or description: related work, papers "
        "like this one."
    ),
    "trending": (
        "What readers interested in the topic are reading right now: hot, popular, "
        "trending, buzzing, currently active research."
    ),
    "useful": (
        "The foundational works most cited by the papers on the topic: seminal, key, "
        "essential, must-read, landmark, classic, important, canonical papers."
    ),
    "reviews": (
        "Review-style coverage of the topic: reviews, literature surveys, overviews, "
        "tutorials, introductions, primers, literature reviews, state-of-the-art summaries. "
        "Not an observational sky survey, which is a data source. "
        "Not a book review: a review of a book or textbook is the bookreview document "
        "type, not a review of the research literature."
    ),
}

SEARCH_KIND_DESCRIPTIONS: dict[str, str] = {
    "topic": "A subject, phenomenon or keyword search.",
    "author": "Publications by a named person or group.",
    "object": "An astronomical object by name or catalogue designation.",
    "paper_reference": "One specific paper described by title, author-year or a famous result.",
    "identifier": "A bibcode, DOI, arXiv id or other record identifier.",
    "mixed": "Two or more of the above carry equal weight.",
}

COLLECTION_DESCRIPTIONS: dict[str, str] = {
    "none": "No discipline restriction.",
    "astronomy": "Astronomy and astrophysics collection.",
    "physics": "Physics collection.",
    "general": "General science collection.",
    "earthscience": "Earth and planetary science collection.",
}

DOCTYPE_DESCRIPTIONS: dict[str, str] = {
    "none": "No document-type restriction.",
    "abstract": "Meeting abstract.",
    "article": "Journal article.",
    "book": "Book or monograph.",
    "bookreview": (
        "Book review: a critique or assessment of a particular book or textbook (book "
        "reviews, textbook reviews, reviews of books). Not a review article surveying a "
        "research topic."
    ),
    "catalog": "Data catalog or high-level data product.",
    "circular": "Printed or electronic circular (e.g. IAU, ATel-style notices).",
    "editorial": "Editorial.",
    "eprint": "Preprint (e.g. arXiv) as a document type.",
    "erratum": "Erratum or correction to a journal article.",
    "inbook": "Chapter or article appearing in a book.",
    "inproceedings": "Paper appearing in conference proceedings.",
    "mastersthesis": "Master's thesis.",
    "misc": "Anything not in the other categories.",
    "newsletter": "Printed or electronic newsletter.",
    "obituary": "Obituary.",
    "phdthesis": "PhD thesis or doctoral dissertation.",
    "pressrelease": "Press release.",
    "proceedings": "Conference proceedings volume as a whole.",
    "proposal": "Observing or funding proposal.",
    "software": "Software package or code record.",
    "talk": "Research talk at a scholarly venue.",
    "techreport": "Technical report.",
}

BIBGROUP_DESCRIPTIONS: dict[str, str] = {
    "none": "No telescope, mission or institution bibliography restriction.",
    "HST": "Hubble Space Telescope.",
    "JWST": "James Webb Space Telescope.",
    "Spitzer": "Spitzer Space Telescope.",
    "Chandra": "Chandra X-ray Observatory.",
    "XMM": "XMM-Newton X-ray observatory.",
    "GALEX": "Galaxy Evolution Explorer (ultraviolet).",
    "Kepler": "Kepler exoplanet mission.",
    "K2": "K2, the extended Kepler mission.",
    "TESS": "Transiting Exoplanet Survey Satellite.",
    "FUSE": "Far Ultraviolet Spectroscopic Explorer.",
    "IUE": "International Ultraviolet Explorer.",
    "EUVE": "Extreme Ultraviolet Explorer.",
    "Copernicus": "Copernicus (OAO-3) satellite.",
    "IRAS": "Infrared Astronomical Satellite.",
    "WISE": "Wide-field Infrared Survey Explorer.",
    "NEOWISE": "NEOWISE asteroid-hunting mission.",
    "Fermi": "Fermi Gamma-ray Space Telescope.",
    "Swift": "Neil Gehrels Swift Observatory (gamma-ray bursts).",
    "RXTE": "Rossi X-ray Timing Explorer.",
    "NuSTAR": "Nuclear Spectroscopic Telescope Array (hard X-ray).",
    "SOHO": "Solar and Heliospheric Observatory.",
    "STEREO": "Solar TErrestrial RElations Observatory.",
    "SDO": "Solar Dynamics Observatory.",
    "ESO/Telescopes": "European Southern Observatory telescopes bibliography.",
    "CFHT": "Canada-France-Hawaii Telescope.",
    "Gemini": "Gemini Observatory (North and South).",
    "Keck": "W. M. Keck Observatory.",
    "VLT": "ESO Very Large Telescope.",
    "Subaru": "Subaru Telescope.",
    "NOAO": "National Optical Astronomy Observatory.",
    "NOIRLab": "NSF NOIRLab (National Optical-Infrared Astronomy Research Laboratory).",
    "CTIO": "Cerro Tololo Inter-American Observatory.",
    "KPNO": "Kitt Peak National Observatory.",
    "Pan-STARRS": "Panoramic Survey Telescope and Rapid Response System.",
    "SDSS": "Sloan Digital Sky Survey.",
    "2MASS": "Two Micron All Sky Survey.",
    "UKIRT": "United Kingdom Infrared Telescope.",
    "ALMA": "Atacama Large Millimeter/submillimeter Array.",
    "JCMT": "James Clerk Maxwell Telescope.",
    "APEX": "Atacama Pathfinder Experiment.",
    "ARECIBO": "Arecibo Observatory radio telescope.",
    "VLA": "Very Large Array (radio).",
    "VLBA": "Very Long Baseline Array.",
    "GBT": "Green Bank Telescope.",
    "LOFAR": "Low-Frequency Array (radio).",
    "MeerKAT": "MeerKAT radio telescope.",
    "SKA": "Square Kilometre Array.",
    "Gaia": "Gaia astrometry mission.",
    "Hipparcos": "Hipparcos astrometry satellite.",
    "CfA": "Center for Astrophysics | Harvard & Smithsonian publications.",
    "NASA PubSpace": "NASA public access repository.",
    "LISA": "Laser Interferometer Space Antenna.",
    "LIGO": "Laser Interferometer Gravitational-Wave Observatory.",
    "SETI": "SETI Institute publications.",
    "ESO": "European Southern Observatory (alias of ESO/Telescopes).",
}
