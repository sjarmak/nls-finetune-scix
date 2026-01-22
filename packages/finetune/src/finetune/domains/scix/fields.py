"""ADS/SciX search field definitions.

Reference: https://ui.adsabs.harvard.edu/help/search/search-syntax
"""

# All searchable ADS fields
ADS_FIELDS = {
    # Core metadata
    "abs": "Abstract, title, and keywords (virtual field)",
    "abstract": "Abstract only",
    "title": "Title only",
    "keyword": "Publisher or author-supplied keywords",
    "full": "Fulltext, acknowledgements, abstract, title, keywords",
    "body": "Full text only (minus acknowledgements)",
    "ack": "Acknowledgements section",
    # Authors
    "author": "Author name (Last, First or Last, F)",
    "^author": "First author only (prefix with ^)",
    "author_count": "Number of authors",
    "orcid": "ORCID identifier",
    "orcid_pub": "ORCID from publishers",
    "orcid_user": "ORCID claimed by ADS users",
    # Affiliations
    "aff": "Raw affiliation string",
    "aff_id": "Canonical affiliation ID",
    "inst": "Curated institution abbreviation",
    # Publication info
    "bibcode": "ADS bibcode identifier",
    "bibstem": "Journal abbreviation (e.g., ApJ, MNRAS)",
    "bibgroup": "Bibliographic group (e.g., HST)",
    "pub": "Publication name",
    "pubdate": "Publication date (YYYY-MM or range)",
    "year": "Publication year",
    "volume": "Volume number",
    "issue": "Issue number",
    "page": "Page number",
    "doctype": "Document type (article, eprint, etc.)",
    # Identifiers
    "doi": "Digital Object Identifier",
    "arXiv": "arXiv identifier",
    "arxiv_class": "arXiv classification",
    "identifier": "Any identifier (bibcode, doi, arXiv)",
    "alternate_bibcode": "Alternate bibcode",
    # Citations and metrics
    "citation_count": "Number of citations",
    "read_count": "Number of reads",
    "cite_read_boost": "Citation/read boost factor",
    "classic_factor": "Classic popularity factor",
    # Data and links
    "data": "Data source links",
    "property": "Record properties (e.g., refereed, openaccess)",
    "esources": "Electronic source types",
    # Astronomy-specific
    "object": "Astronomical object name or coordinates",
    # Other
    "database": "Database (astronomy, physics, general)",
    "lang": "Language of the paper",
    "copyright": "Copyright information",
    "grant": "Grant information",
    "entdate": "Entry date in ADS",
    "editor": "Editor name (for books)",
    "book_author": "Book author",
    "caption": "Figure/table captions",
    "comment": "Comments field",
    "alternate_title": "Alternate title (translations)",
}

# Fields organized by category for training data generation
FIELD_CATEGORIES = {
    "author": ["author", "^author", "author_count", "orcid"],
    "content": ["abs", "abstract", "title", "keyword", "full", "body"],
    "publication": ["bibstem", "pubdate", "year", "volume", "issue", "page", "pub", "doctype"],
    "identifiers": ["bibcode", "doi", "arXiv", "identifier"],
    "metrics": ["citation_count", "read_count"],
    "affiliation": ["aff", "aff_id", "inst"],
    "astronomy": ["object", "arxiv_class"],
    "properties": ["property", "database", "data"],
}

# Common bibstems for synthetic data generation
COMMON_BIBSTEMS = [
    "ApJ",  # Astrophysical Journal
    "MNRAS",  # Monthly Notices of the Royal Astronomical Society
    "A&A",  # Astronomy & Astrophysics
    "AJ",  # Astronomical Journal
    "Nature",  # Nature
    "Science",  # Science
    "PhRvL",  # Physical Review Letters
    "PhRvD",  # Physical Review D
    "ApJL",  # Astrophysical Journal Letters
    "ApJS",  # Astrophysical Journal Supplement
    "PASP",  # Publications of the Astronomical Society of the Pacific
    "ARA&A",  # Annual Review of Astronomy and Astrophysics
    "Icar",  # Icarus
    "P&SS",  # Planetary and Space Science
    "JGRE",  # Journal of Geophysical Research: Planets
]

# Common astronomical objects for synthetic data
COMMON_OBJECTS = [
    "M31",  # Andromeda Galaxy
    "M87",  # Virgo A
    "Sgr A*",  # Sagittarius A*
    "Crab Nebula",
    "Orion Nebula",
    "LMC",  # Large Magellanic Cloud
    "SMC",  # Small Magellanic Cloud
    "NGC 1234",  # Example NGC object
    "HD 209458",  # Famous exoplanet host
    "TRAPPIST-1",
    "Proxima Centauri",
    "Alpha Centauri",
    "Betelgeuse",
    "Vega",
]

# Document types
DOC_TYPES = [
    "article",
    "eprint",
    "inproceedings",
    "book",
    "bookreview",
    "catalog",
    "circular",
    "editorial",
    "erratum",
    "inbook",
    "mastersthesis",
    "misc",
    "newsletter",
    "obituary",
    "phdthesis",
    "pressrelease",
    "proceedings",
    "proposal",
    "software",
    "talk",
    "techreport",
]

# Properties for filtering
PROPERTIES = [
    "refereed",
    "notrefereed",
    "article",
    "openaccess",
    "eprint",
    "data",
    "software",
    "citation",
    "reference",
    "toc",  # Table of contents
    "catalog",
    "associated",
]
