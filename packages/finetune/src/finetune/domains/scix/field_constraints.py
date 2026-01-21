"""ADS/SciX field enumeration constraints for query validation.

This module contains enumerated valid values for ADS fields that have
constrained vocabularies. These are used to validate model-generated
queries at inference time to catch invalid field combinations.

References:
    - ADS Search Syntax: https://ui.adsabs.harvard.edu/help/search/search-syntax
    - ADS Solr Schema: https://github.com/adsabs/montysolr/blob/main/deploy/adsabs/server/solr/collection1/conf/schema.xml
    - ADS Bibgroups: https://ui.adsabs.harvard.edu/help/data_faq/Bibgroups
"""

# Document types indexed by ADS (extension of BibTeX entry types)
# Reference: https://ui.adsabs.harvard.edu/help/search/search-syntax#document-type
DOCTYPES: frozenset[str] = frozenset({
    "abstract",       # Meeting abstract
    "article",        # Journal article
    "book",           # Book (monograph)
    "bookreview",     # Published book review
    "catalog",        # Data catalog or high-level data product
    "circular",       # Printed or electronic circular
    "editorial",      # Editorial
    "eprint",         # Preprinted article (e.g., arXiv)
    "erratum",        # Erratum to a journal article
    "inbook",         # Article appearing in a book
    "inproceedings",  # Article appearing in conference proceedings
    "mastersthesis",  # Masters thesis
    "misc",           # Anything not in above categories
    "newsletter",     # Printed or electronic newsletter
    "obituary",       # Obituary (article with "obituary" in title)
    "phdthesis",      # PhD thesis
    "pressrelease",   # Press release
    "proceedings",    # Conference proceedings book
    "proposal",       # Observing or funding proposal
    "software",       # Software package
    "talk",           # Research talk at scholarly venue
    "techreport",     # Technical report
})

# Properties for filtering records
# Reference: https://ui.adsabs.harvard.edu/help/search/search-syntax#properties
PROPERTIES: frozenset[str] = frozenset({
    # Open access variants
    "ads_openaccess",      # OA version available from ADS
    "author_openaccess",   # Author-submitted OA version available
    "eprint_openaccess",   # OA version from preprint server (arXiv)
    "pub_openaccess",      # OA version from publisher
    "openaccess",          # At least one OA version available
    # Record type properties
    "article",             # Regular article
    "nonarticle",          # Not a regular article (e.g., meeting abstracts)
    "refereed",            # Peer reviewed
    "notrefereed",         # Not peer reviewed
    "eprint",              # The record is an eprint/preprint
    "inproceedings",       # Conference proceeding
    "software",            # Software record
    "catalog",             # Data catalog record
    # Associated resources
    "associated",          # Has associated articles
    "data",                # Has data links
    "esource",             # Electronic source available
    "inspire",             # Record in INSPIRE database
    "library_catalog",     # Record in library catalog
    "presentation",        # Has media presentations
    "toc",                 # Has table of contents
    # OCR status
    "ocr_abstract",        # Abstract generated from OCR
})

# Database collections in ADS
# Reference: https://ui.adsabs.harvard.edu/help/search/search-syntax
DATABASES: frozenset[str] = frozenset({
    "astronomy",   # Astronomy and astrophysics collection
    "physics",     # Physics collection
    "general",     # General science collection
})

# Bibliographic groups curated by institutions/observatories
# Reference: https://ui.adsabs.harvard.edu/help/data_faq/Bibgroups
# These are hand-curated by librarians and contain publications
# using data from specific telescopes/institutions
BIBGROUPS: frozenset[str] = frozenset({
    # Space telescopes
    "HST",              # Hubble Space Telescope
    "JWST",             # James Webb Space Telescope
    "Spitzer",          # Spitzer Space Telescope
    "Chandra",          # Chandra X-ray Observatory
    "XMM",              # XMM-Newton
    "GALEX",            # Galaxy Evolution Explorer
    "Kepler",           # Kepler mission
    "K2",               # K2 mission
    "TESS",             # Transiting Exoplanet Survey Satellite
    "FUSE",             # Far Ultraviolet Spectroscopic Explorer
    "IUE",              # International Ultraviolet Explorer
    "EUVE",             # Extreme Ultraviolet Explorer
    "Copernicus",       # Copernicus satellite
    "IRAS",             # Infrared Astronomical Satellite
    "WISE",             # Wide-field Infrared Survey Explorer
    "NEOWISE",          # NEOWISE mission
    "Fermi",            # Fermi Gamma-ray Space Telescope
    "Swift",            # Swift Observatory
    "RXTE",             # Rossi X-ray Timing Explorer
    "NuSTAR",           # Nuclear Spectroscopic Telescope Array
    # Solar missions
    "SOHO",             # Solar and Heliospheric Observatory
    "STEREO",           # Solar TErrestrial RElations Observatory
    "SDO",              # Solar Dynamics Observatory
    # Ground-based observatories
    "ESO/Telescopes",   # European Southern Observatory
    "CFHT",             # Canada-France-Hawaii Telescope
    "Gemini",           # Gemini Observatory
    "Keck",             # W.M. Keck Observatory
    "VLT",              # Very Large Telescope
    "Subaru",           # Subaru Telescope
    "NOAO",             # National Optical Astronomy Observatory
    "NOIRLab",          # NSF's National Optical-Infrared Astronomy Research Laboratory
    "CTIO",             # Cerro Tololo Inter-American Observatory
    "KPNO",             # Kitt Peak National Observatory
    "Pan-STARRS",       # Panoramic Survey Telescope & Rapid Response System
    "SDSS",             # Sloan Digital Sky Survey
    "2MASS",            # Two Micron All Sky Survey
    "UKIRT",            # United Kingdom Infrared Telescope
    "ALMA",             # Atacama Large Millimeter Array
    "JCMT",             # James Clerk Maxwell Telescope
    "APEX",             # Atacama Pathfinder Experiment
    "ARECIBO",          # Arecibo Observatory
    "VLA",              # Very Large Array
    "VLBA",             # Very Long Baseline Array
    "GBT",              # Green Bank Telescope
    "LOFAR",            # Low-Frequency Array
    "MeerKAT",          # MeerKAT radio telescope
    "SKA",              # Square Kilometre Array
    # Astrometry
    "Gaia",             # Gaia mission
    "Hipparcos",        # Hipparcos satellite
    # Other
    "CfA",              # Center for Astrophysics publications
    "NASA PubSpace",    # NASA public access repository
    "LISA",             # Laser Interferometer Space Antenna
    "LIGO",             # Laser Interferometer Gravitational-Wave Observatory
})

# Electronic source types (esources field)
# These indicate what type of electronic access is available
ESOURCES: frozenset[str] = frozenset({
    "PUB_PDF",          # PDF from publisher
    "PUB_HTML",         # HTML from publisher
    "EPRINT_PDF",       # PDF from preprint server (arXiv)
    "EPRINT_HTML",      # HTML from preprint server
    "AUTHOR_PDF",       # PDF from author
    "AUTHOR_HTML",      # HTML from author
    "ADS_PDF",          # PDF available from ADS
    "ADS_SCAN",         # Scanned article from ADS
})

# Data archive sources (data field)
# Archives that ADS links to for observational data
DATA_SOURCES: frozenset[str] = frozenset({
    "ARI",              # Astronomisches Rechen-Institut
    "BICEP2",           # BICEP2 data
    "Chandra",          # Chandra data archive
    "CXO",              # Chandra X-ray Observatory
    "ESA",              # European Space Agency
    "ESO",              # European Southern Observatory
    "GCPD",             # General Catalogue of Photometric Data
    "GTC",              # Gran Telescopio Canarias
    "HEASARC",          # High Energy Astrophysics Science Archive
    "Herschel",         # Herschel Space Observatory
    "INES",             # IUE Newly Extracted Spectra
    "IRSA",             # Infrared Science Archive
    "ISO",              # Infrared Space Observatory
    "KOA",              # Keck Observatory Archive
    "MAST",             # Mikulski Archive for Space Telescopes
    "NED",              # NASA/IPAC Extragalactic Database
    "NExScI",           # NASA Exoplanet Science Institute
    "NOAO",             # National Optical Astronomy Observatory
    "PDS",              # Planetary Data System
    "SIMBAD",           # SIMBAD database
    "Spitzer",          # Spitzer Heritage Archive
    "TNS",              # Transient Name Server
    "VizieR",           # VizieR catalog service
    "XMM",              # XMM-Newton Science Archive
})

# Combined dict for easier validation lookup
FIELD_ENUMS = {
    "doctype": DOCTYPES,
    "database": DATABASES,
    "property": PROPERTIES,
    "bibgroup": BIBGROUPS,
    "esources": ESOURCES,
    "data": DATA_SOURCES,
}


def get_valid_values(field: str) -> frozenset[str] | None:
    """Get the set of valid values for a constrained field.

    Args:
        field: The ADS field name (e.g., 'doctype', 'property')

    Returns:
        FrozenSet of valid values, or None if field has no constraints
    """
    return FIELD_ENUMS.get(field.lower())


def is_valid_value(field: str, value: str) -> bool:
    """Check if a value is valid for a given constrained field.

    Args:
        field: The ADS field name (e.g., 'doctype', 'property')
        value: The value to check

    Returns:
        True if valid, False if invalid or field has no constraints
    """
    valid_values = get_valid_values(field)
    if valid_values is None:
        return True  # No constraints for this field
    return value.lower() in {v.lower() for v in valid_values}


def suggest_correction(field: str, invalid_value: str) -> list[str]:
    """Suggest possible corrections for an invalid field value.

    Uses simple string matching to find similar valid values.

    Args:
        field: The ADS field name
        invalid_value: The invalid value to find suggestions for

    Returns:
        List of up to 3 suggested valid values, sorted by similarity
    """
    valid_values = get_valid_values(field)
    if valid_values is None:
        return []

    invalid_lower = invalid_value.lower()
    suggestions = []

    for valid in valid_values:
        valid_lower = valid.lower()
        # Exact prefix match
        if valid_lower.startswith(invalid_lower) or invalid_lower.startswith(valid_lower):
            suggestions.append((0, valid))
        # Substring match
        elif invalid_lower in valid_lower or valid_lower in invalid_lower:
            suggestions.append((1, valid))
        # Common prefix
        elif len(invalid_lower) > 2 and len(valid_lower) > 2:
            common = 0
            for a, b in zip(invalid_lower, valid_lower):
                if a == b:
                    common += 1
                else:
                    break
            if common >= 3:
                suggestions.append((2, valid))

    # Sort by match quality and return top 3
    suggestions.sort(key=lambda x: x[0])
    return [s[1] for s in suggestions[:3]]
