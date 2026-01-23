"""Rules-based NER extraction for natural language to IntentSpec conversion.

This module extracts structured intent from natural language search queries
using rule-based patterns. It implements strict operator gating to prevent
the conflation of natural language words with ADS operator syntax.

CRITICAL: Operators are ONLY set when explicit patterns match.
Words like "citing", "references", "similar" as topics do NOT trigger operators.
"""

import re
from datetime import datetime

from .field_constraints import BIBGROUPS, COLLECTIONS, DOCTYPES, PROPERTIES
from .intent_spec import IntentSpec
from .pipeline import is_ads_query  # noqa: E402

# =============================================================================
# SYNONYM MAPS
# =============================================================================
# These map user-friendly terms to valid FIELD_ENUM values

PROPERTY_SYNONYMS: dict[str, str] = {
    # Peer review
    "refereed": "refereed",
    "peer reviewed": "refereed",
    "peer-reviewed": "refereed",
    "reviewed": "refereed",
    # Open access
    "open access": "openaccess",
    "open-access": "openaccess",
    "oa": "openaccess",
    "free": "openaccess",
    # Preprints
    "arxiv": "eprint",
    "preprint": "eprint",
    "preprints": "eprint",
    "eprint": "eprint",
}

DOCTYPE_SYNONYMS: dict[str, str] = {
    "article": "article",
    "articles": "article",
    "journal article": "article",
    "journal articles": "article",
    "paper": "article",
    "papers": "article",
    "publication": "article",
    "publications": "article",
    # Thesis types
    "thesis": "phdthesis",
    "phd": "phdthesis",
    "phd thesis": "phdthesis",
    "dissertation": "phdthesis",
    "masters thesis": "mastersthesis",
    "masters": "mastersthesis",
    "master's thesis": "mastersthesis",
    # Preprints
    "preprint": "eprint",
    "preprints": "eprint",
    "arxiv": "eprint",
    # Conference
    "conference": "inproceedings",
    "conference paper": "inproceedings",
    "conference papers": "inproceedings",
    "proceedings": "inproceedings",
    # Software
    "software": "software",
    "code": "software",
    # Books
    "book": "book",
    "books": "book",
    "monograph": "book",
    # Reviews
    "review": "article",  # NOT reviews operator - just article type
    "review article": "article",
    "review articles": "article",
}

BIBGROUP_SYNONYMS: dict[str, str] = {
    # Space telescopes - human-friendly names to codes
    "hubble": "HST",
    "hubble space telescope": "HST",
    "hst": "HST",
    "webb": "JWST",
    "james webb": "JWST",
    "james webb space telescope": "JWST",
    "jwst": "JWST",
    "spitzer": "Spitzer",
    "spitzer space telescope": "Spitzer",
    "chandra": "Chandra",
    "chandra x-ray": "Chandra",
    "kepler": "Kepler",
    "kepler mission": "Kepler",
    "tess": "TESS",
    "fermi": "Fermi",
    "fermi gamma": "Fermi",
    "gaia": "Gaia",
    "xmm": "XMM",
    "xmm-newton": "XMM",
    # Ground-based
    "sloan": "SDSS",
    "sdss": "SDSS",
    "sloan digital sky survey": "SDSS",
    "alma": "ALMA",
    "vlt": "VLT",
    "very large telescope": "VLT",
    "keck": "Keck",
    "gemini": "Gemini",
    "subaru": "Subaru",
    # Gravitational waves
    "ligo": "LIGO",
    "gravitational wave": "LIGO",
    "gravitational waves": "LIGO",
}

COLLECTION_SYNONYMS: dict[str, str] = {
    "astronomy": "astronomy",
    "astro": "astronomy",
    "astrophysics": "astronomy",
    "physics": "physics",
    "general": "general",
    # Earth science / planetary science
    "earthscience": "earthscience",
    "earth science": "earthscience",
    "earth sciences": "earthscience",
    "geoscience": "earthscience",
    "geosciences": "earthscience",
    "planetary science": "earthscience",
    "planetary sciences": "earthscience",
    "heliophysics": "earthscience",
    "space weather": "earthscience",
}


# =============================================================================
# OPERATOR GATING PATTERNS
# =============================================================================
# CRITICAL: These patterns MUST be explicit and specific.
# Do NOT trigger operators for generic use of these words as topics.

# NOTE: The order of operators in this dict matters! More specific patterns should be
# checked first. Currently, "references" patterns that could conflict with "citations"
# patterns (like "sources cited by" vs "cited by") are handled via pattern specificity.
OPERATOR_PATTERNS: dict[str, list[re.Pattern]] = {
    # IMPORTANT: "references" comes FIRST to handle "X cited by" patterns correctly
    # where X qualifies the pattern (e.g., "sources cited by", "works cited by")
    "references": [
        # Specific "X cited by" patterns (must match before generic "cited by")
        re.compile(r"\bsources?\s+cited\s+by\b", re.IGNORECASE),
        re.compile(r"\bworks?\s+cited\s+by\b", re.IGNORECASE),
        # Existing patterns
        re.compile(r"\breferences?\s+of\b", re.IGNORECASE),
        re.compile(r"\breferences?\s+from\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+referenced\s+by\b", re.IGNORECASE),
        re.compile(r"\bbibliography\s+of\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+did\s+.+\s+cite\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+cited\s+in\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bcited\s+in\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+.+\s+cite\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+papers?\s+does\s+.+\s+cite\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+it\s+cites\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+they\s+cite\b", re.IGNORECASE),
        # More specific "references in" pattern - requires paper/bibliography context
        re.compile(r"\breferences?\s+in\s+(the\s+)?(paper|bibliography|appendix)\b", re.IGNORECASE),
        re.compile(r"\bsources?\s+in\s+(the\s+)?(paper|bibliography|appendix)\b", re.IGNORECASE),
        re.compile(r"\bshow\s+references?\b", re.IGNORECASE),
        re.compile(r"\blist\s+references?\b", re.IGNORECASE),
    ],
    "citations": [
        # Existing patterns
        re.compile(r"\bcited\s+by\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+citing\b", re.IGNORECASE),
        re.compile(r"\bwho\s+cited\b", re.IGNORECASE),
        re.compile(r"\bcitations?\s+to\b", re.IGNORECASE),
        re.compile(r"\bcitations?\s+of\b", re.IGNORECASE),
        re.compile(r"\bfind\s+citations?\b", re.IGNORECASE),
        re.compile(r"\bget\s+citations?\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bpapers?\s+that\s+cite\b", re.IGNORECASE),
        re.compile(r"\bwork\s+citing\b", re.IGNORECASE),
        re.compile(r"\bresearch\s+citing\b", re.IGNORECASE),
        re.compile(r"\bstudies?\s+citing\b", re.IGNORECASE),
        re.compile(r"\barticles?\s+citing\b", re.IGNORECASE),
        re.compile(r"\bworks?\s+that\s+cite\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+which\s+cite\b", re.IGNORECASE),
        re.compile(r"\bshow\s+citations?\b", re.IGNORECASE),
        re.compile(r"\blist\s+citations?\b", re.IGNORECASE),
    ],
    "similar": [
        # Existing patterns
        re.compile(r"\bsimilar\s+to\s+this\s+paper\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+papers?\s+to\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+like\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+similar\s+to\b", re.IGNORECASE),
        re.compile(r"\bfind\s+similar\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\brelated\s+to\b", re.IGNORECASE),
        re.compile(r"\brelated\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bwork\s+similar\s+to\b", re.IGNORECASE),
        re.compile(r"\bstudies?\s+resembling\b", re.IGNORECASE),
        re.compile(r"\bresembles?\b", re.IGNORECASE),
        re.compile(r"\bresembling\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+resembling\b", re.IGNORECASE),
        re.compile(r"\bworks?\s+like\b", re.IGNORECASE),
        re.compile(r"\barticles?\s+like\b", re.IGNORECASE),
        re.compile(r"\bstudies?\s+like\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+work\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+research\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+studies?\b", re.IGNORECASE),
        re.compile(r"\bcomparable\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bcomparable\s+work\b", re.IGNORECASE),
    ],
    "trending": [
        # Existing patterns
        re.compile(r"\btrending\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bwhat'?s?\s+hot\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+papers?\b", re.IGNORECASE),
        re.compile(r"\btrending\s+(in|on|about)\b", re.IGNORECASE),
        re.compile(r"\bcurrently\s+popular\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bhot\s+topics?\b", re.IGNORECASE),
        re.compile(r"\bhot\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bwhat'?s?\s+trending\b", re.IGNORECASE),
        re.compile(r"\btrending\s+research\b", re.IGNORECASE),
        re.compile(r"\btrending\s+topics?\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+research\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+now\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+recently\b", re.IGNORECASE),
        re.compile(r"\brecently\s+popular\b", re.IGNORECASE),
        re.compile(r"\bhot\s+research\b", re.IGNORECASE),
        re.compile(r"\bhot\s+in\b", re.IGNORECASE),
        re.compile(r"\btrending\s+now\b", re.IGNORECASE),
        re.compile(r"\bhot\s+topics?\s+in\b", re.IGNORECASE),
    ],
    "useful": [
        # Existing patterns
        re.compile(r"\bmost\s+useful\b", re.IGNORECASE),
        re.compile(r"\buseful\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bhigh\s+utility\b", re.IGNORECASE),
        re.compile(r"\bhigh-utility\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bhelpful\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bhelpful\s+research\b", re.IGNORECASE),
        re.compile(r"\bhelpful\s+work\b", re.IGNORECASE),
        re.compile(r"\bfoundational\s+work\b", re.IGNORECASE),
        re.compile(r"\bfoundational\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bessential\s+reading\b", re.IGNORECASE),
        re.compile(r"\bessential\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bmust[\s-]?read\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bmust[\s-]?read\b", re.IGNORECASE),
        re.compile(r"\bkey\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bkey\s+references?\b", re.IGNORECASE),
        re.compile(r"\bimportant\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bseminal\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bseminal\s+work\b", re.IGNORECASE),
        re.compile(r"\blandmark\s+papers?\b", re.IGNORECASE),
    ],
    "reviews": [
        # Existing patterns
        re.compile(r"\breview\s+articles?\s+(on|about|of)\b", re.IGNORECASE),
        re.compile(r"\breviews?\s+of\b", re.IGNORECASE),
        re.compile(r"\bfind\s+reviews?\b", re.IGNORECASE),
        re.compile(r"\breview\s+papers?\s+(on|about|of)\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bsurvey\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bsurvey\s+articles?\b", re.IGNORECASE),
        re.compile(r"\boverviews?\s+of\b", re.IGNORECASE),
        re.compile(r"\boverview\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bcomprehensive\s+reviews?\b", re.IGNORECASE),
        re.compile(r"\bcomprehensive\s+survey\b", re.IGNORECASE),
        re.compile(r"\bsurvey\s+(on|about|of)\b", re.IGNORECASE),
        re.compile(r"\breviews?\s+(on|about)\b", re.IGNORECASE),
        re.compile(r"\bstate[\s-]?of[\s-]?the[\s-]?art\s+review\b", re.IGNORECASE),
        re.compile(r"\bliterature\s+review\b", re.IGNORECASE),
        re.compile(r"\breview\s+of\s+the\s+literature\b", re.IGNORECASE),
        re.compile(r"\bsystematic\s+review\b", re.IGNORECASE),
        re.compile(r"\btutorial\s+(on|about)\b", re.IGNORECASE),
        re.compile(r"\btutorial\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bintroduction\s+to\b", re.IGNORECASE),
    ],
}

# Patterns to remove from text after operator is detected
OPERATOR_REMOVAL_PATTERNS: dict[str, list[re.Pattern]] = {
    "references": [
        # Specific "X cited by" patterns
        re.compile(r"\bsources?\s+cited\s+by\b", re.IGNORECASE),
        re.compile(r"\bworks?\s+cited\s+by\b", re.IGNORECASE),
        # Existing patterns
        re.compile(r"\breferences?\s+of\b", re.IGNORECASE),
        re.compile(r"\breferences?\s+from\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+referenced\s+by\b", re.IGNORECASE),
        re.compile(r"\bbibliography\s+of\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+did\s+.+\s+cite\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+cited\s+in\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bcited\s+in\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+.+\s+cite\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+papers?\s+does\s+.+\s+cite\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+it\s+cites\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+they\s+cite\b", re.IGNORECASE),
        re.compile(r"\breferences?\s+in\s+(the\s+)?(paper|bibliography|appendix)\b", re.IGNORECASE),
        re.compile(r"\bsources?\s+in\s+(the\s+)?(paper|bibliography|appendix)\b", re.IGNORECASE),
        re.compile(r"\bshow\s+references?\b", re.IGNORECASE),
        re.compile(r"\blist\s+references?\b", re.IGNORECASE),
    ],
    "citations": [
        # Existing patterns
        re.compile(r"\bcited\s+by\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+citing\b", re.IGNORECASE),
        re.compile(r"\bwho\s+cited\b", re.IGNORECASE),
        re.compile(r"\bcitations?\s+to\b", re.IGNORECASE),
        re.compile(r"\bcitations?\s+of\b", re.IGNORECASE),
        re.compile(r"\bfind\s+citations?\b", re.IGNORECASE),
        re.compile(r"\bget\s+citations?\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bpapers?\s+that\s+cite\b", re.IGNORECASE),
        re.compile(r"\bwork\s+citing\b", re.IGNORECASE),
        re.compile(r"\bresearch\s+citing\b", re.IGNORECASE),
        re.compile(r"\bstudies?\s+citing\b", re.IGNORECASE),
        re.compile(r"\barticles?\s+citing\b", re.IGNORECASE),
        re.compile(r"\bworks?\s+that\s+cite\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+which\s+cite\b", re.IGNORECASE),
        re.compile(r"\bshow\s+citations?\b", re.IGNORECASE),
        re.compile(r"\blist\s+citations?\b", re.IGNORECASE),
    ],
    "similar": [
        # Existing patterns
        re.compile(r"\bsimilar\s+to\s+this\s+paper\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+papers?\s+to\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+like\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+similar\s+to\b", re.IGNORECASE),
        re.compile(r"\bfind\s+similar\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\brelated\s+to\b", re.IGNORECASE),
        re.compile(r"\brelated\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bwork\s+similar\s+to\b", re.IGNORECASE),
        re.compile(r"\bstudies?\s+resembling\b", re.IGNORECASE),
        re.compile(r"\bresembles?\b", re.IGNORECASE),
        re.compile(r"\bresembling\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+resembling\b", re.IGNORECASE),
        re.compile(r"\bworks?\s+like\b", re.IGNORECASE),
        re.compile(r"\barticles?\s+like\b", re.IGNORECASE),
        re.compile(r"\bstudies?\s+like\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+work\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+research\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+studies?\b", re.IGNORECASE),
        re.compile(r"\bcomparable\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bcomparable\s+work\b", re.IGNORECASE),
    ],
    "trending": [
        # Existing patterns
        re.compile(r"\btrending\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bwhat'?s?\s+hot\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+papers?\b", re.IGNORECASE),
        re.compile(r"\btrending\s+(in|on|about)\b", re.IGNORECASE),
        re.compile(r"\bcurrently\s+popular\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bhot\s+topics?\b", re.IGNORECASE),
        re.compile(r"\bhot\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bwhat'?s?\s+trending\b", re.IGNORECASE),
        re.compile(r"\btrending\s+research\b", re.IGNORECASE),
        re.compile(r"\btrending\s+topics?\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+research\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+now\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+recently\b", re.IGNORECASE),
        re.compile(r"\brecently\s+popular\b", re.IGNORECASE),
        re.compile(r"\bhot\s+research\b", re.IGNORECASE),
        re.compile(r"\bhot\s+in\b", re.IGNORECASE),
        re.compile(r"\btrending\s+now\b", re.IGNORECASE),
        re.compile(r"\bhot\s+topics?\s+in\b", re.IGNORECASE),
    ],
    "useful": [
        # Existing patterns
        re.compile(r"\bmost\s+useful\b", re.IGNORECASE),
        re.compile(r"\buseful\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bhigh\s+utility\b", re.IGNORECASE),
        re.compile(r"\bhigh-utility\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bhelpful\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bhelpful\s+research\b", re.IGNORECASE),
        re.compile(r"\bhelpful\s+work\b", re.IGNORECASE),
        re.compile(r"\bfoundational\s+work\b", re.IGNORECASE),
        re.compile(r"\bfoundational\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bessential\s+reading\b", re.IGNORECASE),
        re.compile(r"\bessential\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bmust[\s-]?read\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bmust[\s-]?read\b", re.IGNORECASE),
        re.compile(r"\bkey\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bkey\s+references?\b", re.IGNORECASE),
        re.compile(r"\bimportant\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bseminal\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bseminal\s+work\b", re.IGNORECASE),
        re.compile(r"\blandmark\s+papers?\b", re.IGNORECASE),
    ],
    "reviews": [
        # Existing patterns
        re.compile(r"\breview\s+articles?\s+(on|about|of)\b", re.IGNORECASE),
        re.compile(r"\breviews?\s+of\b", re.IGNORECASE),
        re.compile(r"\bfind\s+reviews?\b", re.IGNORECASE),
        re.compile(r"\breview\s+papers?\s+(on|about|of)\b", re.IGNORECASE),
        # New patterns from US-004
        re.compile(r"\bsurvey\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bsurvey\s+articles?\b", re.IGNORECASE),
        re.compile(r"\boverviews?\s+of\b", re.IGNORECASE),
        re.compile(r"\boverview\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bcomprehensive\s+reviews?\b", re.IGNORECASE),
        re.compile(r"\bcomprehensive\s+survey\b", re.IGNORECASE),
        re.compile(r"\bsurvey\s+(on|about|of)\b", re.IGNORECASE),
        re.compile(r"\breviews?\s+(on|about)\b", re.IGNORECASE),
        re.compile(r"\bstate[\s-]?of[\s-]?the[\s-]?art\s+review\b", re.IGNORECASE),
        re.compile(r"\bliterature\s+review\b", re.IGNORECASE),
        re.compile(r"\breview\s+of\s+the\s+literature\b", re.IGNORECASE),
        re.compile(r"\bsystematic\s+review\b", re.IGNORECASE),
        re.compile(r"\btutorial\s+(on|about)\b", re.IGNORECASE),
        re.compile(r"\btutorial\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bintroduction\s+to\b", re.IGNORECASE),
    ],
}


# =============================================================================
# YEAR EXTRACTION PATTERNS
# =============================================================================

YEAR_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Explicit ranges: "from 2015 to 2020", "2015-2020", "between 2015 and 2020"
    (re.compile(r"\bfrom\s+(\d{4})\s+to\s+(\d{4})\b", re.IGNORECASE), "range"),
    (re.compile(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", re.IGNORECASE), "range"),
    (re.compile(r"\b(\d{4})\s*[-–—]\s*(\d{4})\b"), "range"),
    # Since/after: "since 2020", "after 2019"
    (re.compile(r"\bsince\s+(\d{4})\b", re.IGNORECASE), "since"),
    (re.compile(r"\bafter\s+(\d{4})\b", re.IGNORECASE), "after"),
    (re.compile(r"\bfrom\s+(\d{4})\b", re.IGNORECASE), "since"),
    # Before/until: "before 2020", "until 2019"
    (re.compile(r"\bbefore\s+(\d{4})\b", re.IGNORECASE), "before"),
    (re.compile(r"\buntil\s+(\d{4})\b", re.IGNORECASE), "until"),
    (re.compile(r"\bthrough\s+(\d{4})\b", re.IGNORECASE), "until"),
    # Relative: "last N years", "past N years"
    (re.compile(r"\b(?:last|past)\s+(\d+)\s+years?\b", re.IGNORECASE), "last_n"),
    # Single year: "in 2020", "from 2020"
    (re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE), "exact"),
    # Decade: "in the 1990s", "from the 2000s"
    (re.compile(r"\bin\s+the\s+(\d{4})s\b", re.IGNORECASE), "decade"),
    (re.compile(r"\bthe\s+(\d{4})s\b", re.IGNORECASE), "decade"),
]


# =============================================================================
# AUTHOR EXTRACTION PATTERNS
# =============================================================================

AUTHOR_PATTERNS: list[re.Pattern] = [
    # "by Hawking", "by Stephen Hawking", "by S. Hawking"
    # Use word boundary \b and explicit stopword exclusion with negative lookahead
    re.compile(
        r"\bby\s+([A-Z][a-z]+)(?:\s+([A-Z]\.?))?\b",
        re.IGNORECASE,
    ),
    # "author Hawking", "author: Hawking"
    re.compile(
        r"\bauthors?\s*:?\s+([A-Z][a-z]+)(?:\s+([A-Z]\.?))?\b",
        re.IGNORECASE,
    ),
    # "first author Hawking", "first-author Hawking"
    re.compile(
        r"\bfirst[-\s]?author\s+([A-Z][a-z]+)(?:\s+([A-Z]\.?))?\b",
        re.IGNORECASE,
    ),
    # "Hawking et al." - captures name before et al.
    re.compile(
        r"\b([A-Z][a-z]+)\s+et\s+al\.?\b",
        re.IGNORECASE,
    ),
]

# Noise words to remove from extracted author names
AUTHOR_NOISE_WORDS: set[str] = {
    "the",
    "and",
    "or",
    "about",
    "on",
    "in",
    "from",
    "with",
    "papers",
    "paper",
    "articles",
    "article",
    "publications",
}


# =============================================================================
# STOPWORDS AND NOISE
# =============================================================================

STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "ought",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "for",
    "and",
    "nor",
    "but",
    "or",
    "yet",
    "so",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "of",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    # Domain-specific noise
    "papers",
    "paper",
    "articles",
    "article",
    "publications",
    "publication",
    "studies",
    "study",
    "research",
    "work",
    "works",
    "find",
    "show",
    "get",
    "me",
    "search",
    "look",
    "looking",
    "give",
    "please",
}


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================


def extract_intent(text: str) -> IntentSpec:
    """Extract structured intent from natural language search query.

    This is the main NER function that parses user input into an IntentSpec.
    It implements strict operator gating to prevent malformed queries.

    Args:
        text: Natural language search query from user

    Returns:
        IntentSpec with extracted fields and validated values

    Note:
        If text appears to already be ADS syntax, minimal extraction is done
        and the text is preserved for passthrough validation.
    """
    if not text or not text.strip():
        return IntentSpec(raw_user_text=text)

    # Preserve original text
    original_text = text.strip()
    working_text = original_text

    # Check if already ADS query - minimal extraction
    if is_ads_query(working_text):
        return IntentSpec(
            raw_user_text=original_text,
            confidence={"ads_passthrough": 1.0},
        )

    # Initialize intent
    intent = IntentSpec(raw_user_text=original_text)

    # Extract operator (FIRST - so we can remove operator phrases from text)
    intent.operator, working_text = _extract_operator(working_text)

    # Extract years
    intent.year_from, intent.year_to, working_text = _extract_years(working_text)

    # Extract authors
    intent.authors, working_text = _extract_authors(working_text)

    # Extract enum fields with synonym resolution
    intent.property, working_text = _extract_properties(working_text)
    intent.doctype, working_text = _extract_doctypes(working_text)
    intent.bibgroup, working_text = _extract_bibgroups(working_text)
    intent.collection, working_text = _extract_collections(working_text)

    # Remaining text becomes free text terms (topics)
    intent.free_text_terms, intent.or_terms = _extract_topics(working_text)

    # Set confidence scores
    _set_confidence_scores(intent)

    return intent


# =============================================================================
# EXTRACTION HELPERS
# =============================================================================


def _extract_operator(text: str) -> tuple[str | None, str]:
    """Extract operator from text using strict gating patterns.

    CRITICAL: Only matches explicit operator patterns.
    Words like 'citing' or 'references' as topics do NOT trigger operators.

    Args:
        text: Input text to scan

    Returns:
        Tuple of (operator_name or None, text with operator phrase removed)
    """
    for operator, patterns in OPERATOR_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                # Found operator - remove the triggering phrase
                cleaned = text
                for removal_pattern in OPERATOR_REMOVAL_PATTERNS.get(operator, []):
                    cleaned = removal_pattern.sub(" ", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                return operator, cleaned

    return None, text


def _extract_years(text: str) -> tuple[int | None, int | None, str]:
    """Extract year range from text.

    Args:
        text: Input text to scan

    Returns:
        Tuple of (year_from, year_to, text with year phrases removed)
    """
    current_year = datetime.now().year
    year_from = None
    year_to = None
    cleaned_text = text

    for pattern, pattern_type in YEAR_PATTERNS:
        match = pattern.search(text)
        if match:
            if pattern_type == "range":
                year_from = int(match.group(1))
                year_to = int(match.group(2))
            elif pattern_type == "since":
                year_from = int(match.group(1))
                year_to = current_year
            elif pattern_type == "after":
                year_from = int(match.group(1)) + 1
                year_to = current_year
            elif pattern_type == "before":
                year_to = int(match.group(1)) - 1
            elif pattern_type == "until":
                year_to = int(match.group(1))
            elif pattern_type == "last_n":
                n = int(match.group(1))
                year_from = current_year - n
                year_to = current_year
            elif pattern_type == "exact":
                year_from = int(match.group(1))
                year_to = int(match.group(1))
            elif pattern_type == "decade":
                decade_start = int(match.group(1))
                year_from = decade_start
                year_to = decade_start + 9

            # Remove matched phrase from text
            cleaned_text = pattern.sub(" ", cleaned_text)
            break  # Only extract first year pattern

    # Validate years are reasonable
    if year_from and (year_from < 1800 or year_from > current_year + 5):
        year_from = None
    if year_to and (year_to < 1800 or year_to > current_year + 5):
        year_to = None

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return year_from, year_to, cleaned_text


def _extract_authors(text: str) -> tuple[list[str], str]:
    """Extract author names from text.

    Args:
        text: Input text to scan

    Returns:
        Tuple of (list of author names, text with author phrases removed)
    """
    authors = []
    cleaned_text = text

    for pattern in AUTHOR_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1).strip()
            # Filter out noise words
            if name.lower() not in AUTHOR_NOISE_WORDS and len(name) > 1:
                authors.append(name)
                # Remove the entire matched phrase
                cleaned_text = cleaned_text.replace(match.group(0), " ")

    # Deduplicate while preserving order
    seen = set()
    unique_authors = []
    for author in authors:
        normalized = author.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_authors.append(author)

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return unique_authors, cleaned_text


def _extract_properties(text: str) -> tuple[set[str], str]:
    """Extract property values from text using synonym map.

    Args:
        text: Input text to scan

    Returns:
        Tuple of (set of valid property values, text with property phrases removed)
    """
    properties = set()
    cleaned_text = text.lower()

    # Sort by length descending to match longer phrases first
    for synonym in sorted(PROPERTY_SYNONYMS.keys(), key=len, reverse=True):
        # Use word boundary matching to avoid partial matches (e.g., "oa" in "sloan")
        pattern = re.compile(r"\b" + re.escape(synonym) + r"\b")
        if pattern.search(cleaned_text):
            value = PROPERTY_SYNONYMS[synonym]
            if value in PROPERTIES:  # Validate against enum
                properties.add(value)
                cleaned_text = pattern.sub(" ", cleaned_text)

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return properties, cleaned_text


def _extract_doctypes(text: str) -> tuple[set[str], str]:
    """Extract doctype values from text using synonym map.

    Args:
        text: Input text to scan

    Returns:
        Tuple of (set of valid doctype values, text with doctype phrases removed)
    """
    doctypes = set()
    cleaned_text = text.lower()

    for synonym in sorted(DOCTYPE_SYNONYMS.keys(), key=len, reverse=True):
        # Use word boundary matching to avoid partial matches
        pattern = re.compile(r"\b" + re.escape(synonym) + r"\b")
        if pattern.search(cleaned_text):
            value = DOCTYPE_SYNONYMS[synonym]
            if value in DOCTYPES:
                doctypes.add(value)
                cleaned_text = pattern.sub(" ", cleaned_text)

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return doctypes, cleaned_text


def _extract_bibgroups(text: str) -> tuple[set[str], str]:
    """Extract bibgroup values from text using synonym map.

    Args:
        text: Input text to scan

    Returns:
        Tuple of (set of valid bibgroup values, text with bibgroup phrases removed)
    """
    bibgroups = set()
    cleaned_text = text.lower()

    for synonym in sorted(BIBGROUP_SYNONYMS.keys(), key=len, reverse=True):
        # Use word boundary matching to avoid partial matches
        pattern = re.compile(r"\b" + re.escape(synonym) + r"\b")
        if pattern.search(cleaned_text):
            value = BIBGROUP_SYNONYMS[synonym]
            if value in BIBGROUPS:
                bibgroups.add(value)
                cleaned_text = pattern.sub(" ", cleaned_text)

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return bibgroups, cleaned_text


def _extract_collections(text: str) -> tuple[set[str], str]:
    """Extract collection values from text using synonym map.

    Args:
        text: Input text to scan

    Returns:
        Tuple of (set of valid collection values, text with collection phrases removed)
    """
    collections = set()
    cleaned_text = text.lower()

    for synonym in sorted(COLLECTION_SYNONYMS.keys(), key=len, reverse=True):
        # Use word boundary matching to avoid partial matches
        pattern = re.compile(r"\b" + re.escape(synonym) + r"\b")
        if pattern.search(cleaned_text):
            value = COLLECTION_SYNONYMS[synonym]
            if value in COLLECTIONS:
                collections.add(value)
                cleaned_text = pattern.sub(" ", cleaned_text)

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return collections, cleaned_text


def _extract_topics(text: str) -> tuple[list[str], list[str]]:
    """Extract remaining topic terms from text.

    Removes stopwords and noise, returning meaningful topic phrases.
    Handles "X or Y" patterns by returning them separately for OR combination.

    Args:
        text: Input text with other extractions already removed

    Returns:
        Tuple of (free_text_terms for AND, or_terms for OR combination)
    """
    if not text.strip():
        return [], []

    # Clean term helper
    def clean_term(term: str) -> str:
        words = re.findall(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\b", term.lower())
        meaningful = [w for w in words if w not in STOPWORDS and len(w) > 1]
        return " ".join(meaningful)

    # First, check for "X or Y" patterns
    # This handles cases like "rocks or volcanoes" -> or_terms=["rocks", "volcanoes"]
    or_pattern = re.compile(r"\b(\S+(?:\s+\S+)*?)\s+or\s+(\S+(?:\s+\S+)*?)\b", re.IGNORECASE)
    or_match = or_pattern.search(text)

    if or_match:
        # Found "X or Y" pattern - extract as or_terms
        term1 = or_match.group(1).strip()
        term2 = or_match.group(2).strip()

        clean1 = clean_term(term1)
        clean2 = clean_term(term2)

        # Build or_terms list
        or_terms = []
        if clean1:
            or_terms.append(clean1)
        if clean2:
            or_terms.append(clean2)

        # Handle any remaining text outside the OR pattern as free_text_terms
        remaining = or_pattern.sub(" ", text).strip()
        free_text = []
        if remaining:
            remaining_words = re.findall(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\b", remaining.lower())
            remaining_meaningful = [w for w in remaining_words if w not in STOPWORDS and len(w) > 1]
            if remaining_meaningful:
                free_text.append(" ".join(remaining_meaningful))

        return free_text, or_terms

    # No OR pattern - standard processing into free_text_terms
    # Tokenize
    words = re.findall(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\b", text.lower())

    # Filter stopwords
    meaningful = [w for w in words if w not in STOPWORDS and len(w) > 1]

    if not meaningful:
        return [], []

    # Group remaining words into a single topic phrase
    topic = " ".join(meaningful)
    return ([topic] if topic else []), []


def _set_confidence_scores(intent: IntentSpec) -> None:
    """Set confidence scores for extracted fields.

    Args:
        intent: IntentSpec to update with confidence scores
    """
    confidence = {}

    if intent.operator:
        confidence["operator"] = 0.95  # Pattern match = high confidence
    if intent.year_from or intent.year_to:
        confidence["year"] = 0.9
    if intent.authors:
        confidence["authors"] = 0.85
    if intent.property:
        confidence["property"] = 0.9  # Synonym match = high confidence
    if intent.doctype:
        confidence["doctype"] = 0.9
    if intent.bibgroup:
        confidence["bibgroup"] = 0.9
    if intent.collection:
        confidence["database"] = 0.9
    if intent.free_text_terms:
        confidence["topics"] = 0.7  # Lower - just remaining words
    if intent.or_terms:
        confidence["or_topics"] = 0.85  # Higher - explicit OR pattern detected

    intent.confidence = confidence
