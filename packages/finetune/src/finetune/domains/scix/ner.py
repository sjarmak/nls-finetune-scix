"""Rules-based NER extraction for natural language to IntentSpec conversion.

This module extracts structured intent from natural language search queries
using rule-based patterns. It implements strict operator gating to prevent
the conflation of natural language words with ADS operator syntax.

CRITICAL: Operators are ONLY set when explicit patterns match.
Words like "citing", "references", "similar" as topics do NOT trigger operators.
"""

import re
from datetime import datetime

from .field_constraints import BIBGROUPS, DATABASES, DOCTYPES, PROPERTIES
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

DATABASE_SYNONYMS: dict[str, str] = {
    "astronomy": "astronomy",
    "astro": "astronomy",
    "astrophysics": "astronomy",
    "physics": "physics",
    "general": "general",
}


# =============================================================================
# OPERATOR GATING PATTERNS
# =============================================================================
# CRITICAL: These patterns MUST be explicit and specific.
# Do NOT trigger operators for generic use of these words as topics.

OPERATOR_PATTERNS: dict[str, list[re.Pattern]] = {
    "citations": [
        re.compile(r"\bcited\s+by\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+citing\b", re.IGNORECASE),
        re.compile(r"\bwho\s+cited\b", re.IGNORECASE),
        re.compile(r"\bcitations?\s+to\b", re.IGNORECASE),
        re.compile(r"\bcitations?\s+of\b", re.IGNORECASE),
        re.compile(r"\bfind\s+citations?\b", re.IGNORECASE),
        re.compile(r"\bget\s+citations?\b", re.IGNORECASE),
    ],
    "references": [
        re.compile(r"\breferences?\s+of\b", re.IGNORECASE),
        re.compile(r"\breferences?\s+from\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+referenced\s+by\b", re.IGNORECASE),
        re.compile(r"\bbibliography\s+of\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+did\s+.+\s+cite\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+cited\s+in\b", re.IGNORECASE),
    ],
    "similar": [
        re.compile(r"\bsimilar\s+to\s+this\s+paper\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+papers?\s+to\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+like\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+similar\s+to\b", re.IGNORECASE),
        re.compile(r"\bfind\s+similar\b", re.IGNORECASE),
    ],
    "trending": [
        re.compile(r"\btrending\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bwhat'?s?\s+hot\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+papers?\b", re.IGNORECASE),
        re.compile(r"\btrending\s+(in|on|about)\b", re.IGNORECASE),
        re.compile(r"\bcurrently\s+popular\b", re.IGNORECASE),
    ],
    "useful": [
        re.compile(r"\bmost\s+useful\b", re.IGNORECASE),
        re.compile(r"\buseful\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bhigh\s+utility\b", re.IGNORECASE),
        re.compile(r"\bhigh-utility\b", re.IGNORECASE),
    ],
    "reviews": [
        re.compile(r"\breview\s+articles?\s+(on|about|of)\b", re.IGNORECASE),
        re.compile(r"\breviews?\s+of\b", re.IGNORECASE),
        re.compile(r"\bfind\s+reviews?\b", re.IGNORECASE),
        re.compile(r"\breview\s+papers?\s+(on|about|of)\b", re.IGNORECASE),
    ],
}

# Patterns to remove from text after operator is detected
OPERATOR_REMOVAL_PATTERNS: dict[str, list[re.Pattern]] = {
    "citations": [
        re.compile(r"\bcited\s+by\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+citing\b", re.IGNORECASE),
        re.compile(r"\bwho\s+cited\b", re.IGNORECASE),
        re.compile(r"\bcitations?\s+to\b", re.IGNORECASE),
        re.compile(r"\bcitations?\s+of\b", re.IGNORECASE),
        re.compile(r"\bfind\s+citations?\b", re.IGNORECASE),
        re.compile(r"\bget\s+citations?\b", re.IGNORECASE),
    ],
    "references": [
        re.compile(r"\breferences?\s+of\b", re.IGNORECASE),
        re.compile(r"\breferences?\s+from\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+referenced\s+by\b", re.IGNORECASE),
        re.compile(r"\bbibliography\s+of\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+did\s+.+\s+cite\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+cited\s+in\b", re.IGNORECASE),
    ],
    "similar": [
        re.compile(r"\bsimilar\s+to\s+this\s+paper\b", re.IGNORECASE),
        re.compile(r"\bsimilar\s+papers?\s+to\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+like\b", re.IGNORECASE),
        re.compile(r"\bpapers?\s+similar\s+to\b", re.IGNORECASE),
        re.compile(r"\bfind\s+similar\b", re.IGNORECASE),
    ],
    "trending": [
        re.compile(r"\btrending\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bwhat'?s?\s+hot\b", re.IGNORECASE),
        re.compile(r"\bpopular\s+papers?\b", re.IGNORECASE),
        re.compile(r"\btrending\s+(in|on|about)\b", re.IGNORECASE),
        re.compile(r"\bcurrently\s+popular\b", re.IGNORECASE),
    ],
    "useful": [
        re.compile(r"\bmost\s+useful\b", re.IGNORECASE),
        re.compile(r"\buseful\s+papers?\b", re.IGNORECASE),
        re.compile(r"\bhigh\s+utility\b", re.IGNORECASE),
        re.compile(r"\bhigh-utility\b", re.IGNORECASE),
    ],
    "reviews": [
        re.compile(r"\breview\s+articles?\s+(on|about|of)\b", re.IGNORECASE),
        re.compile(r"\breviews?\s+of\b", re.IGNORECASE),
        re.compile(r"\bfind\s+reviews?\b", re.IGNORECASE),
        re.compile(r"\breview\s+papers?\s+(on|about|of)\b", re.IGNORECASE),
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
    intent.database, working_text = _extract_databases(working_text)

    # Remaining text becomes free text terms (topics)
    intent.free_text_terms = _extract_topics(working_text)

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


def _extract_databases(text: str) -> tuple[set[str], str]:
    """Extract database values from text using synonym map.

    Args:
        text: Input text to scan

    Returns:
        Tuple of (set of valid database values, text with database phrases removed)
    """
    databases = set()
    cleaned_text = text.lower()

    for synonym in sorted(DATABASE_SYNONYMS.keys(), key=len, reverse=True):
        # Use word boundary matching to avoid partial matches
        pattern = re.compile(r"\b" + re.escape(synonym) + r"\b")
        if pattern.search(cleaned_text):
            value = DATABASE_SYNONYMS[synonym]
            if value in DATABASES:
                databases.add(value)
                cleaned_text = pattern.sub(" ", cleaned_text)

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return databases, cleaned_text


def _extract_topics(text: str) -> list[str]:
    """Extract remaining topic terms from text.

    Removes stopwords and noise, returning meaningful topic phrases.

    Args:
        text: Input text with other extractions already removed

    Returns:
        List of topic phrases for abs:/title: fields
    """
    if not text.strip():
        return []

    # Tokenize
    words = re.findall(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\b", text.lower())

    # Filter stopwords
    meaningful = [w for w in words if w not in STOPWORDS and len(w) > 1]

    if not meaningful:
        return []

    # Group remaining words into a single topic phrase
    # In future, could split on natural phrase boundaries
    topic = " ".join(meaningful)
    return [topic] if topic else []


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
    if intent.database:
        confidence["database"] = 0.9
    if intent.free_text_terms:
        confidence["topics"] = 0.7  # Lower - just remaining words

    intent.confidence = confidence
