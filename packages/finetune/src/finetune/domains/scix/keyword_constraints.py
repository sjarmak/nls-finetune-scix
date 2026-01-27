"""UAT (Unified Astronomy Thesaurus) keyword validation for ADS queries.

This module validates keyword: field values against the UAT controlled vocabulary.
The UAT is the standard vocabulary for astronomy keywords adopted by AAS journals
and integrated into ADS search.

References:
    - UAT: https://astrothesaurus.org/
    - UAT GitHub: https://github.com/astrothesaurus/UAT
    - ADS UAT Integration: https://ui.adsabs.harvard.edu/blog/uat-integration
"""

import json
import re
from functools import lru_cache
from pathlib import Path

# Path to UAT vocabulary file (relative to package root)
UAT_VOCAB_PATH = (
    Path(__file__).parent.parent.parent.parent.parent.parent.parent
    / "data"
    / "vocabularies"
    / "uat_keywords.json"
)


@lru_cache(maxsize=1)
def load_uat_vocabulary() -> dict[str, dict]:
    """Load the UAT vocabulary from JSON file.

    Returns:
        Dict mapping lowercase keyword to metadata (name, uri, id)
    """
    if not UAT_VOCAB_PATH.exists():
        # Fallback: return empty dict if vocab not available
        return {}

    with open(UAT_VOCAB_PATH) as f:
        data = json.load(f)

    return data.get("concepts", {})


def is_valid_keyword(keyword: str) -> bool:
    """Check if a keyword is in the UAT vocabulary.

    Args:
        keyword: The keyword value to check (case-insensitive)

    Returns:
        True if valid UAT keyword, False otherwise
    """
    vocab = load_uat_vocabulary()
    if not vocab:
        return True  # If no vocab loaded, assume valid

    return keyword.lower().strip() in vocab


def get_keyword_info(keyword: str) -> dict | None:
    """Get UAT metadata for a keyword.

    Args:
        keyword: The keyword to look up

    Returns:
        Dict with name, uri, id if found, None otherwise
    """
    vocab = load_uat_vocabulary()
    return vocab.get(keyword.lower().strip())


def suggest_keyword_corrections(invalid_keyword: str, max_suggestions: int = 5) -> list[str]:
    """Suggest valid UAT keywords similar to an invalid one.

    Uses word overlap and semantic similarity for matching.

    Args:
        invalid_keyword: The invalid keyword to find suggestions for
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of suggested valid keywords, sorted by relevance
    """
    vocab = load_uat_vocabulary()
    if not vocab:
        return []

    query = invalid_keyword.lower().strip()
    query_words = set(query.split())

    suggestions = []

    for term in vocab.keys():
        term_words = set(term.split())

        # Priority 0: Exact word match with different modifier
        # e.g., "galactic dynamics" -> "galaxy dynamics" (both have "dynamics")
        common_words = query_words & term_words
        if len(common_words) > 0:
            # Score based on word overlap ratio
            overlap_ratio = len(common_words) / max(len(query_words), len(term_words))
            # Prefer terms with same word count
            length_penalty = abs(len(query_words) - len(term_words)) * 0.1
            score = 1 - overlap_ratio + length_penalty
            suggestions.append((score, len(term), term))

        # Priority 1: One word is prefix/suffix of another
        # e.g., "stellar" matches "stellar dynamics"
        elif any(qw in term or tw in query for qw in query_words for tw in term_words):
            for qw in query_words:
                if qw in term:
                    suggestions.append((1.5, len(term), term))
                    break

    # Sort by score, then by length (prefer shorter terms)
    suggestions.sort(key=lambda x: (x[0], x[1]))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in suggestions:
        if s[2] not in seen:
            seen.add(s[2])
            unique.append(s)

    # Return canonical names (not lowercase keys)
    return [vocab[s[2]]["name"] for s in unique[:max_suggestions]]


def extract_keywords_from_query(query: str) -> list[str]:
    """Extract keyword: field values from an ADS query.

    Args:
        query: ADS query string

    Returns:
        List of keyword values found in the query
    """
    # Match keyword:"value" or keyword:value
    pattern = r'keyword:(?:"([^"]+)"|(\S+))'
    matches = re.findall(pattern, query, re.IGNORECASE)

    # Each match is a tuple (quoted_value, unquoted_value)
    return [m[0] or m[1] for m in matches]


def validate_query_keywords(query: str) -> list[dict]:
    """Validate all keyword: values in a query against UAT.

    Args:
        query: ADS query string

    Returns:
        List of validation results for each keyword, with:
        - keyword: the keyword value
        - valid: True/False
        - suggestions: list of suggested corrections if invalid
    """
    keywords = extract_keywords_from_query(query)
    results = []

    for kw in keywords:
        valid = is_valid_keyword(kw)
        result = {
            "keyword": kw,
            "valid": valid,
        }
        if not valid:
            result["suggestions"] = suggest_keyword_corrections(kw)
        results.append(result)

    return results


# Word mappings for common variations
# Maps variant words to their UAT equivalents
WORD_VARIANTS = {
    "galactic": "galaxy",  # "galactic dynamics" -> "galaxy dynamics"
    "stellar": "stellar",  # Already correct
    "extragalactic": "extragalactic",  # Already correct
}

# Common keyword corrections based on UAT vocabulary
# Maps common mistakes to correct UAT terms
KEYWORD_CORRECTIONS = {
    "galactic dynamics": "galaxy dynamics",
    "string theory": None,  # Not in astronomy thesaurus (physics concept)
    "galactic evolution": "galaxy evolution",
}


def apply_keyword_correction(keyword: str) -> str | None:
    """Apply known correction for a keyword.

    Args:
        keyword: The keyword to correct

    Returns:
        Corrected keyword if a known correction exists,
        original if valid, None if no valid alternative
    """
    kw_lower = keyword.lower().strip()

    # Check if already valid
    if is_valid_keyword(kw_lower):
        return keyword

    # Check known corrections first
    if kw_lower in KEYWORD_CORRECTIONS:
        correction = KEYWORD_CORRECTIONS[kw_lower]
        if correction is None:
            return None  # No valid astronomy equivalent
        return correction

    # Try word variant substitution
    # e.g., "galactic dynamics" -> "galaxy dynamics"
    words = kw_lower.split()
    corrected_words = [WORD_VARIANTS.get(w, w) for w in words]
    corrected = " ".join(corrected_words)
    if corrected != kw_lower and is_valid_keyword(corrected):
        return corrected

    # Try to find closest match via suggestions
    suggestions = suggest_keyword_corrections(kw_lower, max_suggestions=1)
    if suggestions:
        return suggestions[0]

    return None


def fix_query_keywords(query: str) -> tuple[str, list[dict]]:
    """Fix invalid keyword: values in a query.

    Args:
        query: ADS query string

    Returns:
        Tuple of (fixed_query, list of changes made)
    """
    changes = []
    fixed_query = query

    for kw in extract_keywords_from_query(query):
        if not is_valid_keyword(kw):
            correction = apply_keyword_correction(kw)
            if correction:
                # Replace in query (handle both quoted and unquoted)
                old_patterns = [f'keyword:"{kw}"', f"keyword:{kw}"]
                new_value = f'keyword:"{correction}"'

                for old in old_patterns:
                    if old in fixed_query:
                        fixed_query = fixed_query.replace(old, new_value)
                        changes.append(
                            {"old": kw, "new": correction, "reason": "corrected to valid UAT term"}
                        )
                        break
            else:
                # No valid correction - flag for removal
                changes.append({"old": kw, "new": None, "reason": "no valid UAT equivalent"})

    return fixed_query, changes
