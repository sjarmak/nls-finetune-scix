"""Post-processing filter for model-generated ADS queries.

Cleans up model output by removing invalid field combinations and
enforcing field enumeration constraints. Logs warnings for removed fields.
"""

import logging
import re

from finetune.domains.scix.field_constraints import FIELD_ENUMS

logger = logging.getLogger(__name__)

# Operator names that may be malformed
OPERATORS = {"citations", "references", "trending", "useful", "similar", "reviews", "topn"}


def _fix_malformed_operators(query: str) -> str:
    """Fix malformed operator syntax like 'citationsauthor:' -> 'citations(author:...')'.

    Model sometimes concatenates operator name directly with field name instead of
    using parentheses. This reconstructs the correct syntax.

    Handles patterns like:
    - citationsauthor:"query" -> citations(author:"query")
    - trendingabs:exoplanet -> trending(abs:exoplanet)
    - usefulabs:"cosmology" -> useful(abs:"cosmology")
    """
    for op in OPERATORS:
        # Pattern: operator followed directly by field name + colon
        # Match: citations + author: -> citations(author:
        pattern = rf"\b{op}((?:author|abs|title|pubdate|bibstem|object|keyword|doctype|property|database|bibgroup|aff|full|identifier):)"

        def replace_op(match: re.Match) -> str:
            field_part = match.group(1)
            logger.warning(f"Fixed malformed operator: {op}{field_part} -> {op}({field_part}")
            return f"{op}({field_part}"

        query = re.sub(pattern, replace_op, query, flags=re.IGNORECASE)

    # Balance parentheses if the fix introduced unbalanced ones
    while query.count("(") > query.count(")"):
        # Add closing paren to last unclosed operator
        # Find the position of the last ( without a matching )
        last_unclosed = -1
        paren_count = 0
        for i, c in enumerate(query):
            if c == "(":
                paren_count += 1
                last_unclosed = i
            elif c == ")":
                paren_count -= 1

        if last_unclosed >= 0:
            # Insert ) at the end, before any trailing operators or whitespace
            query = query + ")"
        else:
            break

    return query


def constrain_query_output(query: str) -> str:
    """Clean up model-generated query by removing invalid field values.

    Removes field:value pairs where the value is not in FIELD_ENUMS.
    Preserves valid field combinations. Handles OR lists, quoted values,
    and parenthesized groups.

    Args:
        query: The raw model-generated query string

    Returns:
        Cleaned query with invalid field values removed

    Example:
        >>> constrain_query_output('doctype:journal property:refereed')
        'property:refereed'
        >>> constrain_query_output('doctype:(article OR journal) abs:exoplanets')
        'doctype:article abs:exoplanets'
    """
    if not query or not query.strip():
        return ""

    result = query.strip()

    # Fix malformed operators (e.g., citationsauthor: -> citations(author:...))
    result = _fix_malformed_operators(result)

    # Process each constrained field
    for field_name, valid_values in FIELD_ENUMS.items():
        valid_lower = {v.lower() for v in valid_values}
        result = _filter_field(result, field_name, valid_lower)

    # Clean up artifacts from removal
    result = _cleanup_query(result)

    return result


def _filter_field(query: str, field_name: str, valid_lower: set[str]) -> str:
    """Filter out invalid values for a specific field.

    Handles:
    - field:value (unquoted)
    - field:"value" (quoted)
    - field:(val1 OR val2 OR val3) (OR list)
    """
    # Pattern for OR list: field:(val1 OR val2)
    or_pattern = rf"\b{field_name}:\s*\(([^)]+)\)"

    def process_or_list(match: re.Match[str]) -> str:
        inner = match.group(1)
        # Split on OR, preserving whitespace for reconstruction
        parts = re.split(r"\s+OR\s+", inner, flags=re.IGNORECASE)
        valid_parts = []
        for part in parts:
            # Strip quotes if present
            clean = part.strip().strip('"')
            if clean.lower() in valid_lower:
                valid_parts.append(part.strip())
            else:
                logger.warning(f"Removed invalid {field_name} value: '{clean}'")

        if not valid_parts:
            # All values invalid - remove entire field expression
            return ""
        elif len(valid_parts) == 1:
            # Single value - no parens needed
            return f"{field_name}:{valid_parts[0]}"
        else:
            return f"{field_name}:({' OR '.join(valid_parts)})"

    query = re.sub(or_pattern, process_or_list, query, flags=re.IGNORECASE)

    # Pattern for quoted value: field:"value"
    quoted_pattern = rf'\b{field_name}:\s*"([^"]*)"'

    def process_quoted(match: re.Match[str]) -> str:
        value = match.group(1)
        if value.lower() in valid_lower:
            return match.group(0)  # Keep as-is
        else:
            logger.warning(f"Removed invalid {field_name} value: '{value}'")
            return ""

    query = re.sub(quoted_pattern, process_quoted, query, flags=re.IGNORECASE)

    # Pattern for unquoted value: field:value
    # Must not match already-processed patterns (quotes, parens)
    unquoted_pattern = rf'\b{field_name}:([^\s()"]+)'

    def process_unquoted(match: re.Match[str]) -> str:
        value = match.group(1)
        if value.lower() in valid_lower:
            return match.group(0)  # Keep as-is
        else:
            logger.warning(f"Removed invalid {field_name} value: '{value}'")
            return ""

    query = re.sub(unquoted_pattern, process_unquoted, query, flags=re.IGNORECASE)

    return query


def _cleanup_query(query: str) -> str:
    """Clean up artifacts from field removal.

    Handles:
    - Trailing/leading operators (AND, OR, NOT)
    - Double operators (AND AND, OR OR)
    - Empty parentheses
    - Extra whitespace
    """
    # Remove empty parentheses (possibly with whitespace)
    query = re.sub(r"\(\s*\)", "", query)

    # Remove leading boolean operators
    query = re.sub(r"^\s*(AND|OR|NOT)\s+", "", query, flags=re.IGNORECASE)

    # Remove trailing boolean operators
    query = re.sub(r"\s+(AND|OR|NOT)\s*$", "", query, flags=re.IGNORECASE)

    # Remove double boolean operators (AND AND -> AND, OR OR -> OR)
    # Also handles AND OR, OR AND combinations
    while True:
        new_query = re.sub(
            r"\b(AND|OR|NOT)\s+(AND|OR)\b",
            r"\2",  # Keep the second operator
            query,
            flags=re.IGNORECASE,
        )
        if new_query == query:
            break
        query = new_query

    # Handle "field:value AND" at start becoming "AND" orphan
    query = re.sub(r"^\s*(AND|OR)\s+", "", query, flags=re.IGNORECASE)

    # Handle "AND field:value" at end becoming orphan "AND"
    query = re.sub(r"\s+(AND|OR|NOT)\s*$", "", query, flags=re.IGNORECASE)

    # Collapse multiple spaces into one
    query = re.sub(r"\s+", " ", query)

    # Remove parentheses that now contain only a single term (no operators)
    # e.g., "(article)" -> "article"
    # BUT preserve operator calls like citations(), trending(), similar(), etc.
    operator_names = r"(citations|references|trending|useful|similar|reviews|topn)"

    def unwrap_single_term(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        full_match = match.group(0)
        start_pos = match.start()

        # Check if this is preceded by an operator name (e.g., citations(), trending())
        # Look at what's before the opening paren in the original query
        prefix = query[:start_pos]
        if re.search(rf"{operator_names}$", prefix, re.IGNORECASE):
            # This is an operator call - preserve the parentheses
            return full_match

        # Check if inner contains no boolean operators (AND, OR, NOT)
        if not re.search(r"\b(AND|OR|NOT)\b", inner, re.IGNORECASE):
            return inner
        return full_match

    query = re.sub(r"\(([^()]+)\)", unwrap_single_term, query)

    # Fix malformed parentheses - remove unbalanced ones
    while True:
        open_count = query.count("(")
        close_count = query.count(")")
        if open_count == close_count:
            break

        if open_count > close_count:
            # Remove rightmost unmatched opening paren
            idx = query.rfind("(")
            query = query[:idx] + query[idx + 1 :]
        else:
            # Remove leftmost unmatched closing paren
            idx = query.find(")")
            query = query[:idx] + query[idx + 1 :]

    return query.strip()
