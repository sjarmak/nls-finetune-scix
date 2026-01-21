"""Post-processing filter for model-generated ADS queries.

Cleans up model output by removing invalid field combinations and
enforcing field enumeration constraints. Logs warnings for removed fields.
"""

import logging
import re

from finetune.domains.scix.field_constraints import FIELD_ENUMS

logger = logging.getLogger(__name__)


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
    or_pattern = rf'\b{field_name}:\s*\(([^)]+)\)'

    def process_or_list(match: re.Match[str]) -> str:
        inner = match.group(1)
        # Split on OR, preserving whitespace for reconstruction
        parts = re.split(r'\s+OR\s+', inner, flags=re.IGNORECASE)
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
    query = re.sub(r'\(\s*\)', '', query)

    # Remove leading boolean operators
    query = re.sub(r'^\s*(AND|OR|NOT)\s+', '', query, flags=re.IGNORECASE)

    # Remove trailing boolean operators
    query = re.sub(r'\s+(AND|OR|NOT)\s*$', '', query, flags=re.IGNORECASE)

    # Remove double boolean operators (AND AND -> AND, OR OR -> OR)
    # Also handles AND OR, OR AND combinations
    while True:
        new_query = re.sub(
            r'\b(AND|OR|NOT)\s+(AND|OR)\b',
            r'\2',  # Keep the second operator
            query,
            flags=re.IGNORECASE,
        )
        if new_query == query:
            break
        query = new_query

    # Handle "field:value AND" at start becoming "AND" orphan
    query = re.sub(r'^\s*(AND|OR)\s+', '', query, flags=re.IGNORECASE)

    # Handle "AND field:value" at end becoming orphan "AND"
    query = re.sub(r'\s+(AND|OR|NOT)\s*$', '', query, flags=re.IGNORECASE)

    # Collapse multiple spaces into one
    query = re.sub(r'\s+', ' ', query)

    # Remove parentheses that now contain only a single term (no operators)
    # e.g., "(article)" -> "article"
    def unwrap_single_term(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        # Check if inner contains no boolean operators (AND, OR, NOT)
        if not re.search(r'\b(AND|OR|NOT)\b', inner, re.IGNORECASE):
            return inner
        return match.group(0)

    query = re.sub(r'\(([^()]+)\)', unwrap_single_term, query)

    # Fix malformed parentheses - remove unbalanced ones
    while True:
        open_count = query.count('(')
        close_count = query.count(')')
        if open_count == close_count:
            break

        if open_count > close_count:
            # Remove rightmost unmatched opening paren
            idx = query.rfind('(')
            query = query[:idx] + query[idx + 1 :]
        else:
            # Remove leftmost unmatched closing paren
            idx = query.find(')')
            query = query[:idx] + query[idx + 1 :]

    return query.strip()
