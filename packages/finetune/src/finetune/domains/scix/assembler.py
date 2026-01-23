"""Deterministic query assembler for building ADS queries from IntentSpec.

This module implements template-based query assembly that produces valid
ADS query syntax by composing validated building blocks. All enum values
are validated against FIELD_ENUMS before assembly.

The assembler is deterministic and never generates arbitrary text.
LLM is only used in the resolver fallback path for paper references.
"""

import logging
import re
from collections.abc import Sequence

from .constrain import constrain_query_output
from .field_constraints import FIELD_ENUMS
from .intent_spec import OPERATORS, IntentSpec
from .pipeline import GoldExample

logger = logging.getLogger(__name__)


def _needs_quotes(value: str) -> bool:
    """Check if a value needs quotes in ADS syntax.

    Multi-word phrases and special characters require quoting.

    Args:
        value: The value to check

    Returns:
        True if value should be quoted
    """
    if not value:
        return False
    # Needs quotes if contains spaces, commas, colons, or other special chars
    return bool(re.search(r"[\s,:\-()]", value))


def _quote_value(value: str) -> str:
    """Quote a value for ADS syntax if needed.

    Args:
        value: The value to potentially quote

    Returns:
        Quoted value if needed, otherwise original value
    """
    if _needs_quotes(value):
        # Escape any internal quotes
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _validate_enum_values(field: str, values: set[str]) -> set[str]:
    """Validate enum values against FIELD_ENUMS.

    Filters out invalid values and logs warnings for each removal.

    Args:
        field: Field name (e.g., 'doctype', 'property')
        values: Set of values to validate

    Returns:
        Set of valid values only
    """
    valid_enum = FIELD_ENUMS.get(field)
    if valid_enum is None:
        return values  # No constraints for this field

    valid_lower = {v.lower() for v in valid_enum}
    valid_values = set()

    for value in values:
        if value.lower() in valid_lower:
            # Find the canonical casing from the enum
            for canonical in valid_enum:
                if canonical.lower() == value.lower():
                    valid_values.add(canonical)
                    break
        else:
            logger.warning(f"Removed invalid {field} value: '{value}'")

    return valid_values


def _build_author_clause(authors: Sequence[str]) -> str:
    """Build author search clause.

    Formats author names for ADS syntax: author:"Last, F"

    Args:
        authors: List of author names

    Returns:
        Author clause string, or empty string if no authors
    """
    if not authors:
        return ""

    clauses = []
    for author in authors:
        # Always quote author names
        clauses.append(f'author:"{author}"')

    if len(clauses) == 1:
        return clauses[0]
    return " ".join(clauses)


def _build_abs_clause(terms: Sequence[str], use_or: bool = False) -> str:
    """Build abstract/topic search clause.

    Args:
        terms: List of topic terms/phrases
        use_or: If True, combine terms with OR instead of AND (implicit)

    Returns:
        Abstract clause string, or empty string if no terms
    """
    if not terms:
        return ""

    clauses = []
    for term in terms:
        quoted = _quote_value(term)
        clauses.append(quoted)

    if len(clauses) == 1:
        return f"abs:{clauses[0]}"

    if use_or:
        # Use OR within parentheses: abs:(term1 OR term2)
        return f"abs:({' OR '.join(clauses)})"
    else:
        # Use implicit AND with separate abs: fields
        return " ".join(f"abs:{c}" for c in clauses)


def _build_year_clause(year_from: int | None, year_to: int | None) -> str:
    """Build pubdate range clause.

    Args:
        year_from: Start year (inclusive)
        year_to: End year (inclusive)

    Returns:
        Pubdate clause string, or empty string if no years
    """
    if year_from is None and year_to is None:
        return ""

    if year_from is not None and year_to is not None:
        return f"pubdate:[{year_from} TO {year_to}]"
    elif year_from is not None:
        return f"pubdate:[{year_from} TO *]"
    else:
        return f"pubdate:[* TO {year_to}]"


def _build_enum_clause(field: str, values: set[str]) -> str:
    """Build clause for an enum-constrained field.

    Validates values against FIELD_ENUMS and builds proper syntax.

    Args:
        field: Field name (e.g., 'doctype', 'property')
        values: Set of values to include

    Returns:
        Field clause string, or empty string if no valid values
    """
    if not values:
        return ""

    # Validate values
    valid_values = _validate_enum_values(field, values)
    if not valid_values:
        return ""

    sorted_values = sorted(valid_values)

    if len(sorted_values) == 1:
        return f"{field}:{sorted_values[0]}"
    else:
        or_list = " OR ".join(sorted_values)
        return f"{field}:({or_list})"


def _build_object_clause(objects: Sequence[str]) -> str:
    """Build astronomical object search clause.

    Args:
        objects: List of object names (e.g., 'M31', 'NGC 1234')

    Returns:
        Object clause string, or empty string if no objects
    """
    if not objects:
        return ""

    clauses = []
    for obj in objects:
        # Object names are typically short, but quote if needed
        quoted = _quote_value(obj)
        clauses.append(f"object:{quoted}")

    if len(clauses) == 1:
        return clauses[0]
    return " ".join(clauses)


def _build_affiliation_clause(affiliations: Sequence[str]) -> str:
    """Build affiliation search clause.

    Args:
        affiliations: List of institutional affiliations

    Returns:
        Affiliation clause string, or empty string if no affiliations
    """
    if not affiliations:
        return ""

    clauses = []
    for aff in affiliations:
        # Affiliations are typically multi-word, always quote
        clauses.append(f'aff:"{aff}"')

    if len(clauses) == 1:
        return clauses[0]
    return " ".join(clauses)


def _wrap_with_operator(query: str, operator: str) -> str:
    """Wrap a query with an operator.

    Args:
        query: The base query string
        operator: Operator name (must be in OPERATORS)

    Returns:
        Query wrapped with operator, e.g., 'citations(query)'

    Raises:
        ValueError: If operator is not valid
    """
    if operator not in OPERATORS:
        raise ValueError(f"Invalid operator: {operator}")

    if not query.strip():
        logger.warning(f"Empty query cannot be wrapped with operator {operator}")
        return ""

    return f"{operator}({query})"


def assemble_query(intent: IntentSpec, examples: list[GoldExample] | None = None) -> str:
    """Assemble an ADS query from an IntentSpec.

    This is the main entry point for query assembly. It builds a valid
    ADS query by composing validated building blocks.

    Pipeline:
    1. Build base clauses from IntentSpec fields
    2. Validate all enum values against FIELD_ENUMS
    3. Join clauses with space (implicit AND)
    4. Apply operator wrapper if set
    5. Run constrain_query_output() as final safety net

    Args:
        intent: Structured intent specification from NER
        examples: Retrieved gold examples (optional, for future guidance)

    Returns:
        Valid ADS query string

    Note:
        The examples parameter is currently unused but reserved for
        future pattern-guided assembly improvements.
    """
    clauses: list[str] = []
    constraint_count_before = 0
    constraint_count_after = 0

    # Build author clause
    if intent.authors:
        author_clause = _build_author_clause(intent.authors)
        if author_clause:
            clauses.append(author_clause)

    # Build abstract/topic clause
    if intent.free_text_terms:
        abs_clause = _build_abs_clause(intent.free_text_terms, use_or=False)
        if abs_clause:
            clauses.append(abs_clause)

    # Build OR'd topic clause (e.g., "rocks or volcanoes" -> abs:(rocks OR volcanoes))
    if intent.or_terms:
        or_clause = _build_abs_clause(intent.or_terms, use_or=True)
        if or_clause:
            clauses.append(or_clause)

    # Build year range clause
    if intent.year_from is not None or intent.year_to is not None:
        year_clause = _build_year_clause(intent.year_from, intent.year_to)
        if year_clause:
            clauses.append(year_clause)

    # Build object clause
    if intent.objects:
        object_clause = _build_object_clause(intent.objects)
        if object_clause:
            clauses.append(object_clause)

    # Build affiliation clause
    if intent.affiliations:
        aff_clause = _build_affiliation_clause(intent.affiliations)
        if aff_clause:
            clauses.append(aff_clause)

    # Build enum-constrained field clauses
    for field_name in ("doctype", "property", "collection", "bibgroup", "esources", "data"):
        values = getattr(intent, field_name)
        if values:
            constraint_count_before += len(values)
            clause = _build_enum_clause(field_name, values)
            if clause:
                clauses.append(clause)
                # Count valid values
                valid_values = _validate_enum_values(field_name, values)
                constraint_count_after += len(valid_values)

    # Join all clauses with space (implicit AND)
    base_query = " ".join(clauses)

    # Apply operator wrapper if set
    if intent.operator:
        if base_query:
            base_query = _wrap_with_operator(base_query, intent.operator)
        else:
            # No base query, operator needs a target
            if intent.operator_target:
                # Use the target as the query
                target = _quote_value(intent.operator_target)
                base_query = _wrap_with_operator(target, intent.operator)
            else:
                logger.warning(
                    f"Operator {intent.operator} requested but no base query or target. "
                    "Returning empty query."
                )
                base_query = ""

    # Safety check: if too many constraints were dropped, simplify
    if constraint_count_before > 0:
        drop_ratio = 1 - (constraint_count_after / constraint_count_before)
        if drop_ratio > 0.5:
            logger.warning(
                f"Dropped {drop_ratio:.0%} of constraints. Falling back to simpler query."
            )
            # Fall back to just topic search if available
            if intent.free_text_terms:
                base_query = _build_abs_clause(intent.free_text_terms)

    # Final safety net: run constraint filter
    final_query = constrain_query_output(base_query)

    return final_query


def validate_query_syntax(query: str) -> tuple[bool, list[str]]:
    """Validate query syntax for common issues.

    Checks for:
    - Balanced parentheses
    - No malformed operator concatenations
    - No invalid enum values

    Args:
        query: Query string to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Check balanced parentheses
    if query.count("(") != query.count(")"):
        errors.append(f"Unbalanced parentheses: {query.count('(')} open, {query.count(')')} close")

    # Check for malformed operator concatenations
    malformed_patterns = [
        r"\bcitationsabs:",
        r"\bcitationsauthor:",
        r"\bcitationstitle:",
        r"\breferencesabs:",
        r"\breferencesauthor:",
        r"\breferencestitle:",
        r"\btrendingabs:",
        r"\busefulabs:",
        r"\bsimilarabs:",
        r"\breviewsabs:",
    ]
    for pattern in malformed_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            errors.append(f"Malformed operator pattern found: {pattern}")

    return len(errors) == 0, errors
