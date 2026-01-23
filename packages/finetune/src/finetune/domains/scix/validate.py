"""ADS/SciX query validation.

Provides both offline linting and API-backed validation against ADS Search API.
"""

import os
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import httpx

from finetune.domains.scix.field_constraints import FIELD_ENUMS, suggest_correction
from finetune.domains.scix.fields import ADS_FIELDS

# Valid field prefixes (including ^ for first author), normalized to lowercase
VALID_FIELD_PREFIXES = {k.lower() for k in ADS_FIELDS.keys()} | {"^author", "^"}


@dataclass
class FieldConstraintError:
    """Error for an invalid field value."""

    field: str
    value: str
    suggestions: list[str] = dataclass_field(default_factory=list)

    def __str__(self) -> str:
        msg = f"Invalid {self.field} value: '{self.value}'"
        if self.suggestions:
            msg += f" (did you mean: {', '.join(self.suggestions)}?)"
        return msg


@dataclass
class ConstraintValidationResult:
    """Result of field constraint validation."""

    valid: bool
    errors: list[FieldConstraintError] = dataclass_field(default_factory=list)

    @property
    def error_messages(self) -> list[str]:
        """Get list of error messages as strings."""
        return [str(e) for e in self.errors]


@dataclass
class ValidationResult:
    """Result of query validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    normalized: str | None = None


def lint_query(query: str) -> ValidationResult:
    """Fast offline linting for common query issues.

    This catches obvious errors without calling the ADS API.
    Use validate_query() for authoritative validation.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not query or not query.strip():
        return ValidationResult(valid=False, errors=["Empty query"], warnings=[])

    query = query.strip()

    # Check for unbalanced quotes
    quote_count = query.count('"')
    if quote_count % 2 != 0:
        errors.append("Unbalanced quotes")

    # Check for unbalanced parentheses
    paren_count = query.count("(") - query.count(")")
    if paren_count != 0:
        errors.append("Unbalanced parentheses")

    # Check for unbalanced brackets
    bracket_count = query.count("[") - query.count("]")
    if bracket_count != 0:
        errors.append("Unbalanced brackets")

    # Check for invalid field prefixes
    field_pattern = r"(\^?[a-z_]+):"
    fields_used = re.findall(field_pattern, query, re.IGNORECASE)
    for field in fields_used:
        field_lower = field.lower()
        # Handle ^author special case
        if field_lower.startswith("^"):
            base_field = field_lower[1:]
            if base_field not in ADS_FIELDS and base_field != "author":
                errors.append(f"Unknown field prefix: {field}")
        elif field_lower not in VALID_FIELD_PREFIXES:
            errors.append(f"Unknown field prefix: {field}")

    # Check for common pubdate range issues
    pubdate_ranges = re.findall(r"pubdate:\s*\[([^\]]+)\]", query, re.IGNORECASE)
    for range_str in pubdate_ranges:
        if " TO " not in range_str.upper():
            errors.append(f"Invalid pubdate range syntax: [{range_str}] (missing 'TO')")
        else:
            parts = range_str.upper().split(" TO ")
            if len(parts) != 2:
                errors.append(f"Invalid pubdate range: [{range_str}]")

    # Check for leading boolean operators
    if re.match(r"^\s*(AND|OR)\s+", query, re.IGNORECASE):
        errors.append("Query cannot start with AND/OR")

    # Check for trailing boolean operators
    if re.search(r"\s+(AND|OR|NOT)\s*$", query, re.IGNORECASE):
        errors.append("Query cannot end with boolean operator")

    # Check for double boolean operators
    if re.search(r"\b(AND|OR|NOT)\s+(AND|OR|NOT)\b", query, re.IGNORECASE):
        errors.append("Consecutive boolean operators")

    # Warnings (not errors)
    if "github.com" in query.lower() or "repo:" in query.lower():
        warnings.append("Query contains code-search-like syntax (not ADS)")

    if re.search(r"lang:[a-z]+", query, re.IGNORECASE) and ":" not in query.replace("lang:", ""):
        warnings.append("lang: field alone may return unexpected results")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_field_constraints(query: str) -> ConstraintValidationResult:
    """Validate that field values in a query match allowed enumerations.

    Checks doctype:, collection:, property:, and bibgroup: field values
    against their allowed sets defined in field_constraints.py.

    Args:
        query: The ADS query string to validate

    Returns:
        ConstraintValidationResult with validity status and list of errors
        with suggestions for invalid values.

    Example:
        >>> result = validate_field_constraints('doctype:journal property:openaccess')
        >>> result.valid
        False
        >>> result.errors[0].field
        'doctype'
        >>> result.errors[0].value
        'journal'
        >>> result.errors[0].suggestions
        ['article']
    """
    errors: list[FieldConstraintError] = []

    # Fields to check - these have constrained vocabularies
    constrained_fields = ["doctype", "database", "property", "bibgroup"]

    for field_name in constrained_fields:
        valid_values = FIELD_ENUMS.get(field_name)
        if valid_values is None:
            continue

        # Build regex pattern to match field:value or field:"value"
        # Handles: field:value, field:"value", field:(val1 OR val2)
        pattern = rf'\b{field_name}:\s*(?:"([^"]+)"|(\([^)]+\))|([^\s()]+))'

        for match in re.finditer(pattern, query, re.IGNORECASE):
            # Get the matched value from whichever group matched
            if match.group(1):  # Quoted value
                value = match.group(1)
            elif match.group(2):  # Parenthesized group (OR list)
                # Parse values from (val1 OR val2 OR val3)
                inner = match.group(2)[1:-1]  # Remove parens
                # Split on OR and strip whitespace
                values = [
                    v.strip().strip('"') for v in re.split(r"\s+OR\s+", inner, flags=re.IGNORECASE)
                ]
                for v in values:
                    if v and v.lower() not in {val.lower() for val in valid_values}:
                        suggestions = suggest_correction(field_name, v)
                        errors.append(
                            FieldConstraintError(
                                field=field_name,
                                value=v,
                                suggestions=suggestions,
                            )
                        )
                continue
            else:  # Unquoted value
                value = match.group(3)

            # Check if the value is valid (case-insensitive)
            if value.lower() not in {v.lower() for v in valid_values}:
                suggestions = suggest_correction(field_name, value)
                errors.append(
                    FieldConstraintError(
                        field=field_name,
                        value=value,
                        suggestions=suggestions,
                    )
                )

    return ConstraintValidationResult(
        valid=len(errors) == 0,
        errors=errors,
    )


def validate_query(
    query: str,
    api_key: str | None = None,
    api_url: str = "https://api.adsabs.harvard.edu/v1/search/query",
) -> ValidationResult:
    """Validate query against ADS Search API.

    This is the authoritative validation - if ADS accepts the query,
    it's valid syntax.

    Args:
        query: The ADS query to validate
        api_key: ADS API key (defaults to ADS_API_KEY env var)
        api_url: ADS API endpoint

    Returns:
        ValidationResult with validity status and any errors
    """
    # First run offline lint
    lint_result = lint_query(query)
    if not lint_result.valid:
        return lint_result

    # Get API key
    api_key = api_key or os.environ.get("ADS_API_KEY")
    if not api_key:
        return ValidationResult(
            valid=False,
            errors=["ADS_API_KEY not set - cannot validate against API"],
            warnings=lint_result.warnings,
        )

    # Call ADS API with minimal response (just check if query parses)
    try:
        response = httpx.get(
            api_url,
            params={
                "q": query,
                "rows": 0,  # Don't return any results
                "fl": "bibcode",  # Minimal field list
            },
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=10.0,
        )

        if response.status_code == 200:
            data = response.json()
            num_found = data.get("response", {}).get("numFound", 0)
            return ValidationResult(
                valid=True,
                errors=[],
                warnings=lint_result.warnings
                + (["Query returned 0 results"] if num_found == 0 else []),
                normalized=query,  # ADS doesn't return normalized form
            )

        elif response.status_code == 400:
            # Parse error from ADS
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("msg", "Unknown parse error")
            except Exception:
                error_msg = response.text[:200]

            return ValidationResult(
                valid=False,
                errors=[f"ADS parse error: {error_msg}"],
                warnings=lint_result.warnings,
            )

        elif response.status_code == 401:
            return ValidationResult(
                valid=False,
                errors=["Invalid ADS API key"],
                warnings=lint_result.warnings,
            )

        elif response.status_code == 429:
            return ValidationResult(
                valid=False,
                errors=["ADS API rate limit exceeded"],
                warnings=lint_result.warnings,
            )

        else:
            return ValidationResult(
                valid=False,
                errors=[f"ADS API error: HTTP {response.status_code}"],
                warnings=lint_result.warnings,
            )

    except httpx.TimeoutException:
        return ValidationResult(
            valid=False,
            errors=["ADS API timeout"],
            warnings=lint_result.warnings,
        )
    except httpx.RequestError as e:
        return ValidationResult(
            valid=False,
            errors=[f"ADS API request error: {e}"],
            warnings=lint_result.warnings,
        )


def validate_nl(nl: str) -> tuple[bool, list[str]]:
    """Validate that natural language doesn't contain ADS syntax.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for ADS syntax markers that shouldn't appear in NL
    syntax_patterns = [
        (r"\bauthor:", "contains 'author:'"),
        (r"\babs:", "contains 'abs:'"),
        (r"\babstract:", "contains 'abstract:'"),
        (r"\btitle:", "contains 'title:'"),
        (r"\bpubdate:", "contains 'pubdate:'"),
        (r"\bbibstem:", "contains 'bibstem:'"),
        (r"\bobject:", "contains 'object:'"),
        (r"\bkeyword:", "contains 'keyword:'"),
        (r"\bdoi:", "contains 'doi:'"),
        (r"\barXiv:", "contains 'arXiv:'"),
        (r"\borcid:", "contains 'orcid:'"),
        (r"\baff:", "contains 'aff:'"),
        (r"\binst:", "contains 'inst:'"),
        (r"\bcitation_count:", "contains 'citation_count:'"),
        (r"\bproperty:", "contains 'property:'"),
        (r"\b(?:database|collection):", "contains 'collection:'"),
        (r"\bdoctype:", "contains 'doctype:'"),
        (r"\bfull:", "contains 'full:'"),
        (r"\bbody:", "contains 'body:'"),
    ]

    for pattern, message in syntax_patterns:
        if re.search(pattern, nl, re.IGNORECASE):
            issues.append(message)

    # Check for range syntax
    if re.search(r"\[[^\]]+\s+TO\s+[^\]]+\]", nl, re.IGNORECASE):
        issues.append("contains range syntax [X TO Y]")

    # Check for ^ prefix (first author)
    if re.search(r"\^[a-z]", nl, re.IGNORECASE):
        issues.append("contains ^ prefix (first author syntax)")

    # Basic quality checks
    if len(nl) < 5:
        issues.append("too short (< 5 chars)")
    if len(nl) > 300:
        issues.append("too long (> 300 chars)")
    if nl.count('"') > 6:
        issues.append("too many quotes")

    return len(issues) == 0, issues
