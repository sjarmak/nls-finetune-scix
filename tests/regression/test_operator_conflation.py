"""Regression tests for operator conflation patterns.

These tests specifically catch the malformed operator patterns that caused
the hybrid NER pipeline refactor. The original end-to-end fine-tuned model
would conflate natural language words like "citing" and "references" with
ADS operator syntax, producing broken queries like `citations(abs:referencesabs:...)`.

This test suite ensures these patterns NEVER appear in pipeline output.

References:
    - PRD US-009: Regression test suite for known failure patterns
    - AGENTS.md: Operator Gating Rule
"""

import re

import pytest

from finetune.domains.scix.field_constraints import FIELD_ENUMS
from finetune.domains.scix.pipeline import process_query


# Malformed operator patterns that should NEVER appear
MALFORMED_PATTERNS = [
    r"citationsabs:",
    r"citationsauthor:",
    r"citationstitle:",
    r"referencesabs:",
    r"referencesauthor:",
    r"referencestitle:",
    r"trendingabs:",
    r"trendingauthor:",
    r"trendingtitle:",
    r"usefulabs:",
    r"usefulauthor:",
    r"usefultitle:",
    r"similarabs:",
    r"similarauthor:",
    r"similartitle:",
    r"reviewsabs:",
    r"reviewsauthor:",
    r"reviewstitle:",
    # Additional patterns from prior failures
    r"usefulcitations\(",
    r"similarreferences\(",
    r"citationsreferences\(",
    r"referencescitations\(",
    r"trending\(abs:referencesabs:",
    r"citations\(abs:citationsabs:",
]


def check_no_malformed_patterns(query: str) -> list[str]:
    """Check query for any malformed operator patterns.

    Args:
        query: ADS query string to check

    Returns:
        List of malformed patterns found (empty if valid)
    """
    found = []
    for pattern in MALFORMED_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            found.append(pattern)
    return found


def check_balanced_parentheses(query: str) -> bool:
    """Check if parentheses are balanced.

    Args:
        query: ADS query string to check

    Returns:
        True if balanced, False otherwise
    """
    count = 0
    for char in query:
        if char == "(":
            count += 1
        elif char == ")":
            count -= 1
        if count < 0:
            return False
    return count == 0


def check_enum_values(query: str) -> list[str]:
    """Check that all enum field values are valid.

    Args:
        query: ADS query string to check

    Returns:
        List of invalid enum values found
    """
    invalid = []
    for field, valid_values in FIELD_ENUMS.items():
        # Match field:value or field:(value1 OR value2)
        pattern = rf"{field}:(\w+)"
        matches = re.findall(pattern, query, re.IGNORECASE)
        for value in matches:
            if value.lower() not in {v.lower() for v in valid_values}:
                invalid.append(f"{field}:{value}")
    return invalid


class TestOperatorConflationRegression:
    """Regression tests for operator word conflation.

    These tests use inputs that previously caused malformed output.
    """

    @pytest.mark.parametrize(
        "nl_input,should_not_trigger_operator",
        [
            # Words like "citing" used as TOPICS, not operators
            ("citing papers in astrophysics", True),
            ("reference materials for stellar spectra", True),
            ("papers about references in bibliographic analysis", True),
            ("the citing behavior of astronomers", True),
            ("useful citations for students", True),
            ("similar references across journals", True),
            ("trending topics in cosmology", True),  # "trending" as adjective
            # Edge cases with operator words embedded
            ("papers about citation analysis", True),
            ("study of reference patterns", True),
            ("bibliographic citation networks", True),
            ("cross-referencing astronomical data", True),
        ],
    )
    def test_operator_words_as_topics_no_operator(self, nl_input: str, should_not_trigger_operator: bool):
        """Operator words used as topics should NOT trigger operators."""
        result = process_query(nl_input)

        # Check no malformed patterns
        malformed = check_no_malformed_patterns(result.final_query)
        assert not malformed, f"Malformed patterns found: {malformed} in query: {result.final_query}"

        # Check balanced parentheses
        assert check_balanced_parentheses(
            result.final_query
        ), f"Unbalanced parentheses in: {result.final_query}"

        # If should not trigger operator, verify operator is None
        if should_not_trigger_operator:
            assert result.intent.operator is None, (
                f"Expected no operator for '{nl_input}', "
                f"but got operator={result.intent.operator} "
                f"in query: {result.final_query}"
            )

    @pytest.mark.parametrize(
        "nl_input",
        [
            # Inputs that SHOULD produce operators (explicit patterns)
            "papers citing the Hubble deep field paper",
            "who cited Einstein's relativity paper",
            "papers cited by Hawking",
            "references of the Planck 2018 cosmology paper",
            "bibliography of the LIGO detection paper",
            "papers referenced by the JWST paper",
            "trending papers on exoplanets",
            "most useful papers on gravitational waves",
            "papers similar to this black hole paper",
        ],
    )
    def test_explicit_operator_patterns_produce_valid_queries(self, nl_input: str):
        """Explicit operator patterns should produce valid queries."""
        result = process_query(nl_input)

        # Check no malformed patterns
        malformed = check_no_malformed_patterns(result.final_query)
        assert not malformed, f"Malformed patterns found: {malformed} in query: {result.final_query}"

        # Check balanced parentheses
        assert check_balanced_parentheses(
            result.final_query
        ), f"Unbalanced parentheses in: {result.final_query}"

        # Should have an operator set
        assert result.intent.operator is not None, (
            f"Expected operator for '{nl_input}', " f"but got None in query: {result.final_query}"
        )


class TestMalformedConcatenationPatterns:
    """Tests that specifically check for malformed concatenation patterns."""

    @pytest.mark.parametrize(
        "nl_input",
        [
            # Previously failing inputs
            "papers about references in stellar spectra",
            "citing patterns in galaxy evolution",
            "useful citations in the field",
            "similar references for comparison",
            "trending citations in cosmology",
            "reference analysis techniques",
            "citation metrics and impact",
            "papers mentioning citing practices",
            "studies about referencing methods",
            "overview of citation networks",
        ],
    )
    def test_no_operator_concatenation(self, nl_input: str):
        """No operator concatenation patterns should appear."""
        result = process_query(nl_input)

        query = result.final_query
        malformed = check_no_malformed_patterns(query)

        assert not malformed, (
            f"REGRESSION: Malformed concatenation found!\n"
            f"Input: {nl_input}\n"
            f"Output: {query}\n"
            f"Patterns: {malformed}"
        )


class TestNestedOperatorEdgeCases:
    """Tests for nested operator requests (should gracefully handle or reject)."""

    @pytest.mark.parametrize(
        "nl_input",
        [
            # Nested/conflicting operator requests
            "citations of references to gravitational waves",
            "references of papers citing black holes",
            "trending papers that cite exoplanet papers",
            "useful papers referencing JWST observations",
        ],
    )
    def test_nested_operator_requests(self, nl_input: str):
        """Nested operator requests should produce valid (single-operator or no-operator) output."""
        result = process_query(nl_input)

        # Check no malformed patterns
        malformed = check_no_malformed_patterns(result.final_query)
        assert not malformed, f"Malformed patterns in: {result.final_query}"

        # Check balanced parentheses
        assert check_balanced_parentheses(result.final_query)

        # Count operators in output - should be at most 1
        operator_count = sum(
            1
            for op in ["citations(", "references(", "trending(", "useful(", "similar(", "reviews("]
            if op in result.final_query
        )
        assert operator_count <= 1, (
            f"Multiple operators in output ({operator_count}): {result.final_query}"
        )


class TestEmptyAndPassthroughInputs:
    """Tests for empty inputs and ADS passthrough."""

    def test_empty_input(self):
        """Empty input should produce valid (possibly empty) response."""
        result = process_query("")

        assert result.success or result.error is not None
        # Should not crash, no malformed patterns
        malformed = check_no_malformed_patterns(result.final_query)
        assert not malformed

    def test_whitespace_only_input(self):
        """Whitespace-only input should produce valid response."""
        result = process_query("   ")

        assert result.success or result.error is not None
        malformed = check_no_malformed_patterns(result.final_query)
        assert not malformed

    @pytest.mark.parametrize(
        "ads_input",
        [
            'author:"Hawking, S" abs:cosmology',
            "doctype:article property:refereed",
            'citations(abs:"gravitational waves")',
            "bibstem:ApJ pubdate:[2020 TO 2023]",
        ],
    )
    def test_ads_query_passthrough(self, ads_input: str):
        """Already-ADS-syntax input should be validated without NER processing."""
        result = process_query(ads_input)

        # Should not add malformed patterns
        malformed = check_no_malformed_patterns(result.final_query)
        assert not malformed, f"Malformed patterns in passthrough: {result.final_query}"

        # Parentheses should remain balanced
        assert check_balanced_parentheses(result.final_query)


class TestEnumValueValidation:
    """Tests that all enum values in output are valid."""

    @pytest.mark.parametrize(
        "nl_input",
        [
            "peer reviewed papers on cosmology",
            "open access arxiv preprints about JWST",
            "refereed journal articles on exoplanets",
            "PhD thesis on stellar evolution",
            "conference proceedings from astronomy meetings",
            "Hubble telescope observations",
            "SDSS survey data papers",
            "papers in the physics database",
        ],
    )
    def test_all_enum_values_valid(self, nl_input: str):
        """All enum field values in output must be in FIELD_ENUMS."""
        result = process_query(nl_input)

        invalid = check_enum_values(result.final_query)
        assert not invalid, (
            f"Invalid enum values found!\n"
            f"Input: {nl_input}\n"
            f"Output: {result.final_query}\n"
            f"Invalid: {invalid}"
        )


class TestParenthesesBalance:
    """Tests specifically for parentheses balance issues."""

    @pytest.mark.parametrize(
        "nl_input",
        [
            "papers citing (or referencing) black holes",
            "trends in (modern) astrophysics",
            "JWST (James Webb Space Telescope) observations",
            "(latest) papers on cosmology",
            "papers about NGC (1234)",
        ],
    )
    def test_inputs_with_parentheses(self, nl_input: str):
        """Inputs containing parentheses should produce balanced output."""
        result = process_query(nl_input)

        assert check_balanced_parentheses(
            result.final_query
        ), f"Unbalanced parentheses in: {result.final_query}"

        # No malformed patterns
        malformed = check_no_malformed_patterns(result.final_query)
        assert not malformed


class TestKnownFailureStrings:
    """Tests that known failure strings never appear."""

    KNOWN_FAILURE_STRINGS = [
        "citations(abs:referencesabs:",
        "citationsabs:referencesabs:",
        "referencesabs:abs:",
        "abs:citationsabs:",
        "abs:referencesabs:",
        "usefulcitations(",
        "similarreferences(",
        "citations(abs:citations",
        "references(abs:references",
    ]

    @pytest.mark.parametrize(
        "nl_input",
        [
            "papers about references in stellar spectra",
            "citing behavior of researchers",
            "reference standards in astronomy",
            "papers with citations about cosmology",
            "referencing patterns in journals",
            "study of citation networks",
            "papers about useful citations",
            "similar reference works",
            "trending citation metrics",
            "review of references",
        ],
    )
    def test_known_failure_strings_never_appear(self, nl_input: str):
        """Known failure strings should never appear in output."""
        result = process_query(nl_input)

        for failure_string in self.KNOWN_FAILURE_STRINGS:
            assert failure_string.lower() not in result.final_query.lower(), (
                f"REGRESSION: Known failure string found!\n"
                f"Input: {nl_input}\n"
                f"Output: {result.final_query}\n"
                f"Failure string: {failure_string}"
            )
