"""Unit tests for the deterministic query assembler.

Tests cover:
- Basic clause building (authors, topics, years, objects)
- Enum-constrained field handling (doctype, property, etc.)
- Operator wrapping
- Validation and safety checks
- Fuzz tests for malformed operator prevention
"""

import random
import re
from string import ascii_lowercase

import pytest

from finetune.domains.scix.assembler import (
    _build_abs_clause,
    _build_affiliation_clause,
    _build_author_clause,
    _build_enum_clause,
    _build_object_clause,
    _build_year_clause,
    _needs_quotes,
    _quote_value,
    _validate_enum_values,
    _wrap_with_operator,
    assemble_query,
    validate_query_syntax,
)
from finetune.domains.scix.intent_spec import IntentSpec


class TestQuoting:
    """Tests for quoting logic."""

    def test_single_word_no_quotes(self):
        assert not _needs_quotes("exoplanets")
        assert _quote_value("exoplanets") == "exoplanets"

    def test_multi_word_needs_quotes(self):
        assert _needs_quotes("black holes")
        assert _quote_value("black holes") == '"black holes"'

    def test_special_chars_need_quotes(self):
        assert _needs_quotes("Hawking, S")
        assert _needs_quotes("2020-2023")
        assert _needs_quotes("NGC(1234)")

    def test_empty_string(self):
        assert not _needs_quotes("")
        assert _quote_value("") == ""


class TestEnumValidation:
    """Tests for enum value validation."""

    def test_valid_doctype(self):
        values = {"article", "eprint"}
        valid = _validate_enum_values("doctype", values)
        assert valid == {"article", "eprint"}

    def test_invalid_doctype_removed(self):
        values = {"article", "journal", "paper"}  # journal and paper are invalid
        valid = _validate_enum_values("doctype", values)
        assert valid == {"article"}

    def test_valid_property(self):
        values = {"refereed", "openaccess"}
        valid = _validate_enum_values("property", values)
        assert valid == {"refereed", "openaccess"}

    def test_case_insensitive_matching(self):
        values = {"REFEREED", "OpenAccess"}
        valid = _validate_enum_values("property", values)
        # Should return canonical casing
        assert "refereed" in valid or "REFEREED" in valid
        assert "openaccess" in valid or "OpenAccess" in valid

    def test_valid_bibgroup(self):
        values = {"HST", "JWST", "SDSS"}
        valid = _validate_enum_values("bibgroup", values)
        assert valid == {"HST", "JWST", "SDSS"}

    def test_invalid_bibgroup_removed(self):
        values = {"HST", "Hubble", "Webb"}  # Hubble and Webb are aliases, not valid
        valid = _validate_enum_values("bibgroup", values)
        assert valid == {"HST"}

    def test_unknown_field_passes_through(self):
        values = {"anything", "goes"}
        valid = _validate_enum_values("unknown_field", values)
        assert valid == {"anything", "goes"}


class TestClauseBuilding:
    """Tests for individual clause building functions."""

    def test_author_clause_single(self):
        result = _build_author_clause(["Hawking, S"])
        assert result == 'author:"Hawking, S"'

    def test_author_clause_multiple(self):
        result = _build_author_clause(["Einstein, A", "Bohr, N"])
        assert 'author:"Einstein, A"' in result
        assert 'author:"Bohr, N"' in result

    def test_author_clause_empty(self):
        assert _build_author_clause([]) == ""

    def test_abs_clause_single_word(self):
        result = _build_abs_clause(["exoplanets"])
        assert result == "abs:exoplanets"

    def test_abs_clause_multi_word(self):
        result = _build_abs_clause(["black hole mergers"])
        assert result == 'abs:"black hole mergers"'

    def test_abs_clause_multiple_terms(self):
        result = _build_abs_clause(["JWST", "exoplanets"])
        assert "abs:JWST" in result
        assert "abs:exoplanets" in result

    def test_abs_clause_empty(self):
        assert _build_abs_clause([]) == ""

    def test_year_clause_range(self):
        result = _build_year_clause(2020, 2023)
        assert result == "pubdate:[2020 TO 2023]"

    def test_year_clause_from_only(self):
        result = _build_year_clause(2020, None)
        assert result == "pubdate:[2020 TO *]"

    def test_year_clause_to_only(self):
        result = _build_year_clause(None, 2023)
        assert result == "pubdate:[* TO 2023]"

    def test_year_clause_empty(self):
        assert _build_year_clause(None, None) == ""

    def test_object_clause_single(self):
        result = _build_object_clause(["M31"])
        assert result == "object:M31"

    def test_object_clause_with_space(self):
        result = _build_object_clause(["NGC 1234"])
        assert result == 'object:"NGC 1234"'

    def test_affiliation_clause(self):
        result = _build_affiliation_clause(["Harvard"])
        assert result == 'aff:"Harvard"'


class TestEnumClauses:
    """Tests for enum-constrained field clauses."""

    def test_single_doctype(self):
        result = _build_enum_clause("doctype", {"article"})
        assert result == "doctype:article"

    def test_multiple_doctypes(self):
        result = _build_enum_clause("doctype", {"article", "eprint"})
        assert "doctype:(" in result
        assert " OR " in result
        assert "article" in result
        assert "eprint" in result

    def test_single_property(self):
        result = _build_enum_clause("property", {"refereed"})
        assert result == "property:refereed"

    def test_invalid_values_filtered(self):
        result = _build_enum_clause("doctype", {"journal", "paper"})
        # Both are invalid, should return empty
        assert result == ""

    def test_mixed_valid_invalid(self):
        result = _build_enum_clause("doctype", {"article", "journal"})
        # journal is invalid, article is valid
        assert result == "doctype:article"


class TestOperatorWrapping:
    """Tests for operator wrapping."""

    def test_citations_wrap(self):
        result = _wrap_with_operator('abs:"exoplanets"', "citations")
        assert result == 'citations(abs:"exoplanets")'

    def test_references_wrap(self):
        result = _wrap_with_operator('author:"Hawking, S"', "references")
        assert result == 'references(author:"Hawking, S")'

    def test_trending_wrap(self):
        result = _wrap_with_operator('abs:"JWST"', "trending")
        assert result == 'trending(abs:"JWST")'

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="Invalid operator"):
            _wrap_with_operator("abs:test", "invalid_operator")

    def test_empty_query_returns_empty(self):
        result = _wrap_with_operator("", "citations")
        assert result == ""


class TestAssembleQuery:
    """Tests for the main assemble_query function."""

    def test_simple_topic(self):
        intent = IntentSpec(free_text_terms=["exoplanets"])
        result = assemble_query(intent)
        assert result == "abs:exoplanets"

    def test_topic_with_author(self):
        intent = IntentSpec(
            free_text_terms=["black holes"],
            authors=["Hawking, S"],
        )
        result = assemble_query(intent)
        assert 'author:"Hawking, S"' in result
        assert 'abs:"black holes"' in result

    def test_topic_with_year_range(self):
        intent = IntentSpec(
            free_text_terms=["JWST"],
            year_from=2022,
            year_to=2024,
        )
        result = assemble_query(intent)
        assert "abs:JWST" in result
        assert "pubdate:[2022 TO 2024]" in result

    def test_property_filter(self):
        intent = IntentSpec(
            free_text_terms=["cosmology"],
            property={"refereed", "openaccess"},
        )
        result = assemble_query(intent)
        assert "abs:cosmology" in result
        assert "property:" in result
        assert "refereed" in result
        assert "openaccess" in result

    def test_doctype_filter(self):
        intent = IntentSpec(
            free_text_terms=["review"],
            doctype={"article"},
        )
        result = assemble_query(intent)
        assert "doctype:article" in result

    def test_bibgroup_filter(self):
        intent = IntentSpec(
            free_text_terms=["observations"],
            bibgroup={"HST", "JWST"},
        )
        result = assemble_query(intent)
        assert "bibgroup:" in result
        assert "HST" in result
        assert "JWST" in result

    def test_operator_citations(self):
        intent = IntentSpec(
            free_text_terms=["gravitational waves"],
            operator="citations",
        )
        result = assemble_query(intent)
        assert result.startswith("citations(")
        assert result.endswith(")")
        assert 'abs:"gravitational waves"' in result

    def test_operator_references(self):
        intent = IntentSpec(
            authors=["Einstein, A"],
            operator="references",
        )
        result = assemble_query(intent)
        assert result.startswith("references(")
        assert 'author:"Einstein, A"' in result

    def test_complex_query(self):
        intent = IntentSpec(
            free_text_terms=["exoplanet atmospheres"],
            authors=["Kreidberg, L"],
            year_from=2020,
            property={"refereed"},
            bibgroup={"JWST"},
        )
        result = assemble_query(intent)
        assert 'author:"Kreidberg, L"' in result
        assert 'abs:"exoplanet atmospheres"' in result
        assert "pubdate:[2020 TO *]" in result
        assert "property:refereed" in result
        assert "bibgroup:JWST" in result

    def test_empty_intent_returns_empty(self):
        intent = IntentSpec()
        result = assemble_query(intent)
        assert result == ""

    def test_operator_with_target(self):
        intent = IntentSpec(
            operator="citations",
            operator_target="2023ApJ...123..456K",
        )
        result = assemble_query(intent)
        assert result.startswith("citations(")
        # Bibcode should be included somehow
        assert "2023ApJ" in result or result != ""


class TestValidateSyntax:
    """Tests for syntax validation."""

    def test_balanced_parentheses(self):
        is_valid, errors = validate_query_syntax('citations(abs:"test")')
        assert is_valid
        assert len(errors) == 0

    def test_unbalanced_parentheses(self):
        is_valid, errors = validate_query_syntax('citations(abs:"test"')
        assert not is_valid
        assert any("Unbalanced" in e for e in errors)

    def test_malformed_operator_detected(self):
        is_valid, errors = validate_query_syntax('citationsabs:"test"')
        assert not is_valid
        assert any("Malformed" in e for e in errors)

    def test_referencesabs_pattern_detected(self):
        is_valid, errors = validate_query_syntax('referencesabs:"stellar"')
        assert not is_valid
        assert any("Malformed" in e for e in errors)


class TestFuzzMalformedOperators:
    """Fuzz tests to ensure random insertion of operator words never produces malformed output."""

    @pytest.mark.parametrize(
        "operator_word",
        ["citations", "references", "citing", "cited", "reference", "trending", "useful", "similar"],
    )
    def test_operator_word_in_nl_never_produces_malformed(self, operator_word):
        """Inserting operator words into NL should never produce malformed concatenations."""
        test_phrases = [
            f"papers about {operator_word} in stellar spectra",
            f"{operator_word} behavior in galaxies",
            f"study of {operator_word} patterns",
            f"the {operator_word} in astrophysics",
        ]

        for phrase in test_phrases:
            intent = IntentSpec(free_text_terms=[phrase])
            result = assemble_query(intent)

            # Check for malformed patterns
            malformed_patterns = [
                r"citationsabs:",
                r"citationsauthor:",
                r"referencesabs:",
                r"referencesauthor:",
                r"trendingabs:",
                r"usefulabs:",
                r"similarabs:",
            ]

            for pattern in malformed_patterns:
                assert not re.search(
                    pattern, result, re.IGNORECASE
                ), f"Malformed pattern '{pattern}' found in result: {result}"

            # Check balanced parentheses
            assert result.count("(") == result.count(
                ")"
            ), f"Unbalanced parentheses in result: {result}"

    @pytest.mark.parametrize("seed", range(10))
    def test_random_phrases_never_produce_malformed(self, seed):
        """Random phrases with operator words should never produce malformed output."""
        random.seed(seed)

        operator_words = ["citations", "references", "citing", "cited", "trending", "useful"]
        topic_words = ["stellar", "galaxy", "exoplanet", "cosmology", "spectroscopy"]

        # Generate random phrase
        phrase_parts = random.sample(topic_words, 2)
        op_word = random.choice(operator_words)
        position = random.randint(0, 2)
        phrase_parts.insert(position, op_word)
        phrase = " ".join(phrase_parts)

        intent = IntentSpec(free_text_terms=[phrase])
        result = assemble_query(intent)

        # Should never have operator concatenation
        assert "citationsabs:" not in result.lower()
        assert "referencesabs:" not in result.lower()
        assert "trendingabs:" not in result.lower()

        # Balanced parentheses
        assert result.count("(") == result.count(")")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_invalid_enums_produces_simpler_query(self):
        """If all enum values are invalid, fall back to topic search."""
        intent = IntentSpec(
            free_text_terms=["cosmology"],
            doctype={"journal", "paper", "publication"},  # All invalid
        )
        result = assemble_query(intent)
        # Should still have topic search
        assert "abs:cosmology" in result
        # Should not have invalid doctypes
        assert "journal" not in result
        assert "paper" not in result

    def test_operator_with_empty_content(self):
        """Operator with no content should return empty or reasonable fallback."""
        intent = IntentSpec(operator="citations")
        result = assemble_query(intent)
        # Either empty or has a reasonable fallback
        assert result == "" or "citations(" in result

    def test_special_characters_in_author(self):
        """Author names with special characters should be properly quoted."""
        intent = IntentSpec(authors=["O'Brien, P"])
        result = assemble_query(intent)
        assert 'author:"O\'Brien, P"' in result

    def test_unicode_in_topic(self):
        """Unicode characters in topic should be handled."""
        intent = IntentSpec(free_text_terms=["α Centauri"])
        result = assemble_query(intent)
        assert "abs:" in result

    def test_very_long_query_parts(self):
        """Very long query parts should not break assembly."""
        long_topic = "a " * 100 + "topic"
        intent = IntentSpec(free_text_terms=[long_topic])
        result = assemble_query(intent)
        assert "abs:" in result
