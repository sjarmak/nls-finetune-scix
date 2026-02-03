"""Tests for constrain_query_output function - post-processing filter for model output."""

import logging


from finetune.domains.scix.constrain import constrain_query_output


class TestConstrainQueryOutput:
    """Tests for constrain_query_output function."""

    # ============================================================
    # Basic valid queries - should remain unchanged
    # ============================================================

    def test_valid_doctype_article_unchanged(self):
        """Valid doctype should remain unchanged."""
        assert constrain_query_output("doctype:article") == "doctype:article"

    def test_valid_property_refereed_unchanged(self):
        """Valid property should remain unchanged."""
        assert constrain_query_output("property:refereed") == "property:refereed"

    def test_valid_database_astronomy_unchanged(self):
        """Valid database should remain unchanged."""
        assert constrain_query_output("database:astronomy") == "database:astronomy"

    def test_valid_combination_preserved(self):
        """Valid field combinations should be preserved."""
        query = "doctype:article AND property:refereed"
        assert constrain_query_output(query) == "doctype:article AND property:refereed"

    # ============================================================
    # Invalid field values - should be removed
    # ============================================================

    def test_invalid_doctype_journal_removed(self):
        """'journal' is not a valid doctype - model hallucination."""
        assert constrain_query_output("doctype:journal") == ""

    def test_invalid_doctype_paper_removed(self):
        """'paper' is not a valid doctype - model hallucination."""
        assert constrain_query_output("doctype:paper") == ""

    def test_invalid_property_peerreviewed_removed(self):
        """'peerreviewed' is not valid - should be 'refereed'."""
        assert constrain_query_output("property:peerreviewed") == ""

    def test_invalid_database_astro_removed(self):
        """'astro' is not valid - should be 'astronomy'."""
        assert constrain_query_output("database:astro") == ""

    def test_invalid_property_reviewed_removed(self):
        """'reviewed' is not valid property."""
        assert constrain_query_output("property:reviewed") == ""

    # ============================================================
    # Mixed valid and invalid - only invalid removed
    # ============================================================

    def test_mixed_valid_invalid_preserves_valid(self):
        """Mix of valid and invalid should keep valid parts."""
        query = "doctype:article AND property:peerreviewed"
        assert constrain_query_output(query) == "doctype:article"

    def test_invalid_between_valid_fields(self):
        """Invalid field between valid ones should be removed."""
        query = "doctype:article property:fake database:astronomy"
        assert constrain_query_output(query) == "doctype:article database:astronomy"

    def test_multiple_invalid_all_removed(self):
        """Multiple invalid values all removed."""
        query = "doctype:journal database:astro"
        assert constrain_query_output(query) == ""

    # ============================================================
    # OR list handling
    # ============================================================

    def test_or_list_all_valid(self):
        """OR list with all valid values preserved."""
        query = "doctype:(article OR eprint)"
        assert constrain_query_output(query) == "doctype:(article OR eprint)"

    def test_or_list_partial_valid(self):
        """OR list with some invalid values filters them out."""
        query = "doctype:(article OR journal OR eprint)"
        result = constrain_query_output(query)
        assert result == "doctype:(article OR eprint)"

    def test_or_list_single_valid_unwrapped(self):
        """OR list reducing to single value removes parens."""
        query = "doctype:(article OR journal)"
        # journal is invalid, only article remains
        assert constrain_query_output(query) == "doctype:article"

    def test_or_list_all_invalid_removed(self):
        """OR list with all invalid values removed entirely."""
        query = "doctype:(journal OR paper)"
        assert constrain_query_output(query) == ""

    # ============================================================
    # Quoted values
    # ============================================================

    def test_quoted_valid_value_preserved(self):
        """Quoted valid values should be preserved."""
        assert constrain_query_output('doctype:"article"') == 'doctype:"article"'

    def test_quoted_invalid_value_removed(self):
        """Quoted invalid values should be removed."""
        assert constrain_query_output('doctype:"journal"') == ""

    # ============================================================
    # Trailing/leading operator cleanup
    # ============================================================

    def test_trailing_and_removed(self):
        """Trailing AND after removal should be cleaned."""
        query = "doctype:journal AND property:refereed"
        assert constrain_query_output(query) == "property:refereed"

    def test_leading_and_removed(self):
        """Leading AND after removal should be cleaned."""
        query = "property:refereed AND doctype:journal"
        assert constrain_query_output(query) == "property:refereed"

    def test_double_and_collapsed(self):
        """Double AND from removal should be collapsed."""
        query = "doctype:article AND doctype:journal AND property:refereed"
        assert constrain_query_output(query) == "doctype:article AND property:refereed"

    def test_trailing_or_removed(self):
        """Trailing OR after removal should be cleaned."""
        query = "property:refereed OR doctype:journal"
        assert constrain_query_output(query) == "property:refereed"

    # ============================================================
    # Edge cases - empty, whitespace, unconstrained fields
    # ============================================================

    def test_empty_string(self):
        """Empty string returns empty."""
        assert constrain_query_output("") == ""

    def test_whitespace_only(self):
        """Whitespace only returns empty."""
        assert constrain_query_output("   ") == ""

    def test_unconstrained_fields_preserved(self):
        """Fields without constraints are preserved."""
        query = 'author:"Einstein" abs:relativity'
        assert constrain_query_output(query) == 'author:"Einstein" abs:relativity'

    def test_mixed_constrained_unconstrained(self):
        """Mix of constrained and unconstrained fields."""
        query = 'author:"Hawking" doctype:journal abs:"black holes"'
        result = constrain_query_output(query)
        assert result == 'author:"Hawking" abs:"black holes"'

    # ============================================================
    # Common model hallucinations
    # ============================================================

    def test_hallucination_doctype_research(self):
        """Model might output 'research' instead of 'article'."""
        assert constrain_query_output("doctype:research") == ""

    def test_hallucination_property_peer_reviewed(self):
        """Model might output 'peer_reviewed' with underscore."""
        assert constrain_query_output("property:peer_reviewed") == ""

    def test_hallucination_doctype_publication(self):
        """Model might output 'publication' generically."""
        assert constrain_query_output("doctype:publication") == ""

    def test_hallucination_database_astrophysics(self):
        """Model might confuse 'astrophysics' with 'astronomy'."""
        assert constrain_query_output("database:astrophysics") == ""

    def test_hallucination_property_open_access(self):
        """Model might output 'open_access' with underscore."""
        assert constrain_query_output("property:open_access") == ""

    def test_hallucination_bibgroup_hubble(self):
        """Model might use 'Hubble' instead of 'HST'."""
        assert constrain_query_output("bibgroup:Hubble") == ""

    def test_hallucination_esources_pdf(self):
        """Model might hallucinate simple 'pdf'."""
        assert constrain_query_output("esources:pdf") == ""

    # ============================================================
    # Complex queries with multiple issues
    # ============================================================

    def test_complex_query_partial_cleanup(self):
        """Complex query with some valid, some invalid fields."""
        query = 'doctype:article AND property:peerreviewed AND database:astronomy abs:"cosmology"'
        result = constrain_query_output(query)
        assert "doctype:article" in result
        assert "database:astronomy" in result
        assert 'abs:"cosmology"' in result
        assert "peerreviewed" not in result

    def test_multiple_or_lists(self):
        """Multiple OR lists in same query."""
        query = "doctype:(article OR journal) property:(refereed OR reviewed)"
        result = constrain_query_output(query)
        assert result == "doctype:article property:refereed"

    # ============================================================
    # Parentheses handling
    # ============================================================

    def test_empty_parens_removed(self):
        """Empty parentheses from removal should be cleaned."""
        query = "doctype:(journal)"
        result = constrain_query_output(query)
        assert result == ""
        assert "()" not in result

    def test_unbalanced_parens_fixed(self):
        """Unbalanced parentheses should be handled."""
        query = "doctype:article ("
        result = constrain_query_output(query)
        assert result.count("(") == result.count(")")

    # ============================================================
    # Case sensitivity
    # ============================================================

    def test_case_insensitive_valid(self):
        """Validation should be case-insensitive."""
        assert constrain_query_output("doctype:ARTICLE") == "doctype:ARTICLE"

    def test_case_insensitive_invalid(self):
        """Invalid check should be case-insensitive."""
        assert constrain_query_output("doctype:JOURNAL") == ""

    # ============================================================
    # Logging warnings
    # ============================================================

    def test_logs_warning_for_removed_field(self, caplog):
        """Should log warning when removing invalid field."""
        with caplog.at_level(logging.WARNING):
            constrain_query_output("doctype:journal")
        assert "Removed invalid doctype value: 'journal'" in caplog.text

    def test_logs_multiple_warnings(self, caplog):
        """Should log warning for each removed value."""
        with caplog.at_level(logging.WARNING):
            constrain_query_output("doctype:journal database:astro")
        assert "doctype" in caplog.text
        assert "database" in caplog.text
