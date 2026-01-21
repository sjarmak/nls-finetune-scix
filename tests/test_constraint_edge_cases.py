"""Unit tests for constraint validation edge cases (US-012).

Tests the constrain_query_output function with synthetic inputs that
simulate the 5 edge cases from the acceptance criteria.
"""

import logging

import pytest

from finetune.domains.scix.constrain import constrain_query_output

# Enable logging to verify constraint violations are logged
logging.basicConfig(level=logging.WARNING)


class TestConstraintEdgeCases:
    """Test 5 constraint edge cases as specified in US-012."""

    def test_edge_case_1_invalid_database_removed(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test 1: 'ADS papers' - model outputs invalid database → post-processing removes it."""
        # Simulated model output with invalid database
        raw_output = "database:ADS abs:papers"

        with caplog.at_level(logging.WARNING):
            constrained = constrain_query_output(raw_output)

        # Invalid database:ADS should be removed
        assert "database:ADS" not in constrained
        assert "database:ads" not in constrained.lower()
        # Valid abs:papers should be kept
        assert "abs:papers" in constrained
        # Constraint violation should be logged
        assert any("database" in record.message.lower() for record in caplog.records)

    def test_edge_case_1b_invalid_database_astrophysics(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test 1b: Invalid database 'astrophysics' should be removed."""
        raw_output = "database:astrophysics abs:exoplanets"

        with caplog.at_level(logging.WARNING):
            constrained = constrain_query_output(raw_output)

        # Invalid database:astrophysics should be removed
        assert "database:astrophysics" not in constrained
        assert "abs:exoplanets" in constrained
        assert any("astrophysics" in record.message for record in caplog.records)

    def test_edge_case_1c_valid_database_kept(self) -> None:
        """Test 1c: Valid database 'astronomy' should be kept."""
        raw_output = "database:astronomy abs:papers"

        constrained = constrain_query_output(raw_output)

        # Valid database:astronomy should be kept
        assert "database:astronomy" in constrained
        assert "abs:papers" in constrained

    def test_edge_case_2_property_refereed_valid(self) -> None:
        """Test 2: 'refereed articles' - property:refereed is valid, should be kept."""
        raw_output = "property:refereed abs:exoplanets"

        constrained = constrain_query_output(raw_output)

        # Valid property:refereed should be kept
        assert "property:refereed" in constrained
        assert "abs:exoplanets" in constrained

    def test_edge_case_2b_invalid_property_peerreviewed(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test 2b: Invalid property 'peerreviewed' should be removed."""
        raw_output = "property:peerreviewed abs:exoplanets"

        with caplog.at_level(logging.WARNING):
            constrained = constrain_query_output(raw_output)

        # Invalid property:peerreviewed should be removed
        assert "property:peerreviewed" not in constrained
        assert "abs:exoplanets" in constrained
        assert any("peerreviewed" in record.message for record in caplog.records)

    def test_edge_case_3_bibgroup_hubble_corrected(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test 3: 'papers by Hubble' - bibgroup:Hubble is invalid, should be removed."""
        raw_output = "bibgroup:Hubble abs:telescope"

        with caplog.at_level(logging.WARNING):
            constrained = constrain_query_output(raw_output)

        # Invalid bibgroup:Hubble should be removed
        assert "bibgroup:Hubble" not in constrained
        assert "bibgroup:hubble" not in constrained.lower()
        # Valid abs:telescope should be kept
        assert "abs:telescope" in constrained
        # Constraint violation should be logged
        assert any("Hubble" in record.message for record in caplog.records)

    def test_edge_case_3b_bibgroup_hst_valid(self) -> None:
        """Test 3b: bibgroup:HST is valid, should be kept."""
        raw_output = "bibgroup:HST abs:telescope"

        constrained = constrain_query_output(raw_output)

        # Valid bibgroup:HST should be kept
        assert "bibgroup:HST" in constrained
        assert "abs:telescope" in constrained

    def test_edge_case_4_doctype_phdthesis_valid(self) -> None:
        """Test 4: 'PhD theses' - doctype:phdthesis is valid, should be kept."""
        raw_output = "doctype:phdthesis abs:black holes"

        constrained = constrain_query_output(raw_output)

        # Valid doctype:phdthesis should be kept
        assert "doctype:phdthesis" in constrained
        assert "abs:black holes" in constrained

    def test_edge_case_4b_doctype_thesis_invalid(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test 4b: Invalid doctype 'thesis' should be removed (should be phdthesis)."""
        raw_output = "doctype:thesis abs:black holes"

        with caplog.at_level(logging.WARNING):
            constrained = constrain_query_output(raw_output)

        # Invalid doctype:thesis should be removed
        assert "doctype:thesis" not in constrained
        assert "abs:black holes" in constrained
        assert any("thesis" in record.message for record in caplog.records)

    def test_edge_case_4c_doctype_journal_invalid(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test 4c: Invalid doctype 'journal' should be removed (should be article)."""
        raw_output = "doctype:journal abs:papers"

        with caplog.at_level(logging.WARNING):
            constrained = constrain_query_output(raw_output)

        # Invalid doctype:journal should be removed
        assert "doctype:journal" not in constrained
        assert "abs:papers" in constrained
        assert any("journal" in record.message for record in caplog.records)

    def test_edge_case_5_property_openaccess_and_data_valid(self) -> None:
        """Test 5: 'data papers with open access' - both properties valid, should be kept."""
        raw_output = "property:openaccess property:data abs:observations"

        constrained = constrain_query_output(raw_output)

        # Both valid properties should be kept
        assert "property:openaccess" in constrained
        assert "property:data" in constrained
        assert "abs:observations" in constrained

    def test_edge_case_5b_invalid_property_open_access(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test 5b: Invalid property 'open_access' should be removed."""
        raw_output = "property:open_access property:data abs:observations"

        with caplog.at_level(logging.WARNING):
            constrained = constrain_query_output(raw_output)

        # Invalid property:open_access should be removed
        assert "property:open_access" not in constrained
        # Valid property:data should be kept
        assert "property:data" in constrained
        assert "abs:observations" in constrained
        assert any("open_access" in record.message for record in caplog.records)

    def test_results_not_empty_after_filtering(self) -> None:
        """Verify results still valid (not empty due to over-aggressive filtering)."""
        # Query with all invalid constrained fields
        raw_output = "doctype:journal database:astrophysics property:peerreviewed abs:exoplanets"

        constrained = constrain_query_output(raw_output)

        # Should not be empty - abs:exoplanets should survive
        assert constrained.strip() != ""
        assert "abs:exoplanets" in constrained
        # All invalid fields should be removed
        assert "doctype:journal" not in constrained
        assert "database:astrophysics" not in constrained
        assert "property:peerreviewed" not in constrained

    def test_mixed_valid_invalid_fields(self) -> None:
        """Test query with mix of valid and invalid fields."""
        raw_output = "doctype:article database:astronomy property:peerreviewed bibgroup:Hubble abs:test"

        constrained = constrain_query_output(raw_output)

        # Valid fields kept
        assert "doctype:article" in constrained
        assert "database:astronomy" in constrained
        # Invalid fields removed
        assert "property:peerreviewed" not in constrained
        assert "bibgroup:Hubble" not in constrained
        assert "bibgroup:hubble" not in constrained.lower()
        # Content kept
        assert "abs:test" in constrained

    def test_or_list_with_invalid_values(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test OR list with mix of valid and invalid values."""
        raw_output = "doctype:(article OR journal OR phdthesis) abs:test"

        with caplog.at_level(logging.WARNING):
            constrained = constrain_query_output(raw_output)

        # Invalid 'journal' should be removed from OR list
        assert "journal" not in constrained
        # Valid values should be kept
        assert "article" in constrained
        assert "phdthesis" in constrained
        assert any("journal" in record.message for record in caplog.records)
