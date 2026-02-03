"""Tests for validate_field_constraints function."""


from finetune.domains.scix.validate import (
    FieldConstraintError,
    validate_field_constraints,
)


class TestValidateFieldConstraints:
    """Tests for validate_field_constraints function."""

    # Valid queries - should pass validation

    def test_valid_doctype_article(self):
        """Valid doctype value should pass."""
        result = validate_field_constraints("doctype:article")
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_doctype_phdthesis(self):
        """PhD thesis doctype should be valid."""
        result = validate_field_constraints("doctype:phdthesis")
        assert result.valid is True

    def test_valid_property_refereed(self):
        """Valid property value should pass."""
        result = validate_field_constraints("property:refereed")
        assert result.valid is True

    def test_valid_database_astronomy(self):
        """Valid database value should pass."""
        result = validate_field_constraints("database:astronomy")
        assert result.valid is True

    def test_valid_database_earthscience(self):
        """Valid earthscience database value should pass."""
        result = validate_field_constraints("database:earthscience")
        assert result.valid is True

    def test_valid_bibgroup_hst(self):
        """Valid bibgroup value should pass."""
        result = validate_field_constraints("bibgroup:HST")
        assert result.valid is True

    def test_valid_bibgroup_seti(self):
        """SETI bibgroup should be valid."""
        result = validate_field_constraints("bibgroup:SETI")
        assert result.valid is True

    def test_valid_bibgroup_eso(self):
        """ESO bibgroup (alias for ESO/Telescopes) should be valid."""
        result = validate_field_constraints("bibgroup:ESO")
        assert result.valid is True

    def test_valid_multiple_fields(self):
        """Query with multiple valid constrained fields should pass."""
        query = 'doctype:article property:refereed database:astronomy bibgroup:JWST'
        result = validate_field_constraints(query)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_complex_query(self):
        """Complex query with valid values should pass."""
        query = 'author:"Einstein, A" doctype:article property:openaccess abs:"relativity"'
        result = validate_field_constraints(query)
        assert result.valid is True

    def test_valid_quoted_values(self):
        """Quoted field values should be validated correctly."""
        result = validate_field_constraints('doctype:"article"')
        assert result.valid is True

    def test_valid_case_insensitive(self):
        """Validation should be case-insensitive."""
        result = validate_field_constraints("doctype:ARTICLE property:REFEREED")
        assert result.valid is True

    # Invalid queries - should fail validation

    def test_invalid_doctype_journal(self):
        """'journal' is not a valid doctype - should be 'article'."""
        result = validate_field_constraints("doctype:journal")
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "doctype"
        assert result.errors[0].value == "journal"

    def test_invalid_doctype_paper(self):
        """'paper' is not a valid doctype."""
        result = validate_field_constraints("doctype:paper")
        assert result.valid is False
        assert result.errors[0].field == "doctype"
        assert result.errors[0].value == "paper"

    def test_invalid_property_peerreviewed(self):
        """'peerreviewed' is not valid - should be 'refereed'."""
        result = validate_field_constraints("property:peerreviewed")
        assert result.valid is False
        assert result.errors[0].field == "property"
        assert result.errors[0].value == "peerreviewed"

    def test_invalid_database_astro(self):
        """'astro' is not valid - should be 'astronomy'."""
        result = validate_field_constraints("database:astro")
        assert result.valid is False
        assert result.errors[0].field == "database"
        assert result.errors[0].value == "astro"
        # Should suggest 'astronomy'
        assert "astronomy" in result.errors[0].suggestions

    def test_invalid_bibgroup_hubble(self):
        """'Hubble' is not valid - should be 'HST'."""
        result = validate_field_constraints("bibgroup:Hubble")
        assert result.valid is False
        assert result.errors[0].field == "bibgroup"
        assert result.errors[0].value == "Hubble"

    def test_multiple_invalid_values(self):
        """Multiple invalid values should all be caught."""
        query = "doctype:journal database:astro"
        result = validate_field_constraints(query)
        assert result.valid is False
        assert len(result.errors) == 2
        fields = {e.field for e in result.errors}
        assert fields == {"doctype", "database"}

    def test_mixed_valid_invalid(self):
        """Mix of valid and invalid should catch invalid ones."""
        query = "doctype:article property:reviewed database:astronomy"
        result = validate_field_constraints(query)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "property"
        assert result.errors[0].value == "reviewed"

    def test_or_list_with_invalid(self):
        """OR list with invalid values should catch them."""
        query = "doctype:(article OR journal OR eprint)"
        result = validate_field_constraints(query)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].value == "journal"

    def test_or_list_all_valid(self):
        """OR list with all valid values should pass."""
        query = "doctype:(article OR eprint OR phdthesis)"
        result = validate_field_constraints(query)
        assert result.valid is True

    # Edge cases

    def test_empty_query(self):
        """Empty query should pass (no constrained fields to check)."""
        result = validate_field_constraints("")
        assert result.valid is True

    def test_query_without_constrained_fields(self):
        """Query with only unconstrained fields should pass."""
        query = 'author:"Einstein" abs:"relativity" pubdate:[1900 TO 1950]'
        result = validate_field_constraints(query)
        assert result.valid is True

    def test_error_messages_property(self):
        """error_messages property should return list of strings."""
        result = validate_field_constraints("doctype:journal")
        assert len(result.error_messages) == 1
        assert "Invalid doctype value: 'journal'" in result.error_messages[0]

    def test_suggestions_included(self):
        """Suggestions should be included for similar values."""
        result = validate_field_constraints("doctype:proceeding")
        assert result.valid is False
        # Should suggest 'proceedings' or 'inproceedings'
        assert len(result.errors[0].suggestions) > 0

    def test_field_constraint_error_str(self):
        """FieldConstraintError string representation."""
        error = FieldConstraintError(
            field="doctype",
            value="journal",
            suggestions=["article"],
        )
        assert "doctype" in str(error)
        assert "journal" in str(error)
        assert "article" in str(error)
        assert "did you mean" in str(error)

    def test_field_constraint_error_no_suggestions(self):
        """FieldConstraintError without suggestions."""
        error = FieldConstraintError(field="doctype", value="xyz")
        assert "xyz" in str(error)
        assert "did you mean" not in str(error)
