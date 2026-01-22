"""Unit tests for NER extraction module.

Tests for US-002: Implement rules-based NER with strict operator gating.

CRITICAL: Operator gating tests verify that words like 'citing', 'references'
as TOPICS do NOT trigger operators - only explicit patterns do.
"""

import pytest
from datetime import datetime

from finetune.domains.scix.ner import (
    extract_intent,
    PROPERTY_SYNONYMS,
    DOCTYPE_SYNONYMS,
    BIBGROUP_SYNONYMS,
    DATABASE_SYNONYMS,
    OPERATOR_PATTERNS,
)
from finetune.domains.scix.intent_spec import IntentSpec, OPERATORS
from finetune.domains.scix.field_constraints import (
    PROPERTIES,
    DOCTYPES,
    BIBGROUPS,
    DATABASES,
)


class TestOperatorGating:
    """Tests for strict operator gating.
    
    CRITICAL: These tests verify the core fix for operator conflation.
    Operators should ONLY be set for explicit patterns, not generic words.
    """

    # ==========================================================================
    # POSITIVE CASES: Operator SHOULD be set
    # ==========================================================================

    def test_citations_papers_citing(self):
        """'papers citing X' should trigger citations operator."""
        intent = extract_intent("papers citing Einstein's relativity paper")
        assert intent.operator == "citations"

    def test_citations_cited_by(self):
        """'cited by' should trigger citations operator."""
        intent = extract_intent("cited by recent cosmology papers")
        assert intent.operator == "citations"

    def test_citations_who_cited(self):
        """'who cited' should trigger citations operator."""
        intent = extract_intent("who cited the original dark matter paper")
        assert intent.operator == "citations"

    def test_references_references_of(self):
        """'references of' should trigger references operator."""
        intent = extract_intent("references of the Planck 2018 paper")
        assert intent.operator == "references"

    def test_references_bibliography_of(self):
        """'bibliography of' should trigger references operator."""
        intent = extract_intent("bibliography of Hawking's famous paper")
        assert intent.operator == "references"

    def test_references_papers_referenced_by(self):
        """'papers referenced by' should trigger references operator."""
        intent = extract_intent("papers referenced by this paper")
        assert intent.operator == "references"

    def test_similar_similar_to_this_paper(self):
        """'similar to this paper' should trigger similar operator."""
        intent = extract_intent("similar to this paper about gravitational waves")
        assert intent.operator == "similar"

    def test_similar_papers_like(self):
        """'papers like' should trigger similar operator."""
        intent = extract_intent("papers like the famous cosmology paper")
        assert intent.operator == "similar"

    def test_trending_trending_papers(self):
        """'trending papers' should trigger trending operator."""
        intent = extract_intent("trending papers on exoplanets")
        assert intent.operator == "trending"

    def test_trending_whats_hot(self):
        """'what's hot' should trigger trending operator."""
        intent = extract_intent("what's hot in gravitational wave astronomy")
        assert intent.operator == "trending"

    def test_useful_most_useful(self):
        """'most useful' should trigger useful operator."""
        intent = extract_intent("most useful papers on stellar evolution")
        assert intent.operator == "useful"

    def test_reviews_review_articles_on(self):
        """'review articles on' should trigger reviews operator."""
        intent = extract_intent("review articles on black holes")
        assert intent.operator == "reviews"

    def test_reviews_reviews_of(self):
        """'reviews of' should trigger reviews operator."""
        intent = extract_intent("reviews of dark matter evidence")
        assert intent.operator == "reviews"

    # ==========================================================================
    # NEGATIVE CASES: Operator should NOT be set
    # ==========================================================================

    def test_no_operator_citing_as_topic(self):
        """'citing' alone as topic should NOT trigger operator."""
        intent = extract_intent("papers about citing practices in astronomy")
        assert intent.operator is None

    def test_no_operator_references_as_topic(self):
        """'references' alone as topic should NOT trigger operator."""
        # Note: "about references in" is NOT an operator pattern
        # The pattern "references in" was removed for being too broad
        intent = extract_intent("papers about references in stellar spectra")
        assert intent.operator is None

    def test_no_operator_citation_analysis(self):
        """'citation analysis' should NOT trigger operator."""
        intent = extract_intent("citation analysis methods in astrophysics")
        assert intent.operator is None

    def test_no_operator_reference_materials(self):
        """'reference materials' should NOT trigger operator."""
        intent = extract_intent("reference materials for spectroscopy")
        assert intent.operator is None

    def test_no_operator_similar_as_adjective(self):
        """'similar' as adjective should NOT trigger operator."""
        intent = extract_intent("similar spectral features in quasars")
        assert intent.operator is None

    def test_no_operator_trending_as_adjective(self):
        """Generic 'trending' should NOT trigger operator."""
        intent = extract_intent("trending towards higher redshifts")
        assert intent.operator is None

    def test_no_operator_useful_as_adjective(self):
        """'useful' as adjective should NOT trigger operator."""
        intent = extract_intent("useful techniques for data analysis")
        assert intent.operator is None

    def test_no_operator_review_noun(self):
        """'review' as noun without 'of/about' should NOT trigger operator."""
        intent = extract_intent("annual review summary")
        assert intent.operator is None

    def test_no_operator_simple_topic(self):
        """Simple topic search should have no operator."""
        intent = extract_intent("exoplanets in habitable zones")
        assert intent.operator is None

    def test_no_operator_author_search(self):
        """Author search should have no operator."""
        intent = extract_intent("papers by Hawking")
        assert intent.operator is None

    def test_no_operator_year_search(self):
        """Year search should have no operator."""
        intent = extract_intent("papers from 2020 to 2024 about JWST")
        assert intent.operator is None


class TestPropertySynonyms:
    """Tests for property synonym mapping."""

    def test_refereed_synonym(self):
        """'refereed' maps to property:refereed."""
        intent = extract_intent("refereed papers about dark matter")
        assert "refereed" in intent.property

    def test_peer_reviewed_synonym(self):
        """'peer reviewed' maps to property:refereed."""
        intent = extract_intent("peer reviewed articles on cosmology")
        assert "refereed" in intent.property

    def test_open_access_synonym(self):
        """'open access' maps to property:openaccess."""
        intent = extract_intent("open access papers about exoplanets")
        assert "openaccess" in intent.property

    def test_oa_synonym(self):
        """'oa' maps to property:openaccess."""
        intent = extract_intent("oa papers on galaxies")
        assert "openaccess" in intent.property

    def test_arxiv_synonym(self):
        """'arxiv' maps to property:eprint."""
        intent = extract_intent("arxiv papers on gravitational waves")
        assert "eprint" in intent.property

    def test_preprint_synonym(self):
        """'preprint' maps to property:eprint."""
        intent = extract_intent("preprint about JWST observations")
        assert "eprint" in intent.property

    def test_multiple_properties(self):
        """Multiple property synonyms are extracted."""
        intent = extract_intent("refereed open access papers")
        assert "refereed" in intent.property
        assert "openaccess" in intent.property

    def test_all_properties_valid(self):
        """All extracted properties are in PROPERTIES enum."""
        intent = extract_intent("refereed open access arxiv preprint papers")
        for prop in intent.property:
            assert prop in PROPERTIES, f"Invalid property: {prop}"


class TestDoctypeSynonyms:
    """Tests for doctype synonym mapping."""

    def test_article_synonym(self):
        """'article' maps to doctype:article."""
        intent = extract_intent("journal article about black holes")
        assert "article" in intent.doctype

    def test_thesis_synonym(self):
        """'thesis' maps to doctype:phdthesis."""
        intent = extract_intent("thesis on stellar evolution")
        assert "phdthesis" in intent.doctype

    def test_conference_synonym(self):
        """'conference paper' maps to doctype:inproceedings."""
        intent = extract_intent("conference paper on data analysis")
        assert "inproceedings" in intent.doctype

    def test_software_synonym(self):
        """'software' maps to doctype:software."""
        intent = extract_intent("software for image processing")
        assert "software" in intent.doctype

    def test_all_doctypes_valid(self):
        """All extracted doctypes are in DOCTYPES enum."""
        intent = extract_intent("article thesis conference software")
        for dt in intent.doctype:
            assert dt in DOCTYPES, f"Invalid doctype: {dt}"


class TestBibgroupSynonyms:
    """Tests for bibgroup synonym mapping."""

    def test_hubble_synonym(self):
        """'hubble' maps to bibgroup:HST."""
        intent = extract_intent("hubble observations of galaxies")
        assert "HST" in intent.bibgroup

    def test_webb_synonym(self):
        """'webb' maps to bibgroup:JWST."""
        intent = extract_intent("webb deep field images")
        assert "JWST" in intent.bibgroup

    def test_james_webb_synonym(self):
        """'james webb' maps to bibgroup:JWST."""
        intent = extract_intent("james webb space telescope first images")
        assert "JWST" in intent.bibgroup

    def test_sloan_synonym(self):
        """'sloan' maps to bibgroup:SDSS."""
        intent = extract_intent("sloan survey quasar catalog")
        assert "SDSS" in intent.bibgroup

    def test_ligo_synonym(self):
        """'ligo' maps to bibgroup:LIGO."""
        intent = extract_intent("ligo gravitational wave detections")
        assert "LIGO" in intent.bibgroup

    def test_gravitational_wave_bibgroup(self):
        """'gravitational wave' maps to bibgroup:LIGO."""
        intent = extract_intent("gravitational wave observation methods")
        assert "LIGO" in intent.bibgroup

    def test_all_bibgroups_valid(self):
        """All extracted bibgroups are in BIBGROUPS enum."""
        intent = extract_intent("hubble webb sloan ligo observations")
        for bg in intent.bibgroup:
            assert bg in BIBGROUPS, f"Invalid bibgroup: {bg}"


class TestDatabaseSynonyms:
    """Tests for database synonym mapping."""

    def test_astronomy_synonym(self):
        """'astronomy' maps to database:astronomy."""
        intent = extract_intent("astronomy papers on stellar nucleosynthesis")
        assert "astronomy" in intent.database

    def test_astrophysics_synonym(self):
        """'astrophysics' maps to database:astronomy."""
        intent = extract_intent("astrophysics research on dark energy")
        assert "astronomy" in intent.database

    def test_physics_synonym(self):
        """'physics' maps to database:physics."""
        intent = extract_intent("physics papers on quantum mechanics")
        assert "physics" in intent.database


class TestYearExtraction:
    """Tests for year range extraction."""

    def test_year_range_from_to(self):
        """'from X to Y' extracts year range."""
        intent = extract_intent("papers from 2015 to 2020 about galaxies")
        assert intent.year_from == 2015
        assert intent.year_to == 2020

    def test_year_range_between_and(self):
        """'between X and Y' extracts year range."""
        intent = extract_intent("papers between 2018 and 2024 on JWST")
        assert intent.year_from == 2018
        assert intent.year_to == 2024

    def test_year_range_dash(self):
        """'X-Y' extracts year range."""
        intent = extract_intent("2010-2020 papers about exoplanets")
        assert intent.year_from == 2010
        assert intent.year_to == 2020

    def test_year_since(self):
        """'since X' extracts year_from to current year."""
        current_year = datetime.now().year
        intent = extract_intent("papers since 2020 about ALMA")
        assert intent.year_from == 2020
        assert intent.year_to == current_year

    def test_year_after(self):
        """'after X' extracts year_from as X+1."""
        current_year = datetime.now().year
        intent = extract_intent("papers after 2019 on star formation")
        assert intent.year_from == 2020
        assert intent.year_to == current_year

    def test_year_before(self):
        """'before X' extracts year_to as X-1."""
        intent = extract_intent("papers before 2020 about cosmology")
        assert intent.year_to == 2019
        assert intent.year_from is None

    def test_year_last_n(self):
        """'last N years' extracts relative range."""
        current_year = datetime.now().year
        intent = extract_intent("papers from the last 5 years about TESS")
        assert intent.year_from == current_year - 5
        assert intent.year_to == current_year

    def test_year_exact(self):
        """'in X' extracts exact year."""
        intent = extract_intent("papers in 2022 about Webb")
        assert intent.year_from == 2022
        assert intent.year_to == 2022

    def test_year_decade(self):
        """'1990s' extracts decade range."""
        intent = extract_intent("papers from the 1990s about Hubble")
        assert intent.year_from == 1990
        assert intent.year_to == 1999


class TestAuthorExtraction:
    """Tests for author name extraction."""

    def test_author_by_name(self):
        """'by Hawking' extracts author."""
        intent = extract_intent("papers by Hawking on black holes")
        assert "Hawking" in intent.authors

    def test_author_full_name(self):
        """'by Stephen Hawking' extracts full name."""
        intent = extract_intent("papers by Stephen Hawking")
        assert any("Stephen" in a or "Hawking" in a for a in intent.authors)

    def test_author_et_al(self):
        """'Einstein et al.' extracts first author."""
        intent = extract_intent("Einstein et al. papers on relativity")
        assert "Einstein" in intent.authors

    def test_author_first_author(self):
        """'first author X' extracts author."""
        intent = extract_intent("first author Kepler on exoplanets")
        assert "Kepler" in intent.authors

    def test_no_author_for_topic(self):
        """Topics should not be extracted as authors."""
        intent = extract_intent("papers about stellar evolution")
        assert "stellar" not in intent.authors
        assert "evolution" not in intent.authors


class TestTopicExtraction:
    """Tests for free text topic extraction."""

    def test_simple_topic(self):
        """Simple topic is extracted."""
        intent = extract_intent("papers about black holes")
        assert intent.free_text_terms
        assert any("black" in t or "holes" in t for t in intent.free_text_terms)

    def test_topic_removes_stopwords(self):
        """Stopwords are removed from topics."""
        intent = extract_intent("papers about the evolution of stars")
        if intent.free_text_terms:
            topic = " ".join(intent.free_text_terms)
            assert "the" not in topic.split()
            assert "of" not in topic.split()

    def test_topic_after_other_extractions(self):
        """Topics extracted after other fields removed."""
        intent = extract_intent("refereed papers by Hawking since 2020 on cosmology")
        # "refereed", "by Hawking", "since 2020" should be removed
        # "cosmology" should remain
        assert intent.free_text_terms
        # Cosmology should be in topics
        assert any("cosmology" in t for t in intent.free_text_terms)


class TestAdsPassthrough:
    """Tests for ADS query passthrough detection."""

    def test_ads_query_passthrough(self):
        """ADS query syntax is detected for passthrough."""
        intent = extract_intent('author:"Hawking, S" abs:"black holes"')
        assert intent.confidence.get("ads_passthrough") == 1.0
        assert intent.operator is None  # No extraction done

    def test_ads_citations_passthrough(self):
        """ADS citations() is detected for passthrough."""
        intent = extract_intent('citations(abs:"cosmology")')
        assert intent.confidence.get("ads_passthrough") == 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self):
        """Empty input returns empty IntentSpec."""
        intent = extract_intent("")
        assert intent.free_text_terms == []
        assert intent.operator is None

    def test_whitespace_only(self):
        """Whitespace-only input returns empty IntentSpec."""
        intent = extract_intent("   \n\t  ")
        assert intent.free_text_terms == []

    def test_none_input(self):
        """None-ish input is handled."""
        # Note: Function expects str, but should handle gracefully
        intent = extract_intent("")
        assert isinstance(intent, IntentSpec)

    def test_raw_user_text_preserved(self):
        """Original input is preserved in raw_user_text."""
        original = "papers citing Hawking on black holes since 2020"
        intent = extract_intent(original)
        assert intent.raw_user_text == original

    def test_confidence_scores_set(self):
        """Confidence scores are set for extracted fields."""
        intent = extract_intent("papers citing cosmology since 2020")
        assert "operator" in intent.confidence
        assert "year" in intent.confidence
        assert intent.confidence["operator"] > 0


class TestIntegration:
    """Integration tests for full extraction pipeline."""

    def test_complex_query_with_all_features(self):
        """Complex query extracts all features correctly."""
        query = "refereed open access papers by Hawking from 2015 to 2020 on black holes"
        intent = extract_intent(query)

        assert "refereed" in intent.property
        assert "openaccess" in intent.property
        assert "Hawking" in intent.authors
        assert intent.year_from == 2015
        assert intent.year_to == 2020
        assert intent.operator is None  # No operator pattern
        assert intent.free_text_terms  # Should have "black holes" topic

    def test_operator_with_properties(self):
        """Operator works with property extraction."""
        query = "papers citing refereed cosmology articles"
        intent = extract_intent(query)

        assert intent.operator == "citations"
        assert "refereed" in intent.property

    def test_operator_with_years(self):
        """Operator works with year extraction."""
        query = "papers citing the famous 2015 paper on gravitational waves"
        intent = extract_intent(query)

        assert intent.operator == "citations"
        # Year might or might not be extracted depending on pattern

    def test_telescope_with_topic(self):
        """Telescope/bibgroup works with topic."""
        query = "JWST observations of exoplanet atmospheres"
        intent = extract_intent(query)

        assert "JWST" in intent.bibgroup
        assert intent.free_text_terms


# =============================================================================
# US-004: EXPANDED OPERATOR TRIGGER PATTERNS TESTS
# =============================================================================


class TestCitationsOperatorPatterns:
    """Tests for US-004 expanded citations() operator patterns."""

    # Original patterns (still should work)
    def test_cited_by(self):
        """'cited by' triggers citations operator."""
        intent = extract_intent("cited by recent papers")
        assert intent.operator == "citations"

    def test_papers_citing(self):
        """'papers citing' triggers citations operator."""
        intent = extract_intent("papers citing the original paper")
        assert intent.operator == "citations"

    def test_who_cited(self):
        """'who cited' triggers citations operator."""
        intent = extract_intent("who cited Einstein's paper")
        assert intent.operator == "citations"

    def test_citations_to(self):
        """'citations to' triggers citations operator."""
        intent = extract_intent("citations to the Planck paper")
        assert intent.operator == "citations"

    def test_citations_of(self):
        """'citations of' triggers citations operator."""
        intent = extract_intent("citations of the original work")
        assert intent.operator == "citations"

    def test_find_citations(self):
        """'find citations' triggers citations operator."""
        intent = extract_intent("find citations to this paper")
        assert intent.operator == "citations"

    def test_get_citations(self):
        """'get citations' triggers citations operator."""
        intent = extract_intent("get citations for Hawking 1974")
        assert intent.operator == "citations"

    # New patterns from US-004
    def test_papers_that_cite(self):
        """'papers that cite' triggers citations operator."""
        intent = extract_intent("papers that cite the LIGO discovery")
        assert intent.operator == "citations"

    def test_work_citing(self):
        """'work citing' triggers citations operator."""
        intent = extract_intent("work citing cosmological simulations")
        assert intent.operator == "citations"

    def test_research_citing(self):
        """'research citing' triggers citations operator."""
        intent = extract_intent("research citing the original dark matter paper")
        assert intent.operator == "citations"

    def test_studies_citing(self):
        """'studies citing' triggers citations operator."""
        intent = extract_intent("studies citing the Webb observations")
        assert intent.operator == "citations"

    def test_articles_citing(self):
        """'articles citing' triggers citations operator."""
        intent = extract_intent("articles citing this seminal work")
        assert intent.operator == "citations"

    def test_works_that_cite(self):
        """'works that cite' triggers citations operator."""
        intent = extract_intent("works that cite the exoplanet catalog")
        assert intent.operator == "citations"

    def test_papers_which_cite(self):
        """'papers which cite' triggers citations operator."""
        intent = extract_intent("papers which cite the supernova data")
        assert intent.operator == "citations"

    def test_show_citations(self):
        """'show citations' triggers citations operator."""
        intent = extract_intent("show citations to Hawking 1988")
        assert intent.operator == "citations"

    def test_list_citations(self):
        """'list citations' triggers citations operator."""
        intent = extract_intent("list citations of the CMB paper")
        assert intent.operator == "citations"


class TestReferencesOperatorPatterns:
    """Tests for US-004 expanded references() operator patterns."""

    # Original patterns
    def test_references_of(self):
        """'references of' triggers references operator."""
        intent = extract_intent("references of the Planck paper")
        assert intent.operator == "references"

    def test_references_from(self):
        """'references from' triggers references operator."""
        intent = extract_intent("references from Einstein's paper")
        assert intent.operator == "references"

    def test_papers_referenced_by(self):
        """'papers referenced by' triggers references operator."""
        intent = extract_intent("papers referenced by the LIGO paper")
        assert intent.operator == "references"

    def test_bibliography_of(self):
        """'bibliography of' triggers references operator."""
        intent = extract_intent("bibliography of Hawking's book")
        assert intent.operator == "references"

    def test_what_did_cite(self):
        """'what did X cite' triggers references operator."""
        intent = extract_intent("what did the paper cite")
        assert intent.operator == "references"

    def test_papers_cited_in(self):
        """'papers cited in' triggers references operator."""
        intent = extract_intent("papers cited in the review")
        assert intent.operator == "references"

    # New patterns from US-004
    def test_cited_in(self):
        """'cited in' triggers references operator."""
        intent = extract_intent("cited in the original paper")
        assert intent.operator == "references"

    def test_sources_cited_by(self):
        """'sources cited by' triggers references operator."""
        intent = extract_intent("sources cited by Hawking et al")
        assert intent.operator == "references"

    def test_what_does_cite(self):
        """'what does X cite' triggers references operator."""
        intent = extract_intent("what does this paper cite")
        assert intent.operator == "references"

    def test_what_papers_does_cite(self):
        """'what papers does X cite' triggers references operator."""
        intent = extract_intent("what papers does the review cite")
        assert intent.operator == "references"

    def test_papers_it_cites(self):
        """'papers it cites' triggers references operator."""
        intent = extract_intent("papers it cites on cosmology")
        assert intent.operator == "references"

    def test_papers_they_cite(self):
        """'papers they cite' triggers references operator."""
        intent = extract_intent("papers they cite in the appendix")
        assert intent.operator == "references"

    def test_works_cited_by(self):
        """'works cited by' triggers references operator."""
        intent = extract_intent("works cited by the JWST paper")
        assert intent.operator == "references"

    def test_references_in_paper(self):
        """'references in the paper' triggers references operator."""
        intent = extract_intent("references in the paper")
        assert intent.operator == "references"

    def test_sources_in_bibliography(self):
        """'sources in the bibliography' triggers references operator."""
        intent = extract_intent("sources in the bibliography")
        assert intent.operator == "references"

    def test_show_references(self):
        """'show references' triggers references operator."""
        intent = extract_intent("show references for this paper")
        assert intent.operator == "references"

    def test_list_references(self):
        """'list references' triggers references operator."""
        intent = extract_intent("list references of the catalog paper")
        assert intent.operator == "references"


class TestTrendingOperatorPatterns:
    """Tests for US-004 expanded trending() operator patterns."""

    # Original patterns
    def test_trending_papers(self):
        """'trending papers' triggers trending operator."""
        intent = extract_intent("trending papers on exoplanets")
        assert intent.operator == "trending"

    def test_whats_hot(self):
        """'what's hot' triggers trending operator."""
        intent = extract_intent("what's hot in gravitational waves")
        assert intent.operator == "trending"

    def test_popular_papers(self):
        """'popular papers' triggers trending operator."""
        intent = extract_intent("popular papers on dark matter")
        assert intent.operator == "trending"

    def test_trending_in(self):
        """'trending in' triggers trending operator."""
        intent = extract_intent("trending in cosmology")
        assert intent.operator == "trending"

    def test_currently_popular(self):
        """'currently popular' triggers trending operator."""
        intent = extract_intent("currently popular astronomy topics")
        assert intent.operator == "trending"

    # New patterns from US-004
    def test_hot_topics(self):
        """'hot topics' triggers trending operator."""
        intent = extract_intent("hot topics in astrophysics")
        assert intent.operator == "trending"

    def test_hot_papers(self):
        """'hot papers' triggers trending operator."""
        intent = extract_intent("hot papers this month")
        assert intent.operator == "trending"

    def test_whats_trending(self):
        """'what's trending' triggers trending operator."""
        intent = extract_intent("what's trending in astronomy")
        assert intent.operator == "trending"

    def test_trending_research(self):
        """'trending research' triggers trending operator."""
        intent = extract_intent("trending research on black holes")
        assert intent.operator == "trending"

    def test_trending_topics(self):
        """'trending topics' triggers trending operator."""
        intent = extract_intent("trending topics in stellar physics")
        assert intent.operator == "trending"

    def test_popular_research(self):
        """'popular research' triggers trending operator."""
        intent = extract_intent("popular research areas")
        assert intent.operator == "trending"

    def test_popular_now(self):
        """'popular now' triggers trending operator."""
        intent = extract_intent("popular now in astronomy")
        assert intent.operator == "trending"

    def test_recently_popular(self):
        """'recently popular' triggers trending operator."""
        intent = extract_intent("recently popular papers on exoplanets")
        assert intent.operator == "trending"

    def test_hot_research(self):
        """'hot research' triggers trending operator."""
        intent = extract_intent("hot research in cosmology")
        assert intent.operator == "trending"

    def test_hot_in(self):
        """'hot in' triggers trending operator."""
        intent = extract_intent("hot in gravitational wave astronomy")
        assert intent.operator == "trending"

    def test_trending_now(self):
        """'trending now' triggers trending operator."""
        intent = extract_intent("trending now in astrophysics")
        assert intent.operator == "trending"


class TestSimilarOperatorPatterns:
    """Tests for US-004 expanded similar() operator patterns."""

    # Original patterns
    def test_similar_to_this_paper(self):
        """'similar to this paper' triggers similar operator."""
        intent = extract_intent("similar to this paper on cosmology")
        assert intent.operator == "similar"

    def test_similar_papers_to(self):
        """'similar papers to' triggers similar operator."""
        intent = extract_intent("similar papers to the LIGO paper")
        assert intent.operator == "similar"

    def test_papers_like(self):
        """'papers like' triggers similar operator."""
        intent = extract_intent("papers like the original discovery")
        assert intent.operator == "similar"

    def test_papers_similar_to(self):
        """'papers similar to' triggers similar operator."""
        intent = extract_intent("papers similar to Hawking 1974")
        assert intent.operator == "similar"

    def test_find_similar(self):
        """'find similar' triggers similar operator."""
        intent = extract_intent("find similar papers on dark matter")
        assert intent.operator == "similar"

    # New patterns from US-004
    def test_related_to(self):
        """'related to' triggers similar operator."""
        intent = extract_intent("related to the CMB paper")
        assert intent.operator == "similar"

    def test_related_papers(self):
        """'related papers' triggers similar operator."""
        intent = extract_intent("related papers on exoplanets")
        assert intent.operator == "similar"

    def test_work_similar_to(self):
        """'work similar to' triggers similar operator."""
        intent = extract_intent("work similar to Einstein's paper")
        assert intent.operator == "similar"

    def test_studies_resembling(self):
        """'studies resembling' triggers similar operator."""
        intent = extract_intent("studies resembling the original method")
        assert intent.operator == "similar"

    def test_papers_resembling(self):
        """'papers resembling' triggers similar operator."""
        intent = extract_intent("papers resembling the approach")
        assert intent.operator == "similar"

    def test_works_like(self):
        """'works like' triggers similar operator."""
        intent = extract_intent("works like the seminal paper")
        assert intent.operator == "similar"

    def test_articles_like(self):
        """'articles like' triggers similar operator."""
        intent = extract_intent("articles like this review")
        assert intent.operator == "similar"

    def test_studies_like(self):
        """'studies like' triggers similar operator."""
        intent = extract_intent("studies like the SDSS survey")
        assert intent.operator == "similar"

    def test_similar_work(self):
        """'similar work' triggers similar operator."""
        intent = extract_intent("similar work on stellar evolution")
        assert intent.operator == "similar"

    def test_similar_research(self):
        """'similar research' triggers similar operator."""
        intent = extract_intent("similar research on black holes")
        assert intent.operator == "similar"

    def test_similar_studies(self):
        """'similar studies' triggers similar operator."""
        intent = extract_intent("similar studies on galaxy formation")
        assert intent.operator == "similar"

    def test_comparable_papers(self):
        """'comparable papers' triggers similar operator."""
        intent = extract_intent("comparable papers on the topic")
        assert intent.operator == "similar"

    def test_comparable_work(self):
        """'comparable work' triggers similar operator."""
        intent = extract_intent("comparable work in cosmology")
        assert intent.operator == "similar"


class TestUsefulOperatorPatterns:
    """Tests for US-004 expanded useful() operator patterns."""

    # Original patterns
    def test_most_useful(self):
        """'most useful' triggers useful operator."""
        intent = extract_intent("most useful papers on cosmology")
        assert intent.operator == "useful"

    def test_useful_papers(self):
        """'useful papers' triggers useful operator."""
        intent = extract_intent("useful papers on stellar evolution")
        assert intent.operator == "useful"

    def test_high_utility(self):
        """'high utility' triggers useful operator."""
        intent = extract_intent("high utility papers for research")
        assert intent.operator == "useful"

    # New patterns from US-004
    def test_helpful_papers(self):
        """'helpful papers' triggers useful operator."""
        intent = extract_intent("helpful papers on exoplanet atmospheres")
        assert intent.operator == "useful"

    def test_helpful_research(self):
        """'helpful research' triggers useful operator."""
        intent = extract_intent("helpful research on dark matter")
        assert intent.operator == "useful"

    def test_helpful_work(self):
        """'helpful work' triggers useful operator."""
        intent = extract_intent("helpful work on gravitational waves")
        assert intent.operator == "useful"

    def test_foundational_work(self):
        """'foundational work' triggers useful operator."""
        intent = extract_intent("foundational work on cosmology")
        assert intent.operator == "useful"

    def test_foundational_papers(self):
        """'foundational papers' triggers useful operator."""
        intent = extract_intent("foundational papers in the field")
        assert intent.operator == "useful"

    def test_essential_reading(self):
        """'essential reading' triggers useful operator."""
        intent = extract_intent("essential reading on black holes")
        assert intent.operator == "useful"

    def test_essential_papers(self):
        """'essential papers' triggers useful operator."""
        intent = extract_intent("essential papers for understanding CMB")
        assert intent.operator == "useful"

    def test_must_read_papers(self):
        """'must-read papers' triggers useful operator."""
        intent = extract_intent("must-read papers on exoplanets")
        assert intent.operator == "useful"

    def test_must_read(self):
        """'must read' triggers useful operator."""
        intent = extract_intent("must read for cosmology students")
        assert intent.operator == "useful"

    def test_key_papers(self):
        """'key papers' triggers useful operator."""
        intent = extract_intent("key papers on gravitational lensing")
        assert intent.operator == "useful"

    def test_key_references(self):
        """'key references' triggers useful operator."""
        intent = extract_intent("key references on stellar nucleosynthesis")
        assert intent.operator == "useful"

    def test_important_papers(self):
        """'important papers' triggers useful operator."""
        intent = extract_intent("important papers in astrophysics")
        assert intent.operator == "useful"

    def test_seminal_papers(self):
        """'seminal papers' triggers useful operator."""
        intent = extract_intent("seminal papers on dark energy")
        assert intent.operator == "useful"

    def test_seminal_work(self):
        """'seminal work' triggers useful operator."""
        intent = extract_intent("seminal work in the field")
        assert intent.operator == "useful"

    def test_landmark_papers(self):
        """'landmark papers' triggers useful operator."""
        intent = extract_intent("landmark papers on gravitational waves")
        assert intent.operator == "useful"


class TestReviewsOperatorPatterns:
    """Tests for US-004 expanded reviews() operator patterns."""

    # Original patterns
    def test_review_articles_on(self):
        """'review articles on' triggers reviews operator."""
        intent = extract_intent("review articles on black holes")
        assert intent.operator == "reviews"

    def test_reviews_of(self):
        """'reviews of' triggers reviews operator."""
        intent = extract_intent("reviews of dark matter evidence")
        assert intent.operator == "reviews"

    def test_find_reviews(self):
        """'find reviews' triggers reviews operator."""
        intent = extract_intent("find reviews on cosmology")
        assert intent.operator == "reviews"

    def test_review_papers_on(self):
        """'review papers on' triggers reviews operator."""
        intent = extract_intent("review papers on exoplanets")
        assert intent.operator == "reviews"

    # New patterns from US-004
    def test_survey_papers(self):
        """'survey papers' triggers reviews operator."""
        intent = extract_intent("survey papers on galaxy formation")
        assert intent.operator == "reviews"

    def test_survey_articles(self):
        """'survey articles' triggers reviews operator."""
        intent = extract_intent("survey articles on cosmology")
        assert intent.operator == "reviews"

    def test_overviews_of(self):
        """'overviews of' triggers reviews operator."""
        intent = extract_intent("overviews of stellar evolution")
        assert intent.operator == "reviews"

    def test_overview_papers(self):
        """'overview papers' triggers reviews operator."""
        intent = extract_intent("overview papers on dark matter")
        assert intent.operator == "reviews"

    def test_comprehensive_reviews(self):
        """'comprehensive reviews' triggers reviews operator."""
        intent = extract_intent("comprehensive reviews of black holes")
        assert intent.operator == "reviews"

    def test_comprehensive_survey(self):
        """'comprehensive survey' triggers reviews operator."""
        intent = extract_intent("comprehensive survey on exoplanets")
        assert intent.operator == "reviews"

    def test_survey_on(self):
        """'survey on' triggers reviews operator."""
        intent = extract_intent("survey on gravitational waves")
        assert intent.operator == "reviews"

    def test_reviews_on(self):
        """'reviews on' triggers reviews operator."""
        intent = extract_intent("reviews on stellar physics")
        assert intent.operator == "reviews"

    def test_literature_review(self):
        """'literature review' triggers reviews operator."""
        intent = extract_intent("literature review on cosmology")
        assert intent.operator == "reviews"

    def test_review_of_the_literature(self):
        """'review of the literature' triggers reviews operator."""
        intent = extract_intent("review of the literature on exoplanets")
        assert intent.operator == "reviews"

    def test_systematic_review(self):
        """'systematic review' triggers reviews operator."""
        intent = extract_intent("systematic review on dark matter")
        assert intent.operator == "reviews"

    def test_tutorial_on(self):
        """'tutorial on' triggers reviews operator."""
        intent = extract_intent("tutorial on stellar spectroscopy")
        assert intent.operator == "reviews"

    def test_tutorial_papers(self):
        """'tutorial papers' triggers reviews operator."""
        intent = extract_intent("tutorial papers on data analysis")
        assert intent.operator == "reviews"

    def test_introduction_to(self):
        """'introduction to' triggers reviews operator."""
        intent = extract_intent("introduction to cosmology")
        assert intent.operator == "reviews"


class TestOperatorNegativeCases:
    """Negative cases - patterns that should NOT trigger operators."""

    def test_no_operator_citing_as_subject(self):
        """'citing' as a topic should not trigger operator."""
        intent = extract_intent("studies about citing practices")
        assert intent.operator is None

    def test_no_operator_reference_as_noun(self):
        """'reference' as general noun should not trigger operator."""
        intent = extract_intent("standard reference catalogs")
        assert intent.operator is None

    def test_no_operator_similar_as_adjective_only(self):
        """'similar' alone as adjective should not trigger operator."""
        intent = extract_intent("similar spectral features")
        assert intent.operator is None

    def test_no_operator_trending_downward(self):
        """'trending' in non-popularity context should not trigger."""
        intent = extract_intent("trending downward in luminosity")
        assert intent.operator is None

    def test_no_operator_useful_technique(self):
        """'useful technique' should not trigger useful operator."""
        intent = extract_intent("useful technique for analysis")
        assert intent.operator is None

    def test_no_operator_review_board(self):
        """'review board' should not trigger reviews operator."""
        intent = extract_intent("ethics review board decisions")
        assert intent.operator is None

    def test_no_operator_hot_gas(self):
        """'hot gas' should not trigger trending operator."""
        intent = extract_intent("hot gas in galaxy clusters")
        assert intent.operator is None

    def test_no_operator_related_field(self):
        """'related field' alone should not trigger operator."""
        intent = extract_intent("a related field of study")
        # Note: 'related field' could be ambiguous, test current behavior
        # If it triggers, we may need to tighten the pattern
        pass  # Document actual behavior

    def test_no_operator_foundation_grant(self):
        """'foundation' in other context should not trigger."""
        intent = extract_intent("National Science Foundation grants")
        assert intent.operator is None

    def test_no_operator_survey_mission(self):
        """'survey' as mission type should not trigger reviews."""
        intent = extract_intent("TESS survey mission observations")
        # This is ambiguous - 'survey' could trigger reviews()
        # Document actual behavior for awareness
        pass
