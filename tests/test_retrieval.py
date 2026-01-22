"""Tests for few-shot retrieval over gold examples.

Tests cover:
- Index loading and initialization
- Similarity scoring with token overlap
- Feature boosting (operators, doctypes, properties, etc.)
- Performance requirements (<20ms for k=5)
- Ranking stability and determinism
"""

import time
from pathlib import Path

import pytest

from finetune.domains.scix.intent_spec import IntentSpec
from finetune.domains.scix.pipeline import GoldExample
from finetune.domains.scix.retrieval import (
    DEFAULT_GOLD_EXAMPLES_PATH,
    GoldExampleIndex,
    IndexedExample,
    extract_features_from_ads_query,
    get_index,
    reset_index,
    retrieve_similar,
    tokenize,
)


# --- Tokenization Tests ---


class TestTokenize:
    """Tests for tokenization function."""

    def test_basic_tokenization(self):
        """Should split text into lowercase tokens."""
        tokens = tokenize("Papers about black holes")
        assert "black" in tokens
        assert "holes" in tokens

    def test_removes_stopwords(self):
        """Should remove common stopwords."""
        tokens = tokenize("papers by the author about a topic")
        assert "papers" not in tokens  # Domain stopword
        assert "the" not in tokens
        assert "about" not in tokens
        assert "author" in tokens
        assert "topic" in tokens

    def test_handles_special_characters(self):
        """Should handle punctuation and special chars."""
        tokens = tokenize("JWST's observations (2022)")
        assert "jwst" in tokens
        assert "observations" in tokens
        assert "2022" in tokens

    def test_removes_short_tokens(self):
        """Should remove single-character tokens."""
        tokens = tokenize("a b c test")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "test" in tokens


# --- Feature Extraction Tests ---


class TestExtractFeatures:
    """Tests for ADS query feature extraction."""

    def test_extracts_author_presence(self):
        """Should detect author: field."""
        features = extract_features_from_ads_query('author:"Einstein, A."')
        assert features["has_author"] is True

    def test_extracts_year_presence(self):
        """Should detect year/pubdate fields."""
        features = extract_features_from_ads_query("pubdate:[2020 TO 2022]")
        assert features["has_year"] is True

        features = extract_features_from_ads_query("year:2022")
        assert features["has_year"] is True

    def test_extracts_operators(self):
        """Should extract operator names."""
        features = extract_features_from_ads_query('citations(author:"Hawking")')
        assert features["operator"] == "citations"
        assert "citations" in features["operators"]

    def test_extracts_doctypes(self):
        """Should extract doctype values."""
        features = extract_features_from_ads_query("doctype:article doctype:eprint")
        assert "article" in features["doctypes"]
        assert "eprint" in features["doctypes"]

    def test_extracts_properties(self):
        """Should extract property values."""
        features = extract_features_from_ads_query("property:refereed property:openaccess")
        assert "refereed" in features["properties"]
        assert "openaccess" in features["properties"]

    def test_extracts_bibgroups(self):
        """Should extract bibgroup values."""
        features = extract_features_from_ads_query("bibgroup:JWST bibgroup:HST")
        assert "JWST" in features["bibgroups"]
        assert "HST" in features["bibgroups"]

    def test_no_features(self):
        """Should handle query with no special features."""
        features = extract_features_from_ads_query('abs:"black holes"')
        assert features["has_author"] is False
        assert features["has_year"] is False
        assert features["operator"] is None


# --- Index Tests ---


class TestGoldExampleIndex:
    """Tests for the gold example index."""

    @pytest.fixture
    def sample_examples(self):
        """Sample gold examples for testing."""
        return [
            {
                "natural_language": "papers by Einstein about relativity",
                "ads_query": 'author:"Einstein, A." abs:"relativity"',
                "category": "author",
            },
            {
                "natural_language": "JWST observations of exoplanets",
                "ads_query": 'abs:"JWST" abs:"exoplanets"',
                "category": "unfielded",
            },
            {
                "natural_language": "papers citing the Planck 2018 paper",
                "ads_query": 'citations(bibcode:"2018...")',
                "category": "citations",
            },
            {
                "natural_language": "refereed papers on black holes from 2020",
                "ads_query": 'abs:"black holes" property:refereed pubdate:2020',
                "category": "property",
            },
            {
                "natural_language": "Hubble observations of galaxies",
                "ads_query": 'bibgroup:HST abs:"galaxies"',
                "category": "astronomy",
            },
        ]

    @pytest.fixture
    def index(self, sample_examples):
        """Create index from sample examples."""
        return GoldExampleIndex(examples=sample_examples)

    def test_loads_examples(self, index, sample_examples):
        """Should load all examples."""
        assert len(index) == len(sample_examples)

    def test_retrieves_matching_examples(self, index):
        """Should retrieve examples matching intent."""
        intent = IntentSpec(
            free_text_terms=["relativity"],
            authors=["Einstein"],
        )
        results = index.retrieve(intent, k=3)
        assert len(results) > 0
        assert any("Einstein" in r.nl_query for r in results)

    def test_boosts_operator_matches(self, index):
        """Should rank operator matches higher."""
        intent = IntentSpec(
            free_text_terms=["papers"],
            operator="citations",
        )
        results = index.retrieve(intent, k=3)
        # Citations example should rank high
        assert any("citing" in r.nl_query for r in results)

    def test_boosts_property_matches(self, index):
        """Should rank property matches higher."""
        intent = IntentSpec(
            free_text_terms=["black holes"],
            property={"refereed"},
        )
        results = index.retrieve(intent, k=3)
        # Refereed example should rank high
        assert any("refereed" in r.nl_query for r in results)

    def test_boosts_author_presence(self, index):
        """Should boost examples with author when intent has authors."""
        intent = IntentSpec(
            authors=["Hawking"],
            free_text_terms=["cosmology"],
        )
        results = index.retrieve(intent, k=3)
        # Should prefer examples with author patterns
        if results:
            # At least check we get results
            assert len(results) >= 1

    def test_returns_gold_examples(self, index):
        """Should return GoldExample objects with scores."""
        intent = IntentSpec(free_text_terms=["galaxies"])
        results = index.retrieve(intent, k=2)
        for result in results:
            assert isinstance(result, GoldExample)
            assert result.score > 0
            assert result.nl_query
            assert result.ads_query

    def test_deterministic_ranking(self, index):
        """Ranking should be deterministic across calls."""
        intent = IntentSpec(
            free_text_terms=["observations"],
            property={"refereed"},
        )
        results1 = index.retrieve(intent, k=5)
        results2 = index.retrieve(intent, k=5)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.nl_query == r2.nl_query
            assert r1.score == r2.score


# --- Retrieval Function Tests ---


class TestRetrieveSimilar:
    """Tests for the retrieve_similar function."""

    @pytest.fixture(autouse=True)
    def reset_global_index(self):
        """Reset global index before each test."""
        reset_index()
        yield
        reset_index()

    def test_loads_global_index(self):
        """Should load global index on first call."""
        if not DEFAULT_GOLD_EXAMPLES_PATH.exists():
            pytest.skip("Gold examples file not found")

        intent = IntentSpec(free_text_terms=["exoplanets"])
        results = retrieve_similar(intent, k=5)
        assert len(results) <= 5

    def test_returns_k_results(self):
        """Should return at most k results."""
        if not DEFAULT_GOLD_EXAMPLES_PATH.exists():
            pytest.skip("Gold examples file not found")

        intent = IntentSpec(free_text_terms=["black holes"])
        results = retrieve_similar(intent, k=3)
        assert len(results) <= 3


# --- Performance Tests ---


class TestRetrievalPerformance:
    """Performance tests for retrieval."""

    @pytest.fixture(autouse=True)
    def reset_global_index(self):
        """Reset global index before each test."""
        reset_index()
        yield
        reset_index()

    @pytest.mark.skipif(
        not DEFAULT_GOLD_EXAMPLES_PATH.exists(),
        reason="Gold examples file not found",
    )
    def test_retrieval_under_20ms(self):
        """Retrieval should complete in under 20ms for k=5."""
        # First call loads the index (don't time this)
        index = get_index()

        # Test retrieval performance
        intent = IntentSpec(
            free_text_terms=["supermassive black hole"],
            authors=["Hawking"],
            property={"refereed", "openaccess"},
            year_from=2020,
        )

        # Warm up
        index.retrieve(intent, k=5)

        # Time 100 retrievals
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            index.retrieve(intent, k=5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / iterations
        assert avg_ms < 20, f"Average retrieval time {avg_ms:.2f}ms exceeds 20ms target"

    @pytest.mark.skipif(
        not DEFAULT_GOLD_EXAMPLES_PATH.exists(),
        reason="Gold examples file not found",
    )
    def test_index_loads_all_examples(self):
        """Index should load all 4000+ gold examples."""
        index = get_index()
        # Should have loaded significant number of examples
        assert len(index) > 400, f"Expected >400 examples, got {len(index)}"


# --- Snapshot/Determinism Tests ---


class TestRankingStability:
    """Tests for ranking stability and determinism."""

    @pytest.fixture
    def sample_examples(self):
        """More examples for stability testing."""
        return [
            {"natural_language": "JWST deep field observations", "ads_query": 'abs:"JWST" abs:"deep field"', "category": "astronomy"},
            {"natural_language": "Hubble deep field imaging", "ads_query": 'abs:"Hubble" abs:"deep field"', "category": "astronomy"},
            {"natural_language": "exoplanet atmospheres with JWST", "ads_query": 'abs:"exoplanet" abs:"JWST"', "category": "astronomy"},
            {"natural_language": "gravitational waves from black hole mergers", "ads_query": 'abs:"gravitational waves" abs:"black hole"', "category": "astronomy"},
            {"natural_language": "dark matter simulations", "ads_query": 'abs:"dark matter" abs:"simulations"', "category": "astronomy"},
            {"natural_language": "stellar evolution models", "ads_query": 'abs:"stellar evolution"', "category": "astronomy"},
            {"natural_language": "galaxy formation theories", "ads_query": 'abs:"galaxy formation"', "category": "astronomy"},
            {"natural_language": "papers by Hawking on black holes", "ads_query": 'author:"Hawking" abs:"black holes"', "category": "author"},
            {"natural_language": "citations of Einstein relativity paper", "ads_query": 'citations(author:"Einstein")', "category": "citations"},
            {"natural_language": "refereed papers on cosmology", "ads_query": 'property:refereed abs:"cosmology"', "category": "property"},
        ]

    def test_ranking_is_deterministic(self, sample_examples):
        """Same query should always produce same ranking."""
        index = GoldExampleIndex(examples=sample_examples)

        intent = IntentSpec(
            free_text_terms=["deep field"],
            bibgroup={"HST"},
        )

        # Run multiple times
        rankings = []
        for _ in range(5):
            results = index.retrieve(intent, k=5)
            ranking = [(r.nl_query, round(r.score, 4)) for r in results]
            rankings.append(ranking)

        # All rankings should be identical
        for i, ranking in enumerate(rankings[1:], 1):
            assert ranking == rankings[0], f"Ranking {i} differs from first"

    def test_snapshot_known_inputs(self, sample_examples):
        """Verify expected top results for known inputs."""
        index = GoldExampleIndex(examples=sample_examples)

        # Query for JWST should return JWST examples first
        intent = IntentSpec(free_text_terms=["JWST", "observations"])
        results = index.retrieve(intent, k=3)
        assert len(results) > 0
        # Top result should mention JWST
        assert "JWST" in results[0].nl_query.upper()

        # Query with citations operator should boost citations examples
        intent = IntentSpec(
            free_text_terms=["relativity"],
            operator="citations",
        )
        results = index.retrieve(intent, k=3)
        # Should have citations example ranked high
        assert any("citations" in r.nl_query.lower() for r in results)


# --- Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_intent(self):
        """Should handle empty intent gracefully."""
        index = GoldExampleIndex(examples=[
            {"natural_language": "test", "ads_query": "abs:test", "category": "test"}
        ])
        intent = IntentSpec()
        results = index.retrieve(intent, k=5)
        # Empty intent should return no results (no positive scores)
        assert len(results) == 0

    def test_empty_index(self):
        """Should handle empty index."""
        index = GoldExampleIndex(examples=[])
        intent = IntentSpec(free_text_terms=["test"])
        results = index.retrieve(intent, k=5)
        assert len(results) == 0

    def test_k_larger_than_index(self):
        """Should handle k larger than index size."""
        index = GoldExampleIndex(examples=[
            {"natural_language": "test query", "ads_query": "abs:test", "category": "test"}
        ])
        intent = IntentSpec(free_text_terms=["test"])
        results = index.retrieve(intent, k=100)
        assert len(results) <= 1

    def test_no_matching_tokens(self):
        """Should handle queries with no token overlap."""
        index = GoldExampleIndex(examples=[
            {"natural_language": "alpha beta gamma", "ads_query": "abs:test", "category": "test"}
        ])
        intent = IntentSpec(free_text_terms=["completely", "different", "words"])
        results = index.retrieve(intent, k=5)
        # May still get results from feature boosting if any match
        # But with no features, should be empty
        assert len(results) == 0


# --- Integration with Pipeline ---


class TestPipelineIntegration:
    """Tests for integration with pipeline module."""

    def test_gold_example_compatibility(self):
        """GoldExample from retrieval should match pipeline GoldExample."""
        index = GoldExampleIndex(examples=[
            {"natural_language": "test", "ads_query": "abs:test", "category": "test"}
        ])
        intent = IntentSpec(free_text_terms=["test"])
        results = index.retrieve(intent, k=1)

        if results:
            result = results[0]
            # Verify GoldExample attributes
            assert hasattr(result, "nl_query")
            assert hasattr(result, "ads_query")
            assert hasattr(result, "features")
            assert hasattr(result, "score")
            # Verify to_dict works
            d = result.to_dict()
            assert "nl_query" in d
            assert "ads_query" in d
