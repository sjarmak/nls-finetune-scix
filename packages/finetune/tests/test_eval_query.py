"""Tests for the ADS-native evaluate_query (result-set overlap semantics)."""

import pytest

import finetune.eval.eval as eval_module
from finetune.eval.eval import SEMANTIC_MATCH_THRESHOLD, evaluate_query


class TestSyntaxValidation:
    def test_invalid_syntax_fails_without_api_call(self):
        result = evaluate_query('abs:"dark matter"', 'abs:"unbalanced')
        assert not result.valid
        assert not result.match
        assert result.overlap == 0.0

    def test_error_output_fails(self):
        result = evaluate_query('abs:"dark matter"', "ERROR: connection refused")
        assert not result.match


class TestIdenticalFastPath:
    def test_identical_queries_match_without_api(self, monkeypatch):
        monkeypatch.delenv("ADS_API_KEY", raising=False)
        result = evaluate_query('abs:"dark matter"', 'abs:"dark matter"')
        assert result.valid
        assert result.match
        assert result.overlap == 1.0

    def test_whitespace_and_case_normalized(self, monkeypatch):
        monkeypatch.delenv("ADS_API_KEY", raising=False)
        result = evaluate_query('abs:"Dark Matter"', '  abs:"dark   matter" ')
        assert result.match


class TestMissingApiKey:
    def test_differing_queries_require_api_key(self, monkeypatch):
        monkeypatch.delenv("ADS_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ADS_API_KEY"):
            evaluate_query('abs:"dark matter"', 'abs:"dark energy"')


class TestResultSetOverlap:
    def _patch_bibcodes(self, monkeypatch, mapping):
        def fake_fetch(query, n=50, api_key=None, **kwargs):
            return mapping[query]

        monkeypatch.setattr(eval_module, "fetch_bibcodes", fake_fetch)

    def test_high_overlap_matches(self, monkeypatch):
        self._patch_bibcodes(
            monkeypatch,
            {
                'abs:"dark matter"': ["a", "b", "c", "d"],
                'abs:"dark matter halos"': ["a", "b", "c"],
            },
        )
        result = evaluate_query(
            'abs:"dark matter"', 'abs:"dark matter halos"', api_key="test-key"
        )
        assert result.valid
        assert result.overlap == 0.75
        assert result.match  # 0.75 >= threshold

    def test_low_overlap_does_not_match(self, monkeypatch):
        self._patch_bibcodes(
            monkeypatch,
            {
                'abs:"dark matter"': ["a", "b", "c", "d"],
                'abs:"exoplanets"': ["x", "y", "z"],
            },
        )
        result = evaluate_query('abs:"dark matter"', 'abs:"exoplanets"', api_key="test-key")
        assert result.valid
        assert result.overlap == 0.0
        assert not result.match

    def test_threshold_boundary(self, monkeypatch):
        # Jaccard exactly at threshold counts as a match
        self._patch_bibcodes(
            monkeypatch,
            {
                'abs:"a b"': ["1", "2", "3", "4"],
                'abs:"a c"': ["1", "2", "5", "6"],
            },
        )
        result = evaluate_query('abs:"a b"', 'abs:"a c"', api_key="test-key")
        assert result.overlap == pytest.approx(2 / 6)
        assert result.match == (result.overlap >= SEMANTIC_MATCH_THRESHOLD)
