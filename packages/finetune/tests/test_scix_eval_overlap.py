"""Tests for result-set overlap scoring in finetune.domains.scix.eval."""

import httpx
import pytest

import finetune.domains.scix.eval as scix_eval
from finetune.domains.scix.eval import (
    GOLD_EMPTY,
    BibcodeFetchError,
    EvalResult,
    compute_overlap_metrics,
    evaluate_pair,
    fetch_bibcodes,
    summarize_results,
)
from finetune.domains.scix.validate import ValidationResult


def _patch(monkeypatch, bibcodes: dict[str, list[str] | Exception], valid: bool = True):
    def fake_fetch(query, n=50, api_key=None, **kwargs):
        value = bibcodes[query]
        if isinstance(value, Exception):
            raise value
        return value

    def fake_validate(query, api_key=None):
        return ValidationResult(valid=valid, errors=[] if valid else ["bad"], warnings=[])

    monkeypatch.setattr(scix_eval, "fetch_bibcodes", fake_fetch)
    monkeypatch.setattr(scix_eval, "validate_query", fake_validate)


def _result(jaccard: float, valid: bool = True, reason: str | None = None) -> EvalResult:
    return EvalResult(
        nl="q",
        expected_query="gold",
        generated_query="gen",
        syntactically_valid=valid,
        syntax_errors=[],
        expected_bibcodes=[],
        generated_bibcodes=[],
        jaccard_overlap=jaccard,
        precision_at_n=jaccard,
        recall_at_n=jaccard,
        category="c",
        unscorable_reason=reason,
    )


class TestOverlapMetrics:
    def test_two_empty_sets_are_not_a_match(self):
        assert compute_overlap_metrics([], []) == (0.0, 0.0, 0.0)

    def test_partial_overlap(self):
        jaccard, precision, recall = compute_overlap_metrics(["a", "b"], ["b", "c"])
        assert jaccard == pytest.approx(1 / 3)
        assert precision == 0.5
        assert recall == 0.5


class TestEvaluatePair:
    def test_empty_gold_is_unscorable_not_a_match(self, monkeypatch):
        # "top 10 papers on dark matter": gold topn() returned nothing and the
        # garbled generated query returned nothing too; this used to score 1.0.
        _patch(monkeypatch, {"gold": [], "gen": []})
        result = evaluate_pair("q", "gold", "gen")
        assert result.unscorable_reason == GOLD_EMPTY
        assert result.jaccard_overlap == 0.0

    def test_empty_gold_is_unscorable_for_invalid_generated_query_too(self, monkeypatch):
        _patch(monkeypatch, {"gold": []}, valid=False)
        result = evaluate_pair("q", "gold", "gen")
        assert result.unscorable_reason == GOLD_EMPTY
        assert not result.syntactically_valid

    def test_invalid_generated_query_on_scorable_item_is_a_miss(self, monkeypatch):
        _patch(monkeypatch, {"gold": ["a"]}, valid=False)
        result = evaluate_pair("q", "gold", "gen")
        assert result.unscorable_reason is None
        assert result.syntax_errors == ["bad"]
        assert result.jaccard_overlap == 0.0

    def test_ads_failure_on_generated_query_is_unscorable(self, monkeypatch):
        _patch(monkeypatch, {"gold": ["a"], "gen": BibcodeFetchError("ADS returned HTTP 504")})
        result = evaluate_pair("q", "gold", "gen")
        assert result.unscorable_reason == "generated: ADS returned HTTP 504"

    def test_ads_failure_on_gold_is_unscorable(self, monkeypatch):
        _patch(monkeypatch, {"gold": BibcodeFetchError("ADS request failed: timeout")})
        result = evaluate_pair("q", "gold", "gen")
        assert result.unscorable_reason == "gold: ADS request failed: timeout"

    def test_scored_pair(self, monkeypatch):
        _patch(monkeypatch, {"gold": ["a", "b"], "gen": ["a", "b"]})
        result = evaluate_pair("q", "gold", "gen", category="c")
        assert result.unscorable_reason is None
        assert result.jaccard_overlap == 1.0
        assert result.category == "c"


class TestSummarize:
    def test_unscorable_items_are_excluded_from_means(self):
        results = [_result(1.0), _result(0.0), _result(0.0, reason=GOLD_EMPTY)]
        summary = summarize_results(results)
        assert summary.total == 3
        assert summary.unscorable == 1
        assert summary.mean_jaccard == 0.5
        assert summary.by_category["c"]["unscorable"] == 1
        assert summary.by_category["c"]["mean_jaccard"] == 0.5

    def test_empty_results(self):
        summary = summarize_results([])
        assert summary.total == 0
        assert summary.by_category == {}


class TestFetchBibcodes:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ADS_API_KEY", raising=False)
        with pytest.raises(BibcodeFetchError, match="ADS_API_KEY"):
            fetch_bibcodes("abs:x")

    def test_non_200_raises(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(429))
        with pytest.raises(BibcodeFetchError, match="HTTP 429"):
            fetch_bibcodes("abs:x", api_key="k")

    def test_transport_error_raises(self, monkeypatch):
        def boom(*a, **k):
            raise httpx.ReadTimeout("timed out")

        monkeypatch.setattr(httpx, "get", boom)
        with pytest.raises(BibcodeFetchError, match="timed out"):
            fetch_bibcodes("abs:x", api_key="k")

    def test_malformed_body_raises(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={}))
        with pytest.raises(BibcodeFetchError, match="malformed"):
            fetch_bibcodes("abs:x", api_key="k")

    def test_empty_result_is_an_empty_list(self, monkeypatch):
        body = {"response": {"docs": []}}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json=body))
        assert fetch_bibcodes("abs:x", api_key="k") == []
