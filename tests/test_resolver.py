"""Tests for the optional LLM resolver module.

These tests verify:
1. Gating logic - LLM is NOT called for normal queries
2. Ambiguous reference detection
3. Hint extraction from user text
4. ADS search resolution (mocked)
5. Timeout handling
6. Fallback behavior
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from finetune.domains.scix.resolver import (
    AMBIGUOUS_REFERENCE_PATTERNS,
    OPERATORS_REQUIRING_TARGET,
    ResolverResult,
    extract_paper_hint,
    needs_resolution,
    resolve_paper_reference,
    resolve_via_ads_search,
    resolve_via_llm,
)


class TestNeedsResolution:
    """Tests for the needs_resolution gating function."""

    def test_no_operator_no_resolution(self):
        """No operator means no resolution needed."""
        assert needs_resolution(None, None, "papers about exoplanets") is False

    def test_non_target_operator_no_resolution(self):
        """Operators like trending don't require target resolution."""
        assert needs_resolution("trending", None, "trending papers about cosmology") is False
        assert needs_resolution("useful", None, "useful papers on dark matter") is False
        assert needs_resolution("reviews", None, "reviews of black hole formation") is False

    def test_target_already_provided_no_resolution(self):
        """If target is explicitly provided, no resolution needed."""
        assert needs_resolution("citations", "2018A&A...641A...1P", "citations of this paper") is False

    def test_citations_with_ambiguous_reference_needs_resolution(self):
        """citations + 'this paper' needs resolution."""
        assert needs_resolution("citations", None, "papers citing this paper") is True

    def test_references_with_famous_paper_needs_resolution(self):
        """references + 'famous paper' needs resolution."""
        assert needs_resolution("references", None, "references of the famous paper about CMB") is True

    def test_similar_with_that_paper_needs_resolution(self):
        """similar + 'that paper' needs resolution."""
        assert needs_resolution("similar", None, "papers similar to that paper") is True

    def test_operator_without_ambiguous_reference_no_resolution(self):
        """Operator present but no ambiguous reference - no resolution."""
        assert needs_resolution("citations", None, "papers citing exoplanet studies") is False

    @pytest.mark.parametrize("pattern_phrase", [
        "this paper",
        "that paper",
        "the paper",
        "famous paper",
        "groundbreaking study",
        "seminal work",
        "landmark paper",
        "original study",
        "classic paper",
        "pioneering work",
        "influential paper",
    ])
    def test_all_ambiguous_patterns_detected(self, pattern_phrase):
        """All defined ambiguous patterns trigger resolution."""
        text = f"papers citing the {pattern_phrase} about cosmology"
        assert needs_resolution("citations", None, text) is True


class TestExtractPaperHint:
    """Tests for hint extraction from user text."""

    def test_extract_about_topic(self):
        """Extract topic from 'paper about X'."""
        hint = extract_paper_hint("the famous paper about cosmic microwave background")
        assert hint is not None
        assert "cosmic microwave background" in hint

    def test_extract_author_possessive(self):
        """Extract author from 'Einstein's famous paper'."""
        hint = extract_paper_hint("Einstein's famous paper on relativity")
        assert hint is not None
        assert "einstein" in hint.lower()

    def test_extract_year(self):
        """Extract year from '2018 paper'."""
        hint = extract_paper_hint("the 2018 paper on Planck results")
        assert hint is not None
        assert "year:2018" in hint

    def test_no_hint_available(self):
        """Return None when no hint can be extracted."""
        hint = extract_paper_hint("papers citing something")
        assert hint is None

    def test_multiple_hints_combined(self):
        """Multiple hints are combined."""
        hint = extract_paper_hint("Hawking's famous paper about black hole radiation from Nature")
        assert hint is not None
        parts = hint.lower()
        assert "hawking" in parts or "black hole" in parts


class TestResolverResult:
    """Tests for ResolverResult dataclass."""

    def test_success_result_to_dict(self):
        """Successful result serializes correctly."""
        result = ResolverResult(
            success=True,
            bibcode="2018A&A...641A...1P",
            paper_title="Planck 2018 results",
            resolution_time_ms=50.0,
            used_llm=False,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["bibcode"] == "2018A&A...641A...1P"
        assert d["paper_title"] == "Planck 2018 results"
        assert d["fallback_reason"] is None

    def test_failure_result_to_dict(self):
        """Failed result includes fallback reason."""
        result = ResolverResult(
            success=False,
            fallback_reason="No papers found",
            resolution_time_ms=25.0,
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["fallback_reason"] == "No papers found"
        assert d["bibcode"] is None


class TestResolveViaAdsSearch:
    """Tests for ADS search resolution (mocked)."""

    @patch("finetune.domains.scix.resolver.httpx.get")
    def test_successful_search(self, mock_get):
        """ADS search returns most-cited paper."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {
                "docs": [
                    {
                        "bibcode": "2018A&A...641A...1P",
                        "title": ["Planck 2018 results. I. Overview"],
                        "citation_count": 5000,
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"ADS_API_KEY": "test-key"}):
            result = resolve_via_ads_search("Planck cosmology")

        assert result.success is True
        assert result.bibcode == "2018A&A...641A...1P"
        assert "Planck 2018" in result.paper_title

    @patch("finetune.domains.scix.resolver.httpx.get")
    def test_no_results_found(self, mock_get):
        """ADS search with no results returns fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": {"docs": []}}
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"ADS_API_KEY": "test-key"}):
            result = resolve_via_ads_search("nonexistent topic xyz123")

        assert result.success is False
        assert "No papers found" in result.fallback_reason

    @patch("finetune.domains.scix.resolver.httpx.get")
    def test_api_error_handled(self, mock_get):
        """ADS API errors are handled gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"ADS_API_KEY": "test-key"}):
            result = resolve_via_ads_search("cosmology")

        assert result.success is False
        assert "HTTP 500" in result.fallback_reason

    def test_no_api_key(self):
        """Missing API key returns fallback."""
        with patch.dict("os.environ", {}, clear=True):
            result = resolve_via_ads_search("cosmology")

        assert result.success is False
        assert "ADS_API_KEY not set" in result.fallback_reason

    @patch("finetune.domains.scix.resolver.httpx.get")
    def test_timeout_handled(self, mock_get):
        """Timeout is handled gracefully."""
        import httpx
        mock_get.side_effect = httpx.TimeoutException("Connection timed out")

        with patch.dict("os.environ", {"ADS_API_KEY": "test-key"}):
            result = resolve_via_ads_search("cosmology")

        assert result.success is False
        assert "timeout" in result.fallback_reason.lower()


class TestResolveViaLlm:
    """Tests for LLM resolution (mocked)."""

    @patch("finetune.domains.scix.resolver.httpx.post")
    def test_successful_llm_response(self, mock_post):
        """LLM returns structured JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"paper_title_guess": "Planck 2018 results", "bibcode_guess": "2018A&A...641A...1P"}'
                }
            }]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = resolve_via_llm("the famous Planck paper")

        assert result is not None
        assert result["paper_title_guess"] == "Planck 2018 results"
        assert result["bibcode_guess"] == "2018A&A...641A...1P"

    @patch("finetune.domains.scix.resolver.httpx.post")
    def test_llm_no_bibcode_guess(self, mock_post):
        """LLM may return only title guess."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"paper_title_guess": "A paper about exoplanets", "bibcode_guess": null}'
                }
            }]
        }
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = resolve_via_llm("that famous exoplanet paper")

        assert result is not None
        assert result["paper_title_guess"] == "A paper about exoplanets"
        assert result["bibcode_guess"] is None

    def test_no_openai_key(self):
        """Missing OpenAI key returns None."""
        with patch.dict("os.environ", {}, clear=True):
            result = resolve_via_llm("the famous paper")

        assert result is None

    @patch("finetune.domains.scix.resolver.httpx.post")
    def test_llm_timeout_returns_none(self, mock_post):
        """LLM timeout returns None."""
        import httpx
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = resolve_via_llm("the famous paper", timeout_ms=100)

        assert result is None


class TestResolvePaperReference:
    """Integration tests for the main resolve_paper_reference function."""

    def test_no_resolution_needed_fast_return(self):
        """When resolution not needed, returns immediately."""
        result = resolve_paper_reference(
            raw_text="papers about exoplanets",
            operator=None,
        )
        assert result.success is False
        assert result.fallback_reason == "Resolution not needed"
        assert result.resolution_time_ms < 10

    def test_operator_without_ambiguous_ref_no_resolution(self):
        """Operator present but no ambiguous reference - fast return."""
        result = resolve_paper_reference(
            raw_text="papers citing exoplanet studies",
            operator="citations",
        )
        assert result.success is False
        assert result.fallback_reason == "Resolution not needed"

    @patch("finetune.domains.scix.resolver.resolve_via_ads_search")
    def test_hint_extraction_triggers_ads_search(self, mock_ads):
        """When hint is extractable, ADS search is tried first."""
        mock_ads.return_value = ResolverResult(
            success=True,
            bibcode="2018A&A...641A...1P",
            paper_title="Planck 2018",
        )

        result = resolve_paper_reference(
            raw_text="papers citing the famous paper about cosmic microwave background",
            operator="citations",
            use_llm=False,
        )

        assert result.success is True
        assert mock_ads.called

    @patch("finetune.domains.scix.resolver.resolve_via_llm")
    @patch("finetune.domains.scix.resolver.resolve_via_ads_search")
    def test_llm_fallback_when_hint_fails(self, mock_ads, mock_llm):
        """LLM is used when ADS search from hint fails."""
        mock_ads.return_value = ResolverResult(
            success=False,
            fallback_reason="No papers found",
        )
        mock_llm.return_value = {
            "paper_title_guess": "Some paper",
            "bibcode_guess": "2020ApJ...test",
        }

        result = resolve_paper_reference(
            raw_text="papers citing that famous paper",
            operator="citations",
            use_llm=True,
        )

        assert result.success is True
        assert result.bibcode == "2020ApJ...test"
        assert result.used_llm is True

    @patch("finetune.domains.scix.resolver.resolve_via_llm")
    @patch("finetune.domains.scix.resolver.resolve_via_ads_search")
    def test_llm_title_guess_triggers_second_ads_search(self, mock_ads, mock_llm):
        """LLM title guess (no bibcode) triggers ADS search."""
        mock_ads.side_effect = [
            ResolverResult(success=False, fallback_reason="No papers found"),
            ResolverResult(success=True, bibcode="2018test", paper_title="Test Paper"),
        ]
        mock_llm.return_value = {
            "paper_title_guess": "Test Paper Title",
            "bibcode_guess": None,
        }

        result = resolve_paper_reference(
            raw_text="papers citing that famous paper",
            operator="citations",
            use_llm=True,
        )

        assert result.success is True
        assert result.used_llm is True
        assert mock_ads.call_count == 2

    def test_use_llm_false_skips_llm(self):
        """use_llm=False skips LLM entirely."""
        with patch("finetune.domains.scix.resolver.resolve_via_llm") as mock_llm:
            result = resolve_paper_reference(
                raw_text="papers citing that famous paper",
                operator="citations",
                use_llm=False,
            )

            mock_llm.assert_not_called()
            assert result.success is False


class TestGatingLogicForNormalQueries:
    """Critical tests: normal queries must NOT trigger LLM."""

    @pytest.mark.parametrize("normal_query", [
        "papers about exoplanets",
        "refereed papers on dark matter",
        "JWST observations from 2022",
        "author:Smith gravitational waves",
        "trending papers on machine learning",
        "reviews of black hole formation",
        "papers by Hawking on radiation",
        "useful papers about spectroscopy",
    ])
    def test_normal_queries_no_llm(self, normal_query):
        """Normal queries never trigger LLM resolution."""
        with patch("finetune.domains.scix.resolver.resolve_via_llm") as mock_llm:
            resolve_paper_reference(raw_text=normal_query, operator=None)
            mock_llm.assert_not_called()

    @pytest.mark.parametrize("operator", ["trending", "useful", "reviews"])
    def test_non_target_operators_no_llm(self, operator):
        """Operators that don't need targets never trigger LLM."""
        with patch("finetune.domains.scix.resolver.resolve_via_llm") as mock_llm:
            resolve_paper_reference(
                raw_text=f"{operator} papers on cosmology",
                operator=operator,
            )
            mock_llm.assert_not_called()


class TestIntegrationWithMockedServices:
    """Full integration tests with mocked external services."""

    @patch("finetune.domains.scix.resolver.httpx.post")
    @patch("finetune.domains.scix.resolver.httpx.get")
    def test_full_resolution_flow(self, mock_get, mock_post):
        """Full flow: ADS fails, LLM succeeds with title, ADS search finds it."""
        mock_get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value={"response": {"docs": []}})),
            MagicMock(status_code=200, json=MagicMock(return_value={
                "response": {
                    "docs": [{
                        "bibcode": "2018A&A...641A...1P",
                        "title": ["Planck 2018 results"],
                        "citation_count": 5000,
                    }]
                }
            })),
        ]
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{
                    "message": {
                        "content": '{"paper_title_guess": "Planck 2018 cosmology paper", "bibcode_guess": null}'
                    }
                }]
            })
        )

        with patch.dict("os.environ", {"ADS_API_KEY": "test", "OPENAI_API_KEY": "test"}):
            result = resolve_paper_reference(
                raw_text="papers citing that famous paper about cosmic microwave background",
                operator="citations",
            )

        assert result.success is True
        assert result.bibcode == "2018A&A...641A...1P"
        assert result.used_llm is True
