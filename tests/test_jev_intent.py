"""Tests for the Jev (TypeSafe System One) typed-classifier intent backend.

The client is exercised through an httpx MockTransport, so no network is
touched. Wire-contract fixtures mirror the smoke call recorded on 2026-09-24
against jev-1.13.0.
"""

import json

import httpx
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from jev_fixtures import choice_answer, jev_payload, mock_jev_client, noul_answer

from finetune.domains.scix.field_constraints import BIBGROUPS, COLLECTIONS, DOCTYPES, PROPERTIES
from finetune.domains.scix.intent_spec import OPERATORS, IntentSpec
from finetune.domains.scix.jev_intent import (
    BOOLEAN_QUESTION_IDS,
    CHOICE_QUESTION_IDS,
    JEV_MODEL,
    PROPERTY_BOOLEANS,
    JevAnswers,
    JevClient,
    JevResponseError,
    apply_answers,
    build_questions,
    build_request,
    extract_intent_jev,
    parse_response,
    request_fingerprint,
)


class TestQuestionSet:
    def test_choice_options_are_exactly_the_legal_enum_values(self):
        q = build_questions()
        assert set(q["operator"]["criteria"]) == {"none", *OPERATORS}
        assert set(q["doctype"]["criteria"]) == {"none", *DOCTYPES}
        assert set(q["bibgroup"]["criteria"]) == {"none", *BIBGROUPS}
        assert set(q["collection"]["criteria"]) == {"none", *COLLECTIONS}

    def test_boolean_questions_use_noul_wire_type(self):
        q = build_questions()
        for qid in BOOLEAN_QUESTION_IDS:
            assert q[qid]["type"] == "noul"
            assert set(q[qid]["criteria"]) == {"true", "false"}
        for qid in CHOICE_QUESTION_IDS:
            assert q[qid]["type"] == "choice"
            assert q[qid]["instructions"]

    def test_reviews_operator_leaves_book_reviews_to_the_doctype(self):
        q = build_questions()
        reviews = q["operator"]["criteria"]["reviews"].lower()
        assert "book review" in reviews
        assert "bookreview" in reviews
        bookreview = q["doctype"]["criteria"]["bookreview"].lower()
        assert "book" in bookreview
        assert "not a review article" in bookreview

    def test_reviews_operator_is_not_an_observational_sky_survey(self):
        reviews = build_questions()["operator"]["criteria"]["reviews"].lower()
        assert "literature surveys" in reviews
        assert "not an observational sky survey" in reviews

    def test_a_catalog_as_topic_is_not_the_catalog_doctype(self):
        catalog = build_questions()["doctype"]["criteria"]["catalog"].lower()
        assert "not papers that present" in catalog
        assert "is a topic" in catalog
        doctype = build_questions()["doctype"]["instructions"]
        assert "catalogs in the astronomy database" in doctype

    def test_bare_keywords_with_a_name_are_a_plain_search(self):
        citations = build_questions()["operator"]["criteria"]["citations"].lower()
        assert "pairs a surname with topic words" in citations
        assert "the word citation used as a topic" in citations

    def test_every_option_has_a_description(self):
        for qid, q in build_questions().items():
            for option, description in q["criteria"].items():
                assert description.strip(), f"{qid}:{option} has no description"


class TestRequest:
    def test_request_pins_model_and_carries_state(self):
        req = build_request("papers citing Planck 2018")
        assert req["model"] == JEV_MODEL
        assert req["state"] == {"query": "papers citing Planck 2018"}
        extraction = {"recency", "first_author", "highly_cited", "ranking"}
        assert set(req["questions"]) == set(build_questions()) | extraction

    def test_context_is_merged_into_state_without_overriding_query(self):
        req = build_request("x", context={"regex_intent": {"operator": None}})
        assert req["state"] == {"query": "x", "regex_intent": {"operator": None}}
        with pytest.raises(ValueError):
            build_request("x", context={"query": "y"})

    def test_fingerprint_is_stable_under_key_order(self):
        a = {"model": "m", "state": {"query": "q", "z": 1}, "questions": {"b": 1, "a": 2}}
        b = {"questions": {"a": 2, "b": 1}, "state": {"z": 1, "query": "q"}, "model": "m"}
        assert request_fingerprint(a) == request_fingerprint(b)
        assert request_fingerprint(a) != request_fingerprint({**a, "model": "other"})


class TestParseResponse:
    def test_parses_all_fields(self):
        answers = parse_response(jev_payload("citations"), latency_ms=123.0, cached=False)
        assert answers.model == JEV_MODEL
        assert answers.choices["operator"].choice == "citations"
        assert answers.choices["operator"].probabilities["none"] == pytest.approx(0.01)
        assert answers.booleans["refereed"] == pytest.approx(0.1)
        assert answers.input_tokens == 500
        assert answers.latency_ms == 123.0
        assert answers.cached is False

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda p: p["answers"].pop("operator"),
            lambda p: p["answers"]["operator"].update(choice="topn"),
            lambda p: p["answers"]["refereed"].update(noul=1.5),
            lambda p: p["answers"]["refereed"].update(type="choice"),
            lambda p: p.pop("usage"),
            lambda p: p.update(model=""),
            lambda p: p["answers"]["operator"].pop("probabilities"),
        ],
    )
    def test_rejects_malformed_payloads(self, mutation):
        payload = jev_payload()
        mutation(payload)
        with pytest.raises(JevResponseError):
            parse_response(payload, latency_ms=1.0, cached=False)


class TestApplyAnswers:
    def _answers(self, **overrides) -> JevAnswers:
        return parse_response(jev_payload(**overrides), latency_ms=1.0, cached=False)

    def test_none_operator_leaves_operator_unset(self):
        base = IntentSpec(raw_user_text="t", free_text_terms=["citation analysis"])
        out = apply_answers(base, self._answers())
        assert out.operator is None
        assert out.free_text_terms == ["citation analysis"]

    def test_operator_and_enums_are_taken_from_jev(self):
        base = IntentSpec(raw_user_text="t", authors=["Smith, J"], year_from=2020)
        answers = self._answers(
            operator="references",
            doctype=choice_answer("phdthesis", {"none": 0.1, "phdthesis": 0.9}),
            bibgroup=choice_answer("JWST", {"none": 0.2, "JWST": 0.8}),
            collection=choice_answer("physics", {"none": 0.3, "physics": 0.7}),
            refereed=noul_answer(0.95),
            openaccess=noul_answer(0.51),
            eprint=noul_answer(0.49),
        )
        out = apply_answers(base, answers)
        assert out.operator == "references"
        assert out.doctype == {"phdthesis"}
        assert out.bibgroup == {"JWST"}
        assert out.collection == {"physics"}
        assert out.property == {"refereed", "openaccess"}
        assert out.authors == ["Smith, J"] and out.year_from == 2020
        assert out.confidence["operator"] == pytest.approx(0.94)
        assert out.confidence["doctype"] == pytest.approx(0.9)
        assert out.confidence["property.eprint"] == pytest.approx(0.49)
        assert out.confidence["refers_to_specific_paper"] == pytest.approx(0.2)

    def test_does_not_mutate_input(self):
        base = IntentSpec(raw_user_text="t", property={"refereed"})
        apply_answers(base, self._answers(operator="similar"))
        assert base.operator is None and base.property == {"refereed"}

    def test_preserves_property_values_not_owned_by_jev(self):
        base = IntentSpec(property={"data", "refereed"})
        out = apply_answers(base, self._answers())
        assert out.property == {"data"}

    @given(
        unowned=st.sets(st.sampled_from(sorted(PROPERTIES - set(PROPERTY_BOOLEANS)))),
        refereed=st.booleans(),
        openaccess=st.booleans(),
        eprint=st.booleans(),
    )
    @example(unowned={"ads_openaccess"}, refereed=False, openaccess=False, eprint=False)
    @settings(database=None)
    def test_replaces_only_jev_owned_property_values(
        self,
        unowned: set[str],
        refereed: bool,
        openaccess: bool,
        eprint: bool,
    ) -> None:
        base = IntentSpec(property=unowned | set(PROPERTY_BOOLEANS))
        answers = self._answers(
            refereed=noul_answer(float(refereed)),
            openaccess=noul_answer(float(openaccess)),
            eprint=noul_answer(float(eprint)),
        )
        out = apply_answers(base, answers)
        expected = unowned | {
            name
            for name, selected in (
                ("refereed", refereed),
                ("openaccess", openaccess),
                ("eprint", eprint),
            )
            if selected
        }
        assert out.property == expected


class TestClientCache:
    def test_second_identical_request_is_served_from_cache(self, tmp_path):
        calls: list[dict] = []
        client = mock_jev_client(tmp_path, [jev_payload("trending")], calls)
        first = client.classify("what's hot in exoplanets")
        second = client.classify("what's hot in exoplanets")
        assert len(calls) == 1
        assert first.cached is False and second.cached is True
        assert second.choices["operator"].choice == "trending"
        rows = [json.loads(line) for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["fingerprint"] == request_fingerprint(calls[0])

    def test_cache_can_be_bypassed_for_stability_repeats(self, tmp_path):
        calls: list[dict] = []
        client = mock_jev_client(tmp_path, [jev_payload("useful"), jev_payload("useful")], calls)
        client.classify("q")
        client.classify("q", use_cache=False)
        assert len(calls) == 2

    def test_http_error_propagates(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "bad key"})

        client = JevClient(
            api_key="k", cache_path=tmp_path / "c.jsonl", transport=httpx.MockTransport(handler)
        )
        with pytest.raises(httpx.HTTPStatusError):
            client.classify("q")

    def test_missing_api_key_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            JevClient(api_key="", cache_path=tmp_path / "c.jsonl")


class TestExtractIntentJev:
    def test_composes_regex_extractors_with_jev_gating(self, tmp_path):
        calls: list[dict] = []
        client = mock_jev_client(tmp_path, [jev_payload("citations")], calls)
        text = "journal articles that build on dark energy work by Smith since 2019"
        intent = extract_intent_jev(text, client)
        assert intent.operator == "citations"
        assert intent.year_from == 2019
        assert intent.authors == ["Smith"]
        assert "dark energy" in " ".join(intent.free_text_terms)
        assert intent.doctype == set(), "Jev's 'none' overrides the regex 'journal articles' map"
        assert intent.raw_user_text == text
        assert calls[0]["state"] == {"query": text}

    def test_regex_intent_can_be_sent_as_context(self, tmp_path):
        calls: list[dict] = []
        client = mock_jev_client(tmp_path, [jev_payload()], calls)
        extract_intent_jev("dark energy since 2019", client, include_regex_state=True)
        assert calls[0]["state"]["regex_intent"]["year_from"] == 2019

    def test_ads_syntax_passes_through_without_calling_jev(self, tmp_path):
        calls: list[dict] = []
        client = mock_jev_client(tmp_path, [], calls)
        intent = extract_intent_jev('author:"Smith, J" year:2020', client)
        assert intent.confidence == {"ads_passthrough": 1.0}
        assert calls == []


class TestPipelineBackends:
    """process_query routes to the selected intent backend."""

    def test_regex_backend_never_calls_jev(self, tmp_path):
        from finetune.domains.scix.pipeline import process_query

        calls: list[dict] = []
        client = mock_jev_client(tmp_path, [], calls)
        result = process_query("papers citing dark energy surveys", "regex", client)
        assert calls == []
        assert result.debug_info.intent_backend == "regex"
        assert result.debug_info.classifier_called is False
        assert result.intent.operator == "citations"

    def test_jev_backend_always_calls_jev(self, tmp_path):
        from finetune.domains.scix.pipeline import process_query

        calls: list[dict] = []
        client = mock_jev_client(tmp_path, [jev_payload("none")], calls)
        result = process_query("papers citing dark energy surveys", "jev", client)
        assert len(calls) == 1
        assert result.debug_info.classifier_called is True
        assert result.intent.operator is None, "Jev's answer wins over the regex match"
        assert result.final_query

    def test_gated_backend_skips_jev_when_regex_found_an_operator(self, tmp_path):
        from finetune.domains.scix.pipeline import process_query

        calls: list[dict] = []
        client = mock_jev_client(tmp_path, [], calls)
        result = process_query("papers citing dark energy surveys", "jev_gated", client)
        assert calls == []
        assert result.debug_info.classifier_called is False

    def test_gated_backend_calls_jev_when_regex_found_no_operator(self, tmp_path):
        from finetune.domains.scix.pipeline import process_query

        calls: list[dict] = []
        client = mock_jev_client(tmp_path, [jev_payload("similar")], calls)
        result = process_query("work along the lines of the Planck results", "jev_gated", client)
        assert len(calls) == 1
        assert result.debug_info.classifier_called is True
        assert result.intent.operator == "similar"

    def test_jev_backends_require_a_client(self):
        from finetune.domains.scix.pipeline import process_query

        with pytest.raises(ValueError):
            process_query("x", "jev", None)
        with pytest.raises(ValueError):
            process_query("x", "bogus", None)  # type: ignore[arg-type]
