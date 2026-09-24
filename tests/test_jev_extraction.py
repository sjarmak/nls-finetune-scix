"""Jev owns recency, topic choice, first author and highly cited (bead nls-finetune-scix-bh9).

Code proposes mechanical candidates (topic sub-spans, recency windows); Jev
picks. All traffic runs over an httpx MockTransport.
"""

import pytest
from jev_fixtures import choice_answer, jev_payload, mock_jev_client, noul_answer

from finetune.domains.scix.assembler import assemble_query
from finetune.domains.scix.intent_spec import IntentSpec
from finetune.domains.scix.jev_intent import (
    EXTRACTION_QUESTION_IDS,
    HIGHLY_CITED_MIN_CITATIONS,
    RECENCY_WINDOWS,
    JevResponseError,
    apply_answers,
    build_questions,
    build_request,
    classify_and_extract,
    parse_response,
    topic_candidates,
)
from finetune.domains.scix.ner import extract_intent


def _answers(request: dict, **overrides):
    return parse_response(
        jev_payload(**overrides), latency_ms=1.0, cached=False, questions=request["questions"]
    )


class TestQuestionSet:
    def test_gating_set_is_unchanged_so_arm_c_keeps_its_prompt(self):
        assert not set(EXTRACTION_QUESTION_IDS) & set(build_questions())

    def test_request_asks_the_extraction_questions(self):
        questions = build_request("recent papers on asteroids")["questions"]
        assert questions["recency"]["type"] == "choice"
        assert set(questions["recency"]["criteria"]) == {"none", *RECENCY_WINDOWS}
        assert questions["first_author"]["type"] == "noul"
        assert questions["highly_cited"]["type"] == "noul"
        assert "topic" not in questions

    def test_topic_question_offers_none_plus_the_candidates(self):
        req = build_request("q", topic_candidates=("recent asteroids", "recent", "asteroids"))
        options = req["questions"]["topic"]["criteria"]
        assert set(options) == {"none", "recent asteroids", "recent", "asteroids"}
        assert all(text.strip() for text in options.values())


class TestTopicCandidates:
    def test_all_contiguous_sub_spans_longest_first(self):
        intent = IntentSpec(free_text_terms=["recent dark matter"])
        assert topic_candidates(intent) == (
            "recent dark matter",
            "recent dark",
            "dark matter",
            "recent",
            "dark",
            "matter",
        )

    @pytest.mark.parametrize(
        "intent",
        [
            IntentSpec(),
            IntentSpec(free_text_terms=["a", "b"]),
            IntentSpec(free_text_terms=["rocks"], or_terms=["rocks", "volcanoes"]),
            IntentSpec(free_text_terms=["one two three four five six seven"]),
        ],
    )
    def test_no_question_when_the_topic_is_not_one_short_phrase(self, intent):
        assert topic_candidates(intent) == ()

    def test_a_span_equal_to_the_none_option_is_dropped(self):
        assert "none" not in topic_candidates(IntentSpec(free_text_terms=["none left"]))


class TestParseResponse:
    def test_topic_must_be_one_of_the_offered_candidates(self):
        req = build_request("q", topic_candidates=("asteroids",))
        bad = choice_answer("comets", {"none": 0.1, "comets": 0.9})
        with pytest.raises(JevResponseError):
            _answers(req, topic=bad)

    def test_topic_answer_is_required_when_asked(self):
        req = build_request("q", topic_candidates=("asteroids",))
        with pytest.raises(JevResponseError):
            _answers(req)

    def test_extraction_answers_are_required(self):
        payload = jev_payload()
        payload["answers"].pop("recency")
        with pytest.raises(JevResponseError):
            parse_response(payload, latency_ms=1.0, cached=False)


class TestApplyAnswers:
    def test_recency_window_ends_at_the_reference_year(self):
        req = build_request("q")
        answers = _answers(
            req, recency=choice_answer("last_3_years", {"none": 0.1, "last_3_years": 0.9})
        )
        out = apply_answers(IntentSpec(), answers, reference_year=2025)
        assert (out.year_from, out.year_to) == (2023, 2025)
        assert out.confidence["recency"] == pytest.approx(0.9)

    def test_explicit_years_win_over_recency(self):
        req = build_request("q")
        answers = _answers(
            req, recency=choice_answer("last_2_years", {"none": 0.1, "last_2_years": 0.9})
        )
        out = apply_answers(IntentSpec(year_from=2010, year_to=2012), answers, reference_year=2025)
        assert (out.year_from, out.year_to) == (2010, 2012)

    def test_topic_choice_replaces_the_regex_phrase(self):
        req = build_request("q", topic_candidates=("recent asteroids", "recent", "asteroids"))
        topic = choice_answer("asteroids", {"none": 0.01, "recent": 0.01, "asteroids": 0.98})
        out = apply_answers(
            IntentSpec(free_text_terms=["recent asteroids"]), _answers(req, topic=topic)
        )
        assert out.free_text_terms == ["asteroids"]
        assert out.confidence["topic"] == pytest.approx(0.98)

    def test_topic_none_clears_the_phrase(self):
        req = build_request("q", topic_candidates=("latest",))
        topic = choice_answer("none", {"none": 0.9, "latest": 0.1})
        out = apply_answers(IntentSpec(free_text_terms=["latest"]), _answers(req, topic=topic))
        assert out.free_text_terms == []

    def test_first_author_needs_an_author(self):
        req = build_request("q")
        answers = _answers(req, first_author=noul_answer(0.9))
        assert apply_answers(IntentSpec(authors=["Riess, A"]), answers).first_author is True
        assert apply_answers(IntentSpec(), answers).first_author is False

    def test_highly_cited_sets_the_citation_floor(self):
        req = build_request("q")
        out = apply_answers(IntentSpec(), _answers(req, highly_cited=noul_answer(0.8)))
        assert out.min_citations == HIGHLY_CITED_MIN_CITATIONS
        low = apply_answers(IntentSpec(), _answers(req, highly_cited=noul_answer(0.2)))
        assert low.min_citations is None


class TestEndToEnd:
    def test_recent_papers_on_asteroids(self, tmp_path):
        calls: list[dict] = []
        payload = jev_payload(
            recency=choice_answer("last_3_years", {"none": 0.01, "last_3_years": 0.99}),
            topic=choice_answer("asteroids", {"none": 0.0, "recent": 0.0, "asteroids": 1.0}),
        )
        client = mock_jev_client(tmp_path, [payload], calls)
        intent, _ = classify_and_extract("recent papers on asteroids", client, reference_year=2025)
        assert set(calls[0]["questions"]["topic"]["criteria"]) >= {"asteroids", "recent"}
        assert assemble_query(intent) == "abs:asteroids pubdate:[2023 TO 2025]"

    def test_highly_cited_first_author_papers(self, tmp_path):
        payload = jev_payload(first_author=noul_answer(0.95), highly_cited=noul_answer(0.9))
        client = mock_jev_client(tmp_path, [payload], [])
        intent, _ = classify_and_extract("highly cited first author papers by Riess", client)
        query = assemble_query(intent)
        assert 'author:"^Riess"' in query
        assert "citation_count:[100 TO *]" in query


class TestAssembler:
    def test_first_author_carets_only_the_first_name(self):
        intent = IntentSpec(authors=["Fry", "Fields"], first_author=True)
        assert assemble_query(intent) == 'author:"^Fry" author:"Fields"'

    def test_citation_floor_clause(self):
        intent = IntentSpec(free_text_terms=["asteroids"], min_citations=100)
        assert assemble_query(intent) == "abs:asteroids citation_count:[100 TO *]"


class TestReferenceYear:
    def test_last_n_years_anchor_on_the_reference_year(self):
        intent = extract_intent("exoplanet papers from the last 5 years", reference_year=2025)
        assert intent.year_to == 2025
