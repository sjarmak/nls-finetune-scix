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
    MAX_AUTHOR_CANDIDATES,
    RECENCY_WINDOWS,
    JevResponseError,
    ads_author,
    apply_answers,
    author_candidates,
    bibgroup_candidates,
    build_questions,
    build_request,
    classify_and_extract,
    parse_response,
    phrase_groupings,
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


class TestBibgroupCandidates:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("JWST or HST papers on exoplanet atmospheres", ("HST", "JWST")),
            ("Hubble deep field observations", ("HST",)),
            ("papers using Atacama Large Millimeter Array", ("ALMA",)),
            ("X-ray bursts research with Rossi XTE", ("RXTE",)),
            ("stellar populations studies using Gemini North", ("Gemini",)),
            ("asteroid surveys with panstarrs", ("Pan-STARRS",)),
            ("Wide-field Infrared Survey Explorer asteroids", ("WISE",)),
            ("recent papers on asteroids", ()),
        ],
    )
    def test_facilities_named_in_the_text(self, text, expected):
        assert bibgroup_candidates(text) == expected

    def test_no_facility_leaves_the_bibgroup_question_out(self):
        questions = build_request("dark matter", bibgroup_candidates=())["questions"]
        assert "bibgroup" not in questions

    def test_named_facilities_are_the_only_options(self):
        questions = build_request("JWST papers", bibgroup_candidates=("JWST",))["questions"]
        assert set(questions["bibgroup"]["criteria"]) == {"none", "JWST"}

    def test_without_candidates_every_bibgroup_is_offered(self):
        assert len(build_request("q")["questions"]["bibgroup"]["criteria"]) > 50

    def test_unasked_bibgroup_is_empty_and_has_no_confidence(self):
        request = build_request("dark matter", bibgroup_candidates=())
        answers = _answers(request)
        intent = apply_answers(IntentSpec(bibgroup={"HST"}), answers)
        assert intent.bibgroup == set()
        assert "bibgroup" not in intent.confidence

    def test_classify_and_extract_narrows_the_request(self):
        calls: list[dict] = []
        chandra = choice_answer("Chandra", {"none": 0.2, "Chandra": 0.8})
        client = mock_jev_client(None, [jev_payload(), jev_payload(bibgroup=chandra)], calls)
        classify_and_extract("dark matter halos", client)
        intent, _ = classify_and_extract("Chandra observations of dark matter halos", client)
        assert intent.bibgroup == {"Chandra"}
        assert "bibgroup" not in calls[0]["questions"]
        assert set(calls[1]["questions"]["bibgroup"]["criteria"]) == {"none", "Chandra"}


class TestAuthors:
    """Code offers capitalized words as names; Jev decides which are people."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("papers that cite Jarmak", ("Jarmak",)),
            ("papers by Hawking on black holes", ("Hawking",)),
            ("JWST papers on brown dwarfs", ()),
            ("recent papers on asteroids", ()),
            ("Chandra observations of Einstein rings", ("Chandra", "Einstein")),
            ("papers by hawking", ("hawking",)),
            ("papers by Übler published in 2019", ("Übler",)),
            ("Sara Seager", ("Sara Seager", "Sara", "Seager")),
            ("papers by A. G. Riess", ("A. G. Riess",)),
            ("Riess, A. G.", ("Riess, A. G.",)),
            ("similar work to Kaltenegger's biosignature models", ("Kaltenegger",)),
            (
                "sara seager exoplanet atmospheres",
                ("sara seager", "seager exoplanet", "exoplanet atmospheres"),
            ),
            ("Event Horizon Telescope images", ("Event", "Horizon", "Telescope")),
            ("Pieter van Dokkum dwarf galaxies", ("Pieter van Dokkum", "van Dokkum", "Pieter")),
            ("van Dokkum, P. G.", ("van Dokkum, P. G.",)),
        ],
    )
    def test_candidates(self, text, expected):
        assert author_candidates(text, extract_intent(text)) == expected

    def test_candidates_are_capped(self):
        text = "Alpha Beta Gamma Delta Epsilon Zeta"
        assert len(author_candidates(text, extract_intent(text))) == MAX_AUTHOR_CANDIDATES

    def test_one_yes_no_question_per_candidate(self):
        questions = build_request("q", author_candidates=("Jarmak", "Kurtz"))["questions"]
        assert questions["author_0"]["type"] == "noul"
        assert "'Jarmak'" in questions["author_0"]["instructions"]
        assert "'Kurtz'" in questions["author_1"]["instructions"]
        assert "author_0" not in build_request("q")["questions"]

    def test_papers_that_cite_a_surname(self):
        payload = jev_payload("citations", author_0=noul_answer(0.95))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("papers that cite Jarmak", client)
        assert intent.authors == ["Jarmak"]
        assert assemble_query(intent) == 'citations(author:"Jarmak")'

    def test_a_rejected_name_stays_in_the_topic(self):
        payload = jev_payload("citations", author_0=noul_answer(0.05))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("papers citing Maxwell", client)
        assert intent.authors == []
        assert assemble_query(intent) == "citations(abs:maxwell)"

    def test_jev_can_overrule_a_regex_author(self):
        payload = jev_payload(author_0=noul_answer(0.05))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("papers by Hubble", client)
        assert intent.authors == []
        assert intent.first_author is False
        assert "hubble" in " ".join(intent.free_text_terms).lower()

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Sara Seager", "Seager, Sara"),
            ("A. G. Riess", "Riess, A. G."),
            ("Riess, A. G.", "Riess, A. G."),
            ("sara seager", "Seager, Sara"),
            ("Seager", "Seager"),
            ("Pieter van Dokkum", "van Dokkum, Pieter"),
            ("richard de grijs", "de Grijs, Richard"),
            ("de Grijs", "de Grijs"),
        ],
    )
    def test_names_are_written_last_name_first(self, name, expected):
        assert ads_author(name) == expected

    def test_a_full_name_is_one_author(self):
        payload = jev_payload(author_0=noul_answer(0.95))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("Sara Seager exoplanet atmospheres", client)
        assert intent.authors == ["Seager, Sara"]
        assert assemble_query(intent) == 'author:"Seager, Sara" abs:"exoplanet atmospheres"'

    def test_two_surnames_written_together_can_be_two_people(self):
        payload = jev_payload(
            author_0=noul_answer(0.1), author_1=noul_answer(0.9), author_2=noul_answer(0.9)
        )
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("Madau Dickinson star formation history", client)
        assert intent.authors == ["Madau", "Dickinson"]

    def test_a_lowercase_name_is_found(self):
        payload = jev_payload(
            author_0=noul_answer(0.97), author_1=noul_answer(0.1), author_2=noul_answer(0.02)
        )
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("andy casey stellar spectra", client)
        assert assemble_query(intent) == 'author:"Casey, Andy" abs:"stellar spectra"'

    def test_a_regex_name_covered_by_an_accepted_span_stays_out_of_the_topic(self):
        names = ("Sara Seager", "Sara", "Seager")
        req = build_request("q", author_candidates=names)
        answers = _answers(
            req, author_0=noul_answer(0.95), author_1=noul_answer(0.1), author_2=noul_answer(0.1)
        )
        intent = IntentSpec(authors=["Sara Seager"], free_text_terms=["exoplanets"])
        out = apply_answers(intent, answers, author_candidates=names)
        assert out.authors == ["Seager, Sara"]
        assert out.free_text_terms == ["exoplanets"]

    def test_confidence_is_recorded_per_name(self):
        payload = jev_payload("citations", author_0=noul_answer(0.95))
        intent, _ = classify_and_extract(
            "papers that cite Jarmak", mock_jev_client(None, [payload], [])
        )
        assert intent.confidence["author.Jarmak"] == 0.95


class TestPhraseGrouping:
    """Code lists every contiguous grouping of the topic words; Jev picks one."""

    def test_every_grouping_whole_phrase_first(self):
        intent = IntentSpec(free_text_terms=["atmospheric escape sub-neptunes"])
        assert phrase_groupings(intent) == (
            "atmospheric escape sub-neptunes",
            "atmospheric | escape sub-neptunes",
            "atmospheric escape | sub-neptunes",
            "atmospheric | escape | sub-neptunes",
        )

    @pytest.mark.parametrize(
        "terms",
        [["dark energy"], ["one two three four five six"], ["a b c", "d e f"], []],
    )
    def test_not_asked_for_short_long_or_several_phrases(self, terms):
        assert phrase_groupings(IntentSpec(free_text_terms=terms)) == ()

    def test_question_offers_the_groupings(self):
        groupings = ("a b c", "a | b c", "a b | c", "a | b | c")
        questions = build_request("q", phrase_groupings=groupings)["questions"]
        assert list(questions["phrasing"]["criteria"]) == list(groupings)
        assert "phrasing" not in build_request("q")["questions"]

    def test_chosen_grouping_is_cut_to_the_chosen_topic(self):
        text = "recent papers on atmospheric escape from sub-Neptunes"
        phrase = extract_intent(text).free_text_terms[0]
        assert phrase == "recent atmospheric escape sub-neptunes"
        grouping = "recent | atmospheric escape | sub-neptunes"
        payload = jev_payload(
            recency=choice_answer("last_3_years", {"none": 0.01, "last_3_years": 0.99}),
            topic=choice_answer(
                "atmospheric escape sub-neptunes",
                {"none": 0.0, "atmospheric escape sub-neptunes": 1.0},
            ),
            phrasing=choice_answer(grouping, {grouping: 0.9, phrase: 0.1}),
        )
        intent, _ = classify_and_extract(
            text, mock_jev_client(None, [payload], []), reference_year=2026
        )
        assert intent.free_text_terms == ["atmospheric escape", "sub-neptunes"]
        assert assemble_query(intent) == (
            'abs:"atmospheric escape" abs:"sub-neptunes" pubdate:[2024 TO 2026]'
        )

    def test_whole_phrase_answer_keeps_one_phrase(self):
        payload = jev_payload()
        intent, _ = classify_and_extract(
            "supermassive black hole growth", mock_jev_client(None, [payload], [])
        )
        assert intent.free_text_terms == ["supermassive black hole growth"]

    def test_no_topic_means_no_terms(self):
        payload = jev_payload(
            topic=choice_answer("none", {"none": 1.0, "supermassive black hole growth": 0.0}),
            phrasing=choice_answer(
                "supermassive | black hole | growth",
                {"supermassive | black hole | growth": 1.0},
            ),
        )
        intent, _ = classify_and_extract(
            "supermassive black hole growth", mock_jev_client(None, [payload], [])
        )
        assert intent.free_text_terms == []
