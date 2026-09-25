"""Jev owns recency, topic choice, first author and highly cited (bead nls-finetune-scix-bh9).

Code proposes mechanical candidates (topic sub-spans, recency windows); Jev
picks. All traffic runs over an httpx MockTransport.
"""

import pytest
from jev_fixtures import choice_answer, jev_payload, mock_jev_client, noul_answer

from finetune.domains.scix.assembler import assemble_query
from finetune.domains.scix.field_constraints import BIBGROUPS
from finetune.domains.scix.intent_spec import IntentSpec
from finetune.domains.scix.jev_intent import (
    EXTRACTION_QUESTION_IDS,
    HIGHLY_CITED_MIN_CITATIONS,
    MAX_AUTHOR_CANDIDATES,
    MAX_AUTHOR_READINGS,
    MAX_JOIN_QUESTIONS,
    RECENCY_WINDOWS,
    JevResponseError,
    ads_author,
    apply_answers,
    author_candidates,
    author_readings,
    bibgroup_candidates,
    build_questions,
    build_request,
    classify_and_extract,
    parse_response,
    topic_candidates,
    word_pairs,
)
from finetune.domains.scix.ner import extract_intent


def reading(choice: str, probabilities: dict[str, float] | None = None) -> dict:
    """An ``author_reading`` answer; ``probabilities`` override the default 0.9 on ``choice``."""
    return choice_answer(choice, {choice: 0.9, **(probabilities or {})})


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
            ("X-ray bursts research with Rossi XTE", ()),
            ("radio jets with the Very Large Array", ("NRAO",)),
            ("stellar populations studies using Gemini North", ("Gemini",)),
            ("asteroid surveys with panstarrs", ("Pan-STARRS",)),
            ("Wide-field Infrared Survey Explorer asteroids", ()),
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
        criteria = build_request("q")["questions"]["bibgroup"]["criteria"]
        assert set(criteria) == {"none", *BIBGROUPS}

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
                (
                    "sara seager",
                    "seager exoplanet",
                    "exoplanet atmospheres",
                    "sara",
                    "seager",
                    "exoplanet",
                    "atmospheres",
                ),
            ),
            ("accomazzi europa", ("accomazzi europa", "accomazzi", "europa")),
            ("Event Horizon Telescope images", ("Event", "Horizon", "Telescope")),
            ("Pieter van Dokkum dwarf galaxies", ("Pieter van Dokkum", "van Dokkum", "Pieter")),
            ("van Dokkum, P. G.", ("van Dokkum, P. G.",)),
        ],
    )
    def test_candidates(self, text, expected):
        assert author_candidates(text, extract_intent(text)) == expected

    def test_candidates_are_capped(self):
        text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota"
        assert len(author_candidates(text, extract_intent(text))) == MAX_AUTHOR_CANDIDATES

    def test_readings_are_sets_of_names_that_share_no_word(self):
        assert author_readings(("Jarmak Cassini", "Jarmak", "Cassini")) == (
            (),
            ("Jarmak Cassini",),
            ("Jarmak",),
            ("Cassini",),
            ("Jarmak", "Cassini"),
        )

    def test_readings_are_capped_and_keep_every_name_as_a_person(self):
        names = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")
        readings = author_readings(names)
        assert len(readings) == MAX_AUTHOR_READINGS
        assert readings[0] == ()
        assert readings[-1] == names

    def test_capped_readings_keep_every_name_as_written(self):
        names = ("Sara Seager", "Alpha", "Beta", "Gamma", "Sara", "Seager")
        readings = author_readings(names)
        assert len(readings) == MAX_AUTHOR_READINGS
        assert readings[-1] == ("Sara Seager", "Alpha", "Beta", "Gamma")

    def test_a_hyphenated_surname_and_its_first_part_are_not_two_people(self):
        assert author_readings(("El-Badry", "El")) == ((), ("El-Badry",), ("El",))

    def test_a_hyphenated_author_leaves_no_part_in_the_topic(self):
        payload = jev_payload(author_reading=reading("El-Badry"))
        intent, _ = classify_and_extract(
            "papers by El-Badry on stellar binaries", mock_jev_client(None, [payload], [])
        )
        assert intent.authors == ["El-Badry"]
        assert intent.free_text_terms == ["stellar binaries"]

    def test_one_reading_question_for_all_candidates(self):
        names = ("Jarmak Cassini", "Jarmak", "Cassini")
        question = build_request("q", author_candidates=names)["questions"]["author_reading"]
        assert question["type"] == "choice"
        assert list(question["criteria"]) == [
            "none",
            "Jarmak Cassini",
            "Jarmak",
            "Cassini",
            "Jarmak|Cassini",
        ]
        jarmak = question["criteria"]["Jarmak"]
        assert jarmak.startswith("'Jarmak' is one person")
        assert "'Cassini' is not a person here" in jarmak
        assert question["criteria"]["Jarmak|Cassini"] == (
            "'Jarmak' and 'Cassini' are different people, each an author."
        )
        assert "author_reading" not in build_request("q")["questions"]

    def test_a_mission_next_to_a_surname_is_the_topic(self):
        payload = jev_payload(author_reading=reading("Jarmak", {"none": 0.3, "Cassini": 0.1}))
        intent, _ = classify_and_extract("Jarmak Cassini", mock_jev_client(None, [payload], []))
        assert intent.authors == ["Jarmak"]
        assert assemble_query(intent) == 'author:"Jarmak" abs:cassini'

    def test_papers_that_cite_a_surname(self):
        payload = jev_payload("citations", author_reading=reading("Jarmak"))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("papers that cite Jarmak", client)
        assert intent.authors == ["Jarmak"]
        assert assemble_query(intent) == 'citations(author:"Jarmak")'

    @pytest.mark.parametrize("subject", ["ARP299", "NGC 3690"])
    def test_cite_after_a_topic_leaves_the_topic_when_jev_picks_citations(self, subject):
        text = (
            f'papers about {subject} that cite "A Digital Archive of HI 21 Centimeter '
            'Line Spectra of Optically Targeted Galaxies"'
        )
        calls: list[dict] = []
        client = mock_jev_client(None, [jev_payload("citations")], calls)
        intent, _ = classify_and_extract(text, client)
        assert intent.operator == "citations"
        offered = " ".join(q.get("instructions", "") for q in calls[0]["questions"].values())
        assert "'cite'" not in offered
        assert "cite" not in {w for term in intent.free_text_terms for w in term.split()}
        assert "abs:cite" not in assemble_query(intent)

    def test_a_rejected_name_stays_in_the_topic(self):
        payload = jev_payload("citations", author_reading=reading("none"))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("papers citing Maxwell", client)
        assert intent.authors == []
        assert assemble_query(intent) == "citations(abs:maxwell)"

    def test_jev_can_overrule_a_regex_author(self):
        payload = jev_payload(author_reading=reading("none"))
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
        payload = jev_payload(author_reading=reading("Sara Seager"))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("Sara Seager exoplanet atmospheres", client)
        assert intent.authors == ["Seager, Sara"]
        assert assemble_query(intent) == 'author:"Seager, Sara" abs:"exoplanet atmospheres"'

    def test_two_surnames_written_together_can_be_two_people(self):
        payload = jev_payload(author_reading=reading("Madau|Dickinson"))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("Madau Dickinson star formation history", client)
        assert intent.authors == ["Madau", "Dickinson"]

    def test_a_lowercase_name_is_found(self):
        payload = jev_payload(author_reading=reading("andy casey"))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("andy casey stellar spectra", client)
        assert assemble_query(intent) == 'author:"Casey, Andy" abs:"stellar spectra"'

    def test_a_lowercase_surname_next_to_a_mission_is_one_author(self):
        payload = jev_payload(author_reading=reading("accomazzi"))
        client = mock_jev_client(None, [payload], [])
        intent, _ = classify_and_extract("accomazzi europa", client)
        assert assemble_query(intent) == 'author:"Accomazzi" abs:europa'

    def test_a_regex_name_covered_by_the_chosen_span_stays_out_of_the_topic(self):
        names = ("Sara Seager", "Sara", "Seager")
        req = build_request("q", author_candidates=names)
        answers = _answers(req, author_reading=reading("Sara Seager"))
        intent = IntentSpec(authors=["Sara Seager"], free_text_terms=["exoplanets"])
        out = apply_answers(intent, answers, author_candidates=names)
        assert out.authors == ["Seager, Sara"]
        assert out.free_text_terms == ["exoplanets"]

    def test_confidence_is_recorded_per_name_across_readings(self):
        probabilities = {"Jarmak": 0.6, "Jarmak|Cassini": 0.25, "none": 0.15}
        payload = jev_payload(author_reading=reading("Jarmak", probabilities))
        intent, _ = classify_and_extract("Jarmak Cassini", mock_jev_client(None, [payload], []))
        assert intent.confidence["author_reading"] == 0.6
        assert intent.confidence["author.Jarmak"] == pytest.approx(0.85)
        assert intent.confidence["author.Cassini"] == pytest.approx(0.25)
        assert intent.confidence["author.Jarmak Cassini"] == 0.0


class TestPhraseSplitting:
    """Code asks about each adjacent word pair; Jev says which pairs are one term."""

    def test_each_pair_of_a_long_phrase_is_asked(self):
        intent = IntentSpec(free_text_terms=["tidal disruption small bodies", "dark energy"])
        assert word_pairs(intent) == (
            ("tidal disruption small bodies", 0),
            ("tidal disruption small bodies", 1),
            ("tidal disruption small bodies", 2),
        )

    def test_a_phrase_past_the_cap_stays_whole(self):
        long = " ".join(f"w{i}" for i in range(MAX_JOIN_QUESTIONS + 2))
        intent = IntentSpec(free_text_terms=["a b c", long])
        assert word_pairs(intent) == (("a b c", 0), ("a b c", 1))

    def test_question_names_the_two_words(self):
        pairs = (("stis ultraviolet spectroscopy", 1),)
        questions = build_request("q", word_pairs=pairs)["questions"]
        assert "'ultraviolet' and 'spectroscopy'" in questions["join_0"]["instructions"]
        assert not any(q.startswith("join_") for q in build_request("q")["questions"])

    def test_unjoined_pairs_split_the_phrase(self):
        text = "Numerical simulations of tidal disruption of small bodies"
        joins = [0.58, 0.07, 0.89, 0.09, 0.65]
        payload = jev_payload(**{f"join_{k}": noul_answer(p) for k, p in enumerate(joins)})
        intent, _ = classify_and_extract(text, mock_jev_client(None, [payload], []))
        assert assemble_query(intent) == (
            'abs:"numerical simulations" abs:"tidal disruption" abs:"small bodies"'
        )

    def test_split_is_cut_to_the_chosen_topic(self):
        text = "recent papers on atmospheric escape from sub-Neptunes"
        phrase = extract_intent(text).free_text_terms[0]
        assert phrase == "recent atmospheric escape sub-neptunes"
        payload = jev_payload(
            recency=choice_answer("last_3_years", {"none": 0.01, "last_3_years": 0.99}),
            topic=choice_answer(
                "atmospheric escape sub-neptunes",
                {"none": 0.0, "atmospheric escape sub-neptunes": 1.0},
            ),
            join_0=noul_answer(0.1),
            join_1=noul_answer(0.9),
            join_2=noul_answer(0.1),
        )
        intent, _ = classify_and_extract(
            text, mock_jev_client(None, [payload], []), reference_year=2026
        )
        assert intent.free_text_terms == ["atmospheric escape", "sub-neptunes"]
        assert assemble_query(intent) == (
            'abs:"atmospheric escape" abs:"sub-neptunes" pubdate:[2024 TO 2026]'
        )

    def test_joined_pairs_keep_one_phrase(self):
        payload = jev_payload()
        intent, _ = classify_and_extract(
            "supermassive black hole growth", mock_jev_client(None, [payload], [])
        )
        assert intent.free_text_terms == ["supermassive black hole growth"]

    def test_no_topic_means_no_terms(self):
        payload = jev_payload(
            topic=choice_answer("none", {"none": 1.0, "supermassive black hole growth": 0.0}),
            join_0=noul_answer(0.1),
        )
        intent, _ = classify_and_extract(
            "supermassive black hole growth", mock_jev_client(None, [payload], [])
        )
        assert intent.free_text_terms == []
