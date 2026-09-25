"""Named-paper lookup: ADS proposes candidates, Jev picks the paper."""

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from jev_fixtures import (
    TEST_API_KEY,
    answering_client,
    choice_answer,
    handler_client,
    jev_payload,
    noul_answer,
    with_topic_answer,
)

from finetune.domains.scix.assembler import assemble_query
from finetune.domains.scix.intent_spec import IntentSpec
from finetune.domains.scix.paper_lookup import (
    DESCRIBES_PREFIX,
    MAX_CANDIDATES,
    MAX_SEARCHES,
    MAX_TERM_SEARCHES,
    PAPER_QUESTION,
    ADSPaperSearch,
    PaperCandidate,
    PaperSearchError,
    candidate_searches,
    needs_lookup,
    paper_request,
    pool_candidates,
    resolve_paper,
)
from finetune.domains.scix.pipeline import process_query

GILLON = {
    "bibcode": "2017Natur.542..456G",
    "title": ["Seven temperate terrestrial planets around the nearby ultracool dwarf star"],
    "first_author": "Gillon, Michaël",
    "year": "2017",
    "citation_count": 1409,
}
ORMEL = {
    "bibcode": "2017A&A...604A...1O",
    "title": ["Formation of TRAPPIST-1 and other compact systems"],
    "first_author": "Ormel, Chris W.",
    "year": "2017",
    "citation_count": 155,
}
PENZIAS = {
    "bibcode": "1965ApJ...142..419P",
    "title": ["A Measurement of Excess Antenna Temperature at 4080 Mc/s."],
    "first_author": "Penzias, A. A.",
    "year": "1965",
    "citation_count": 2000,
}
TRAPPIST_TEXT = "Papers that cite the original TRAPPIST-1 seven-planet paper"


def candidate(bibcode: str) -> PaperCandidate:
    return PaperCandidate(bibcode, f"title {bibcode}", "Author, A.", "2000", 1)


def ads_search(docs_for, calls: list[str] | None = None) -> ADSPaperSearch:
    """ADSPaperSearch whose ADS answers each query ``q`` with ``docs_for(q)``."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {TEST_API_KEY}"
        params = parse_qs(urlparse(str(request.url)).query)
        assert params["sort"] == ["citation_count desc"]
        query = params["q"][0]
        if calls is not None:
            calls.append(query)
        return httpx.Response(200, json={"response": {"docs": docs_for(query)}})

    return ADSPaperSearch(api_key=TEST_API_KEY, transport=httpx.MockTransport(handler))


def jev_answering_paper(
    paper_choice: str,
    calls: list[dict],
    operator: str = "citations",
    confidence: float = 0.9,
    restricting: frozenset[str] = frozenset(),
):
    """Jev that classifies as ``operator`` about one paper and picks ``paper_choice``.

    Every term and author describes the paper except those in ``restricting``.
    """
    payload = jev_payload(operator, refers_to_specific_paper=noul_answer(0.92))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        questions = body["questions"]
        if PAPER_QUESTION in questions:
            options = questions[PAPER_QUESTION]["criteria"]
            rest = (1.0 - confidence) / (len(options) - 1)
            probabilities = {o: confidence if o == paper_choice else rest for o in options}
            answers = {PAPER_QUESTION: choice_answer(paper_choice, probabilities)}
            for qid, question in questions.items():
                if qid.startswith(DESCRIBES_PREFIX):
                    named = any(f"'{r}'" in question["instructions"] for r in restricting)
                    answers[qid] = noul_answer(0.1 if named else 0.9)
            return httpx.Response(
                200, json={"model": "jev", "answers": answers, "usage": {"input_tokens": 300}}
            )
        return httpx.Response(200, json=with_topic_answer(payload, body))

    return handler_client(handler)


def paper_intent(**overrides) -> IntentSpec:
    fields = {
        "operator": "citations",
        "free_text_terms": ["original", "trappist-1", "seven-planet"],
        "confidence": {"refers_to_specific_paper": 0.92},
    }
    return IntentSpec(**{**fields, **overrides})


class TestNeedsLookup:
    def test_citations_of_one_named_paper(self):
        assert needs_lookup(paper_intent())

    @pytest.mark.parametrize("operator", ["references", "similar"])
    def test_other_operators_with_a_target(self, operator):
        assert needs_lookup(paper_intent(operator=operator))

    @pytest.mark.parametrize("operator", [None, "reviews", "useful", "trending"])
    def test_operators_without_a_target(self, operator):
        assert not needs_lookup(paper_intent(operator=operator))

    def test_topic_request_is_not_looked_up(self):
        assert not needs_lookup(paper_intent(confidence={"refers_to_specific_paper": 0.3}))

    def test_request_without_the_jev_answer_is_not_looked_up(self):
        assert not needs_lookup(paper_intent(confidence={}))

    def test_target_already_set(self):
        assert not needs_lookup(paper_intent(operator_target="2017Natur.542..456G"))


class TestCandidateSearches:
    def test_all_terms_full_text_then_each_term(self):
        assert candidate_searches(paper_intent()) == (
            'abs:original abs:"trappist-1" abs:"seven-planet"',
            'full:original full:"trappist-1" full:"seven-planet"',
            "abs:original",
            'abs:"trappist-1"',
            'abs:"seven-planet"',
        )

    def test_single_word_is_searched_in_abstracts_and_full_text(self):
        assert candidate_searches(paper_intent(free_text_terms=["emcee"])) == (
            "abs:emcee",
            "full:emcee",
        )

    def test_single_phrase_is_also_searched_one_word_at_a_time(self):
        intent = paper_intent(free_text_terms=["2mass all-sky survey"])
        assert candidate_searches(intent) == (
            'abs:"2mass all-sky survey"',
            'full:"2mass all-sky survey"',
            "abs:2mass",
            'abs:"all-sky"',
            "abs:survey",
        )

    def test_term_searches_are_capped(self):
        terms = [f"term{i}" for i in range(MAX_TERM_SEARCHES + 3)]
        searches = candidate_searches(paper_intent(free_text_terms=terms))
        assert len(searches) == 2 + MAX_TERM_SEARCHES

    def test_phrase_word_searches_are_capped(self):
        phrase = " ".join(f"word{i}" for i in range(MAX_TERM_SEARCHES + 3))
        searches = candidate_searches(paper_intent(free_text_terms=[phrase]))
        assert len(searches) == 2 + MAX_TERM_SEARCHES

    def test_authors_are_not_searched_without_the_terms(self):
        intent = paper_intent(free_text_terms=["radiation"], authors=["Hawking"])
        searches = candidate_searches(intent)
        assert 'author:"Hawking"' not in searches
        assert all(s.startswith('author:"Hawking"') for s in searches)

    def test_every_search_stays_within_the_bound(self):
        intent = paper_intent(
            free_text_terms=[f"term{i}" for i in range(MAX_TERM_SEARCHES + 3)], authors=["Riess"]
        )
        assert len(candidate_searches(intent)) == MAX_SEARCHES

    def test_authors_and_explicit_year_describe_the_paper(self):
        intent = paper_intent(
            free_text_terms=["trappist-1"],
            authors=["Gillon"],
            year_from=2017,
            year_to=2017,
            confidence={"refers_to_specific_paper": 0.9, "year": 0.9},
        )
        query, full_text = candidate_searches(intent)
        assert "author:" in query and "Gillon" in query
        assert 'abs:"trappist-1"' in query and "2017" in query
        assert 'full:"trappist-1"' in full_text and "Gillon" in full_text and "2017" in full_text

    def test_recency_window_is_not_searched(self):
        intent = paper_intent(free_text_terms=["ligo"], year_from=2024, year_to=2026)
        assert candidate_searches(intent) == ("abs:ligo", "full:ligo")

    def test_authors_without_terms_are_one_search(self):
        intent = paper_intent(free_text_terms=[], authors=["Hawking"])
        assert candidate_searches(intent) == ('author:"Hawking"',)

    def test_nothing_to_search(self):
        assert candidate_searches(paper_intent(free_text_terms=[])) == ()


class TestPoolCandidates:
    def test_round_robin_by_rank_and_dedupe(self):
        first = [candidate("a"), candidate("b"), candidate("c")]
        second = [candidate("x"), candidate("a"), candidate("y")]
        pooled = [c.bibcode for c in pool_candidates([first, second])]
        assert pooled == ["a", "x", "b", "c", "y"]

    def test_capped(self):
        results = [[candidate(f"{s}-{r}") for r in range(5)] for s in range(5)]
        pooled = pool_candidates(results)
        assert len(pooled) == MAX_CANDIDATES
        assert {c.bibcode for c in pooled[:5]} == {f"{s}-0" for s in range(5)}

    def test_empty(self):
        assert pool_candidates([[], []]) == []


class TestADSPaperSearch:
    def test_parses_docs(self):
        search = ads_search(lambda q: [GILLON])
        (paper,) = search.top_cited('abs:"trappist-1"')
        assert paper == PaperCandidate(
            "2017Natur.542..456G", GILLON["title"][0], "Gillon, Michaël", "2017", 1409
        )

    def test_missing_optional_fields(self):
        (paper,) = ads_search(lambda q: [{"bibcode": "2000X"}]).top_cited("abs:x")
        assert paper.bibcode == "2000X" and paper.citation_count == 0

    def test_doc_without_bibcode_fails(self):
        with pytest.raises(PaperSearchError):
            ads_search(lambda q: [{"title": ["no bibcode"]}]).top_cited("abs:x")

    @pytest.mark.parametrize(
        "doc",
        [
            {"bibcode": "2000X", "citation_count": "N/A"},
            {"bibcode": "2000X", "citation_count": 1.5},
            {"bibcode": "2000X", "title": "a string, not a list"},
        ],
    )
    def test_malformed_doc_fails(self, doc):
        with pytest.raises(PaperSearchError):
            ads_search(lambda q: [doc]).top_cited("abs:x")

    def test_body_without_docs_fails(self):
        search = ADSPaperSearch(
            api_key=TEST_API_KEY,
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"error": "x"})),
        )
        with pytest.raises(PaperSearchError):
            search.top_cited("abs:x")

    def test_http_error_raises(self):
        search = ADSPaperSearch(
            api_key=TEST_API_KEY,
            transport=httpx.MockTransport(lambda r: httpx.Response(500)),
        )
        with pytest.raises(httpx.HTTPStatusError):
            search.top_cited("abs:x")

    def test_search_all_keeps_query_order(self):
        search = ads_search(lambda q: [{"bibcode": q}])
        results = search.search_all(("abs:a", "abs:b", "abs:c"))
        assert [r[0].bibcode for r in results] == ["abs:a", "abs:b", "abs:c"]

    def test_requires_key(self):
        with pytest.raises(ValueError):
            ADSPaperSearch(api_key="")


class TestPaperRequest:
    def test_offers_none_and_each_candidate(self):
        papers = [candidate("a"), candidate("b")]
        request = paper_request("text", papers, "jev-model")
        question = request["questions"][PAPER_QUESTION]
        assert question["type"] == "choice"
        assert list(question["criteria"]) == ["none", "a", "b"]
        assert "title a" in question["criteria"]["a"]
        assert request["state"] == {"query": "text"} and request["model"] == "jev-model"
        assert set(request["questions"]) == {PAPER_QUESTION}

    def test_asks_whether_each_term_and_author_describes_the_paper(self):
        request = paper_request("text", [candidate("a")], "m", ("arp299", "Kurtz"))
        questions = request["questions"]
        assert questions["describes_0"]["type"] == "noul"
        assert "'arp299'" in questions["describes_0"]["instructions"]
        assert "'Kurtz'" in questions["describes_1"]["instructions"]


class TestResolvePaper:
    def test_chosen_paper_becomes_the_target(self):
        calls: list[dict] = []
        search = ads_search(lambda q: [ORMEL, GILLON])
        intent, lookup = resolve_paper(
            TRAPPIST_TEXT, paper_intent(), jev_answering_paper(GILLON["bibcode"], calls), search
        )
        assert intent.operator_target == GILLON["bibcode"]
        assert intent.free_text_terms == [] and intent.authors == []
        assert intent.confidence[PAPER_QUESTION] == pytest.approx(0.9)
        assert lookup.bibcode == GILLON["bibcode"] and lookup.title == GILLON["title"][0]
        assert lookup.candidates == (ORMEL["bibcode"], GILLON["bibcode"])
        assert len(lookup.searches) == 5

    def test_restricting_terms_and_authors_stay_outside_the_operator(self):
        calls: list[dict] = []
        search = ads_search(lambda q: [GILLON])
        before = paper_intent(
            free_text_terms=["arp299", "trappist-1"], authors=["Kurtz"], first_author=True
        )
        client = jev_answering_paper(
            GILLON["bibcode"], calls, restricting=frozenset({"arp299", "Kurtz"})
        )
        intent, _ = resolve_paper(TRAPPIST_TEXT, before, client, search)
        assert intent.free_text_terms == ["arp299"]
        assert intent.authors == ["Kurtz"] and intent.first_author is True
        assert assemble_query(intent) == (
            'citations(bibcode:2017Natur.542..456G) author:"^Kurtz" abs:arp299'
        )

    def test_a_describing_author_leaves_and_takes_first_author_with_it(self):
        calls: list[dict] = []
        search = ads_search(lambda q: [GILLON])
        before = paper_intent(authors=["Gillon"], first_author=True)
        intent, _ = resolve_paper(
            TRAPPIST_TEXT, before, jev_answering_paper(GILLON["bibcode"], calls), search
        )
        assert intent.authors == [] and intent.first_author is False

    def test_a_low_confidence_pick_keeps_the_topic_search(self):
        calls: list[dict] = []
        search = ads_search(lambda q: [ORMEL, GILLON])
        before = paper_intent()
        client = jev_answering_paper(GILLON["bibcode"], calls, confidence=0.3)
        intent, lookup = resolve_paper(TRAPPIST_TEXT, before, client, search)
        assert intent == before
        assert lookup.bibcode is None and lookup.confidence == pytest.approx(0.3)

    def test_none_keeps_the_topic_search(self):
        calls: list[dict] = []
        search = ads_search(lambda q: [ORMEL])
        before = paper_intent()
        intent, lookup = resolve_paper(
            TRAPPIST_TEXT, before, jev_answering_paper("none", calls), search
        )
        assert intent == before
        assert lookup.bibcode is None and lookup.candidates == (ORMEL["bibcode"],)

    def test_no_candidates_skips_jev(self):
        calls: list[dict] = []
        search = ads_search(lambda q: [])
        before = paper_intent()
        intent, lookup = resolve_paper(
            TRAPPIST_TEXT, before, jev_answering_paper("none", calls), search
        )
        assert intent == before and calls == [] and lookup.candidates == ()

    def test_explicit_year_is_consumed_recency_is_kept(self):
        calls: list[dict] = []
        search = ads_search(lambda q: [GILLON])
        explicit = paper_intent(
            year_from=2017, year_to=2017, confidence={"refers_to_specific_paper": 0.9, "year": 0.9}
        )
        client = jev_answering_paper(GILLON["bibcode"], calls)
        resolved, _ = resolve_paper(TRAPPIST_TEXT, explicit, client, search)
        assert (resolved.year_from, resolved.year_to) == (None, None)
        recent, _ = resolve_paper(
            TRAPPIST_TEXT, paper_intent(year_from=2024, year_to=2026), client, search
        )
        assert (recent.year_from, recent.year_to) == (2024, 2026)


class TestPipelineLookup:
    def test_paper_named_in_what_does_x_cite_resolves(self):
        calls: list[dict] = []
        searches: list[str] = []
        search = ads_search(lambda q: [PENZIAS], searches)
        result = process_query(
            "what papers does the CMB discovery paper cite",
            "jev",
            jev_answering_paper(PENZIAS["bibcode"], calls, operator="references"),
            2026,
            paper_search=search,
        )
        assert searches[0] == 'abs:"cmb discovery"'
        assert result.final_query == "references(bibcode:1965ApJ...142..419P)"

    def test_unresolved_named_paper_keeps_its_topic(self):
        result = process_query(
            "what papers does the CMB discovery paper cite",
            "jev",
            answering_client(jev_payload("references")),
            2026,
        )
        assert result.final_query == 'references(abs:"cmb discovery")'

    def test_named_paper_resolves_to_bibcode(self):
        calls: list[dict] = []
        search = ads_search(lambda q: [ORMEL, GILLON])
        result = process_query(
            TRAPPIST_TEXT,
            "jev",
            jev_answering_paper(GILLON["bibcode"], calls),
            2026,
            paper_search=search,
        )
        assert result.final_query == "citations(bibcode:2017Natur.542..456G)"
        assert result.debug_info.paper_lookup["bibcode"] == GILLON["bibcode"]
        assert result.debug_info.paper_lookup_error is None

    def test_ads_failure_keeps_the_topic_search_and_says_why(self):
        calls: list[dict] = []
        search = ADSPaperSearch(
            api_key=TEST_API_KEY, transport=httpx.MockTransport(lambda r: httpx.Response(503))
        )
        result = process_query(
            TRAPPIST_TEXT, "jev", jev_answering_paper(GILLON["bibcode"], calls), 2026, search
        )
        assert result.final_query.startswith("citations(abs:")
        assert "HTTPStatusError" in result.debug_info.paper_lookup_error
        assert result.debug_info.paper_lookup is None

    def test_malformed_ads_doc_keeps_the_topic_search(self):
        calls: list[dict] = []
        search = ads_search(lambda q: [{"bibcode": "2000X", "citation_count": "N/A"}])
        result = process_query(
            TRAPPIST_TEXT, "jev", jev_answering_paper(GILLON["bibcode"], calls), 2026, search
        )
        assert result.final_query.startswith("citations(abs:")
        assert "PaperSearchError" in result.debug_info.paper_lookup_error

    def test_jev_failure_on_the_paper_question_keeps_the_topic_search(self):
        payload = jev_payload("citations", refers_to_specific_paper=noul_answer(0.92))

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if PAPER_QUESTION in body["questions"]:
                return httpx.Response(500)
            return httpx.Response(200, json=with_topic_answer(payload, body))

        search = ads_search(lambda q: [GILLON])
        result = process_query(TRAPPIST_TEXT, "jev", handler_client(handler), 2026, search)
        assert result.final_query.startswith("citations(abs:")
        assert result.debug_info.paper_lookup_error

    def test_without_paper_search_nothing_is_looked_up(self):
        calls: list[dict] = []
        result = process_query(
            TRAPPIST_TEXT, "jev", jev_answering_paper(GILLON["bibcode"], calls), 2026
        )
        assert result.final_query.startswith("citations(abs:")
        assert result.debug_info.paper_lookup is None
        assert len(calls) == 1

    def test_regex_backend_never_looks_up(self):
        calls: list[str] = []
        search = ads_search(lambda q: [GILLON], calls)
        result = process_query(TRAPPIST_TEXT, "regex", None, 2026, paper_search=search)
        assert calls == [] and result.debug_info.paper_lookup is None
