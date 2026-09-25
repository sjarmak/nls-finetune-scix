"""Resolve the one paper a request names to its bibcode: code proposes, Jev decides.

"Papers that cite the original TRAPPIST-1 seven-planet paper" is about one
paper, but the topic words alone search for every paper that mentions them.
When Jev says the request names a specific paper and the operator needs a
target (citations, references, similar), code searches ADS with the topic
words and author names, pools the most-cited hits, and asks Jev which
candidate the request means. The chosen paper becomes the operator's target,
``citations(bibcode:2017Natur.542..456G)``. In the same request Jev says, for
each topic term and author, whether it describes the named paper or narrows
the papers the user wants ("papers about ARP299 that cite <title>"): the
describing ones leave the query, the others stay outside the operator. When
Jev picks none, or picks with confidence below the decision threshold, the
query stays a topic search.

Searches: all topic terms together in abstracts, then in full text (the
GW150914 abstract never says "LIGO"), then each term alone (a descriptive
word such as "original" narrows the combined search to nothing), or, for a
single phrase, each of its words ("2mass all-sky survey" is not how the 2MASS
paper words it), then the author names without the terms (the Salpeter 1955
record has no abstract). Every search keeps the author names and any explicit
year. Explicit years describe the paper ("the Riess 1998 paper"); a recency
window Jev set ("recent papers citing ...") stays on the citing papers,
outside the operator.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import httpx

from .assembler import _quote_value, assemble_query
from .intent_spec import IntentSpec
from .jev_intent import BOOLEAN_DECISION_THRESHOLD, NONE_OPTION, JevClient

ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
ADS_TIMEOUT_S = 10.0
OPERATORS_WITH_TARGET = frozenset({"citations", "references", "similar"})
CANDIDATES_PER_SEARCH = 5
MAX_TERM_SEARCHES = 4
MAX_SEARCHES = 2 + MAX_TERM_SEARCHES
MAX_CANDIDATES = 12
PAPER_QUESTION = "paper"
DESCRIBES_PREFIX = "describes_"


class PaperSearchError(ValueError):
    """The ADS response did not match the expected contract."""


PAPER_LOOKUP_FAILURES: tuple[type[Exception], ...] = (httpx.HTTPError, PaperSearchError)
"""What ``ADSPaperSearch.top_cited`` raises when ADS is unusable."""


@dataclass(frozen=True)
class PaperCandidate:
    bibcode: str
    title: str
    first_author: str
    year: str
    citation_count: int

    def describe(self) -> str:
        return (
            f'"{self.title}" by {self.first_author} ({self.year}), '
            f"cited {self.citation_count} times"
        )


@dataclass(frozen=True)
class PaperLookup:
    """What the lookup did, for debug output and telemetry."""

    searches: tuple[str, ...]
    candidates: tuple[str, ...]
    bibcode: str | None = None
    title: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "searches": list(self.searches),
            "candidates": list(self.candidates),
            "bibcode": self.bibcode,
            "title": self.title,
            "confidence": self.confidence,
        }


class ADSPaperSearch:
    """ADS search client for candidate papers, most cited first."""

    def __init__(
        self,
        api_key: str,
        timeout_s: float = ADS_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        max_concurrent_searches: int = MAX_SEARCHES,
    ) -> None:
        if not api_key:
            raise ValueError("ADS_API_KEY is required for paper lookup")
        self._http = httpx.Client(
            headers={"authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrent_searches, thread_name_prefix="paper-search"
        )

    def top_cited(self, query: str, rows: int = CANDIDATES_PER_SEARCH) -> list[PaperCandidate]:
        response = self._http.get(
            ADS_SEARCH_URL,
            params={
                "q": query,
                "rows": rows,
                "sort": "citation_count desc",
                "fl": "bibcode,title,first_author,year,citation_count",
            },
        )
        response.raise_for_status()
        try:
            docs = response.json()["response"]["docs"]
        except (ValueError, KeyError, TypeError) as error:
            raise PaperSearchError(f"ADS response for {query!r} lacks response.docs") from error
        return [_candidate(doc) for doc in docs]

    def search_all(self, queries: tuple[str, ...]) -> list[list[PaperCandidate]]:
        """``top_cited`` for each query, run concurrently, results in query order."""
        return list(self._pool.map(self.top_cited, queries))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._http.close()


def _candidate(doc: object) -> PaperCandidate:
    if not isinstance(doc, dict) or not isinstance(doc.get("bibcode"), str):
        raise PaperSearchError(f"ADS doc without a bibcode: {doc!r}")
    titles = doc.get("title") or ["(untitled)"]
    citation_count = doc.get("citation_count", 0)
    if (
        not isinstance(titles, list)
        or isinstance(citation_count, bool)
        or not isinstance(citation_count, int)
    ):
        raise PaperSearchError(f"ADS doc with a malformed title or citation_count: {doc!r}")
    return PaperCandidate(
        bibcode=doc["bibcode"],
        title=str(titles[0]),
        first_author=str(doc.get("first_author") or "unknown author"),
        year=str(doc.get("year") or "unknown year"),
        citation_count=citation_count,
    )


def needs_lookup(intent: IntentSpec, threshold: float = BOOLEAN_DECISION_THRESHOLD) -> bool:
    """True when Jev says the request names one paper and the operator needs a target."""
    return (
        intent.operator in OPERATORS_WITH_TARGET
        and not intent.operator_target
        and intent.confidence.get("refers_to_specific_paper", 0.0) >= threshold
    )


def _has_explicit_year(intent: IntentSpec) -> bool:
    return "year" in intent.confidence


def _full_text_search(describe: IntentSpec, terms: list[str]) -> str:
    """``describe`` with ``terms`` searched in the full text instead of abstracts."""
    full = " ".join(f"full:{_quote_value(t)}" for t in terms)
    return " ".join(q for q in (assemble_query(describe), full) if q)


def _narrower_term_sets(terms: list[str]) -> list[list[str]]:
    """Each term alone, or each word of a lone phrase, capped."""
    parts = terms[0].split() if len(terms) == 1 else terms
    return [[p] for p in parts[:MAX_TERM_SEARCHES]] if len(parts) > 1 else []


def candidate_searches(intent: IntentSpec) -> tuple[str, ...]:
    """ADS queries for candidate papers, at most ``MAX_SEARCHES``.

    All terms together in abstracts and in full text, then each term (or word of
    a lone phrase) alone. Authors are searched alone only when there are no
    terms: an author-only search offers the author's other well-cited papers,
    which splits the pick between them.
    """
    years = (
        {"year_from": intent.year_from, "year_to": intent.year_to}
        if _has_explicit_year(intent)
        else {}
    )
    describe = IntentSpec(authors=list(intent.authors), first_author=intent.first_author, **years)
    terms = intent.free_text_terms
    if not terms:
        return tuple(q for q in [assemble_query(describe)] if q)
    combined, *narrower = (
        assemble_query(replace(describe, free_text_terms=list(ts)))
        for ts in [terms, *_narrower_term_sets(terms)]
    )
    queries = [combined, _full_text_search(describe, terms), *narrower]
    return tuple(dict.fromkeys(q for q in queries if q))


def pool_candidates(results: list[list[PaperCandidate]]) -> list[PaperCandidate]:
    """Round-robin by rank across searches, first occurrence wins, capped."""
    pooled: dict[str, PaperCandidate] = {}
    for rank in range(max((len(r) for r in results), default=0)):
        for result in results:
            if rank < len(result):
                pooled.setdefault(result[rank].bibcode, result[rank])
    return list(pooled.values())[:MAX_CANDIDATES]


def _describes_question(descriptor: str) -> dict:
    return {
        "type": "noul",
        "instructions": (
            f"In this request, is '{descriptor}' part of how the user names or describes the "
            "one paper they refer to (for example 'the X paper', 'X results', 'X et al. work', "
            "'cite \"<title>\"'), rather than a separate condition on the papers they want "
            "back (for example 'X papers that cite ...', 'papers by X citing ...')?"
        ),
        "criteria": {
            "true": f"'{descriptor}' is part of naming the referred-to paper.",
            "false": f"'{descriptor}' is a separate condition on the papers the user wants back.",
        },
    }


def _descriptors(intent: IntentSpec) -> tuple[str, ...]:
    """Topic terms, then authors: what may describe the named paper."""
    return (*intent.free_text_terms, *intent.authors)


def paper_request(
    text: str, candidates: list[PaperCandidate], model: str, descriptors: tuple[str, ...] = ()
) -> dict:
    """The paper choice plus one ``describes_<i>`` yes/no question per descriptor."""
    criteria = {
        NONE_OPTION: "None of these candidates is the paper the request refers to.",
        **{c.bibcode: c.describe() for c in candidates},
    }
    question = {
        "type": "choice",
        "instructions": (
            "The request refers to one specific paper. Which of these candidate papers "
            "is it? Answer none when no candidate is that paper."
        ),
        "criteria": criteria,
    }
    questions = {
        PAPER_QUESTION: question,
        **{f"{DESCRIBES_PREFIX}{i}": _describes_question(d) for i, d in enumerate(descriptors)},
    }
    return {"model": model, "state": {"query": text}, "questions": questions}


def resolve_paper(
    text: str,
    intent: IntentSpec,
    client: JevClient,
    search: ADSPaperSearch,
    threshold: float = BOOLEAN_DECISION_THRESHOLD,
) -> tuple[IntentSpec, PaperLookup]:
    """Return ``intent`` with the named paper as operator target, when Jev picks one.

    A pick below ``threshold`` confidence is not applied. Terms and authors
    Jev says describe the paper leave the intent; the rest stay.

    Raises what ``ADSPaperSearch`` and ``JevClient.answer`` raise
    (``PAPER_LOOKUP_FAILURES``, ``JEV_FAILURES``).
    """
    searches = candidate_searches(intent)
    if not searches:
        return intent, PaperLookup(searches=(), candidates=())
    candidates = pool_candidates(search.search_all(searches))
    offered = tuple(c.bibcode for c in candidates)
    if not candidates:
        return intent, PaperLookup(searches=searches, candidates=offered)
    descriptors = _descriptors(intent)
    answers = client.answer(paper_request(text, candidates, client.model, descriptors))
    answer = answers.choices[PAPER_QUESTION]
    lookup = PaperLookup(searches=searches, candidates=offered, confidence=answer.confidence)
    if answer.choice == NONE_OPTION or answer.confidence < threshold:
        return intent, lookup
    chosen = next(c for c in candidates if c.bibcode == answer.choice)
    keep = [answers.booleans[f"{DESCRIBES_PREFIX}{i}"] < threshold for i in range(len(descriptors))]
    n_terms = len(intent.free_text_terms)
    terms = [t for t, k in zip(intent.free_text_terms, keep[:n_terms], strict=True) if k]
    authors = [a for a, k in zip(intent.authors, keep[n_terms:], strict=True) if k]
    explicit_year = _has_explicit_year(intent)
    resolved = replace(
        intent,
        operator_target=chosen.bibcode,
        free_text_terms=terms,
        or_terms=[],
        authors=authors,
        first_author=intent.first_author and bool(authors),
        year_from=None if explicit_year else intent.year_from,
        year_to=None if explicit_year else intent.year_to,
        confidence={**intent.confidence, PAPER_QUESTION: answer.confidence},
    )
    return resolved, replace(lookup, bibcode=chosen.bibcode, title=chosen.title)
