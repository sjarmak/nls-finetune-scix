"""Jev (TypeSafe System One) typed classifiers for IntentSpec gating fields.

Jev answers bounded questions whose options are exactly the enum values that
are already legal in :class:`IntentSpec`. It never emits ADS syntax. This
module is IO, schema validation, caching and a mechanical mapping from
answers to IntentSpec fields; every semantic decision is the model's.

Wire contract (observed 2026-09-24 against ``jev-1.13.0``):

    POST https://api.typesafe.ai/v1/systemone
    authorization: Bearer <TYPESAFE_API_KEY>
    {"model": ..., "state": {...}, "questions": {id: {type, instructions, criteria}}}

Boolean questions are sent with ``type: "noul"`` and come back as
``{"type": "noul", "noul": p}``; choice questions come back as
``{"type": "choice", "choice", "confidence", "probabilities"}``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import httpx

from .field_constraints import BIBGROUPS, COLLECTIONS, DOCTYPES
from .intent_spec import OPERATORS, IntentSpec
from .jev_option_text import (
    BIBGROUP_DESCRIPTIONS,
    COLLECTION_DESCRIPTIONS,
    DOCTYPE_DESCRIPTIONS,
    OPERATOR_DESCRIPTIONS,
    SEARCH_KIND_DESCRIPTIONS,
)

SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"
DEFAULT_CACHE_PATH = Path("data/cache/jev_systemone.jsonl")

logger = logging.getLogger(__name__)
BOOLEAN_DECISION_THRESHOLD = 0.5
NONE_OPTION = "none"

CHOICE_QUESTION_IDS: tuple[str, ...] = (
    "operator",
    "search_kind",
    "doctype",
    "bibgroup",
    "collection",
)
BOOLEAN_QUESTION_IDS: tuple[str, ...] = (
    "refereed",
    "openaccess",
    "eprint",
    "needs_clarification",
    "refers_to_specific_paper",
)
PROPERTY_BOOLEANS: tuple[str, ...] = ("refereed", "openaccess", "eprint")

NAMED_TOPIC_QUESTION = "named_topic"
EXTRACTION_QUESTION_IDS: tuple[str, ...] = (
    "recency",
    "first_author",
    "highly_cited",
    "topic",
    NAMED_TOPIC_QUESTION,
)
MIN_JOINED_TOKENS = 3
"""Shortest topic phrase whose adjacent word pairs each get a ``join_<i>`` question."""
MAX_JOIN_QUESTIONS = 12
"""Most word-pair questions per query; a phrase that would pass the cap stays whole."""
MAX_AUTHOR_CANDIDATES = 10
"""Most names offered per query."""
MAX_AUTHOR_READINGS = 16
"""Most readings in the ``author_reading`` question: every reading of four single names."""
AUTHOR_READING_QUESTION = "author_reading"
PUBLICATION_YEAR_QUESTION = "publication_year"
_BARE_YEAR = re.compile(r"(?<![\w+\-/.])\d{4}(?![\w+\-/])")
EARLIEST_YEAR = 1800
_WORD = re.compile(r"[^\W\d_][\w'\-]*")
"""Questions that replace regex keyword rules. Kept out of ``build_questions`` so
the arm C (``llm_intent``) prompt, which mirrors the gating set, is unchanged."""
RECENCY_WINDOWS: dict[str, int] = {
    "this_year": 1,
    "last_2_years": 2,
    "last_3_years": 3,
    "last_5_years": 5,
    "last_10_years": 10,
}
"""Window length in years, ending at the request's reference year."""
HIGHLY_CITED_MIN_CITATIONS = 100
"""Most common floor for "highly cited" in the gold set (citation_count:[100 TO *])."""
MAX_TOPIC_TOKENS = 6
"""Longer topic phrases would offer too many sub-spans; the regex phrase is kept."""


class JevResponseError(ValueError):
    """The System One response did not match the expected contract."""


JEV_FAILURES: tuple[type[Exception], ...] = (httpx.HTTPError, JevResponseError)
"""Everything ``JevClient.classify`` raises when System One is unusable."""


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class JevAnswers:
    """Validated answers for one query, plus the bookkeeping the eval needs."""

    choices: dict[str, ChoiceAnswer]
    booleans: dict[str, float]
    model: str
    input_tokens: int
    latency_ms: float
    cached: bool
    fingerprint: str = ""
    raw: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Question set
# -----------------------------------------------------------------------------


def _choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _boolean(instructions: str, when_true: str, when_false: str) -> dict:
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {"true": when_true, "false": when_false},
    }


def build_questions() -> dict[str, dict]:
    """The fixed System One question set. Options are the legal enum values."""
    _check_covers("operator", OPERATOR_DESCRIPTIONS, OPERATORS)
    _check_covers("doctype", DOCTYPE_DESCRIPTIONS, DOCTYPES)
    _check_covers("bibgroup", BIBGROUP_DESCRIPTIONS, BIBGROUPS)
    _check_covers("collection", COLLECTION_DESCRIPTIONS, COLLECTIONS)
    return {
        "operator": _choice(
            "The user is searching the NASA ADS / SciX astronomy literature database. "
            "Which result-set operator, if any, do they want applied? Decide from what "
            "the user wants back, not from the words used: the same intent is phrased "
            "many ways. Choose 'none' when operator-like words (citations, references, "
            "similar, trending, useful, reviews) are the topic of the search rather than "
            "an instruction, and when a citation or read count is a numeric filter.",
            OPERATOR_DESCRIPTIONS,
        ),
        "search_kind": _choice(
            "What is the primary thing the user is searching by?",
            SEARCH_KIND_DESCRIPTIONS,
        ),
        "refereed": _boolean(
            "Does the user restrict results to peer-reviewed (refereed) publications?",
            "The user asks for peer-reviewed, refereed or published-journal results only.",
            "No peer-review restriction is expressed.",
        ),
        "openaccess": _boolean(
            "Does the user restrict results to open-access (freely readable) publications?",
            "The user asks for open access, free-to-read or freely available results.",
            "No open-access restriction is expressed.",
        ),
        "eprint": _boolean(
            "Does the user restrict results to preprints (arXiv e-prints)?",
            "The user asks for preprints, arXiv postings or e-prints.",
            "No preprint restriction is expressed.",
        ),
        "doctype": _choice(
            "Which single document type does the user restrict results to? Generic words "
            "such as papers, publications, articles, work, research or studies do not name a "
            "type; choose 'none' for them. Choose a type only when a specific kind of document "
            "is asked for (a thesis, a book, conference proceedings, software, journal articles). "
            "A word naming a data product in the topic (a flare catalog, a star catalog, an "
            "atlas) asks for papers about or presenting it: choose 'none'. Choose 'catalog' only "
            "when catalog records are the whole request (catalogs in the astronomy database, "
            "VizieR tables, catalog entries).",
            DOCTYPE_DESCRIPTIONS,
        ),
        "bibgroup": _choice(
            "Which curated telescope, mission or institution bibliography does the user "
            "restrict results to? Choose the facility when the user wants papers from, using, "
            "or by the team of that facility (JWST papers, ALMA observations, the Kepler mission). "
            "Choose 'none' when no facility is named, or when the name is part of a topic, "
            "object or person (Hubble constant, Hubble deep field, Chandrasekhar).",
            BIBGROUP_DESCRIPTIONS,
        ),
        "collection": _choice(
            "Which ADS discipline collection does the user restrict results to? "
            "Choose 'none' when no collection is requested.",
            COLLECTION_DESCRIPTIONS,
        ),
        "needs_clarification": _boolean(
            "Is the request too ambiguous or underspecified to search without asking the "
            "user a clarifying question?",
            "The request is ambiguous, contradictory or too vague to act on.",
            "The request is specific enough to search.",
        ),
        "refers_to_specific_paper": _boolean(
            "Does the request point at one specific paper or work that would have to be "
            "looked up before the search can run?",
            "A specific identifiable paper is referenced (by title, first author and year, "
            "a well-known result, or an identifier).",
            "The request is about a topic, author or field, not one particular paper.",
        ),
    }


def build_extraction_questions(
    topic_candidates: tuple[str, ...] = (), named_topic_candidates: tuple[str, ...] = ()
) -> dict[str, dict]:
    """Questions that decide what the regex keyword rules used to.

    ``topic`` is asked only when there are candidates; its options are the
    candidate phrases plus ``none``. ``named_topic`` is the same question over
    the phrase with the facility names left in, used when Jev picks no facility.
    """
    questions = {
        "recency": _choice(
            "Does the user limit results to recently published work without giving explicit "
            "years? Choose the window their wording implies. Choose 'none' when no recency is "
            "expressed, when explicit years or a date range are given, when a word such as "
            "'new' or 'up-to-date' is part of a name, title or topic (New Horizons, new "
            "physics), or when the user asks for trending or popular papers, which is about "
            "current reads, not publication date.",
            {
                NONE_OPTION: "No limit to recent work is expressed, including trending or "
                "popular papers.",
                "this_year": "Only work from the current year (this year, so far this year).",
                "last_2_years": "The newest work: latest, newest, new, just published, "
                "or from last year.",
                "last_3_years": "Recent work with no stated window: recent, recently, lately, "
                "current, nowadays.",
                "last_5_years": "The past few or past several years.",
                "last_10_years": "The past decade or the last ten years.",
            },
        ),
        "first_author": _boolean(
            "Does the user ask for papers where a named person is the first (lead) author, "
            "rather than any author?",
            "A named person is asked for as first author or lead author.",
            "A named person may be any author, or no person is named.",
        ),
        "highly_cited": _boolean(
            "Does the user restrict results to highly cited papers?",
            "The user wants only highly cited, heavily cited, well cited, influential, "
            "seminal or landmark papers, with no stated number.",
            "No citation-based restriction is expressed; citations are the operator (papers "
            "citing X) or the topic; the user asks to rank or sort (most cited, highest "
            "cited, top 10, top N by citations), which orders results rather than filtering "
            "them; the user states a citation count; or the user asks for popular or trending "
            "papers, which is about reads, not citations.",
        ),
    }
    if topic_candidates:
        questions["topic"] = _topic_question(topic_candidates)
    if named_topic_candidates:
        questions[NAMED_TOPIC_QUESTION] = _topic_question(named_topic_candidates)
    return questions


def bare_year(text: str, intent: IntentSpec, reference_year: int | None = None) -> int | None:
    """The first plausible year in ``text`` when the regex found no explicit years.

    "Jensen, E. 2020" and "hubble 1929" carry a year with no "in" or "since";
    so does "the Planck 2018 results", where it is a name. Jev decides which
    (``publication_year``). Catalog numbers (PSR 1913+16, SN 1987A) are not
    candidates, nor are years before 1800 or more than five years ahead.
    """
    if intent.year_from is not None or intent.year_to is not None:
        return None
    latest = (reference_year if reference_year is not None else datetime.now(UTC).year) + 5
    years = (int(m.group()) for m in _BARE_YEAR.finditer(text))
    return next((y for y in years if EARLIEST_YEAR <= y <= latest), None)


def _publication_year_question(year: int) -> dict:
    return _boolean(
        f"In this request, is {year} the publication year of the papers the user wants, or "
        "of the one paper they refer to?",
        f"{year} is when the wanted papers, or the referred-to paper, were published "
        f"(exoplanet atmospheres {year}, Jensen {year}, the Riess {year} paper).",
        f"{year} is part of a name or label that is not a publication date: a data release, "
        "survey or result named after a year (the Planck 2018 results were published in "
        "2020, DESI 2024), an object designation or an event.",
    )


def _topic_question(candidates: tuple[str, ...]) -> dict:
    return _choice(
        "Which phrase, taken from the request, names the subject the user wants papers "
        "about? Choose the phrase holding only the subject words: leave out words about "
        "recency, document type, citation counts or popularity (recent, latest, new, "
        "papers, highly cited). Choose 'none' when no offered phrase names the subject.",
        {
            NONE_OPTION: "No offered phrase names the subject.",
            **{c: f"The subject is exactly '{c}'." for c in candidates},
        },
    )


def _join_question(phrase: str, first: str, second: str) -> dict:
    return _boolean(
        f"In the search subject '{phrase}', do the adjacent words '{first}' and '{second}' "
        "belong to one fixed term that papers write together (a technical term or name such "
        "as 'dark matter', 'atmospheric escape', 'black hole', 'cosmic microwave background')?",
        f"'{first} {second}' is part of one fixed term, so the words should be searched as one "
        "phrase.",
        f"'{first}' and '{second}' are separate concepts; papers may use them apart.",
    )


def author_readings(names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Sets of ``names`` that could all be people at once: no two share a word.

    The empty reading comes first, then readings with more names; among
    readings of two or more names, those with fewer multi-word names come
    first, so "riess scolnic hubble constant" offers Riess and Scolnic as two
    people before the pairs of adjacent words fill the cap. At most
    ``MAX_AUTHOR_READINGS``. When the cap cuts, the last slot goes to every
    name as written (each name that shares no word with an earlier one), so
    a request listing five authors still offers all five. "Jarmak Cassini"
    gives (), ("Jarmak Cassini",), ("Jarmak",), ("Cassini",) and
    ("Jarmak", "Cassini").
    """
    readings = sorted(
        (
            people
            for k in range(len(names) + 1)
            for people in itertools.combinations(names, k)
            if all(not name_words(a) & name_words(b) for a, b in itertools.combinations(people, 2))
        ),
        key=lambda people: (len(people), len(people) > 1 and sum(" " in n for n in people)),
    )
    if len(readings) <= MAX_AUTHOR_READINGS:
        return tuple(readings)
    every: list[str] = []
    for name in names:
        if not any(name_words(name) & name_words(taken) for taken in every):
            every.append(name)
    kept = [r for r in readings[: MAX_AUTHOR_READINGS - 1] if r != tuple(every)]
    return (*kept, tuple(every))


def reading_key(people: tuple[str, ...]) -> str:
    return "|".join(people) or NONE_OPTION


def _quoted(names: list[str]) -> str:
    return " and ".join(f"'{name}'" for name in names)


def _reading_text(people: tuple[str, ...], names: tuple[str, ...]) -> str:
    taken = {word for person in people for word in name_words(person)}
    words = (w.strip(".,") for name in names for w in name.split() if not _is_initial(w))
    others = list(dict.fromkeys(w for w in words if w.lower() not in taken))
    if not people:
        text = "No one in the request is named as a person"
    elif len(people) == 1:
        text = f"{_quoted(list(people))} is one person, an author"
    else:
        text = f"{_quoted(list(people))} are different people, each an author"
    if others:
        verb = "is" if len(others) == 1 else "are"
        text += (
            f"; {_quoted(others)} {verb} not a person here but what the papers are about "
            "(a mission, spacecraft, telescope, survey, instrument, object, place, "
            "institution or theory), part of a topic phrase (Hawking radiation, Einstein "
            "ring), or an ordinary word"
        )
    return text + "."


def _author_reading_question(names: tuple[str, ...]) -> dict:
    return _choice(
        "Which of the capitalized names in this request are people the user names as "
        "authors (someone whose papers they want, or whose work the papers they want cite "
        "or are cited by), and which are not people? Missions, spacecraft, telescopes and "
        "surveys are often named after people (Cassini, Hubble, Kepler, Herschel, Planck, "
        "Gaia); in a search request such a name usually means the mission, not the person. "
        "In a short keyword request, a surname next to a subject usually names an author of "
        "papers on that subject.",
        {reading_key(people): _reading_text(people, names) for people in author_readings(names)},
    )


_NAME_TOKEN = re.compile(r"[^\W\d_][\w'\u2019\-]*\.?")
_POSSESSIVE = re.compile(r"['\u2019]s$")


_PARTICLES = frozenset({"van", "von", "de", "der", "den", "del", "della", "di", "da", "du", "ter"})
"""Lowercase surname particles that belong to the capitalized word after them (van Dokkum)."""


def _is_initial(word: str) -> bool:
    return len(word.rstrip(".")) == 1 and word[0].isupper()


def _is_particle(word: str) -> bool:
    return word in _PARTICLES


def _is_capitalized(word: str) -> bool:
    return word[0].isupper() and not (len(word.rstrip(".")) > 1 and word.rstrip(".").isupper())


def _name_runs(text: str) -> list[tuple[str, list[str]]]:
    """Runs of capitalized words and initials that touch in ``text`` ("Sara Seager", "Riess, A. G.").

    Each run is its text and its words. Words touch when only spaces, or a
    comma before an initial, separate them. Surname particles join the
    capitalized word after them ("de Grijs"). Possessive 's is dropped.
    All-caps acronyms (JWST) end a run.
    """
    runs: list[tuple[int, int, list[str]]] = []
    particles: list[re.Match[str]] = []
    for match in _NAME_TOKEN.finditer(text):
        word = _POSSESSIVE.sub("", match.group())
        if _is_particle(word):
            particles.append(match)
            continue
        if not _is_capitalized(word):
            runs.append((match.end(), match.end(), []))
            particles = []
            continue
        first = particles[0].start() if particles else match.start()
        joined = [*(p.group() for p in particles), word]
        start, end, words = runs[-1] if runs else (0, 0, [])
        gap = text[end:first].strip()
        if words and (gap == "" or (gap == "," and _is_initial(word) and not particles)):
            runs[-1] = (start, match.end(), [*words, *joined])
        else:
            runs.append((first, match.end(), joined))
        particles = []
    return [(_POSSESSIVE.sub("", text[start:end]), words) for start, end, words in runs if words]


def _surnames(run: list[str]) -> list[str]:
    """The full words of ``run`` with their particles ("de Grijs"), without initials."""
    names: list[str] = []
    particles: list[str] = []
    for word in run:
        if _is_particle(word):
            particles.append(word)
        elif not _is_initial(word):
            names.append(" ".join([*particles, word.rstrip(".")]))
            particles = []
    return names


def _run_candidates(span: str, run: list[str]) -> list[str]:
    """One span for a run of up to three surnames with its initials; single words otherwise too."""
    full = _surnames(run)
    if not full:
        return []
    spans = [span] if len(full) <= 3 and span not in full else []
    singles = full if len(full) > 1 or not spans else []
    return spans + singles


def _lowercase_runs(text: str, n: int) -> list[list[str]]:
    """Runs of ``n`` adjacent content words of an all-lowercase request."""
    from .ner import STOPWORDS

    if any(c.isupper() for c in text):
        return []
    words = [_POSSESSIVE.sub("", w) for w in _WORD.findall(text)]
    return [
        run
        for run in (words[i : i + n] for i in range(len(words) - n + 1))
        if all(w not in STOPWORDS and len(w) > 1 for w in run)
    ]


def _lowercase_pairs(text: str) -> list[str]:
    """Adjacent content-word pairs of an all-lowercase request (``sara seager exoplanets``)."""
    return [" ".join(run) for run in _lowercase_runs(text, 2)]


def _lowercase_triples(text: str) -> list[str]:
    """Adjacent content-word triples of an all-lowercase request (``jocelyn bell burnell``)."""
    return [" ".join(run) for run in _lowercase_runs(text, 3)]


def author_candidates(text: str, intent: IntentSpec) -> tuple[str, ...]:
    """Name spans, the regex names, then single words, at most ``MAX_AUTHOR_CANDIDATES``.

    A span is a full name written together ("Sara Seager", "A. G. Riess",
    "Riess, A. G.") or, in an all-lowercase request, a pair of adjacent words.
    Single words are offered too, so Jev can call "Madau Dickinson" two people
    and "accomazzi europa" one author plus a topic. Lowercase triples
    ("jocelyn bell burnell") come last, so a long request loses them to the
    cap before its pairs and single words. All-caps words (JWST, ALMA) are
    acronyms, not surnames, and are not offered.
    """
    runs = [_run_candidates(span, run) for span, run in _name_runs(text)]
    spans = [c for r in runs for c in r if " " in c]
    singles = [c for r in runs for c in r if " " not in c]
    pairs = _lowercase_pairs(text)
    lowercase_singles = [word for pair in pairs for word in pair.split()]
    unique: dict[str, str] = {}
    triples = _lowercase_triples(text)
    for name in (*spans, *pairs, *intent.authors, *singles, *lowercase_singles, *triples):
        unique.setdefault(name.lower(), name)
    return tuple(unique.values())[:MAX_AUTHOR_CANDIDATES]


def name_words(name: str) -> set[str]:
    """Lowercase words of ``name`` without initials, periods or commas.

    A hyphenated surname also gives its parts ("El-Badry": el-badry, el,
    badry), since the regex topic splits it there.
    """
    words = {w.lower().strip(".,") for w in name.split() if not _is_initial(w)}
    return {part for word in words for part in (word, *word.split("-"))} - {""}


def ads_author(name: str) -> str:
    """``name`` in ADS order: "Sara Seager" -> "Seager, Sara"; "Pieter van Dokkum" -> "van Dokkum, Pieter".

    A single surname or a name that already has a comma is kept; an
    all-lowercase name is capitalized, except its surname particles.
    """
    words = name.split()
    if name.islower():
        words = [w if _is_particle(w) else w[:1].upper() + w[1:] for w in words]
    last = len(words) - 1
    while last > 0 and _is_particle(words[last - 1]):
        last -= 1
    if "," in name or last == 0:
        return " ".join(words)
    return f"{' '.join(words[last:])}, {' '.join(words[:last])}"


def topic_candidates(intent: IntentSpec) -> tuple[str, ...]:
    """Contiguous sub-spans of the regex topic phrase, longest first.

    Empty (no topic question) unless the regex found exactly one topic phrase,
    no OR'd topics, and at most ``MAX_TOPIC_TOKENS`` words.
    """
    if intent.or_terms or len(intent.free_text_terms) != 1:
        return ()
    tokens = intent.free_text_terms[0].split()
    if len(tokens) > MAX_TOPIC_TOKENS:
        return ()
    spans = (
        " ".join(tokens[start : start + length])
        for length in range(len(tokens), 0, -1)
        for start in range(len(tokens) - length + 1)
    )
    return tuple(dict.fromkeys(s for s in spans if s.lower() != NONE_OPTION))


def word_pairs(*intents: IntentSpec) -> tuple[tuple[str, int], ...]:
    """(phrase, i) for each pair of adjacent words i, i+1 in the regex topic phrases of ``intents``.

    Only phrases of at least ``MIN_JOINED_TOKENS`` words are split, and a
    phrase whose pairs would pass ``MAX_JOIN_QUESTIONS`` is left whole. Each
    pair becomes a ``join_<k>`` question; Jev decides which pairs are one term.
    """
    pairs: list[tuple[str, int]] = []
    for phrase in dict.fromkeys(term for intent in intents for term in intent.free_text_terms):
        gaps = len(phrase.split()) - 1
        if gaps + 1 >= MIN_JOINED_TOKENS and len(pairs) + gaps <= MAX_JOIN_QUESTIONS:
            pairs += [(phrase, i) for i in range(gaps)]
    return tuple(pairs)


def _split_terms(terms: list[str], joined: dict[tuple[str, int], bool]) -> list[str]:
    """Each term cut between adjacent words Jev did not join.

    A term may be a sub-span of an asked phrase (the topic Jev chose); its
    gaps map to the phrase's gaps. A term from no asked phrase stays whole.
    """
    phrases = list(dict.fromkeys(phrase for phrase, _ in joined))
    split: list[str] = []
    for term in terms:
        words = term.split()
        offset = next(
            (
                (phrase, start)
                for phrase in phrases
                for start in range(len(phrase.split()) - len(words) + 1)
                if phrase.split()[start : start + len(words)] == words
            ),
            None,
        )
        if offset is None:
            split.append(term)
            continue
        phrase, start = offset
        current = [words[0]]
        for i, word in enumerate(words[1:]):
            if not joined[(phrase, start + i)]:
                split.append(" ".join(current))
                current = []
            current.append(word)
        split.append(" ".join(current))
    return split


def _name_pattern(name: str) -> re.Pattern[str]:
    """Whole-word pattern for a facility name, tolerant of spaces, hyphens and slashes."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    return re.compile(r"\b" + r"[\s\-/]*".join(map(re.escape, words)) + r"\b")


@cache
def _bibgroup_name_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Every name a bibgroup goes by: its code, its described name, and the NER synonyms."""
    from .ner import BIBGROUP_SYNONYMS

    names = [(code, code) for code in BIBGROUPS]
    names += [
        (code, re.split(r" \(| \||\.", text, maxsplit=1)[0])
        for code, text in BIBGROUP_DESCRIPTIONS.items()
        if code != NONE_OPTION
    ]
    names += [(code, synonym) for synonym, code in BIBGROUP_SYNONYMS.items()]
    return tuple((code, _name_pattern(name)) for code, name in names)


def bibgroup_candidates(text: str) -> tuple[str, ...]:
    """Bibgroups whose name appears in the request, in code order.

    This is string matching on facility names only. Whether a named facility is
    a restriction or part of a topic (Hubble constant) stays Jev's decision;
    code only keeps the other fifty-odd options out of the request.
    """
    lowered = text.lower()
    found = {code for code, pattern in _bibgroup_name_patterns() if pattern.search(lowered)}
    return tuple(sorted(found))


def _check_covers(name: str, descriptions: dict[str, str], values: frozenset[str]) -> None:
    expected = {NONE_OPTION, *values}
    if set(descriptions) != expected:
        missing = sorted(expected - set(descriptions))
        extra = sorted(set(descriptions) - expected)
        raise ValueError(f"{name} descriptions out of sync: missing={missing} extra={extra}")


# -----------------------------------------------------------------------------
# Request
# -----------------------------------------------------------------------------


def build_request(
    text: str,
    context: dict | None = None,
    model: str = JEV_MODEL,
    topic_candidates: tuple[str, ...] = (),
    bibgroup_candidates: tuple[str, ...] | None = None,
    author_candidates: tuple[str, ...] = (),
    word_pairs: tuple[tuple[str, int], ...] = (),
    named_topic_candidates: tuple[str, ...] = (),
    bare_year: int | None = None,
) -> dict:
    """Build the System One request body. ``context`` is merged into ``state``.

    ``author_candidates`` get one choice question, ``author_reading``, over
    which of them are people; each of ``word_pairs`` gets a yes/no question,
    ``join_<k>``; a ``bare_year`` gets ``publication_year``.

    With ``bibgroup_candidates`` the bibgroup question offers only those
    facilities plus ``none``, and is left out when there are none; without it
    the question offers every bibgroup.
    """
    state: dict = {"query": text}
    if context:
        if "query" in context:
            raise ValueError("context may not override the 'query' state key")
        state.update(context)
    questions = {
        **build_questions(),
        **build_extraction_questions(topic_candidates, named_topic_candidates),
    }
    if bibgroup_candidates is not None:
        bibgroup = questions.pop("bibgroup")
        if bibgroup_candidates:
            offered = (NONE_OPTION, *bibgroup_candidates)
            criteria = {k: v for k, v in bibgroup["criteria"].items() if k in offered}
            questions["bibgroup"] = {**bibgroup, "criteria": criteria}
    if author_candidates:
        questions[AUTHOR_READING_QUESTION] = _author_reading_question(author_candidates)
    if bare_year is not None:
        questions[PUBLICATION_YEAR_QUESTION] = _publication_year_question(bare_year)
    for k, (phrase, i) in enumerate(word_pairs):
        words = phrase.split()
        questions[f"join_{k}"] = _join_question(phrase, words[i], words[i + 1])
    return {"model": model, "state": state, "questions": questions}


def request_fingerprint(request: dict) -> str:
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Response
# -----------------------------------------------------------------------------


def parse_response(
    payload: dict,
    latency_ms: float,
    cached: bool,
    fingerprint: str = "",
    questions: dict[str, dict] | None = None,
) -> JevAnswers:
    """Validate a System One response against the questions asked. Fails fast.

    ``questions`` defaults to the request's question set without a topic question.
    """
    if not isinstance(payload, dict):
        raise JevResponseError("response is not an object")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise JevResponseError("response lacks a model id")
    usage = payload.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("input_tokens"), int):
        raise JevResponseError("response lacks usage.input_tokens")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise JevResponseError("response lacks answers")

    if questions is None:
        questions = build_request("")["questions"]
    choices = {
        qid: _parse_choice(qid, answers.get(qid), question)
        for qid, question in questions.items()
        if question["type"] == "choice"
    }
    booleans = {
        qid: _parse_noul(qid, answers.get(qid))
        for qid, question in questions.items()
        if question["type"] == "noul"
    }
    return JevAnswers(
        choices=choices,
        booleans=booleans,
        model=model,
        input_tokens=usage["input_tokens"],
        latency_ms=latency_ms,
        cached=cached,
        fingerprint=fingerprint,
        raw=payload,
    )


def _parse_choice(qid: str, answer: object, question: dict) -> ChoiceAnswer:
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise JevResponseError(f"{qid}: expected a choice answer, got {answer!r}")
    options = set(question["criteria"])
    choice = answer.get("choice")
    if choice not in options:
        raise JevResponseError(f"{qid}: choice {choice!r} is not one of the offered options")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise JevResponseError(f"{qid}: missing probabilities")
    parsed: dict[str, float] = {}
    for option, p in probabilities.items():
        if option not in options:
            raise JevResponseError(f"{qid}: probability for unknown option {option!r}")
        parsed[option] = _probability(qid, p)
    confidence = _probability(qid, answer.get("confidence"))
    return ChoiceAnswer(choice=choice, confidence=confidence, probabilities=parsed)


def _parse_noul(qid: str, answer: object) -> float:
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise JevResponseError(f"{qid}: expected a noul answer, got {answer!r}")
    return _probability(qid, answer.get("noul"))


def _probability(qid: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise JevResponseError(f"{qid}: probability {value!r} is not a number")
    if not 0.0 <= float(value) <= 1.0:
        raise JevResponseError(f"{qid}: probability {value!r} outside [0, 1]")
    return float(value)


# -----------------------------------------------------------------------------
# Client with JSONL cache
# -----------------------------------------------------------------------------


class JevClient:
    """Synchronous System One client with an append-only JSONL response cache.

    Cache rows are keyed by the SHA-256 of the canonical request body, so any
    change to the question text, model id or state produces a new row.

    ``classify`` raises ``httpx.HTTPError`` (status errors, timeouts, network
    errors) and ``JevResponseError`` (a body that is not JSON or breaks the
    answer contract). Those are the failure classes callers may degrade on;
    see ``JEV_FAILURES``.

    ``timeout_s`` is a wall-clock bound on each call. httpx applies its
    timeout per phase (connect, write, pool, and each socket read), so a
    response that trickles in could otherwise run far past it. Each POST runs
    on a small worker pool and ``classify`` stops waiting at the deadline with
    ``httpx.TimeoutException``; the abandoned request still ends at httpx's
    own per-phase limits. At most ``max_concurrent_calls`` requests are in
    flight; further calls queue, and the queue wait counts toward the deadline.

    The cache is best-effort. A cache file that cannot be written is logged
    and the answer is still returned. With ``cache_path=None`` nothing is
    memoised, so a long-running server does not grow without bound.
    """

    def __init__(
        self,
        api_key: str,
        cache_path: Path | None = DEFAULT_CACHE_PATH,
        model: str = JEV_MODEL,
        base_url: str = SYSTEM_ONE_URL,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        max_concurrent_calls: int = 16,
    ) -> None:
        if not api_key:
            raise ValueError("TYPESAFE_API_KEY is required for the Jev intent backend")
        self.model = model
        self.cache_path = cache_path
        self.timeout_s = timeout_s
        self._http = httpx.Client(
            headers={"authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )
        self._base_url = base_url
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrent_calls, thread_name_prefix="jev-http"
        )
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = self._load_cache() if cache_path else {}

    def _load_cache(self) -> dict[str, dict]:
        assert self.cache_path is not None
        if not self.cache_path.exists():
            return {}
        rows: dict[str, dict] = {}
        with self.cache_path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{self.cache_path}:{lineno}: corrupt cache row") from error
                rows[row["fingerprint"]] = row
        return rows

    def classify(
        self,
        text: str,
        context: dict | None = None,
        use_cache: bool = True,
        topic_candidates: tuple[str, ...] = (),
        bibgroup_candidates: tuple[str, ...] | None = None,
        author_candidates: tuple[str, ...] = (),
        word_pairs: tuple[tuple[str, int], ...] = (),
        named_topic_candidates: tuple[str, ...] = (),
        bare_year: int | None = None,
    ) -> JevAnswers:
        request = build_request(
            text,
            context=context,
            model=self.model,
            topic_candidates=topic_candidates,
            bibgroup_candidates=bibgroup_candidates,
            author_candidates=author_candidates,
            word_pairs=word_pairs,
            named_topic_candidates=named_topic_candidates,
            bare_year=bare_year,
        )
        return self.answer(request, use_cache=use_cache)

    def answer(self, request: dict, use_cache: bool = True) -> JevAnswers:
        """Send any System One request body and validate the answers to its questions."""
        fingerprint = request_fingerprint(request)
        questions = request["questions"]
        if use_cache:
            with self._lock:
                row = self._cache.get(fingerprint)
            if row is not None:
                return parse_response(
                    row["response"],
                    row["latency_ms"],
                    cached=True,
                    fingerprint=fingerprint,
                    questions=questions,
                )

        started = time.perf_counter()
        response = self._post(request)
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as error:
            raise JevResponseError("response body is not JSON") from error
        answers = parse_response(
            payload, latency_ms, cached=False, fingerprint=fingerprint, questions=questions
        )
        self._record(fingerprint, request, payload, latency_ms)
        return answers

    def _post(self, request: dict) -> httpx.Response:
        future = self._pool.submit(self._http.post, self._base_url, json=request)
        try:
            return future.result(timeout=self.timeout_s)
        except FutureTimeout as error:
            future.cancel()
            raise httpx.TimeoutException(
                f"no System One response within {self.timeout_s:g}s"
            ) from error

    def _record(self, fingerprint: str, request: dict, payload: dict, latency_ms: float) -> None:
        row = {
            "fingerprint": fingerprint,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "latency_ms": round(latency_ms, 1),
            "request": request,
            "response": payload,
        }
        if self.cache_path is None:
            return
        with self._lock:
            self._cache[fingerprint] = row
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as error:
                logger.warning("Jev cache write to %s failed: %s", self.cache_path, error)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._http.close()


# -----------------------------------------------------------------------------
# Answers -> IntentSpec
# -----------------------------------------------------------------------------


def apply_answers(
    intent: IntentSpec,
    answers: JevAnswers,
    boolean_threshold: float = BOOLEAN_DECISION_THRESHOLD,
    reference_year: int | None = None,
    author_candidates: tuple[str, ...] = (),
    word_pairs: tuple[tuple[str, int], ...] = (),
    named: IntentSpec | None = None,
    bare_year: int | None = None,
) -> IntentSpec:
    """Return a new IntentSpec with Jev's answers applied.

    With ``author_candidates`` the authors are the reading Jev picks; their
    words leave the topic, and a regex author Jev rejects joins it.
    ``named`` is ``intent`` with the facility names the regex took out of the
    topic left in; when Jev picks no facility its topic phrases and the
    ``named_topic`` answer are used, so "Hubble constant" keeps "hubble".
    With ``word_pairs`` each topic phrase is cut between the words Jev does
    not join into one term.

    Names and explicit years on ``intent`` are kept. Operator, doctype,
    bibgroup, collection and the three property flags are replaced by Jev's
    answers; bibgroup is empty when the question was not asked (no facility
    named). A ``bare_year`` Jev calls a publication year becomes the year
    range and leaves the topic. Jev also sets the recency window (only when no
    years were found; it ends at ``reference_year``, default the current year), the
    first-author flag (only when there are authors), the citation floor and,
    when a topic question was asked, the topic phrase. ``confidence`` holds
    the model's actual probabilities.
    """
    operator = answers.choices["operator"]
    doctype = answers.choices["doctype"]
    bibgroup = answers.choices.get("bibgroup")
    collection = answers.choices["collection"]
    recency = answers.choices["recency"]
    topic = answers.choices.get("topic")
    free_text_terms, or_terms = intent.free_text_terms, intent.or_terms
    if named is not None and (bibgroup is None or bibgroup.choice == NONE_OPTION):
        topic = answers.choices.get(NAMED_TOPIC_QUESTION)
        free_text_terms, or_terms = named.free_text_terms, named.or_terms
    join_probability = {pair: answers.booleans[f"join_{k}"] for k, pair in enumerate(word_pairs)}
    properties = {name for name in PROPERTY_BOOLEANS if answers.booleans[name] >= boolean_threshold}
    reading = answers.choices.get(AUTHOR_READING_QUESTION) if author_candidates else None
    readings = {reading_key(people): people for people in author_readings(author_candidates)}
    author_probability = {
        name: sum(p for key, p in reading.probabilities.items() if name in readings[key])
        for name in (author_candidates if reading else ())
    }
    confidence = {
        "operator": operator.confidence,
        "search_kind": answers.choices["search_kind"].confidence,
        "doctype": doctype.confidence,
        "database": collection.confidence,
        "recency": recency.confidence,
        "needs_clarification": answers.booleans["needs_clarification"],
        "refers_to_specific_paper": answers.booleans["refers_to_specific_paper"],
        "first_author": answers.booleans["first_author"],
        "highly_cited": answers.booleans["highly_cited"],
        **{f"property.{name}": answers.booleans[name] for name in PROPERTY_BOOLEANS},
        **{
            k: v
            for k, v in intent.confidence.items()
            if k in ("year", "authors", "topics", "or_topics")
        },
        **({"topic": topic.confidence} if topic else {}),
        **{f"join.{phrase}#{i}": p for (phrase, i), p in join_probability.items()},
        **({"bibgroup": bibgroup.confidence} if bibgroup else {}),
        **({AUTHOR_READING_QUESTION: reading.confidence} if reading else {}),
        **{f"author.{name}": p for name, p in author_probability.items()},
    }
    year_from, year_to = intent.year_from, intent.year_to
    year_probability = answers.booleans.get(PUBLICATION_YEAR_QUESTION) if bare_year else None
    publication_year = (
        year_probability is not None
        and year_probability >= boolean_threshold
        and year_from is None
        and year_to is None
    )
    if publication_year:
        year_from = year_to = bare_year
        confidence["year"] = year_probability
    if year_from is None and year_to is None and recency.choice != NONE_OPTION:
        year_to = reference_year if reference_year is not None else datetime.now(UTC).year
        year_from = year_to - RECENCY_WINDOWS[recency.choice] + 1
    if topic is not None:
        free_text_terms = [] if topic.choice == NONE_OPTION else [topic.choice]
    if join_probability:
        joined = {pair: p >= boolean_threshold for pair, p in join_probability.items()}
        free_text_terms = _split_terms(free_text_terms, joined)
    authors = intent.authors
    if reading is not None:
        authors = list(readings[reading.choice])
        people = {word for name in authors for word in name_words(name)}
        rejected = [name.lower() for name in intent.authors if not name_words(name) <= people]
        authors = [ads_author(name) for name in authors]
        kept = (" ".join(w for w in t.split() if w.lower() not in people) for t in free_text_terms)
        free_text_terms = [t for t in kept if t] + rejected
    if publication_year:
        year_word = str(bare_year)
        kept = (" ".join(w for w in t.split() if w != year_word) for t in free_text_terms)
        free_text_terms = [t for t in kept if t]
    highly_cited = answers.booleans["highly_cited"] >= boolean_threshold
    return replace(
        intent,
        operator=None if operator.choice == NONE_OPTION else operator.choice,
        doctype=set() if doctype.choice == NONE_OPTION else {doctype.choice},
        bibgroup=set() if bibgroup is None or bibgroup.choice == NONE_OPTION else {bibgroup.choice},
        collection=set() if collection.choice == NONE_OPTION else {collection.choice},
        property=properties,
        year_from=year_from,
        year_to=year_to,
        free_text_terms=free_text_terms,
        or_terms=or_terms,
        authors=authors,
        first_author=bool(authors) and answers.booleans["first_author"] >= boolean_threshold,
        min_citations=HIGHLY_CITED_MIN_CITATIONS if highly_cited else intent.min_citations,
        confidence=confidence,
    )


def classify_and_extract(
    text: str,
    client: JevClient,
    include_regex_state: bool = False,
    use_cache: bool = True,
    regex_intent: IntentSpec | None = None,
    reference_year: int | None = None,
) -> tuple[IntentSpec, JevAnswers | None]:
    """Compose Jev's answers with the regex extractors for names and explicit years.

    Returns the composed IntentSpec and the raw answers (None when the text is
    ADS syntax and Jev was not called). With ``include_regex_state`` the regex
    IntentSpec is sent to Jev as extra state (experiment arm D). Pass
    ``regex_intent`` when the caller already ran ``extract_intent(text)``.
    The regex topic phrase is only a candidate source: Jev picks one of its
    sub-spans (see ``topic_candidates``). When the regex took a facility name
    out of the topic, the phrase with the name left in is offered too
    (``named_topic``), for when Jev says no facility is meant. Relative dates end at
    ``reference_year``, default the current year.
    """
    from .ner import extract_intent, extract_intent_with_operator

    if regex_intent is None:
        regex_intent = extract_intent(text, reference_year)
    if regex_intent.confidence.get("ads_passthrough"):
        return regex_intent, None
    context = {"regex_intent": regex_intent.to_dict()} if include_regex_state else None
    named = extract_intent(text, reference_year, keep_facility_words=True)
    facility_in_topic = _topics(named) != _topics(regex_intent)
    names = author_candidates(text, regex_intent)
    year = bare_year(text, regex_intent, reference_year)
    pairs = word_pairs(regex_intent, *([named] if facility_in_topic else []))
    answers = client.classify(
        text,
        context=context,
        use_cache=use_cache,
        topic_candidates=topic_candidates(regex_intent),
        bibgroup_candidates=bibgroup_candidates(text),
        author_candidates=names,
        word_pairs=pairs,
        named_topic_candidates=topic_candidates(named) if facility_in_topic else (),
        bare_year=year,
    )
    operator = answers.choices["operator"].choice
    operator = None if operator == NONE_OPTION else operator
    base = extract_intent_with_operator(text, operator, reference_year)
    named_base = (
        extract_intent_with_operator(text, operator, reference_year, keep_facility_words=True)
        if facility_in_topic
        else None
    )
    intent = apply_answers(
        base,
        answers,
        reference_year=reference_year,
        author_candidates=names,
        word_pairs=pairs,
        named=named_base,
        bare_year=year,
    )
    return intent, answers


def _topics(intent: IntentSpec) -> tuple[list[str], list[str]]:
    return intent.free_text_terms, intent.or_terms


def extract_intent_jev(
    text: str,
    client: JevClient,
    include_regex_state: bool = False,
) -> IntentSpec:
    """IntentSpec from Jev's answers plus regex names and explicit years."""
    intent, _ = classify_and_extract(text, client, include_regex_state=include_regex_state)
    return intent
