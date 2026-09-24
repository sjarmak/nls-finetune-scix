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
import json
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
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
            "is asked for (a thesis, a book, conference proceedings, software, journal articles).",
            DOCTYPE_DESCRIPTIONS,
        ),
        "bibgroup": _choice(
            "Which curated telescope, mission or institution bibliography does the user "
            "restrict results to? Choose the facility when the user wants papers from, using, "
            "or by the team of that facility (JWST papers, ALMA observations, the Gaia mission). "
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


def _check_covers(name: str, descriptions: dict[str, str], values: frozenset[str]) -> None:
    expected = {NONE_OPTION, *values}
    if set(descriptions) != expected:
        missing = sorted(expected - set(descriptions))
        extra = sorted(set(descriptions) - expected)
        raise ValueError(f"{name} descriptions out of sync: missing={missing} extra={extra}")


# -----------------------------------------------------------------------------
# Request
# -----------------------------------------------------------------------------


def build_request(text: str, context: dict | None = None, model: str = JEV_MODEL) -> dict:
    """Build the System One request body. ``context`` is merged into ``state``."""
    state: dict = {"query": text}
    if context:
        if "query" in context:
            raise ValueError("context may not override the 'query' state key")
        state.update(context)
    return {"model": model, "state": state, "questions": build_questions()}


def request_fingerprint(request: dict) -> str:
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Response
# -----------------------------------------------------------------------------


def parse_response(
    payload: dict, latency_ms: float, cached: bool, fingerprint: str = ""
) -> JevAnswers:
    """Validate a System One response against the question set. Fails fast."""
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

    questions = build_questions()
    choices = {
        qid: _parse_choice(qid, answers.get(qid), questions[qid]) for qid in CHOICE_QUESTION_IDS
    }
    booleans = {qid: _parse_noul(qid, answers.get(qid)) for qid in BOOLEAN_QUESTION_IDS}
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
    """

    def __init__(
        self,
        api_key: str,
        cache_path: Path | None = DEFAULT_CACHE_PATH,
        model: str = JEV_MODEL,
        base_url: str = SYSTEM_ONE_URL,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
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
        self, text: str, context: dict | None = None, use_cache: bool = True
    ) -> JevAnswers:
        request = build_request(text, context=context, model=self.model)
        fingerprint = request_fingerprint(request)
        if use_cache:
            with self._lock:
                row = self._cache.get(fingerprint)
            if row is not None:
                return parse_response(
                    row["response"], row["latency_ms"], cached=True, fingerprint=fingerprint
                )

        started = time.perf_counter()
        response = self._http.post(self._base_url, json=request)
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as error:
            raise JevResponseError("response body is not JSON") from error
        answers = parse_response(payload, latency_ms, cached=False, fingerprint=fingerprint)
        self._record(fingerprint, request, payload, latency_ms)
        return answers

    def _record(self, fingerprint: str, request: dict, payload: dict, latency_ms: float) -> None:
        row = {
            "fingerprint": fingerprint,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "latency_ms": round(latency_ms, 1),
            "request": request,
            "response": payload,
        }
        with self._lock:
            self._cache[fingerprint] = row
            if self.cache_path is None:
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._http.close()


# -----------------------------------------------------------------------------
# Answers -> IntentSpec
# -----------------------------------------------------------------------------


def apply_answers(
    intent: IntentSpec,
    answers: JevAnswers,
    boolean_threshold: float = BOOLEAN_DECISION_THRESHOLD,
) -> IntentSpec:
    """Return a new IntentSpec with Jev's gating fields applied.

    Names, years and topics on ``intent`` are kept. Operator, doctype,
    bibgroup, collection and the three property flags are replaced by Jev's
    answers, and ``confidence`` holds the model's actual probabilities.
    """
    operator = answers.choices["operator"]
    doctype = answers.choices["doctype"]
    bibgroup = answers.choices["bibgroup"]
    collection = answers.choices["collection"]
    properties = {name for name in PROPERTY_BOOLEANS if answers.booleans[name] >= boolean_threshold}
    confidence = {
        "operator": operator.confidence,
        "search_kind": answers.choices["search_kind"].confidence,
        "doctype": doctype.confidence,
        "bibgroup": bibgroup.confidence,
        "database": collection.confidence,
        "needs_clarification": answers.booleans["needs_clarification"],
        "refers_to_specific_paper": answers.booleans["refers_to_specific_paper"],
        **{f"property.{name}": answers.booleans[name] for name in PROPERTY_BOOLEANS},
        **{
            k: v
            for k, v in intent.confidence.items()
            if k in ("year", "authors", "topics", "or_topics")
        },
    }
    return replace(
        intent,
        operator=None if operator.choice == NONE_OPTION else operator.choice,
        doctype=set() if doctype.choice == NONE_OPTION else {doctype.choice},
        bibgroup=set() if bibgroup.choice == NONE_OPTION else {bibgroup.choice},
        collection=set() if collection.choice == NONE_OPTION else {collection.choice},
        property=properties,
        confidence=confidence,
    )


def classify_and_extract(
    text: str,
    client: JevClient,
    include_regex_state: bool = False,
    use_cache: bool = True,
) -> tuple[IntentSpec, JevAnswers | None]:
    """Compose Jev gating answers with the regex extractors for names, years, topics.

    Returns the composed IntentSpec and the raw answers (None when the text is
    ADS syntax and Jev was not called). With ``include_regex_state`` the regex
    IntentSpec is sent to Jev as extra state (experiment arm D).
    """
    from .ner import extract_intent, extract_intent_with_operator

    regex_intent = extract_intent(text)
    if regex_intent.confidence.get("ads_passthrough"):
        return regex_intent, None
    context = {"regex_intent": regex_intent.to_dict()} if include_regex_state else None
    answers = client.classify(text, context=context, use_cache=use_cache)
    operator = answers.choices["operator"].choice
    base = extract_intent_with_operator(text, None if operator == NONE_OPTION else operator)
    return apply_answers(base, answers), answers


def extract_intent_jev(
    text: str,
    client: JevClient,
    include_regex_state: bool = False,
) -> IntentSpec:
    """IntentSpec from Jev gating plus regex names, years and topics."""
    intent, _ = classify_and_extract(text, client, include_regex_state=include_regex_state)
    return intent
