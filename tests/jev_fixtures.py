"""Shared System One wire fixtures for tests that exercise the Jev backend.

Payloads mirror the smoke call recorded on 2026-09-24 against jev-1.13.0.
Clients run over an httpx MockTransport, so no test touches the network.
"""

import json
from collections.abc import Callable
from pathlib import Path

import httpx

from finetune.domains.scix.intent_spec import OPERATORS
from finetune.domains.scix.jev_intent import JEV_MODEL, JevClient

TEST_API_KEY = "test-key"


def choice_answer(choice: str, probabilities: dict[str, float]) -> dict:
    return {
        "type": "choice",
        "choice": choice,
        "confidence": probabilities[choice],
        "probabilities": probabilities,
    }


def noul_answer(p: float) -> dict:
    return {"type": "noul", "noul": p}


def operator_probabilities(operator: str, confidence: float | None = None) -> dict[str, float]:
    """Operator distribution with ``operator`` on top; the rest share the remainder."""
    options = ["none", *sorted(OPERATORS)]
    top = 1.0 - 0.01 * (len(options) - 1) if confidence is None else confidence
    rest = (1.0 - top) / (len(options) - 1)
    return {o: (top if o == operator else rest) for o in options}


def jev_payload(
    operator: str = "none", operator_confidence: float | None = None, **overrides
) -> dict:
    """A complete System One response for the question set without a topic question."""
    answers = {
        "operator": choice_answer(operator, operator_probabilities(operator, operator_confidence)),
        "search_kind": choice_answer(
            "topic",
            {
                "topic": 0.9,
                "author": 0.02,
                "object": 0.02,
                "paper_reference": 0.02,
                "identifier": 0.02,
                "mixed": 0.02,
            },
        ),
        "refereed": noul_answer(0.1),
        "openaccess": noul_answer(0.05),
        "eprint": noul_answer(0.02),
        "doctype": choice_answer("none", {"none": 0.97, "article": 0.03}),
        "bibgroup": choice_answer("none", {"none": 0.99, "HST": 0.01}),
        "collection": choice_answer("none", {"none": 0.8, "astronomy": 0.2}),
        "needs_clarification": noul_answer(0.1),
        "refers_to_specific_paper": noul_answer(0.2),
        "recency": choice_answer("none", {"none": 0.95, "last_3_years": 0.05}),
        "first_author": noul_answer(0.1),
        "highly_cited": noul_answer(0.05),
    }
    answers.update(overrides)
    usage = {"input_tokens": 500, "output_tokens": 0}
    return {"model": JEV_MODEL, "answers": answers, "usage": usage}


def handler_client(
    handler: Callable[[httpx.Request], httpx.Response],
    cache_path: Path | None = None,
    timeout_s: float = 30.0,
) -> JevClient:
    """A JevClient whose HTTP traffic is answered by ``handler``."""
    return JevClient(
        api_key=TEST_API_KEY,
        cache_path=cache_path,
        timeout_s=timeout_s,
        transport=httpx.MockTransport(handler),
    )


def with_topic_answer(payload: dict, request: dict) -> dict:
    """``payload`` plus defaults for the candidate questions it leaves unanswered.

    An unanswered topic gets the longest candidate, which is the whole regex
    phrase, so tests that do not set a topic keep the regex topic. An
    unanswered author question is yes exactly when the regex found that name,
    so tests that do not set authors keep the regex authors.
    """
    from finetune.domains.scix.ner import extract_intent

    questions, answers = request["questions"], dict(payload["answers"])
    topic = questions.get("topic")
    if topic is not None and "topic" not in answers:
        longest = next(option for option in topic["criteria"] if option != "none")
        answers["topic"] = choice_answer(longest, {"none": 0.05, longest: 0.95})
    phrasing = questions.get("phrasing")
    if phrasing is not None and "phrasing" not in answers:
        whole = next(iter(phrasing["criteria"]))
        answers["phrasing"] = choice_answer(whole, {whole: 1.0})
    regex_authors = {a.lower() for a in extract_intent(request["state"]["query"]).authors}
    for qid, question in questions.items():
        if qid.startswith("author_") and qid not in answers:
            named = any(f"'{a}'" in question["instructions"].lower() for a in regex_authors)
            answers[qid] = noul_answer(0.9 if named else 0.1)
    return {**payload, "answers": answers}


def answering_client(
    payload: dict, calls: list[dict] | None = None, cache_path: Path | None = None
) -> JevClient:
    """A JevClient that answers every request with ``payload`` (plus ``with_topic_answer``).

    Each request body is appended to ``calls`` when given.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        return httpx.Response(200, json=with_topic_answer(payload, body))

    return handler_client(handler, cache_path=cache_path)


def mock_jev_client(tmp_path: Path | None, payloads: list[dict], calls: list[dict]) -> JevClient:
    """A JevClient that answers the n-th request with ``payloads[n]``.

    Each request body is appended to ``calls``. With ``tmp_path`` the client
    keeps a JSONL cache there; with None it runs uncached. A topic question
    the payload leaves unanswered gets ``with_topic_answer``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {TEST_API_KEY}"
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json=with_topic_answer(payloads[len(calls) - 1], body))

    return handler_client(handler, cache_path=tmp_path / "cache.jsonl" if tmp_path else None)
