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
    """A complete System One response for the standard question set."""
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


def mock_jev_client(tmp_path: Path | None, payloads: list[dict], calls: list[dict]) -> JevClient:
    """A JevClient that answers the n-th request with ``payloads[n]``.

    Each request body is appended to ``calls``. With ``tmp_path`` the client
    keeps a JSONL cache there; with None it runs uncached.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {TEST_API_KEY}"
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=payloads[len(calls) - 1])

    return handler_client(handler, cache_path=tmp_path / "cache.jsonl" if tmp_path else None)
