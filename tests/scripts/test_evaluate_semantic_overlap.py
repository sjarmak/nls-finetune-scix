"""Tests for the --intent-backend and --dataset wiring in evaluate_semantic_overlap."""

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from evaluate_semantic_overlap import (  # noqa: E402
    build_jev_client,
    generate_via_pipeline,
    load_cases,
)

from finetune.domains.scix.jev_intent import JevClient  # noqa: E402


def _jev_payload(operator: str) -> dict:
    options = ["none", "citations", "references", "similar", "trending", "useful", "reviews"]
    probs = {o: (1.0 if o == operator else 0.0) for o in options}
    choice = lambda c, opts: {  # noqa: E731
        "type": "choice",
        "choice": c,
        "confidence": 1.0,
        "probabilities": {o: (1.0 if o == c else 0.0) for o in opts},
    }
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 10},
        "answers": {
            "operator": {
                "type": "choice",
                "choice": operator,
                "confidence": 1.0,
                "probabilities": probs,
            },
            "search_kind": choice(
                "topic", ["topic", "author", "object", "paper_reference", "identifier", "mixed"]
            ),
            "doctype": choice("none", ["none"]),
            "bibgroup": choice("none", ["none"]),
            "collection": choice(
                "none", ["none", "astronomy", "physics", "general", "earthscience"]
            ),
            **{
                b: {"type": "noul", "noul": 0.0}
                for b in (
                    "refereed",
                    "openaccess",
                    "eprint",
                    "needs_clarification",
                    "refers_to_specific_paper",
                )
            },
        },
    }


def test_generate_via_pipeline_uses_jev_backend(tmp_path: Path):
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=_jev_payload("citations"))

    client = JevClient(
        api_key="k", cache_path=tmp_path / "c.jsonl", transport=httpx.MockTransport(handler)
    )
    nl = "papers that build on dark energy work by Smith"
    regex_query = generate_via_pipeline(nl)
    jev_query = generate_via_pipeline(nl, "jev", client)
    assert calls and calls[0]["state"]["query"] == nl
    assert "citations(" in jev_query
    assert regex_query != jev_query


def test_build_jev_client_requires_key(monkeypatch, tmp_path: Path):
    assert build_jev_client("regex", tmp_path / "c.jsonl") is None
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        build_jev_client("jev", tmp_path / "c.jsonl")


def test_load_cases_val_shape():
    cases = load_cases("val", Path("unused"))
    assert len(cases) > 400
    test, category, _ = cases[0]
    assert category == "val"
    assert test["natural_language"] and test["expected_query"] and test["id"].startswith("val-")
