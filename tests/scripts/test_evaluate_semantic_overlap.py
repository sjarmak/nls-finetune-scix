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
    semantic_match_rate_of,
)
from jev_fixtures import jev_payload, with_topic_answer  # noqa: E402

from finetune.domains.scix.jev_intent import JevClient  # noqa: E402


def test_generate_via_pipeline_uses_jev_backend(tmp_path: Path):
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json=with_topic_answer(jev_payload("citations"), body))

    client = JevClient(
        api_key="k", cache_path=tmp_path / "c.jsonl", transport=httpx.MockTransport(handler)
    )
    nl = "papers that build on dark energy work by Smith"
    regex_query = generate_via_pipeline(nl)
    jev_query = generate_via_pipeline(nl, "jev", client)
    assert calls and calls[0]["state"]["query"] == nl
    assert "citations(" in jev_query
    assert regex_query != jev_query


def test_generate_via_pipeline_anchors_recency_to_benchmark_year():
    query = generate_via_pipeline("dark energy papers from the last 3 years")
    assert "pubdate:[2022 TO 2025]" in query


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


def test_semantic_match_rate_excludes_unscorable_items():
    from finetune.domains.scix.eval import GOLD_EMPTY, EvalResult

    def result(jaccard, valid=True, reason=None):
        return EvalResult("q", "g", "x", valid, [], [], [], jaccard, 0.0, 0.0, None, reason)

    results = [result(0.9), result(0.1), result(0.0, valid=False), result(0.0, reason=GOLD_EMPTY)]
    assert semantic_match_rate_of(results) == pytest.approx(1 / 3)
    assert semantic_match_rate_of([result(0.0, reason=GOLD_EMPTY)]) == 0.0
