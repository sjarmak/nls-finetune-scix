"""Tests for scripts/summarize_intent_shadow.py (mechanical aggregation only)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from summarize_intent_shadow import (  # noqa: E402
    format_summary,
    load_shadow_rows,
    main,
    summarize,
)


def _intent(operator=None, doctype=(), bibgroup=(), collection=(), prop=()) -> dict:
    return {
        "operator": operator,
        "doctype": list(doctype),
        "bibgroup": list(bibgroup),
        "collection": list(collection),
        "property": list(prop),
        "free_text_terms": ["ignored"],
    }


def _shadow(
    nl,
    served,
    shadow,
    disagreements,
    called=True,
    error=None,
    conf=0.9,
    path="pipeline",
    cached=False,
) -> dict:
    return {
        "record_type": "intent_shadow",
        "timestamp": "2026-09-24T00:00:00+00:00",
        "request_id": nl,
        "nl_query": nl,
        "served_backend": "regex",
        "served_path": path,
        "shadow_backend": "jev_gated",
        "served_intent": served,
        "shadow_intent": shadow,
        "disagreements": disagreements,
        "disagree": bool(disagreements),
        "classifier_called": called,
        "classifier_cached": cached,
        "classifier_error": error,
        "classifier_operator_confidence": None if error or not called else conf,
        "shadow_latency_ms": 200.0,
    }


ROWS = [
    _shadow("a", _intent(), _intent("similar"), ["operator"], conf=0.8),
    _shadow(
        "b",
        _intent(doctype=["article"]),
        _intent("citations"),
        ["operator", "doctype"],
        conf=0.44,
    ),
    _shadow("c", _intent("citations"), _intent("citations"), [], called=False),
    _shadow("d", _intent(), _intent(), [], error="ReadTimeout: timed out"),
]


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "telemetry.jsonl"
    request_row = {"record_type": "request", "nl_query": "x", "path": "pipeline"}
    lines = [json.dumps(request_row)] + [json.dumps(r) for r in rows] + [""]
    path.write_text("\n".join(lines))
    return path


def test_load_keeps_only_shadow_rows(tmp_path):
    rows = load_shadow_rows(_write(tmp_path, ROWS))
    assert [r["nl_query"] for r in rows] == ["a", "b", "c", "d"]


def test_load_rejects_corrupt_json_with_line_number(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"record_type": "intent_shadow"\n')
    with pytest.raises(ValueError, match=":1:"):
        load_shadow_rows(path)


def test_load_rejects_shadow_rows_missing_fields(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"record_type": "intent_shadow", "nl_query": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        load_shadow_rows(path)


def test_summary_counts():
    summary = summarize(ROWS)
    assert summary["shadow_rows"] == 4
    assert summary["classifier_called"] == 3
    assert summary["classifier_errors"] == 1
    assert summary["disagreeing"] == 2
    assert summary["by_field"] == {
        "operator": 2,
        "doctype": 1,
        "bibgroup": 0,
        "collection": 0,
        "property": 0,
        "year_from": 0,
        "year_to": 0,
        "free_text_terms": 0,
        "first_author": 0,
        "min_citations": 0,
        "ranking": 0,
        "ranking_limit": 0,
    }


def test_summary_lists_disagreeing_queries_with_both_values():
    queries = summarize(ROWS)["disagreeing_queries"]
    assert [q["nl_query"] for q in queries] == ["a", "b"]
    assert queries[1]["served"] == {"operator": None, "doctype": ["article"]}
    assert queries[1]["shadow"] == {"operator": "citations", "doctype": []}
    assert queries[1]["classifier_operator_confidence"] == 0.44


def test_empty_log_summarizes_to_zeros():
    summary = summarize([])
    assert summary["shadow_rows"] == 0 and summary["disagreeing_queries"] == []


def test_format_mentions_counts_and_queries():
    text = format_summary(summarize(ROWS))
    assert "shadow rows: 4" in text
    assert "operator: 2" in text
    assert "b" in text and "citations" in text


def test_main_prints_summary(tmp_path, capsys):
    assert main([str(_write(tmp_path, ROWS))]) == 0
    assert "disagreeing (pipeline-served rows): 2" in capsys.readouterr().out


def test_main_json_output(tmp_path, capsys):
    assert main([str(_write(tmp_path, ROWS)), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["disagreeing"] == 2


def test_model_served_rows_are_counted_but_not_compared():
    rows = [
        *ROWS,
        _shadow("e", _intent(), _intent("useful"), ["operator"], path="model"),
    ]
    summary = summarize(rows)
    assert summary["shadow_rows"] == 5
    assert summary["served_by_model"] == 1
    assert summary["disagreeing"] == 2
    assert [q["nl_query"] for q in summary["disagreeing_queries"]] == ["a", "b"]


def test_cache_answers_are_not_counted_as_billed():
    rows = [*ROWS, _shadow("e", _intent(), _intent(), [], cached=True)]
    summary = summarize(rows)
    assert summary["classifier_called"] == 4
    assert summary["classifier_billed"] == 3
