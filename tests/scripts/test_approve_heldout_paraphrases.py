"""Tests for scripts/approve_heldout_paraphrases.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from approve_heldout_paraphrases import apply_review, parse_notes  # noqa: E402


def _doc() -> dict:
    return {
        "schema": "heldout-paraphrases-v1",
        "review_status": "pending",
        "items": [
            {"id": "a", "nl": "x", "review": {"status": "pending", "reviewer": None, "note": ""}},
            {"id": "b", "nl": "y", "review": {"status": "pending", "reviewer": None, "note": ""}},
            {"id": "c", "nl": "z", "review": {"status": "pending", "reviewer": None, "note": ""}},
        ],
    }


def test_approves_all_but_rejected_and_records_notes():
    out = apply_review(_doc(), reviewer="S", reject=["b"], notes={"a": "fine", "b": "unclear"})
    by_id = {i["id"]: i["review"] for i in out["items"]}
    assert by_id["a"] == {"status": "approved", "reviewer": "S", "note": "fine"}
    assert by_id["b"] == {"status": "rejected", "reviewer": "S", "note": "unclear"}
    assert by_id["c"] == {"status": "approved", "reviewer": "S", "note": ""}
    assert out["review_status"].startswith("reviewed by S on ")
    assert "2 approved, 1 rejected" in out["review_status"]


def test_input_document_is_not_mutated():
    doc = _doc()
    apply_review(doc, reviewer="S", reject=[], notes={})
    assert all(i["review"]["status"] == "pending" for i in doc["items"])


def test_unknown_id_is_an_error():
    with pytest.raises(SystemExit, match="unknown item id"):
        apply_review(_doc(), reviewer="S", reject=["nope"], notes={})
    with pytest.raises(SystemExit, match="unknown item id"):
        apply_review(_doc(), reviewer="S", reject=[], notes={"nope": "x"})


def test_parse_notes():
    assert parse_notes(["a=fine", "b=has = sign"]) == {"a": "fine", "b": "has = sign"}
    with pytest.raises(SystemExit, match="ID=text"):
        parse_notes(["novalue"])


def test_round_trip_writes_json(tmp_path):
    path = tmp_path / "h.json"
    path.write_text(json.dumps(_doc()))
    from approve_heldout_paraphrases import main

    assert main(["--path", str(path), "--reviewer", "S", "--reject", "c"]) == 0
    saved = json.loads(path.read_text())
    assert [i["review"]["status"] for i in saved["items"]] == ["approved", "approved", "rejected"]
