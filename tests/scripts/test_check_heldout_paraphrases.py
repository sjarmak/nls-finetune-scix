"""Tests for scripts/check_heldout_paraphrases.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_heldout_paraphrases import (  # noqa: E402
    load_heldout,
    regex_problems,
    run_checks,
    validate_item,
)


def _item(**overrides) -> dict:
    base = {
        "id": "x-1",
        "stratum": "operator_positive",
        "nl": "papers that build on the Planck 2018 results",
        "labels": {
            "operator": "citations",
            "property": [],
            "doctype": [],
            "bibgroup": [],
            "collection": [],
            "search_kind": "paper_reference",
            "needs_clarification": False,
            "refers_to_specific_paper": True,
        },
    }
    base.update(overrides)
    return base


def test_schema_validation_catches_illegal_labels():
    bad = _item()
    bad["labels"] = {**bad["labels"], "operator": "topn", "doctype": ["novel"], "search_kind": "?"}
    problems = validate_item(bad)
    assert any("operator" in p for p in problems)
    assert any("doctype" in p for p in problems)
    assert any("search_kind" in p for p in problems)
    assert validate_item(_item()) == []


def test_in_pattern_positive_is_flagged():
    assert regex_problems(_item(nl="papers citing the Planck 2018 results"))
    assert regex_problems(_item()) == []


def test_in_map_enum_synonym_is_flagged():
    item = _item(
        stratum="enum_synonym", target_field="property", nl="open access papers on pulsars"
    )
    item["labels"] = {**item["labels"], "operator": "none", "property": ["openaccess"]}
    assert regex_problems(item)
    item["nl"] = "pulsar papers anyone can read without paying"
    assert regex_problems(item) == []


def test_shipped_heldout_file_passes_all_mechanical_checks():
    doc = load_heldout()
    failures, counts = run_checks(doc)
    assert failures == []
    assert counts["operator_positive"] >= 40
    assert counts["operator_negative"] >= 40
    assert counts["enum_synonym"] >= 40
    assert counts["ambiguous"] >= 30
    assert doc["excluded_from_comparison_arms"] == ["claude-fable-5-1"]
