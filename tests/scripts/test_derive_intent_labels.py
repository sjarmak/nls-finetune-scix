"""Tests for scripts/derive_intent_labels.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from derive_intent_labels import (  # noqa: E402
    derive_labels,
    load_benchmark_items,
    load_val_items,
)


@pytest.mark.parametrize(
    "query, operator, unsupported",
    [
        ('abs:"dark energy"', "none", None),
        ('citations(abs:"dark energy")', "citations", None),
        ("references(bibgroup:JWST)", "references", None),
        ("topn(10, bibgroup:JWST, citation_count)", "none", "topn"),
        ('similar(author:"Smith, J") year:2020', "similar", None),
        ('property:refereed trending(abs:"exoplanets")', "trending", None),
    ],
)
def test_operator_labels(query, operator, unsupported):
    labels = derive_labels(query)
    assert labels.operator == operator
    assert labels.unsupported_operator == unsupported


def test_enum_values_are_canonicalised_and_validated():
    labels = derive_labels(
        'doctype:phdthesis property:(refereed OR openaccess) bibgroup:"NASA PubSpace" '
        "database:astronomy bibgroup:jwst property:bogus"
    )
    assert labels.doctype == {"phdthesis"}
    assert labels.property == {"refereed", "openaccess"}
    assert labels.bibgroup == {"NASA PubSpace", "JWST"}
    assert labels.collection == {"astronomy"}
    assert labels.unparsed_values == ("property:bogus",)


def test_year_and_author_flags():
    assert derive_labels("pubdate:[2020-01 TO 2020-12] author:Smith").has_year
    assert derive_labels('author:"^Smith"').has_author
    assert not derive_labels('abs:"year of the comet"').has_year


def test_benchmark_operator_field_agrees_with_derived_labels():
    items = load_benchmark_items()
    checked = 0
    for item in items:
        expected = item["meta"]["operator"]
        if expected is None:
            continue
        labels = derive_labels(item["gold_query"])
        if expected == "topn":
            assert labels.operator == "none" and labels.unsupported_operator == "topn", item["id"]
        else:
            assert labels.operator == expected, item["id"]
        checked += 1
    assert checked >= 100


def test_benchmark_enum_field_agrees_with_derived_labels():
    field_map = {"database": "collection"}
    checked = 0
    for item in load_benchmark_items():
        field, value = item["meta"]["enum_field"], item["meta"]["enum_value"]
        if not field:
            continue
        labels = derive_labels(item["gold_query"]).to_dict()
        assert value in labels[field_map.get(field, field)], item["id"]
        checked += 1
    assert checked >= 50


def test_val_loader_strips_prompt_scaffolding():
    items = load_val_items()
    assert len(items) > 400
    assert not any(i["nl"].startswith("Query:") or "\nDate:" in i["nl"] for i in items)
    assert all(i["gold_query"] for i in items)
