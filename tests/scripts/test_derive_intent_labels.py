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


@pytest.mark.parametrize(
    "query, years",
    [
        ("year:2023-2025", (2023, 2025)),
        ("year:2019", (2019, 2019)),
        ("pubdate:[2023-01 TO 2026-12]", (2023, 2026)),
        ("year:[2020 TO *]", (2020, None)),
        ("pubdate:[* TO 2010]", (None, 2010)),
        ('abs:"year of the comet"', (None, None)),
    ],
)
def test_year_range(query, years):
    labels = derive_labels(query)
    assert (labels.year_from, labels.year_to) == years


def test_first_author_and_citation_floor():
    assert derive_labels('author:"^Fry" author:"Fields"').first_author
    assert derive_labels('author:("^Parnell")').first_author
    assert not derive_labels('author:"Fields"').first_author
    assert derive_labels("abs:x citation_count:[100 TO *]").min_citations == 100
    assert derive_labels("abs:x").min_citations is None


def test_topic_tokens_come_from_abstract_and_title_clauses():
    labels = derive_labels(
        'abs:(black AND hole AND merger) title:"dark energy" author:"^Smith" abs:quasars'
    )
    assert labels.topic_tokens == {"black", "hole", "merger", "dark", "energy", "quasars"}


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
