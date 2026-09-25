"""Tests for scripts/evaluate_keyword_queries.py and its dataset (no network)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_keyword_queries import (  # noqa: E402
    DATASET,
    author_key,
    authors_match,
    metrics_by_stratum,
    score_item,
    summarize,
    term_key,
    topics_match,
    word_prf,
)


def _gold(**overrides) -> dict:
    base = {
        "authors": [],
        "topic_terms": [],
        "bibgroup": [],
        "object": [],
        "year_from": None,
        "year_to": None,
        "operator": "none",
        "first_author": False,
    }
    return {**base, **overrides}


def _pred(**overrides) -> dict:
    gold = _gold(**overrides)
    gold.pop("object")
    return gold


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Seager, Sara", ("seager", "s")),
        ("Seager, S.", ("seager", "s")),
        ("Sara Seager", ("seager", "s")),
        ("seager", ("seager", None)),
        ("^Casey, C", ("casey", "c")),
        ("El-Badry, Kareem", ("elbadry", "k")),
        ("Bell Burnell, Jocelyn", ("bellburnell", "j")),
        ("Gómez", ("gomez", None)),
    ],
)
def test_author_key_normalises_order_case_and_initials(name, key):
    assert author_key(name) == key


def test_authors_match_ignores_order_and_missing_initials():
    assert authors_match(["Kurtz", "Accomazzi"], ["Accomazzi, A", "kurtz"])
    assert authors_match(["Seager, Sara"], ["Seager"])


def test_authors_match_rejects_wrong_initial_extra_or_missing_names():
    assert not authors_match(["Seager, Sara"], ["Seager, D"])
    assert not authors_match(["Accomazzi"], ["Europa, Accomazzi"])
    assert not authors_match(["Accomazzi"], ["Accomazzi", "Europa"])
    assert not authors_match(["Riess", "Scolnic"], ["Riess"])
    assert not authors_match(["Kurtz", "Kurtz"], ["Kurtz", "Accomazzi"])


def test_topics_match_respects_grouping_but_not_case_or_hyphens():
    assert topics_match(["tidal disruption events", "x-ray"], ["X ray", "Tidal disruption events"])
    assert not topics_match(["dark matter halo profiles"], ["dark matter", "halo profiles"])
    assert topics_match([], [])


def test_term_key_drops_punctuation():
    assert term_key('"Sgr A*"') == "sgr a"


def test_word_prf_ignores_grouping():
    assert word_prf(["dark matter halo profiles"], ["dark matter", "halo profiles"])["f1"] == 1.0
    out = word_prf(["europa"], ["accomazzi europa"])
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 0)
    assert word_prf([], [])["f1"] == 1.0
    assert word_prf(["cassini"], [])["f1"] == 0.0


def test_score_item_exact_needs_every_field():
    gold = _gold(authors=["Accomazzi"], topic_terms=["europa"])
    right = score_item(gold, _pred(authors=["Accomazzi, A"], topic_terms=["Europa"]))
    assert right["exact"] and right["authors"] and right["topic"]
    wrong = score_item(gold, _pred(authors=["Europa, Accomazzi"]))
    assert not wrong["exact"] and not wrong["authors"] and not wrong["topic"]
    assert wrong["bibgroup"] and wrong["years"] and wrong["operator"]


def test_score_item_years_bibgroup_operator():
    gold = _gold(topic_terms=["brown dwarfs"], bibgroup=["JWST"], year_from=2020, year_to=2020)
    pred = _pred(topic_terms=["brown dwarfs"], bibgroup=["jwst"], year_from=2020, year_to=None)
    scored = score_item(gold, pred)
    assert scored["bibgroup"] and not scored["years"] and not scored["exact"]
    assert not score_item(gold, {**pred, "operator": "reviews"})["operator"]


def _row(stratum: str, exact: bool, hits: int | None, gold_hits: int = 5) -> dict:
    gold = _gold(topic_terms=["x"])
    pred = _pred(topic_terms=["x" if exact else "y"])
    return {
        "stratum": stratum,
        "score": score_item(gold, pred),
        "ads_hits": hits,
        "ads_error": None if hits is not None else "HTTP 400",
        "gold_hits": gold_hits,
    }


def test_summarize_rates():
    rows = [_row("a", True, 10), _row("a", False, 0), _row("b", False, None, gold_hits=0)]
    out = summarize(rows)
    assert out["n"] == 3
    assert out["exact_accuracy"] == pytest.approx(1 / 3)
    assert out["hits_gt0_rate"] == pytest.approx(1 / 3)
    assert out["ads_error_rate"] == pytest.approx(1 / 3)
    assert out["hits_gt0_rate_where_gold_finds"] == pytest.approx(1 / 2)
    assert out["topic_word_f1_micro"] == pytest.approx(2 / (2 + 2 + 2))
    assert summarize([]) == {"n": 0}


def test_metrics_by_stratum_splits_rows():
    rows = [_row("a", True, 1), _row("b", False, 1)]
    out = metrics_by_stratum(rows)
    assert out["overall"]["n"] == 2
    assert out["by_stratum"]["a"]["exact_accuracy"] == 1.0
    assert out["by_stratum"]["b"]["exact_accuracy"] == 0.0


def test_dataset_is_well_formed():
    doc = json.loads(DATASET.read_text(encoding="utf-8"))
    items = doc["items"]
    assert len({i["id"] for i in items}) == len(items)
    assert Counter(i["stratum"] for i in items) == doc["strata"]
    for item in items:
        gold = item["gold"]
        assert set(gold) == {
            "authors",
            "topic_terms",
            "bibgroup",
            "object",
            "year_from",
            "year_to",
            "operator",
            "first_author",
        }, item["id"]
        assert item["query"] and item["intent"] and item["gold_query"], item["id"]
        assert isinstance(item["ambiguous"], bool), item["id"]
        assert not item["ambiguous"] or item.get("note"), item["id"]
        assert "object:" not in item["gold_query"], item["id"]
