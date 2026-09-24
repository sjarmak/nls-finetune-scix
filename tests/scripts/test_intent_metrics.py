"""Unit tests for the pure metric functions used by evaluate_intent_classifiers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from intent_metrics import (  # noqa: E402
    accuracy_at_coverage,
    cost_summary,
    expected_calibration_error,
    extraction_metrics,
    false_positive_rate,
    latency_summary,
    operator_report,
    selective_curve,
    set_f1,
    stability,
)


def test_operator_report_macro_f1_and_per_class():
    pairs = [
        ("none", "none"),
        ("none", "citations"),
        ("citations", "citations"),
        ("references", "none"),
    ]
    rep = operator_report(pairs)
    assert rep["n"] == 4 and rep["accuracy"] == 0.5
    assert set(rep["per_class"]) == {"none", "citations", "references"}
    assert rep["per_class"]["citations"]["precision"] == 0.5
    assert rep["per_class"]["citations"]["recall"] == 1.0
    assert rep["per_class"]["references"]["f1"] == 0.0
    expected_macro = (rep["per_class"]["none"]["f1"] + 2 / 3 + 0.0) / 3
    assert rep["macro_f1"] == pytest.approx(expected_macro)


def test_false_positive_rate_counts_only_gold_none():
    pairs = [("none", "none"), ("none", "similar"), ("citations", "none"), ("none", "none")]
    fpr = false_positive_rate(pairs)
    assert fpr == {"n": 3, "false_positives": 1, "rate": pytest.approx(1 / 3)}


def test_set_f1_micro():
    pairs = [
        ({"refereed"}, {"refereed"}),
        (set(), {"eprint"}),
        ({"openaccess", "refereed"}, {"refereed"}),
    ]
    out = set_f1(pairs)
    assert out["tp"] == 2 and out["fp"] == 1 and out["fn"] == 1
    assert out["f1"] == pytest.approx(2 / 3)
    assert out["exact_match"] == pytest.approx(1 / 3)
    assert set_f1([(set(), set())])["f1"] is None


def test_selective_curve_and_coverage_target():
    rows = [(0.99, True), (0.9, True), (0.7, False), (0.55, True), (0.4, False)]
    curve = selective_curve(rows, thresholds=(0.5, 0.8))
    assert curve[0] == {"threshold": 0.5, "coverage": 0.8, "accuracy": 0.75, "n_answered": 4}
    assert curve[1]["n_answered"] == 2 and curve[1]["accuracy"] == 1.0
    at90 = accuracy_at_coverage(rows, 0.9)
    assert at90["coverage"] == 0.8 and at90["threshold"] == 0.55
    assert at90["accuracy"] == 0.75


def test_ece_perfect_and_overconfident():
    perfect = [(1.0, True)] * 5 + [(0.0, False)] * 5
    assert expected_calibration_error(perfect)["ece"] == pytest.approx(0.0)
    over = [(0.95, False)] * 4
    out = expected_calibration_error(over)
    assert out["ece"] == pytest.approx(0.95)
    assert out["bins"][9]["n"] == 4 and out["bins"][9]["accuracy"] == 0.0


def test_stability_fraction_changed():
    rows = [
        {"item_id": "a", "repeat": 1, "operator": "none", "doctype": ()},
        {"item_id": "a", "repeat": 2, "operator": "citations", "doctype": ()},
        {"item_id": "b", "repeat": 1, "operator": "none", "doctype": ("article",)},
        {"item_id": "b", "repeat": 2, "operator": "none", "doctype": ("article",)},
        {"item_id": "c", "repeat": 1, "operator": "none", "doctype": ()},
    ]
    out = stability(rows, ("operator", "doctype"))
    assert out["n_items"] == 2 and out["repeats"] == 2
    assert out["operator"] == 0.5 and out["doctype"] == 0.0


def test_latency_and_cost():
    lat = latency_summary([100.0, 200.0, 300.0, 400.0, 1000.0])
    assert lat["p50"] == 300.0 and lat["p95"] == pytest.approx(880.0) and lat["mean"] == 400.0
    assert latency_summary([])["p95"] is None
    rows = [
        {"input_tokens": 2000, "classifier_called": True},
        {"input_tokens": None, "classifier_called": False},
    ]
    cost = cost_summary(rows, rate_in_per_mtok=0.042)
    assert cost["mean_input_tokens"] == 1000.0
    assert cost["mean_usd_per_query"] == pytest.approx(2000 / 1e6 * 0.042 / 2)
    assert cost["classifier_call_rate"] == 0.5


def test_summary_table_tolerates_an_arm_with_no_rows():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from evaluate_intent_classifiers import arm_metrics, summary_table

    table = summary_table({"C": arm_metrics([])})
    assert "n/a" in table and "C llm_haiku_4_5" in table


def _row(labels: dict, prediction: dict) -> dict:
    base_labels = {
        "year_from": None,
        "year_to": None,
        "first_author": False,
        "has_author": False,
        "min_citations": None,
        "topic_tokens": [],
    }
    base_pred = {
        "year_from": None,
        "year_to": None,
        "first_author": False,
        "min_citations": None,
        "free_text_terms": [],
        "or_terms": [],
    }
    return {"labels": {**base_labels, **labels}, "prediction": {**base_pred, **prediction}}


def test_extraction_metrics_score_each_field():
    rows = [
        _row(
            {"year_from": 2023, "year_to": 2025, "topic_tokens": ["asteroids"]},
            {"year_from": 2023, "year_to": 2025, "free_text_terms": ["asteroids"]},
        ),
        _row(
            {"topic_tokens": ["dark", "matter"], "min_citations": 100},
            {"free_text_terms": ["recent dark matter"], "min_citations": 100},
        ),
        _row({"has_author": True, "first_author": True}, {"first_author": False}),
    ]
    m = extraction_metrics(rows)
    assert m["year"]["exact_match"] == pytest.approx(1.0)
    assert m["year"]["n"] == 3
    assert m["year_when_gold_has_one"]["n"] == 1
    assert m["first_author"]["n"] == 1
    assert m["first_author"]["exact_match"] == 0.0
    assert m["citation_floor"]["f1"] == pytest.approx(1.0)
    assert m["topic_tokens"]["tp"] == 3
    assert m["topic_tokens"]["fp"] == 1


def test_extraction_metrics_skip_rows_without_extraction_labels():
    assert extraction_metrics([{"labels": {"operator": "none"}, "prediction": {}}]) is None


def test_arm_metrics_report_extraction_for_derived_labels():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from evaluate_intent_classifiers import _derived, arm_metrics, run_regex, summary_table

    item = _derived(
        {
            "id": "g-1",
            "nl": "dark energy papers from the last 3 years",
            "gold_query": 'abs:"dark energy" pubdate:[2022 TO 2025]',
            "source": "gold",
        }
    )
    row = {**item, "arm": "A", "repeat": 1, "error": None, "prediction": run_regex(item["nl"])}
    metrics = arm_metrics([row])
    assert metrics["extraction"]["year"]["exact_match"] == 1.0
    assert metrics["extraction"]["topic_tokens"]["f1"] == 1.0
    assert "year EM" in summary_table({"A": metrics})
