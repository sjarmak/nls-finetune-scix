"""Shadow comparison of the served regex intent against a Jev backend."""

import httpx
import pytest
from jev_fixtures import answering_client, choice_answer, handler_client, jev_payload

from finetune.domains.scix.intent_shadow import (
    SHADOW_COMPARED_FIELDS,
    SHADOW_RECORD_TYPE,
    intent_disagreements,
    shadow_record,
)
from finetune.domains.scix.ner import extract_intent

GATED_QUERY = "work along the lines of the Planck results"


def _answering(payload: dict):
    return answering_client(payload)


def test_compared_fields_are_the_ones_jev_decides():
    assert SHADOW_COMPARED_FIELDS == (
        "operator",
        "doctype",
        "bibgroup",
        "collection",
        "property",
        "year_from",
        "year_to",
        "free_text_terms",
        "first_author",
        "min_citations",
    )


def test_disagreements_list_differing_fields_in_a_fixed_order():
    served = extract_intent("refereed papers on quasars").to_dict()
    shadow = {**served, "operator": "similar", "property": [], "bibgroup": ["JWST"]}
    assert intent_disagreements(served, shadow) == ["operator", "bibgroup", "property"]


def test_identical_intents_agree():
    served = extract_intent("refereed papers on quasars").to_dict()
    assert intent_disagreements(served, dict(served)) == []


def test_record_captures_both_intents_and_the_disagreement():
    served = extract_intent(GATED_QUERY).to_dict()
    record = shadow_record(
        GATED_QUERY, served, "jev_gated", _answering(jev_payload("similar", 0.81)), "pipeline"
    )
    assert record["record_type"] == SHADOW_RECORD_TYPE
    assert record["nl_query"] == GATED_QUERY
    assert record["served_backend"] == "regex"
    assert record["served_path"] == "pipeline"
    assert record["shadow_backend"] == "jev_gated"
    assert record["served_intent"] == served
    assert record["shadow_intent"]["operator"] == "similar"
    assert record["classifier_called"] is True
    assert record["classifier_cached"] is False
    assert record["classifier_error"] is None
    assert record["classifier_operator_confidence"] == pytest.approx(0.81)
    assert record["shadow_latency_ms"] >= 0
    assert "operator" in record["disagreements"]
    assert record["disagree"] is True


def test_enum_disagreement_is_reported_per_field():
    served = extract_intent(GATED_QUERY).to_dict()
    payload = jev_payload(
        "none", doctype=choice_answer("phdthesis", {"none": 0.1, "phdthesis": 0.9})
    )
    record = shadow_record(GATED_QUERY, served, "jev_gated", _answering(payload), "pipeline")
    assert "doctype" in record["disagreements"]
    assert "operator" not in record["disagreements"]


def test_failed_jev_call_is_recorded_not_raised():
    served = extract_intent(GATED_QUERY).to_dict()
    client = handler_client(lambda request: httpx.Response(502))
    record = shadow_record(GATED_QUERY, served, "jev_gated", client, "pipeline")
    assert record["classifier_called"] is True
    assert "502" in record["classifier_error"]
    assert record["classifier_operator_confidence"] is None
    assert record["shadow_intent"] == served
    assert record["disagree"] is False


def test_closed_gate_records_no_call():
    query = "papers citing dark energy surveys"
    served = extract_intent(query).to_dict()
    client = handler_client(lambda request: pytest.fail("gate should stay closed"))
    record = shadow_record(query, served, "jev_gated", client, "pipeline")
    assert record["classifier_called"] is False
    assert record["disagreements"] == []


def test_record_keeps_the_path_that_served_the_request():
    served = extract_intent(GATED_QUERY).to_dict()
    record = shadow_record(
        GATED_QUERY, served, "jev_gated", _answering(jev_payload("similar")), "model"
    )
    assert record["served_path"] == "model"
