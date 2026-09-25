"""Pipeline confidence uses Jev's operator confidence when Jev answered.

On the held-out paraphrase set Jev's operator accuracy is 100% above 0.73
confidence and its only miss sits at 0.44, so a 0.5 routing threshold on the
combined confidence sends that miss to the model.
"""

import httpx
import pytest
from jev_fixtures import answering_client as _answering
from jev_fixtures import handler_client, jev_payload

from finetune.domains.scix.pipeline import process_query

GATED_QUERY = "work along the lines of the Planck results"


def test_low_jev_operator_confidence_lowers_pipeline_confidence():
    result = process_query(
        GATED_QUERY, "jev_gated", _answering(jev_payload("similar", operator_confidence=0.44))
    )
    assert result.intent.operator == "similar"
    assert result.debug_info.structural_confidence == pytest.approx(0.9)
    assert result.debug_info.classifier_operator_confidence == pytest.approx(0.44)
    assert result.confidence == pytest.approx(0.44)


def test_confident_jev_leaves_the_structural_confidence_in_charge():
    result = process_query(
        GATED_QUERY, "jev_gated", _answering(jev_payload("similar", operator_confidence=0.97))
    )
    assert result.debug_info.classifier_operator_confidence == pytest.approx(0.97)
    assert result.confidence == pytest.approx(0.9)


def test_underspecified_query_stays_low_even_when_jev_is_sure():
    result = process_query(
        "quasars", "jev_gated", _answering(jev_payload("none", operator_confidence=0.99))
    )
    assert result.debug_info.classifier_called is True
    assert result.debug_info.classifier_operator_confidence == pytest.approx(0.99)
    assert result.confidence == pytest.approx(0.3)
    assert result.debug_info.fallback_reason


def test_failed_jev_call_routes_on_structural_confidence():
    client = handler_client(lambda request: httpx.Response(500))
    result = process_query(GATED_QUERY, "jev_gated", client)
    assert result.debug_info.classifier_operator_confidence is None
    assert result.confidence == result.debug_info.structural_confidence


@pytest.mark.parametrize(
    ("backend", "query"),
    [("regex", GATED_QUERY), ("jev_gated", "papers citing dark energy surveys")],
)
def test_no_classifier_answer_means_no_classifier_confidence(backend, query):
    client = handler_client(lambda request: pytest.fail("Jev must not be called"))
    result = process_query(query, backend, client)
    assert result.debug_info.classifier_called is False
    assert result.debug_info.classifier_operator_confidence is None
    assert result.confidence == result.debug_info.structural_confidence


def test_operator_without_a_subject_is_an_empty_query_with_no_confidence():
    """Nothing names what to take the references of, so nothing is served from the pipeline."""
    result = process_query("what does this paper cite", "regex")
    assert result.intent.operator == "references"
    assert result.final_query == ""
    assert result.confidence == 0.0
    assert "references" in result.debug_info.fallback_reason
