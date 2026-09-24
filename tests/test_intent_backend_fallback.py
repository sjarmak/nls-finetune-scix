"""Jev failures degrade to the regex intent instead of failing the pipeline.

A Jev outage, timeout or contract break must not turn into a pipeline error,
because the hybrid server routes pipeline errors to the fine-tuned model, the
path the Jev backend exists to avoid.
"""

import logging

import httpx
import pytest
from jev_fixtures import handler_client, jev_payload

from finetune.domains.scix.jev_intent import JevResponseError
from finetune.domains.scix.ner import extract_intent
from finetune.domains.scix.pipeline import extract_intent_with_backend, process_query

# The regex finds no operator here, so jev_gated opens the gate and calls Jev.
GATED_QUERY = "work along the lines of the Planck results"


def _status(code: int):
    return lambda request: httpx.Response(code, json={"error": "upstream"})


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _not_json(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"<html>gateway</html>")


def _wrong_shape(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 1}})


FAILURES = {
    "http_500": _status(500),
    "http_401": _status(401),
    "timeout": _timeout,
    "connect_error": _connect_error,
    "malformed_body": _not_json,
    "contract_violation": _wrong_shape,
}


@pytest.mark.parametrize("backend", ["jev", "jev_gated"])
@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_jev_failure_returns_the_regex_intent(backend, failure):
    client = handler_client(FAILURES[failure])
    extraction = extract_intent_with_backend(GATED_QUERY, backend, client)
    assert extraction.intent == extract_intent(GATED_QUERY)
    assert extraction.classifier_called is True, "a call was attempted"
    assert extraction.classifier_error
    assert extraction.classifier_succeeded is False


def test_error_names_the_failure_class():
    extraction = extract_intent_with_backend(
        GATED_QUERY, "jev_gated", handler_client(FAILURES["timeout"])
    )
    assert extraction.classifier_error.startswith("ReadTimeout:")


def test_failure_is_logged_as_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        extract_intent_with_backend(GATED_QUERY, "jev_gated", handler_client(_status(503)))
    assert any(
        r.levelno == logging.WARNING and "regex intent" in r.getMessage() for r in caplog.records
    )


def test_success_reports_no_error():
    client = handler_client(lambda request: httpx.Response(200, json=jev_payload("similar")))
    extraction = extract_intent_with_backend(GATED_QUERY, "jev_gated", client)
    assert extraction.intent.operator == "similar"
    assert extraction.classifier_called is True
    assert extraction.classifier_error is None
    assert extraction.classifier_succeeded is True


def test_unrelated_errors_are_not_swallowed():
    def boom(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("bug in our code, not a Jev failure")

    with pytest.raises(RuntimeError):
        extract_intent_with_backend(GATED_QUERY, "jev_gated", handler_client(boom))


def test_process_query_surfaces_the_classifier_error():
    result = process_query(GATED_QUERY, "jev_gated", handler_client(_status(500)))
    assert result.success is True
    assert result.final_query
    assert result.debug_info.classifier_called is True
    assert "500" in result.debug_info.classifier_error
    assert result.to_dict()["debug_info"]["classifier_error"] == result.debug_info.classifier_error


def test_client_reports_non_json_body_as_a_contract_error():
    with pytest.raises(JevResponseError):
        handler_client(_not_json).classify("anything")


def test_client_timeout_is_configurable():
    client = handler_client(_status(200), timeout_s=2.0)
    assert client.timeout_s == 2.0
