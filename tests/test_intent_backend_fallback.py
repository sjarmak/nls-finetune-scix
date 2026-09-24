"""Jev failures degrade to the regex intent instead of failing the pipeline.

A Jev outage, timeout or contract break must not turn into a pipeline error,
because the hybrid server routes pipeline errors to the fine-tuned model, the
path the Jev backend exists to avoid.
"""

import logging
import time

import httpx
import pytest
from jev_fixtures import handler_client, jev_payload, mock_jev_client

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


def _slow(delay_s: float):
    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(delay_s)
        return httpx.Response(200, json=jev_payload("citations"))

    return handler


def test_timeout_is_a_wall_clock_bound_on_the_whole_call():
    # httpx applies its timeout per phase; a response that takes longer than
    # timeout_s in total must still end the call at the deadline.
    client = handler_client(_slow(1.0), timeout_s=0.1)
    started = time.perf_counter()
    extraction = extract_intent_with_backend(GATED_QUERY, "jev_gated", client)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.6
    assert extraction.intent == extract_intent(GATED_QUERY)
    assert extraction.classifier_error.startswith("TimeoutException")


def test_unwritable_cache_still_serves_the_jev_answer(tmp_path, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    calls: list[dict] = []
    client = mock_jev_client(None, [jev_payload("citations")], calls)
    client.cache_path = blocker / "cache.jsonl"
    with caplog.at_level(logging.WARNING):
        extraction = extract_intent_with_backend(GATED_QUERY, "jev_gated", client)
    assert extraction.classifier_succeeded
    assert extraction.intent.operator == "citations"
    assert len(calls) == 1
    assert "Jev cache write" in caplog.text


def test_cache_hit_is_reported_as_cached_not_billed(tmp_path):
    calls: list[dict] = []
    client = mock_jev_client(tmp_path, [jev_payload("citations")], calls)
    first = extract_intent_with_backend(GATED_QUERY, "jev_gated", client)
    second = extract_intent_with_backend(GATED_QUERY, "jev_gated", client)
    assert len(calls) == 1
    assert (first.classifier_called, first.classifier_cached) == (True, False)
    assert (second.classifier_called, second.classifier_cached) == (True, True)
    assert second.intent == first.intent


def test_uncached_client_does_not_memoise_answers():
    calls: list[dict] = []
    client = mock_jev_client(None, [jev_payload("citations"), jev_payload("citations")], calls)
    extract_intent_with_backend(GATED_QUERY, "jev", client)
    second = extract_intent_with_backend(GATED_QUERY, "jev", client)
    assert len(calls) == 2
    assert second.classifier_cached is False
