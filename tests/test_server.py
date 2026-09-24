"""Automated tests for the hybrid inference server (docker/server.py).

server.py reads its configuration from the environment at import, so each
test imports a fresh copy of the module under a unique name with the
environment it needs. No model is loaded: the fine-tuned model is replaced by
a stub generate_query, and Jev by a JevClient over an httpx MockTransport.
torch and transformers are imported only by load_model / generate_query and
by device auto-detection, which DEVICE=cpu skips.
"""

import importlib.util
import itertools
import json
import logging
import sys
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from jev_fixtures import handler_client, jev_payload

SERVER_PATH = Path(__file__).resolve().parents[1] / "docker" / "server.py"
SERVER_ENV = (
    "MODEL_NAME",
    "DEVICE",
    "PORT",
    "ROUTING_MODE",
    "PIPELINE_CONFIDENCE_THRESHOLD",
    "TELEMETRY_LOG",
    "INTENT_BACKEND",
    "JEV_CACHE_PATH",
    "JEV_TIMEOUT_S",
    "SHADOW_INTENT_BACKEND",
    "TYPESAFE_API_KEY",
)
MODEL_QUERY = 'abs:"from the model"'
# The regex finds no operator here, so jev_gated calls Jev.
GATED_QUERY = "work along the lines of the Planck results"
# The regex finds an operator with high confidence, so jev_gated skips Jev.
REGEX_OPERATOR_QUERY = "papers citing dark energy surveys"
_module_ids = itertools.count()


@pytest.fixture
def load_server(monkeypatch, tmp_path):
    """Import docker/server.py with ``env`` on top of a hermetic base env."""
    loaded = []

    def load(**env: str | None):
        """A value of None leaves that variable unset, overriding the base env."""
        for name in SERVER_ENV:
            monkeypatch.delenv(name, raising=False)
        base = {"DEVICE": "cpu", "JEV_CACHE_PATH": str(tmp_path / "jev_cache.jsonl")}
        for name, value in {**base, **env}.items():
            if value is not None:
                monkeypatch.setenv(name, value)
        module_name = f"nls_server_under_test_{next(_module_ids)}"
        spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
        return module

    yield load
    for module in loaded:
        if module.shadow_executor is not None:
            module.shadow_executor.shutdown(wait=True, cancel_futures=True)


def _with_model(server) -> list[list]:
    """Pretend the fine-tuned model is loaded; returns the recorded calls."""
    calls: list[list] = []

    def generate_query(messages, max_tokens=256):
        calls.append(messages)
        return MODEL_QUERY, 11, 7

    server.model = object()
    server.generate_query = generate_query
    return calls


def _answering(payload: dict, calls: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    return handler_client(handler)


def _post_pipeline(server, nl: str) -> dict:
    response = TestClient(server.app).post(
        "/pipeline", json={"messages": [{"role": "user", "content": f"Query: {nl}\nDate: x"}]}
    )
    assert response.status_code == 200
    return response.json()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jev_env(**env: str) -> dict:
    return {"INTENT_BACKEND": "jev_gated", "TYPESAFE_API_KEY": "test-key", **env}


# -----------------------------------------------------------------------------
# Configuration validated at import
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"INTENT_BACKEND": "llm"}, "INTENT_BACKEND"),
        ({"ROUTING_MODE": "fast"}, "ROUTING_MODE"),
        (_jev_env(JEV_TIMEOUT_S="0"), "JEV_TIMEOUT_S"),
        ({"INTENT_BACKEND": "jev_gated"}, "TYPESAFE_API_KEY"),
        ({"SHADOW_INTENT_BACKEND": "regex", "TELEMETRY_LOG": "t.jsonl"}, "jev or jev_gated"),
        (_jev_env(SHADOW_INTENT_BACKEND="jev", TELEMETRY_LOG="t.jsonl"), "INTENT_BACKEND=regex"),
        ({"SHADOW_INTENT_BACKEND": "jev_gated", "TYPESAFE_API_KEY": "k"}, "TELEMETRY_LOG"),
        (
            {"SHADOW_INTENT_BACKEND": "jev", "TELEMETRY_LOG": "t", "ROUTING_MODE": "model"},
            "ROUTING_MODE=model",
        ),
    ],
)
def test_invalid_configuration_fails_at_import(load_server, env, message):
    with pytest.raises(ValueError, match=message):
        load_server(**env)


def test_jev_timeout_reaches_the_client(load_server):
    assert load_server(**_jev_env()).jev_client.timeout_s == 2.0
    assert load_server(**_jev_env(JEV_TIMEOUT_S="0.75")).jev_client.timeout_s == 0.75


def test_jev_cache_path_defaults_to_the_local_cache_file(load_server, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    server = load_server(**_jev_env(JEV_CACHE_PATH=None))
    assert server.jev_client.cache_path == Path("data/cache/jev_systemone.jsonl")


def test_empty_jev_cache_path_disables_the_cache_file(load_server, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    server = load_server(**_jev_env(JEV_CACHE_PATH=""))
    assert server.jev_client.cache_path is None
    payload = jev_payload("similar", operator_confidence=0.97)
    server.jev_client = handler_client(
        lambda request: httpx.Response(200, json=payload), cache_path=server.jev_client.cache_path
    )
    body = _post_pipeline(server, GATED_QUERY)
    assert body["pipeline_result"]["debug_info"]["classifier_called"] is True
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_imports_without_torch_or_transformers(load_server, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    server = load_server(DEVICE="")
    health = TestClient(server.app).get("/health").json()
    assert health["model_loaded"] is False and health["device"] == "cpu"
    assert _post_pipeline(server, REGEX_OPERATOR_QUERY)["path"] == "pipeline"


def test_startup_loads_the_model(load_server):
    server = load_server()
    loads = []
    server.load_model = lambda: loads.append(True)
    with TestClient(server.app):
        assert loads == [True]


def test_startup_degrades_to_pipeline_only_when_the_model_fails(load_server):
    server = load_server()

    def failing_load():
        raise OSError("no weights here")

    server.load_model = failing_load
    with TestClient(server.app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        body = client.post(
            "/pipeline", json={"messages": [{"role": "user", "content": REGEX_OPERATOR_QUERY}]}
        ).json()
        assert body["path"] == "pipeline"


def test_pipeline_routing_mode_never_loads_the_model(load_server):
    server = load_server(ROUTING_MODE="pipeline")
    server.load_model = lambda: pytest.fail("pipeline mode must not load the model")
    with TestClient(server.app) as client:
        assert client.get("/health").json()["routing_mode"] == "pipeline"


# -----------------------------------------------------------------------------
# /health
# -----------------------------------------------------------------------------


def test_health_reports_the_intent_configuration(load_server):
    health = TestClient(load_server().app).get("/health").json()
    assert health["intent_backend"] == "regex"
    assert health["jev_timeout_s"] == 2.0
    assert health["shadow_intent_backend"] is None


def test_health_reports_jev_and_shadow_settings(load_server, tmp_path):
    gated = TestClient(load_server(**_jev_env(JEV_TIMEOUT_S="1.5")).app).get("/health").json()
    assert gated["intent_backend"] == "jev_gated" and gated["jev_timeout_s"] == 1.5
    shadow = load_server(
        SHADOW_INTENT_BACKEND="jev_gated",
        TYPESAFE_API_KEY="test-key",
        TELEMETRY_LOG=str(tmp_path / "t.jsonl"),
    )
    health = TestClient(shadow.app).get("/health").json()
    assert health["intent_backend"] == "regex"
    assert health["shadow_intent_backend"] == "jev_gated"


# -----------------------------------------------------------------------------
# /pipeline debug fields and routing
# -----------------------------------------------------------------------------


def test_regex_backend_debug_fields(load_server):
    body = _post_pipeline(load_server(), REGEX_OPERATOR_QUERY)
    debug = body["pipeline_result"]["debug_info"]
    assert body["path"] == "pipeline" and body["fallback"] is False
    assert debug["intent_backend"] == "regex"
    assert debug["classifier_called"] is False
    assert debug["classifier_operator_confidence"] is None
    assert body["pipeline_result"]["confidence"] == pytest.approx(0.9)


def test_jev_gated_debug_fields_when_jev_answers(load_server):
    server = load_server(**_jev_env())
    calls: list = []
    server.jev_client = _answering(jev_payload("similar", operator_confidence=0.97), calls)
    _with_model(server)
    body = _post_pipeline(server, GATED_QUERY)
    result = body["pipeline_result"]
    assert len(calls) == 1
    assert body["path"] == "pipeline"
    assert result["intent"]["operator"] == "similar"
    assert result["debug_info"]["intent_backend"] == "jev_gated"
    assert result["debug_info"]["classifier_called"] is True
    assert result["debug_info"]["classifier_error"] is None
    assert result["debug_info"]["classifier_operator_confidence"] == pytest.approx(0.97)
    assert result["confidence"] == pytest.approx(0.9)
    assert "similar(" in body["choices"][0]["message"]["content"]


def test_jev_gated_skips_jev_when_regex_is_sure(load_server):
    server = load_server(**_jev_env())
    server.jev_client = handler_client(lambda request: pytest.fail("Jev must not be called"))
    debug = _post_pipeline(server, REGEX_OPERATOR_QUERY)["pipeline_result"]["debug_info"]
    assert debug["intent_backend"] == "jev_gated" and debug["classifier_called"] is False


def _server_error(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "upstream"})


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


@pytest.mark.parametrize(("handler", "marker"), [(_server_error, "500"), (_timeout, "ReadTimeout")])
def test_jev_failure_serves_the_regex_intent_not_the_model(load_server, handler, marker):
    server = load_server(**_jev_env())
    server.jev_client = handler_client(handler)
    model_calls = _with_model(server)
    body = _post_pipeline(server, GATED_QUERY)
    debug = body["pipeline_result"]["debug_info"]
    assert body["path"] == "pipeline" and body["error"] is None
    assert model_calls == []
    assert debug["classifier_called"] is True
    assert marker in debug["classifier_error"]
    assert debug["classifier_operator_confidence"] is None
    assert body["pipeline_result"]["intent"]["operator"] is None


def test_low_jev_operator_confidence_routes_to_the_model(load_server, tmp_path):
    log = tmp_path / "telemetry.jsonl"
    server = load_server(**_jev_env(TELEMETRY_LOG=str(log)))
    server.jev_client = _answering(jev_payload("similar", operator_confidence=0.44))
    model_calls = _with_model(server)
    body = _post_pipeline(server, GATED_QUERY)
    result = body["pipeline_result"]
    assert body["path"] == "model" and body["fallback"] is True
    assert len(model_calls) == 1
    assert body["choices"][0]["message"]["content"] == MODEL_QUERY
    assert result["debug_info"]["intent_backend"] == "jev_gated"
    assert result["debug_info"]["classifier_called"] is True
    assert result["debug_info"]["classifier_operator_confidence"] == pytest.approx(0.44)
    assert "classifier operator confidence 0.44" in result["debug_info"]["fallback_reason"]
    (row,) = _rows(log)
    assert row["path"] == "model"
    assert row["classifier_operator_confidence"] == pytest.approx(0.44)


def test_low_jev_confidence_is_served_when_no_model_is_loaded(load_server, tmp_path):
    log = tmp_path / "telemetry.jsonl"
    server = load_server(**_jev_env(TELEMETRY_LOG=str(log)))
    server.jev_client = _answering(jev_payload("similar", operator_confidence=0.44))
    body = _post_pipeline(server, GATED_QUERY)
    assert body["path"] == "pipeline"
    (row,) = _rows(log)
    assert row["fallback_reason"] == (
        "classifier operator confidence 0.44 below threshold 0.50; served without model fallback"
    )


def test_chat_completions_uses_the_same_routing(load_server):
    server = load_server(**_jev_env())
    server.jev_client = handler_client(_server_error)
    model_calls = _with_model(server)
    response = TestClient(server.app).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": GATED_QUERY}]}
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] != MODEL_QUERY
    assert model_calls == []


# -----------------------------------------------------------------------------
# Telemetry
# -----------------------------------------------------------------------------

REQUEST_ROW_FIELDS = {
    "record_type",
    "timestamp",
    "request_id",
    "nl_query",
    "generated_query",
    "path",
    "confidence",
    "fallback_reason",
    "latency_ms",
    "routing_mode",
    "intent_backend",
    "classifier_called",
    "classifier_error",
    "structural_confidence",
    "classifier_operator_confidence",
}


def test_request_row_fields(load_server, tmp_path):
    log = tmp_path / "telemetry.jsonl"
    server = load_server(**_jev_env(TELEMETRY_LOG=str(log)))
    server.jev_client = _answering(jev_payload("similar", operator_confidence=0.97))
    _post_pipeline(server, GATED_QUERY)
    (row,) = _rows(log)
    assert set(row) == REQUEST_ROW_FIELDS
    assert row["record_type"] == "request"
    assert row["intent_backend"] == "jev_gated"
    assert row["classifier_called"] is True
    assert row["classifier_operator_confidence"] == pytest.approx(0.97)
    assert row["structural_confidence"] == pytest.approx(0.9)


# -----------------------------------------------------------------------------
# Shadow mode
# -----------------------------------------------------------------------------

SHADOW_ROW_FIELDS = {
    "record_type",
    "timestamp",
    "request_id",
    "nl_query",
    "served_backend",
    "shadow_backend",
    "served_intent",
    "shadow_intent",
    "disagreements",
    "disagree",
    "classifier_called",
    "classifier_error",
    "classifier_operator_confidence",
    "shadow_latency_ms",
}


def _shadow_server(load_server, tmp_path: Path):
    log = tmp_path / "telemetry.jsonl"
    server = load_server(
        SHADOW_INTENT_BACKEND="jev_gated",
        TYPESAFE_API_KEY="test-key",
        TELEMETRY_LOG=str(log),
    )
    return server, log


def _drain(server) -> None:
    server.shadow_executor.shutdown(wait=True)


def test_shadow_serves_regex_and_logs_both_intents(load_server, tmp_path):
    server, log = _shadow_server(load_server, tmp_path)
    server.jev_client = _answering(jev_payload("similar", operator_confidence=0.81))
    body = _post_pipeline(server, GATED_QUERY)
    _drain(server)

    debug = body["pipeline_result"]["debug_info"]
    assert debug["intent_backend"] == "regex" and debug["classifier_called"] is False
    assert body["pipeline_result"]["intent"]["operator"] is None, "served intent is the regex one"

    request_row, shadow_row = sorted(_rows(log), key=lambda r: r["record_type"] != "request")
    assert request_row["record_type"] == "request"
    assert set(shadow_row) == SHADOW_ROW_FIELDS
    assert shadow_row["record_type"] == "intent_shadow"
    assert shadow_row["request_id"] == request_row["request_id"]
    assert shadow_row["served_intent"] == body["pipeline_result"]["intent"]
    assert shadow_row["shadow_intent"]["operator"] == "similar"
    assert shadow_row["disagreements"] == ["operator"]
    assert shadow_row["classifier_operator_confidence"] == pytest.approx(0.81)


def test_shadow_runs_off_the_request_path(load_server, tmp_path):
    server, log = _shadow_server(load_server, tmp_path)
    release = threading.Event()

    def slow(request: httpx.Request) -> httpx.Response:
        assert release.wait(timeout=10), "test never released the shadow call"
        return httpx.Response(200, json=jev_payload("similar"))

    server.jev_client = handler_client(slow)
    body = _post_pipeline(server, GATED_QUERY)
    assert body["path"] == "pipeline", "the response returned while Jev was still blocked"
    assert [r["record_type"] for r in _rows(log)] == ["request"]
    release.set()
    _drain(server)
    assert [r["record_type"] for r in _rows(log)] == ["request", "intent_shadow"]


def test_shadow_jev_failure_is_recorded(load_server, tmp_path):
    server, log = _shadow_server(load_server, tmp_path)
    server.jev_client = handler_client(_server_error)
    _post_pipeline(server, GATED_QUERY)
    _drain(server)
    shadow_row = next(r for r in _rows(log) if r["record_type"] == "intent_shadow")
    assert "500" in shadow_row["classifier_error"]
    assert shadow_row["disagree"] is False


def test_shadow_crash_is_logged_never_raised(load_server, tmp_path, caplog):
    server, log = _shadow_server(load_server, tmp_path)

    def crash(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("unexpected")

    server.jev_client = handler_client(crash)
    with caplog.at_level(logging.ERROR, logger="nls-server"):
        body = _post_pipeline(server, GATED_QUERY)
        _drain(server)
    assert body["path"] == "pipeline" and body["error"] is None
    assert [r["record_type"] for r in _rows(log)] == ["request"]
    assert any("Shadow intent run failed" in r.getMessage() for r in caplog.records)


def test_shadow_is_skipped_when_the_queue_is_full(load_server, tmp_path, caplog):
    server, log = _shadow_server(load_server, tmp_path)
    server.jev_client = handler_client(lambda request: pytest.fail("no slot, no call"))
    server._shadow_slots = threading.BoundedSemaphore(1)
    server._shadow_slots.acquire()
    with caplog.at_level(logging.WARNING, logger="nls-server"):
        body = _post_pipeline(server, GATED_QUERY)
    _drain(server)
    assert body["path"] == "pipeline"
    assert [r["record_type"] for r in _rows(log)] == ["request"]
    assert any("queue full" in r.getMessage() for r in caplog.records)


def test_shutdown_drains_running_shadow_runs(load_server, tmp_path):
    server, log = _shadow_server(load_server, tmp_path)
    server.jev_client = _answering(jev_payload("similar"))
    server.load_model = lambda: None
    with TestClient(server.app) as client:
        client.post("/pipeline", json={"messages": [{"role": "user", "content": GATED_QUERY}]})
    assert sorted(r["record_type"] for r in _rows(log)) == ["intent_shadow", "request"]


def test_no_shadow_rows_without_shadow_mode(load_server, tmp_path):
    log = tmp_path / "telemetry.jsonl"
    server = load_server(TELEMETRY_LOG=str(log))
    assert server.shadow_executor is None
    _post_pipeline(server, GATED_QUERY)
    assert [r["record_type"] for r in _rows(log)] == ["request"]
