"""Tests for the general-LLM comparison arm (llm_intent). No network."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune.domains.scix.intent_spec import OPERATORS, IntentSpec
from finetune.domains.scix.jev_intent import build_questions
from finetune.domains.scix.llm_intent import (
    UNKNOWN,
    LlmClient,
    LlmResponseError,
    apply_llm_answers,
    build_prompt,
    build_request,
    build_schema,
    classify_and_extract_llm,
    cli_command,
    cli_environment,
    parse_cli_result,
    parse_output,
    request_fingerprint,
)


def _values(**overrides) -> dict:
    base = {
        "operator": "none",
        "search_kind": "topic",
        "doctype": "none",
        "bibgroup": "none",
        "collection": "none",
        "refereed": "false",
        "openaccess": "false",
        "eprint": "false",
        "needs_clarification": "false",
        "refers_to_specific_paper": "false",
    }
    base.update(overrides)
    return base


class FakeAnthropic:
    """Stands in for anthropic.Anthropic; records calls and replays canned outputs."""

    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.outputs[len(self.calls) - 1]
        return SimpleNamespace(
            stop_reason="end_turn",
            model="claude-haiku-4-5-20251001",
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=900, output_tokens=40),
        )


class TestSchemaAndPrompt:
    def test_schema_mirrors_jev_options_plus_unknown(self):
        schema = build_schema()
        assert set(schema["properties"]["operator"]["enum"]) == {"none", UNKNOWN, *OPERATORS}
        assert schema["properties"]["refereed"]["enum"] == ["true", "false", UNKNOWN]
        assert set(schema["required"]) == set(build_questions())

    def test_prompt_contains_every_option_description(self):
        prompt = build_prompt()
        for qid, q in build_questions().items():
            assert f"## {qid}" in prompt
            for option, description in q["criteria"].items():
                assert f"- {option}: {description}" in prompt


class TestParseOutput:
    def test_rejects_values_outside_schema(self):
        with pytest.raises(LlmResponseError):
            parse_output(json.dumps(_values(operator="topn")), "m", (1, 1), 1.0, False, "")
        with pytest.raises(LlmResponseError):
            parse_output("not json", "m", (1, 1), 1.0, False, "")
        with pytest.raises(LlmResponseError):
            parse_output(json.dumps({"operator": "none"}), "m", (1, 1), 1.0, False, "")


class TestApply:
    def test_unknown_abstains_and_scores_zero_confidence(self):
        answers = parse_output(
            json.dumps(_values(operator=UNKNOWN, doctype="phdthesis", refereed="true")),
            "m",
            (1, 1),
            1.0,
            False,
            "",
        )
        base = IntentSpec(raw_user_text="t", authors=["Smith"])
        out = apply_llm_answers(base, answers)
        assert out.operator is None and out.confidence["operator"] == 0.0
        assert out.doctype == {"phdthesis"} and out.confidence["doctype"] == 1.0
        assert out.property == {"refereed"}
        assert out.authors == ["Smith"] and base.doctype == set()


class TestClient:
    def test_uses_structured_output_and_caches(self, tmp_path: Path):
        fake = FakeAnthropic([json.dumps(_values(operator="citations"))])
        client = LlmClient(cache_path=tmp_path / "c.jsonl", anthropic_client=fake)
        first = client.classify("papers that build on Planck 2018")
        second = client.classify("papers that build on Planck 2018")
        assert len(fake.calls) == 1
        assert fake.calls[0]["output_config"]["format"]["type"] == "json_schema"
        assert fake.calls[0]["model"] == "claude-haiku-4-5"
        assert first.values["operator"] == "citations" and second.cached is True
        assert first.input_tokens == 900

    def test_compose_with_regex(self, tmp_path: Path):
        fake = FakeAnthropic([json.dumps(_values(operator="similar"))])
        client = LlmClient(cache_path=tmp_path / "c.jsonl", anthropic_client=fake)
        intent, answers = classify_and_extract_llm(
            "work by Smith since 2019 on dark energy", client
        )
        assert intent.operator == "similar"
        assert intent.authors == ["Smith"] and intent.year_from == 2019
        assert answers is not None and answers.values["operator"] == "similar"


def _cli_payload(structured: dict, **overrides) -> str:
    payload = {
        "is_error": False,
        "result": json.dumps(structured),
        "structured_output": structured,
        "duration_api_ms": 2403,
        "modelUsage": {
            "claude-haiku-4-5": {
                "inputTokens": 3816,
                "cacheCreationInputTokens": 0,
                "cacheReadInputTokens": 0,
                "outputTokens": 227,
            }
        },
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestCliTransport:
    def test_command_isolates_the_question_set(self):
        request = build_request("papers by Hawking", transport="cli")
        cmd = cli_command(request, '{"query": "papers by Hawking"}')
        assert cmd[:2] == ["claude", "-p"]
        assert "--exclude-dynamic-system-prompt-sections" in cmd
        assert cmd[cmd.index("--setting-sources") + 1] == ""
        assert cmd[cmd.index("--tools") + 1] == ""
        assert json.loads(cmd[cmd.index("--json-schema") + 1]) == build_schema()
        assert cmd[cmd.index("--system-prompt") + 1] == build_prompt()
        assert cmd[-1] == '{"query": "papers by Hawking"}'

    def test_environment_drops_api_key_and_disables_thinking(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
        env = cli_environment()
        assert "ANTHROPIC_API_KEY" not in env
        assert env["MAX_THINKING_TOKENS"] == "0"

    def test_parse_cli_result_sums_cached_input_tokens(self):
        structured = _values(operator="citations")
        text, model, usage, latency = parse_cli_result(
            _cli_payload(
                structured,
                modelUsage={
                    "claude-haiku-4-5": {
                        "inputTokens": 3,
                        "cacheCreationInputTokens": 100,
                        "cacheReadInputTokens": 34725,
                        "outputTokens": 220,
                    }
                },
            )
        )
        assert json.loads(text) == structured
        assert model == "claude-haiku-4-5"
        assert usage == (34828, 220)
        assert latency == 2403.0

    def test_parse_cli_result_rejects_errors_and_missing_output(self):
        with pytest.raises(LlmResponseError, match="Not logged in"):
            parse_cli_result(_cli_payload({}, is_error=True, result="Not logged in"))
        with pytest.raises(LlmResponseError, match="structured_output"):
            parse_cli_result(_cli_payload({}, structured_output=None))
        with pytest.raises(LlmResponseError, match="not JSON"):
            parse_cli_result("Warning: no stdin")

    def test_client_uses_runner_and_separate_cache_rows(self, tmp_path: Path):
        structured = _values(operator="reviews")
        calls: list[list[str]] = []

        def runner(command: list[str]) -> str:
            calls.append(command)
            return _cli_payload(structured)

        client = LlmClient(cache_path=tmp_path / "c.jsonl", transport="cli", cli_runner=runner)
        first = client.classify("survey papers on galaxy evolution")
        second = client.classify("survey papers on galaxy evolution")
        assert len(calls) == 1
        assert first.values["operator"] == "reviews" and first.input_tokens == 3816
        assert second.cached and second.values == first.values
        sdk_request = build_request("survey papers on galaxy evolution")
        assert request_fingerprint(sdk_request) != first.fingerprint
        row = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[0])
        assert row["transport"] == "cli"

    def test_transport_is_validated(self):
        with pytest.raises(ValueError, match="transport"):
            LlmClient(cache_path=None, transport="grpc")
