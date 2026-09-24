"""General-LLM comparison arm: the Jev question set answered by Claude Haiku 4.5.

Same questions, same option descriptions, same output fields as
``jev_intent``, plus an explicit ``unknown`` per field so the model can
abstain. Exists so a Jev win can be attributed to Jev rather than to asking
narrow typed questions. IO, schema validation and caching only.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .intent_spec import IntentSpec
from .jev_intent import (
    BOOLEAN_QUESTION_IDS,
    CHOICE_QUESTION_IDS,
    NONE_OPTION,
    PROPERTY_BOOLEANS,
    build_questions,
    request_fingerprint,
)

LLM_MODEL = "claude-haiku-4-5"
DEFAULT_CACHE_PATH = Path("data/cache/llm_intent.jsonl")
UNKNOWN = "unknown"
MAX_TOKENS = 256
TRANSPORTS = ("sdk", "cli")
CLI_TIMEOUT_S = 120


class LlmResponseError(ValueError):
    """The model output did not match the requested schema."""


@dataclass(frozen=True)
class LlmAnswers:
    """One structured classification. Values are option strings or 'unknown'."""

    values: dict[str, str]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool
    fingerprint: str = ""
    raw: dict = field(default_factory=dict)

    def is_unknown(self, qid: str) -> bool:
        return self.values[qid] == UNKNOWN


def build_schema() -> dict:
    """JSON schema with the same options as the Jev questions plus 'unknown'."""
    questions = build_questions()
    properties: dict[str, Any] = {}
    for qid in CHOICE_QUESTION_IDS:
        properties[qid] = {"type": "string", "enum": [*questions[qid]["criteria"], UNKNOWN]}
    for qid in BOOLEAN_QUESTION_IDS:
        properties[qid] = {"type": "string", "enum": ["true", "false", UNKNOWN]}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_prompt() -> str:
    """System prompt rendered from the Jev question set so both arms share criteria."""
    lines = [
        "You classify natural-language literature-search requests for the NASA ADS / SciX "
        "astronomy database into typed fields. Answer every field. Use 'unknown' when the "
        "request does not let you decide; do not guess.",
        "",
    ]
    for qid, question in build_questions().items():
        lines.append(f"## {qid}")
        lines.append(question["instructions"])
        for option, description in question["criteria"].items():
            lines.append(f"- {option}: {description}")
        lines.append("")
    return "\n".join(lines)


def build_request(
    text: str, context: dict | None = None, model: str = LLM_MODEL, transport: str = "sdk"
) -> dict:
    if transport not in TRANSPORTS:
        raise ValueError(f"transport must be one of {TRANSPORTS}, got {transport!r}")
    state: dict = {"query": text}
    if context:
        if "query" in context:
            raise ValueError("context may not override the 'query' state key")
        state.update(context)
    request = {
        "model": model,
        "system": build_prompt(),
        "state": state,
        "schema": build_schema(),
    }
    if transport != "sdk":
        request["transport"] = transport
    return request


def cli_command(request: dict, user_content: str) -> list[str]:
    """`claude -p` invocation that asks the same question through the logged-in account.

    Settings, CLAUDE.md, MCP servers and the dynamic system-prompt sections are
    all excluded so the model sees only the shared question set; thinking is
    disabled through the environment (see `cli_environment`).
    """
    return [
        "claude",
        "-p",
        "--model",
        request["model"],
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(request["schema"]),
        "--system-prompt",
        request["system"],
        "--exclude-dynamic-system-prompt-sections",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--tools",
        "",
        "--no-session-persistence",
        "--max-turns",
        "2",  # the structured-output answer is delivered as a tool call, which counts as a turn
        user_content,
    ]


def cli_environment() -> dict[str, str]:
    """Environment for the CLI: no API key (so the OAuth login is used), no thinking."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["MAX_THINKING_TOKENS"] = "0"
    return env


def run_cli(command: list[str]) -> str:
    """Execute the CLI and return its stdout. Raises on a non-zero exit."""
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=cli_environment(),
        stdin=subprocess.DEVNULL,
        timeout=CLI_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        raise LlmResponseError(
            f"claude CLI exited {completed.returncode}: {completed.stderr.strip()[:300]}"
        )
    return completed.stdout


def parse_cli_result(stdout: str) -> tuple[str, str, tuple[int, int], float]:
    """(output_text, model, (input_tokens, output_tokens), latency_ms) from `--output-format json`.

    Input tokens include cache creation and cache reads, since those are what
    the account is charged for. Latency is the CLI's reported API time, not
    process wall time.
    """
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise LlmResponseError(f"claude CLI output is not JSON: {stdout[:200]!r}") from error
    if payload.get("is_error"):
        raise LlmResponseError(f"claude CLI error: {str(payload.get('result'))[:300]}")
    structured = payload.get("structured_output")
    if not isinstance(structured, dict):
        raise LlmResponseError("claude CLI returned no structured_output")
    model_usage = payload.get("modelUsage") or {}
    if len(model_usage) != 1:
        raise LlmResponseError(f"expected one model in modelUsage, got {sorted(model_usage)}")
    model, usage = next(iter(model_usage.items()))
    input_tokens = sum(
        int(usage.get(k, 0))
        for k in ("inputTokens", "cacheCreationInputTokens", "cacheReadInputTokens")
    )
    latency = payload.get("duration_api_ms")
    if not isinstance(latency, (int, float)):
        raise LlmResponseError("claude CLI result has no duration_api_ms")
    return (
        json.dumps(structured),
        model,
        (input_tokens, int(usage.get("outputTokens", 0))),
        float(latency),
    )


def parse_output(
    text: str, model: str, usage: tuple[int, int], latency_ms: float, cached: bool, fingerprint: str
) -> LlmAnswers:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise LlmResponseError(f"model output is not JSON: {text[:200]!r}") from error
    if not isinstance(payload, dict):
        raise LlmResponseError("model output is not an object")
    schema = build_schema()["properties"]
    values: dict[str, str] = {}
    for qid, spec in schema.items():
        value = payload.get(qid)
        if value not in spec["enum"]:
            raise LlmResponseError(f"{qid}: {value!r} is not one of the allowed options")
        values[qid] = value
    return LlmAnswers(
        values=values,
        model=model,
        input_tokens=usage[0],
        output_tokens=usage[1],
        latency_ms=latency_ms,
        cached=cached,
        fingerprint=fingerprint,
        raw=payload,
    )


class LlmClient:
    """Structured-output client with a JSONL cache.

    ``transport="sdk"`` calls the Messages API with an API key; ``"cli"``
    shells out to ``claude -p`` so the logged-in Claude account is used
    instead. Cache rows are keyed by request fingerprint, which includes the
    transport, so the two never share rows.
    """

    def __init__(
        self,
        cache_path: Path | None = DEFAULT_CACHE_PATH,
        model: str = LLM_MODEL,
        anthropic_client: Any | None = None,
        transport: str = "sdk",
        cli_runner: Callable[[list[str]], str] = run_cli,
    ) -> None:
        if transport not in TRANSPORTS:
            raise ValueError(f"transport must be one of {TRANSPORTS}, got {transport!r}")
        if transport == "sdk" and anthropic_client is None:
            import anthropic

            anthropic_client = anthropic.Anthropic()
        self._client = anthropic_client
        self._cli_runner = cli_runner
        self.transport = transport
        self.model = model
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = self._load_cache() if cache_path else {}

    def _load_cache(self) -> dict[str, dict]:
        assert self.cache_path is not None
        if not self.cache_path.exists():
            return {}
        rows: dict[str, dict] = {}
        with self.cache_path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{self.cache_path}:{lineno}: corrupt cache row") from error
                rows[row["fingerprint"]] = row
        return rows

    def classify(
        self, text: str, context: dict | None = None, use_cache: bool = True
    ) -> LlmAnswers:
        request = build_request(text, context=context, model=self.model, transport=self.transport)
        fingerprint = request_fingerprint(request)
        if use_cache:
            with self._lock:
                row = self._cache.get(fingerprint)
            if row is not None:
                return parse_output(
                    row["output_text"],
                    row["model"],
                    tuple(row["usage"]),
                    row["latency_ms"],
                    cached=True,
                    fingerprint=fingerprint,
                )

        user_content = json.dumps(request["state"], ensure_ascii=False)
        if self.transport == "cli":
            output_text, model, usage, latency_ms = parse_cli_result(
                self._cli_runner(cli_command(request, user_content))
            )
        else:
            output_text, model, usage, latency_ms = self._call_sdk(request, user_content)
        answers = parse_output(output_text, model, usage, latency_ms, False, fingerprint)
        self._record(fingerprint, request, output_text, model, usage, latency_ms)
        return answers

    def _call_sdk(
        self, request: dict, user_content: str
    ) -> tuple[str, str, tuple[int, int], float]:
        started = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=request["system"],
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": request["schema"]}},
        )
        latency_ms = (time.perf_counter() - started) * 1000
        if message.stop_reason not in ("end_turn", "stop_sequence"):
            raise LlmResponseError(f"unexpected stop_reason {message.stop_reason!r}")
        output_text = "".join(block.text for block in message.content if block.type == "text")
        usage = (message.usage.input_tokens, message.usage.output_tokens)
        return output_text, message.model, usage, latency_ms

    def _record(
        self,
        fingerprint: str,
        request: dict,
        output_text: str,
        model: str,
        usage: tuple[int, int],
        latency_ms: float,
    ) -> None:
        row = {
            "fingerprint": fingerprint,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "latency_ms": round(latency_ms, 1),
            "model": model,
            "usage": list(usage),
            "request_model": request["model"],
            "transport": request.get("transport", "sdk"),
            "state": request["state"],
            "output_text": output_text,
        }
        with self._lock:
            self._cache[fingerprint] = row
            if self.cache_path is None:
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_llm_answers(intent: IntentSpec, answers: LlmAnswers) -> IntentSpec:
    """Mirror of jev_intent.apply_answers. 'unknown' leaves a field empty.

    Confidence is 1.0 for an answered field and 0.0 for 'unknown', which is
    the only abstention signal a general LLM offers.
    """
    v = answers.values

    def single(qid: str) -> set[str]:
        return set() if v[qid] in (NONE_OPTION, UNKNOWN) else {v[qid]}

    confidence = {qid: 0.0 if v[qid] == UNKNOWN else 1.0 for qid in v}
    confidence = {
        "operator": confidence["operator"],
        "search_kind": confidence["search_kind"],
        "doctype": confidence["doctype"],
        "bibgroup": confidence["bibgroup"],
        "database": confidence["collection"],
        "needs_clarification": 1.0 if v["needs_clarification"] == "true" else 0.0,
        "refers_to_specific_paper": 1.0 if v["refers_to_specific_paper"] == "true" else 0.0,
        **{f"property.{name}": 1.0 if v[name] == "true" else 0.0 for name in PROPERTY_BOOLEANS},
        **{
            k: c
            for k, c in intent.confidence.items()
            if k in ("year", "authors", "topics", "or_topics")
        },
    }
    return replace(
        intent,
        operator=None if v["operator"] in (NONE_OPTION, UNKNOWN) else v["operator"],
        doctype=single("doctype"),
        bibgroup=single("bibgroup"),
        collection=single("collection"),
        property={name for name in PROPERTY_BOOLEANS if v[name] == "true"},
        confidence=confidence,
    )


def classify_and_extract_llm(
    text: str, client: LlmClient, use_cache: bool = True, reference_year: int | None = None
) -> tuple[IntentSpec, LlmAnswers | None]:
    """Arm C: LLM gating composed with regex names, years and topics.

    ``reference_year`` anchors relative year phrases, as in ``extract_intent``.
    """
    from .ner import extract_intent, extract_intent_with_operator

    regex_intent = extract_intent(text, reference_year)
    if regex_intent.confidence.get("ads_passthrough"):
        return regex_intent, None
    answers = client.classify(text, use_cache=use_cache)
    operator = answers.values["operator"]
    base = extract_intent_with_operator(
        text, None if operator in (NONE_OPTION, UNKNOWN) else operator, reference_year
    )
    return apply_llm_answers(base, answers), answers
