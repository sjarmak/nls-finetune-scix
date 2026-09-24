"""Shadow comparison of the served regex intent against a Jev intent backend.

The hybrid server can serve the regex IntentSpec while running a Jev backend
on the same query off the request path. This module builds the telemetry
record for one such run. The comparison is mechanical equality on the fields
Jev decides; names, years and topics come from the regex in every backend and
are not compared.
"""

import time

from .jev_intent import JevClient
from .pipeline import IntentBackend, extract_intent_with_backend

SHADOW_RECORD_TYPE = "intent_shadow"
SHADOW_COMPARED_FIELDS: tuple[str, ...] = (
    "operator",
    "doctype",
    "bibgroup",
    "collection",
    "property",
)


def intent_disagreements(served: dict, shadow: dict) -> list[str]:
    """Compared fields whose values differ, in SHADOW_COMPARED_FIELDS order.

    Both arguments are ``IntentSpec.to_dict()`` output, whose enum sets are
    sorted lists, so list equality is set equality.
    """
    return [name for name in SHADOW_COMPARED_FIELDS if served.get(name) != shadow.get(name)]


def shadow_record(
    nl_query: str,
    served_intent: dict,
    shadow_backend: IntentBackend,
    jev_client: JevClient,
) -> dict:
    """Run ``shadow_backend`` on ``nl_query`` and compare it with the served intent.

    Jev failures do not raise: the backend falls back to the regex intent and
    the reason lands in ``classifier_error``, so such a row shows no
    disagreement and is counted as an error by the summary script.
    """
    started = time.perf_counter()
    extraction = extract_intent_with_backend(nl_query, shadow_backend, jev_client)
    latency_ms = (time.perf_counter() - started) * 1000
    shadow_intent = extraction.intent.to_dict()
    disagreements = intent_disagreements(served_intent, shadow_intent)
    return {
        "record_type": SHADOW_RECORD_TYPE,
        "nl_query": nl_query,
        "served_backend": "regex",
        "shadow_backend": shadow_backend,
        "served_intent": served_intent,
        "shadow_intent": shadow_intent,
        "disagreements": disagreements,
        "disagree": bool(disagreements),
        "classifier_called": extraction.classifier_called,
        "classifier_error": extraction.classifier_error,
        "classifier_operator_confidence": (
            extraction.intent.confidence["operator"] if extraction.classifier_succeeded else None
        ),
        "shadow_latency_ms": round(latency_ms, 1),
    }
