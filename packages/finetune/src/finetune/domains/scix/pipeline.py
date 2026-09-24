"""Hybrid NER pipeline for natural language to ADS query conversion.

This module implements the main pipeline that orchestrates:
1. NER extraction → IntentSpec
2. Few-shot retrieval → similar gold examples
3. Deterministic assembly → valid ADS query

The pipeline is designed to be fast (<50ms local) and deterministic.
LLM calls are only made in the fallback resolver path.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from .intent_spec import IntentSpec

if TYPE_CHECKING:
    from .jev_intent import JevClient

IntentBackend = Literal["regex", "jev", "jev_gated"]
INTENT_BACKENDS: tuple[str, ...] = ("regex", "jev", "jev_gated")
GATED_CONFIDENCE_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


@dataclass
class GoldExample:
    """A gold example from gold_examples.json for few-shot guidance.

    Attributes:
        nl_query: Original natural language query
        ads_query: Corresponding ADS query syntax
        features: Feature summary (operators, fields used, etc.)
        score: Retrieval similarity score (set during retrieval)
    """

    nl_query: str
    ads_query: str
    features: dict = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class DebugInfo:
    """Debugging information for pipeline execution.

    Attributes:
        ner_time_ms: Time spent in NER extraction
        retrieval_time_ms: Time spent in retrieval
        assembly_time_ms: Time spent in query assembly
        total_time_ms: Total pipeline time
        constraint_corrections: Fields that were corrected/removed
        fallback_reason: Reason if fallback path was taken
        raw_extracted: Raw NER extraction before validation
        intent_backend: Which intent extractor produced the IntentSpec
        classifier_called: Whether a Jev classifier call was attempted, even one
            that failed
        classifier_error: Why the Jev call failed, when it did; the regex
            intent was used instead
        structural_confidence: compute_pipeline_confidence on the final intent
        classifier_operator_confidence: Jev's operator confidence, when Jev
            answered; routing_confidence combines it with the structural one
    """

    ner_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    assembly_time_ms: float = 0.0
    total_time_ms: float = 0.0
    constraint_corrections: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    raw_extracted: dict | None = None
    intent_backend: str = "regex"
    classifier_called: bool = False
    classifier_error: str | None = None
    structural_confidence: float | None = None
    classifier_operator_confidence: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PipelineResult:
    """Result of the hybrid NER pipeline.

    Attributes:
        intent: Extracted and validated IntentSpec
        retrieved_examples: Top-k similar gold examples
        final_query: Assembled ADS query string
        debug_info: Timing and debugging information
        success: Whether pipeline completed successfully
        error: Error message if success is False
    """

    intent: IntentSpec
    retrieved_examples: list[GoldExample]
    final_query: str
    debug_info: DebugInfo
    confidence: float = 1.0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "intent": self.intent.to_dict(),
            "retrieved_examples": [ex.to_dict() for ex in self.retrieved_examples],
            "final_query": self.final_query,
            "debug_info": self.debug_info.to_dict(),
            "confidence": self.confidence,
            "success": self.success,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def compute_pipeline_confidence(intent: IntentSpec) -> tuple[float, str | None]:
    """Compute a confidence score for the pipeline's extraction quality.

    Returns:
        Tuple of (confidence float 0-1, fallback_reason or None)

    High confidence (0.9): authors, operators, or year ranges extracted
    Medium (0.7): multi-word topics with some structure
    Low (0.3): short query, nothing structured — likely ambiguous input
    """
    has_authors = bool(intent.authors)
    has_operator = intent.operator is not None
    has_years = intent.year_from is not None or intent.year_to is not None
    has_constraints = intent.has_constraints()

    # High confidence: structured fields were extracted
    if has_authors or has_operator or has_years:
        return (0.9, None)

    # Medium confidence: multi-word topics or constraints
    total_topic_words = sum(len(t.split()) for t in intent.free_text_terms) + sum(
        len(t.split()) for t in intent.or_terms
    )

    if has_constraints or total_topic_words >= 3:
        return (0.7, None)

    if total_topic_words >= 2:
        return (0.5, None)

    # Low confidence: short/ambiguous input
    return (0.3, "short query with no structured fields extracted")


@dataclass(frozen=True)
class IntentExtraction:
    """Outcome of the intent stage.

    ``classifier_called`` is True whenever a Jev call was attempted, including
    one that failed; ``classifier_error`` is set exactly when it failed, and
    the intent is then the regex intent. Keeping both lets telemetry count
    attempted (billed or timed-out) calls separately from answered ones.
    """

    intent: IntentSpec
    classifier_called: bool = False
    classifier_error: str | None = None

    @property
    def classifier_succeeded(self) -> bool:
        """True when Jev answered and its fields are in ``intent``."""
        return self.classifier_called and self.classifier_error is None


def routing_confidence(structural: float, classifier_operator: float | None) -> float:
    """The confidence the server routes on.

    Without a Jev answer this is the structural heuristic. With one it is the
    minimum of the two, because they catch different failures and either one
    alone sinks the assembled query: the structural score says whether the
    regex found enough names, years and topics to build a specific query
    (Jev does not touch those), and Jev's operator confidence says whether
    the operator decision can be trusted (the structural score gives any
    operator a flat 0.9). Taking the minimum serves from the pipeline only
    when both clear the threshold.
    """
    if classifier_operator is None:
        return structural
    return min(structural, classifier_operator)


def extract_intent_with_backend(
    nl_text: str,
    intent_backend: IntentBackend,
    jev_client: "JevClient | None",
) -> IntentExtraction:
    """Run the selected intent extractor.

    regex      - the shipped rules-based extractor.
    jev        - Jev decides operator, enum fields and gates; regex keeps names,
                 years and topics.
    jev_gated  - regex first; Jev only when regex finds no operator or the
                 regex intent scores below GATED_CONFIDENCE_THRESHOLD.

    When a Jev call fails (HTTP error, timeout, network error, or a response
    that breaks the contract) the regex intent is returned with the failure
    in ``classifier_error``, so a Jev outage degrades to the regex backend
    instead of failing the request. Any other exception propagates.
    """
    from .ner import extract_intent

    if intent_backend not in INTENT_BACKENDS:
        raise ValueError(f"intent_backend must be one of {INTENT_BACKENDS}, got {intent_backend!r}")
    if intent_backend == "regex":
        return IntentExtraction(extract_intent(nl_text))
    if jev_client is None:
        raise ValueError(f"intent_backend={intent_backend!r} requires a JevClient")

    regex_intent = extract_intent(nl_text)
    if intent_backend == "jev_gated":
        if regex_intent.confidence.get("ads_passthrough"):
            return IntentExtraction(regex_intent)
        confidence, _ = compute_pipeline_confidence(regex_intent)
        if regex_intent.operator is not None and confidence >= GATED_CONFIDENCE_THRESHOLD:
            return IntentExtraction(regex_intent)
    return _classify_or_fall_back(nl_text, regex_intent, jev_client)


def _classify_or_fall_back(
    nl_text: str, regex_intent: IntentSpec, jev_client: "JevClient"
) -> IntentExtraction:
    from .jev_intent import JEV_FAILURES, extract_intent_jev

    try:
        intent = extract_intent_jev(nl_text, jev_client)
    except JEV_FAILURES as error:
        reason = f"{type(error).__name__}: {error}"
        logger.warning("Jev classifier failed, serving the regex intent: %s", reason)
        return IntentExtraction(regex_intent, classifier_called=True, classifier_error=reason)
    called = not regex_intent.confidence.get("ads_passthrough")
    return IntentExtraction(intent, classifier_called=called)


def process_query(
    nl_text: str,
    intent_backend: IntentBackend = "regex",
    jev_client: "JevClient | None" = None,
) -> PipelineResult:
    """Process a natural language query through the hybrid NER pipeline.

    This is the main entry point for converting natural language to ADS query.

    Pipeline stages:
    1. Intent extraction - Parse NL to structured IntentSpec (see
       extract_intent_with_backend for the backends)
    2. Few-shot retrieval - Find similar gold examples for guidance
    3. Query assembly - Build ADS query deterministically

    Args:
        nl_text: Natural language search query from user
        intent_backend: "regex" (default), "jev" or "jev_gated"
        jev_client: Required for the Jev backends

    Returns:
        PipelineResult containing:
        - intent: Extracted IntentSpec
        - retrieved_examples: Similar gold examples
        - final_query: Valid ADS query string
        - debug_info: Timing and debugging info
    """
    start_time = time.perf_counter()
    debug_info = DebugInfo(intent_backend=intent_backend)

    # Stage 1: Intent extraction
    ner_start = time.perf_counter()
    extraction = extract_intent_with_backend(nl_text, intent_backend, jev_client)
    intent = extraction.intent
    debug_info.classifier_called = extraction.classifier_called
    debug_info.classifier_error = extraction.classifier_error
    debug_info.ner_time_ms = (time.perf_counter() - ner_start) * 1000
    debug_info.raw_extracted = intent.to_dict()

    # Stage 2: Few-shot Retrieval
    from .retrieval import retrieve_similar

    retrieval_start = time.perf_counter()
    retrieved_examples = retrieve_similar(intent, k=5)
    debug_info.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

    # Stage 3: Query Assembly
    from .assembler import assemble_query

    assembly_start = time.perf_counter()
    final_query = assemble_query(intent, retrieved_examples)
    debug_info.assembly_time_ms = (time.perf_counter() - assembly_start) * 1000

    # Confidence: structural heuristic, capped by Jev's operator confidence
    structural, fallback_reason = compute_pipeline_confidence(intent)
    if fallback_reason:
        debug_info.fallback_reason = fallback_reason
    debug_info.structural_confidence = structural
    if extraction.classifier_succeeded:
        debug_info.classifier_operator_confidence = intent.confidence["operator"]
    confidence = routing_confidence(structural, debug_info.classifier_operator_confidence)

    # Total timing
    debug_info.total_time_ms = (time.perf_counter() - start_time) * 1000

    return PipelineResult(
        intent=intent,
        retrieved_examples=retrieved_examples,
        final_query=final_query,
        debug_info=debug_info,
        confidence=confidence,
        success=True,
    )


def is_ads_query(text: str) -> bool:
    """Check if text appears to already be an ADS query.

    Detects if the user has provided raw ADS syntax rather than
    natural language. If so, we skip NER and just validate.

    Args:
        text: Input text to check

    Returns:
        True if text contains ADS field tokens
    """
    # Common ADS field prefixes
    ads_patterns = [
        "author:",
        "abs:",
        "title:",
        "pubdate:",
        "bibstem:",
        "doctype:",
        "property:",
        "collection:",
        "bibgroup:",
        "object:",
        "aff:",
        "citations(",
        "references(",
        "trending(",
        "useful(",
        "similar(",
        "reviews(",
    ]
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in ads_patterns)
