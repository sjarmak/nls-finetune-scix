"""Hybrid NER pipeline for natural language to ADS query conversion.

This module implements the main pipeline that orchestrates:
1. NER extraction → IntentSpec
2. Few-shot retrieval → similar gold examples
3. Deterministic assembly → valid ADS query

The pipeline is designed to be fast (<50ms local) and deterministic.
LLM calls are only made in the fallback resolver path.
"""

import json
import time
from dataclasses import asdict, dataclass, field

from .intent_spec import IntentSpec


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
    """

    ner_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    assembly_time_ms: float = 0.0
    total_time_ms: float = 0.0
    constraint_corrections: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    raw_extracted: dict | None = None

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
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "intent": self.intent.to_dict(),
            "retrieved_examples": [ex.to_dict() for ex in self.retrieved_examples],
            "final_query": self.final_query,
            "debug_info": self.debug_info.to_dict(),
            "success": self.success,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def process_query(nl_text: str) -> PipelineResult:
    """Process a natural language query through the hybrid NER pipeline.

    This is the main entry point for converting natural language to ADS query.

    Pipeline stages:
    1. NER extraction - Parse NL to structured IntentSpec
    2. Few-shot retrieval - Find similar gold examples for guidance
    3. Query assembly - Build ADS query deterministically

    Args:
        nl_text: Natural language search query from user

    Returns:
        PipelineResult containing:
        - intent: Extracted IntentSpec
        - retrieved_examples: Similar gold examples
        - final_query: Valid ADS query string
        - debug_info: Timing and debugging info

    Note:
        This is a skeleton implementation. Each stage will be implemented
        in US-002 (NER), US-003 (retrieval), and US-004 (assembly).
    """
    start_time = time.perf_counter()
    debug_info = DebugInfo()

    # Stage 1: NER Extraction
    # TODO: Implement in US-002 (ner.py)
    ner_start = time.perf_counter()
    intent = IntentSpec(raw_user_text=nl_text)

    # Placeholder: use entire text as topic
    # Real implementation in ner.py will do proper extraction
    intent.free_text_terms = [nl_text]  # Fallback: use entire text as topic

    debug_info.ner_time_ms = (time.perf_counter() - ner_start) * 1000
    debug_info.raw_extracted = intent.to_dict()

    # Stage 2: Few-shot Retrieval
    # TODO: Implement in US-003 (retrieval.py)
    retrieval_start = time.perf_counter()
    retrieved_examples: list[GoldExample] = []
    debug_info.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

    # Stage 3: Query Assembly
    # TODO: Implement in US-004 (assembler.py)
    assembly_start = time.perf_counter()

    # Placeholder: simple abs: query from free text
    if intent.free_text_terms:
        topic = intent.free_text_terms[0]
        if " " in topic:
            final_query = f'abs:"{topic}"'
        else:
            final_query = f"abs:{topic}"
    else:
        final_query = ""

    debug_info.assembly_time_ms = (time.perf_counter() - assembly_start) * 1000

    # Total timing
    debug_info.total_time_ms = (time.perf_counter() - start_time) * 1000

    return PipelineResult(
        intent=intent,
        retrieved_examples=retrieved_examples,
        final_query=final_query,
        debug_info=debug_info,
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
        "database:",
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
