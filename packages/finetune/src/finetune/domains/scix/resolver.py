"""Optional LLM resolver for ambiguous paper references.

This module provides fallback resolution for queries like:
- "papers citing the famous paper about X"
- "references of that groundbreaking study on Y"

The LLM is called ONLY when:
1. An operator requires a target (citations, references, similar)
2. The target is not explicitly provided (e.g., no bibcode given)
3. The user's query contains ambiguous references ("this paper", "that famous paper")

Most queries should NOT trigger LLM resolution - it's a fallback path.
"""

import logging
import os
import re
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

LLM_TIMEOUT_MS = 300
ADS_TIMEOUT_S = 5.0
ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"

OPERATORS_REQUIRING_TARGET = frozenset({"citations", "references", "similar"})

AMBIGUOUS_REFERENCE_PATTERNS = [
    r"\bthis paper\b",
    r"\bthat paper\b",
    r"\bthe paper\b",
    r"\bfamous paper\b",
    r"\bgroundbreaking (paper|study|work)\b",
    r"\bseminal (paper|study|work)\b",
    r"\blandmark (paper|study|work)\b",
    r"\boriginal (paper|study|work)\b",
    r"\bclassic (paper|study|work)\b",
    r"\bpioneering (paper|study|work)\b",
    r"\binfluential (paper|study|work)\b",
]


@dataclass
class ResolverResult:
    """Result of paper reference resolution.

    Attributes:
        success: Whether resolution succeeded
        bibcode: Resolved bibcode (if successful)
        paper_title: Title of resolved paper (if successful)
        fallback_reason: Reason for fallback (if not successful)
        resolution_time_ms: Time spent in resolution
        used_llm: Whether LLM was called
    """

    success: bool
    bibcode: str | None = None
    paper_title: str | None = None
    fallback_reason: str | None = None
    resolution_time_ms: float = 0.0
    used_llm: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "bibcode": self.bibcode,
            "paper_title": self.paper_title,
            "fallback_reason": self.fallback_reason,
            "resolution_time_ms": self.resolution_time_ms,
            "used_llm": self.used_llm,
        }


def needs_resolution(operator: str | None, operator_target: str | None, raw_text: str) -> bool:
    """Check if LLM resolution is needed for this query.

    Resolution is needed only when:
    1. Operator is one that requires a target
    2. No explicit target is provided
    3. Text contains ambiguous paper references

    Args:
        operator: The operator from IntentSpec (may be None)
        operator_target: Explicit target if provided (e.g., bibcode)
        raw_text: Original user input text

    Returns:
        True if LLM resolution should be attempted
    """
    if operator is None:
        return False

    if operator not in OPERATORS_REQUIRING_TARGET:
        return False

    if operator_target is not None and operator_target.strip():
        return False

    text_lower = raw_text.lower()
    for pattern in AMBIGUOUS_REFERENCE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    return False


def extract_paper_hint(raw_text: str) -> str | None:
    """Extract hints about which paper the user is referring to.

    Looks for context clues like:
    - "the Planck 2018 paper"
    - "Einstein's famous paper"
    - "the paper about cosmic microwave background"

    Args:
        raw_text: Original user input

    Returns:
        Extracted hint string or None
    """
    text_lower = raw_text.lower()
    hints = []

    about_match = re.search(r"(?:paper|study|work)\s+(?:about|on)\s+([^,\.]+)", text_lower)
    if about_match:
        hints.append(about_match.group(1).strip())

    author_match = re.search(r"(\w+(?:'s)?)\s+(?:famous|seminal|classic)\s+paper", text_lower)
    if author_match:
        hints.append(author_match.group(1).strip())

    year_match = re.search(r"(?:the\s+)?(\d{4})\s+(?:paper|study)", text_lower)
    if year_match:
        hints.append(f"year:{year_match.group(1)}")

    journal_match = re.search(r"(?:published in|from)\s+(\w+)", text_lower)
    if journal_match:
        hints.append(journal_match.group(1).strip())

    return " ".join(hints) if hints else None


def resolve_via_ads_search(
    topic_hint: str,
    api_key: str | None = None,
) -> ResolverResult:
    """Resolve paper reference via ADS search (deterministic).

    Searches ADS for the topic hint and returns the most-cited paper
    as a deterministic selection strategy.

    Args:
        topic_hint: Search terms extracted from user query
        api_key: ADS API key (defaults to ADS_API_KEY env var)

    Returns:
        ResolverResult with resolved bibcode or fallback reason
    """
    start_time = time.perf_counter()

    api_key = api_key or os.environ.get("ADS_API_KEY")
    if not api_key:
        return ResolverResult(
            success=False,
            fallback_reason="ADS_API_KEY not set",
            resolution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    search_query = f'abs:"{topic_hint}"' if " " in topic_hint else f"abs:{topic_hint}"

    try:
        response = httpx.get(
            ADS_API_URL,
            params={
                "q": search_query,
                "rows": 1,
                "sort": "citation_count desc",
                "fl": "bibcode,title,citation_count",
            },
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=ADS_TIMEOUT_S,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            return ResolverResult(
                success=False,
                fallback_reason=f"ADS API error: HTTP {response.status_code}",
                resolution_time_ms=elapsed_ms,
            )

        data = response.json()
        docs = data.get("response", {}).get("docs", [])

        if not docs:
            return ResolverResult(
                success=False,
                fallback_reason=f"No papers found for: {topic_hint}",
                resolution_time_ms=elapsed_ms,
            )

        top_doc = docs[0]
        bibcode = top_doc.get("bibcode")
        title = top_doc.get("title", ["Unknown"])[0]

        return ResolverResult(
            success=True,
            bibcode=bibcode,
            paper_title=title,
            resolution_time_ms=elapsed_ms,
            used_llm=False,
        )

    except httpx.TimeoutException:
        return ResolverResult(
            success=False,
            fallback_reason="ADS API timeout",
            resolution_time_ms=(time.perf_counter() - start_time) * 1000,
        )
    except httpx.RequestError as e:
        return ResolverResult(
            success=False,
            fallback_reason=f"ADS API request error: {e}",
            resolution_time_ms=(time.perf_counter() - start_time) * 1000,
        )


def resolve_via_llm(
    raw_text: str,
    context: str | None = None,
    timeout_ms: float = LLM_TIMEOUT_MS,
) -> dict | None:
    """Call LLM to interpret ambiguous paper reference.

    The LLM returns structured JSON with:
    - paper_title_guess: Best guess at paper title
    - bibcode_guess: Optional bibcode if known

    Args:
        raw_text: User's original query
        context: Additional context about the query
        timeout_ms: Maximum time to wait for LLM response

    Returns:
        Dict with paper_title_guess and bibcode_guess, or None on failure
    """
    start_time = time.perf_counter()

    llm_api_key = os.environ.get("OPENAI_API_KEY")
    if not llm_api_key:
        logger.warning("OPENAI_API_KEY not set, skipping LLM resolution")
        return None

    prompt = f"""You are helping resolve a paper reference from a search query.

User query: "{raw_text}"
{f'Context: {context}' if context else ''}

Based on this query, identify which specific paper the user is referring to.
Return JSON with:
- paper_title_guess: Your best guess at the paper title
- bibcode_guess: The ADS bibcode if you know it (e.g., "2018A&A...641A...1P"), otherwise null

Focus on famous, highly-cited papers that match the description.
Return only the JSON object, no other text."""

    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 150,
            },
            timeout=timeout_ms / 1000.0,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if elapsed_ms > timeout_ms:
            logger.warning(f"LLM took {elapsed_ms:.1f}ms, exceeding {timeout_ms}ms limit")
            return None

        if response.status_code != 200:
            logger.warning(f"LLM API error: HTTP {response.status_code}")
            return None

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if json_match:
            import json

            return json.loads(json_match.group())
        else:
            logger.warning(f"LLM response not valid JSON: {content[:100]}")
            return None

    except httpx.TimeoutException:
        logger.warning(f"LLM timeout after {timeout_ms}ms")
        return None
    except Exception as e:
        logger.warning(f"LLM resolution error: {e}")
        return None


def resolve_paper_reference(
    raw_text: str,
    operator: str | None = None,
    operator_target: str | None = None,
    context: str | None = None,
    use_llm: bool = True,
) -> ResolverResult:
    """Main entry point for paper reference resolution.

    Attempts to resolve ambiguous paper references to a specific bibcode.
    Uses a two-stage approach:
    1. Extract hints from the query text
    2. Search ADS for the most-cited match (deterministic)
    3. Optionally use LLM for more complex cases

    Args:
        raw_text: User's original query
        operator: The operator requiring a target (if any)
        operator_target: Explicit target if already provided
        context: Additional context for LLM
        use_llm: Whether to attempt LLM resolution (default True)

    Returns:
        ResolverResult with resolved bibcode or fallback reason
    """
    start_time = time.perf_counter()

    if not needs_resolution(operator, operator_target, raw_text):
        return ResolverResult(
            success=False,
            fallback_reason="Resolution not needed",
            resolution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    hint = extract_paper_hint(raw_text)
    if hint:
        result = resolve_via_ads_search(hint)
        if result.success:
            result.resolution_time_ms = (time.perf_counter() - start_time) * 1000
            return result

    if use_llm:
        llm_result = resolve_via_llm(raw_text, context)
        if llm_result:
            if llm_result.get("bibcode_guess"):
                return ResolverResult(
                    success=True,
                    bibcode=llm_result["bibcode_guess"],
                    paper_title=llm_result.get("paper_title_guess"),
                    resolution_time_ms=(time.perf_counter() - start_time) * 1000,
                    used_llm=True,
                )

            if llm_result.get("paper_title_guess"):
                ads_result = resolve_via_ads_search(llm_result["paper_title_guess"])
                if ads_result.success:
                    ads_result.used_llm = True
                    ads_result.resolution_time_ms = (time.perf_counter() - start_time) * 1000
                    return ads_result

    return ResolverResult(
        success=False,
        fallback_reason="Could not resolve paper reference",
        resolution_time_ms=(time.perf_counter() - start_time) * 1000,
        used_llm=use_llm,
    )
