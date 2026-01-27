"""Reusable HTML stripping utilities for ADS abstract text.

ADS abstracts frequently contain HTML tags (<SUB>, <SUP>, <I>, <B>, <A>, <P>,
<BR>, MathML fragments, etc.) and HTML entities (&amp;, &lt;, &gt;).  These
corrupt NER tokenizer span offsets and must be removed before any NLP pipeline.

Primary entry point:
    strip_html_tags(text: str) -> str

For annotated data that already carries span annotations, use:
    clean_abstract_and_remap_spans(text, spans) -> (clean_text, remapped_spans)
"""

from __future__ import annotations

import html
import re
from typing import Any

# ---------------------------------------------------------------------------
# Tag / entity patterns
# ---------------------------------------------------------------------------

# Match any HTML/XML tag (opening, closing, self-closing).  Handles:
#   - nested tags:  <A href="...">
#   - self-closing: <BR/>, <BR />
#   - closing tags: </SUB>
#   - case variants: <sub>, <SUB>, <Sub>
_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:[^>]*)?>", re.DOTALL)

# Collapse runs of whitespace that may result from tag removal
_MULTI_SPACE_RE = re.compile(r"[ \t]+")


def strip_html_tags(text: str) -> str:
    """Strip HTML tags and decode HTML entities from *text*.

    1. Remove all HTML/XML tags (preserving their text content).
    2. Decode named and numeric HTML entities (``&amp;`` → ``&``).
    3. Collapse resulting runs of horizontal whitespace to a single space.
    4. Strip leading/trailing whitespace.

    Returns the cleaned plain-text string.
    """
    # Step 1 – remove tags
    cleaned = _TAG_RE.sub("", text)

    # Step 2 – decode HTML entities (handles &amp; &lt; &gt; &#123; etc.)
    cleaned = html.unescape(cleaned)

    # Step 3 – collapse horizontal whitespace (preserve newlines)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)

    # Step 4 – strip outer whitespace
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Offset re-mapping for annotated spans
# ---------------------------------------------------------------------------


def _build_offset_map(original: str) -> list[int]:
    """Build a mapping from *original* char positions → *cleaned* char positions.

    Returns a list of length ``len(original)`` where ``map[i]`` is the index in
    the cleaned string that character ``original[i]`` maps to.  Characters that
    are inside tags receive ``-1`` (they don't appear in the cleaned output).

    The cleaned string is computed using the same logic as ``strip_html_tags``
    minus the final whitespace collapse/strip (those are length-preserving and
    handled separately).
    """
    mapping: list[int] = []
    clean_pos = 0
    i = 0
    n = len(original)

    while i < n:
        # Check if we're at the start of an HTML tag
        tag_match = _TAG_RE.match(original, i)
        if tag_match:
            tag_len = tag_match.end() - tag_match.start()
            mapping.extend([-1] * tag_len)
            i += tag_len
            continue

        # Check if we're at the start of an HTML entity
        entity_match = re.match(r"&(?:#[0-9]+|#x[0-9a-fA-F]+|[A-Za-z]+);", original[i:])
        if entity_match:
            entity_text = entity_match.group(0)
            decoded = html.unescape(entity_text)
            # The first character of the entity maps to the start of decoded text
            mapping.append(clean_pos)
            # Remaining characters of the entity are "consumed"
            for _ in range(1, len(entity_text)):
                mapping.append(-1)
            clean_pos += len(decoded)
            i += len(entity_text)
            continue

        # Regular character — maps 1:1
        mapping.append(clean_pos)
        clean_pos += 1
        i += 1

    return mapping


def _remap_span(
    span: dict[str, Any],
    offset_map: list[int],
    clean_text: str,
) -> dict[str, Any] | None:
    """Remap a single span's start/end offsets from original → clean text.

    Returns a new span dict with updated offsets, or ``None`` if the span
    cannot be validly mapped (e.g. it falls entirely inside a tag).
    """
    orig_start: int = span["start"]
    orig_end: int = span["end"]

    # Find the first non-tag character at or after orig_start
    new_start: int | None = None
    for pos in range(orig_start, min(orig_end, len(offset_map))):
        if offset_map[pos] != -1:
            new_start = offset_map[pos]
            break

    if new_start is None:
        return None  # entire span was inside a tag

    # Find the last non-tag character before orig_end
    new_end: int | None = None
    for pos in range(min(orig_end - 1, len(offset_map) - 1), orig_start - 1, -1):
        if offset_map[pos] != -1:
            new_end = offset_map[pos] + 1
            break

    if new_end is None or new_end <= new_start:
        return None

    remapped_surface = clean_text[new_start:new_end]

    # Build new span (immutable — create new dict)
    return {
        **span,
        "start": new_start,
        "end": new_end,
        "surface": remapped_surface,
    }


def clean_abstract_and_remap_spans(
    text: str,
    spans: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Clean HTML from *text* and remap all *spans* to new offsets.

    Returns ``(clean_text, remapped_spans)`` where every span in
    ``remapped_spans`` has its ``start``, ``end``, and ``surface``
    updated to match the cleaned text.  Spans that cannot be validly
    remapped are dropped.
    """
    clean_text = strip_html_tags(text)
    offset_map = _build_offset_map(text)

    remapped: list[dict[str, Any]] = []
    for span in spans:
        new_span = _remap_span(span, offset_map, clean_text)
        if new_span is not None:
            remapped.append(new_span)

    return clean_text, remapped
