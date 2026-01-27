"""Unit tests for scripts/html_utils.py — HTML stripping and span remapping."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from html_utils import (
    clean_abstract_and_remap_spans,
    strip_html_tags,
)

# ---------------------------------------------------------------------------
# strip_html_tags — basic cases
# ---------------------------------------------------------------------------


class TestStripHtmlTags:
    def test_no_html(self) -> None:
        assert strip_html_tags("Hello world") == "Hello world"

    def test_empty_string(self) -> None:
        assert strip_html_tags("") == ""

    def test_only_whitespace(self) -> None:
        assert strip_html_tags("   ") == ""

    # --- tag removal ---

    def test_simple_tags(self) -> None:
        assert strip_html_tags("<B>bold</B>") == "bold"

    def test_subscript_superscript(self) -> None:
        text = "Ω<SUB>M</SUB> and Ω<SUP>2</SUP>"
        assert strip_html_tags(text) == "ΩM and Ω2"

    def test_nested_tags(self) -> None:
        text = "<P><B>Hello</B> <I>World</I></P>"
        assert strip_html_tags(text) == "Hello World"

    def test_self_closing_tag(self) -> None:
        text = "line one<BR/>line two"
        assert strip_html_tags(text) == "line oneline two"

    def test_self_closing_with_space(self) -> None:
        text = "line one<BR />line two"
        assert strip_html_tags(text) == "line oneline two"

    def test_anchor_tag_with_attributes(self) -> None:
        text = 'Click <A href="http://example.com">here</A> now'
        assert strip_html_tags(text) == "Click here now"

    def test_empty_tags(self) -> None:
        text = "before<INLINE></INLINE>after"
        assert strip_html_tags(text) == "beforeafter"

    def test_mixed_case_tags(self) -> None:
        text = "<sub>x</Sub>"
        assert strip_html_tags(text) == "x"

    def test_tag_with_multiline_attribute(self) -> None:
        text = '<A\nhref="url">link</A>'
        assert strip_html_tags(text) == "link"

    # --- entity decoding ---

    def test_amp_entity(self) -> None:
        assert strip_html_tags("a &amp; b") == "a & b"

    def test_lt_gt_entities(self) -> None:
        assert strip_html_tags("x &lt; y &gt; z") == "x < y > z"

    def test_numeric_entity(self) -> None:
        assert strip_html_tags("&#60;") == "<"

    def test_hex_entity(self) -> None:
        assert strip_html_tags("&#x3C;") == "<"

    # --- whitespace handling ---

    def test_collapse_multiple_spaces(self) -> None:
        text = "word1   word2    word3"
        assert strip_html_tags(text) == "word1 word2 word3"

    def test_tag_removal_creates_spaces(self) -> None:
        text = "before<P>middle</P>after"
        assert strip_html_tags(text) == "beforemiddleafter"

    def test_preserves_newlines(self) -> None:
        text = "line1\nline2\nline3"
        assert strip_html_tags(text) == "line1\nline2\nline3"

    # --- real ADS abstract patterns ---

    def test_ads_subscript_pattern(self) -> None:
        """Typical ADS pattern: Ω<SUB>M</SUB>."""
        text = "Ω<SUB>M</SUB>, and cosmological-constant energy density, Ω<SUB>Λ</SUB>"
        expected = "ΩM, and cosmological-constant energy density, ΩΛ"
        assert strip_html_tags(text) == expected

    def test_ads_mathml_pattern(self) -> None:
        """MathML tags should also be stripped."""
        text = "The value <MML>x</MML> is significant"
        assert strip_html_tags(text) == "The value x is significant"

    def test_malformed_html(self) -> None:
        """Unclosed tags should still be handled (matching what regex catches)."""
        text = "some text <B>bold without close"
        assert strip_html_tags(text) == "some text bold without close"

    def test_combined_tags_and_entities(self) -> None:
        text = "<SUP>a &amp; b</SUP> in &lt;context&gt;"
        assert strip_html_tags(text) == "a & b in <context>"


# ---------------------------------------------------------------------------
# clean_abstract_and_remap_spans
# ---------------------------------------------------------------------------


class TestCleanAbstractAndRemapSpans:
    def test_no_html_spans_unchanged(self) -> None:
        """When there's no HTML, spans should remain identical."""
        text = "cosmic microwave background"
        spans = [
            {
                "surface": "cosmic microwave background",
                "start": 0,
                "end": 27,
                "type": "topic",
                "canonical_id": "uat:123",
                "source_vocabulary": "uat",
            }
        ]
        clean, remapped = clean_abstract_and_remap_spans(text, spans)
        assert clean == text
        assert len(remapped) == 1
        assert remapped[0]["start"] == 0
        assert remapped[0]["end"] == 27
        assert remapped[0]["surface"] == "cosmic microwave background"

    def test_span_after_removed_tag(self) -> None:
        """Span offset must shift left when a tag is removed before it."""
        text = "<B>bold</B> cosmic rays"
        # "cosmic rays" starts at index 12 in original
        spans = [
            {
                "surface": "cosmic rays",
                "start": 12,
                "end": 23,
                "type": "topic",
                "canonical_id": "uat:456",
                "source_vocabulary": "uat",
            }
        ]
        clean, remapped = clean_abstract_and_remap_spans(text, spans)
        assert clean == "bold cosmic rays"
        assert len(remapped) == 1
        assert clean[remapped[0]["start"] : remapped[0]["end"]] == "cosmic rays"

    def test_span_containing_tag(self) -> None:
        """Span wrapping text with an inner tag: Ω<SUB>M</SUB>."""
        text = "The density Ω<SUB>M</SUB> is measured"
        # "Ω<SUB>M</SUB>" spans from index 12 to 25
        spans = [
            {
                "surface": "Ω",
                "start": 12,
                "end": 13,
                "type": "topic",
                "canonical_id": "sweet:omega",
                "source_vocabulary": "sweet",
            }
        ]
        clean, remapped = clean_abstract_and_remap_spans(text, spans)
        assert clean == "The density ΩM is measured"
        assert len(remapped) == 1
        assert clean[remapped[0]["start"] : remapped[0]["end"]] == "Ω"

    def test_span_with_entity(self) -> None:
        """Span containing an HTML entity should decode correctly."""
        text = "a &amp; b"
        spans = [
            {
                "surface": "&amp;",
                "start": 2,
                "end": 7,
                "type": "topic",
                "canonical_id": "test:amp",
                "source_vocabulary": "test",
            }
        ]
        clean, remapped = clean_abstract_and_remap_spans(text, spans)
        assert clean == "a & b"
        assert len(remapped) == 1
        assert remapped[0]["start"] == 2
        assert remapped[0]["end"] == 3
        assert remapped[0]["surface"] == "&"

    def test_multiple_spans_shifted(self) -> None:
        """Multiple spans should all be correctly remapped."""
        text = "<I>alpha</I> and <I>beta</I> rays"
        # In original: "alpha" at 3..8, "beta" at 20..24, "rays" at 29..33
        spans = [
            {
                "surface": "alpha",
                "start": 3,
                "end": 8,
                "type": "topic",
                "canonical_id": "uat:alpha",
                "source_vocabulary": "uat",
            },
            {
                "surface": "beta",
                "start": 20,
                "end": 24,
                "type": "topic",
                "canonical_id": "uat:beta",
                "source_vocabulary": "uat",
            },
            {
                "surface": "rays",
                "start": 29,
                "end": 33,
                "type": "topic",
                "canonical_id": "uat:rays",
                "source_vocabulary": "uat",
            },
        ]
        clean, remapped = clean_abstract_and_remap_spans(text, spans)
        assert clean == "alpha and beta rays"
        assert len(remapped) == 3
        for span in remapped:
            assert clean[span["start"] : span["end"]] == span["surface"]

    def test_span_entirely_inside_tag_is_dropped(self) -> None:
        """A span whose entire range falls within a removed tag should be dropped."""
        text = '<A href="test">link</A> text'
        # Span covering the href attribute (inside the tag)
        spans = [
            {
                "surface": "href",
                "start": 3,
                "end": 7,
                "type": "topic",
                "canonical_id": "test:x",
                "source_vocabulary": "test",
            }
        ]
        clean, remapped = clean_abstract_and_remap_spans(text, spans)
        assert clean == "link text"
        assert len(remapped) == 0

    def test_preserves_extra_span_fields(self) -> None:
        """Extra fields on span dicts should be preserved."""
        text = "<B>hello</B> world"
        spans = [
            {
                "surface": "hello",
                "start": 3,
                "end": 8,
                "type": "topic",
                "canonical_id": "test:hello",
                "source_vocabulary": "test",
                "extra_field": "keep_me",
            }
        ]
        clean, remapped = clean_abstract_and_remap_spans(text, spans)
        assert len(remapped) == 1
        assert remapped[0]["extra_field"] == "keep_me"
        assert clean[remapped[0]["start"] : remapped[0]["end"]] == "hello"

    def test_empty_spans_list(self) -> None:
        """Empty spans list should return empty."""
        clean, remapped = clean_abstract_and_remap_spans("<B>text</B>", [])
        assert clean == "text"
        assert remapped == []

    def test_real_world_pattern(self) -> None:
        """Test with a pattern similar to real ADS data."""
        text = (
            "We report measurements of the mass density, "
            "Ω<SUB>M</SUB>, and cosmological-constant energy density, "
            "Ω<SUB>Λ</SUB>, of the universe"
        )
        spans = [
            {
                "surface": "mass density",
                "start": 33,
                "end": 45,
                "type": "topic",
                "canonical_id": "sweet:mass_density",
                "source_vocabulary": "sweet",
            },
            {
                "surface": "universe",
                "start": text.index("universe"),
                "end": text.index("universe") + len("universe"),
                "type": "topic",
                "canonical_id": "uat:universe",
                "source_vocabulary": "uat",
            },
        ]
        clean, remapped = clean_abstract_and_remap_spans(text, spans)
        assert "<SUB>" not in clean
        assert "ΩM" in clean
        assert "ΩΛ" in clean
        for span in remapped:
            extracted = clean[span["start"] : span["end"]]
            assert extracted == span["surface"], (
                f"Span mismatch: expected '{span['surface']}', got '{extracted}'"
            )
