"""Unit tests for scripts/reannotate_ads_abstracts.py — re-annotation with curated vocab."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from annotate_ads_abstracts import (
    CatalogEntry,
    SpanAnnotation,
    build_keyword_index,
    find_annotation_spans,
)
from reannotate_ads_abstracts import (
    compute_comparison_stats,
    load_sweet_curated,
    reannotate_abstract,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_catalog_entry(
    entry_id: str = "test:1",
    label: str = "test label",
    aliases: list[str] | None = None,
    entry_type: str = "topic",
    source_vocabulary: str = "sweet",
    domain_tags: list[str] | None = None,
) -> CatalogEntry:
    return CatalogEntry(
        entry_id=entry_id,
        label=label,
        aliases=aliases or [],
        entry_type=entry_type,
        source_vocabulary=source_vocabulary,
        domain_tags=domain_tags or ["earthscience"],
    )


def _make_record(
    bibcode: str = "2020TEST...1A",
    abstract: str = "The photosphere of Mars has thermal emission.",
    abstract_clean: str | None = None,
    spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "bibcode": bibcode,
        "title": "Test Paper",
        "abstract": abstract,
        "abstract_clean": abstract_clean or abstract,
        "database": ["astronomy"],
        "keywords": [],
        "year": "2020",
        "doctype": "article",
        "citation_count": 10,
        "domain_category": "astronomy",
        "spans": spans or [],
    }


# ---------------------------------------------------------------------------
# load_sweet_curated
# ---------------------------------------------------------------------------


class TestLoadSweetCurated:
    def test_loads_curated_entries(self, tmp_path: Path) -> None:
        entries = [
            {"id": "sweet:x/Photosphere", "label": "Photosphere", "aliases": [],
             "source_vocabulary": "sweet", "domain_tags": ["earthscience"]},
            {"id": "sweet:y/Ionosphere", "label": "Ionosphere", "aliases": ["ionic layer"],
             "source_vocabulary": "sweet", "domain_tags": ["earthscience"]},
        ]
        path = tmp_path / "sweet_curated.jsonl"
        path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        result = load_sweet_curated(path)
        assert len(result) == 2
        assert result[0].entry_id == "sweet:x/Photosphere"
        assert result[0].label == "Photosphere"
        assert result[0].entry_type == "topic"
        assert result[0].source_vocabulary == "sweet"
        assert result[1].aliases == ["ionic layer"]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_sweet_curated(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        result = load_sweet_curated(path)
        assert result == []

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        content = (
            '{"id": "sweet:a/X", "label": "Xray", "aliases": [], '
            '"source_vocabulary": "sweet", "domain_tags": ["earthscience"]}\n'
            "\n"
            '{"id": "sweet:b/Y", "label": "Ylem", "aliases": [], '
            '"source_vocabulary": "sweet", "domain_tags": ["earthscience"]}\n'
        )
        path = tmp_path / "curated.jsonl"
        path.write_text(content, encoding="utf-8")
        result = load_sweet_curated(path)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# reannotate_abstract
# ---------------------------------------------------------------------------


class TestReannotateAbstract:
    def test_uses_abstract_clean(self) -> None:
        """Re-annotation should use abstract_clean, not abstract."""
        record = _make_record(
            abstract="<SUB>photosphere</SUB> is hot",
            abstract_clean="photosphere is hot",
        )
        entry = _make_catalog_entry(
            label="photosphere",
            source_vocabulary="uat",
        )
        index = build_keyword_index([entry], min_label_length=4)
        result = reannotate_abstract(record, index)

        assert len(result["spans"]) == 1
        span = result["spans"][0]
        # Offset should match abstract_clean, not abstract (with HTML)
        assert span["surface"] == "photosphere"
        assert span["start"] == 0
        assert span["end"] == 11
        assert result["abstract_clean"][span["start"] : span["end"]] == "photosphere"

    def test_falls_back_to_abstract_when_no_clean(self) -> None:
        record = {
            "bibcode": "2020TEST",
            "title": "Test",
            "abstract": "The photosphere is hot",
            "database": [],
            "keywords": [],
            "year": "2020",
            "doctype": "article",
            "citation_count": 0,
            "domain_category": "astronomy",
            "spans": [],
        }
        entry = _make_catalog_entry(
            label="photosphere",
            source_vocabulary="uat",
        )
        index = build_keyword_index([entry], min_label_length=4)
        result = reannotate_abstract(record, index)

        assert len(result["spans"]) == 1
        assert result["spans"][0]["surface"] == "photosphere"

    def test_preserves_record_fields(self) -> None:
        record = _make_record(
            bibcode="2021MARS...1B",
            abstract="plain text with photosphere",
            abstract_clean="plain text with photosphere",
        )
        entry = _make_catalog_entry(label="photosphere")
        index = build_keyword_index([entry], min_label_length=4)
        result = reannotate_abstract(record, index)

        assert result["bibcode"] == "2021MARS...1B"
        assert result["title"] == "Test Paper"
        assert result["year"] == "2020"
        assert result["domain_category"] == "astronomy"
        assert "abstract_clean" in result

    def test_curated_sweet_excludes_stopword_surfaces(self) -> None:
        """Words in STOPWORD_SURFACES are excluded even if in the keyword index."""
        record = _make_record(
            abstract="The analysis of the model uses a method",
            abstract_clean="The analysis of the model uses a method",
        )
        # "analysis" and "model" are in STOPWORD_SURFACES
        entries = [
            _make_catalog_entry(
                entry_id="sweet:a/Analysis",
                label="analysis",
                source_vocabulary="sweet",
            ),
            _make_catalog_entry(
                entry_id="sweet:b/Model",
                label="model",
                source_vocabulary="sweet",
            ),
            _make_catalog_entry(
                entry_id="sweet:c/Method",
                label="method",
                source_vocabulary="sweet",
            ),
        ]
        index = build_keyword_index(entries, min_label_length=4)
        result = reannotate_abstract(record, index)

        surfaces = [s["surface"] for s in result["spans"]]
        assert "analysis" not in surfaces
        assert "model" not in surfaces
        assert "method" not in surfaces

    def test_multiple_vocabulary_spans(self) -> None:
        record = _make_record(
            abstract="The photosphere of Mars shows thermal emission from regolith",
            abstract_clean="The photosphere of Mars shows thermal emission from regolith",
        )
        entries = [
            _make_catalog_entry(
                entry_id="uat:photosphere",
                label="photosphere",
                source_vocabulary="uat",
                entry_type="topic",
            ),
            _make_catalog_entry(
                entry_id="sweet:thermal",
                label="thermal emission",
                source_vocabulary="sweet",
                entry_type="topic",
            ),
            _make_catalog_entry(
                entry_id="gcmd:regolith",
                label="regolith",
                source_vocabulary="gcmd",
                entry_type="topic",
            ),
        ]
        index = build_keyword_index(entries, min_label_length=4)
        result = reannotate_abstract(record, index)

        vocabs = {s["source_vocabulary"] for s in result["spans"]}
        assert "uat" in vocabs
        assert "sweet" in vocabs
        assert "gcmd" in vocabs

    def test_span_offsets_are_valid(self) -> None:
        record = _make_record(
            abstract_clean="The ionosphere and magnetosphere interact",
        )
        entries = [
            _make_catalog_entry(label="ionosphere"),
            _make_catalog_entry(label="magnetosphere"),
        ]
        index = build_keyword_index(entries, min_label_length=4)
        result = reannotate_abstract(record, index)

        text = result["abstract_clean"]
        for span in result["spans"]:
            assert text[span["start"] : span["end"]] == span["surface"]


# ---------------------------------------------------------------------------
# compute_comparison_stats
# ---------------------------------------------------------------------------


class TestComputeComparisonStats:
    def test_basic_comparison(self) -> None:
        old_records = [
            _make_record(
                bibcode="2020A",
                spans=[
                    {"surface": "present", "start": 0, "end": 7,
                     "type": "topic", "source_vocabulary": "sweet"},
                    {"surface": "model", "start": 10, "end": 15,
                     "type": "topic", "source_vocabulary": "sweet"},
                    {"surface": "Mars", "start": 20, "end": 24,
                     "type": "entity", "source_vocabulary": "planetary"},
                ],
            ),
        ]
        new_records = [
            _make_record(
                bibcode="2020A",
                spans=[
                    {"surface": "Mars", "start": 20, "end": 24,
                     "type": "entity", "source_vocabulary": "planetary"},
                ],
            ),
        ]

        stats = compute_comparison_stats(old_records, new_records)

        assert stats["summary"]["old_total_spans"] == 3
        assert stats["summary"]["new_total_spans"] == 1
        assert stats["summary"]["reduction"] == 2
        assert stats["summary"]["reduction_pct"] == 66.7

    def test_per_vocabulary_breakdown(self) -> None:
        old_records = [
            _make_record(
                bibcode="2020A",
                spans=[
                    {"surface": "x", "start": 0, "end": 1,
                     "type": "topic", "source_vocabulary": "sweet"},
                    {"surface": "y", "start": 2, "end": 3,
                     "type": "topic", "source_vocabulary": "sweet"},
                    {"surface": "z", "start": 4, "end": 5,
                     "type": "topic", "source_vocabulary": "uat"},
                ],
            ),
        ]
        new_records = [
            _make_record(
                bibcode="2020A",
                spans=[
                    {"surface": "z", "start": 4, "end": 5,
                     "type": "topic", "source_vocabulary": "uat"},
                ],
            ),
        ]

        stats = compute_comparison_stats(old_records, new_records)
        vocab_map = {v["vocabulary"]: v for v in stats["by_vocabulary"]}

        assert vocab_map["sweet"]["old_count"] == 2
        assert vocab_map["sweet"]["new_count"] == 0
        assert vocab_map["sweet"]["reduction"] == 2
        assert vocab_map["uat"]["old_count"] == 1
        assert vocab_map["uat"]["new_count"] == 1

    def test_empty_records(self) -> None:
        stats = compute_comparison_stats([], [])
        assert stats["summary"]["old_total_spans"] == 0
        assert stats["summary"]["new_total_spans"] == 0

    def test_per_abstract_summary(self) -> None:
        old_records = [
            _make_record(
                bibcode="A",
                spans=[
                    {"surface": "x", "start": 0, "end": 1,
                     "type": "topic", "source_vocabulary": "sweet"},
                ] * 10,
            ),
            _make_record(
                bibcode="B",
                spans=[
                    {"surface": "y", "start": 0, "end": 1,
                     "type": "topic", "source_vocabulary": "sweet"},
                ] * 5,
            ),
        ]
        new_records = [
            _make_record(
                bibcode="A",
                spans=[
                    {"surface": "x", "start": 0, "end": 1,
                     "type": "topic", "source_vocabulary": "sweet"},
                ] * 3,
            ),
            _make_record(
                bibcode="B",
                spans=[
                    {"surface": "y", "start": 0, "end": 1,
                     "type": "topic", "source_vocabulary": "sweet"},
                ] * 2,
            ),
        ]

        stats = compute_comparison_stats(old_records, new_records)
        per_ab = stats["per_abstract_summary"]
        assert per_ab["max_reduction"] == 7  # 10-3
        assert per_ab["min_reduction"] == 3  # 5-2
        assert per_ab["avg_reduction"] == 5.0  # (7+3)/2

    def test_type_breakdown(self) -> None:
        old_records = [
            _make_record(
                bibcode="A",
                spans=[
                    {"surface": "x", "start": 0, "end": 1,
                     "type": "topic", "source_vocabulary": "sweet"},
                    {"surface": "y", "start": 2, "end": 3,
                     "type": "entity", "source_vocabulary": "ror"},
                ],
            ),
        ]
        new_records = [
            _make_record(
                bibcode="A",
                spans=[
                    {"surface": "z", "start": 4, "end": 5,
                     "type": "entity", "source_vocabulary": "ror"},
                ],
            ),
        ]

        stats = compute_comparison_stats(old_records, new_records)
        assert stats["old_by_type"] == {"entity": 1, "topic": 1}
        assert stats["new_by_type"] == {"entity": 1}


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_reannotation_reduces_sweet_spans(self) -> None:
        """Curated SWEET should produce fewer spans than full SWEET."""
        text = "The thermosphere atmosphere shows convection patterns in the ionosphere"

        # Full SWEET would include common words like "atmosphere"
        full_sweet_entries = [
            _make_catalog_entry(
                entry_id="sweet:a/Thermosphere",
                label="thermosphere",
                source_vocabulary="sweet",
            ),
            _make_catalog_entry(
                entry_id="sweet:b/Atmosphere",
                label="atmosphere",
                source_vocabulary="sweet",
            ),
            _make_catalog_entry(
                entry_id="sweet:c/Convection",
                label="convection",
                source_vocabulary="sweet",
            ),
            _make_catalog_entry(
                entry_id="sweet:d/Ionosphere",
                label="ionosphere",
                source_vocabulary="sweet",
            ),
        ]

        # Curated removes "atmosphere" (common English word)
        curated_sweet_entries = [
            _make_catalog_entry(
                entry_id="sweet:a/Thermosphere",
                label="thermosphere",
                source_vocabulary="sweet",
            ),
            _make_catalog_entry(
                entry_id="sweet:c/Convection",
                label="convection",
                source_vocabulary="sweet",
            ),
            _make_catalog_entry(
                entry_id="sweet:d/Ionosphere",
                label="ionosphere",
                source_vocabulary="sweet",
            ),
        ]

        full_index = build_keyword_index(full_sweet_entries, min_label_length=4)
        curated_index = build_keyword_index(curated_sweet_entries, min_label_length=4)

        full_spans = find_annotation_spans(text, full_index)
        curated_spans = find_annotation_spans(text, curated_index)

        assert len(curated_spans) < len(full_spans)
        # atmosphere should appear in full but not curated
        full_surfaces = {s.surface.lower() for s in full_spans}
        curated_surfaces = {s.surface.lower() for s in curated_spans}
        assert "atmosphere" in full_surfaces
        assert "atmosphere" not in curated_surfaces

    def test_html_clean_text_used_for_offsets(self) -> None:
        """Spans should have offsets matching abstract_clean, not raw abstract."""
        record = _make_record(
            abstract="The <SUB>photosphere</SUB> layer",
            abstract_clean="The photosphere layer",
        )
        entry = _make_catalog_entry(
            label="photosphere",
            source_vocabulary="uat",
        )
        index = build_keyword_index([entry], min_label_length=4)
        result = reannotate_abstract(record, index)

        assert len(result["spans"]) == 1
        span = result["spans"][0]
        clean = result["abstract_clean"]
        assert clean[span["start"] : span["end"]] == "photosphere"
        # In clean text, "photosphere" starts at position 4 ("The ")
        assert span["start"] == 4
