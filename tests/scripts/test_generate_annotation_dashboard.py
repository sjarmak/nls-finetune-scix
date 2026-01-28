"""Unit tests for scripts/generate_annotation_dashboard.py — dashboard generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from generate_annotation_dashboard import (
    generate_dashboard,
    inject_data_into_template,
    load_jsonl,
    merge_data,
    parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_REANNOTATED = [
    {
        "bibcode": "2020ApJ...900..100A",
        "title": "Galaxy Formation Study",
        "abstract": "<SUB>test</SUB> abstract",
        "abstract_clean": "test abstract about galaxies",
        "domain_category": "astronomy",
        "citation_count": 42,
        "spans": [
            {
                "surface": "galaxies",
                "start": 19,
                "end": 27,
                "type": "topic",
                "canonical_id": "uat:123",
                "source_vocabulary": "uat",
            }
        ],
    },
    {
        "bibcode": "2021GeoRL..48..200B",
        "title": "Climate Modeling",
        "abstract": "Earth climate",
        "abstract_clean": "Earth climate patterns in the tropics",
        "domain_category": "earthscience",
        "citation_count": 15,
        "spans": [
            {
                "surface": "climate",
                "start": 6,
                "end": 13,
                "type": "topic",
                "canonical_id": "gcmd:456",
                "source_vocabulary": "gcmd",
            },
            {
                "surface": "tropics",
                "start": 29,
                "end": 36,
                "type": "topic",
                "canonical_id": "sweet:789",
                "source_vocabulary": "sweet",
            },
        ],
    },
]

SAMPLE_PREDICTIONS = [
    {
        "bibcode": "2020ApJ...900..100A",
        "title": "Galaxy Formation Study",
        "abstract_clean": "test abstract about galaxies",
        "domain_category": "astronomy",
        "spans": [
            {
                "surface": "galaxies",
                "start": 19,
                "end": 27,
                "type": "topic",
                "confidence": 0.95,
            },
            {
                "surface": "abstract",
                "start": 5,
                "end": 13,
                "type": "topic",
                "confidence": 0.45,
            },
        ],
    },
    {
        "bibcode": "2021GeoRL..48..200B",
        "title": "Climate Modeling",
        "abstract_clean": "Earth climate patterns in the tropics",
        "domain_category": "earthscience",
        "spans": [
            {
                "surface": "climate patterns",
                "start": 6,
                "end": 22,
                "type": "topic",
                "confidence": 0.87,
            }
        ],
    },
]

MINIMAL_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
<script>
const DATA = /*DATA_PLACEHOLDER*/[];
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        records = [{"bibcode": "A"}, {"bibcode": "B"}]
        path.write_text("\n".join(json.dumps(r) for r in records))

        result = load_jsonl(path)
        assert len(result) == 2
        assert result[0]["bibcode"] == "A"
        assert result[1]["bibcode"] == "B"

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n\n')

        result = load_jsonl(path)
        assert len(result) == 2

    def test_skips_malformed_lines(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text('{"good": true}\nnot json\n{"also_good": true}\n')

        result = load_jsonl(path)
        assert len(result) == 2
        assert result[0]["good"] is True
        assert result[1]["also_good"] is True

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "malformed JSON" in captured.err

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")

        result = load_jsonl(path)
        assert result == []


# ---------------------------------------------------------------------------
# merge_data
# ---------------------------------------------------------------------------


class TestMergeData:
    def test_merges_matching_bibcodes(self) -> None:
        merged = merge_data(SAMPLE_REANNOTATED, SAMPLE_PREDICTIONS)
        assert len(merged) == 2

    def test_sorted_by_bibcode(self) -> None:
        merged = merge_data(SAMPLE_REANNOTATED, SAMPLE_PREDICTIONS)
        bibcodes = [r["bibcode"] for r in merged]
        assert bibcodes == sorted(bibcodes)

    def test_auto_spans_from_reannotated(self) -> None:
        merged = merge_data(SAMPLE_REANNOTATED, SAMPLE_PREDICTIONS)
        astro = next(r for r in merged if r["bibcode"] == "2020ApJ...900..100A")
        assert len(astro["auto_spans"]) == 1
        assert astro["auto_spans"][0]["surface"] == "galaxies"
        assert astro["auto_spans"][0]["source_vocabulary"] == "uat"

    def test_model_spans_from_predictions(self) -> None:
        merged = merge_data(SAMPLE_REANNOTATED, SAMPLE_PREDICTIONS)
        astro = next(r for r in merged if r["bibcode"] == "2020ApJ...900..100A")
        assert len(astro["model_spans"]) == 2
        assert astro["model_spans"][0]["confidence"] == 0.95

    def test_preserves_metadata(self) -> None:
        merged = merge_data(SAMPLE_REANNOTATED, SAMPLE_PREDICTIONS)
        astro = next(r for r in merged if r["bibcode"] == "2020ApJ...900..100A")
        assert astro["title"] == "Galaxy Formation Study"
        assert astro["abstract_clean"] == "test abstract about galaxies"
        assert astro["domain_category"] == "astronomy"
        assert astro["citation_count"] == 42

    def test_missing_predictions_gets_empty_model_spans(self) -> None:
        reannotated = [SAMPLE_REANNOTATED[0]]
        predictions: list[dict[str, Any]] = []  # no predictions at all

        merged = merge_data(reannotated, predictions)
        assert len(merged) == 1
        assert merged[0]["model_spans"] == []

    def test_expected_output_keys(self) -> None:
        merged = merge_data(SAMPLE_REANNOTATED, SAMPLE_PREDICTIONS)
        expected_keys = {
            "bibcode",
            "title",
            "abstract_clean",
            "domain_category",
            "citation_count",
            "auto_spans",
            "model_spans",
        }
        for record in merged:
            assert set(record.keys()) == expected_keys

    def test_does_not_include_raw_abstract(self) -> None:
        """The merged output should NOT include the raw HTML abstract."""
        merged = merge_data(SAMPLE_REANNOTATED, SAMPLE_PREDICTIONS)
        for record in merged:
            assert "abstract" not in record

    def test_multiple_auto_spans(self) -> None:
        merged = merge_data(SAMPLE_REANNOTATED, SAMPLE_PREDICTIONS)
        earth = next(r for r in merged if r["bibcode"] == "2021GeoRL..48..200B")
        assert len(earth["auto_spans"]) == 2
        surfaces = {s["surface"] for s in earth["auto_spans"]}
        assert surfaces == {"climate", "tropics"}

    def test_defaults_for_missing_fields(self) -> None:
        """Records with missing optional fields get sensible defaults."""
        minimal = [{"bibcode": "2020TEST", "spans": []}]
        merged = merge_data(minimal, [])
        assert merged[0]["title"] == ""
        assert merged[0]["abstract_clean"] == ""
        assert merged[0]["domain_category"] == "unknown"
        assert merged[0]["citation_count"] == 0


# ---------------------------------------------------------------------------
# inject_data_into_template
# ---------------------------------------------------------------------------


class TestInjectData:
    def test_replaces_placeholder(self) -> None:
        data = [{"bibcode": "TEST", "auto_spans": [], "model_spans": []}]
        result = inject_data_into_template(MINIMAL_TEMPLATE, data)
        assert "DATA_PLACEHOLDER" not in result
        assert '"bibcode"' in result or "'bibcode'" in result

    def test_roundtrip_json(self) -> None:
        data = [{"bibcode": "TEST", "value": 42}]
        result = inject_data_into_template(MINIMAL_TEMPLATE, data)
        # Extract the JSON between "const DATA = " and ";\n"
        start = result.index("const DATA = ") + len("const DATA = ")
        end = result.index(";\n", start)
        parsed = json.loads(result[start:end])
        assert parsed == data

    def test_preserves_unicode(self) -> None:
        data = [{"text": "H\u03b1 emission \u2014 galaxy"}]
        result = inject_data_into_template(MINIMAL_TEMPLATE, data)
        assert "H\u03b1" in result
        assert "\u2014" in result

    def test_raises_on_missing_placeholder(self) -> None:
        bad_template = "<html><body>No placeholder here</body></html>"
        with pytest.raises(ValueError, match="placeholder"):
            inject_data_into_template(bad_template, [])

    def test_only_replaces_first_occurrence(self) -> None:
        double_template = MINIMAL_TEMPLATE + "\n// /*DATA_PLACEHOLDER*/[] second"
        data = [{"test": True}]
        result = inject_data_into_template(double_template, data)
        # Second occurrence should remain
        assert "/*DATA_PLACEHOLDER*/[]" in result


# ---------------------------------------------------------------------------
# generate_dashboard (integration)
# ---------------------------------------------------------------------------


class TestGenerateDashboard:
    def _write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records))

    def test_generates_html_file(self, tmp_path: Path) -> None:
        reann_path = tmp_path / "reannotated.jsonl"
        pred_path = tmp_path / "predictions.jsonl"
        template_path = tmp_path / "template.html"
        output_path = tmp_path / "output.html"

        self._write_jsonl(reann_path, SAMPLE_REANNOTATED)
        self._write_jsonl(pred_path, SAMPLE_PREDICTIONS)
        template_path.write_text(MINIMAL_TEMPLATE)

        stats = generate_dashboard(reann_path, pred_path, template_path, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "const DATA = " in content
        assert "DATA_PLACEHOLDER" not in content

    def test_stats_are_correct(self, tmp_path: Path) -> None:
        reann_path = tmp_path / "reannotated.jsonl"
        pred_path = tmp_path / "predictions.jsonl"
        template_path = tmp_path / "template.html"
        output_path = tmp_path / "output.html"

        self._write_jsonl(reann_path, SAMPLE_REANNOTATED)
        self._write_jsonl(pred_path, SAMPLE_PREDICTIONS)
        template_path.write_text(MINIMAL_TEMPLATE)

        stats = generate_dashboard(reann_path, pred_path, template_path, output_path)

        assert stats["total_abstracts"] == 2
        assert stats["total_auto_spans"] == 3  # 1 + 2
        assert stats["total_model_spans"] == 3  # 2 + 1
        assert stats["auto_spans_per_abstract"] == 1.5
        assert stats["model_spans_per_abstract"] == 1.5
        assert set(stats["domains"]) == {"astronomy", "earthscience"}
        assert stats["auto_span_types"]["topic"] == 3
        assert stats["model_span_types"]["topic"] == 3
        assert stats["output_size_kb"] > 0

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        reann_path = tmp_path / "reannotated.jsonl"
        pred_path = tmp_path / "predictions.jsonl"
        template_path = tmp_path / "template.html"
        output_path = tmp_path / "nested" / "dir" / "output.html"

        self._write_jsonl(reann_path, SAMPLE_REANNOTATED)
        self._write_jsonl(pred_path, SAMPLE_PREDICTIONS)
        template_path.write_text(MINIMAL_TEMPLATE)

        generate_dashboard(reann_path, pred_path, template_path, output_path)
        assert output_path.exists()

    def test_raises_on_empty_reannotated(self, tmp_path: Path) -> None:
        reann_path = tmp_path / "empty.jsonl"
        pred_path = tmp_path / "predictions.jsonl"
        template_path = tmp_path / "template.html"
        output_path = tmp_path / "output.html"

        reann_path.write_text("")
        self._write_jsonl(pred_path, SAMPLE_PREDICTIONS)
        template_path.write_text(MINIMAL_TEMPLATE)

        with pytest.raises(ValueError, match="No records"):
            generate_dashboard(reann_path, pred_path, template_path, output_path)

    def test_raises_on_empty_predictions(self, tmp_path: Path) -> None:
        reann_path = tmp_path / "reannotated.jsonl"
        pred_path = tmp_path / "empty.jsonl"
        template_path = tmp_path / "template.html"
        output_path = tmp_path / "output.html"

        self._write_jsonl(reann_path, SAMPLE_REANNOTATED)
        pred_path.write_text("")
        template_path.write_text(MINIMAL_TEMPLATE)

        with pytest.raises(ValueError, match="No records"):
            generate_dashboard(reann_path, pred_path, template_path, output_path)

    def test_embedded_data_is_valid_json(self, tmp_path: Path) -> None:
        """The embedded JSON should be parseable from the generated HTML."""
        reann_path = tmp_path / "reannotated.jsonl"
        pred_path = tmp_path / "predictions.jsonl"
        template_path = tmp_path / "template.html"
        output_path = tmp_path / "output.html"

        self._write_jsonl(reann_path, SAMPLE_REANNOTATED)
        self._write_jsonl(pred_path, SAMPLE_PREDICTIONS)
        template_path.write_text(MINIMAL_TEMPLATE)

        generate_dashboard(reann_path, pred_path, template_path, output_path)

        content = output_path.read_text()
        start = content.index("const DATA = ") + len("const DATA = ")
        end = content.index(";\n", start)
        parsed = json.loads(content[start:end])

        assert len(parsed) == 2
        assert all("auto_spans" in r for r in parsed)
        assert all("model_spans" in r for r in parsed)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.reannotated == Path("data/evaluation/ads_sample_reannotated.jsonl")
        assert args.predictions == Path("data/evaluation/ads_sample_predictions.jsonl")
        assert args.template == Path("scripts/annotation_dashboard_template.html")
        assert args.output == Path("data/evaluation/review_ner_annotations.html")

    def test_custom_paths(self) -> None:
        args = parse_args([
            "--reannotated", "/tmp/r.jsonl",
            "--predictions", "/tmp/p.jsonl",
            "--template", "/tmp/t.html",
            "--output", "/tmp/o.html",
        ])
        assert args.reannotated == Path("/tmp/r.jsonl")
        assert args.predictions == Path("/tmp/p.jsonl")
        assert args.template == Path("/tmp/t.html")
        assert args.output == Path("/tmp/o.html")
