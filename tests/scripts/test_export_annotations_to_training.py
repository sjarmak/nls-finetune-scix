"""Unit tests for export_annotations_to_training.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from export_annotations_to_training import (
    ConversionStats,
    SpanValidationError,
    convert_to_enrichment_record,
    process_annotations,
    validate_span_offsets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dashboard_record():
    """A sample dashboard export record."""
    text = "This study investigates stellar evolution in the Milky Way galaxy using optical filters."
    return {
        "bibcode": "2024ApJ...900..123A",
        "title": "A Study of Stellar Evolution",
        "abstract_clean": text,
        "domain_category": "astronomy",
        "spans": [
            {
                "surface": "stellar evolution",
                "start": text.index("stellar evolution"),
                "end": text.index("stellar evolution") + len("stellar evolution"),
                "type": "topic",
                "source": "auto",
                "canonical_id": "uat:1600",
                "source_vocabulary": "uat",
                "confidence": 1.0,
            },
            {
                "surface": "Milky Way",
                "start": text.index("Milky Way"),
                "end": text.index("Milky Way") + len("Milky Way"),
                "type": "topic",
                "source": "model",
                "canonical_id": "uat:1054",
                "source_vocabulary": "uat",
                "confidence": 0.95,
            },
            {
                "surface": "optical filters",
                "start": text.index("optical filters"),
                "end": text.index("optical filters") + len("optical filters"),
                "type": "topic",
                "source": "agreement",
                "canonical_id": "uat:2331",
                "source_vocabulary": "uat",
                "confidence": 0.98,
            },
        ],
        "review_status": "reviewed",
        "notes": "Good quality annotations",
    }


@pytest.fixture
def sample_dashboard_record_with_entity():
    """A sample dashboard record with entity span."""
    text = "Research conducted at the Massachusetts Institute of Technology on climate models."
    return {
        "bibcode": "2024Sci...350..456B",
        "title": "Research at MIT",
        "abstract_clean": text,
        "domain_category": "multidisciplinary",
        "spans": [
            {
                "surface": "Massachusetts Institute of Technology",
                "start": text.index("Massachusetts Institute of Technology"),
                "end": text.index("Massachusetts Institute of Technology") + len("Massachusetts Institute of Technology"),
                "type": "entity",
                "source": "auto",
                "canonical_id": "ror:042nb2s44",
                "source_vocabulary": "ror",
                "confidence": 1.0,
            },
            {
                "surface": "climate models",
                "start": text.index("climate models"),
                "end": text.index("climate models") + len("climate models"),
                "type": "topic",
                "source": "user",
                "canonical_id": "",
                "source_vocabulary": "human",
                "confidence": 1.0,
            },
        ],
        "review_status": "reviewed",
        "notes": "",
    }


@pytest.fixture
def sample_dashboard_record_invalid_offsets():
    """A sample dashboard record with invalid span offsets."""
    text = "This is a test abstract with some text."
    return {
        "bibcode": "2024Nat...500..789C",
        "title": "Invalid Offsets Example",
        "abstract_clean": text,
        "domain_category": "earthscience",
        "spans": [
            {
                "surface": "some text",
                "start": text.index("some text"),
                "end": text.index("some text") + len("some text"),
                "type": "topic",
                "source": "user",
                "canonical_id": "",
                "source_vocabulary": "human",
                "confidence": 1.0,
            },
            {
                "surface": "wrong surface",
                "start": 10,
                "end": 14,
                "type": "topic",
                "source": "user",
                "canonical_id": "",
                "source_vocabulary": "human",
                "confidence": 1.0,
            },
        ],
        "review_status": "reviewed",
        "notes": "",
    }


# ---------------------------------------------------------------------------
# Tests: validate_span_offsets
# ---------------------------------------------------------------------------


def test_validate_span_offsets_valid():
    """Test validation with all valid spans."""
    text = "This is a test with stellar evolution and optical filters."
    spans = [
        {"surface": "stellar evolution", "start": 20, "end": 37},
        {"surface": "optical filters", "start": 42, "end": 57},
    ]

    errors = validate_span_offsets(text, spans)
    assert len(errors) == 0


def test_validate_span_offsets_invalid():
    """Test validation with invalid span offsets."""
    text = "This is a test abstract."
    spans = [
        {"surface": "wrong text", "start": 10, "end": 14},  # actual: "test"
    ]

    errors = validate_span_offsets(text, spans)
    assert len(errors) == 1
    assert errors[0].expected == "wrong text"
    assert errors[0].actual == "test"
    assert errors[0].span_start == 10
    assert errors[0].span_end == 14


def test_validate_span_offsets_empty():
    """Test validation with no spans."""
    text = "Some text"
    spans = []

    errors = validate_span_offsets(text, spans)
    assert len(errors) == 0


def test_validate_span_offsets_multiple_errors():
    """Test validation with multiple invalid spans."""
    text = "The quick brown fox jumps over the lazy dog."
    spans = [
        {"surface": "quick", "start": 4, "end": 9},  # valid
        {"surface": "wrong", "start": 10, "end": 15},  # actual: "brown"
        {"surface": "fox", "start": 16, "end": 19},  # valid
        {"surface": "bad", "start": 40, "end": 43},  # actual: "dog"
    ]

    errors = validate_span_offsets(text, spans)
    assert len(errors) == 2
    assert errors[0].expected == "wrong"
    assert errors[0].actual == "brown"
    assert errors[1].expected == "bad"
    assert errors[1].actual == "dog"


def test_validate_span_offsets_context():
    """Test that validation errors include context."""
    text = "A" * 50 + "TARGET" + "B" * 50
    spans = [
        {"surface": "WRONG", "start": 50, "end": 56},  # actual: "TARGET"
    ]

    errors = validate_span_offsets(text, spans)
    assert len(errors) == 1
    # Context should include 20 chars before and after
    assert "A" * 20 in errors[0].text_snippet
    assert "TARGET" in errors[0].text_snippet
    assert "B" * 20 in errors[0].text_snippet


# ---------------------------------------------------------------------------
# Tests: convert_to_enrichment_record
# ---------------------------------------------------------------------------


def test_convert_to_enrichment_record_basic(sample_dashboard_record):
    """Test basic conversion of dashboard record to enrichment format."""
    result = convert_to_enrichment_record(sample_dashboard_record, "train")

    assert result["id"].startswith("enr_hum_")
    assert len(result["id"]) == 18  # enr_hum_ + 10 hex chars
    assert result["text"] == sample_dashboard_record["abstract_clean"]
    assert result["text_type"] == "abstract"
    assert len(result["spans"]) == 3
    assert len(result["topics"]) == 3  # All spans are topic type with canonical_id
    assert result["provenance"]["source"] == "human_annotation"
    assert result["provenance"]["original_bibcode"] == "2024ApJ...900..123A"
    assert result["provenance"]["domain_category"] == "astronomy"
    assert result["provenance"]["review_status"] == "reviewed"
    assert result["provenance"]["notes"] == "Good quality annotations"


def test_convert_to_enrichment_record_spans_format(sample_dashboard_record):
    """Test that spans are converted to the correct format."""
    result = convert_to_enrichment_record(sample_dashboard_record, "train")

    span = result["spans"][0]
    assert span["surface"] == "stellar evolution"
    assert span["start"] == sample_dashboard_record["spans"][0]["start"]
    assert span["end"] == sample_dashboard_record["spans"][0]["end"]
    assert span["type"] == "topic"
    assert span["canonical_id"] == "uat:1600"
    assert span["source_vocabulary"] == "uat"
    assert span["confidence"] == 1.0


def test_convert_to_enrichment_record_topics_format(sample_dashboard_record):
    """Test that topics list is populated correctly."""
    result = convert_to_enrichment_record(sample_dashboard_record, "train")

    assert len(result["topics"]) == 3

    topic = result["topics"][0]
    assert topic["concept_id"] == "uat:1600"
    assert topic["label"] == "stellar evolution"
    assert topic["source_vocabulary"] == "uat"
    assert topic["confidence"] == 1.0

    # Check that topic concept_ids are unique
    concept_ids = [t["concept_id"] for t in result["topics"]]
    assert len(concept_ids) == len(set(concept_ids))


def test_convert_to_enrichment_record_with_entity(sample_dashboard_record_with_entity):
    """Test conversion with entity-type spans (should not appear in topics)."""
    result = convert_to_enrichment_record(sample_dashboard_record_with_entity, "val")

    assert len(result["spans"]) == 2
    assert result["spans"][0]["type"] == "entity"
    assert result["spans"][1]["type"] == "topic"

    # Only topic spans with canonical_id should appear in topics
    assert len(result["topics"]) == 0  # Second span has empty canonical_id


def test_convert_to_enrichment_record_missing_fields(sample_dashboard_record):
    """Test conversion when optional fields are missing."""
    # Remove optional fields
    record = sample_dashboard_record.copy()
    record["spans"] = [
        {
            "surface": "test",
            "start": 0,
            "end": 4,
            "type": "topic",
            "source": "user",
        }
    ]
    del record["review_status"]
    del record["notes"]
    del record["domain_category"]

    result = convert_to_enrichment_record(record, "train")

    # Should use defaults
    assert result["spans"][0]["canonical_id"] == ""
    assert result["spans"][0]["source_vocabulary"] == "human"
    assert result["spans"][0]["confidence"] == 1.0
    assert result["provenance"]["domain_category"] == "unknown"
    assert result["provenance"]["review_status"] == "reviewed"
    assert result["provenance"]["notes"] == ""


def test_convert_to_enrichment_record_unique_ids():
    """Test that different records get different IDs."""
    record1 = {
        "bibcode": "2024ApJ...900..123A",
        "title": "Title 1",
        "abstract_clean": "Text 1",
        "domain_category": "astronomy",
        "spans": [],
        "review_status": "reviewed",
        "notes": "",
    }

    record2 = {
        "bibcode": "2024ApJ...900..456B",
        "title": "Title 2",
        "abstract_clean": "Text 2",
        "domain_category": "astronomy",
        "spans": [],
        "review_status": "reviewed",
        "notes": "",
    }

    result1 = convert_to_enrichment_record(record1, "train")
    result2 = convert_to_enrichment_record(record2, "train")

    assert result1["id"] != result2["id"]


def test_convert_to_enrichment_record_deterministic_ids():
    """Test that the same record produces the same ID."""
    record = {
        "bibcode": "2024ApJ...900..123A",
        "title": "Title",
        "abstract_clean": "Text",
        "domain_category": "astronomy",
        "spans": [],
        "review_status": "reviewed",
        "notes": "",
    }

    result1 = convert_to_enrichment_record(record, "train")
    result2 = convert_to_enrichment_record(record, "train")

    assert result1["id"] == result2["id"]


# ---------------------------------------------------------------------------
# Tests: process_annotations
# ---------------------------------------------------------------------------


def test_process_annotations_basic(sample_dashboard_record, tmp_path):
    """Test basic end-to-end processing."""
    input_file = tmp_path / "input.jsonl"
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    # Write input
    with open(input_file, "w") as f:
        f.write(json.dumps(sample_dashboard_record) + "\n")

    # Process
    stats = process_annotations(input_file, train_file, val_file, train_fraction=1.0)

    # Check stats
    assert stats.total_records == 1
    assert stats.train_records == 1
    assert stats.val_records == 0
    assert stats.total_spans == 3
    assert stats.spans_by_type["topic"] == 3
    assert len(stats.validation_failures) == 0

    # Check output files
    assert train_file.exists()
    with open(train_file, "r") as f:
        train_data = [json.loads(line) for line in f]
    assert len(train_data) == 1
    assert train_data[0]["text_type"] == "abstract"


def test_process_annotations_train_val_split(
    sample_dashboard_record, sample_dashboard_record_with_entity, tmp_path
):
    """Test train/val splitting."""
    input_file = tmp_path / "input.jsonl"
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    # Write 10 records
    with open(input_file, "w") as f:
        for i in range(10):
            record = sample_dashboard_record.copy()
            record["bibcode"] = f"2024ApJ...900..{100+i:03d}A"
            f.write(json.dumps(record) + "\n")

    # Process with 70/30 split
    stats = process_annotations(input_file, train_file, val_file, train_fraction=0.7)

    assert stats.total_records == 10
    assert stats.train_records == 7
    assert stats.val_records == 3


def test_process_annotations_validation_failures(
    sample_dashboard_record_invalid_offsets, tmp_path
):
    """Test that validation failures are detected and reported."""
    input_file = tmp_path / "input.jsonl"
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    # Write input with invalid offsets
    with open(input_file, "w") as f:
        f.write(json.dumps(sample_dashboard_record_invalid_offsets) + "\n")

    # Process
    stats = process_annotations(input_file, train_file, val_file, train_fraction=1.0)

    # Should detect 1 validation failure (second span is wrong)
    assert len(stats.validation_failures) == 1
    assert stats.records_with_failures == 1
    assert stats.validation_failures[0].bibcode == "2024Nat...500..789C"
    assert stats.validation_failures[0].expected == "wrong surface"
    assert stats.validation_failures[0].actual == "test"


def test_process_annotations_empty_input(tmp_path):
    """Test handling of empty input file."""
    input_file = tmp_path / "input.jsonl"
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    # Create empty file
    input_file.touch()

    # Process
    stats = process_annotations(input_file, train_file, val_file)

    assert stats.total_records == 0


def test_process_annotations_invalid_json(tmp_path):
    """Test handling of invalid JSON lines."""
    input_file = tmp_path / "input.jsonl"
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    # Write mix of valid and invalid JSON
    with open(input_file, "w") as f:
        f.write('{"valid": "json"}\n')
        f.write('invalid json line\n')
        f.write('{"another": "valid"}\n')

    # Process (should skip invalid line and print warning)
    stats = process_annotations(input_file, train_file, val_file)

    # Should only process valid lines
    assert stats.total_records == 2


def test_process_annotations_creates_directories(sample_dashboard_record, tmp_path):
    """Test that output directories are created if they don't exist."""
    input_file = tmp_path / "input.jsonl"
    train_file = tmp_path / "nested" / "dir" / "train.jsonl"
    val_file = tmp_path / "nested" / "dir" / "val.jsonl"

    # Write input
    with open(input_file, "w") as f:
        f.write(json.dumps(sample_dashboard_record) + "\n")

    # Process (should create nested/dir/)
    stats = process_annotations(input_file, train_file, val_file)

    assert train_file.exists()
    assert val_file.exists()


def test_process_annotations_reproducible_splitting(sample_dashboard_record, tmp_path):
    """Test that splitting is reproducible with the same seed."""
    input_file = tmp_path / "input.jsonl"

    # Write 10 records
    with open(input_file, "w") as f:
        for i in range(10):
            record = sample_dashboard_record.copy()
            record["bibcode"] = f"2024ApJ...900..{100+i:03d}A"
            f.write(json.dumps(record) + "\n")

    # Process twice with same seed
    train_file1 = tmp_path / "train1.jsonl"
    val_file1 = tmp_path / "val1.jsonl"
    stats1 = process_annotations(
        input_file, train_file1, val_file1, train_fraction=0.7, seed=42
    )

    train_file2 = tmp_path / "train2.jsonl"
    val_file2 = tmp_path / "val2.jsonl"
    stats2 = process_annotations(
        input_file, train_file2, val_file2, train_fraction=0.7, seed=42
    )

    # Read output files
    with open(train_file1, "r") as f:
        train1_bibcodes = [
            json.loads(line)["provenance"]["original_bibcode"] for line in f
        ]
    with open(train_file2, "r") as f:
        train2_bibcodes = [
            json.loads(line)["provenance"]["original_bibcode"] for line in f
        ]

    # Should have same bibcodes in same order
    assert train1_bibcodes == train2_bibcodes


def test_process_annotations_stats_accuracy(
    sample_dashboard_record, sample_dashboard_record_with_entity, tmp_path
):
    """Test that stats are calculated accurately."""
    input_file = tmp_path / "input.jsonl"
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    # Write 2 records with known spans
    with open(input_file, "w") as f:
        f.write(json.dumps(sample_dashboard_record) + "\n")  # 3 spans, all topic
        f.write(
            json.dumps(sample_dashboard_record_with_entity) + "\n"
        )  # 2 spans, 1 entity + 1 topic

    # Process
    stats = process_annotations(input_file, train_file, val_file, train_fraction=1.0)

    assert stats.total_records == 2
    assert stats.total_spans == 5
    assert stats.spans_by_type["topic"] == 4
    assert stats.spans_by_type["entity"] == 1
    assert stats.spans_by_vocabulary["uat"] == 3
    assert stats.spans_by_vocabulary["ror"] == 1
    assert stats.spans_by_vocabulary["human"] == 1


# ---------------------------------------------------------------------------
# Tests: ConversionStats
# ---------------------------------------------------------------------------


def test_conversion_stats_initialization():
    """Test ConversionStats initialization."""
    stats = ConversionStats()

    assert stats.total_records == 0
    assert stats.total_spans == 0
    assert stats.spans_by_type == {}
    assert stats.spans_by_vocabulary == {}
    assert stats.spans_by_source == {}
    assert stats.validation_failures == []
    assert stats.records_with_failures == 0
    assert stats.train_records == 0
    assert stats.val_records == 0


def test_span_validation_error_dataclass():
    """Test SpanValidationError dataclass."""
    error = SpanValidationError(
        bibcode="2024ApJ...900..123A",
        span_surface="test",
        span_start=0,
        span_end=4,
        expected="test",
        actual="real",
        text_snippet="...real text...",
    )

    assert error.bibcode == "2024ApJ...900..123A"
    assert error.expected == "test"
    assert error.actual == "real"
