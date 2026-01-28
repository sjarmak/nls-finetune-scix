"""Unit tests for scripts/predict_ads_abstracts.py — NER prediction on ADS abstracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from predict_ads_abstracts import (
    BIO_LABELS,
    BIO_TO_SPAN_TYPE,
    ID2LABEL,
    LABEL2ID,
    PredictedSpan,
    load_records,
    predict_spans_for_text,
)


# ---------------------------------------------------------------------------
# BIO label schema
# ---------------------------------------------------------------------------


class TestBioLabelSchema:
    def test_nine_labels(self) -> None:
        assert len(BIO_LABELS) == 9

    def test_o_tag_is_first(self) -> None:
        assert BIO_LABELS[0] == "O"
        assert LABEL2ID["O"] == 0

    def test_b_topic_present(self) -> None:
        assert "B-topic" in LABEL2ID
        assert "I-topic" in LABEL2ID

    def test_b_institution_present(self) -> None:
        assert "B-institution" in LABEL2ID
        assert "I-institution" in LABEL2ID

    def test_b_author_present(self) -> None:
        assert "B-author" in LABEL2ID
        assert "I-author" in LABEL2ID

    def test_b_date_range_present(self) -> None:
        assert "B-date_range" in LABEL2ID
        assert "I-date_range" in LABEL2ID

    def test_id2label_roundtrip(self) -> None:
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_bio_to_span_type_mapping(self) -> None:
        assert BIO_TO_SPAN_TYPE["institution"] == "entity"


# ---------------------------------------------------------------------------
# PredictedSpan
# ---------------------------------------------------------------------------


class TestPredictedSpan:
    def test_create_span(self) -> None:
        span = PredictedSpan(
            surface="dark matter",
            start=0,
            end=11,
            type="topic",
            confidence=0.95,
        )
        assert span.surface == "dark matter"
        assert span.start == 0
        assert span.end == 11
        assert span.type == "topic"
        assert span.confidence == 0.95

    def test_to_dict(self) -> None:
        span = PredictedSpan(
            surface="Harvard",
            start=10,
            end=17,
            type="entity",
            confidence=0.88,
        )
        d = span.to_dict()
        assert d["surface"] == "Harvard"
        assert d["start"] == 10
        assert d["end"] == 17
        assert d["type"] == "entity"
        assert d["confidence"] == 0.88

    def test_frozen(self) -> None:
        span = PredictedSpan(
            surface="test",
            start=0,
            end=4,
            type="topic",
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            span.surface = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_records
# ---------------------------------------------------------------------------


class TestLoadRecords:
    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        data = [
            {"bibcode": "ABC", "title": "Test1", "abstract_clean": "Hello world"},
            {"bibcode": "DEF", "title": "Test2", "abstract_clean": "Foo bar"},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")

        records = load_records(path)
        assert len(records) == 2
        assert records[0]["bibcode"] == "ABC"
        assert records[1]["bibcode"] == "DEF"

    def test_skip_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write('{"bibcode": "A"}\n\n\n{"bibcode": "B"}\n')

        records = load_records(path)
        assert len(records) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        records = load_records(path)
        assert len(records) == 0


# ---------------------------------------------------------------------------
# predict_spans_for_text — mock-based tests
# ---------------------------------------------------------------------------


def _make_mock_model_and_tokenizer(
    text: str,
    tag_ids: list[int],
    offsets: list[tuple[int, int]],
    logits_raw: list[list[float]] | None = None,
) -> tuple[Any, Any]:
    """Create mock tokenizer and model that return deterministic outputs.

    Args:
        text: The input text.
        tag_ids: Predicted tag IDs per token (including special tokens).
        offsets: Offset mapping per token.
        logits_raw: Optional raw logit values per token. If None, uses
                     high confidence (10.0) for predicted tag.
    """
    import torch

    seq_len = len(tag_ids)
    num_labels = len(BIO_LABELS)

    # Build logits tensor
    if logits_raw is not None:
        logits = torch.tensor([logits_raw], dtype=torch.float32)
    else:
        # Default: high logit for predicted tag, low for others
        logits_data = []
        for tid in tag_ids:
            row = [-5.0] * num_labels
            row[tid] = 10.0
            logits_data.append(row)
        logits = torch.tensor([logits_data], dtype=torch.float32)

    # Mock tokenizer
    tokenizer = MagicMock()
    encoding_dict: dict[str, Any] = {
        "input_ids": torch.tensor([[0] * seq_len]),
        "attention_mask": torch.tensor([[1] * seq_len]),
        "offset_mapping": torch.tensor([offsets]),
    }

    class EncodingProxy(dict):
        def pop(self, key: str, *args: Any) -> Any:
            return dict.pop(self, key, *args)

    encoding = EncodingProxy(encoding_dict)
    tokenizer.return_value = encoding

    # Mock model
    model = MagicMock()
    output = MagicMock()
    output.logits = logits
    model.return_value = output

    return tokenizer, model


class TestPredictSpansForText:
    def test_single_b_topic_span(self) -> None:
        text = "dark matter is cool"
        # Tokens: [CLS], dark, matter, is, cool, [SEP], [PAD]...
        tag_ids = [0, LABEL2ID["B-topic"], LABEL2ID["I-topic"], 0, 0, 0]
        offsets = [(0, 0), (0, 4), (5, 11), (12, 14), (15, 19), (0, 0)]

        tokenizer, model = _make_mock_model_and_tokenizer(text, tag_ids, offsets)
        spans = predict_spans_for_text(text, tokenizer, model)

        assert len(spans) == 1
        assert spans[0].surface == "dark matter"
        assert spans[0].start == 0
        assert spans[0].end == 11
        assert spans[0].type == "topic"
        assert spans[0].confidence > 0.99  # very high logit

    def test_institution_maps_to_entity(self) -> None:
        text = "Harvard University"
        tag_ids = [0, LABEL2ID["B-institution"], LABEL2ID["I-institution"], 0]
        offsets = [(0, 0), (0, 7), (8, 18), (0, 0)]

        tokenizer, model = _make_mock_model_and_tokenizer(text, tag_ids, offsets)
        spans = predict_spans_for_text(text, tokenizer, model)

        assert len(spans) == 1
        assert spans[0].type == "entity"
        assert spans[0].surface == "Harvard University"

    def test_multiple_spans(self) -> None:
        text = "dark matter near Gale Crater"
        # Two spans: topic + entity
        tag_ids = [
            0,
            LABEL2ID["B-topic"],
            LABEL2ID["I-topic"],
            0,
            LABEL2ID["B-institution"],
            LABEL2ID["I-institution"],
            0,
        ]
        offsets = [
            (0, 0),
            (0, 4),
            (5, 11),
            (12, 16),
            (17, 21),
            (22, 28),
            (0, 0),
        ]

        tokenizer, model = _make_mock_model_and_tokenizer(text, tag_ids, offsets)
        spans = predict_spans_for_text(text, tokenizer, model)

        assert len(spans) == 2
        assert spans[0].surface == "dark matter"
        assert spans[0].type == "topic"
        assert spans[1].surface == "Gale Crater"
        assert spans[1].type == "entity"

    def test_no_entities_all_o(self) -> None:
        text = "just plain text"
        tag_ids = [0, 0, 0, 0, 0]
        offsets = [(0, 0), (0, 4), (5, 10), (11, 15), (0, 0)]

        tokenizer, model = _make_mock_model_and_tokenizer(text, tag_ids, offsets)
        spans = predict_spans_for_text(text, tokenizer, model)

        assert len(spans) == 0

    def test_consecutive_b_tags_create_separate_spans(self) -> None:
        text = "alpha beta gamma"
        # Three consecutive B-topic tags (no I- continuation)
        tag_ids = [
            0,
            LABEL2ID["B-topic"],
            LABEL2ID["B-topic"],
            LABEL2ID["B-topic"],
            0,
        ]
        offsets = [(0, 0), (0, 5), (6, 10), (11, 16), (0, 0)]

        tokenizer, model = _make_mock_model_and_tokenizer(text, tag_ids, offsets)
        spans = predict_spans_for_text(text, tokenizer, model)

        assert len(spans) == 3
        assert spans[0].surface == "alpha"
        assert spans[1].surface == "beta"
        assert spans[2].surface == "gamma"

    def test_type_mismatch_closes_span(self) -> None:
        text = "dark matter near Gale"
        # B-topic followed by I-institution (mismatch)
        tag_ids = [
            0,
            LABEL2ID["B-topic"],
            LABEL2ID["I-institution"],
            0,
            0,
            0,
        ]
        offsets = [(0, 0), (0, 4), (5, 11), (12, 16), (17, 21), (0, 0)]

        tokenizer, model = _make_mock_model_and_tokenizer(text, tag_ids, offsets)
        spans = predict_spans_for_text(text, tokenizer, model)

        # B-topic at "dark" closed by mismatched I-institution
        # I-institution without matching B- is treated as O
        assert len(spans) == 1
        assert spans[0].surface == "dark"
        assert spans[0].type == "topic"

    def test_span_at_end_of_sequence(self) -> None:
        text = "cosmic rays"
        # Span extends to end of real tokens
        tag_ids = [0, LABEL2ID["B-topic"], LABEL2ID["I-topic"], 0]
        offsets = [(0, 0), (0, 6), (7, 11), (0, 0)]

        tokenizer, model = _make_mock_model_and_tokenizer(text, tag_ids, offsets)
        spans = predict_spans_for_text(text, tokenizer, model)

        assert len(spans) == 1
        assert spans[0].surface == "cosmic rays"

    def test_confidence_reflects_softmax(self) -> None:
        """Confidence should be the softmax probability of the B- tag."""
        import torch

        text = "test"
        b_topic_id = LABEL2ID["B-topic"]
        tag_ids = [0, b_topic_id, 0]
        offsets = [(0, 0), (0, 4), (0, 0)]

        # Build logits where B-topic has logit=2.0, others=0.0
        num_labels = len(BIO_LABELS)
        logits_raw = [
            [0.0] * num_labels,  # CLS
            [0.0] * num_labels,  # "test" token
            [0.0] * num_labels,  # SEP
        ]
        logits_raw[1][b_topic_id] = 2.0

        # Expected softmax probability
        logits_tensor = torch.tensor(logits_raw[1], dtype=torch.float32)
        expected_conf = torch.softmax(logits_tensor, dim=0)[b_topic_id].item()

        tokenizer, model = _make_mock_model_and_tokenizer(
            text, tag_ids, offsets, logits_raw
        )
        spans = predict_spans_for_text(text, tokenizer, model)

        assert len(spans) == 1
        assert abs(spans[0].confidence - round(expected_conf, 4)) < 0.001

    def test_date_range_span(self) -> None:
        text = "from 2020 to 2023"
        tag_ids = [
            0,
            0,
            LABEL2ID["B-date_range"],
            LABEL2ID["I-date_range"],
            LABEL2ID["I-date_range"],
            0,
        ]
        offsets = [(0, 0), (0, 4), (5, 9), (10, 12), (13, 17), (0, 0)]

        tokenizer, model = _make_mock_model_and_tokenizer(text, tag_ids, offsets)
        spans = predict_spans_for_text(text, tokenizer, model)

        assert len(spans) == 1
        assert spans[0].surface == "2020 to 2023"
        assert spans[0].type == "date_range"

    def test_author_span(self) -> None:
        text = "by John Smith"
        tag_ids = [
            0,
            0,
            LABEL2ID["B-author"],
            LABEL2ID["I-author"],
            0,
        ]
        offsets = [(0, 0), (0, 2), (3, 7), (8, 13), (0, 0)]

        tokenizer, model = _make_mock_model_and_tokenizer(text, tag_ids, offsets)
        spans = predict_spans_for_text(text, tokenizer, model)

        assert len(spans) == 1
        assert spans[0].surface == "John Smith"
        assert spans[0].type == "author"
