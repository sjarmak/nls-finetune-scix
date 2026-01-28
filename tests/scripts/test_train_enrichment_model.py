"""Tests for scripts/train_enrichment_model.py curriculum learning functionality."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from train_enrichment_model import (  # noqa: E402
    TrainConfig,
    load_enrichment_records,
    train_curriculum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_enrichment_records() -> list[dict[str, Any]]:
    """Sample enrichment records for testing."""
    return [
        {
            "id": "enr_001",
            "text": "Mars rover found water ice.",
            "text_type": "title",
            "spans": [
                {"surface": "Mars", "start": 0, "end": 4, "type": "topic"},
                {"surface": "water ice", "start": 17, "end": 26, "type": "topic"},
            ],
        },
        {
            "id": "enr_002",
            "text": "The James Webb Space Telescope observes distant galaxies.",
            "text_type": "abstract",
            "spans": [
                {"surface": "James Webb Space Telescope", "start": 4, "end": 30, "type": "entity"},
                {"surface": "galaxies", "start": 49, "end": 57, "type": "topic"},
            ],
        },
    ]


@pytest.fixture
def sample_human_records() -> list[dict[str, Any]]:
    """Sample human-annotated records for testing."""
    return [
        {
            "id": "enr_hum_001",
            "text": "Exoplanet atmospheres reveal chemical composition.",
            "text_type": "title",
            "spans": [
                {"surface": "Exoplanet", "start": 0, "end": 9, "type": "topic"},
                {"surface": "atmospheres", "start": 10, "end": 21, "type": "topic"},
                {"surface": "chemical composition", "start": 29, "end": 49, "type": "topic"},
            ],
        },
    ]


@pytest.fixture
def temp_data_files(
    sample_enrichment_records: list[dict[str, Any]],
    sample_human_records: list[dict[str, Any]],
) -> dict[str, Path]:
    """Create temporary JSONL files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Synthetic data files
        synthetic_train = tmp_path / "synthetic_train.jsonl"
        synthetic_val = tmp_path / "synthetic_val.jsonl"

        with open(synthetic_train, "w", encoding="utf-8") as f:
            for record in sample_enrichment_records:
                f.write(json.dumps(record) + "\n")

        with open(synthetic_val, "w", encoding="utf-8") as f:
            for record in sample_enrichment_records[:1]:
                f.write(json.dumps(record) + "\n")

        # Human annotation files
        human_train = tmp_path / "human_train.jsonl"
        human_val = tmp_path / "human_val.jsonl"

        with open(human_train, "w", encoding="utf-8") as f:
            for record in sample_human_records:
                f.write(json.dumps(record) + "\n")

        with open(human_val, "w", encoding="utf-8") as f:
            for record in sample_human_records[:1]:
                f.write(json.dumps(record) + "\n")

        yield {
            "synthetic_train": synthetic_train,
            "synthetic_val": synthetic_val,
            "human_train": human_train,
            "human_val": human_val,
            "output_dir": tmp_path / "output",
        }


# ---------------------------------------------------------------------------
# Tests for load_enrichment_records
# ---------------------------------------------------------------------------


def test_load_enrichment_records(temp_data_files: dict[str, Path]) -> None:
    """Test loading enrichment records from JSONL."""
    records = load_enrichment_records(temp_data_files["synthetic_train"])
    assert len(records) == 2
    assert records[0]["id"] == "enr_001"
    assert records[0]["text"] == "Mars rover found water ice."
    assert len(records[0]["spans"]) == 2


def test_load_enrichment_records_empty_lines(temp_data_files: dict[str, Path]) -> None:
    """Test that empty lines are skipped."""
    tmpfile = temp_data_files["synthetic_train"].parent / "empty_lines.jsonl"
    with open(tmpfile, "w", encoding="utf-8") as f:
        f.write('{"id": "enr_001", "text": "Test"}\n')
        f.write("\n")
        f.write("\n")
        f.write('{"id": "enr_002", "text": "Test2"}\n')

    records = load_enrichment_records(tmpfile)
    assert len(records) == 2


# ---------------------------------------------------------------------------
# Tests for curriculum learning configuration
# ---------------------------------------------------------------------------


def test_train_config_default_values() -> None:
    """Test TrainConfig default values."""
    config = TrainConfig()
    assert config.curriculum is False
    assert config.human_train_file is None
    assert config.human_val_file is None
    assert config.phase1_epochs == 3
    assert config.phase2_epochs == 5
    assert config.phase2_learning_rate == 5e-6


def test_train_config_curriculum_values() -> None:
    """Test TrainConfig with curriculum learning parameters."""
    config = TrainConfig(
        curriculum=True,
        human_train_file="data/human_train.jsonl",
        human_val_file="data/human_val.jsonl",
        phase1_epochs=2,
        phase2_epochs=4,
        phase2_learning_rate=1e-6,
    )
    assert config.curriculum is True
    assert config.human_train_file == "data/human_train.jsonl"
    assert config.human_val_file == "data/human_val.jsonl"
    assert config.phase1_epochs == 2
    assert config.phase2_epochs == 4
    assert config.phase2_learning_rate == 1e-6


# ---------------------------------------------------------------------------
# Tests for curriculum learning training
# ---------------------------------------------------------------------------


def test_train_curriculum_has_human_data_check(
    temp_data_files: dict[str, Path],
) -> None:
    """Test that curriculum learning checks for human data existence."""
    # Test with existing human data files
    config = TrainConfig(
        curriculum=True,
        train_file=str(temp_data_files["synthetic_train"]),
        val_file=str(temp_data_files["synthetic_val"]),
        human_train_file=str(temp_data_files["human_train"]),
        human_val_file=str(temp_data_files["human_val"]),
        output_dir=str(temp_data_files["output_dir"]),
        phase1_epochs=2,
        phase2_epochs=2,
    )

    # Verify human data files exist
    assert Path(config.human_train_file).exists()
    assert Path(config.human_val_file).exists()

    # Test with missing human data files - should fall back to regular training
    config_no_human = TrainConfig(
        curriculum=True,
        train_file=str(temp_data_files["synthetic_train"]),
        val_file=str(temp_data_files["synthetic_val"]),
        human_train_file="nonexistent.jsonl",
        human_val_file="nonexistent.jsonl",
        output_dir=str(temp_data_files["output_dir"]),
    )

    # Verify human data files don't exist
    assert not Path(config_no_human.human_train_file).exists()
    assert not Path(config_no_human.human_val_file).exists()


@patch("train_enrichment_model.train")
def test_train_curriculum_fallback_missing_human_data(
    mock_train: MagicMock,
    temp_data_files: dict[str, Path],
) -> None:
    """Test that curriculum learning falls back to regular training when human data is missing."""
    mock_train.return_value = 0

    config = TrainConfig(
        curriculum=True,
        train_file=str(temp_data_files["synthetic_train"]),
        val_file=str(temp_data_files["synthetic_val"]),
        human_train_file="nonexistent.jsonl",
        human_val_file="nonexistent.jsonl",
        output_dir=str(temp_data_files["output_dir"]),
    )

    result = train_curriculum(config)

    assert result == 0
    # Should fall back to regular train()
    mock_train.assert_called_once_with(config)


@patch("train_enrichment_model.train")
def test_train_curriculum_fallback_no_human_files_specified(
    mock_train: MagicMock,
    temp_data_files: dict[str, Path],
) -> None:
    """Test fallback when human files are not specified at all."""
    mock_train.return_value = 0

    config = TrainConfig(
        curriculum=True,
        train_file=str(temp_data_files["synthetic_train"]),
        val_file=str(temp_data_files["synthetic_val"]),
        human_train_file=None,
        human_val_file=None,
        output_dir=str(temp_data_files["output_dir"]),
    )

    result = train_curriculum(config)

    assert result == 0
    mock_train.assert_called_once_with(config)


def test_curriculum_learning_phase_configuration(temp_data_files: dict[str, Path]) -> None:
    """Test that curriculum learning phase parameters are correctly configured."""
    config = TrainConfig(
        curriculum=True,
        train_file=str(temp_data_files["synthetic_train"]),
        val_file=str(temp_data_files["synthetic_val"]),
        human_train_file=str(temp_data_files["human_train"]),
        human_val_file=str(temp_data_files["human_val"]),
        output_dir=str(temp_data_files["output_dir"]),
        phase1_epochs=2,
        phase2_epochs=3,
        phase2_learning_rate=1e-6,
        learning_rate=2e-5,
    )

    # Verify phase 1 settings
    assert config.phase1_epochs == 2
    assert config.learning_rate == 2e-5

    # Verify phase 2 settings
    assert config.phase2_epochs == 3
    assert config.phase2_learning_rate == 1e-6

    # Verify phase 2 has lower learning rate than phase 1
    assert config.phase2_learning_rate < config.learning_rate


def test_train_curriculum_missing_synthetic_train_file() -> None:
    """Test that curriculum learning fails if synthetic train file is missing."""
    config = TrainConfig(
        curriculum=True,
        train_file="nonexistent_train.jsonl",
        val_file="nonexistent_val.jsonl",
        human_train_file="nonexistent_human_train.jsonl",
        human_val_file="nonexistent_human_val.jsonl",
        output_dir="output",
    )

    result = train_curriculum(config)
    assert result == 1


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
