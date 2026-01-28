"""Tests for scripts/curate_sweet_vocabulary.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from curate_sweet_vocabulary import (
    EXTENDED_STOPWORDS,
    build_high_frequency_set,
    classify_entry,
    curate_sweet_vocabulary,
    is_borderline_scientific,
    is_single_common_english_word,
    load_sweet_catalog,
)


def _make_entry(label: str, entry_id: str = "sweet:test/Test") -> dict:
    return {
        "id": entry_id,
        "label": label,
        "aliases": [],
        "parents": [],
        "children": [],
        "source_id": "sweet",
        "domain_tags": ["earthscience"],
        "source_vocabulary": "sweet",
    }


class TestExtendedStopwords:
    def test_has_at_least_500_words(self) -> None:
        assert len(EXTENDED_STOPWORDS) >= 500

    def test_contains_nltk_stopwords(self) -> None:
        for word in ["the", "is", "and", "or", "but", "for", "with"]:
            assert word in EXTENDED_STOPWORDS

    def test_contains_extended_words(self) -> None:
        for word in ["however", "therefore", "current", "general", "result"]:
            assert word in EXTENDED_STOPWORDS


class TestLoadSweetCatalog:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        entries = [_make_entry("alpha"), _make_entry("beta")]
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = load_sweet_catalog(path)
        assert len(result) == 2
        assert result[0]["label"] == "alpha"
        assert result[1]["label"] == "beta"

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        entry = _make_entry("gamma")
        path.write_text(json.dumps(entry) + "\n\n\n")
        result = load_sweet_catalog(path)
        assert len(result) == 1


class TestBuildHighFrequencySet:
    def test_returns_frozenset(self) -> None:
        result = build_high_frequency_set(100)
        assert isinstance(result, frozenset)

    def test_contains_common_words(self) -> None:
        result = build_high_frequency_set(5000)
        for word in ["the", "and", "water", "animal"]:
            assert word in result

    def test_correct_size(self) -> None:
        result = build_high_frequency_set(100)
        assert len(result) == 100


class TestIsSingleCommonEnglishWord:
    def test_common_words_detected(self) -> None:
        for word in ["reservoir", "delta", "variance", "difference"]:
            assert is_single_common_english_word(word), f"{word} should be common"

    def test_scientific_terms_not_detected(self) -> None:
        for word in ["photoionization", "sandur", "esker"]:
            assert not is_single_common_english_word(word), f"{word} should not be common"

    def test_multi_word_returns_false(self) -> None:
        assert not is_single_common_english_word("mountain breeze")
        assert not is_single_common_english_word("trace metal")


class TestClassifyEntry:
    @pytest.fixture()
    def high_freq(self) -> frozenset[str]:
        return build_high_frequency_set(5000)

    def test_short_label_removed(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("a")
        reason, category = classify_entry(entry, high_freq)
        assert reason == "fewer_than_4_characters"
        assert category == "short"

    def test_three_char_label_removed(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("ice")
        reason, category = classify_entry(entry, high_freq)
        assert reason == "fewer_than_4_characters"
        assert category == "short"

    def test_stopword_removed(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("however")
        reason, category = classify_entry(entry, high_freq)
        assert reason == "in_stopword_list"
        assert category == "stopword"

    def test_high_frequency_removed(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("water")
        reason, category = classify_entry(entry, high_freq)
        assert reason is not None  # Could be stopword or high_frequency
        assert category in ("stopword", "high_frequency")

    def test_common_english_removed(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("reservoir")
        reason, category = classify_entry(entry, high_freq)
        assert reason is not None
        assert category in ("high_frequency", "common_english")

    def test_scientific_term_kept(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("photoionization")
        reason, category = classify_entry(entry, high_freq)
        assert reason is None
        assert category == "kept"

    def test_mineral_name_kept(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("Hypercinnabar")
        reason, category = classify_entry(entry, high_freq)
        assert reason is None
        assert category == "kept"

    def test_multi_word_scientific_kept(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("mountain breeze")
        reason, category = classify_entry(entry, high_freq)
        assert reason is None
        assert category == "kept"

    def test_case_insensitive_matching(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("The")
        reason, _ = classify_entry(entry, high_freq)
        assert reason == "fewer_than_4_characters"

    def test_whitespace_stripped(self, high_freq: frozenset[str]) -> None:
        entry = _make_entry("  ice  ")
        reason, category = classify_entry(entry, high_freq)
        assert reason == "fewer_than_4_characters"
        assert category == "short"


class TestIsBorderlineScientific:
    def test_four_char_label(self) -> None:
        assert is_borderline_scientific(_make_entry("silt"))

    def test_six_char_label(self) -> None:
        assert is_borderline_scientific(_make_entry("farad1"))

    def test_three_char_not_borderline(self) -> None:
        assert not is_borderline_scientific(_make_entry("ice"))

    def test_seven_char_not_borderline(self) -> None:
        assert not is_borderline_scientific(_make_entry("sandstone"))


class TestCurateSweetVocabulary:
    def test_full_pipeline(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.jsonl"
        output_dir = tmp_path / "output"

        entries = [
            _make_entry("a", "sweet:test/Short"),              # short: removed
            _make_entry("however", "sweet:test/Stopword"),     # stopword: removed
            _make_entry("water", "sweet:test/HighFreq"),       # high freq: removed
            _make_entry("reservoir", "sweet:test/Common"),     # common: removed
            _make_entry("photoionization", "sweet:test/Sci"),  # scientific: kept
            _make_entry("Hypercinnabar", "sweet:test/Min"),    # mineral: kept
            _make_entry("mountain breeze", "sweet:test/MW"),   # multi-word: kept
            _make_entry("silt", "sweet:test/Borderline"),      # borderline kept 4-char
        ]
        input_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        report = curate_sweet_vocabulary(
            input_path=input_path, output_dir=output_dir
        )

        assert report["summary"]["total_input_entries"] == 8
        assert report["summary"]["curated_entries"] >= 3
        assert report["summary"]["removed_entries"] >= 3

        # Verify output files exist
        assert (output_dir / "sweet_curated.jsonl").exists()
        assert (output_dir / "sweet_removed.jsonl").exists()
        assert (output_dir / "sweet_borderline.jsonl").exists()
        assert (output_dir / "sweet_curation_report.json").exists()

        # Verify curated entries
        curated = []
        with open(output_dir / "sweet_curated.jsonl") as f:
            for line in f:
                curated.append(json.loads(line))
        curated_ids = {e["id"] for e in curated}
        assert "sweet:test/Sci" in curated_ids
        assert "sweet:test/Min" in curated_ids
        assert "sweet:test/MW" in curated_ids

        # Verify removed entries have reason
        removed = []
        with open(output_dir / "sweet_removed.jsonl") as f:
            for line in f:
                removed.append(json.loads(line))
        assert all("removal_reason" in e for e in removed)

        # Verify report JSON is valid
        with open(output_dir / "sweet_curation_report.json") as f:
            saved_report = json.load(f)
        assert saved_report["summary"]["total_input_entries"] == 8

    def test_empty_input(self, tmp_path: Path) -> None:
        input_path = tmp_path / "empty.jsonl"
        output_dir = tmp_path / "output"
        input_path.write_text("")
        report = curate_sweet_vocabulary(
            input_path=input_path, output_dir=output_dir
        )
        assert report["summary"]["total_input_entries"] == 0
        assert report["summary"]["curated_entries"] == 0
        assert report["summary"]["removed_entries"] == 0

    def test_all_scientific_terms_kept(self, tmp_path: Path) -> None:
        input_path = tmp_path / "scientific.jsonl"
        output_dir = tmp_path / "output"
        entries = [
            _make_entry("photoionization"),
            _make_entry("Hypercinnabar"),
            _make_entry("stratocumulus"),
        ]
        input_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        report = curate_sweet_vocabulary(
            input_path=input_path, output_dir=output_dir
        )
        assert report["summary"]["curated_entries"] == 3
        assert report["summary"]["removed_entries"] == 0

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.jsonl"
        output_dir = tmp_path / "nested" / "output"
        input_path.write_text(json.dumps(_make_entry("photoionization")) + "\n")
        curate_sweet_vocabulary(input_path=input_path, output_dir=output_dir)
        assert output_dir.exists()
