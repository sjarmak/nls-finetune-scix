"""Tests for entity_linker module: exact match, fuzzy match, type filtering, persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from finetune.dataset_agent.entity_linker import (
    LinkResult,
    LinkingIndex,
    build_linking_index,
    link_span,
    load_linking_index,
    save_linking_index,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@pytest.fixture()
def topic_catalog(tmp_path: Path) -> Path:
    """Create a small topic catalog JSONL file."""
    records = [
        {
            "id": "uat:123",
            "label": "Exoplanets",
            "aliases": ["Extra-solar planets", "Exoplanet"],
            "parents": [],
            "children": [],
            "source_vocabulary": "uat",
            "domain_tags": ["astronomy"],
        },
        {
            "id": "sweet:456",
            "label": "Hydrosphere",
            "aliases": ["Water cycle"],
            "parents": [],
            "children": [],
            "source_vocabulary": "sweet",
            "domain_tags": ["earthscience"],
        },
        {
            "id": "gcmd:789",
            "label": "Atmospheric Chemistry",
            "aliases": ["Atmos Chemistry"],
            "parents": [],
            "children": [],
            "source_vocabulary": "gcmd",
            "domain_tags": ["earthscience"],
        },
    ]
    path = tmp_path / "topic_catalog.jsonl"
    _write_jsonl(path, records)
    return path


@pytest.fixture()
def entity_catalog(tmp_path: Path) -> Path:
    """Create a small entity catalog JSONL file."""
    records = [
        {
            "id": "ror:04nt9b154",
            "label": "Hubble Space Telescope",
            "aliases": ["HST"],
            "metadata": {},
            "source_vocabulary": "ror",
            "domain_tags": ["multidisciplinary"],
            "entity_subtype": "observatory",
        },
        {
            "id": "planetary:Mars/1234",
            "label": "Olympus Mons",
            "aliases": ["Nix Olympica"],
            "metadata": {},
            "source_vocabulary": "planetary",
            "domain_tags": ["planetary"],
            "entity_subtype": "mons",
        },
    ]
    path = tmp_path / "entity_catalog.jsonl"
    _write_jsonl(path, records)
    return path


@pytest.fixture()
def index(topic_catalog: Path, entity_catalog: Path) -> LinkingIndex:
    """Build a linking index from the test catalogs."""
    return build_linking_index(topic_catalog, entity_catalog)


# ---------------------------------------------------------------------------
# Exact match tests
# ---------------------------------------------------------------------------

class TestExactMatch:
    """Exact match tests: case-insensitive, labels, aliases."""

    def test_exact_match_topic_label(self, index: LinkingIndex) -> None:
        """Exact match on a topic label returns confidence=1.0."""
        result = link_span("Exoplanets", "topic", index)
        assert result is not None
        assert result.canonical_id == "uat:123"
        assert result.source_vocabulary == "uat"
        assert result.confidence == 1.0
        assert result.match_type == "exact"

    def test_exact_match_case_insensitive(self, index: LinkingIndex) -> None:
        """Exact match is case-insensitive."""
        result = link_span("exoplanets", "topic", index)
        assert result is not None
        assert result.canonical_id == "uat:123"
        assert result.confidence == 1.0

    def test_exact_match_mixed_case(self, index: LinkingIndex) -> None:
        """Mixed case still matches."""
        result = link_span("EXOPLANETS", "topic", index)
        assert result is not None
        assert result.canonical_id == "uat:123"

    def test_exact_match_alias(self, index: LinkingIndex) -> None:
        """Alias lookup returns exact match."""
        result = link_span("Extra-solar planets", "topic", index)
        assert result is not None
        assert result.canonical_id == "uat:123"
        assert result.confidence == 1.0
        assert result.match_type == "exact"

    def test_exact_match_alias_case_insensitive(self, index: LinkingIndex) -> None:
        """Alias lookup is also case-insensitive."""
        result = link_span("hst", "entity", index)
        assert result is not None
        assert result.canonical_id == "ror:04nt9b154"

    def test_exact_match_entity_label(self, index: LinkingIndex) -> None:
        """Exact match on an entity label."""
        result = link_span("Olympus Mons", "entity", index)
        assert result is not None
        assert result.canonical_id == "planetary:Mars/1234"
        assert result.source_vocabulary == "planetary"
        assert result.confidence == 1.0

    def test_exact_match_entity_alias(self, index: LinkingIndex) -> None:
        """Entity alias resolves to correct entry."""
        result = link_span("Nix Olympica", "entity", index)
        assert result is not None
        assert result.canonical_id == "planetary:Mars/1234"


# ---------------------------------------------------------------------------
# Type filtering tests
# ---------------------------------------------------------------------------

class TestTypeFiltering:
    """Topics should only match topic catalog, entities only match entity catalog."""

    def test_topic_surface_not_matched_as_entity(self, index: LinkingIndex) -> None:
        """A topic surface form should NOT match when span_type is 'entity'."""
        result = link_span("Exoplanets", "entity", index)
        assert result is None

    def test_entity_surface_not_matched_as_topic(self, index: LinkingIndex) -> None:
        """An entity surface form should NOT match when span_type is 'topic'."""
        result = link_span("Hubble Space Telescope", "topic", index)
        assert result is None

    def test_alias_type_filtering(self, index: LinkingIndex) -> None:
        """Alias filtering respects type as well."""
        result = link_span("HST", "topic", index)
        assert result is None

    def test_entity_alias_type_filtering(self, index: LinkingIndex) -> None:
        """Entity alias should not match as topic."""
        result = link_span("Nix Olympica", "topic", index)
        assert result is None


# ---------------------------------------------------------------------------
# Fuzzy match tests
# ---------------------------------------------------------------------------

class TestFuzzyMatch:
    """Fuzzy matching via Levenshtein distance."""

    def test_fuzzy_match_small_typo(self, index: LinkingIndex) -> None:
        """Small typo in surface should fuzzy-match."""
        result = link_span("Exoplanet", "topic", index)
        # "Exoplanet" is also an alias, so this should be exact
        assert result is not None
        assert result.canonical_id == "uat:123"

    def test_fuzzy_match_minor_misspelling(self, index: LinkingIndex) -> None:
        """Minor misspelling should fuzzy-match if similarity >= 0.85."""
        result = link_span("Olympus Monss", "entity", index)
        assert result is not None
        assert result.canonical_id == "planetary:Mars/1234"
        assert result.match_type == "fuzzy"
        assert result.confidence == 0.8

    def test_fuzzy_match_respects_threshold(self, index: LinkingIndex) -> None:
        """Very different string should not fuzzy-match."""
        result = link_span("Completely different text", "topic", index)
        assert result is None

    def test_fuzzy_match_type_filtering(self, index: LinkingIndex) -> None:
        """Fuzzy match should respect type filtering."""
        # "Olympus Monss" is close to "Olympus Mons" (entity), should not match as topic
        result = link_span("Olympus Monss", "topic", index)
        assert result is None

    def test_fuzzy_match_custom_threshold(self, index: LinkingIndex) -> None:
        """Custom threshold should be respected."""
        # With very high threshold, slight misspellings won't match
        result = link_span("Olympus Monss", "entity", index, fuzzy_threshold=0.99)
        assert result is None


# ---------------------------------------------------------------------------
# No match tests
# ---------------------------------------------------------------------------

class TestNoMatch:
    """Cases where no match should be found."""

    def test_no_match_unknown_surface(self, index: LinkingIndex) -> None:
        """Unknown surface returns None."""
        result = link_span("Completely Unknown Concept", "topic", index)
        assert result is None

    def test_no_match_empty_surface(self, index: LinkingIndex) -> None:
        """Empty string returns None."""
        result = link_span("", "topic", index)
        assert result is None

    def test_no_match_wrong_type(self, index: LinkingIndex) -> None:
        """Surface exists in catalog but wrong type returns None."""
        result = link_span("Exoplanets", "entity", index)
        assert result is None


# ---------------------------------------------------------------------------
# Index building tests
# ---------------------------------------------------------------------------

class TestBuildIndex:
    """Index building from catalogs."""

    def test_build_from_catalogs(self, topic_catalog: Path, entity_catalog: Path) -> None:
        """Index builds successfully from real catalog files."""
        idx = build_linking_index(topic_catalog, entity_catalog)
        assert len(idx.exact_map) > 0
        assert len(idx.entries) > 0

    def test_build_missing_topic_catalog(self, tmp_path: Path, entity_catalog: Path) -> None:
        """Missing topic catalog still builds (entities only)."""
        idx = build_linking_index(tmp_path / "nonexistent.jsonl", entity_catalog)
        assert len(idx.exact_map) > 0
        # Should contain entity entries only
        result = link_span("Olympus Mons", "entity", idx)
        assert result is not None

    def test_build_missing_entity_catalog(self, topic_catalog: Path, tmp_path: Path) -> None:
        """Missing entity catalog still builds (topics only)."""
        idx = build_linking_index(topic_catalog, tmp_path / "nonexistent.jsonl")
        assert len(idx.exact_map) > 0
        result = link_span("Exoplanets", "topic", idx)
        assert result is not None

    def test_build_both_missing(self, tmp_path: Path) -> None:
        """Both catalogs missing builds an empty index."""
        idx = build_linking_index(
            tmp_path / "no_topics.jsonl",
            tmp_path / "no_entities.jsonl",
        )
        assert len(idx.exact_map) == 0
        assert len(idx.entries) == 0

    def test_index_contains_aliases(self, index: LinkingIndex) -> None:
        """Index should contain entries for aliases, not just labels."""
        assert "extra-solar planets" in index.exact_map
        assert "hst" in index.exact_map
        assert "nix olympica" in index.exact_map

    def test_index_entry_count(self, index: LinkingIndex) -> None:
        """Index should have entries for all surface forms."""
        # Topics: Exoplanets(1+2aliases=3), Hydrosphere(1+1=2), Atmospheric Chemistry(1+1=2) = 7
        # Entities: Hubble Space Telescope(1+1=2), Olympus Mons(1+1=2) = 4
        # Total = 11
        assert len(index.entries) == 11


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    """Index serialization and deserialization."""

    def test_save_and_load(self, index: LinkingIndex, tmp_path: Path) -> None:
        """Save and reload produces functionally identical index."""
        path = tmp_path / "index.json"
        save_linking_index(index, path)
        loaded = load_linking_index(path)

        assert len(loaded.exact_map) == len(index.exact_map)
        assert len(loaded.entries) == len(index.entries)

    def test_loaded_index_works(self, index: LinkingIndex, tmp_path: Path) -> None:
        """A loaded index can be used for linking."""
        path = tmp_path / "index.json"
        save_linking_index(index, path)
        loaded = load_linking_index(path)

        result = link_span("Exoplanets", "topic", loaded)
        assert result is not None
        assert result.canonical_id == "uat:123"

    def test_loaded_index_fuzzy_works(self, index: LinkingIndex, tmp_path: Path) -> None:
        """A loaded index supports fuzzy matching."""
        path = tmp_path / "index.json"
        save_linking_index(index, path)
        loaded = load_linking_index(path)

        result = link_span("Olympus Monss", "entity", loaded)
        assert result is not None
        assert result.match_type == "fuzzy"

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Loading from nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_linking_index(tmp_path / "nonexistent.json")

    def test_save_creates_parent_dirs(self, tmp_path: Path, index: LinkingIndex) -> None:
        """Save creates parent directories if needed."""
        path = tmp_path / "sub" / "dir" / "index.json"
        save_linking_index(index, path)
        assert path.exists()

    def test_serialized_format_is_json(self, index: LinkingIndex, tmp_path: Path) -> None:
        """Saved file is valid JSON."""
        path = tmp_path / "index.json"
        save_linking_index(index, path)
        with open(path) as f:
            data = json.load(f)
        assert "exact_map" in data
        assert "entries" in data


# ---------------------------------------------------------------------------
# LinkResult tests
# ---------------------------------------------------------------------------

class TestLinkResult:
    """LinkResult dataclass behavior."""

    def test_to_dict(self) -> None:
        """LinkResult.to_dict() produces correct dict."""
        result = LinkResult(
            canonical_id="uat:123",
            source_vocabulary="uat",
            confidence=1.0,
            match_type="exact",
        )
        d = result.to_dict()
        assert d == {
            "canonical_id": "uat:123",
            "source_vocabulary": "uat",
            "confidence": 1.0,
            "match_type": "exact",
        }

    def test_frozen(self) -> None:
        """LinkResult is immutable."""
        result = LinkResult("uat:123", "uat", 1.0, "exact")
        with pytest.raises(AttributeError):
            result.confidence = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Alias resolution tests
# ---------------------------------------------------------------------------

class TestAliasResolution:
    """Verify alias resolution works across different vocabularies."""

    def test_water_cycle_alias(self, index: LinkingIndex) -> None:
        """SWEET alias 'Water cycle' resolves to Hydrosphere."""
        result = link_span("Water cycle", "topic", index)
        assert result is not None
        assert result.canonical_id == "sweet:456"
        assert result.source_vocabulary == "sweet"

    def test_atmos_chemistry_alias(self, index: LinkingIndex) -> None:
        """GCMD alias 'Atmos Chemistry' resolves correctly."""
        result = link_span("Atmos Chemistry", "topic", index)
        assert result is not None
        assert result.canonical_id == "gcmd:789"
        assert result.source_vocabulary == "gcmd"

    def test_hst_alias_resolves(self, index: LinkingIndex) -> None:
        """HST alias resolves to Hubble Space Telescope."""
        result = link_span("HST", "entity", index)
        assert result is not None
        assert result.canonical_id == "ror:04nt9b154"
        assert result.source_vocabulary == "ror"
