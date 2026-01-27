"""Tests for entity_linker module: exact match, fuzzy match, embedding match, type filtering, persistence."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from finetune.dataset_agent.entity_linker import (
    CatalogEntry,
    EmbeddingIndex,
    LinkingIndex,
    LinkResult,
    build_embedding_index,
    build_linking_index,
    link_span,
    link_span_cascade,
    link_span_embedding,
    load_embedding_index,
    load_linking_index,
    save_embedding_index,
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


# ---------------------------------------------------------------------------
# Embedding index fixtures
# ---------------------------------------------------------------------------

def _make_mock_model(dim: int = 384) -> MagicMock:
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    model = MagicMock()

    def _encode(texts: list[str], normalize_embeddings: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        rng = np.random.RandomState(42)
        embs = rng.randn(len(texts), dim).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embs = embs / norms
        return embs

    model.encode = _encode
    return model


def _make_catalog_entries() -> list[CatalogEntry]:
    """Create test catalog entries for embedding index."""
    return [
        CatalogEntry(
            canonical_id="uat:123",
            source_vocabulary="uat",
            span_type="topic",
            label="Exoplanets",
            aliases=["Extra-solar planets"],
        ),
        CatalogEntry(
            canonical_id="sweet:456",
            source_vocabulary="sweet",
            span_type="topic",
            label="Hydrosphere",
            aliases=["Water cycle"],
        ),
        CatalogEntry(
            canonical_id="ror:04nt9b154",
            source_vocabulary="ror",
            span_type="entity",
            label="Hubble Space Telescope",
            aliases=["HST"],
        ),
        CatalogEntry(
            canonical_id="planetary:Mars/1234",
            source_vocabulary="planetary",
            span_type="entity",
            label="Olympus Mons",
            aliases=["Nix Olympica"],
        ),
    ]


@pytest.fixture()
def mock_model():
    """Provide a mock sentence transformer model."""
    return _make_mock_model()


@pytest.fixture()
def embedding_index(mock_model: MagicMock) -> EmbeddingIndex:
    """Build an embedding index using the mock model."""
    entries = _make_catalog_entries()
    with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
        return build_embedding_index(entries, model_name="mock-model")


# ---------------------------------------------------------------------------
# Embedding index building tests
# ---------------------------------------------------------------------------

class TestBuildEmbeddingIndex:
    """Tests for build_embedding_index."""

    def test_builds_successfully(self, embedding_index: EmbeddingIndex) -> None:
        """Embedding index builds with correct shape."""
        # 4 entries * 2 surface forms each = 8 surfaces
        assert embedding_index.embeddings.shape[0] == 8
        assert embedding_index.embeddings.shape[1] == 384
        assert len(embedding_index.entries) == 8

    def test_empty_catalog(self, mock_model: MagicMock) -> None:
        """Empty catalog produces empty index."""
        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            idx = build_embedding_index([], model_name="mock-model")
        assert idx.embeddings.shape[0] == 0
        assert len(idx.entries) == 0

    def test_model_name_stored(self, embedding_index: EmbeddingIndex) -> None:
        """Model name is stored in the index."""
        assert embedding_index.model_name == "mock-model"

    def test_entries_have_correct_metadata(self, embedding_index: EmbeddingIndex) -> None:
        """Each entry has required metadata fields."""
        for entry in embedding_index.entries:
            assert "canonical_id" in entry
            assert "source_vocabulary" in entry
            assert "span_type" in entry
            assert "surface" in entry

    def test_embeddings_are_normalized(self, embedding_index: EmbeddingIndex) -> None:
        """Embeddings should be L2-normalized (unit vectors)."""
        norms = np.linalg.norm(embedding_index.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Embedding linking tests
# ---------------------------------------------------------------------------

class TestLinkSpanEmbedding:
    """Tests for link_span_embedding."""

    def test_match_above_threshold(self, embedding_index: EmbeddingIndex, mock_model: MagicMock) -> None:
        """A query that produces high similarity should return a match."""
        # Override encode to return the first entry's embedding (perfect match)
        first_emb = embedding_index.embeddings[0:1].copy()

        def _encode_match(texts, normalize_embeddings=True, show_progress_bar=False):
            return first_emb

        mock_model.encode = _encode_match
        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            result = link_span_embedding("Exoplanets", "topic", embedding_index, threshold=0.75)

        assert result is not None
        assert result.canonical_id == "uat:123"
        assert result.match_type == "embedding"
        assert result.confidence >= 0.75

    def test_no_match_below_threshold(self, embedding_index: EmbeddingIndex, mock_model: MagicMock) -> None:
        """A query with low similarity should return None."""
        # Return a random vector that won't match well
        rng = np.random.RandomState(999)
        random_emb = rng.randn(1, 384).astype(np.float32)
        random_emb = random_emb / np.linalg.norm(random_emb)

        def _encode_nomatch(texts, normalize_embeddings=True, show_progress_bar=False):
            return random_emb

        mock_model.encode = _encode_nomatch
        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            result = link_span_embedding("Completely random text", "topic", embedding_index, threshold=0.99)

        assert result is None

    def test_type_filtering(self, embedding_index: EmbeddingIndex, mock_model: MagicMock) -> None:
        """Embedding match should respect type filtering."""
        # Use a topic entry embedding but query as entity
        topic_emb = embedding_index.embeddings[0:1].copy()

        def _encode_typed(texts, normalize_embeddings=True, show_progress_bar=False):
            return topic_emb

        mock_model.encode = _encode_typed
        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            result = link_span_embedding("Exoplanets", "entity", embedding_index, threshold=0.75)

        # Topic entry should not match when querying as entity type
        # (unless an entity entry happens to have similar embedding from the random mock)
        # This test verifies type filtering is applied
        if result is not None:
            assert result.canonical_id in ("ror:04nt9b154", "planetary:Mars/1234")

    def test_empty_surface(self, embedding_index: EmbeddingIndex, mock_model: MagicMock) -> None:
        """Empty or whitespace surface returns None."""
        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            result = link_span_embedding("", "topic", embedding_index)
        assert result is None

    def test_empty_index(self, mock_model: MagicMock) -> None:
        """Empty embedding index returns None."""
        empty_idx = EmbeddingIndex(
            embeddings=np.zeros((0, 384), dtype=np.float32),
            entries=[],
            model_name="mock-model",
        )
        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            result = link_span_embedding("anything", "topic", empty_idx)
        assert result is None

    def test_confidence_is_cosine_similarity(self, embedding_index: EmbeddingIndex, mock_model: MagicMock) -> None:
        """Confidence value should be the cosine similarity score."""
        first_emb = embedding_index.embeddings[0:1].copy()

        def _encode_exact(texts, normalize_embeddings=True, show_progress_bar=False):
            return first_emb

        mock_model.encode = _encode_exact
        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            result = link_span_embedding("Exoplanets", "topic", embedding_index, threshold=0.5)

        assert result is not None
        # Perfect match with itself should give cosine sim of 1.0
        assert result.confidence == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Embedding index persistence tests
# ---------------------------------------------------------------------------

class TestEmbeddingPersistence:
    """Tests for save/load embedding index."""

    def test_save_and_load_roundtrip(self, embedding_index: EmbeddingIndex, tmp_path: Path) -> None:
        """Save and reload produces functionally identical index."""
        save_embedding_index(embedding_index, tmp_path / "emb_index")
        loaded = load_embedding_index(tmp_path / "emb_index")

        np.testing.assert_array_equal(loaded.embeddings, embedding_index.embeddings)
        assert loaded.entries == embedding_index.entries
        assert loaded.model_name == embedding_index.model_name

    def test_saved_files_exist(self, embedding_index: EmbeddingIndex, tmp_path: Path) -> None:
        """Save creates expected files."""
        out_dir = tmp_path / "emb_index"
        save_embedding_index(embedding_index, out_dir)

        assert (out_dir / "embedding_vectors.npy").exists()
        assert (out_dir / "embedding_metadata.json").exists()

    def test_metadata_is_valid_json(self, embedding_index: EmbeddingIndex, tmp_path: Path) -> None:
        """Metadata file is valid JSON."""
        out_dir = tmp_path / "emb_index"
        save_embedding_index(embedding_index, out_dir)

        with open(out_dir / "embedding_metadata.json") as f:
            data = json.load(f)
        assert "entries" in data
        assert "model_name" in data

    def test_loaded_index_produces_results(
        self, embedding_index: EmbeddingIndex, mock_model: MagicMock, tmp_path: Path
    ) -> None:
        """Loaded index can be used for linking."""
        out_dir = tmp_path / "emb_index"
        save_embedding_index(embedding_index, out_dir)
        loaded = load_embedding_index(out_dir)

        # Use perfect match embedding
        first_emb = loaded.embeddings[0:1].copy()

        def _encode_reloaded(texts, normalize_embeddings=True, show_progress_bar=False):
            return first_emb

        mock_model.encode = _encode_reloaded
        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            result = link_span_embedding("Exoplanets", "topic", loaded, threshold=0.75)

        assert result is not None

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Loading from nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_embedding_index(tmp_path / "nonexistent")

    def test_creates_parent_dirs(self, embedding_index: EmbeddingIndex, tmp_path: Path) -> None:
        """Save creates parent directories if needed."""
        out_dir = tmp_path / "deep" / "nested" / "dir"
        save_embedding_index(embedding_index, out_dir)
        assert (out_dir / "embedding_vectors.npy").exists()

    def test_embeddings_shape_preserved(self, embedding_index: EmbeddingIndex, tmp_path: Path) -> None:
        """Embedding matrix shape is preserved through save/load cycle."""
        out_dir = tmp_path / "emb_index"
        save_embedding_index(embedding_index, out_dir)
        loaded = load_embedding_index(out_dir)
        assert loaded.embeddings.shape == embedding_index.embeddings.shape


# ---------------------------------------------------------------------------
# Full cascade tests
# ---------------------------------------------------------------------------

class TestCascade:
    """Tests for link_span_cascade (exact -> fuzzy -> embedding)."""

    def test_cascade_exact_match_first(self, index: LinkingIndex, embedding_index: EmbeddingIndex) -> None:
        """Cascade returns exact match without touching embedding."""
        result = link_span_cascade("Exoplanets", "topic", index, embedding_index)
        assert result is not None
        assert result.match_type == "exact"
        assert result.confidence == 1.0

    def test_cascade_fuzzy_match_second(self, index: LinkingIndex, embedding_index: EmbeddingIndex) -> None:
        """Cascade returns fuzzy match if exact fails."""
        result = link_span_cascade("Olympus Monss", "entity", index, embedding_index)
        assert result is not None
        assert result.match_type == "fuzzy"
        assert result.confidence == 0.8

    def test_cascade_embedding_fallback(
        self, index: LinkingIndex, embedding_index: EmbeddingIndex, mock_model: MagicMock
    ) -> None:
        """Cascade falls through to embedding if exact and fuzzy both fail."""
        # Use an entity embedding for a surface that won't exact/fuzzy match
        entity_idx = next(
            i for i, e in enumerate(embedding_index.entries) if e["span_type"] == "entity"
        )
        entity_emb = embedding_index.embeddings[entity_idx : entity_idx + 1].copy()

        def _encode_fallback(texts, normalize_embeddings=True, show_progress_bar=False):
            return entity_emb

        mock_model.encode = _encode_fallback
        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            result = link_span_cascade(
                "The Famous Hubble Observatory",
                "entity",
                index,
                embedding_index,
                embedding_threshold=0.5,
            )

        assert result is not None
        assert result.match_type == "embedding"

    def test_cascade_no_match(self, index: LinkingIndex, mock_model: MagicMock) -> None:
        """Cascade returns None if all stages fail."""
        empty_emb = EmbeddingIndex(
            embeddings=np.zeros((0, 384), dtype=np.float32),
            entries=[],
            model_name="mock-model",
        )
        result = link_span_cascade("xyzzy_nonexistent", "topic", index, empty_emb)
        assert result is None

    def test_cascade_without_embedding_index(self, index: LinkingIndex) -> None:
        """Cascade works without an embedding index (only exact+fuzzy)."""
        result = link_span_cascade("Exoplanets", "topic", index)
        assert result is not None
        assert result.match_type == "exact"

    def test_cascade_without_embedding_no_match(self, index: LinkingIndex) -> None:
        """Cascade without embedding returns None for unmatched surface."""
        result = link_span_cascade("xyzzy_nonexistent", "topic", index)
        assert result is None


# ---------------------------------------------------------------------------
# Integration test: full cascade on test spans
# ---------------------------------------------------------------------------

class TestCascadeIntegration:
    """Integration test: run the full cascade on a batch of spans."""

    def test_cascade_on_100_spans(
        self, index: LinkingIndex, embedding_index: EmbeddingIndex, mock_model: MagicMock
    ) -> None:
        """Run cascade on 100 synthetic spans and verify cascade ordering."""
        # Create 100 test spans: mix of exact, fuzzy, embedding, and no-match
        test_spans = (
            # Exact match spans (25)
            [("Exoplanets", "topic")] * 5
            + [("Hydrosphere", "topic")] * 5
            + [("Hubble Space Telescope", "entity")] * 5
            + [("Olympus Mons", "entity")] * 5
            + [("HST", "entity")] * 5
            # Fuzzy match spans (25)
            + [("Exoplanetts", "topic")] * 5
            + [("Hydrospherre", "topic")] * 5
            + [("Olympus Monss", "entity")] * 5
            + [("Water cycl", "topic")] * 5
            + [("Atmos Chemistri", "topic")] * 5
            # Spans that should go to embedding or no match (50)
            + [("stellar nucleosynthesis", "topic")] * 10
            + [("radio telescope array", "entity")] * 10
            + [("completely random text xyzzy", "topic")] * 10
            + [("quantum chromodynamics", "topic")] * 10
            + [("ALMA observatory Chile", "entity")] * 10
        )
        assert len(test_spans) == 100

        exact_count = 0
        fuzzy_count = 0
        embedding_count = 0
        no_match_count = 0

        with patch("finetune.dataset_agent.entity_linker._load_sentence_transformer", return_value=mock_model):
            for surface, span_type in test_spans:
                result = link_span_cascade(
                    surface, span_type, index, embedding_index, embedding_threshold=0.75
                )
                if result is None:
                    no_match_count += 1
                elif result.match_type == "exact":
                    exact_count += 1
                elif result.match_type == "fuzzy":
                    fuzzy_count += 1
                elif result.match_type == "embedding":
                    embedding_count += 1

        # Exact matches should work for the 25 exact spans
        assert exact_count == 25
        # Some fuzzy matches should work
        assert fuzzy_count >= 10
        # Embedding and no-match cover the rest
        assert exact_count + fuzzy_count + embedding_count + no_match_count == 100
        # Cascade ordering: exact is tried first (validated by exact_count == 25)
        # Fuzzy is tried before embedding (validated by the fuzzy spans matching)
