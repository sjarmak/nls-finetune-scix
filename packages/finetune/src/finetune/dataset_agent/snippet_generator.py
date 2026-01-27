"""Synthetic text snippet generator for enrichment training data.

Generates title-like and abstract-like text snippets containing known
entity/topic mentions from the unified catalogs. Each snippet includes
ground-truth span annotations for training NER/extraction models.

Output schema per snippet:
    {
        "id": str,
        "text": str,
        "text_type": "title" | "abstract",
        "spans": [
            {
                "surface": str,
                "start": int,
                "end": int,
                "type": "topic" | "entity",
                "canonical_id": str,
                "source_vocabulary": str,
            }
        ],
        "domain_tags": [str],
        "provenance": {"template_id": str, "slot_entries": [str]},
    }
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune.dataset_agent.schemas import EntityEntry, TopicEntry
from finetune.dataset_agent.writers import JSONLReader, JSONLWriter

# ---------------------------------------------------------------------------
# Title templates
# ---------------------------------------------------------------------------
# Each template has {topic}, {entity}, or both as placeholders.
# They produce short phrases of ~5-15 words.

TITLE_TEMPLATES: list[dict[str, Any]] = [
    # topic only
    {"text": "Observations of {topic} in the solar atmosphere", "slots": ["topic"]},
    {"text": "New constraints on {topic} from satellite data", "slots": ["topic"]},
    {"text": "A survey of {topic} across stellar populations", "slots": ["topic"]},
    {"text": "Mapping {topic} using multi-wavelength imaging", "slots": ["topic"]},
    {"text": "Temporal variability of {topic} in planetary atmospheres", "slots": ["topic"]},
    {"text": "Modeling {topic} with Monte Carlo simulations", "slots": ["topic"]},
    {"text": "The role of {topic} in galaxy evolution", "slots": ["topic"]},
    {"text": "Detection of {topic} signatures in spectroscopic data", "slots": ["topic"]},
    {"text": "Characterization of {topic} in protoplanetary disks", "slots": ["topic"]},
    {"text": "On the relationship between {topic} and stellar mass", "slots": ["topic"]},
    # entity only
    {"text": "Surface geology of {entity} from orbital observations", "slots": ["entity"]},
    {"text": "Topographic analysis of {entity} using altimetry data", "slots": ["entity"]},
    {"text": "Research output from {entity} in the past decade", "slots": ["entity"]},
    {"text": "Collaborative studies involving {entity} and partner institutions", "slots": ["entity"]},
    {"text": "Impact cratering history near {entity}", "slots": ["entity"]},
    {"text": "Mineralogical mapping around {entity} from spectral data", "slots": ["entity"]},
    # topic + entity
    {"text": "{topic} observed near {entity} by remote sensing", "slots": ["topic", "entity"]},
    {"text": "{topic} measurements at {entity} during opposition", "slots": ["topic", "entity"]},
    {"text": "Evidence of {topic} at {entity} from radar data", "slots": ["topic", "entity"]},
    {"text": "{entity} as a laboratory for studying {topic}", "slots": ["topic", "entity"]},
    {"text": "Investigating {topic} patterns surrounding {entity}", "slots": ["topic", "entity"]},
    # two topics
    {"text": "Linking {topic} and {topic2} in stellar environments", "slots": ["topic", "topic2"]},
    {"text": "Interaction between {topic} and {topic2} in planetary science", "slots": ["topic", "topic2"]},
]

# ---------------------------------------------------------------------------
# Abstract templates
# ---------------------------------------------------------------------------
# Multi-sentence templates combining 2-4 catalog entries.

ABSTRACT_TEMPLATES: list[dict[str, Any]] = [
    {
        "text": (
            "We present new observations of {topic} obtained from ground-based telescopes. "
            "Our analysis reveals significant variability in {topic} across different epochs. "
            "These findings have implications for models of {topic2} and related processes."
        ),
        "slots": ["topic", "topic2"],
    },
    {
        "text": (
            "This study examines the distribution of {topic} near {entity}. "
            "Using high-resolution imaging, we identify spatial correlations between {topic} intensity and surface morphology. "
            "The results suggest that {entity} plays a key role in local {topic} dynamics."
        ),
        "slots": ["topic", "entity"],
    },
    {
        "text": (
            "We report the detection of {topic} at {entity} using spectroscopic methods. "
            "The measured abundances are consistent with theoretical predictions for {topic2}. "
            "Further monitoring of {entity} is recommended to track temporal evolution."
        ),
        "slots": ["topic", "entity", "topic2"],
    },
    {
        "text": (
            "A comprehensive survey of {topic} across multiple planetary bodies is presented. "
            "Data from {entity} and surrounding regions reveal distinct patterns. "
            "We compare our results to laboratory measurements and find good agreement. "
            "The implications for {topic2} models are discussed."
        ),
        "slots": ["topic", "entity", "topic2"],
    },
    {
        "text": (
            "Observations from {entity} provide new insights into {topic}. "
            "Our spectral analysis identifies key signatures associated with {topic}. "
            "These results contribute to understanding the broader context of {topic2} in this environment."
        ),
        "slots": ["topic", "entity", "topic2"],
    },
    {
        "text": (
            "We investigate the relationship between {topic} and {topic2} using numerical simulations. "
            "The models reproduce observed trends in {topic} evolution. "
            "Predictions for future observations near {entity} are provided."
        ),
        "slots": ["topic", "entity", "topic2"],
    },
    {
        "text": (
            "The formation and evolution of {entity} is analyzed in the context of {topic}. "
            "Comparative analysis with {entity2} reveals shared geological processes. "
            "Our findings support the hypothesis that {topic} drives surface modification on both features."
        ),
        "slots": ["topic", "entity", "entity2"],
    },
    {
        "text": (
            "Multi-epoch data of {topic} near {entity} are analyzed. "
            "We detect statistically significant changes in {topic} levels over the observation period. "
            "The inferred rates are consistent with models invoking {topic2} as a driving mechanism."
        ),
        "slots": ["topic", "entity", "topic2"],
    },
]

# ---------------------------------------------------------------------------
# Span annotation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanAnnotation:
    """A single span annotation within generated text."""

    surface: str
    start: int
    end: int
    type: str  # "topic" or "entity"
    canonical_id: str
    source_vocabulary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "canonical_id": self.canonical_id,
            "source_vocabulary": self.source_vocabulary,
        }


@dataclass(frozen=True)
class Snippet:
    """A generated text snippet with span annotations."""

    id: str
    text: str
    text_type: str  # "title" or "abstract"
    spans: list[SpanAnnotation]
    domain_tags: list[str]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "text_type": self.text_type,
            "spans": [s.to_dict() for s in self.spans],
            "domain_tags": self.domain_tags,
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SnippetGeneratorConfig:
    """Configuration for the snippet generator.

    Attributes:
        title_count: Number of title-like snippets to produce.
        abstract_count: Number of abstract-like snippets to produce.
        domain_weights: Mapping of domain tag to relative weight for sampling.
            If empty, all domains are equally weighted.
        seed: Random seed for reproducibility. None for non-deterministic.
    """

    title_count: int = 500
    abstract_count: int = 500
    domain_weights: dict[str, float] = field(default_factory=dict)
    seed: int | None = 42


@dataclass
class SnippetGeneratorStats:
    """Statistics from snippet generation."""

    titles_generated: int = 0
    abstracts_generated: int = 0
    total_spans: int = 0
    spans_by_type: dict[str, int] = field(default_factory=dict)
    spans_by_domain: dict[str, int] = field(default_factory=dict)
    spans_by_vocabulary: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Catalog loading helpers
# ---------------------------------------------------------------------------


def _load_topics(path: Path) -> list[TopicEntry]:
    """Load topic entries from a JSONL catalog file."""
    if not path.exists():
        return []
    return [TopicEntry.from_dict(d) for d in JSONLReader(path)]


def _load_entities(path: Path) -> list[EntityEntry]:
    """Load entity entries from a JSONL catalog file."""
    if not path.exists():
        return []
    return [EntityEntry.from_dict(d) for d in JSONLReader(path)]


def _group_by_domain(
    topics: list[TopicEntry],
    entities: list[EntityEntry],
) -> dict[str, dict[str, list[TopicEntry | EntityEntry]]]:
    """Group catalog entries by domain tag.

    Returns:
        Mapping of domain -> {"topics": [...], "entities": [...]}
    """
    grouped: dict[str, dict[str, list[TopicEntry | EntityEntry]]] = {}
    for entry in topics:
        for tag in entry.domain_tags or ["untagged"]:
            bucket = grouped.setdefault(tag, {"topics": [], "entities": []})
            bucket["topics"].append(entry)
    for entry in entities:
        for tag in entry.domain_tags or ["untagged"]:
            bucket = grouped.setdefault(tag, {"topics": [], "entities": []})
            bucket["entities"].append(entry)
    return grouped


# ---------------------------------------------------------------------------
# Snippet ID generation
# ---------------------------------------------------------------------------


def _snippet_id(text_type: str, index: int) -> str:
    """Generate a deterministic snippet ID."""
    raw = f"snippet_{text_type}_{index}"
    hash_val = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"snip_{text_type[:3]}_{hash_val}"


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------


def _pick_entry(
    entries: list[TopicEntry | EntityEntry],
    rng: random.Random,
    used_ids: set[str],
) -> TopicEntry | EntityEntry | None:
    """Pick a random catalog entry, preferring unused ones."""
    if not entries:
        return None
    # Try unused first
    unused = [e for e in entries if e.id not in used_ids]
    pool = unused if unused else entries
    return rng.choice(pool)


def _surface_label(entry: TopicEntry | EntityEntry) -> str:
    """Choose a surface form for the entry (label or alias)."""
    return entry.label


def _build_spans(
    text: str,
    fills: list[tuple[str, str, TopicEntry | EntityEntry]],
) -> list[SpanAnnotation]:
    """Build non-overlapping span annotations for filled text.

    Args:
        text: The generated text after slot filling.
        fills: List of (placeholder_tag, surface_text, catalog_entry) triples
               in the order they appear in the text.

    Returns:
        List of SpanAnnotation sorted by start offset.
    """
    spans: list[SpanAnnotation] = []
    search_from = 0
    for _tag, surface, entry in fills:
        idx = text.find(surface, search_from)
        if idx < 0:
            continue
        entry_type = "topic" if isinstance(entry, TopicEntry) else "entity"
        spans.append(
            SpanAnnotation(
                surface=surface,
                start=idx,
                end=idx + len(surface),
                type=entry_type,
                canonical_id=entry.id,
                source_vocabulary=entry.source_vocabulary,
            )
        )
        search_from = idx + len(surface)
    return sorted(spans, key=lambda s: s.start)


def _select_domain(
    grouped: dict[str, dict[str, list[TopicEntry | EntityEntry]]],
    rng: random.Random,
    weights: dict[str, float],
) -> str:
    """Select a domain tag based on configured weights."""
    domains = list(grouped.keys())
    if not domains:
        return "untagged"
    if weights:
        w = [weights.get(d, 1.0) for d in domains]
    else:
        w = [1.0] * len(domains)
    return rng.choices(domains, weights=w, k=1)[0]


def _fill_template(
    template: dict[str, Any],
    topics: list[TopicEntry | EntityEntry],
    entities: list[EntityEntry | TopicEntry],
    rng: random.Random,
) -> tuple[str, list[tuple[str, str, TopicEntry | EntityEntry]]] | None:
    """Fill a template's slots with catalog entries.

    Returns:
        (filled_text, fills_list) or None if we couldn't fill all required slots.
    """
    slots = template["slots"]
    text = template["text"]
    fills: list[tuple[str, str, TopicEntry | EntityEntry]] = []
    used_ids: set[str] = set()

    for slot in slots:
        if slot.startswith("topic"):
            entry = _pick_entry(topics, rng, used_ids)
        elif slot.startswith("entity"):
            entry = _pick_entry(entities, rng, used_ids)
        else:
            entry = None

        if entry is None:
            return None

        surface = _surface_label(entry)
        placeholder = "{" + slot + "}"
        text = text.replace(placeholder, surface, 1)
        fills.append((slot, surface, entry))
        used_ids.add(entry.id)

    return text, fills


def _collect_domain_tags(
    fills: list[tuple[str, str, TopicEntry | EntityEntry]],
) -> list[str]:
    """Collect unique domain tags from all filled entries."""
    tags: set[str] = set()
    for _, _, entry in fills:
        for tag in entry.domain_tags:
            tags.add(tag)
    return sorted(tags)


def generate_snippets(
    topics: list[TopicEntry],
    entities: list[EntityEntry],
    config: SnippetGeneratorConfig | None = None,
) -> tuple[list[Snippet], SnippetGeneratorStats]:
    """Generate synthetic snippets from catalog entries.

    Args:
        topics: Topic catalog entries.
        entities: Entity catalog entries.
        config: Generation configuration.

    Returns:
        (list of Snippet, SnippetGeneratorStats)
    """
    cfg = config or SnippetGeneratorConfig()
    rng = random.Random(cfg.seed)
    stats = SnippetGeneratorStats()

    grouped = _group_by_domain(topics, entities)

    snippets: list[Snippet] = []

    def _generate_batch(
        templates: list[dict[str, Any]],
        text_type: str,
        count: int,
    ) -> None:
        for i in range(count):
            domain = _select_domain(grouped, rng, cfg.domain_weights)
            bucket = grouped.get(domain, {"topics": [], "entities": []})
            domain_topics = bucket["topics"]
            domain_entities = bucket["entities"]

            template = rng.choice(templates)

            result = _fill_template(template, domain_topics, domain_entities, rng)
            if result is None:
                continue

            text, fills = result
            span_annots = _build_spans(text, fills)

            # Validate spans
            valid_spans = [
                s for s in span_annots if s.surface == text[s.start : s.end]
            ]

            if not valid_spans:
                continue

            snippet = Snippet(
                id=_snippet_id(text_type, len(snippets)),
                text=text,
                text_type=text_type,
                spans=valid_spans,
                domain_tags=_collect_domain_tags(fills),
                provenance={
                    "template_id": f"{text_type}_{templates.index(template)}",
                    "slot_entries": [e.id for _, _, e in fills],
                },
            )
            snippets.append(snippet)

            # Update stats
            if text_type == "title":
                stats.titles_generated += 1
            else:
                stats.abstracts_generated += 1
            for s in valid_spans:
                stats.total_spans += 1
                stats.spans_by_type[s.type] = stats.spans_by_type.get(s.type, 0) + 1
                for dtag in snippet.domain_tags:
                    stats.spans_by_domain[dtag] = (
                        stats.spans_by_domain.get(dtag, 0) + 1
                    )
                stats.spans_by_vocabulary[s.source_vocabulary] = (
                    stats.spans_by_vocabulary.get(s.source_vocabulary, 0) + 1
                )

    _generate_batch(TITLE_TEMPLATES, "title", cfg.title_count)
    _generate_batch(ABSTRACT_TEMPLATES, "abstract", cfg.abstract_count)

    return snippets, stats


# ---------------------------------------------------------------------------
# File-based API
# ---------------------------------------------------------------------------


def generate_snippets_from_catalogs(
    topic_catalog_path: Path,
    entity_catalog_path: Path,
    output_path: Path,
    config: SnippetGeneratorConfig | None = None,
) -> tuple[int, str, SnippetGeneratorStats]:
    """Generate snippets from catalog files and write to JSONL.

    Args:
        topic_catalog_path: Path to topic_catalog.jsonl
        entity_catalog_path: Path to entity_catalog.jsonl
        output_path: Path for output snippets JSONL file.
        config: Generation configuration.

    Returns:
        (snippet_count, sha256_checksum, stats)
    """
    topics = _load_topics(topic_catalog_path)
    entities = _load_entities(entity_catalog_path)

    snippets, stats = generate_snippets(topics, entities, config)

    with JSONLWriter(output_path) as writer:
        for snippet in snippets:
            writer.write_line(snippet)

    return writer.line_count, writer.checksum, stats
