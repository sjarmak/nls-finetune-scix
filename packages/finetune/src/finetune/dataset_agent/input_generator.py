"""NL input generator for dataset generation pipeline.

This module generates diverse natural language inputs by filling template slots
with values from topic and entity catalogs. It supports:

- Slot filling from topic_catalog.jsonl and entity_catalog_*.jsonl
- Configurable noise options: stopwords, paraphrase variants, alias sampling
- Deterministic output with seed parameter

Output: pairs/inputs.jsonl with NLInput records
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune.dataset_agent.schemas import EntityEntry, NLInput, Template, TopicEntry
from finetune.dataset_agent.writers import JSONLReader, JSONLWriter


@dataclass
class InputGeneratorConfig:
    """Configuration for NL input generation.

    Attributes:
        seed: Random seed for deterministic output. If None, uses system random.
        samples_per_template: Number of inputs to generate per template (0 = all combinations)
        samples_per_nl_variant: Number of inputs to generate per NL template variant
        enable_stopword_noise: Add optional stopwords (e.g., "please", "show me")
        enable_paraphrase_variants: Use alternate NL templates from same template
        enable_alias_sampling: Use aliases from catalogs instead of primary labels
        alias_sample_rate: Probability of using alias instead of primary label (0.0-1.0)
        max_aliases_per_slot: Maximum number of different aliases to sample per slot
        author_names: List of author names to use for author slots
        literal_values: Dict mapping slot name patterns to literal values
    """

    seed: int | None = None
    samples_per_template: int = 100
    samples_per_nl_variant: int = 0  # 0 means use all NL variants
    enable_stopword_noise: bool = True
    enable_paraphrase_variants: bool = True
    enable_alias_sampling: bool = True
    alias_sample_rate: float = 0.3
    max_aliases_per_slot: int = 5
    author_names: list[str] = field(default_factory=list)
    literal_values: dict[str, list[str]] = field(default_factory=dict)


# Default author names for author slots
DEFAULT_AUTHOR_NAMES = [
    "Einstein",
    "Hubble",
    "Sagan",
    "Hawking",
    "Feynman",
    "Penrose",
    "Rees",
    "Tyson",
    "Thorne",
    "Chandrasekhar",
    "Bethe",
    "Fowler",
    "Rubin",
    "Faber",
    "Perlmutter",
    "Schmidt",
    "Riess",
    "Guth",
    "Linde",
    "Susskind",
    "Witten",
    "Strominger",
    "Vafa",
    "Maldacena",
    "Polchinski",
    "Einstein, A.",
    "Hubble, E.",
    "Sagan, C.",
    "Hawking, S.",
    "Feynman, R.",
]


# Default literal values for common slot types
DEFAULT_LITERAL_VALUES = {
    "phrase": [
        "dark matter",
        "gravitational waves",
        "black hole",
        "exoplanet",
        "supernova",
        "cosmic microwave background",
        "galaxy formation",
        "stellar evolution",
        "pulsar timing",
        "quasar",
    ],
    "year": ["2020", "2021", "2022", "2023", "2024"],
    "start_year": ["2015", "2018", "2020", "2022"],
    "end_year": ["2023", "2024", "2025", "*"],
    "bibcode": [
        "2020ApJ...895L...1P",
        "2019Natur.574..211H",
        "2016PhRvL.116f1102A",
        "2018ApJ...853L...1P",
    ],
}


# Optional stopwords to inject for natural language variation
STOPWORD_PREFIXES = [
    "",
    "find ",
    "search for ",
    "show me ",
    "get me ",
    "I want ",
    "can you find ",
    "please find ",
    "look up ",
    "retrieve ",
]

STOPWORD_SUFFIXES = [
    "",
    " please",
    " for me",
    " if possible",
]


@dataclass
class SlotValue:
    """A value to fill a template slot.

    Attributes:
        text: The text to insert into the template
        source_id: ID from the catalog (if from topic/entity catalog)
        is_alias: Whether this is an alias or the primary label
        slot_name: Name of the slot this value is for
    """

    text: str
    source_id: str | None = None
    is_alias: bool = False
    slot_name: str = ""


@dataclass
class InputGeneratorStats:
    """Statistics from input generation.

    Attributes:
        templates_processed: Number of templates processed
        inputs_generated: Total number of inputs generated
        topics_used: Number of unique topics used
        entities_used: Number of unique entities used
        aliases_used: Number of times aliases were used instead of primary labels
        stopwords_added: Number of inputs with stopword noise
        nl_variants_used: Number of different NL template variants used
    """

    templates_processed: int = 0
    inputs_generated: int = 0
    topics_used: int = 0
    entities_used: int = 0
    aliases_used: int = 0
    stopwords_added: int = 0
    nl_variants_used: int = 0


class InputGenerator:
    """Generator for NL inputs from templates and catalogs.

    This class generates NL inputs by:
    1. Loading topic and entity catalogs
    2. For each template, sampling slot values from catalogs
    3. Filling NL template variants with sampled values
    4. Optionally adding noise (stopwords, alias variants)

    All operations are deterministic when a seed is provided.
    """

    def __init__(
        self,
        config: InputGeneratorConfig | None = None,
        topic_catalog: list[TopicEntry] | None = None,
        entity_catalogs: dict[str, list[EntityEntry]] | None = None,
    ) -> None:
        """Initialize the input generator.

        Args:
            config: Generator configuration. Uses defaults if None.
            topic_catalog: Pre-loaded topic catalog. Load from file if None.
            entity_catalogs: Pre-loaded entity catalogs keyed by type (e.g., "institutions").
        """
        self.config = config or InputGeneratorConfig()
        self._rng = random.Random(self.config.seed)

        # Catalogs
        self._topic_catalog = topic_catalog or []
        self._entity_catalogs = entity_catalogs or {}

        # Index for fast lookup
        self._topic_by_id: dict[str, TopicEntry] = {}
        self._entity_by_id: dict[str, EntityEntry] = {}

        # Build indexes
        self._build_indexes()

        # Stats
        self._stats = InputGeneratorStats()
        self._topics_used_ids: set[str] = set()
        self._entities_used_ids: set[str] = set()

    def _build_indexes(self) -> None:
        """Build lookup indexes for catalogs."""
        for entry in self._topic_catalog:
            self._topic_by_id[entry.id] = entry

        for catalog in self._entity_catalogs.values():
            for entry in catalog:
                self._entity_by_id[entry.id] = entry

    def load_topic_catalog(self, path: Path) -> None:
        """Load topic catalog from JSONL file.

        Args:
            path: Path to topic_catalog.jsonl
        """
        reader = JSONLReader(path)
        self._topic_catalog = [TopicEntry.from_dict(d) for d in reader]
        self._build_indexes()

    def load_entity_catalog(self, path: Path, catalog_type: str = "institutions") -> None:
        """Load entity catalog from JSONL file.

        Args:
            path: Path to entity_catalog_*.jsonl
            catalog_type: Type of entities (e.g., "institutions")
        """
        reader = JSONLReader(path)
        self._entity_catalogs[catalog_type] = [EntityEntry.from_dict(d) for d in reader]
        self._build_indexes()

    def _get_author_names(self) -> list[str]:
        """Get list of author names for author slots."""
        if self.config.author_names:
            return self.config.author_names
        return DEFAULT_AUTHOR_NAMES

    def _get_literal_values(self, slot_name: str) -> list[str]:
        """Get literal values for a slot name.

        Args:
            slot_name: Name of the slot

        Returns:
            List of possible literal values
        """
        # Check config first
        if slot_name in self.config.literal_values:
            return self.config.literal_values[slot_name]

        # Check defaults
        if slot_name in DEFAULT_LITERAL_VALUES:
            return DEFAULT_LITERAL_VALUES[slot_name]

        # Generic fallback
        return [f"<{slot_name}>"]

    def _sample_topic_values(
        self,
        n: int,
        slot_name: str,
    ) -> list[SlotValue]:
        """Sample topic values for a slot.

        Args:
            n: Number of values to sample
            slot_name: Name of the slot

        Returns:
            List of SlotValue objects
        """
        if not self._topic_catalog:
            return []

        values: list[SlotValue] = []
        sampled_topics = self._rng.sample(
            self._topic_catalog,
            min(n, len(self._topic_catalog)),
        )

        for topic in sampled_topics:
            # Decide whether to use alias
            use_alias = (
                self.config.enable_alias_sampling
                and topic.aliases
                and self._rng.random() < self.config.alias_sample_rate
            )

            if use_alias:
                alias = self._rng.choice(topic.aliases[: self.config.max_aliases_per_slot])
                values.append(
                    SlotValue(
                        text=alias,
                        source_id=topic.id,
                        is_alias=True,
                        slot_name=slot_name,
                    )
                )
            else:
                values.append(
                    SlotValue(
                        text=topic.label,
                        source_id=topic.id,
                        is_alias=False,
                        slot_name=slot_name,
                    )
                )

        return values

    def _sample_entity_values(
        self,
        n: int,
        slot_name: str,
        catalog_type: str = "institutions",
    ) -> list[SlotValue]:
        """Sample entity values for a slot.

        Args:
            n: Number of values to sample
            slot_name: Name of the slot
            catalog_type: Type of entity catalog to sample from

        Returns:
            List of SlotValue objects
        """
        catalog = self._entity_catalogs.get(catalog_type, [])
        if not catalog:
            return []

        values: list[SlotValue] = []
        sampled_entities = self._rng.sample(
            catalog,
            min(n, len(catalog)),
        )

        for entity in sampled_entities:
            use_alias = (
                self.config.enable_alias_sampling
                and entity.aliases
                and self._rng.random() < self.config.alias_sample_rate
            )

            if use_alias:
                alias = self._rng.choice(entity.aliases[: self.config.max_aliases_per_slot])
                values.append(
                    SlotValue(
                        text=alias,
                        source_id=entity.id,
                        is_alias=True,
                        slot_name=slot_name,
                    )
                )
            else:
                values.append(
                    SlotValue(
                        text=entity.label,
                        source_id=entity.id,
                        is_alias=False,
                        slot_name=slot_name,
                    )
                )

        return values

    def _sample_slot_values(
        self,
        template: Template,
        n_per_slot: int,
    ) -> dict[str, list[SlotValue]]:
        """Sample values for all slots in a template.

        Args:
            template: Template with slots to fill
            n_per_slot: Number of values to sample per slot

        Returns:
            Dict mapping slot names to lists of SlotValue
        """
        slot_values: dict[str, list[SlotValue]] = {}

        for slot_name, slot_def in template.slots.items():
            slot_type = slot_def.type

            if slot_type == "topic":
                slot_values[slot_name] = self._sample_topic_values(n_per_slot, slot_name)
            elif slot_type == "entity":
                # Get catalog type from constraints if specified
                catalog_type = slot_def.constraints.get("catalog", "institutions")
                # Map catalog name to type
                if "institution" in catalog_type.lower():
                    catalog_type = "institutions"
                slot_values[slot_name] = self._sample_entity_values(
                    n_per_slot, slot_name, catalog_type
                )
            elif slot_type == "author":
                author_names = self._get_author_names()
                sampled = self._rng.sample(
                    author_names,
                    min(n_per_slot, len(author_names)),
                )
                slot_values[slot_name] = [
                    SlotValue(text=name, slot_name=slot_name) for name in sampled
                ]
            elif slot_type == "literal":
                literals = self._get_literal_values(slot_name)
                sampled = self._rng.sample(
                    literals,
                    min(n_per_slot, len(literals)),
                )
                slot_values[slot_name] = [
                    SlotValue(text=val, slot_name=slot_name) for val in sampled
                ]
            elif slot_type == "date":
                # Handle date slots by looking up slot name
                literals = self._get_literal_values(slot_name)
                sampled = self._rng.sample(
                    literals,
                    min(n_per_slot, len(literals)),
                )
                slot_values[slot_name] = [
                    SlotValue(text=val, slot_name=slot_name) for val in sampled
                ]
            elif slot_type == "bibcode":
                literals = self._get_literal_values("bibcode")
                sampled = self._rng.sample(
                    literals,
                    min(n_per_slot, len(literals)),
                )
                slot_values[slot_name] = [
                    SlotValue(text=val, slot_name=slot_name) for val in sampled
                ]
            else:
                # Fallback for unknown slot types
                slot_values[slot_name] = [
                    SlotValue(text=f"<{slot_name}>", slot_name=slot_name)
                ]

        return slot_values

    def _apply_stopword_noise(self, text: str) -> tuple[str, bool]:
        """Apply optional stopword noise to input text.

        Args:
            text: Original NL text

        Returns:
            Tuple of (modified text, whether noise was added)
        """
        if not self.config.enable_stopword_noise:
            return text, False

        # Randomly decide whether to add noise
        if self._rng.random() < 0.5:
            return text, False

        prefix = self._rng.choice(STOPWORD_PREFIXES)
        suffix = self._rng.choice(STOPWORD_SUFFIXES)

        # If adding prefix, adjust capitalization
        if prefix:
            # Lowercase the first character of original text if prefix added
            if text and text[0].isupper():
                text = text[0].lower() + text[1:]

        result = f"{prefix}{text}{suffix}".strip()
        noise_added = bool(prefix or suffix)

        return result, noise_added

    def _generate_input_id(self, template_id: str, variant_idx: int, combo_idx: int) -> str:
        """Generate a unique input ID.

        Args:
            template_id: ID of the template
            variant_idx: Index of the NL template variant
            combo_idx: Index of the slot combination

        Returns:
            Unique input ID
        """
        # Create deterministic hash-based ID
        base = f"{template_id}:{variant_idx}:{combo_idx}:{self.config.seed}"
        hash_val = hashlib.md5(base.encode()).hexdigest()[:8]
        return f"input_{template_id}_{hash_val}"

    def _fill_template(
        self,
        nl_template: str,
        slot_values: dict[str, SlotValue],
    ) -> str:
        """Fill a template string with slot values.

        Args:
            nl_template: NL template with {slot} placeholders
            slot_values: Dict mapping slot names to SlotValue

        Returns:
            Filled template string
        """
        result = nl_template
        for slot_name, value in slot_values.items():
            result = result.replace(f"{{{slot_name}}}", value.text)
        return result

    def generate_from_template(
        self,
        template: Template,
        max_inputs: int | None = None,
    ) -> Iterator[NLInput]:
        """Generate NL inputs from a single template.

        Args:
            template: Template to generate inputs from
            max_inputs: Maximum number of inputs to generate. Uses config default if None.

        Yields:
            NLInput objects
        """
        max_inputs = max_inputs or self.config.samples_per_template

        # Sample slot values
        n_samples = max_inputs if max_inputs > 0 else 100
        slot_values_lists = self._sample_slot_values(template, n_samples)

        # Determine NL variants to use
        nl_templates = template.nl_templates
        if self.config.samples_per_nl_variant > 0:
            # Limit to configured number per variant
            nl_templates = nl_templates[: self.config.samples_per_nl_variant]
        elif not self.config.enable_paraphrase_variants:
            # Use only first variant
            nl_templates = nl_templates[:1]

        # Track unique NL variants used
        variants_used: set[int] = set()

        # Generate combinations
        generated_count = 0
        combo_idx = 0

        # Simple approach: iterate through slot value indices
        slot_names = list(slot_values_lists.keys())

        # Handle templates with no slots
        if not slot_names:
            # Generate inputs using just the NL templates (no slot filling needed)
            generated_count = 0
            variants_used: set[int] = set()
            for variant_idx, nl_template in enumerate(nl_templates):
                if max_inputs and generated_count >= max_inputs:
                    break
                variants_used.add(variant_idx)
                user_text = nl_template
                user_text, noise_added = self._apply_stopword_noise(user_text)
                if noise_added:
                    self._stats.stopwords_added += 1
                input_id = self._generate_input_id(template.id, variant_idx, 0)
                yield NLInput(
                    input_id=input_id,
                    user_text=user_text,
                    template_id=template.id,
                    filled_slots={},
                    source_ids=[],
                )
                generated_count += 1
            self._stats.templates_processed += 1
            self._stats.inputs_generated += generated_count
            self._stats.nl_variants_used += len(variants_used)
            return

        # Calculate max combinations - but if any slot is empty, return no inputs
        max_combos = 1
        for values in slot_values_lists.values():
            if not values:
                # Slot requires catalog values but none available - can't generate
                self._stats.templates_processed += 1
                return
            max_combos *= len(values)

        # Generate inputs
        for combo_idx in range(min(max_combos, max_inputs or max_combos)):
            if max_inputs and generated_count >= max_inputs:
                break

            # Select slot values for this combination
            current_slot_values: dict[str, SlotValue] = {}
            remaining = combo_idx

            for slot_name in slot_names:
                values_list = slot_values_lists[slot_name]
                if values_list:
                    idx = remaining % len(values_list)
                    remaining //= len(values_list)
                    current_slot_values[slot_name] = values_list[idx]

            # Generate for each NL variant
            for variant_idx, nl_template in enumerate(nl_templates):
                if max_inputs and generated_count >= max_inputs:
                    break

                variants_used.add(variant_idx)

                # Fill the template
                user_text = self._fill_template(nl_template, current_slot_values)

                # Apply stopword noise
                user_text, noise_added = self._apply_stopword_noise(user_text)
                if noise_added:
                    self._stats.stopwords_added += 1

                # Collect source IDs
                source_ids = [
                    v.source_id for v in current_slot_values.values() if v.source_id
                ]

                # Track alias usage
                for v in current_slot_values.values():
                    if v.is_alias:
                        self._stats.aliases_used += 1
                    if v.source_id:
                        if v.source_id in self._topic_by_id:
                            self._topics_used_ids.add(v.source_id)
                        elif v.source_id in self._entity_by_id:
                            self._entities_used_ids.add(v.source_id)

                # Create filled_slots dict for output
                filled_slots: dict[str, Any] = {
                    slot_name: {
                        "value": value.text,
                        "source_id": value.source_id,
                        "is_alias": value.is_alias,
                    }
                    for slot_name, value in current_slot_values.items()
                }

                input_id = self._generate_input_id(template.id, variant_idx, combo_idx)

                yield NLInput(
                    input_id=input_id,
                    user_text=user_text,
                    template_id=template.id,
                    filled_slots=filled_slots,
                    source_ids=source_ids,
                )

                generated_count += 1

        self._stats.templates_processed += 1
        self._stats.inputs_generated += generated_count
        self._stats.nl_variants_used += len(variants_used)
        # Update topic/entity counts
        self._stats.topics_used = len(self._topics_used_ids)
        self._stats.entities_used = len(self._entities_used_ids)

    def generate_from_templates(
        self,
        templates: list[Template],
        max_per_template: int | None = None,
    ) -> Iterator[NLInput]:
        """Generate NL inputs from multiple templates.

        Args:
            templates: List of templates to generate from
            max_per_template: Maximum inputs per template. Uses config default if None.

        Yields:
            NLInput objects
        """
        for template in templates:
            yield from self.generate_from_template(template, max_per_template)

    def generate_to_file(
        self,
        templates: list[Template],
        output_path: Path,
        max_per_template: int | None = None,
    ) -> tuple[str, int, InputGeneratorStats]:
        """Generate NL inputs and write to JSONL file.

        Args:
            templates: List of templates to generate from
            output_path: Path to output inputs.jsonl file
            max_per_template: Maximum inputs per template

        Returns:
            Tuple of (SHA256 checksum, line count, generation stats)
        """
        # Reset stats
        self._stats = InputGeneratorStats()
        self._topics_used_ids = set()
        self._entities_used_ids = set()

        with JSONLWriter(output_path) as writer:
            for nl_input in self.generate_from_templates(templates, max_per_template):
                writer.write_line(nl_input)

        # Update final stats
        self._stats.topics_used = len(self._topics_used_ids)
        self._stats.entities_used = len(self._entities_used_ids)

        return writer.checksum, writer.line_count, self._stats

    @property
    def stats(self) -> InputGeneratorStats:
        """Get generation statistics."""
        return self._stats


def generate_inputs(
    templates: list[Template],
    output_path: Path,
    topic_catalog: list[TopicEntry] | None = None,
    entity_catalogs: dict[str, list[EntityEntry]] | None = None,
    config: InputGeneratorConfig | None = None,
) -> tuple[str, int, InputGeneratorStats]:
    """Convenience function to generate NL inputs.

    Args:
        templates: List of templates to generate from
        output_path: Path to output inputs.jsonl file
        topic_catalog: Pre-loaded topic catalog
        entity_catalogs: Pre-loaded entity catalogs keyed by type
        config: Generator configuration

    Returns:
        Tuple of (SHA256 checksum, line count, generation stats)
    """
    generator = InputGenerator(
        config=config,
        topic_catalog=topic_catalog,
        entity_catalogs=entity_catalogs,
    )
    return generator.generate_to_file(templates, output_path)


def load_and_generate_inputs(
    templates: list[Template],
    output_path: Path,
    topic_catalog_path: Path | None = None,
    entity_catalog_paths: dict[str, Path] | None = None,
    config: InputGeneratorConfig | None = None,
) -> tuple[str, int, InputGeneratorStats]:
    """Load catalogs and generate NL inputs.

    Args:
        templates: List of templates to generate from
        output_path: Path to output inputs.jsonl file
        topic_catalog_path: Path to topic_catalog.jsonl
        entity_catalog_paths: Dict mapping catalog type to path
        config: Generator configuration

    Returns:
        Tuple of (SHA256 checksum, line count, generation stats)
    """
    generator = InputGenerator(config=config)

    if topic_catalog_path and topic_catalog_path.exists():
        generator.load_topic_catalog(topic_catalog_path)

    if entity_catalog_paths:
        for catalog_type, path in entity_catalog_paths.items():
            if path.exists():
                generator.load_entity_catalog(path, catalog_type)

    return generator.generate_to_file(templates, output_path)
