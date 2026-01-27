"""Enrichment label generator for dataset generation pipeline.

This module generates enrichment labels from validated pairs for training
entity/topic extraction models. It extracts structured labels from the
filled slots in pairs.

Features:
- Extract topic labels from topic slots
- Extract institution labels from entity slots
- Extract author labels from author slots
- Extract date range labels from date slots
- Track provenance (template_id, slot mapping)

Output: enrichment/enrichment_labels.jsonl with EnrichmentLabel records
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from finetune.dataset_agent.schemas import (
    EnrichmentLabel,
    Label,
    LabelType,
    Pair,
)
from finetune.dataset_agent.writers import JSONLReader, JSONLWriter


@dataclass
class EnrichmentGeneratorConfig:
    """Configuration for enrichment label generation.

    Attributes:
        include_topics: Generate topic labels from topic slots
        include_institutions: Generate institution labels from entity slots
        include_authors: Generate author labels from author slots
        include_dates: Generate date range labels from date slots
        require_source_id: Only generate labels for slots with source_id
    """

    include_topics: bool = True
    include_institutions: bool = True
    include_authors: bool = True
    include_dates: bool = True
    require_source_id: bool = False


@dataclass
class EnrichmentGeneratorStats:
    """Statistics from enrichment label generation.

    Attributes:
        pairs_processed: Number of pairs processed
        labels_generated: Total number of labels generated
        examples_generated: Number of enrichment examples generated
        topics_extracted: Number of topic labels extracted
        institutions_extracted: Number of institution labels extracted
        authors_extracted: Number of author labels extracted
        dates_extracted: Number of date range labels extracted
    """

    pairs_processed: int = 0
    labels_generated: int = 0
    examples_generated: int = 0
    topics_extracted: int = 0
    institutions_extracted: int = 0
    authors_extracted: int = 0
    dates_extracted: int = 0


def generate_example_id(pair_id: str) -> str:
    """Generate a unique example ID for an enrichment label.

    Args:
        pair_id: The pair ID to base the example ID on

    Returns:
        Unique example ID
    """
    # Create deterministic hash-based ID
    hash_val = hashlib.md5(pair_id.encode()).hexdigest()[:8]
    return f"enrich_{hash_val}"


def extract_label_from_slot(
    slot_name: str,
    slot_data: dict,
    user_text: str,
    config: EnrichmentGeneratorConfig,
) -> Label | None:
    """Extract a label from a filled slot.

    Args:
        slot_name: Name of the slot
        slot_data: Slot data dict with value, source_id, etc.
        user_text: The user text to find span in
        config: Generator configuration

    Returns:
        Label if extractable, None otherwise
    """
    # Get slot value
    if isinstance(slot_data, dict):
        value = slot_data.get("value", "")
        source_id = slot_data.get("source_id")
        slot_type = slot_data.get("type", slot_name)
    else:
        value = str(slot_data)
        source_id = None
        slot_type = slot_name

    # Skip if no value
    if not value:
        return None

    # Skip if source_id required but missing
    if config.require_source_id and not source_id:
        return None

    # Determine entity type based on slot name/type
    entity_type: str | None = None
    if "topic" in slot_type.lower() or "topic" in slot_name.lower():
        if not config.include_topics:
            return None
        entity_type = LabelType.TOPIC.value
    elif (
        "institution" in slot_type.lower()
        or "institution" in slot_name.lower()
        or "entity" in slot_type.lower()
    ):
        if not config.include_institutions:
            return None
        entity_type = LabelType.INSTITUTION.value
    elif "author" in slot_type.lower() or "author" in slot_name.lower():
        if not config.include_authors:
            return None
        entity_type = LabelType.AUTHOR.value
    elif "date" in slot_type.lower() or "date" in slot_name.lower():
        if not config.include_dates:
            return None
        entity_type = LabelType.DATE_RANGE.value
    else:
        # Unknown slot type, skip
        return None

    # Find text span in user_text (case-insensitive search)
    text_span = value
    start_char = None
    end_char = None

    lower_user_text = user_text.lower()
    lower_value = value.lower()
    idx = lower_user_text.find(lower_value)
    if idx >= 0:
        # Found the span - use original case from user_text
        text_span = user_text[idx : idx + len(value)]
        start_char = idx
        end_char = idx + len(value)

    # Use source_id if available, otherwise generate from value
    entity_id = source_id if source_id else f"{entity_type}:{value.replace(' ', '_')}"

    return Label(
        entity_id=entity_id,
        entity_type=entity_type,
        text_span=text_span,
        start_char=start_char,
        end_char=end_char,
    )


class EnrichmentGenerator:
    """Generator for enrichment labels from pairs.

    Extracts structured labels from filled slots in validated pairs
    for training entity/topic extraction models.
    """

    def __init__(self, config: EnrichmentGeneratorConfig | None = None) -> None:
        """Initialize the enrichment generator.

        Args:
            config: Generator configuration
        """
        self.config = config or EnrichmentGeneratorConfig()
        self._stats = EnrichmentGeneratorStats()

    def generate_from_pair(self, pair: Pair) -> EnrichmentLabel | None:
        """Generate enrichment labels from a single pair.

        Args:
            pair: The validated pair to extract labels from

        Returns:
            EnrichmentLabel with extracted labels, or None if no labels extracted
        """
        self._stats.pairs_processed += 1

        labels: list[dict] = []

        # Extract labels from each filled slot
        for slot_name, slot_data in pair.filled_slots.items():
            label = extract_label_from_slot(
                slot_name, slot_data, pair.user_text, self.config
            )
            if label:
                labels.append(label.to_dict())

                # Update stats by type
                if label.entity_type == LabelType.TOPIC.value:
                    self._stats.topics_extracted += 1
                elif label.entity_type == LabelType.INSTITUTION.value:
                    self._stats.institutions_extracted += 1
                elif label.entity_type == LabelType.AUTHOR.value:
                    self._stats.authors_extracted += 1
                elif label.entity_type == LabelType.DATE_RANGE.value:
                    self._stats.dates_extracted += 1

        # Only generate example if we have labels
        if not labels:
            return None

        self._stats.labels_generated += len(labels)
        self._stats.examples_generated += 1

        # Create enrichment label with provenance
        return EnrichmentLabel(
            example_id=generate_example_id(pair.pair_id),
            user_text=pair.user_text,
            labels=labels,
            provenance={
                "pair_id": pair.pair_id,
                "template_id": pair.template_id,
                "slot_mapping": {k: v for k, v in pair.filled_slots.items()},
            },
        )

    def generate_from_pairs(self, pairs: list[Pair]) -> list[EnrichmentLabel]:
        """Generate enrichment labels from multiple pairs.

        Args:
            pairs: List of validated pairs

        Returns:
            List of EnrichmentLabel objects
        """
        labels = []
        for pair in pairs:
            label = self.generate_from_pair(pair)
            if label:
                labels.append(label)
        return labels

    def generate_to_file(
        self,
        pairs: list[Pair],
        output_path: Path,
    ) -> tuple[str, int, EnrichmentGeneratorStats]:
        """Generate enrichment labels and write to JSONL file.

        Args:
            pairs: List of validated pairs
            output_path: Path to output enrichment_labels.jsonl file

        Returns:
            Tuple of (SHA256 checksum, line count, stats)
        """
        # Reset stats
        self._stats = EnrichmentGeneratorStats()

        with JSONLWriter(output_path) as writer:
            for pair in pairs:
                label = self.generate_from_pair(pair)
                if label:
                    writer.write_line(label)

        return writer.checksum, writer.line_count, self._stats

    def generate_from_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> tuple[str, int, EnrichmentGeneratorStats]:
        """Load pairs from file and generate enrichment labels.

        Args:
            input_path: Path to input pairs.jsonl file
            output_path: Path to output enrichment_labels.jsonl file

        Returns:
            Tuple of (SHA256 checksum, line count, stats)
        """
        # Reset stats
        self._stats = EnrichmentGeneratorStats()

        reader = JSONLReader(input_path)
        pairs = [Pair.from_dict(d) for d in reader]

        return self.generate_to_file(pairs, output_path)

    @property
    def stats(self) -> EnrichmentGeneratorStats:
        """Get generation statistics."""
        return self._stats


def generate_enrichment_labels(
    pairs: list[Pair],
    output_path: Path,
    config: EnrichmentGeneratorConfig | None = None,
) -> tuple[str, int, EnrichmentGeneratorStats]:
    """Convenience function to generate enrichment labels.

    Args:
        pairs: List of validated pairs
        output_path: Path to output enrichment_labels.jsonl file
        config: Generator configuration

    Returns:
        Tuple of (SHA256 checksum, line count, stats)
    """
    generator = EnrichmentGenerator(config=config)
    return generator.generate_to_file(pairs, output_path)


def generate_enrichment_labels_from_file(
    input_path: Path,
    output_path: Path,
    config: EnrichmentGeneratorConfig | None = None,
) -> tuple[str, int, EnrichmentGeneratorStats]:
    """Convenience function to generate enrichment labels from file.

    Args:
        input_path: Path to input pairs.jsonl file
        output_path: Path to output enrichment_labels.jsonl file
        config: Generator configuration

    Returns:
        Tuple of (SHA256 checksum, line count, stats)
    """
    generator = EnrichmentGenerator(config=config)
    return generator.generate_from_file(input_path, output_path)
