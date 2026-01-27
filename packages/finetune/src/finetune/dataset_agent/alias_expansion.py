"""Alias expansion utilities for topic and entity catalogs.

This module provides consistent alias generation for real-world spelling and
formatting variants. It applies expansion rules to existing aliases and generates
new variants based on configurable rules.

Expansion rules:
- Case variants: lowercase, uppercase, title case
- Hyphen/space swaps: "black-hole" <-> "black hole"
- Punctuation stripping: "M87*" -> "M87"
- Acronym handling: Extract potential acronyms from multi-word names

The expansion is deterministic - running on the same input produces the same output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from finetune.dataset_agent.schemas import EntityEntry, TopicEntry

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class AliasExpansionConfig:
    """Configuration for alias expansion rules.

    Attributes:
        max_aliases: Maximum number of aliases per entry (including originals)
        enable_case_variants: Generate lowercase, uppercase, title case variants
        enable_hyphen_space_swap: Convert hyphens to spaces and vice versa
        enable_punctuation_strip: Remove punctuation marks
        enable_acronym_extraction: Extract acronyms from multi-word names
        min_acronym_words: Minimum words required to generate acronym
        max_acronym_length: Maximum length of generated acronyms
        strip_diacritics: Convert accented characters to ASCII equivalents
    """

    max_aliases: int = 50
    enable_case_variants: bool = True
    enable_hyphen_space_swap: bool = True
    enable_punctuation_strip: bool = True
    enable_acronym_extraction: bool = True
    min_acronym_words: int = 2
    max_acronym_length: int = 10
    strip_diacritics: bool = True


# Default configuration
DEFAULT_CONFIG = AliasExpansionConfig()


def normalize_for_comparison(s: str) -> str:
    """Normalize a string for case-insensitive comparison.

    Args:
        s: String to normalize

    Returns:
        Lowercased, whitespace-normalized string
    """
    return " ".join(s.lower().split())


def strip_diacritics(s: str) -> str:
    """Remove diacritical marks from a string.

    Converts accented characters to their ASCII equivalents.
    E.g., "café" -> "cafe", "naïve" -> "naive"

    Args:
        s: String with potential diacritics

    Returns:
        String with diacritics removed
    """
    # Normalize to decomposed form (separates base char from combining marks)
    normalized = unicodedata.normalize("NFD", s)
    # Remove combining characters (diacritics)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    return stripped


def generate_case_variants(s: str) -> list[str]:
    """Generate case variants of a string.

    Args:
        s: Input string

    Returns:
        List of unique case variants: [lowercase, uppercase, title case]
    """
    variants = []
    lower = s.lower()
    upper = s.upper()
    title = s.title()

    # Only add variants that differ from original
    seen = {s}
    for v in [lower, upper, title]:
        if v not in seen:
            variants.append(v)
            seen.add(v)

    return variants


def swap_hyphen_space(s: str) -> list[str]:
    """Generate hyphen/space swap variants.

    "black-hole" -> "black hole"
    "black hole" -> "black-hole"

    Args:
        s: Input string

    Returns:
        List of swap variants (may be empty if no hyphens/multi-word spaces)
    """
    variants = []

    # Hyphen to space
    if "-" in s:
        space_variant = s.replace("-", " ")
        if space_variant != s:
            variants.append(space_variant)

    # Space to hyphen (only for multi-word strings)
    words = s.split()
    if len(words) > 1:
        hyphen_variant = "-".join(words)
        if hyphen_variant != s and hyphen_variant not in variants:
            variants.append(hyphen_variant)

    return variants


def strip_punctuation(s: str) -> str | None:
    """Strip punctuation from a string.

    Removes: asterisks, apostrophes, quotes, periods, commas, etc.
    Preserves: hyphens (handled separately), alphanumerics, spaces

    Args:
        s: Input string

    Returns:
        String with punctuation removed, or None if result is same as input
    """
    # Keep alphanumeric, spaces, and hyphens
    stripped = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    # Collapse multiple spaces
    stripped = " ".join(stripped.split())

    if stripped and stripped != s:
        return stripped
    return None


def extract_acronym(s: str, min_words: int = 2, max_length: int = 10) -> str | None:
    """Extract an acronym from a multi-word name.

    Takes first letter of each capitalized word or significant word.

    Args:
        s: Input string (e.g., "National Aeronautics and Space Administration")
        min_words: Minimum words required to generate acronym
        max_length: Maximum length of generated acronym

    Returns:
        Acronym string (e.g., "NASA") or None if not applicable
    """
    # Split into words
    words = s.split()

    if len(words) < min_words:
        return None

    # Skip common articles and prepositions
    skip_words = {"a", "an", "and", "the", "of", "for", "in", "on", "at", "to", "by"}

    # Extract first letters from significant words
    letters = []
    for word in words:
        if word.lower() not in skip_words:
            # Use first character, uppercased
            if word:
                letters.append(word[0].upper())

    if len(letters) < min_words:
        return None

    acronym = "".join(letters)

    # Check length constraints
    if len(acronym) > max_length or len(acronym) < 2:
        return None

    return acronym


def is_already_acronym(s: str) -> bool:
    """Check if a string is already an acronym (all caps, short).

    Args:
        s: String to check

    Returns:
        True if string appears to be an acronym
    """
    # Remove spaces and check
    cleaned = s.replace(" ", "").replace("-", "")
    # All uppercase and reasonably short
    return len(cleaned) <= 10 and cleaned.isupper() and cleaned.isalpha()


def expand_alias(
    alias: str,
    config: AliasExpansionConfig | None = None,
) -> list[str]:
    """Expand a single alias into multiple variants.

    Args:
        alias: Original alias string
        config: Expansion configuration (uses DEFAULT_CONFIG if None)

    Returns:
        List of expanded aliases (including original)
    """
    if config is None:
        config = DEFAULT_CONFIG

    variants: list[str] = [alias]
    seen: set[str] = {normalize_for_comparison(alias)}

    def add_variant(v: str) -> None:
        """Add a variant if not already present."""
        if not v or not v.strip():
            return
        normalized = normalize_for_comparison(v)
        if normalized not in seen:
            variants.append(v)
            seen.add(normalized)

    # Strip diacritics first (can generate new base variants)
    if config.strip_diacritics:
        stripped = strip_diacritics(alias)
        if stripped != alias:
            add_variant(stripped)
            # Also process the stripped version
            variants_to_process = [alias, stripped]
        else:
            variants_to_process = [alias]
    else:
        variants_to_process = [alias]

    # Process each base variant
    for base in variants_to_process:
        # Case variants
        if config.enable_case_variants:
            for case_var in generate_case_variants(base):
                add_variant(case_var)

        # Hyphen/space swaps
        if config.enable_hyphen_space_swap:
            for swap_var in swap_hyphen_space(base):
                add_variant(swap_var)
                # Also add case variants of swapped
                if config.enable_case_variants:
                    for case_var in generate_case_variants(swap_var):
                        add_variant(case_var)

        # Punctuation stripping
        if config.enable_punctuation_strip:
            stripped_punc = strip_punctuation(base)
            if stripped_punc:
                add_variant(stripped_punc)
                # Also add case variants of stripped
                if config.enable_case_variants:
                    for case_var in generate_case_variants(stripped_punc):
                        add_variant(case_var)

        # Acronym extraction (only if not already an acronym)
        if config.enable_acronym_extraction and not is_already_acronym(base):
            acronym = extract_acronym(
                base,
                min_words=config.min_acronym_words,
                max_length=config.max_acronym_length,
            )
            if acronym:
                add_variant(acronym)
                # Lowercase acronym variant
                add_variant(acronym.lower())

    return variants


def expand_aliases(
    aliases: list[str],
    preferred_label: str,
    config: AliasExpansionConfig | None = None,
) -> list[str]:
    """Expand a list of aliases, including the preferred label.

    Generates variants for both the preferred label and all existing aliases,
    then deduplicates and limits to max_aliases.

    Args:
        aliases: Existing aliases
        preferred_label: The primary/preferred label
        config: Expansion configuration

    Returns:
        Expanded and deduplicated list of aliases (excludes preferred_label)
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Track normalized forms we've seen (for deduplication)
    seen: set[str] = set()

    # The preferred label is excluded from the alias list
    pref_normalized = normalize_for_comparison(preferred_label)
    seen.add(pref_normalized)

    result: list[str] = []

    def add_if_new(alias: str) -> None:
        """Add alias if it's new and not the preferred label."""
        normalized = normalize_for_comparison(alias)
        if normalized not in seen and alias.strip():
            seen.add(normalized)
            result.append(alias)

    # Expand the preferred label to generate aliases
    for variant in expand_alias(preferred_label, config):
        if normalize_for_comparison(variant) != pref_normalized:
            add_if_new(variant)

    # Expand each existing alias
    for alias in aliases:
        # Add the original first (preserves case from source)
        add_if_new(alias)
        # Then add variants
        for variant in expand_alias(alias, config):
            add_if_new(variant)

    # Enforce max_aliases limit
    if len(result) > config.max_aliases:
        result = result[: config.max_aliases]

    return result


def expand_topic_entry(
    entry: TopicEntry,
    config: AliasExpansionConfig | None = None,
) -> TopicEntry:
    """Expand aliases for a TopicEntry.

    Args:
        entry: Topic entry to expand
        config: Expansion configuration

    Returns:
        New TopicEntry with expanded aliases
    """
    expanded_aliases = expand_aliases(entry.aliases, entry.label, config)

    return TopicEntry(
        id=entry.id,
        label=entry.label,
        aliases=expanded_aliases,
        parents=entry.parents,
        children=entry.children,
        source_id=entry.source_id,
    )


def expand_entity_entry(
    entry: EntityEntry,
    config: AliasExpansionConfig | None = None,
) -> EntityEntry:
    """Expand aliases for an EntityEntry.

    Args:
        entry: Entity entry to expand
        config: Expansion configuration

    Returns:
        New EntityEntry with expanded aliases
    """
    expanded_aliases = expand_aliases(entry.aliases, entry.label, config)

    return EntityEntry(
        id=entry.id,
        label=entry.label,
        aliases=expanded_aliases,
        metadata=entry.metadata,
        source_id=entry.source_id,
    )


def expand_topic_catalog(
    entries: Iterator[TopicEntry],
    config: AliasExpansionConfig | None = None,
) -> Iterator[TopicEntry]:
    """Expand aliases for all entries in a topic catalog.

    Args:
        entries: Iterator of TopicEntry objects
        config: Expansion configuration

    Yields:
        TopicEntry objects with expanded aliases
    """
    for entry in entries:
        yield expand_topic_entry(entry, config)


def expand_entity_catalog(
    entries: Iterator[EntityEntry],
    config: AliasExpansionConfig | None = None,
) -> Iterator[EntityEntry]:
    """Expand aliases for all entries in an entity catalog.

    Args:
        entries: Iterator of EntityEntry objects
        config: Expansion configuration

    Yields:
        EntityEntry objects with expanded aliases
    """
    for entry in entries:
        yield expand_entity_entry(entry, config)


@dataclass
class AliasExpansionStats:
    """Statistics from alias expansion.

    Attributes:
        entries_processed: Total entries processed
        original_aliases: Total aliases before expansion
        expanded_aliases: Total aliases after expansion
        aliases_added: Net new aliases added
        entries_with_new_aliases: Entries that gained at least one new alias
    """

    entries_processed: int = 0
    original_aliases: int = 0
    expanded_aliases: int = 0
    aliases_added: int = 0
    entries_with_new_aliases: int = 0


def expand_topic_catalog_to_file(
    input_path: str,
    output_path: str,
    config: AliasExpansionConfig | None = None,
) -> tuple[int, str, AliasExpansionStats]:
    """Read topic catalog, expand aliases, and write to new file.

    Args:
        input_path: Path to input topic_catalog.jsonl
        output_path: Path for output expanded_topic_catalog.jsonl
        config: Expansion configuration

    Returns:
        Tuple of (entry count, SHA256 checksum, expansion stats)
    """
    from pathlib import Path

    from finetune.dataset_agent.writers import JSONLReader, JSONLWriter

    if config is None:
        config = DEFAULT_CONFIG

    stats = AliasExpansionStats()

    reader = JSONLReader(Path(input_path))
    writer = JSONLWriter(Path(output_path))

    with writer:
        for line_data in reader:
            entry = TopicEntry.from_dict(line_data)
            original_count = len(entry.aliases)

            expanded = expand_topic_entry(entry, config)
            expanded_count = len(expanded.aliases)

            stats.entries_processed += 1
            stats.original_aliases += original_count
            stats.expanded_aliases += expanded_count
            if expanded_count > original_count:
                stats.entries_with_new_aliases += 1

            writer.write_line(expanded)

    stats.aliases_added = stats.expanded_aliases - stats.original_aliases

    return stats.entries_processed, writer.checksum, stats


def expand_entity_catalog_to_file(
    input_path: str,
    output_path: str,
    config: AliasExpansionConfig | None = None,
) -> tuple[int, str, AliasExpansionStats]:
    """Read entity catalog, expand aliases, and write to new file.

    Args:
        input_path: Path to input entity_catalog.jsonl
        output_path: Path for output expanded_entity_catalog.jsonl
        config: Expansion configuration

    Returns:
        Tuple of (entry count, SHA256 checksum, expansion stats)
    """
    from pathlib import Path

    from finetune.dataset_agent.writers import JSONLReader, JSONLWriter

    if config is None:
        config = DEFAULT_CONFIG

    stats = AliasExpansionStats()

    reader = JSONLReader(Path(input_path))
    writer = JSONLWriter(Path(output_path))

    with writer:
        for line_data in reader:
            entry = EntityEntry.from_dict(line_data)
            original_count = len(entry.aliases)

            expanded = expand_entity_entry(entry, config)
            expanded_count = len(expanded.aliases)

            stats.entries_processed += 1
            stats.original_aliases += original_count
            stats.expanded_aliases += expanded_count
            if expanded_count > original_count:
                stats.entries_with_new_aliases += 1

            writer.write_line(expanded)

    stats.aliases_added = stats.expanded_aliases - stats.original_aliases

    return stats.entries_processed, writer.checksum, stats
