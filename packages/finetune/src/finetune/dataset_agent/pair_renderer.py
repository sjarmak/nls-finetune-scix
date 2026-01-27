"""Pair renderer for dataset generation pipeline.

This module renders ADS queries from templates and produces training pairs.
It takes NLInput records and generates Pair records with canonical ADS queries.

Features:
- Render ads_query from template + filled slots
- Canonicalize query formatting (whitespace, parentheses, quoting)
- Track provenance (template_id, entity IDs)

Output: pairs/pairs.jsonl with Pair records
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetune.dataset_agent.schemas import NLInput, Pair, Template
from finetune.dataset_agent.writers import JSONLReader, JSONLWriter


@dataclass
class PairRendererConfig:
    """Configuration for pair rendering.

    Attributes:
        canonicalize_whitespace: Normalize whitespace in queries
        canonicalize_quotes: Ensure consistent quote formatting
        canonicalize_parentheses: Normalize parentheses spacing
        strip_trailing_whitespace: Remove trailing whitespace from queries
    """

    canonicalize_whitespace: bool = True
    canonicalize_quotes: bool = True
    canonicalize_parentheses: bool = True
    strip_trailing_whitespace: bool = True


@dataclass
class PairRendererStats:
    """Statistics from pair rendering.

    Attributes:
        inputs_processed: Number of NL inputs processed
        pairs_generated: Number of pairs successfully generated
        pairs_failed: Number of pairs that failed rendering
        templates_used: Number of unique templates used
    """

    inputs_processed: int = 0
    pairs_generated: int = 0
    pairs_failed: int = 0
    templates_used: int = 0


def canonicalize_query(query: str, config: PairRendererConfig | None = None) -> str:
    """Canonicalize an ADS query string.

    Applies the following normalizations:
    - Normalize whitespace (collapse multiple spaces, trim)
    - Normalize parentheses spacing (no space after open, before close)
    - Ensure consistent quote formatting

    Args:
        query: Raw ADS query string
        config: Renderer configuration. Uses defaults if None.

    Returns:
        Canonicalized query string
    """
    config = config or PairRendererConfig()
    result = query

    if config.canonicalize_whitespace:
        # Collapse multiple spaces into one
        result = re.sub(r" +", " ", result)
        # Trim leading/trailing whitespace
        result = result.strip()

    if config.canonicalize_parentheses:
        # Remove space after opening parenthesis
        result = re.sub(r"\(\s+", "(", result)
        # Remove space before closing parenthesis
        result = re.sub(r"\s+\)", ")", result)
        # Ensure space before opening parenthesis (except at start or after another open paren)
        result = re.sub(r"(\S)\((?!\()", r"\1 (", result)
        # But don't add space after field: or operator(
        result = re.sub(r"(\w): \(", r"\1:(", result)
        # Restore operator parens (citations(, references(, etc.)
        operators = [
            "citations",
            "references",
            "trending",
            "similar",
            "useful",
            "reviews",
            "topn",
            "pos",
        ]
        for op in operators:
            result = re.sub(rf"{op} \(", f"{op}(", result)

    if config.canonicalize_quotes:
        # Normalize curly quotes to straight quotes
        result = result.replace(""", '"').replace(""", '"')
        result = result.replace("'", "'").replace("'", "'")

    if config.strip_trailing_whitespace:
        result = result.rstrip()

    return result


def render_ads_query(
    template: Template,
    filled_slots: dict[str, Any],
    config: PairRendererConfig | None = None,
) -> str:
    """Render an ADS query from a template and filled slots.

    Args:
        template: Template with ads_query_template
        filled_slots: Dict mapping slot names to slot data (with 'value' key)
        config: Renderer configuration

    Returns:
        Rendered and canonicalized ADS query string
    """
    query = template.ads_query_template

    # Fill in slot values
    for slot_name, slot_data in filled_slots.items():
        # slot_data can be a dict with 'value' key or just a string
        if isinstance(slot_data, dict):
            value = slot_data.get("value", "")
        else:
            value = str(slot_data)
        query = query.replace(f"{{{slot_name}}}", value)

    # Canonicalize the result
    return canonicalize_query(query, config)


def generate_pair_id(input_id: str, template_id: str) -> str:
    """Generate a unique pair ID from input and template IDs.

    Args:
        input_id: The NL input ID
        template_id: The template ID

    Returns:
        Unique pair ID
    """
    # Create deterministic hash-based ID
    base = f"{input_id}:{template_id}"
    hash_val = hashlib.md5(base.encode()).hexdigest()[:8]
    return f"pair_{template_id}_{hash_val}"


class PairRenderer:
    """Renderer for NL input to ADS query pairs.

    This class takes NL inputs and their associated templates to produce
    Pair records with rendered ADS queries.
    """

    def __init__(
        self,
        templates: dict[str, Template],
        config: PairRendererConfig | None = None,
    ) -> None:
        """Initialize the pair renderer.

        Args:
            templates: Dict mapping template IDs to Template objects
            config: Renderer configuration
        """
        self.templates = templates
        self.config = config or PairRendererConfig()
        self._stats = PairRendererStats()
        self._templates_used: set[str] = set()

    def render_pair(self, nl_input: NLInput) -> Pair | None:
        """Render a single NL input to a Pair.

        Args:
            nl_input: The NL input to render

        Returns:
            Rendered Pair, or None if template not found
        """
        self._stats.inputs_processed += 1

        template = self.templates.get(nl_input.template_id)
        if template is None:
            self._stats.pairs_failed += 1
            return None

        self._templates_used.add(nl_input.template_id)

        # Render the ADS query
        ads_query = render_ads_query(template, nl_input.filled_slots, self.config)

        # Generate pair ID
        pair_id = generate_pair_id(nl_input.input_id, nl_input.template_id)

        # Create the pair with provenance
        pair = Pair(
            pair_id=pair_id,
            user_text=nl_input.user_text,
            ads_query=ads_query,
            template_id=nl_input.template_id,
            filled_slots=nl_input.filled_slots,
            validation_tier=0,
            validation_errors=[],
        )

        self._stats.pairs_generated += 1
        return pair

    def render_pairs(self, nl_inputs: list[NLInput]) -> Iterator[Pair]:
        """Render multiple NL inputs to pairs.

        Args:
            nl_inputs: List of NL inputs to render

        Yields:
            Pair objects (skips inputs where template not found)
        """
        for nl_input in nl_inputs:
            pair = self.render_pair(nl_input)
            if pair is not None:
                yield pair

    def render_to_file(
        self,
        nl_inputs: list[NLInput],
        output_path: Path,
    ) -> tuple[str, int, PairRendererStats]:
        """Render NL inputs and write pairs to JSONL file.

        Args:
            nl_inputs: List of NL inputs to render
            output_path: Path to output pairs.jsonl file

        Returns:
            Tuple of (SHA256 checksum, line count, render stats)
        """
        # Reset stats
        self._stats = PairRendererStats()
        self._templates_used = set()

        with JSONLWriter(output_path) as writer:
            for pair in self.render_pairs(nl_inputs):
                writer.write_line(pair)

        # Update final stats
        self._stats.templates_used = len(self._templates_used)

        return writer.checksum, writer.line_count, self._stats

    def render_from_inputs_file(
        self,
        inputs_path: Path,
        output_path: Path,
    ) -> tuple[str, int, PairRendererStats]:
        """Load NL inputs from file and render to pairs file.

        Args:
            inputs_path: Path to inputs.jsonl file
            output_path: Path to output pairs.jsonl file

        Returns:
            Tuple of (SHA256 checksum, line count, render stats)
        """
        # Reset stats
        self._stats = PairRendererStats()
        self._templates_used = set()

        reader = JSONLReader(inputs_path)

        with JSONLWriter(output_path) as writer:
            for input_dict in reader:
                nl_input = NLInput.from_dict(input_dict)
                pair = self.render_pair(nl_input)
                if pair is not None:
                    writer.write_line(pair)

        # Update final stats
        self._stats.templates_used = len(self._templates_used)

        return writer.checksum, writer.line_count, self._stats

    @property
    def stats(self) -> PairRendererStats:
        """Get rendering statistics."""
        return self._stats


def render_pairs(
    nl_inputs: list[NLInput],
    templates: dict[str, Template],
    output_path: Path,
    config: PairRendererConfig | None = None,
) -> tuple[str, int, PairRendererStats]:
    """Convenience function to render NL inputs to pairs.

    Args:
        nl_inputs: List of NL inputs to render
        templates: Dict mapping template IDs to Template objects
        output_path: Path to output pairs.jsonl file
        config: Renderer configuration

    Returns:
        Tuple of (SHA256 checksum, line count, render stats)
    """
    renderer = PairRenderer(templates=templates, config=config)
    return renderer.render_to_file(nl_inputs, output_path)


def render_pairs_from_file(
    inputs_path: Path,
    templates: dict[str, Template],
    output_path: Path,
    config: PairRendererConfig | None = None,
) -> tuple[str, int, PairRendererStats]:
    """Convenience function to render pairs from input file.

    Args:
        inputs_path: Path to inputs.jsonl file
        templates: Dict mapping template IDs to Template objects
        output_path: Path to output pairs.jsonl file
        config: Renderer configuration

    Returns:
        Tuple of (SHA256 checksum, line count, render stats)
    """
    renderer = PairRenderer(templates=templates, config=config)
    return renderer.render_from_inputs_file(inputs_path, output_path)
