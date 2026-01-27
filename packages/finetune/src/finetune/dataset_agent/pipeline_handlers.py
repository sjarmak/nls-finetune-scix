"""Pipeline stage handlers for dataset generation.

This module implements handler functions for each pipeline stage.
Each handler takes (run_dir: Path, config: PipelineConfig) and returns list[Path] of outputs.

Stages:
1. fetch - Download data sources (UAT, ROR)
2. normalize - Normalize raw data to catalog format
3. expand_aliases - Expand alias variants in catalogs
4. load_templates - Load and validate templates
5. generate_inputs - Generate NL inputs from templates
6. render_pairs - Render ADS queries from inputs
7. validate_local - Tier 1 local syntax validation
8. validate_backend - Tier 2/3 backend validation (optional)
9. generate_enrichment - Generate enrichment labels
10. report - Generate reports and manifest checksums
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune.dataset_agent.alias_expansion import (
    AliasExpansionConfig,
    expand_entity_catalog_to_file,
    expand_topic_catalog_to_file,
)
from finetune.dataset_agent.backend_validator import (
    BackendValidator,
    BackendValidatorConfig,
    ValidationMode,
)
from finetune.dataset_agent.enrichment_generator import (
    EnrichmentGenerator,
    EnrichmentGeneratorConfig,
)
from finetune.dataset_agent.gcmd_normalizer import normalize_gcmd_to_catalog
from finetune.dataset_agent.git_downloader import (
    GitDownloader,
    create_source_entry_from_git_download,
)
from finetune.dataset_agent.http_downloader import (
    HTTPDownloader,
    create_source_entry_from_download,
)
from finetune.dataset_agent.input_generator import (
    InputGenerator,
    InputGeneratorConfig,
)
from finetune.dataset_agent.local_validator import (
    LocalValidator,
    LocalValidatorConfig,
)
from finetune.dataset_agent.pair_renderer import (
    PairRenderer,
    PairRendererConfig,
)
from finetune.dataset_agent.planetary_normalizer import normalize_planetary_to_catalog
from finetune.dataset_agent.report_generator import (
    ReportGenerator,
    ReportGeneratorConfig,
)
from finetune.dataset_agent.ror_normalizer import normalize_ror_to_catalog
from finetune.dataset_agent.schemas import (
    RunManifest,
    SourceType,
    Stage,
    Template,
)
from finetune.dataset_agent.sources import (
    SourcesConfig,
    create_source_manifest,
    load_sources_config,
    save_source_manifest,
)
from finetune.dataset_agent.sweet_normalizer import normalize_sweet_to_catalog
from finetune.dataset_agent.template_loader import (
    get_default_templates_dir,
    load_templates_from_directory,
)
from finetune.dataset_agent.uat_normalizer import normalize_uat_to_catalog

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline.

    Attributes:
        sources_config_path: Path to sources.yaml
        templates_dir: Path to templates directory (uses default if None)
        seed: Random seed for deterministic generation
        samples_per_template: Number of samples per template (0 = unlimited)
        skip_backend: Skip backend validation stage
        enable_backend_validation: Enable online backend validation (vs offline)
        alias_expansion_config: Config for alias expansion
        input_generator_config: Config for input generation
    """

    sources_config_path: Path | None = None
    templates_dir: Path | None = None
    seed: int | None = 42
    samples_per_template: int = 100
    skip_backend: bool = False
    enable_backend_validation: bool = True
    alias_expansion_config: AliasExpansionConfig | None = None
    input_generator_config: InputGeneratorConfig | None = None


@dataclass
class PipelineState:
    """Shared state across pipeline stages.

    This stores intermediate results that need to be passed between stages.
    """

    sources_config: SourcesConfig | None = None
    source_entries: list[Any] = field(default_factory=list)
    topics_count: int = 0
    entities_count: int = 0
    entries_by_source: dict[str, int] = field(default_factory=dict)
    entries_by_domain: dict[str, int] = field(default_factory=dict)
    templates: list[Template] = field(default_factory=list)
    templates_by_id: dict[str, Template] = field(default_factory=dict)
    inputs_count: int = 0
    pairs_valid_count: int = 0
    pairs_quarantined_count: int = 0
    enrichment_count: int = 0
    backend_pass_rate: float | None = None
    errors_by_type: dict[str, int] = field(default_factory=dict)
    pinned_revisions: dict[str, str] = field(default_factory=dict)


def handle_fetch(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 1: Fetch data sources (UAT, ROR).

    Downloads configured sources into raw/ directory.
    Updates source_manifest.json with actual checksums and metadata.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of downloaded file paths
    """
    logger.info("Stage: fetch - Downloading data sources")

    raw_dir = run_dir / "raw"
    manifests_dir = run_dir / "manifests"
    outputs: list[Path] = []

    # Load sources config
    if config.sources_config_path is None:
        # Use default sources.yaml from package
        default_sources = Path(__file__).parent / "config" / "sources.yaml"
        if not default_sources.exists():
            raise FileNotFoundError(f"Default sources.yaml not found: {default_sources}")
        sources_config = load_sources_config(default_sources)
    else:
        sources_config = load_sources_config(config.sources_config_path)

    state.sources_config = sources_config
    logger.info(f"Loaded {len(sources_config.sources)} source(s)")

    # Set up git cache directory
    git_cache_dir = run_dir.parent / ".git_cache"
    git_cache_dir.mkdir(parents=True, exist_ok=True)

    # Download each source
    http_downloader = HTTPDownloader()
    git_downloader = GitDownloader(cache_dir=git_cache_dir)

    source_entries = []

    for source in sources_config.sources:
        logger.info(f"Downloading source: {source.id} ({source.type.value})")

        if source.type == SourceType.HTTP:
            result = http_downloader.download(source, raw_dir)
            entry = create_source_entry_from_download(source, result, raw_dir)
            outputs.append(result.local_path)
        elif source.type == SourceType.GIT:
            result = git_downloader.download(source, raw_dir)
            entry = create_source_entry_from_git_download(source, result, raw_dir)
            outputs.append(result.local_path)
            # Track pinned revisions for reproduce info
            if result.resolved_commit:
                state.pinned_revisions[source.id] = result.resolved_commit
        else:
            logger.warning(f"Unknown source type: {source.type}")
            continue

        source_entries.append(entry)
        logger.info(f"  Downloaded: {entry.local_path} ({entry.checksum_sha256[:16] if entry.checksum_sha256 else 'no checksum'}...)")

    state.source_entries = source_entries

    # Update source manifest with actual metadata
    manifest = create_source_manifest(source_entries)
    manifest_path = manifests_dir / "source_manifest.json"
    save_source_manifest(manifest, manifest_path)
    outputs.append(manifest_path)

    logger.info(f"Fetch complete: {len(outputs)} files downloaded")
    return outputs


def _resolve_normalizer(source_id: str, normalizer_field: str | None) -> str | None:
    """Resolve the normalizer identifier for a source.

    Falls back to the source ID if no explicit normalizer field is set.

    Args:
        source_id: The source's unique identifier
        normalizer_field: The optional 'normalizer' field from source config

    Returns:
        Normalizer identifier string, or None if unrecognized
    """
    return normalizer_field or source_id


def _find_raw_file(raw_dir: Path, source_id: str) -> Path | None:
    """Find the raw downloaded file for a source.

    Checks common archive formats in priority order.

    Args:
        raw_dir: Directory containing raw downloaded files
        source_id: Source identifier used in filenames

    Returns:
        Path to the raw file, or None if not found
    """
    candidates = [
        raw_dir / f"{source_id}.tar.gz",
        raw_dir / f"{source_id}.zip",
        raw_dir / f"{source_id}.json",
        raw_dir / f"{source_id}.rdf",
        raw_dir / f"{source_id}.ttl",
        raw_dir / f"{source_id}.shp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _dispatch_normalizer(
    normalizer_id: str,
    raw_path: Path,
    output_path: Path,
    source_id: str,
) -> tuple[int, str] | None:
    """Dispatch to the correct normalizer function.

    Args:
        normalizer_id: Normalizer identifier (e.g., 'uat', 'ror')
        raw_path: Path to the raw source file
        output_path: Path for normalized output catalog
        source_id: Source identifier for provenance

    Returns:
        Tuple of (entry count, checksum), or None if normalizer not recognized
    """
    if normalizer_id == "uat":
        return normalize_uat_to_catalog(raw_path, output_path, source_id=source_id)
    elif normalizer_id == "ror":
        return normalize_ror_to_catalog(
            raw_path, output_path, source_id=source_id, active_only=True
        )
    elif normalizer_id == "sweet":
        return normalize_sweet_to_catalog(raw_path, output_path, source_id=source_id)
    elif normalizer_id == "gcmd":
        return normalize_gcmd_to_catalog(raw_path, output_path, source_id=source_id)
    elif normalizer_id == "planetary":
        return normalize_planetary_to_catalog(raw_path, output_path, source_id=source_id)
    else:
        return None


def _catalog_type_for_normalizer(normalizer_id: str) -> str:
    """Determine catalog type (topic or entity) for a normalizer.

    Args:
        normalizer_id: Normalizer identifier

    Returns:
        'topic' or 'entity'
    """
    topic_normalizers = {"uat", "sweet", "gcmd"}
    if normalizer_id in topic_normalizers:
        return "topic"
    return "entity"


def _catalog_filename(normalizer_id: str, source_id: str) -> str:
    """Generate a per-source catalog filename.

    Args:
        normalizer_id: Normalizer identifier
        source_id: Source identifier

    Returns:
        Filename like 'topic_catalog_uat.jsonl' or 'entity_catalog_ror.jsonl'
    """
    catalog_type = _catalog_type_for_normalizer(normalizer_id)
    return f"{catalog_type}_catalog_{source_id}.jsonl"


def _merge_catalogs(
    source_catalog_paths: list[Path],
    merged_output_path: Path,
) -> tuple[int, str]:
    """Merge multiple per-source catalog files into a single unified catalog.

    Args:
        source_catalog_paths: List of per-source catalog JSONL file paths
        merged_output_path: Path for the merged output file

    Returns:
        Tuple of (total entry count, checksum of merged file)
    """
    from finetune.dataset_agent.writers import JSONLWriter

    writer = JSONLWriter(merged_output_path)
    count = 0

    with writer:
        for catalog_path in source_catalog_paths:
            if not catalog_path.exists():
                continue
            with open(catalog_path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        entry = json.loads(stripped)
                        writer.write_line(entry)
                        count += 1

    return count, writer.checksum


def handle_normalize(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 2: Normalize raw data to catalog format.

    Dispatches to the correct normalizer for each configured source based on
    the 'normalizer' field (or source ID as fallback). Each normalizer writes
    its own per-source catalog file. After all sources are processed, per-source
    catalogs are merged into unified topic_catalog.jsonl and entity_catalog.jsonl.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of catalog file paths
    """
    logger.info("Stage: normalize - Normalizing raw data to catalogs")

    raw_dir = run_dir / "raw"
    normalized_dir = run_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    topic_catalog_paths: list[Path] = []
    entity_catalog_paths: list[Path] = []

    # Dispatch normalization for each configured source
    sources = state.sources_config.sources if state.sources_config else []
    for source in sources:
        normalizer_id = _resolve_normalizer(source.id, source.normalizer)
        if normalizer_id is None:
            logger.warning(f"No normalizer for source: {source.id}")
            continue

        raw_path = _find_raw_file(raw_dir, source.id)
        if raw_path is None:
            logger.warning(f"Raw file not found for source: {source.id}")
            continue

        catalog_filename = _catalog_filename(normalizer_id, source.id)
        catalog_path = normalized_dir / catalog_filename

        logger.info(f"Normalizing {source.id} (normalizer={normalizer_id}) to {catalog_path}")

        try:
            result = _dispatch_normalizer(normalizer_id, raw_path, catalog_path, source.id)
        except Exception as e:
            logger.warning(f"Normalization failed for {source.id} (continuing): {e}")
            continue

        if result is None:
            logger.warning(f"Unknown normalizer '{normalizer_id}' for source: {source.id}")
            continue

        count, checksum = result
        outputs.append(catalog_path)
        logger.info(f"  {count} entries normalized (checksum: {checksum[:16]}...)")

        # Track per-source counts
        state.entries_by_source[source.id] = count

        catalog_type = _catalog_type_for_normalizer(normalizer_id)
        if catalog_type == "topic":
            topic_catalog_paths.append(catalog_path)
            state.topics_count += count
        else:
            entity_catalog_paths.append(catalog_path)
            state.entities_count += count

    # Merge per-source catalogs into unified files
    if topic_catalog_paths:
        merged_topic_path = normalized_dir / "topic_catalog.jsonl"
        logger.info(f"Merging {len(topic_catalog_paths)} topic catalog(s) into {merged_topic_path}")
        count, checksum = _merge_catalogs(topic_catalog_paths, merged_topic_path)
        outputs.append(merged_topic_path)
        logger.info(f"  Merged topic catalog: {count} entries (checksum: {checksum[:16]}...)")

    if entity_catalog_paths:
        merged_entity_path = normalized_dir / "entity_catalog.jsonl"
        logger.info(f"Merging {len(entity_catalog_paths)} entity catalog(s) into {merged_entity_path}")
        count, checksum = _merge_catalogs(entity_catalog_paths, merged_entity_path)
        outputs.append(merged_entity_path)
        logger.info(f"  Merged entity catalog: {count} entries (checksum: {checksum[:16]}...)")

    # Tally per-domain counts from merged catalogs
    domain_counts: dict[str, int] = {}
    for catalog_path in [normalized_dir / "topic_catalog.jsonl", normalized_dir / "entity_catalog.jsonl"]:
        if catalog_path.exists():
            with open(catalog_path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        entry = json.loads(stripped)
                        for tag in entry.get("domain_tags", []):
                            domain_counts[tag] = domain_counts.get(tag, 0) + 1
    state.entries_by_domain = domain_counts

    logger.info(f"Normalize complete: {len(outputs)} catalogs created")
    return outputs


def handle_expand_aliases(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 3: Expand alias variants in catalogs.

    Generates case variants, hyphen/space swaps, and acronyms.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of expanded catalog file paths
    """
    logger.info("Stage: expand_aliases - Expanding alias variants")

    normalized_dir = run_dir / "normalized"
    outputs: list[Path] = []

    expansion_config = config.alias_expansion_config or AliasExpansionConfig()

    # Expand topic catalog
    topic_input = normalized_dir / "topic_catalog.jsonl"
    if topic_input.exists():
        topic_output = normalized_dir / "expanded_topic_catalog.jsonl"
        logger.info(f"Expanding topic aliases to {topic_output}")
        count, checksum, stats = expand_topic_catalog_to_file(
            str(topic_input), str(topic_output), expansion_config
        )
        outputs.append(topic_output)
        logger.info(
            f"  {count} topics, {stats.aliases_added} aliases added "
            f"({stats.entries_with_new_aliases} entries gained aliases)"
        )

    # Expand entity catalog (unified or legacy institutions-only)
    entity_input = normalized_dir / "entity_catalog.jsonl"
    if not entity_input.exists():
        entity_input = normalized_dir / "entity_catalog_institutions.jsonl"
    if entity_input.exists():
        entity_output = normalized_dir / "expanded_entity_catalog.jsonl"
        logger.info(f"Expanding entity aliases to {entity_output}")
        count, checksum, stats = expand_entity_catalog_to_file(
            str(entity_input), str(entity_output), expansion_config
        )
        outputs.append(entity_output)
        logger.info(
            f"  {count} entities, {stats.aliases_added} aliases added "
            f"({stats.entries_with_new_aliases} entries gained aliases)"
        )

    logger.info(f"Alias expansion complete: {len(outputs)} catalogs expanded")
    return outputs


def handle_load_templates(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 4: Load and validate templates.

    Loads template YAML files from templates directory.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of template manifest paths
    """
    logger.info("Stage: load_templates - Loading and validating templates")

    templates_dir = config.templates_dir or get_default_templates_dir()
    manifests_dir = run_dir / "manifests"
    outputs: list[Path] = []

    logger.info(f"Loading templates from {templates_dir}")
    result = load_templates_from_directory(templates_dir)

    if result.errors:
        for error in result.errors:
            logger.warning(f"Template error: {error}")

    state.templates = result.templates
    state.templates_by_id = result.templates_by_id

    logger.info(f"Loaded {len(result.templates)} templates")

    # Write template manifest
    template_manifest = {
        "templates_dir": str(templates_dir),
        "templates_count": len(result.templates),
        "templates": [t.to_dict() for t in result.templates],
        "errors": [str(e) for e in result.errors],
    }
    template_manifest_path = manifests_dir / "template_manifest.json"
    with open(template_manifest_path, "w") as f:
        json.dump(template_manifest, f, indent=2)
    outputs.append(template_manifest_path)

    logger.info(f"Templates loaded: {len(result.templates)} templates, {len(result.errors)} errors")
    return outputs


def handle_generate_inputs(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 5: Generate NL inputs from templates.

    Fills templates with catalog values to create diverse NL inputs.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of input file paths
    """
    logger.info("Stage: generate_inputs - Generating NL inputs from templates")

    normalized_dir = run_dir / "normalized"
    pairs_dir = run_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # Configure input generator
    gen_config = config.input_generator_config or InputGeneratorConfig(
        seed=config.seed,
        samples_per_template=config.samples_per_template,
    )
    gen_config.seed = config.seed
    gen_config.samples_per_template = config.samples_per_template

    generator = InputGenerator(config=gen_config)

    # Load expanded catalogs (or fall back to regular catalogs)
    topic_catalog_path = normalized_dir / "expanded_topic_catalog.jsonl"
    if not topic_catalog_path.exists():
        topic_catalog_path = normalized_dir / "topic_catalog.jsonl"

    if topic_catalog_path.exists():
        logger.info(f"Loading topic catalog from {topic_catalog_path}")
        generator.load_topic_catalog(topic_catalog_path)

    entity_catalog_path = normalized_dir / "expanded_entity_catalog.jsonl"
    if not entity_catalog_path.exists():
        entity_catalog_path = normalized_dir / "expanded_entity_catalog_institutions.jsonl"
    if not entity_catalog_path.exists():
        entity_catalog_path = normalized_dir / "entity_catalog.jsonl"
    if not entity_catalog_path.exists():
        entity_catalog_path = normalized_dir / "entity_catalog_institutions.jsonl"

    if entity_catalog_path.exists():
        logger.info(f"Loading entity catalog from {entity_catalog_path}")
        generator.load_entity_catalog(entity_catalog_path, "institutions")

    # Generate inputs
    inputs_path = pairs_dir / "inputs.jsonl"
    logger.info(f"Generating inputs to {inputs_path}")
    checksum, count, stats = generator.generate_to_file(
        state.templates,
        inputs_path,
        max_per_template=config.samples_per_template if config.samples_per_template > 0 else None,
    )
    state.inputs_count = count
    outputs.append(inputs_path)

    logger.info(
        f"Generated {count} inputs "
        f"({stats.templates_processed} templates, {stats.topics_used} topics, "
        f"{stats.entities_used} entities, {stats.aliases_used} alias usages)"
    )
    return outputs


def handle_render_pairs(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 6: Render ADS queries from inputs.

    Creates NL input / ADS query pairs from generated inputs.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of pair file paths
    """
    logger.info("Stage: render_pairs - Rendering ADS queries from inputs")

    pairs_dir = run_dir / "pairs"
    outputs: list[Path] = []

    inputs_path = pairs_dir / "inputs.jsonl"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Inputs file not found: {inputs_path}")

    # Create renderer
    renderer = PairRenderer(
        templates=state.templates_by_id,
        config=PairRendererConfig(),
    )

    # Render pairs
    pairs_path = pairs_dir / "unvalidated_pairs.jsonl"
    logger.info(f"Rendering pairs to {pairs_path}")
    checksum, count, stats = renderer.render_from_inputs_file(inputs_path, pairs_path)
    outputs.append(pairs_path)

    logger.info(
        f"Rendered {count} pairs "
        f"({stats.templates_used} templates, {stats.pairs_failed} failed)"
    )
    return outputs


def handle_validate_local(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 7: Tier 1 local syntax validation.

    Validates queries locally for syntax, field names, and constraints.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of validated and quarantine file paths
    """
    logger.info("Stage: validate_local - Running Tier 1 local validation")

    pairs_dir = run_dir / "pairs"
    outputs: list[Path] = []

    unvalidated_path = pairs_dir / "unvalidated_pairs.jsonl"
    if not unvalidated_path.exists():
        raise FileNotFoundError(f"Unvalidated pairs file not found: {unvalidated_path}")

    validator = LocalValidator(config=LocalValidatorConfig())

    valid_path = pairs_dir / "local_validated_pairs.jsonl"
    quarantine_path = pairs_dir / "local_quarantine.jsonl"

    logger.info("Validating pairs locally")
    valid_cksum, valid_count, q_cksum, q_count, stats = validator.validate_from_file(
        unvalidated_path, valid_path, quarantine_path
    )
    outputs.extend([valid_path, quarantine_path])

    state.pairs_valid_count = valid_count
    state.pairs_quarantined_count = q_count
    if stats.errors_by_type:
        state.errors_by_type.update(stats.errors_by_type)

    # If skipping backend validation, copy local validated to final pairs.jsonl
    if config.skip_backend:
        import shutil
        final_pairs_path = pairs_dir / "pairs.jsonl"
        final_quarantine_path = pairs_dir / "quarantine.jsonl"
        shutil.copy(valid_path, final_pairs_path)
        shutil.copy(quarantine_path, final_quarantine_path)
        outputs.extend([final_pairs_path, final_quarantine_path])
        logger.info("  (Copied to final pairs.jsonl - backend validation skipped)")

    logger.info(
        f"Local validation: {valid_count} valid, {q_count} quarantined "
        f"({stats.pairs_processed} processed)"
    )
    return outputs


def handle_validate_backend(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 8: Tier 2/3 backend validation (optional).

    Validates queries against ADS API for syntax and results.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of validated and quarantine file paths
    """
    logger.info("Stage: validate_backend - Running Tier 2/3 backend validation")

    pairs_dir = run_dir / "pairs"
    manifests_dir = run_dir / "manifests"
    outputs: list[Path] = []

    local_validated_path = pairs_dir / "local_validated_pairs.jsonl"
    if not local_validated_path.exists():
        raise FileNotFoundError(f"Local validated pairs file not found: {local_validated_path}")

    # Determine validation mode
    mode = (
        ValidationMode.ONLINE
        if config.enable_backend_validation
        else ValidationMode.OFFLINE
    )

    cache_path = manifests_dir / "validation_cache.jsonl"
    validator = BackendValidator(
        config=BackendValidatorConfig(
            mode=mode,
            cache_path=cache_path,
            require_results=False,
        )
    )

    valid_path = pairs_dir / "pairs.jsonl"
    quarantine_path = pairs_dir / "quarantine.jsonl"

    logger.info(f"Validating pairs against backend (mode: {mode.value})")
    valid_cksum, valid_count, q_cksum, q_count, stats = validator.validate_from_file(
        local_validated_path, valid_path, quarantine_path
    )
    outputs.extend([valid_path, quarantine_path, cache_path])

    state.pairs_valid_count = valid_count
    state.pairs_quarantined_count += q_count
    if stats.errors_by_type:
        state.errors_by_type.update(stats.errors_by_type)

    # Calculate pass rate
    total = stats.pairs_processed
    if total > 0:
        state.backend_pass_rate = valid_count / total

    logger.info(
        f"Backend validation: {valid_count} valid, {q_count} quarantined "
        f"({stats.cache_hits} cache hits, {stats.api_calls} API calls)"
    )
    return outputs


def handle_generate_enrichment(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 9: Generate enrichment labels.

    Extracts entity/topic labels from validated pairs for enrichment training.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of enrichment file paths
    """
    logger.info("Stage: generate_enrichment - Generating enrichment labels")

    pairs_dir = run_dir / "pairs"
    enrichment_dir = run_dir / "enrichment"
    enrichment_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # Use backend-validated pairs if available, otherwise local validated pairs
    valid_pairs_path = pairs_dir / "pairs.jsonl"
    if not valid_pairs_path.exists():
        valid_pairs_path = pairs_dir / "local_validated_pairs.jsonl"
    if not valid_pairs_path.exists():
        logger.warning("No validated pairs file found")
        return outputs

    generator = EnrichmentGenerator(config=EnrichmentGeneratorConfig())

    enrichment_path = enrichment_dir / "enrichment_labels.jsonl"
    logger.info(f"Generating enrichment labels to {enrichment_path}")
    checksum, count, stats = generator.generate_from_file(
        valid_pairs_path, enrichment_path
    )
    state.enrichment_count = count
    outputs.append(enrichment_path)

    logger.info(
        f"Generated {count} enrichment examples "
        f"({stats.labels_generated} labels: {stats.topics_extracted} topics, "
        f"{stats.institutions_extracted} institutions, {stats.authors_extracted} authors)"
    )
    return outputs


def handle_report(
    run_dir: Path,
    config: PipelineConfig,
    state: PipelineState,
) -> list[Path]:
    """Stage 10: Generate reports and manifest checksums.

    Produces summary report and updates run manifest with checksums.

    Args:
        run_dir: Run directory
        config: Pipeline configuration
        state: Pipeline state to update

    Returns:
        List of report file paths
    """
    logger.info("Stage: report - Generating reports and manifest checksums")

    manifests_dir = run_dir / "manifests"
    outputs: list[Path] = []

    # Load existing run manifest
    manifest_path = manifests_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        manifest = RunManifest.from_dict(manifest_data)
    else:
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    generator = ReportGenerator(config=ReportGeneratorConfig())

    report, updated_manifest, stats = generator.generate_and_save(
        run_dir=run_dir,
        manifest=manifest,
        sources_count=len(state.source_entries),
        topics_count=state.topics_count,
        entities_count=state.entities_count,
        entries_by_source=state.entries_by_source,
        entries_by_domain=state.entries_by_domain,
        templates_count=len(state.templates),
        inputs_count=state.inputs_count,
        pairs_valid_count=state.pairs_valid_count,
        pairs_quarantined_count=state.pairs_quarantined_count,
        enrichment_labels_count=state.enrichment_count,
        backend_pass_rate=state.backend_pass_rate,
        errors_by_type=state.errors_by_type,
        config_path=config.sources_config_path,
        seed=config.seed,
        pinned_revisions=state.pinned_revisions,
    )

    outputs.append(run_dir / "reports" / "summary.json")
    outputs.append(manifest_path)

    logger.info(
        f"Report generated: {stats.artifacts_processed} artifacts, "
        f"{stats.total_size_bytes} bytes total"
    )
    return outputs


# Handler registry mapping stage to handler function
STAGE_HANDLERS = {
    Stage.FETCH: handle_fetch,
    Stage.NORMALIZE: handle_normalize,
    Stage.EXPAND_ALIASES: handle_expand_aliases,
    Stage.LOAD_TEMPLATES: handle_load_templates,
    Stage.GENERATE_INPUTS: handle_generate_inputs,
    Stage.RENDER_PAIRS: handle_render_pairs,
    Stage.VALIDATE_LOCAL: handle_validate_local,
    Stage.VALIDATE_BACKEND: handle_validate_backend,
    Stage.GENERATE_ENRICHMENT: handle_generate_enrichment,
    Stage.REPORT: handle_report,
}


def get_handler_for_stage(stage: Stage) -> callable:
    """Get the handler function for a pipeline stage.

    Args:
        stage: The pipeline stage

    Returns:
        Handler function that takes (run_dir, config, state) and returns list[Path]
    """
    return STAGE_HANDLERS.get(stage)
