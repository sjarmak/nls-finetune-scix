#!/usr/bin/env python3
"""US-002: Continue pipeline from expand_aliases using existing US-001 run dir.

This script:
1. Runs pipeline stages 3-10 (expand_aliases through report) on the existing
   normalized catalogs from US-001
2. Generates snippets from the real unified catalogs (multi-round for diversity)
3. Builds the unified enrichment dataset with train/val/test splits

Usage:
    python scripts/run_us002_pipeline.py \
        --run-dir data/datasets/agent_runs/run_20260127_174306_999adfdd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "finetune" / "src"))

from finetune.dataset_agent.enrichment_dataset_builder import (
    DatasetBuilderConfig,
    build_enrichment_dataset_from_files,
)
from finetune.dataset_agent.pipeline_handlers import (
    STAGE_HANDLERS,
    PipelineConfig,
    PipelineState,
)
from finetune.dataset_agent.schemas import Stage
from finetune.dataset_agent.snippet_generator import (
    SnippetGeneratorConfig,
    generate_snippets,
)
from finetune.dataset_agent.sources import load_sources_config
from finetune.dataset_agent.stage_runner import StageRunner, StageStatus
from finetune.dataset_agent.writers import JSONLWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_pipeline_state_from_run(run_dir: Path, sources_config_path: Path) -> PipelineState:
    """Reconstruct pipeline state from an existing run directory."""
    state = PipelineState()
    state.sources_config = load_sources_config(sources_config_path)

    normalized_dir = run_dir / "normalized"

    for catalog_file in sorted(normalized_dir.glob("*_catalog_*.jsonl")):
        source_id = catalog_file.stem.split("_catalog_")[-1]
        count = sum(1 for line in open(catalog_file, encoding="utf-8") if line.strip())
        state.entries_by_source[source_id] = count

        if "topic" in catalog_file.stem:
            state.topics_count += count
        else:
            state.entities_count += count

    domain_counts: dict[str, int] = {}
    for merged_name in ["topic_catalog.jsonl", "entity_catalog.jsonl"]:
        merged_path = normalized_dir / merged_name
        if merged_path.exists():
            with open(merged_path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        entry = json.loads(stripped)
                        for tag in entry.get("domain_tags", []):
                            domain_counts[tag] = domain_counts.get(tag, 0) + 1
    state.entries_by_domain = domain_counts

    logger.info(f"Loaded state: {state.topics_count} topics, {state.entities_count} entities")
    logger.info(f"  Sources: {state.entries_by_source}")
    logger.info(f"  Domains: {state.entries_by_domain}")

    return state


def run_pipeline_continuation(
    run_dir: Path, sources_config_path: Path, samples_per_template: int
) -> bool:
    """Run pipeline stages 3-10 on an existing run directory."""
    logger.info(f"Continuing pipeline in: {run_dir}")

    stages_to_run = [
        Stage.EXPAND_ALIASES,
        Stage.LOAD_TEMPLATES,
        Stage.GENERATE_INPUTS,
        Stage.RENDER_PAIRS,
        Stage.VALIDATE_LOCAL,
        Stage.GENERATE_ENRICHMENT,
        Stage.REPORT,
    ]

    pipeline_config = PipelineConfig(
        sources_config_path=sources_config_path,
        seed=42,
        samples_per_template=samples_per_template,
        skip_backend=True,
        enable_backend_validation=False,
    )

    pipeline_state = load_pipeline_state_from_run(run_dir, sources_config_path)

    stage_runner = StageRunner(run_dir)

    for stage, handler in STAGE_HANDLERS.items():

        def create_wrapped_handler(h, pc, ps):
            def wrapped(rd: Path) -> list[Path]:
                return h(rd, pc, ps)

            return wrapped

        stage_runner.register_handler(
            stage, create_wrapped_handler(handler, pipeline_config, pipeline_state)
        )

    logger.info("=" * 60)
    logger.info("PIPELINE CONTINUATION (stages 3-10)")
    logger.info("=" * 60)

    for stage in stages_to_run:
        logger.info(f"\n>>> Stage: {stage.value}")
        result = stage_runner.run_stage(stage)

        if result.status == StageStatus.COMPLETED:
            logger.info(
                f"    [COMPLETE] {len(result.outputs)} outputs, {result.duration_seconds:.1f}s"
            )
        elif result.status == StageStatus.SKIPPED:
            logger.info("    [SKIPPED]")
        elif result.status == StageStatus.FAILED:
            logger.error(f"    [FAILED] {result.error}")
            return False

    logger.info("\nPipeline stages complete.")
    return True


def run_snippet_generation(
    run_dir: Path,
    title_count: int,
    abstract_count: int,
    num_rounds: int = 5,
) -> Path | None:
    """Generate snippets using multiple seed rounds for diversity.

    Each round uses a different seed to produce diverse combinations.
    Results are deduplicated by text content and merged into one file.
    """
    normalized_dir = run_dir / "normalized"
    enrichment_dir = run_dir / "enrichment"
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    topic_catalog = normalized_dir / "topic_catalog.jsonl"
    entity_catalog = normalized_dir / "entity_catalog.jsonl"
    snippets_path = enrichment_dir / "snippets.jsonl"

    if not topic_catalog.exists() or not entity_catalog.exists():
        logger.error("Missing unified catalogs for snippet generation")
        return None

    logger.info("=" * 60)
    logger.info("SNIPPET GENERATION (multi-round)")
    logger.info("=" * 60)
    logger.info(f"  Topics: {topic_catalog}")
    logger.info(f"  Entities: {entity_catalog}")
    logger.info(f"  Output: {snippets_path}")
    logger.info(
        f"  Rounds: {num_rounds}, per-round: {title_count} titles, {abstract_count} abstracts"
    )

    # Import internal loader functions
    from finetune.dataset_agent.snippet_generator import _load_entities, _load_topics

    topics = _load_topics(topic_catalog)
    entities = _load_entities(entity_catalog)

    all_snippet_dicts: list[dict] = []
    seen_hashes: set[str] = set()
    total_titles = 0
    total_abstracts = 0
    total_spans = 0

    for round_idx in range(num_rounds):
        seed = 42 + round_idx * 1000
        config = SnippetGeneratorConfig(
            title_count=title_count,
            abstract_count=abstract_count,
            seed=seed,
        )
        snippets, stats = generate_snippets(topics, entities, config)
        round_new = 0

        for snippet in snippets:
            text_hash = hashlib.md5(snippet.text.encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                all_snippet_dicts.append(snippet.to_dict())
                round_new += 1
                if snippet.text_type == "title":
                    total_titles += 1
                else:
                    total_abstracts += 1
                total_spans += len(snippet.spans)

        logger.info(
            f"  Round {round_idx + 1}/{num_rounds} (seed={seed}): "
            f"{len(snippets)} generated, {round_new} new unique"
        )

    # Write deduplicated snippets
    with JSONLWriter(snippets_path) as writer:
        for snippet_dict in all_snippet_dicts:
            writer.write_line(snippet_dict)

    logger.info(f"\nTotal unique snippets: {len(all_snippet_dicts)}")
    logger.info(f"  Titles: {total_titles}, Abstracts: {total_abstracts}")
    logger.info(f"  Total spans: {total_spans}")

    return snippets_path


def run_enrichment_dataset_build(run_dir: Path, builder_config: DatasetBuilderConfig) -> bool:
    """Build the unified enrichment dataset from snippets + pair labels."""
    enrichment_dir = run_dir / "enrichment"

    snippets_path = enrichment_dir / "snippets.jsonl"
    pair_labels_path = enrichment_dir / "enrichment_labels.jsonl"

    if not snippets_path.exists():
        logger.error(f"Missing snippets: {snippets_path}")
        return False

    if not pair_labels_path.exists():
        logger.warning(f"No pair labels found at {pair_labels_path}, using snippets only")
        pair_labels_path.write_text("")

    logger.info("=" * 60)
    logger.info("ENRICHMENT DATASET BUILD")
    logger.info("=" * 60)
    logger.info(f"  Snippets: {snippets_path}")
    logger.info(f"  Pair labels: {pair_labels_path}")
    logger.info(f"  Output dir: {enrichment_dir}")
    logger.info(f"  Min examples: {builder_config.min_examples}")

    total_count, stats = build_enrichment_dataset_from_files(
        snippets_path, pair_labels_path, enrichment_dir, builder_config
    )

    logger.info(f"\nEnrichment dataset built: {total_count} total records")
    logger.info(f"  Train: {stats.train_count}")
    logger.info(f"  Val: {stats.val_count}")
    logger.info(f"  Test: {stats.test_count}")
    logger.info(f"  Snippets loaded: {stats.snippets_loaded}")
    logger.info(f"  Pair labels loaded: {stats.pair_labels_loaded}")
    logger.info(f"  Total records: {stats.total_records}")

    # Check coverage
    coverage_path = enrichment_dir / "enrichment_coverage.json"
    if coverage_path.exists():
        with open(coverage_path) as f:
            coverage = json.load(f)
        logger.info("\nCoverage report:")
        if "vocabulary_counts" in coverage:
            for vocab, count in coverage["vocabulary_counts"].items():
                logger.info(f"  {vocab}: {count}")
        if "domain_counts" in coverage:
            for domain, count in coverage["domain_counts"].items():
                logger.info(f"  {domain}: {count}")

    return total_count >= builder_config.min_examples


def main():
    parser = argparse.ArgumentParser(description="US-002: Run full pipeline with real catalogs")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/datasets/agent_runs/run_20260127_174306_999adfdd"),
        help="Existing run directory from US-001",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("packages/finetune/src/finetune/dataset_agent/config/sources.yaml"),
        help="Path to sources.yaml",
    )
    parser.add_argument(
        "--samples-per-template",
        type=int,
        default=10,
        help="Samples per template (10 for small sample)",
    )
    parser.add_argument(
        "--title-count",
        type=int,
        default=2500,
        help="Number of title snippets per round",
    )
    parser.add_argument(
        "--abstract-count",
        type=int,
        default=2500,
        help="Number of abstract snippets per round",
    )
    parser.add_argument(
        "--snippet-rounds",
        type=int,
        default=8,
        help="Number of snippet generation rounds with different seeds",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=5000,
        help="Minimum enrichment dataset examples",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip pipeline stages (only run snippet gen + dataset build)",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)

    # Step 1: Continue pipeline stages
    if not args.skip_pipeline:
        logger.info("STEP 1: Continue pipeline stages (expand_aliases through report)")
        success = run_pipeline_continuation(args.run_dir, args.sources, args.samples_per_template)
        if not success:
            logger.error("Pipeline continuation failed")
            sys.exit(1)
    else:
        logger.info("STEP 1: Skipped (--skip-pipeline)")

    # Step 2: Generate snippets (multi-round)
    logger.info("\nSTEP 2: Generate snippets from real catalogs")
    snippets_path = run_snippet_generation(
        args.run_dir,
        title_count=args.title_count,
        abstract_count=args.abstract_count,
        num_rounds=args.snippet_rounds,
    )
    if snippets_path is None:
        logger.error("Snippet generation failed")
        sys.exit(1)

    # Step 3: Build enrichment dataset
    logger.info("\nSTEP 3: Build enrichment dataset")
    builder_config = DatasetBuilderConfig(
        min_examples=args.min_examples,
        snippet_multiplier=1,  # We handle multiplicity via multi-round generation
        seed=42,
        min_per_vocabulary=100,
    )
    success = run_enrichment_dataset_build(args.run_dir, builder_config)

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("US-002 COMPLETE: All steps passed")
        logger.info("=" * 60)
    else:
        logger.warning("\n" + "=" * 60)
        logger.warning("US-002: Dataset built but below minimum threshold")
        logger.warning("=" * 60)

    # Print summary
    enrichment_dir = args.run_dir / "enrichment"
    reports_dir = args.run_dir / "reports"
    logger.info("\nKey outputs:")
    for path in sorted(enrichment_dir.glob("*.jsonl")):
        lines = sum(1 for _ in open(path))
        logger.info(f"  {path.name}: {lines} records")
    summary = reports_dir / "summary.json"
    if summary.exists():
        logger.info(f"  {summary.name}: pipeline summary")


if __name__ == "__main__":
    main()
