#!/usr/bin/env python3
"""US-003: Generate full-scale enrichment dataset (10K+ examples).

Reuses normalized catalogs from the US-001 pipeline run and generates a
production-scale enrichment dataset for SciBERT NER training.

Strategy:
  - Multi-round snippet generation (12+ rounds, different seeds)
  - Higher per-round counts to ensure >10K unique snippets after dedup
  - Balanced coverage across all 5 source vocabularies (>= 500 each)
  - Spot-check 20 random examples for byte-exact span offsets
  - Save final dataset to data/datasets/enrichment/ for Colab upload

Usage:
    python scripts/run_us003_fullscale.py \
        --run-dir data/datasets/agent_runs/run_20260127_174306_999adfdd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages" / "finetune" / "src")
)

from finetune.dataset_agent.enrichment_dataset_builder import (
    DatasetBuilderConfig,
    build_enrichment_dataset_from_files,
)
from finetune.dataset_agent.snippet_generator import (
    SnippetGeneratorConfig,
    generate_snippets,
    _load_entities,
    _load_topics,
)
from finetune.dataset_agent.writers import JSONLWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "datasets" / "enrichment"

MIN_TOTAL = 10000
MIN_PER_VOCABULARY = 500
REQUIRED_VOCABULARIES = {"uat", "sweet", "gcmd", "ror", "planetary"}


def run_snippet_generation(
    topic_catalog: Path,
    entity_catalog: Path,
    output_path: Path,
    title_count: int,
    abstract_count: int,
    num_rounds: int,
) -> dict[str, int]:
    """Generate snippets with multi-round diversity.

    Returns vocabulary count mapping for coverage tracking.
    """
    logger.info("Loading catalogs...")
    topics = _load_topics(topic_catalog)
    entities = _load_entities(entity_catalog)
    logger.info(f"  Topics: {len(topics)}, Entities: {len(entities)}")

    all_snippet_dicts: list[dict] = []
    seen_hashes: set[str] = set()
    total_titles = 0
    total_abstracts = 0
    total_spans = 0
    vocab_counts: dict[str, int] = {}

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
                snippet_dict = snippet.to_dict()
                all_snippet_dicts.append(snippet_dict)
                round_new += 1
                if snippet.text_type == "title":
                    total_titles += 1
                else:
                    total_abstracts += 1
                total_spans += len(snippet.spans)
                for span in snippet.spans:
                    vocab = span.source_vocabulary
                    vocab_counts[vocab] = vocab_counts.get(vocab, 0) + 1

        logger.info(
            f"  Round {round_idx + 1}/{num_rounds} (seed={seed}): "
            f"{len(snippets)} generated, {round_new} new unique, "
            f"cumulative={len(all_snippet_dicts)}"
        )

        # Check if we have enough
        if len(all_snippet_dicts) >= MIN_TOTAL:
            all_below_min = all(
                vocab_counts.get(v, 0) >= MIN_PER_VOCABULARY
                for v in REQUIRED_VOCABULARIES
            )
            if all_below_min:
                logger.info(
                    f"  Reached {len(all_snippet_dicts)} snippets with "
                    f"all vocabularies >= {MIN_PER_VOCABULARY}. Stopping early."
                )
                break

    with JSONLWriter(output_path) as writer:
        for snippet_dict in all_snippet_dicts:
            writer.write_line(snippet_dict)

    logger.info(f"\nSnippet generation complete:")
    logger.info(f"  Total unique: {len(all_snippet_dicts)}")
    logger.info(f"  Titles: {total_titles}, Abstracts: {total_abstracts}")
    logger.info(f"  Total spans: {total_spans}")
    logger.info(f"  Vocabulary coverage:")
    for vocab in sorted(vocab_counts.keys()):
        logger.info(f"    {vocab}: {vocab_counts[vocab]}")

    return vocab_counts


def spot_check_spans(snippets_path: Path, sample_size: int = 20) -> tuple[int, int]:
    """Spot-check random examples for byte-exact span offsets.

    Returns (checked_count, pass_count).
    """
    logger.info(f"\nSpot-checking {sample_size} random examples for span accuracy...")
    lines: list[str] = []
    with open(snippets_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)

    rng = random.Random(99)
    sample_indices = rng.sample(range(len(lines)), min(sample_size, len(lines)))
    checked = 0
    passed = 0

    for idx in sample_indices:
        record = json.loads(lines[idx])
        text = record.get("text", "")
        spans = record.get("spans", [])
        all_ok = True

        for span in spans:
            surface = span["surface"]
            start = span["start"]
            end = span["end"]
            actual = text[start:end]
            if actual != surface:
                logger.warning(
                    f"  MISMATCH in {record.get('id', '?')}: "
                    f"expected '{surface}' at [{start}:{end}], got '{actual}'"
                )
                all_ok = False

        checked += 1
        if all_ok:
            passed += 1

    logger.info(f"  Spot check: {passed}/{checked} passed ({passed*100//checked}%)")
    return checked, passed


def build_dataset(
    snippets_path: Path,
    pair_labels_path: Path | None,
    output_dir: Path,
) -> tuple[int, dict]:
    """Build the enrichment dataset with train/val/test splits."""
    logger.info(f"\nBuilding enrichment dataset...")
    logger.info(f"  Snippets: {snippets_path}")
    logger.info(f"  Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    effective_pair_path = pair_labels_path
    if effective_pair_path is None or not effective_pair_path.exists():
        # Create empty pair labels file
        empty_path = output_dir / "_empty_pairs.jsonl"
        empty_path.write_text("")
        effective_pair_path = empty_path

    config = DatasetBuilderConfig(
        min_examples=MIN_TOTAL,
        snippet_multiplier=1,
        seed=42,
        min_per_vocabulary=MIN_PER_VOCABULARY,
    )

    total_count, stats = build_enrichment_dataset_from_files(
        snippets_path, effective_pair_path, output_dir, config
    )

    logger.info(f"\nDataset built: {total_count} total records")
    logger.info(f"  Train: {stats.train_count}")
    logger.info(f"  Val: {stats.val_count}")
    logger.info(f"  Test: {stats.test_count}")

    # Read coverage report
    coverage_path = output_dir / "enrichment_coverage.json"
    coverage = {}
    if coverage_path.exists():
        with open(coverage_path) as f:
            coverage = json.load(f)
        logger.info("\nCoverage report:")
        for key in ["by_source_vocabulary", "by_label_type", "by_domain", "by_text_type"]:
            if key in coverage:
                logger.info(f"  {key}:")
                for k, v in sorted(coverage[key].items()):
                    logger.info(f"    {k}: {v}")

    return total_count, coverage


def copy_to_final_output(enrichment_dir: Path, final_dir: Path) -> None:
    """Copy dataset files to the final output directory."""
    final_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "enrichment_labels.jsonl",
        "enrichment_train.jsonl",
        "enrichment_val.jsonl",
        "enrichment_test.jsonl",
        "enrichment_coverage.json",
    ]

    for filename in files_to_copy:
        src = enrichment_dir / filename
        dst = final_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"  Copied {filename} to {final_dir}")
        else:
            logger.warning(f"  Missing: {filename}")


def validate_dataset(final_dir: Path) -> bool:
    """Validate the final dataset meets all acceptance criteria."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)

    all_pass = True

    # Check file existence
    required_files = [
        "enrichment_train.jsonl",
        "enrichment_val.jsonl",
        "enrichment_test.jsonl",
        "enrichment_coverage.json",
    ]
    for fname in required_files:
        path = final_dir / fname
        if not path.exists():
            logger.error(f"  FAIL: Missing {fname}")
            all_pass = False
        else:
            count = sum(1 for line in open(path, encoding="utf-8") if line.strip())
            logger.info(f"  OK: {fname} ({count} records)")

    # Check total count
    total = 0
    for split_name in ["enrichment_train.jsonl", "enrichment_val.jsonl", "enrichment_test.jsonl"]:
        path = final_dir / split_name
        if path.exists():
            total += sum(1 for line in open(path, encoding="utf-8") if line.strip())

    if total < MIN_TOTAL:
        logger.error(f"  FAIL: Total records ({total}) < {MIN_TOTAL}")
        all_pass = False
    else:
        logger.info(f"  OK: Total records = {total} (>= {MIN_TOTAL})")

    # Check split ratios
    train_path = final_dir / "enrichment_train.jsonl"
    val_path = final_dir / "enrichment_val.jsonl"
    test_path = final_dir / "enrichment_test.jsonl"
    if train_path.exists() and val_path.exists() and test_path.exists():
        train_n = sum(1 for line in open(train_path, encoding="utf-8") if line.strip())
        val_n = sum(1 for line in open(val_path, encoding="utf-8") if line.strip())
        test_n = sum(1 for line in open(test_path, encoding="utf-8") if line.strip())
        split_total = train_n + val_n + test_n
        if split_total > 0:
            train_frac = train_n / split_total
            val_frac = val_n / split_total
            test_frac = test_n / split_total
            logger.info(
                f"  Splits: train={train_frac:.1%} ({train_n}), "
                f"val={val_frac:.1%} ({val_n}), test={test_frac:.1%} ({test_n})"
            )

    # Check vocabulary coverage
    coverage_path = final_dir / "enrichment_coverage.json"
    if coverage_path.exists():
        with open(coverage_path) as f:
            coverage = json.load(f)
        vocab_counts = coverage.get("by_source_vocabulary", {})
        for vocab in REQUIRED_VOCABULARIES:
            count = vocab_counts.get(vocab, 0)
            if count < MIN_PER_VOCABULARY:
                logger.error(
                    f"  FAIL: Vocabulary '{vocab}' has {count} spans "
                    f"(need >= {MIN_PER_VOCABULARY})"
                )
                all_pass = False
            else:
                logger.info(f"  OK: Vocabulary '{vocab}' = {count} spans")

    # Spot-check spans
    labels_path = final_dir / "enrichment_labels.jsonl"
    if labels_path.exists():
        checked, passed = spot_check_spans(labels_path, sample_size=20)
        if passed < checked:
            logger.error(f"  FAIL: Span accuracy {passed}/{checked}")
            all_pass = False
        else:
            logger.info(f"  OK: Span accuracy {passed}/{checked}")

    if all_pass:
        logger.info("\n  ALL VALIDATIONS PASSED")
    else:
        logger.error("\n  SOME VALIDATIONS FAILED")

    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="US-003: Generate full-scale enrichment dataset (10K+)"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/datasets/agent_runs/run_20260127_174306_999adfdd"),
        help="Existing run directory with normalized catalogs from US-001",
    )
    parser.add_argument(
        "--title-count",
        type=int,
        default=3000,
        help="Title snippets per round",
    )
    parser.add_argument(
        "--abstract-count",
        type=int,
        default=3000,
        help="Abstract snippets per round",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=15,
        help="Maximum number of snippet generation rounds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Final output directory for enrichment dataset",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)

    normalized_dir = args.run_dir / "normalized"
    topic_catalog = normalized_dir / "topic_catalog.jsonl"
    entity_catalog = normalized_dir / "entity_catalog.jsonl"

    if not topic_catalog.exists() or not entity_catalog.exists():
        logger.error("Missing unified catalogs. Run US-001 first.")
        sys.exit(1)

    # Working directory for intermediate files
    enrichment_work_dir = args.run_dir / "enrichment_fullscale"
    enrichment_work_dir.mkdir(parents=True, exist_ok=True)
    snippets_path = enrichment_work_dir / "snippets.jsonl"

    logger.info("=" * 60)
    logger.info("US-003: Full-scale enrichment dataset generation")
    logger.info("=" * 60)
    logger.info(f"  Topic catalog: {topic_catalog}")
    logger.info(f"  Entity catalog: {entity_catalog}")
    logger.info(f"  Target: >= {MIN_TOTAL} records, >= {MIN_PER_VOCABULARY}/vocabulary")
    logger.info(f"  Rounds: up to {args.num_rounds}")
    logger.info(f"  Per-round: {args.title_count} titles + {args.abstract_count} abstracts")
    logger.info(f"  Output: {args.output_dir}")

    # Step 1: Generate snippets
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Snippet generation (multi-round)")
    logger.info("=" * 60)
    vocab_counts = run_snippet_generation(
        topic_catalog=topic_catalog,
        entity_catalog=entity_catalog,
        output_path=snippets_path,
        title_count=args.title_count,
        abstract_count=args.abstract_count,
        num_rounds=args.num_rounds,
    )

    # Step 2: Build dataset with splits
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Build enrichment dataset with train/val/test splits")
    logger.info("=" * 60)
    # Check for any existing pair labels from the pipeline run
    pair_labels_path = args.run_dir / "enrichment" / "enrichment_labels.jsonl"
    total_count, coverage = build_dataset(
        snippets_path=snippets_path,
        pair_labels_path=pair_labels_path if pair_labels_path.exists() else None,
        output_dir=enrichment_work_dir,
    )

    # Step 3: Copy to final output
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Copy to final output directory")
    logger.info("=" * 60)
    copy_to_final_output(enrichment_work_dir, args.output_dir)

    # Step 4: Validate
    all_pass = validate_dataset(args.output_dir)

    if all_pass:
        logger.info("\n" + "=" * 60)
        logger.info("US-003 COMPLETE: Full-scale dataset ready for Colab upload")
        logger.info(f"  Location: {args.output_dir}")
        logger.info("=" * 60)
    else:
        logger.warning("\n" + "=" * 60)
        logger.warning("US-003: Dataset generated but some validations failed")
        logger.warning("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
