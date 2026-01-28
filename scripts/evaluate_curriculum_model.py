"""Evaluate curriculum-trained enrichment model and compare with baseline.

Runs evaluation on:
1. Synthetic test set (from enrichment_test.jsonl)
2. Real-world test set (from human annotations or ADS samples)

Compares to:
- Keyword baseline
- Synthetic-only model (from previous phase)

Produces:
- reports/enrichment_eval_v2.json — structured metrics and comparison
- reports/enrichment_eval_v2.md — narrative report with gap analysis

Usage:
    # Full evaluation with human annotations
    python scripts/evaluate_curriculum_model.py \\
        --model-dir models/enrichment_ner_v2 \\
        --synthetic-test data/datasets/enrichment/enrichment_test.jsonl \\
        --real-world-test data/evaluation/ads_sample_all_reannotated.jsonl \\
        --baseline-results reports/enrichment_model_eval.json

    # Synthetic-only evaluation (human data pending)
    python scripts/evaluate_curriculum_model.py \\
        --model-dir models/enrichment_ner_v2 \\
        --synthetic-test data/datasets/enrichment/enrichment_test.jsonl \\
        --baseline-results reports/enrichment_model_eval.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalConfig:
    """Configuration for curriculum model evaluation."""

    model_dir: str
    synthetic_test: str
    real_world_test: str | None
    baseline_results: str | None
    output_dir: str
    synthetic_f1_threshold: float = 0.70
    real_world_f1_threshold: float = 0.50


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------


def run_synthetic_evaluation(model_dir: Path, test_file: Path, output_dir: Path) -> dict[str, Any]:
    """Run evaluation on synthetic test set using evaluate_enrichment_model.py."""
    print(f"\nRunning synthetic test set evaluation...")
    print(f"  Model: {model_dir}")
    print(f"  Test file: {test_file}")

    # Create temp directory for evaluation output
    temp_eval_dir = output_dir / "temp_synthetic_eval"
    temp_eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/evaluate_enrichment_model.py",
        "--model-dir",
        str(model_dir),
        "--test-file",
        str(test_file),
        "--output-dir",
        str(temp_eval_dir),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"Error running synthetic evaluation: {e}")
        return {}

    # Read the generated enrichment_model_eval.json
    result_file = temp_eval_dir / "enrichment_model_eval.json"
    if result_file.exists():
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_real_world_evaluation(
    model_dir: Path, test_file: Path, output_dir: Path
) -> dict[str, Any]:
    """Run evaluation on real-world test set using evaluate_real_world.py."""
    print(f"\nRunning real-world test set evaluation...")
    print(f"  Model: {model_dir}")
    print(f"  Test file: {test_file}")

    # Create temp directory for evaluation output
    temp_eval_dir = output_dir / "temp_real_world_eval"
    temp_eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/evaluate_real_world.py",
        "--model-dir",
        str(model_dir),
        "--annotations",
        str(test_file),
        "--output-dir",
        str(temp_eval_dir),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"Error running real-world evaluation: {e}")
        return {}

    # Read the generated real_world_eval.json
    result_file = temp_eval_dir / "real_world_eval.json"
    if result_file.exists():
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_baseline_results(baseline_path: Path) -> dict[str, Any]:
    """Load baseline evaluation results from previous phase."""
    if not baseline_path.exists():
        print(f"Warning: Baseline results not found at {baseline_path}")
        return {}

    with open(baseline_path, encoding="utf-8") as f:
        return json.load(f)


def calculate_gap_delta(
    v2_synthetic_f1: float,
    v2_real_world_f1: float,
    v1_synthetic_f1: float,
    v1_real_world_f1: float,
) -> dict[str, Any]:
    """Calculate synthetic-to-real gap improvement."""
    v1_gap = v1_synthetic_f1 - v1_real_world_f1
    v2_gap = v2_synthetic_f1 - v2_real_world_f1
    gap_improvement = v1_gap - v2_gap

    return {
        "v1_gap": round(v1_gap, 4),
        "v2_gap": round(v2_gap, 4),
        "improvement": round(gap_improvement, 4),
        "improvement_pct": round(gap_improvement / v1_gap * 100, 2) if v1_gap != 0 else 0.0,
    }


def assess_go_no_go(
    synthetic_f1: float,
    real_world_f1: float | None,
    config: EvalConfig,
) -> dict[str, Any]:
    """Assess go/no-go based on thresholds."""
    synthetic_pass = synthetic_f1 >= config.synthetic_f1_threshold
    real_world_pass = (
        real_world_f1 >= config.real_world_f1_threshold if real_world_f1 is not None else None
    )

    if real_world_pass is None:
        decision = "CONDITIONAL GO" if synthetic_pass else "NO GO"
        reason = (
            "Synthetic F1 passes threshold. Real-world evaluation pending."
            if synthetic_pass
            else f"Synthetic F1 {synthetic_f1:.4f} below threshold {config.synthetic_f1_threshold}"
        )
    else:
        both_pass = synthetic_pass and real_world_pass
        decision = "GO" if both_pass else "CONDITIONAL GO" if synthetic_pass else "NO GO"

        if both_pass:
            reason = "Both synthetic and real-world F1 pass thresholds"
        elif synthetic_pass and not real_world_pass:
            reason = f"Synthetic passes but real-world F1 {real_world_f1:.4f} below threshold {config.real_world_f1_threshold}"
        else:
            reason = f"Synthetic F1 {synthetic_f1:.4f} below threshold {config.synthetic_f1_threshold}"

    return {
        "decision": decision,
        "reason": reason,
        "synthetic_f1": round(synthetic_f1, 4),
        "synthetic_threshold": config.synthetic_f1_threshold,
        "synthetic_pass": synthetic_pass,
        "real_world_f1": round(real_world_f1, 4) if real_world_f1 is not None else None,
        "real_world_threshold": config.real_world_f1_threshold,
        "real_world_pass": real_world_pass,
    }


def generate_json_report(
    config: EvalConfig,
    synthetic_results: dict[str, Any],
    real_world_results: dict[str, Any],
    baseline_results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate JSON report with comparison metrics."""
    report: dict[str, Any] = {
        "model_dir": config.model_dir,
        "evaluation_type": "curriculum_learning_comparison",
        "synthetic_evaluation": synthetic_results,
        "real_world_evaluation": real_world_results if real_world_results else None,
        "baseline_comparison": {},
    }

    # Extract key metrics
    v2_synthetic_f1 = synthetic_results.get("overall", {}).get("f1", 0.0)
    v2_real_world_f1 = real_world_results.get("overall", {}).get("f1") if real_world_results else None

    # Compare with baseline
    if baseline_results:
        v1_synthetic_f1 = baseline_results.get("overall", {}).get("f1", 0.0)
        v1_real_world_f1 = 0.0949  # From progress.txt evaluation results

        if v2_real_world_f1 is not None:
            gap_delta = calculate_gap_delta(
                v2_synthetic_f1, v2_real_world_f1, v1_synthetic_f1, v1_real_world_f1
            )
            report["baseline_comparison"]["gap_analysis"] = gap_delta

        report["baseline_comparison"]["synthetic_f1_delta"] = round(
            v2_synthetic_f1 - v1_synthetic_f1, 4
        )
        if v2_real_world_f1 is not None:
            report["baseline_comparison"]["real_world_f1_delta"] = round(
                v2_real_world_f1 - v1_real_world_f1, 4
            )

    # Go/no-go assessment
    report["go_no_go"] = assess_go_no_go(v2_synthetic_f1, v2_real_world_f1, config)

    # Save report
    report_path = output_dir / "enrichment_eval_v2.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nJSON report saved to {report_path}")


def generate_markdown_report(
    config: EvalConfig,
    synthetic_results: dict[str, Any],
    real_world_results: dict[str, Any],
    baseline_results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate markdown report with narrative analysis."""
    lines: list[str] = []

    lines.append("# Curriculum Learning Model Evaluation Report (v2)")
    lines.append("")
    lines.append(f"**Model Directory:** `{config.model_dir}`")
    lines.append("")

    # Synthetic evaluation
    lines.append("## Synthetic Test Set Evaluation")
    lines.append("")
    v2_synthetic_f1 = synthetic_results.get("overall", {}).get("f1", 0.0)
    lines.append(f"- **Overall F1:** {v2_synthetic_f1:.4f}")
    lines.append(
        f"- **Precision:** {synthetic_results.get('overall', {}).get('precision', 0.0):.4f}"
    )
    lines.append(f"- **Recall:** {synthetic_results.get('overall', {}).get('recall', 0.0):.4f}")
    lines.append("")

    # Real-world evaluation
    if real_world_results:
        lines.append("## Real-World Test Set Evaluation")
        lines.append("")
        v2_real_world_f1 = real_world_results.get("overall", {}).get("f1", 0.0)
        lines.append(f"- **Overall F1:** {v2_real_world_f1:.4f}")
        lines.append(
            f"- **Precision:** {real_world_results.get('overall', {}).get('precision', 0.0):.4f}"
        )
        lines.append(f"- **Recall:** {real_world_results.get('overall', {}).get('recall', 0.0):.4f}")
        lines.append("")
    else:
        lines.append("## Real-World Test Set Evaluation")
        lines.append("")
        lines.append("*Real-world evaluation pending. Human annotations not yet available.*")
        lines.append("")
        v2_real_world_f1 = None

    # Baseline comparison
    if baseline_results:
        lines.append("## Comparison with Baseline (Synthetic-Only Model)")
        lines.append("")
        v1_synthetic_f1 = baseline_results.get("overall", {}).get("f1", 0.0)
        v1_real_world_f1 = 0.0949  # From progress.txt

        lines.append(f"- **V1 Synthetic F1:** {v1_synthetic_f1:.4f}")
        lines.append(f"- **V2 Synthetic F1:** {v2_synthetic_f1:.4f}")
        lines.append(f"- **Synthetic F1 Delta:** {v2_synthetic_f1 - v1_synthetic_f1:+.4f}")
        lines.append("")

        if v2_real_world_f1 is not None:
            lines.append(f"- **V1 Real-World F1:** {v1_real_world_f1:.4f}")
            lines.append(f"- **V2 Real-World F1:** {v2_real_world_f1:.4f}")
            lines.append(f"- **Real-World F1 Delta:** {v2_real_world_f1 - v1_real_world_f1:+.4f}")
            lines.append("")

            # Gap analysis
            gap_delta = calculate_gap_delta(
                v2_synthetic_f1, v2_real_world_f1, v1_synthetic_f1, v1_real_world_f1
            )
            lines.append("### Synthetic-to-Real Gap Analysis")
            lines.append("")
            lines.append(f"- **V1 Gap:** {gap_delta['v1_gap']:.4f}")
            lines.append(f"- **V2 Gap:** {gap_delta['v2_gap']:.4f}")
            lines.append(f"- **Gap Improvement:** {gap_delta['improvement']:+.4f} ({gap_delta['improvement_pct']:+.2f}%)")
            lines.append("")

            if gap_delta["improvement"] > 0:
                lines.append("✅ **Gap reduced:** Curriculum learning successfully reduced the synthetic-to-real gap.")
            else:
                lines.append("⚠️ **Gap not reduced:** Curriculum learning did not reduce the gap.")
            lines.append("")

    # Go/no-go assessment
    assessment = assess_go_no_go(v2_synthetic_f1, v2_real_world_f1, config)
    lines.append("## Go/No-Go Assessment")
    lines.append("")
    lines.append(f"**Decision:** {assessment['decision']}")
    lines.append("")
    lines.append(f"**Reason:** {assessment['reason']}")
    lines.append("")
    lines.append(f"- Synthetic F1 threshold: {assessment['synthetic_threshold']:.2f} → **{assessment['synthetic_pass'] and 'PASS' or 'FAIL'}**")
    if assessment["real_world_f1"] is not None:
        lines.append(f"- Real-world F1 threshold: {assessment['real_world_threshold']:.2f} → **{assessment['real_world_pass'] and 'PASS' or 'FAIL'}**")
    else:
        lines.append(f"- Real-world F1 threshold: {assessment['real_world_threshold']:.2f} → **PENDING**")
    lines.append("")

    # Save report
    report_path = output_dir / "enrichment_eval_v2.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown report saved to {report_path}")


# -------------------------------------------------------------------------
# Main evaluation pipeline
# -------------------------------------------------------------------------


def evaluate(config: EvalConfig) -> int:
    """Run full curriculum model evaluation pipeline."""
    model_dir = Path(config.model_dir)
    synthetic_test = Path(config.synthetic_test)
    output_dir = Path(config.output_dir)

    # Validate inputs
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return 1

    if not synthetic_test.exists():
        print(f"Error: Synthetic test file not found: {synthetic_test}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run synthetic evaluation
    synthetic_results = run_synthetic_evaluation(model_dir, synthetic_test, output_dir)

    # Run real-world evaluation (if data available)
    real_world_results: dict[str, Any] = {}
    if config.real_world_test:
        real_world_test = Path(config.real_world_test)
        if real_world_test.exists():
            real_world_results = run_real_world_evaluation(model_dir, real_world_test, output_dir)
        else:
            print(f"\nWarning: Real-world test file not found: {real_world_test}")
            print("Skipping real-world evaluation.")

    # Load baseline results
    baseline_results: dict[str, Any] = {}
    if config.baseline_results:
        baseline_path = Path(config.baseline_results)
        baseline_results = load_baseline_results(baseline_path)

    # Generate reports
    generate_json_report(
        config, synthetic_results, real_world_results, baseline_results, output_dir
    )
    generate_markdown_report(
        config, synthetic_results, real_world_results, baseline_results, output_dir
    )

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

    return 0


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> EvalConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate curriculum-trained enrichment model and compare with baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the trained model",
    )
    parser.add_argument(
        "--synthetic-test",
        type=str,
        required=True,
        help="Path to synthetic test set (enrichment_test.jsonl)",
    )
    parser.add_argument(
        "--real-world-test",
        type=str,
        default=None,
        help="Path to real-world test set (optional, human annotations)",
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Path to baseline evaluation results JSON (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)",
    )
    parser.add_argument(
        "--synthetic-f1-threshold",
        type=float,
        default=0.70,
        help="Synthetic F1 threshold for go/no-go (default: 0.70)",
    )
    parser.add_argument(
        "--real-world-f1-threshold",
        type=float,
        default=0.50,
        help="Real-world F1 threshold for go/no-go (default: 0.50)",
    )

    args = parser.parse_args()

    return EvalConfig(
        model_dir=args.model_dir,
        synthetic_test=args.synthetic_test,
        real_world_test=args.real_world_test,
        baseline_results=args.baseline_results,
        output_dir=args.output_dir,
        synthetic_f1_threshold=args.synthetic_f1_threshold,
        real_world_f1_threshold=args.real_world_f1_threshold,
    )


def main() -> int:
    """CLI entry point."""
    config = parse_args()
    return evaluate(config)


if __name__ == "__main__":
    sys.exit(main())
