#!/usr/bin/env python3
"""
Curate generated training examples using LLM judge and syntax validation.

This script:
1. Loads all generated example files from data/datasets/generated/*.json
2. Validates each example through:
   - LLM judge for semantic quality (score 1-5)
   - ADS syntax validator for syntactic correctness
3. Outputs:
   - curated_*.json files with high-quality examples (score >= 3, no syntax errors)
   - quarantine_report.json with rejected examples and failure reasons

Usage:
    python scripts/curate_generated_examples.py [--dry-run] [--skip-llm]
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Add packages/finetune/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))

from finetune.domains.scix.validate import (
    lint_query,
    validate_field_constraints,
    validate_nl,
)

# Optional: Try to import Anthropic for LLM judge
try:
    from anthropic import Anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# LLM Judge prompt for evaluating NL-query pair quality
LLM_JUDGE_PROMPT = """You are evaluating the quality of a training example for an NL-to-ADS-query system.

Given the following natural language (NL) and ADS query pair, rate the quality on a scale of 1-5:

**Natural Language:** {nl}
**ADS Query:** {ads_query}

Evaluation criteria:
1. **Semantic alignment**: Does the query correctly capture the intent of the natural language?
2. **Naturalness**: Does the NL sound like something a real user would say?
3. **Completeness**: Does the query include all relevant constraints from the NL?
4. **Correctness**: Is the ADS syntax correct (fields, operators, values)?

Score guidelines:
- 5: Perfect alignment, natural phrasing, complete and correct
- 4: Good alignment, minor issues in phrasing or completeness
- 3: Acceptable, captures main intent but has some gaps
- 2: Poor alignment or unnatural phrasing, missing key constraints
- 1: Incorrect or misleading, doesn't match the NL intent

Respond with ONLY a JSON object:
{{"score": <1-5>, "reason": "<brief explanation>"}}"""


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""

    score: int
    reason: str


@dataclass
class ValidationResult:
    """Combined validation result for an example."""

    example: dict
    syntax_valid: bool
    syntax_errors: list[str]
    constraint_valid: bool
    constraint_errors: list[str]
    nl_valid: bool
    nl_issues: list[str]
    llm_score: int
    llm_reason: str
    passed: bool
    failure_reasons: list[str]


def get_anthropic_client() -> "Anthropic | None":
    """Get Anthropic client if API key is available."""
    if not HAS_ANTHROPIC:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def llm_judge(client: "Anthropic", nl: str, ads_query: str, retries: int = 2) -> JudgeResult:
    """Use LLM to evaluate the quality of an NL-query pair.

    Args:
        client: Anthropic client
        nl: Natural language input
        ads_query: ADS query
        retries: Number of retries on failure

    Returns:
        JudgeResult with score (1-5) and reason
    """
    prompt = LLM_JUDGE_PROMPT.format(nl=nl, ads_query=ads_query)

    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text.strip()

            # Parse JSON response
            try:
                data = json.loads(content)
                score = int(data.get("score", 0))
                reason = data.get("reason", "No reason provided")
                if 1 <= score <= 5:
                    return JudgeResult(score=score, reason=reason)
            except (json.JSONDecodeError, ValueError, KeyError):
                # Try to extract score from text if JSON parsing fails
                import re

                match = re.search(r'"score"\s*:\s*(\d)', content)
                if match:
                    score = int(match.group(1))
                    if 1 <= score <= 5:
                        return JudgeResult(score=score, reason="Parsed from response")

        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            return JudgeResult(score=0, reason=f"API error: {e}")

    return JudgeResult(score=0, reason="Failed to get valid response")


def validate_example(
    example: dict,
    client: "Anthropic | None" = None,
    skip_llm: bool = False,
) -> ValidationResult:
    """Validate a single NL-query example.

    Args:
        example: Dict with natural_language, ads_query, category keys
        client: Anthropic client for LLM judge (optional)
        skip_llm: If True, skip LLM judge (assign score 5 to all)

    Returns:
        ValidationResult with all validation details
    """
    nl = example.get("natural_language", "")
    ads_query = example.get("ads_query", "")

    failure_reasons = []

    # 1. Validate NL (no ADS syntax leakage)
    nl_valid, nl_issues = validate_nl(nl)
    if not nl_valid:
        failure_reasons.append(f"NL issues: {', '.join(nl_issues)}")

    # 2. Validate ADS query syntax
    lint_result = lint_query(ads_query)
    syntax_valid = lint_result.valid
    syntax_errors = lint_result.errors
    if not syntax_valid:
        failure_reasons.append(f"Syntax errors: {', '.join(syntax_errors)}")

    # 3. Validate field constraints (doctype, property, bibgroup, database values)
    constraint_result = validate_field_constraints(ads_query)
    constraint_valid = constraint_result.valid
    constraint_errors = constraint_result.error_messages
    if not constraint_valid:
        failure_reasons.append(f"Constraint errors: {', '.join(constraint_errors)}")

    # 4. LLM judge for semantic quality
    if skip_llm or client is None:
        llm_score = 5  # Default to passing if skipping LLM
        llm_reason = "LLM judge skipped"
    else:
        judge_result = llm_judge(client, nl, ads_query)
        llm_score = judge_result.score
        llm_reason = judge_result.reason
        if llm_score < 3:
            failure_reasons.append(f"LLM score {llm_score}: {llm_reason}")
        elif llm_score == 0:
            failure_reasons.append(f"LLM judge failed: {llm_reason}")

    # Determine overall pass/fail
    passed = (
        nl_valid
        and syntax_valid
        and constraint_valid
        and llm_score >= 3
    )

    return ValidationResult(
        example=example,
        syntax_valid=syntax_valid,
        syntax_errors=syntax_errors,
        constraint_valid=constraint_valid,
        constraint_errors=constraint_errors,
        nl_valid=nl_valid,
        nl_issues=nl_issues,
        llm_score=llm_score,
        llm_reason=llm_reason,
        passed=passed,
        failure_reasons=failure_reasons,
    )


def process_file(
    input_path: Path,
    output_dir: Path,
    client: "Anthropic | None" = None,
    skip_llm: bool = False,
    delay: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    """Process a single generated examples file.

    Args:
        input_path: Path to generated examples JSON file
        output_dir: Directory for output files
        client: Anthropic client for LLM judge
        skip_llm: If True, skip LLM judge
        delay: Delay between LLM calls

    Returns:
        Tuple of (curated_examples, quarantine_records)
    """
    with open(input_path) as f:
        examples = json.load(f)

    print(f"  Processing {len(examples)} examples from {input_path.name}")

    curated = []
    quarantine = []

    for i, example in enumerate(examples):
        result = validate_example(example, client, skip_llm)

        if result.passed:
            # Add LLM score to example for transparency
            curated_example = example.copy()
            curated_example["llm_score"] = result.llm_score
            curated.append(curated_example)
        else:
            # Record quarantine details
            quarantine_record = {
                "example": example,
                "syntax_valid": result.syntax_valid,
                "syntax_errors": result.syntax_errors,
                "constraint_valid": result.constraint_valid,
                "constraint_errors": result.constraint_errors,
                "nl_valid": result.nl_valid,
                "nl_issues": result.nl_issues,
                "llm_score": result.llm_score,
                "llm_reason": result.llm_reason,
                "failure_reasons": result.failure_reasons,
            }
            quarantine.append(quarantine_record)

        # Progress indicator
        if (i + 1) % 20 == 0 or i == len(examples) - 1:
            print(f"    Validated {i + 1}/{len(examples)}: {len(curated)} passed, {len(quarantine)} quarantined")

        # Rate limiting for LLM calls
        if not skip_llm and client is not None:
            time.sleep(delay)

    return curated, quarantine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate generated examples with LLM judge and syntax validation"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/datasets/generated",
        help="Input directory with generated example files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets/generated",
        help="Output directory for curated files (default: same as input)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM judge (syntax validation only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process one example from each file and print results",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between LLM API calls in seconds",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all generated example files (exclude curated_* files)
    generated_files = sorted(
        f for f in input_dir.glob("*.json")
        if not f.name.startswith("curated_")
        and not f.name.startswith("quarantine_")
    )

    if not generated_files:
        print(f"No generated example files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(generated_files)} generated example files:")
    for f in generated_files:
        print(f"  - {f.name}")
    print()

    # Initialize LLM client
    client = None
    if not args.skip_llm:
        client = get_anthropic_client()
        if client:
            print("LLM judge enabled (using Claude Sonnet)")
        else:
            print("Warning: ANTHROPIC_API_KEY not set, LLM judge disabled")
            args.skip_llm = True
    else:
        print("LLM judge disabled (--skip-llm)")

    print()

    # Dry run mode
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - Testing one example from each file")
        print("=" * 60)

        for file_path in generated_files:
            with open(file_path) as f:
                examples = json.load(f)
            if not examples:
                continue

            example = examples[0]
            result = validate_example(example, client, args.skip_llm)

            print(f"\n{file_path.name}:")
            print(f"  NL: {example['natural_language'][:60]}...")
            print(f"  Query: {example['ads_query'][:60]}...")
            print(f"  Syntax valid: {result.syntax_valid}")
            print(f"  Constraints valid: {result.constraint_valid}")
            print(f"  NL valid: {result.nl_valid}")
            print(f"  LLM score: {result.llm_score} ({result.llm_reason})")
            print(f"  PASSED: {result.passed}")
            if not result.passed:
                print(f"  Failures: {', '.join(result.failure_reasons)}")

        return

    # Process all files
    output_dir.mkdir(parents=True, exist_ok=True)

    all_curated = []
    all_quarantine = []
    stats_by_file: dict[str, dict] = {}

    print("=" * 60)
    print("PROCESSING GENERATED EXAMPLES")
    print("=" * 60)

    for file_path in generated_files:
        print(f"\n{file_path.name}:")

        curated, quarantine = process_file(
            file_path,
            output_dir,
            client,
            args.skip_llm,
            args.delay,
        )

        all_curated.extend(curated)
        all_quarantine.extend(quarantine)

        # Track stats per file
        base_name = file_path.stem  # e.g., "collection_examples"
        stats_by_file[base_name] = {
            "total": len(curated) + len(quarantine),
            "curated": len(curated),
            "quarantined": len(quarantine),
            "pass_rate": len(curated) / (len(curated) + len(quarantine)) * 100
            if (len(curated) + len(quarantine)) > 0
            else 0,
        }

        # Save curated file for this input
        curated_path = output_dir / f"curated_{base_name}.json"
        with open(curated_path, "w") as f:
            json.dump(curated, f, indent=2)
        print(f"  → Saved {len(curated)} curated examples to {curated_path.name}")

    # Save quarantine report
    quarantine_report = {
        "summary": {
            "total_processed": len(all_curated) + len(all_quarantine),
            "total_curated": len(all_curated),
            "total_quarantined": len(all_quarantine),
            "overall_pass_rate": len(all_curated) / (len(all_curated) + len(all_quarantine)) * 100
            if (len(all_curated) + len(all_quarantine)) > 0
            else 0,
        },
        "by_file": stats_by_file,
        "quarantined_examples": all_quarantine,
    }

    quarantine_path = output_dir / "quarantine_report.json"
    with open(quarantine_path, "w") as f:
        json.dump(quarantine_report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("CURATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal processed: {len(all_curated) + len(all_quarantine)}")
    print(f"Curated (passed): {len(all_curated)}")
    print(f"Quarantined (failed): {len(all_quarantine)}")
    print(f"Overall pass rate: {quarantine_report['summary']['overall_pass_rate']:.1f}%")

    print("\nBy file:")
    for file_name, stats in stats_by_file.items():
        print(
            f"  {file_name}: "
            f"{stats['curated']}/{stats['total']} passed "
            f"({stats['pass_rate']:.1f}%)"
        )

    # Analyze quarantine reasons
    if all_quarantine:
        print("\nQuarantine reasons:")
        reason_counts: dict[str, int] = {}
        for q in all_quarantine:
            for reason in q.get("failure_reasons", []):
                # Extract the type of failure
                if "Syntax errors" in reason:
                    reason_counts["Syntax errors"] = reason_counts.get("Syntax errors", 0) + 1
                elif "Constraint errors" in reason:
                    reason_counts["Constraint errors"] = reason_counts.get("Constraint errors", 0) + 1
                elif "NL issues" in reason:
                    reason_counts["NL issues"] = reason_counts.get("NL issues", 0) + 1
                elif "LLM score" in reason:
                    reason_counts["Low LLM score"] = reason_counts.get("Low LLM score", 0) + 1
                elif "LLM judge failed" in reason:
                    reason_counts["LLM judge failed"] = reason_counts.get("LLM judge failed", 0) + 1

        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print(f"\nOutputs:")
    print(f"  - Curated files: curated_*.json in {output_dir}")
    print(f"  - Quarantine report: {quarantine_path}")


if __name__ == "__main__":
    main()
