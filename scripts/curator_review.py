#!/usr/bin/env python3
"""LLM Curator Review for Gold Examples.

Reviews training data quality using Claude to identify issues like:
- Author initials being guessed when not provided
- Wrong category assignments
- Invalid query syntax
- Mismatched NL to query mappings
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import anthropic

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))

from finetune.domains.scix.field_constraints import FIELD_ENUMS, DOCTYPES, PROPERTIES, COLLECTIONS, BIBGROUPS
from finetune.domains.scix.validate import lint_query, validate_field_constraints

REVIEW_PROMPT = """You are a data quality reviewer for ADS/SciX search query training data.

Review this training example and identify any issues:

Natural Language Input: {nl}
Generated ADS Query: {query}
Category: {category}

## Review Criteria

1. **Author Initials**: If the NL mentions an author by last name only (e.g., "papers by Hawking"),
   the query should NOT include specific initials (e.g., author:"Hawking, S." is WRONG).
   Correct: author:"Hawking" or author:"^Hawking"

2. **Category Validity**: Category should be descriptive, not "unfielded". Valid categories include:
   author, first_author, topic, operator, property, doctype, collection, bibgroup, year, object, etc.

3. **Query-NL Alignment**: The query should accurately represent the NL intent.
   - If NL asks for "papers by X", query should use author: not abs:
   - If NL mentions a topic, it should be in abs: or title:
   - Year mentions should use year: or pubdate:

4. **Syntax Validity**: Check for:
   - Balanced parentheses and quotes
   - Valid field names (author:, abs:, title:, pubdate:, etc.)
   - Valid enum values for doctype:, property:, collection:, bibgroup:

5. **Common Hallucinations**:
   - doctype:journal (should be doctype:article)
   - property:peer-reviewed (should be property:refereed)
   - database:astrophysics (should be collection:astronomy)

## Response Format

Respond with a JSON object:
{{
  "has_issues": true/false,
  "issues": [
    {{
      "type": "author_initials|category|alignment|syntax|hallucination",
      "severity": "high|medium|low",
      "description": "Brief description of the issue",
      "suggested_fix": "Corrected query or category"
    }}
  ],
  "recommended_action": "keep|fix|remove"
}}

If no issues, respond with:
{{
  "has_issues": false,
  "issues": [],
  "recommended_action": "keep"
}}
"""

@dataclass
class ReviewResult:
    """Result of reviewing a single example."""
    index: int
    nl: str
    query: str
    category: str
    has_issues: bool
    issues: list[dict] = field(default_factory=list)
    recommended_action: str = "keep"
    error: str | None = None


def quick_lint(example: dict) -> list[dict]:
    """Quick local validation before LLM review."""
    issues = []
    nl = example.get("natural_language", "")
    query = example.get("ads_query", "")
    category = example.get("category", "")

    # Check for unfielded category
    if category == "unfielded":
        issues.append({
            "type": "category",
            "severity": "medium",
            "description": "Category is 'unfielded' - should be properly categorized",
            "suggested_fix": None
        })

    # Check for author initial guessing
    # Pattern: NL has just last name, query has "Last, F." format
    import re

    # Find author names in NL (simple heuristic: "by LastName")
    nl_authors = re.findall(r'\bby\s+([A-Z][a-z]+)\b', nl)

    # Find author patterns in query with initials
    query_authors_with_initials = re.findall(r'author:\s*"([^"]+,\s*[A-Z]\.?\s*[A-Z]?\.?)"', query)

    for nl_author in nl_authors:
        for q_author in query_authors_with_initials:
            if nl_author.lower() in q_author.lower():
                # Check if initials were added
                if re.search(r',\s*[A-Z]\.', q_author):
                    # Check if NL had the initial
                    nl_has_initial = re.search(rf'\b{nl_author}\s+[A-Z]\.?\b', nl, re.IGNORECASE)
                    if not nl_has_initial:
                        issues.append({
                            "type": "author_initials",
                            "severity": "high",
                            "description": f"Author '{nl_author}' in NL got initials '{q_author}' in query without user providing them",
                            "suggested_fix": f'author:"{nl_author}"'
                        })

    # Run syntax lint
    lint_result = lint_query(query)
    if not lint_result.valid:
        for error in lint_result.errors:
            issues.append({
                "type": "syntax",
                "severity": "high",
                "description": error,
                "suggested_fix": None
            })

    # Check field constraints
    constraint_result = validate_field_constraints(query)
    if not constraint_result.valid:
        for error in constraint_result.errors:
            issues.append({
                "type": "hallucination",
                "severity": "high",
                "description": str(error),
                "suggested_fix": None
            })

    # Check for abs: containing author-like content
    if re.search(r'abs:\s*\([^)]*\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', query):
        # Might be author names in abs field
        if 'by' in nl.lower() and 'author:' not in query.lower():
            issues.append({
                "type": "alignment",
                "severity": "high",
                "description": "NL mentions 'by [name]' but query uses abs: instead of author:",
                "suggested_fix": None
            })

    return issues


def review_with_llm(example: dict, client: anthropic.Anthropic) -> dict:
    """Review example using Claude."""
    prompt = REVIEW_PROMPT.format(
        nl=example.get("natural_language", ""),
        query=example.get("ads_query", ""),
        category=example.get("category", "")
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON response
        text = response.content[0].text
        # Find JSON in response
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"has_issues": False, "issues": [], "recommended_action": "keep", "parse_error": text}
    except Exception as e:
        return {"has_issues": True, "issues": [{"type": "error", "description": str(e)}], "recommended_action": "review"}


def review_batch(examples: list[dict], use_llm: bool = False, sample_size: int | None = None) -> list[ReviewResult]:
    """Review a batch of examples."""
    results = []

    if sample_size:
        import random
        examples = random.sample(examples, min(sample_size, len(examples)))

    client = None
    if use_llm:
        client = anthropic.Anthropic()

    for i, example in enumerate(examples):
        nl = example.get("natural_language", "")
        query = example.get("ads_query", "")
        category = example.get("category", "")

        # Quick local lint
        local_issues = quick_lint(example)

        # LLM review if enabled and has potential issues or random sample
        llm_result = None
        if use_llm and (local_issues or i % 10 == 0):
            llm_result = review_with_llm(example, client)

        # Combine results
        all_issues = local_issues.copy()
        if llm_result and llm_result.get("has_issues"):
            all_issues.extend(llm_result.get("issues", []))

        # Deduplicate issues by description
        seen = set()
        unique_issues = []
        for issue in all_issues:
            desc = issue.get("description", "")
            if desc not in seen:
                seen.add(desc)
                unique_issues.append(issue)

        has_issues = len(unique_issues) > 0
        action = "keep"
        if has_issues:
            high_severity = any(i.get("severity") == "high" for i in unique_issues)
            action = "fix" if high_severity else "review"

        results.append(ReviewResult(
            index=i,
            nl=nl,
            query=query,
            category=category,
            has_issues=has_issues,
            issues=unique_issues,
            recommended_action=action
        ))

        if (i + 1) % 100 == 0:
            print(f"Reviewed {i + 1}/{len(examples)} examples...")

    return results


def generate_report(results: list[ReviewResult], output_path: Path) -> dict:
    """Generate a summary report of the review."""
    total = len(results)
    with_issues = [r for r in results if r.has_issues]

    # Group by issue type
    by_type = {}
    for r in with_issues:
        for issue in r.issues:
            issue_type = issue.get("type", "unknown")
            if issue_type not in by_type:
                by_type[issue_type] = []
            by_type[issue_type].append({
                "index": r.index,
                "nl": r.nl,
                "query": r.query,
                "category": r.category,
                "issue": issue
            })

    # Group by recommended action
    by_action = {}
    for r in results:
        action = r.recommended_action
        if action not in by_action:
            by_action[action] = 0
        by_action[action] += 1

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_reviewed": total,
            "with_issues": len(with_issues),
            "issue_rate": f"{len(with_issues) / total * 100:.1f}%" if total > 0 else "0%",
            "by_action": by_action,
            "by_issue_type": {k: len(v) for k, v in by_type.items()}
        },
        "issues_by_type": by_type,
        "flagged_examples": [
            {
                "index": r.index,
                "nl": r.nl,
                "query": r.query,
                "category": r.category,
                "issues": r.issues,
                "recommended_action": r.recommended_action
            }
            for r in with_issues
        ]
    }

    # Write report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Review gold examples for quality issues")
    parser.add_argument("--input", "-i", default="data/datasets/raw/gold_examples.json",
                        help="Input JSON file to review")
    parser.add_argument("--output", "-o", default="data/datasets/evaluations/curator_review.json",
                        help="Output report file")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use Claude for additional review (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--sample", "-n", type=int, default=None,
                        help="Review only N random samples")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode - only local lint, no LLM")

    args = parser.parse_args()

    # Load examples
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} examples from {input_path}")

    # Review
    use_llm = args.use_llm and not args.quick
    if use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set, falling back to local-only review")
        use_llm = False

    print(f"Reviewing {'with LLM' if use_llm else 'local-only'}...")
    results = review_batch(examples, use_llm=use_llm, sample_size=args.sample)

    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = generate_report(results, output_path)

    # Print summary
    print("\n" + "="*60)
    print("CURATOR REVIEW SUMMARY")
    print("="*60)
    print(f"Total reviewed: {report['summary']['total_reviewed']}")
    print(f"With issues: {report['summary']['with_issues']} ({report['summary']['issue_rate']})")
    print("\nBy recommended action:")
    for action, count in report['summary']['by_action'].items():
        print(f"  {action}: {count}")
    print("\nBy issue type:")
    for issue_type, count in report['summary']['by_issue_type'].items():
        print(f"  {issue_type}: {count}")
    print(f"\nFull report saved to: {output_path}")

    # Show some examples
    if report['flagged_examples']:
        print("\n" + "-"*60)
        print("SAMPLE FLAGGED EXAMPLES (first 5):")
        print("-"*60)
        for ex in report['flagged_examples'][:5]:
            print(f"\nNL: {ex['nl']}")
            print(f"Query: {ex['query']}")
            print(f"Category: {ex['category']}")
            print(f"Issues:")
            for issue in ex['issues']:
                print(f"  - [{issue.get('severity', 'unknown')}] {issue.get('type')}: {issue.get('description')}")
            if ex['issues'] and ex['issues'][0].get('suggested_fix'):
                print(f"  Suggested fix: {ex['issues'][0]['suggested_fix']}")


if __name__ == "__main__":
    main()
