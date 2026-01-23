#!/usr/bin/env python3
"""Strict curation of gold_examples.json.

Identifies and flags examples that violate training data quality criteria:

1. NO exact titles in abs: - topics should be broken into terms with OR/AND
2. NO bibcodes/identifiers as query output - these are lookups, not NL translation
3. NO over-quoted phrases - only quote when exact phrase is explicitly requested
4. NO over-inference - only extract what's explicitly stated
5. YES doctype:article when "papers/articles/publications" mentioned
6. YES break topics into boolean expressions (OR for alternatives, AND for requirements)
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy

@dataclass
class CurationIssue:
    """A curation issue found in an example."""
    issue_type: str
    severity: str  # "remove", "fix", "review"
    description: str
    suggested_fix: str | None = None


@dataclass
class CurationResult:
    """Result of curating a single example."""
    index: int
    nl: str
    query: str
    category: str
    action: str  # "keep", "fix", "remove", "review"
    issues: list[CurationIssue] = field(default_factory=list)
    fixed_query: str | None = None
    fixed_category: str | None = None


def check_bibcode_in_query(query: str) -> CurationIssue | None:
    """Check if query contains a bibcode instead of proper search syntax."""
    # Bibcode patterns: YYYY[journal][volume][page][author initial]
    # e.g., 2022A&A...658L..10L, 2020ApJ...900..123S
    bibcode_pattern = r'\b\d{4}[A-Za-z&]+\.{0,3}\d+[A-Z]?\.\.[.\d]+[A-Z]?\b'

    if re.search(bibcode_pattern, query):
        return CurationIssue(
            issue_type="bibcode_as_query",
            severity="remove",
            description="Query contains bibcode - this is a lookup, not NL translation",
            suggested_fix=None
        )
    return None


def check_exact_title_in_abs(nl: str, query: str) -> CurationIssue | None:
    """Check if abs: contains what looks like an exact paper title."""
    # Extract content from abs:(...)
    abs_match = re.search(r'abs:\s*\(([^)]+)\)', query)
    if not abs_match:
        return None

    abs_content = abs_match.group(1)

    # Signs of exact title:
    # 1. Very long (>50 chars) with no boolean operators
    # 2. Contains articles/prepositions typical of titles
    # 3. Capitalized words in sequence

    if len(abs_content) > 50 and not re.search(r'\b(OR|AND)\b', abs_content):
        return CurationIssue(
            issue_type="exact_title_in_abs",
            severity="fix",
            description=f"abs:() contains what looks like exact title ({len(abs_content)} chars, no boolean ops)",
            suggested_fix="Break into topic terms with OR/AND operators"
        )

    # Check for title-like patterns (The X of Y, Mapping X in Y, etc.)
    title_patterns = [
        r'^(The|A|An)\s+\w+',  # Starts with article
        r'\b(of|in|for|with|from|by|on|to)\s+the\b',  # "of the", "in the"
        r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # Multiple capitalized words
    ]

    for pattern in title_patterns:
        if re.search(pattern, abs_content):
            return CurationIssue(
                issue_type="exact_title_in_abs",
                severity="fix",
                description="abs:() contains title-like phrase pattern",
                suggested_fix="Extract key topic terms only"
            )

    return None


def check_over_quoted_phrase(query: str) -> CurationIssue | None:
    """Check for unnecessarily quoted multi-word phrases."""
    # Find quoted phrases in abs:/title: fields
    quoted_pattern = r'(abs|title):\s*"([^"]+)"'

    for match in re.finditer(quoted_pattern, query):
        phrase = match.group(2)
        words = phrase.split()

        # Flag if:
        # 1. More than 3 words (unlikely to need exact phrase)
        # 2. Contains common words that don't need exact matching
        if len(words) > 3:
            return CurationIssue(
                issue_type="over_quoted",
                severity="review",
                description=f"Long quoted phrase '{phrase}' - consider breaking into terms",
                suggested_fix=f"abs:({' OR '.join(words[:3])})"
            )

        # Check for generic terms that don't need quoting
        generic_terms = {"observations", "studies", "research", "analysis", "data", "results"}
        if any(w.lower() in generic_terms for w in words) and len(words) == 2:
            return CurationIssue(
                issue_type="over_quoted",
                severity="review",
                description=f"Quoted phrase '{phrase}' contains generic term",
                suggested_fix=f"Just use the specific term: abs:{words[0]}"
            )

    return None


def check_missing_doctype(nl: str, query: str) -> CurationIssue | None:
    """Check if 'papers/articles' in NL but no doctype in query."""
    paper_words = {"paper", "papers", "article", "articles", "publication", "publications"}
    nl_lower = nl.lower()

    has_paper_word = any(word in nl_lower for word in paper_words)
    has_doctype = "doctype:" in query.lower()

    if has_paper_word and not has_doctype:
        return CurationIssue(
            issue_type="missing_doctype",
            severity="fix",
            description="NL mentions 'papers/articles' but query missing doctype:article",
            suggested_fix="Add doctype:article"
        )

    return None


def check_over_inference(nl: str, query: str) -> CurationIssue | None:
    """Check for inferred content not explicitly in NL."""
    # Extract years from query
    query_years = set(re.findall(r'\b(19|20)\d{2}\b', query))
    nl_years = set(re.findall(r'\b(19|20)\d{2}\b', nl))

    # Years in query but not in NL = over-inference
    inferred_years = query_years - nl_years
    if inferred_years:
        return CurationIssue(
            issue_type="over_inference",
            severity="review",
            description=f"Query contains years {inferred_years} not in NL",
            suggested_fix="Remove inferred years"
        )

    return None


def check_abs_without_operators(query: str) -> CurationIssue | None:
    """Check for multi-term abs: without boolean operators."""
    # Match abs:"multi word phrase" or abs:(multi word phrase)
    patterns = [
        r'abs:\s*"([^"]+)"',
        r'abs:\s*\(([^)]+)\)'
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            content = match.group(1)
            words = content.split()

            # If 3+ words and no boolean operators, flag it
            if len(words) >= 3 and not re.search(r'\b(OR|AND|NOT)\b', content, re.IGNORECASE):
                return CurationIssue(
                    issue_type="missing_boolean_ops",
                    severity="review",
                    description=f"Multi-word abs content without boolean operators: '{content[:50]}...'",
                    suggested_fix=f"Consider: abs:({' OR '.join(words[:4])})"
                )

    return None


def curate_example(index: int, example: dict) -> CurationResult:
    """Apply all curation checks to an example."""
    nl = example.get("natural_language", "")
    query = example.get("ads_query", "")
    category = example.get("category", "")

    issues = []

    # Run all checks
    checks = [
        check_bibcode_in_query(query),
        check_exact_title_in_abs(nl, query),
        check_over_quoted_phrase(query),
        check_missing_doctype(nl, query),
        check_over_inference(nl, query),
        check_abs_without_operators(query),
    ]

    for issue in checks:
        if issue:
            issues.append(issue)

    # Determine action
    if any(i.severity == "remove" for i in issues):
        action = "remove"
    elif any(i.severity == "fix" for i in issues):
        action = "fix"
    elif any(i.severity == "review" for i in issues):
        action = "review"
    else:
        action = "keep"

    return CurationResult(
        index=index,
        nl=nl,
        query=query,
        category=category,
        action=action,
        issues=issues
    )


def curate_dataset(examples: list[dict]) -> tuple[list[CurationResult], dict]:
    """Curate entire dataset."""
    results = []
    stats = {
        "total": len(examples),
        "keep": 0,
        "fix": 0,
        "remove": 0,
        "review": 0,
        "by_issue_type": {}
    }

    for i, example in enumerate(examples):
        result = curate_example(i, example)
        results.append(result)
        stats[result.action] += 1

        for issue in result.issues:
            issue_type = issue.issue_type
            if issue_type not in stats["by_issue_type"]:
                stats["by_issue_type"][issue_type] = 0
            stats["by_issue_type"][issue_type] += 1

        if (i + 1) % 500 == 0:
            print(f"Curated {i + 1}/{len(examples)} examples...")

    return results, stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Strict curation of gold examples")
    parser.add_argument("--input", "-i", default="data/datasets/raw/gold_examples.json")
    parser.add_argument("--output", "-o", default="data/datasets/evaluations/strict_curation.json")
    parser.add_argument("--remove-flagged", action="store_true",
                        help="Output cleaned dataset without flagged examples")
    parser.add_argument("--clean-output", default=None,
                        help="Path to write cleaned dataset")

    args = parser.parse_args()

    # Load
    with open(args.input) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")

    # Curate
    results, stats = curate_dataset(examples)

    # Print summary
    print("\n" + "="*60)
    print("STRICT CURATION SUMMARY")
    print("="*60)
    print(f"Total: {stats['total']}")
    print(f"  Keep: {stats['keep']} ({stats['keep']/stats['total']*100:.1f}%)")
    print(f"  Fix: {stats['fix']} ({stats['fix']/stats['total']*100:.1f}%)")
    print(f"  Remove: {stats['remove']} ({stats['remove']/stats['total']*100:.1f}%)")
    print(f"  Review: {stats['review']} ({stats['review']/stats['total']*100:.1f}%)")
    print("\nBy issue type:")
    for issue_type, count in sorted(stats["by_issue_type"].items(), key=lambda x: -x[1]):
        print(f"  {issue_type}: {count}")

    # Save report
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": stats,
        "flagged": [
            {
                "index": r.index,
                "nl": r.nl,
                "query": r.query,
                "category": r.category,
                "action": r.action,
                "issues": [
                    {
                        "type": i.issue_type,
                        "severity": i.severity,
                        "description": i.description,
                        "suggested_fix": i.suggested_fix
                    }
                    for i in r.issues
                ]
            }
            for r in results if r.action != "keep"
        ]
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output}")

    # Show samples
    print("\n" + "-"*60)
    print("SAMPLE ISSUES:")
    print("-"*60)

    for action in ["remove", "fix", "review"]:
        samples = [r for r in results if r.action == action][:3]
        if samples:
            print(f"\n=== {action.upper()} ({len([r for r in results if r.action == action])} total) ===")
            for r in samples:
                print(f"\nNL: {r.nl}")
                print(f"Query: {r.query}")
                for issue in r.issues:
                    print(f"  [{issue.severity}] {issue.issue_type}: {issue.description}")
                    if issue.suggested_fix:
                        print(f"    Fix: {issue.suggested_fix}")

    # Write cleaned dataset if requested
    if args.remove_flagged and args.clean_output:
        keep_indices = {r.index for r in results if r.action == "keep"}
        cleaned = [ex for i, ex in enumerate(examples) if i in keep_indices]
        with open(args.clean_output, "w") as f:
            json.dump(cleaned, f, indent=2)
        print(f"\nCleaned dataset ({len(cleaned)} examples) saved to: {args.clean_output}")


if __name__ == "__main__":
    main()
