#!/usr/bin/env python3
"""LLM Curator Review for Gold Examples using GPT-5.2.

Reviews training data quality with full context on ADS/SciX syntax,
field constraints, and project goals.

Issues detected:
- Author initials being guessed when not provided
- Wrong category assignments
- Invalid query syntax
- Mismatched NL to query mappings
- Over-quoted phrases that should use boolean operators
- Exact titles in abs: that should be topic terms
- Missing doctype:article when "papers" mentioned
- Bibcode lookups (should be removed from training data)
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

try:
    import openai
except ImportError:
    print("Error: openai package required. Install with: pip install openai")
    sys.exit(1)

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))

from finetune.domains.scix.field_constraints import (
    FIELD_ENUMS, DOCTYPES, PROPERTIES, COLLECTIONS, BIBGROUPS
)
from finetune.domains.scix.validate import lint_query, validate_field_constraints

# Comprehensive documentation context for the LLM
ADS_DOCUMENTATION = """
## ADS/SciX Search Query Syntax Reference

### Valid Fields
- `abs:` - Search abstract, title, keywords (most common for topics)
- `title:` - Search title only
- `author:` - Author search (use "Last" or "Last, F" format)
- `=author:` - Exact author match
- `^author:` or `author:"^Last"` - First author search
- `aff:` - Affiliation search
- `year:` - Single year (e.g., year:2020)
- `pubdate:` - Date range (e.g., pubdate:[2020 TO 2024])
- `doctype:` - Document type filter
- `property:` - Property filter
- `collection:` - Collection/database filter (astronomy, physics, general, earthscience)
- `bibgroup:` - Bibliographic group (telescope/mission, e.g., HST, JWST)
- `object:` - Astronomical object (resolved via SIMBAD)
- `bibcode:` - Exact bibcode lookup
- `identifier:` - DOI, arXiv ID, or other identifier

### Valid Doctypes
{doctypes}

### Valid Properties
{properties}

### Valid Collections
{collections}

### Valid Bibgroups (telescopes/missions)
{bibgroups}

### Second-Order Operators
- `citations(query)` - Find papers that cite papers matching query
- `references(query)` - Find papers cited by papers matching query
- `similar(query)` - Find similar papers
- `trending(query)` - Find trending papers on topic
- `useful(query)` - Find highly useful papers
- `reviews(query)` - Find review articles

### Boolean Operators
- `AND` - Both terms required (implicit between separate abs: clauses)
- `OR` - Either term acceptable (use within parentheses: `abs:(term1 OR term2)`)
- `NOT` - Exclude term
- Parentheses for grouping

### Project Goals
This training data is for a model that translates natural language to ADS queries.
Key principles:
1. LITERAL translation - only extract what user explicitly states
2. NO fabrication - don't invent author initials, years, or details not in input
3. NO over-quoting - only quote when exact phrase is explicitly requested
4. Topics should use boolean operators, not exact phrases
5. Add `doctype:article` when user mentions "papers", "articles", "publications"
6. Bibcode lookups are NOT translation - they should be removed from training data
"""

REVIEW_PROMPT = """You are a strict data quality reviewer for ADS/SciX search query training data.
Your job is to identify training examples that would teach the model bad habits.

{documentation}

---

## Example to Review

**Natural Language Input:** {nl}
**Generated ADS Query:** {query}
**Category:** {category}

---

## Review Criteria (Check ALL of these carefully)

### 1. Author Initials Fabrication (CRITICAL)
If NL mentions author by last name only (e.g., "papers by Hawking"), query should NOT include initials.
- WRONG: `author:"Hawking, S."` (where did S. come from?)
- CORRECT: `author:"Hawking"` or `author:"^Hawking"`
Exception: If NL explicitly says "S. Hawking" or "Stephen Hawking", initials are OK.

### 2. Exact Titles in abs: (CRITICAL)
The abs: field should contain topic keywords, NOT exact paper titles.
If the abs: content looks like a paper title (capitalized words, "The X of Y", etc.), it's WRONG.
- WRONG: `abs:(Mapping the Spatial Distribution of H_2 in Nearby Galaxies)`
- CORRECT: `abs:(H2 OR "molecular hydrogen") abs:galaxies abs:Spitzer`

### 3. Over-Quoted Phrases
Only use quotes when the user explicitly requests an exact phrase.
Multi-word topics should use boolean operators (AND for all required, OR for alternatives).
- WRONG: `abs:"JWST observations of exoplanets"`
- CORRECT: `abs:(JWST AND exoplanets)` or `abs:JWST abs:exoplanets`

### 4. Boolean Logic (Important)
When breaking up topics, use:
- AND when all terms must appear (more restrictive, usually correct)
- OR when alternatives are acceptable (user says "X or Y")
- WRONG for general topics: `abs:(failed OR novae OR blue)` - too permissive!
- CORRECT: `abs:(novae AND blue)` or just the most distinctive terms

### 5. Missing doctype:article
If NL mentions "papers", "articles", "publications", "studies" (academic works),
the query should include `doctype:article`.
- NL: "papers on dark matter" → query should have `doctype:article`

### 6. Bibcode Lookups (REMOVE from training)
Queries that are just bibcode lookups are NOT translation - they're retrieval.
These should be flagged for removal.
- WRONG as training: NL "find 2020ApJ...900..123S" → query "bibcode:2020ApJ...900..123S"

### 7. Category Validity
Category should be descriptive (author, topic, operator, etc.), not "unfielded".

### 8. Query-NL Alignment
The query should represent ONLY what the NL explicitly states.
- If NL says "by Smith", use author: not abs:
- If NL mentions a year, use year: or pubdate:
- Don't add constraints the user didn't request

### 9. Syntax Validity
- Balanced parentheses and quotes
- Valid field names
- Valid enum values for doctype:, property:, collection:, bibgroup:

---

## Response Format (JSON only, no explanation)

```json
{{
  "has_issues": true/false,
  "issues": [
    {{
      "type": "author_initials|exact_title|over_quoted|boolean_logic|missing_doctype|bibcode_lookup|category|alignment|syntax",
      "severity": "critical|high|medium",
      "description": "Brief description",
      "suggested_fix": "Corrected query or null"
    }}
  ],
  "recommended_action": "keep|fix|remove"
}}
```

If no issues: `{{"has_issues": false, "issues": [], "recommended_action": "keep"}}`
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


def get_documentation_context() -> str:
    """Build comprehensive documentation context for the LLM."""
    return ADS_DOCUMENTATION.format(
        doctypes=", ".join(sorted(DOCTYPES)),
        properties=", ".join(sorted(PROPERTIES)),
        collections=", ".join(sorted(COLLECTIONS)),
        bibgroups=", ".join(sorted(BIBGROUPS))
    )


def quick_lint(example: dict) -> list[dict]:
    """Quick local validation before LLM review."""
    import re

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

    # Check for bibcode in query (these should be removed from training)
    bibcode_pattern = r'\b\d{4}[A-Za-z&]+\.{0,3}\d+[A-Z]?\.\.[.\d]+[A-Z]?\b'
    if re.search(bibcode_pattern, query):
        issues.append({
            "type": "bibcode_lookup",
            "severity": "critical",
            "description": "Query contains bibcode - this is a lookup, not NL translation",
            "suggested_fix": None
        })

    # Check for author initial guessing
    nl_authors = re.findall(r'\bby\s+([A-Z][a-z]+)\b', nl)
    query_authors_with_initials = re.findall(r'author:\s*"([^"]+,\s*[A-Z]\.?\s*[A-Z]?\.?)"', query)

    for nl_author in nl_authors:
        for q_author in query_authors_with_initials:
            if nl_author.lower() in q_author.lower():
                if re.search(r',\s*[A-Z]\.', q_author):
                    nl_has_initial = re.search(rf'\b{nl_author}\s+[A-Z]\.?\b', nl, re.IGNORECASE)
                    if not nl_has_initial:
                        issues.append({
                            "type": "author_initials",
                            "severity": "critical",
                            "description": f"Author '{nl_author}' in NL got initials '{q_author}' without user providing them",
                            "suggested_fix": f'author:"{nl_author}"'
                        })

    # Check for exact titles in abs: (long content without boolean operators)
    abs_match = re.search(r'abs:\s*\(([^)]+)\)', query)
    if abs_match:
        abs_content = abs_match.group(1)
        words = abs_content.split()
        if len(abs_content) > 50 and not re.search(r'\b(OR|AND)\b', abs_content):
            issues.append({
                "type": "exact_title",
                "severity": "high",
                "description": f"abs:() contains exact title ({len(abs_content)} chars, no boolean ops)",
                "suggested_fix": "Break into topic terms with AND operators"
            })

    # Check for missing doctype:article when "papers" mentioned
    paper_words = {"paper", "papers", "article", "articles", "publication", "publications"}
    has_paper_word = any(word in nl.lower() for word in paper_words)
    has_doctype = "doctype:" in query.lower()
    if has_paper_word and not has_doctype:
        issues.append({
            "type": "missing_doctype",
            "severity": "medium",
            "description": "NL mentions 'papers' but query missing doctype:article",
            "suggested_fix": "Add doctype:article"
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

    # Check for abs: containing author-like content when "by" is in NL
    if re.search(r'abs:\s*\([^)]*\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', query):
        if 'by' in nl.lower() and 'author:' not in query.lower():
            issues.append({
                "type": "alignment",
                "severity": "high",
                "description": "NL mentions 'by [name]' but query uses abs: instead of author:",
                "suggested_fix": None
            })

    return issues


def review_with_llm(example: dict, client: openai.OpenAI, documentation: str) -> dict:
    """Review example using GPT-5.2."""
    import re

    prompt = REVIEW_PROMPT.format(
        documentation=documentation,
        nl=example.get("natural_language", ""),
        query=example.get("ads_query", ""),
        category=example.get("category", "")
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            max_tokens=1024,
            temperature=0.1,  # Low temperature for consistent reviews
            messages=[
                {"role": "system", "content": "You are a strict data quality reviewer. Respond ONLY with valid JSON, no explanations."},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.choices[0].message.content
        # Find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"has_issues": False, "issues": [], "recommended_action": "keep", "parse_error": text}
    except Exception as e:
        return {"has_issues": True, "issues": [{"type": "error", "description": str(e)}], "recommended_action": "review"}


def review_batch(
    examples: list[dict],
    use_llm: bool = False,
    sample_size: int | None = None,
    llm_sample_rate: int = 20
) -> list[ReviewResult]:
    """Review a batch of examples.

    Args:
        examples: List of training examples to review
        use_llm: Whether to use GPT-5.2 for additional review
        sample_size: Optional limit on total examples to review
        llm_sample_rate: Review every Nth example with LLM (default 20)
    """
    results = []

    if sample_size:
        import random
        examples = random.sample(examples, min(sample_size, len(examples)))

    client = None
    documentation = None
    if use_llm:
        client = openai.OpenAI()
        documentation = get_documentation_context()

    for i, example in enumerate(examples):
        nl = example.get("natural_language", "")
        query = example.get("ads_query", "")
        category = example.get("category", "")

        # Quick local lint
        local_issues = quick_lint(example)

        # LLM review if enabled and (has potential issues OR random sample)
        llm_result = None
        if use_llm and (local_issues or i % llm_sample_rate == 0):
            llm_result = review_with_llm(example, client, documentation)

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
            critical = any(i.get("severity") == "critical" for i in unique_issues)
            high = any(i.get("severity") == "high" for i in unique_issues)
            action = "remove" if critical else ("fix" if high else "review")

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
        "model_used": "gpt-5.2",
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Review gold examples for quality issues (GPT-5.2)")
    parser.add_argument("--input", "-i", default="data/datasets/raw/gold_examples.json",
                        help="Input JSON file to review")
    parser.add_argument("--output", "-o", default="data/datasets/evaluations/curator_review_gpt52.json",
                        help="Output report file")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use GPT-5.2 for additional review (requires OPENAI_API_KEY)")
    parser.add_argument("--sample", "-n", type=int, default=None,
                        help="Review only N random samples")
    parser.add_argument("--llm-sample-rate", type=int, default=20,
                        help="Review every Nth example with LLM (default: 20)")
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
    if use_llm and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set, falling back to local-only review")
        use_llm = False

    print(f"Reviewing {'with GPT-5.2' if use_llm else 'local-only'}...")
    results = review_batch(
        examples,
        use_llm=use_llm,
        sample_size=args.sample,
        llm_sample_rate=args.llm_sample_rate
    )

    # Generate report
    output_path = Path(args.output)
    report = generate_report(results, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("CURATOR REVIEW SUMMARY (GPT-5.2)")
    print("=" * 60)
    print(f"Total reviewed: {report['summary']['total_reviewed']}")
    print(f"With issues: {report['summary']['with_issues']} ({report['summary']['issue_rate']})")
    print("\nBy recommended action:")
    for action, count in report['summary']['by_action'].items():
        print(f"  {action}: {count}")
    print("\nBy issue type:")
    for issue_type, count in sorted(report['summary']['by_issue_type'].items(), key=lambda x: -x[1]):
        print(f"  {issue_type}: {count}")
    print(f"\nFull report saved to: {output_path}")

    # Show some examples
    if report['flagged_examples']:
        print("\n" + "-" * 60)
        print("SAMPLE FLAGGED EXAMPLES (first 5):")
        print("-" * 60)
        for ex in report['flagged_examples'][:5]:
            print(f"\nNL: {ex['nl']}")
            print(f"Query: {ex['query']}")
            print(f"Category: {ex['category']}")
            print(f"Action: {ex['recommended_action']}")
            print("Issues:")
            for issue in ex['issues']:
                print(f"  - [{issue.get('severity', 'unknown')}] {issue.get('type')}: {issue.get('description')}")
                if issue.get('suggested_fix'):
                    print(f"    Fix: {issue['suggested_fix']}")


if __name__ == "__main__":
    main()
