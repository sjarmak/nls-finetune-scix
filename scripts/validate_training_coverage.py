#!/usr/bin/env python3
"""
Validate training data coverage using Claude as a judge.
Checks if gold_examples.json captures the full ADS syntax space.
"""

import json
from collections import Counter

import anthropic

def load_gold_examples() -> list[dict]:
    """Load gold examples."""
    with open("data/datasets/raw/gold_examples.json") as f:
        return json.load(f)

def get_syntax_patterns(examples: list[dict]) -> dict:
    """Extract syntax patterns from examples."""
    patterns = {
        "field_types": Counter(),
        "operators": Counter(),
        "boolean_ops": Counter(),
        "author_formats": Counter(),
        "has_quoted_phrases": 0,
        "has_ranges": 0,
        "has_boolean_combinations": 0,
    }
    
    for ex in examples:
        query = ex.get("query", "")
        
        # Field types
        for field in ["author:", "^author:", "abs:", "title:", "pubdate:", "bibstem:", 
                      "aff:", "object:", "data:", "doi:", "arXiv:", "identifier:",
                      "citation_count:", "read_count:", "author_count:", "orcid:",
                      "arxiv_class:", "doctype:", "property:", "database:", "full:",
                      "volume:", "issue:", "page:", "eid:"]:
            if field in query:
                patterns["field_types"][field.rstrip(":")] += 1
        
        # Operators
        for op in ["trending(", "similar(", "useful(", "reviews(", "citations(", 
                   "references(", "topn("]:
            if op in query:
                patterns["operators"][op.rstrip("(")] += 1
        
        # Boolean ops
        for bool_op in [" OR ", " AND ", " NOT ", " -"]:
            if bool_op in query:
                patterns["boolean_ops"][bool_op.strip()] += 1
        
        # Author formats
        if "author:" in query:
            # Extract author format
            import re
            match = re.search(r'author:([^"\s]*(?:"[^"]*")?)', query)
            if match:
                author_part = match.group(1)
                if '"' in author_part:
                    if ", " in author_part:
                        patterns["author_formats"]["lastname, initial"] += 1
                    elif "," in author_part:
                        patterns["author_formats"]["lastname,initial"] += 1
                    else:
                        patterns["author_formats"]["quoted_name"] += 1
                else:
                    patterns["author_formats"]["unquoted_lastname"] += 1
            
            # First author marker
            if "^author:" in query:
                patterns["author_formats"]["^author (first author)"] += 1
        
        if '"' in query:
            patterns["has_quoted_phrases"] += 1
        
        if "[" in query and " TO " in query:
            patterns["has_ranges"] += 1
        
        if any(op in query for op in [" OR ", " AND ", "NOT ", "-"]):
            patterns["has_boolean_combinations"] += 1
    
    return patterns

def ask_claude_for_gaps(examples: list[dict], patterns: dict) -> str:
    """Ask Claude to identify potential gaps in training data."""
    client = anthropic.Anthropic()
    
    # Sample a few examples to include in the prompt
    sample = json.dumps(examples[:10], indent=2)
    patterns_str = json.dumps({k: dict(v) if isinstance(v, Counter) else v for k, v in patterns.items()}, indent=2)
    
    prompt = f"""You are an expert in ADS (Astrophysics Data System) search query syntax.

Here are statistics from {len(examples)} training examples for a model that translates natural language to ADS queries:

PATTERNS FOUND:
{patterns_str}

SAMPLE EXAMPLES:
{sample}

Please analyze if these examples adequately cover the ADS syntax space. Specifically:

1. **Author queries**: Are there enough variations of lastname-only, "Lastname, Initial", first author (^author:), author_count, etc.?
2. **Field coverage**: Are all important ADS fields represented (author, abstract, title, publication, citations, etc.)?
3. **Operators**: Do we have enough examples of second-order operators (trending, similar, citations, reviews, useful, references)?
4. **Boolean logic**: Are AND, OR, NOT combinations adequately represented?
5. **Range queries**: Do we have pubdate ranges, citation_count ranges, etc.?
6. **Data scope**: Are there papers from different domains (cosmology, exoplanets, stellar evolution, gravitational waves, etc.)?
7. **Edge cases**: Missing patterns like wildcards (*), nested operators, NEAR proximity, etc.?

Return ONLY a JSON object like:
{{
  "coverage_score": 0.0-1.0,
  "strengths": ["strength1", "strength2"],
  "critical_gaps": ["gap1", "gap2"],
  "medium_gaps": ["gap3"],
  "recommendations": ["rec1", "rec2"],
  "author_pattern_assessment": "detailed assessment of author query coverage"
}}

Be specific about what's missing and why it matters for training the model.
"""
    
    response = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

def main():
    examples = load_gold_examples()
    patterns = get_syntax_patterns(examples)
    
    print("=" * 70)
    print("TRAINING DATA COVERAGE ANALYSIS")
    print("=" * 70)
    print(f"\nTotal examples: {len(examples)}")
    
    print("\n" + "=" * 70)
    print("FIELD COVERAGE")
    print("=" * 70)
    for field, count in patterns["field_types"].most_common(15):
        print(f"  {field:25} {count:4} examples")
    
    print("\n" + "=" * 70)
    print("AUTHOR FORMAT PATTERNS")
    print("=" * 70)
    for fmt, count in patterns["author_formats"].most_common():
        print(f"  {fmt:30} {count:4} examples")
    
    print("\n" + "=" * 70)
    print("OPERATOR COVERAGE")
    print("=" * 70)
    for op, count in patterns["operators"].most_common():
        print(f"  {op:25} {count:4} examples")
    
    print("\n" + "=" * 70)
    print("STRUCTURAL PATTERNS")
    print("=" * 70)
    print(f"  Quoted phrases:           {patterns['has_quoted_phrases']:4} examples")
    print(f"  Range queries:            {patterns['has_ranges']:4} examples")
    print(f"  Boolean combinations:     {patterns['has_boolean_combinations']:4} examples")
    
    print("\n" + "=" * 70)
    print("CLAUDE ASSESSMENT")
    print("=" * 70)
    print("\nAnalyzing with Claude Opus 4.1...")
    assessment = ask_claude_for_gaps(examples, patterns)
    
    # Parse and pretty-print the assessment
    try:
        import json
        assessment_obj = json.loads(assessment)
        print(f"\n✓ Coverage Score: {assessment_obj.get('coverage_score', 'N/A')}")
        
        print("\n✓ Strengths:")
        for strength in assessment_obj.get('strengths', []):
            print(f"  • {strength}")
        
        print("\n⚠ Critical Gaps:")
        for gap in assessment_obj.get('critical_gaps', []):
            print(f"  • {gap}")
        
        if assessment_obj.get('medium_gaps'):
            print("\n⚠ Medium Priority Gaps:")
            for gap in assessment_obj.get('medium_gaps', []):
                print(f"  • {gap}")
        
        print("\n💡 Recommendations:")
        for rec in assessment_obj.get('recommendations', []):
            print(f"  • {rec}")
        
        if 'author_pattern_assessment' in assessment_obj:
            print("\n📋 Author Pattern Assessment:")
            print(f"  {assessment_obj['author_pattern_assessment']}")
        
        # Save full assessment
        with open("data/datasets/coverage_assessment.json", "w") as f:
            json.dump(assessment_obj, f, indent=2)
        print("\n✓ Full assessment saved to data/datasets/coverage_assessment.json")
    except json.JSONDecodeError:
        print(assessment)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
