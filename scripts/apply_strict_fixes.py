#!/usr/bin/env python3
"""Apply strict curation fixes to gold_examples.json.

Fixes:
1. Add doctype:article when NL contains "papers", "articles", "publications"
2. Remove bibcode examples (save separately for recommendation system)
3. Break up quoted phrases into boolean expressions
4. Break up multi-word abs:() without operators into OR expressions
"""

import json
import re
from pathlib import Path
from copy import deepcopy
from datetime import datetime


def has_paper_mention(nl: str) -> bool:
    """Check if NL mentions papers/articles/publications."""
    paper_words = {
        "paper", "papers",
        "article", "articles",
        "publication", "publications",
        "study", "studies",  # Often implies article
    }
    nl_lower = nl.lower()
    return any(re.search(rf'\b{word}\b', nl_lower) for word in paper_words)


def has_bibcode(query: str) -> bool:
    """Check if query contains a bibcode."""
    # Bibcode patterns: YYYY[journal][volume][page][author initial]
    bibcode_pattern = r'\b\d{4}[A-Za-z&]+\.{0,3}\d+[A-Z]?\.\.[.\d]+[A-Z]?\b'
    return bool(re.search(bibcode_pattern, query))


def add_doctype_article(query: str) -> str:
    """Add doctype:article to query if not present."""
    if "doctype:" in query.lower():
        return query
    return query.strip() + " doctype:article"


def break_quoted_phrase(phrase: str, max_words: int = 2) -> str:
    """Break a quoted phrase into AND expression if too long.

    Args:
        phrase: The phrase content (without quotes)
        max_words: Max words to keep as quoted phrase

    Returns:
        Either "phrase" if short enough, or (word1 AND word2 AND ...) if long
    """
    words = phrase.split()

    # Keep short phrases quoted
    if len(words) <= max_words:
        return f'"{phrase}"'

    # Extended stopwords - filter out generic/common words aggressively
    stopwords = {
        # Articles and prepositions
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
        "by", "from", "and", "or", "is", "are", "was", "were", "be",
        "its", "their", "this", "that", "these", "those", "as", "into",
        # Generic research terms
        "observations", "studies", "research", "analysis", "data",
        "results", "survey", "study", "using", "based", "new", "recent",
        "first", "toward", "towards", "between", "among", "within",
        # Very common astronomy terms (too generic to be useful alone)
        "observations", "properties", "characteristics", "nature",
        "evidence", "detection", "discovery", "search", "finding",
        # Adjectives that are too generic
        "failed", "blue", "red", "large", "small", "high", "low",
        "early", "late", "young", "old", "bright", "faint", "massive",
    }

    # Extract only distinctive/unique terms (longer, less common words)
    meaningful = [
        w for w in words
        if w.lower() not in stopwords
        and len(w) > 3  # Require 4+ characters for uniqueness
        and not w.isdigit()  # Skip pure numbers
    ]

    # If we filtered too much, be less aggressive
    if len(meaningful) < 2:
        meaningful = [
            w for w in words
            if w.lower() not in {"the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from", "and", "or"}
            and len(w) > 2
        ][:3]

    # Limit to 3 most distinctive terms (AND is restrictive, don't over-filter)
    meaningful = meaningful[:3]

    if len(meaningful) == 0:
        # Fallback: just use first significant word
        for w in words:
            if len(w) > 2:
                return w
        return words[0] if words else ""

    if len(meaningful) == 1:
        return meaningful[0]

    # Use AND - requires ALL terms to match (more precise)
    return "(" + " AND ".join(meaningful) + ")"


def fix_quoted_phrases(query: str) -> tuple[str, list[str]]:
    """Break up over-quoted phrases in abs: and title: fields."""
    fixes = []
    result = query

    # Pattern for quoted content in abs: or title:
    pattern = r'(abs|title):\s*"([^"]+)"'

    def replace_quoted(match):
        field = match.group(1)
        phrase = match.group(2)
        words = phrase.split()

        # Keep short phrases (1-2 meaningful words)
        if len(words) <= 2:
            return match.group(0)

        # Break long phrases
        new_content = break_quoted_phrase(phrase)
        fixes.append(f"Broke up '{phrase[:30]}...' into {new_content[:50]}")
        return f"{field}:{new_content}"

    result = re.sub(pattern, replace_quoted, result)
    return result, fixes


def fix_abs_parentheses(query: str) -> tuple[str, list[str]]:
    """Fix abs:(...) content without boolean operators."""
    fixes = []
    result = query

    # Pattern for abs:(...) without quotes inside
    pattern = r'abs:\s*\(([^)]+)\)'

    def replace_abs(match):
        content = match.group(1)

        # Skip if already has boolean operators
        if re.search(r'\b(OR|AND|NOT)\b', content, re.IGNORECASE):
            return match.group(0)

        # Skip if it's a quoted phrase inside
        if content.startswith('"') and content.endswith('"'):
            return match.group(0)

        words = content.split()

        # Skip short content
        if len(words) <= 2:
            return match.group(0)

        # Check if this looks like a title (has articles, prepositions in sequence)
        title_indicators = ["the", "of", "in", "a", "an", "for", "with", "by", "to"]
        title_word_count = sum(1 for w in words if w.lower() in title_indicators)

        if title_word_count >= 2 or len(words) > 5:
            # This looks like a title - extract key terms
            # Extended stopwords for better filtering
            stopwords = {
                "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
                "by", "from", "and", "or", "is", "are", "was", "were", "be",
                "its", "their", "this", "that", "these", "those", "as", "into",
                # Generic terms
                "observations", "properties", "characteristics", "nature",
                "evidence", "detection", "discovery", "search", "finding",
                "new", "recent", "first", "toward", "towards",
                # Too-generic adjectives
                "failed", "blue", "red", "large", "small", "high", "low",
            }
            # Only keep distinctive terms (4+ chars)
            meaningful = [
                w for w in words
                if w.lower() not in stopwords
                and len(w) > 3
                and not w.isdigit()
            ]

            if len(meaningful) >= 2:
                # Take up to 3 most distinctive terms with AND
                terms = meaningful[:3]
                new_content = " AND ".join(terms)
                fixes.append(f"Extracted terms from title-like content: {new_content}")
                return f"abs:({new_content})"

        return match.group(0)

    result = re.sub(pattern, replace_abs, result)
    return result, fixes


def process_example(example: dict) -> tuple[dict | None, list[str], bool]:
    """Process a single example.

    Returns:
        Tuple of (fixed_example or None if removed, list of fixes, was_removed)
    """
    nl = example.get("natural_language", "")
    query = example.get("ads_query", "")
    fixes = []

    # Check for bibcode - these get removed
    if has_bibcode(query):
        return None, ["Removed: contains bibcode"], True

    # Start with original
    fixed = deepcopy(example)

    # Add doctype:article if NL mentions papers
    if has_paper_mention(nl) and "doctype:" not in query.lower():
        fixed["ads_query"] = add_doctype_article(query)
        fixes.append("Added doctype:article")
        query = fixed["ads_query"]

    # Fix quoted phrases
    new_query, phrase_fixes = fix_quoted_phrases(query)
    if phrase_fixes:
        fixed["ads_query"] = new_query
        fixes.extend(phrase_fixes)
        query = new_query

    # Fix abs:() without operators
    new_query, abs_fixes = fix_abs_parentheses(query)
    if abs_fixes:
        fixed["ads_query"] = new_query
        fixes.extend(abs_fixes)

    return fixed, fixes, False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply strict fixes to gold examples")
    parser.add_argument("--input", "-i", default="data/datasets/raw/gold_examples.json")
    parser.add_argument("--output", "-o", default="data/datasets/raw/gold_examples.json")
    parser.add_argument("--removed-output", default="data/datasets/raw/bibcode_examples.json",
                        help="Where to save removed bibcode examples")
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--backup", "-b", action="store_true")

    args = parser.parse_args()

    # Load
    with open(args.input) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")

    # Process
    fixed_examples = []
    removed_examples = []
    all_fixes = []
    stats = {
        "total": len(examples),
        "kept": 0,
        "removed": 0,
        "modified": 0,
        "doctype_added": 0,
        "phrases_fixed": 0,
        "abs_fixed": 0,
    }

    for i, example in enumerate(examples):
        fixed, fixes, was_removed = process_example(example)

        if was_removed:
            removed_examples.append(example)
            stats["removed"] += 1
        else:
            fixed_examples.append(fixed)
            stats["kept"] += 1

            if fixes:
                stats["modified"] += 1
                for fix in fixes:
                    if "doctype:article" in fix:
                        stats["doctype_added"] += 1
                    elif "Broke up" in fix:
                        stats["phrases_fixed"] += 1
                    elif "Extracted terms" in fix:
                        stats["abs_fixed"] += 1

                all_fixes.append({
                    "index": i,
                    "nl": example.get("natural_language", ""),
                    "original_query": example.get("ads_query", ""),
                    "fixed_query": fixed.get("ads_query", ""),
                    "fixes": fixes
                })

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(examples)}...")

    # Print summary
    print("\n" + "="*60)
    print("STRICT FIX SUMMARY")
    print("="*60)
    print(f"Total: {stats['total']}")
    print(f"Kept: {stats['kept']}")
    print(f"Removed (bibcodes): {stats['removed']}")
    print(f"Modified: {stats['modified']}")
    print(f"  - doctype:article added: {stats['doctype_added']}")
    print(f"  - Quoted phrases fixed: {stats['phrases_fixed']}")
    print(f"  - abs:() content fixed: {stats['abs_fixed']}")

    # Show samples
    print("\n" + "-"*60)
    print("SAMPLE FIXES (first 10):")
    print("-"*60)
    for fix in all_fixes[:10]:
        print(f"\nNL: {fix['nl']}")
        print(f"Original: {fix['original_query']}")
        print(f"Fixed: {fix['fixed_query']}")
        print(f"Changes: {', '.join(fix['fixes'])}")

    print("\n" + "-"*60)
    print("REMOVED EXAMPLES (first 5):")
    print("-"*60)
    for ex in removed_examples[:5]:
        print(f"\nNL: {ex.get('natural_language', '')}")
        print(f"Query: {ex.get('ads_query', '')}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return

    # Backup if requested
    if args.backup:
        backup_path = Path(args.input).with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(backup_path, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"\nBackup saved to: {backup_path}")

    # Save fixed examples
    with open(args.output, "w") as f:
        json.dump(fixed_examples, f, indent=2)
    print(f"Fixed examples ({len(fixed_examples)}) saved to: {args.output}")

    # Save removed examples (for recommendation system)
    with open(args.removed_output, "w") as f:
        json.dump(removed_examples, f, indent=2)
    print(f"Removed examples ({len(removed_examples)}) saved to: {args.removed_output}")

    # Save fix report
    report_path = "data/datasets/evaluations/strict_fix_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "stats": stats,
            "fixes": all_fixes[:200],  # First 200 for review
            "removed_count": len(removed_examples)
        }, f, indent=2)
    print(f"Fix report saved to: {report_path}")


if __name__ == "__main__":
    main()
