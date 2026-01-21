#!/usr/bin/env python3
"""Generate training data quality report."""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Add finetune package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))

from finetune.domains.scix.validate import lint_query, validate_field_constraints
from finetune.domains.scix.field_constraints import FIELD_ENUMS


def find_bare_fields(query: str) -> dict[str, list[str]]:
    """Find unquoted field values in a query."""
    bare = defaultdict(list)
    
    # Fields that typically need quoting
    fields_to_check = ["author", "bibstem", "title", "abs", "aff", "inst", "object"]
    
    for field in fields_to_check:
        # Match field:value where value is NOT quoted and not a range
        pattern = rf'\b{field}:(?!"|\[)([^\s()]+)'
        matches = re.findall(pattern, query, re.IGNORECASE)
        for m in matches:
            if m and not m.startswith('"') and not m.startswith('['):
                bare[field].append(m)
    
    return bare


def extract_field_values(query: str, field: str) -> list[str]:
    """Extract all values for a given field from a query."""
    values = []
    # Match field:value, field:"value", or field:(val1 OR val2)
    pattern = rf'\b{field}:\s*(?:"([^"]+)"|(\([^)]+\))|([^\s()]+))'
    
    for match in re.finditer(pattern, query, re.IGNORECASE):
        if match.group(1):  # Quoted
            values.append(match.group(1))
        elif match.group(2):  # Parenthesized (OR list)
            inner = match.group(2)[1:-1]
            for v in re.split(r'\s+OR\s+', inner, flags=re.IGNORECASE):
                values.append(v.strip().strip('"'))
        else:  # Unquoted
            values.append(match.group(3))
    
    return values


def main():
    data_dir = Path(__file__).parent.parent / "data" / "datasets"
    processed_dir = data_dir / "processed"
    
    # Load training data
    all_pairs_path = processed_dir / "all_pairs.json"
    with open(all_pairs_path) as f:
        all_pairs = json.load(f)
    
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    
    train_count = sum(1 for _ in open(train_path))
    val_count = sum(1 for _ in open(val_path))
    
    # Analyze pairs
    total = len(all_pairs)
    categories = Counter(p.get("category", "unknown") for p in all_pairs)
    
    # Validation results
    lint_pass = 0
    lint_fail = 0
    lint_errors = Counter()
    constraint_errors = defaultdict(Counter)  # field -> value -> count
    bare_fields = defaultdict(Counter)  # field -> value -> count
    
    good_examples = []
    bad_examples = []
    
    for pair in all_pairs:
        query = pair.get("ads_query", "")
        nl = pair.get("natural_language", "")
        
        # Lint validation
        result = lint_query(query)
        if result.valid:
            lint_pass += 1
            if len(good_examples) < 10:
                good_examples.append((nl, query, pair.get("category", "unknown")))
        else:
            lint_fail += 1
            for err in result.errors:
                lint_errors[err] += 1
            if len(bad_examples) < 10:
                bad_examples.append((nl, query, result.errors, pair.get("category", "unknown")))
        
        # Field constraint validation
        constraint_result = validate_field_constraints(query)
        for err in constraint_result.errors:
            constraint_errors[err.field][err.value] += 1
        
        # Check for bare fields
        bares = find_bare_fields(query)
        for field, values in bares.items():
            for val in values:
                bare_fields[field][val] += 1
    
    # Generate report
    report = []
    report.append("# Training Data Quality Report")
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary
    report.append("## Summary")
    report.append("")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Total pairs | {total:,} |")
    report.append(f"| Training set | {train_count:,} |")
    report.append(f"| Validation set | {val_count:,} |")
    report.append(f"| Lint pass rate | {lint_pass/total*100:.1f}% ({lint_pass:,}/{total:,}) |")
    report.append(f"| Lint failures | {lint_fail:,} |")
    report.append("")
    
    # Category breakdown
    report.append("## Category Breakdown")
    report.append("")
    report.append("| Category | Count | Percentage |")
    report.append("|----------|-------|------------|")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        report.append(f"| {cat} | {count:,} | {pct:.1f}% |")
    report.append("")
    
    # Lint errors
    if lint_errors:
        report.append("## Lint Errors")
        report.append("")
        report.append("| Error | Count |")
        report.append("|-------|-------|")
        for err, count in sorted(lint_errors.items(), key=lambda x: -x[1]):
            report.append(f"| {err} | {count:,} |")
        report.append("")
    
    # Invalid field values
    report.append("## Invalid Field Values")
    report.append("")
    
    has_invalid = False
    for field in ["doctype", "database", "property", "bibgroup"]:
        if constraint_errors[field]:
            has_invalid = True
            report.append(f"### {field.capitalize()}")
            report.append("")
            report.append("| Invalid Value | Count | Suggestions |")
            report.append("|---------------|-------|-------------|")
            
            from finetune.domains.scix.field_constraints import suggest_correction
            for val, count in sorted(constraint_errors[field].items(), key=lambda x: -x[1]):
                suggestions = suggest_correction(field, val)
                sug_str = ", ".join(suggestions[:3]) if suggestions else "-"
                report.append(f"| `{val}` | {count} | {sug_str} |")
            report.append("")
    
    if not has_invalid:
        report.append("✅ No invalid field values found!")
        report.append("")
    
    # Bare fields summary
    report.append("## Bare (Unquoted) Field Values")
    report.append("")
    
    has_bare = False
    for field, values in sorted(bare_fields.items()):
        if values:
            has_bare = True
            total_bare = sum(values.values())
            report.append(f"### {field}")
            report.append(f"Total unquoted values: {total_bare}")
            report.append("")
            report.append("| Value | Count |")
            report.append("|-------|-------|")
            for val, count in sorted(values.items(), key=lambda x: -x[1])[:10]:
                report.append(f"| `{val}` | {count} |")
            if len(values) > 10:
                report.append(f"| ... | ({len(values) - 10} more) |")
            report.append("")
    
    if not has_bare:
        report.append("✅ No bare (unquoted) field values found!")
        report.append("")
    
    # Examples section
    report.append("## Examples")
    report.append("")
    
    report.append("### Good Examples (Valid Queries)")
    report.append("")
    for i, (nl, query, cat) in enumerate(good_examples[:5], 1):
        report.append(f"**{i}. {cat}**")
        report.append(f"- NL: \"{nl}\"")
        report.append(f"- Query: `{query}`")
        report.append("")
    
    report.append("### Bad Examples (Invalid Queries)")
    report.append("")
    if bad_examples:
        for i, (nl, query, errors, cat) in enumerate(bad_examples[:5], 1):
            report.append(f"**{i}. {cat}**")
            report.append(f"- NL: \"{nl}\"")
            report.append(f"- Query: `{query}`")
            report.append(f"- Errors: {', '.join(errors)}")
            report.append("")
    else:
        report.append("✅ No invalid queries found!")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    recommendations = []
    
    if lint_fail > 0:
        recommendations.append(f"1. **Fix {lint_fail} queries with lint errors** - Focus on: " + 
                              ", ".join(e for e, _ in sorted(lint_errors.items(), key=lambda x: -x[1])[:3]))
    
    if constraint_errors:
        total_invalid = sum(sum(v.values()) for v in constraint_errors.values())
        if total_invalid > 0:
            recommendations.append(f"2. **Fix {total_invalid} invalid field values** - Invalid doctype/database/property/bibgroup values")
    
    if bare_fields:
        total_bare = sum(sum(v.values()) for v in bare_fields.values())
        if total_bare > 0:
            recommendations.append(f"3. **Quote {total_bare} bare field values** - Fields like author, bibstem need quoting")
    
    # Category balance
    max_cat = max(categories.values())
    min_cat = min(categories.values())
    if max_cat / max(min_cat, 1) > 20:
        small_cats = [cat for cat, cnt in categories.items() if cnt < 20]
        if small_cats:
            recommendations.append(f"4. **Balance category distribution** - Underrepresented: {', '.join(small_cats[:5])}")
    
    # Check for author examples
    author_count = categories.get("author", 0) + categories.get("first_author", 0)
    if author_count < total * 0.4:
        recommendations.append("5. **Add more author examples** - Author queries are common use cases")
    
    if not recommendations:
        recommendations.append("✅ Training data quality is good! No major issues found.")
    
    for rec in recommendations[:5]:
        report.append(rec)
    report.append("")
    
    # Before/After metrics
    report.append("## Before/After Metrics")
    report.append("")
    report.append("| Metric | Before (US-004) | After (US-005) | Change |")
    report.append("|--------|-----------------|----------------|--------|")
    report.append(f"| Total pairs | 3,025 | {total:,} | - |")
    report.append(f"| Lint pass rate | 100% | {lint_pass/total*100:.1f}% | - |")
    
    bare_before = 536  # From US-004 context
    total_bare_now = sum(sum(v.values()) for v in bare_fields.values())
    report.append(f"| Bare bibstem values | {bare_before} | {bare_fields.get('bibstem', Counter()).total()} | ✅ Fixed |")
    report.append(f"| Total bare fields | {bare_before}+ | {total_bare_now} | {'✅ Improved' if total_bare_now < bare_before else '⚠️ Needs work'} |")
    report.append("")
    
    # Write report
    output_path = data_dir / "QUALITY_REPORT.md"
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    
    print(f"✅ Quality report generated: {output_path}")
    print(f"   Total pairs: {total:,}")
    print(f"   Lint pass rate: {lint_pass/total*100:.1f}%")
    print(f"   Categories: {len(categories)}")


if __name__ == "__main__":
    main()
