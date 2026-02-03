#!/usr/bin/env python3
"""Apply corrected fixes to review items from v5 curator review.

Categories:
1. ESTABLISHED TERMS (keep quoted): gravitational waves, dark energy, weak lensing,
   black holes, solar wind, magnetic field
2. SUBDISCIPLINES (abs:term OR keyword:"phrase" collection:X): infrared astronomy,
   string theory, infrared spectroscopy, galactic dynamics
3. GENERIC COMPOUND (split with AND): stellar evolution, radial velocity, galaxy formation,
   galaxy clusters, detector sensitivity, radiative transfer
4. SINGLE-WORD QUOTED (remove quotes): exoplanets, photometry, magnetar
5. SPECIAL CASES: redshift surveys -> reviews(abs:redshift)
"""

import json
from pathlib import Path

# Indices by category from review file
ESTABLISHED_TERMS_INDICES = {
    # gravitational waves
    3280, 3300, 3320, 3340, 3360, 3380, 3400, 3420, 3440, 3460, 3480, 3500, 3540, 3560, 3580, 3720,
    # dark energy
    2940, 3740, 3760,
    # weak lensing
    3060, 3100, 3780,
    # black holes
    4160,
    # solar wind
    3800,
    # magnetic field
    3960,
}

SUBDISCIPLINE_INDICES = {
    # infrared astronomy
    3000, 3080, 3120,
    # string theory
    4200,
    # infrared spectroscopy
    3840,
    # galactic dynamics
    4380,
}

GENERIC_COMPOUND_INDICES = {
    1040,  # stellar evolution
    1240,  # radial velocity (with measurements)
    1520,  # galaxy formation
    4420,  # galaxy clusters
    4140,  # detector sensitivity
    4280,  # radiative transfer
}

SINGLE_WORD_INDICES = {
    3020,  # exoplanets
    4020,  # photometry
    1460,  # magnetar (partial - also has X-ray)
}

SPECIAL_INDICES = {
    4440,  # redshift surveys
    2400,  # Chandra X-ray
}


def fix_subdiscipline(query: str, nl: str) -> str:
    """Transform subdiscipline phrases to: abs:term OR keyword:"phrase" collection:X"""
    # Detect the subdiscipline phrase
    subdisciplines = {
        "infrared astronomy": ("infrared", "infrared astronomy", "astronomy"),
        "string theory": ("string", "string theory", "physics"),
        "infrared spectroscopy": ("infrared", "infrared spectroscopy", "astronomy"),
        "galactic dynamics": ("galactic", "galactic dynamics", "astronomy"),
    }

    for phrase, (term, kw_phrase, collection) in subdisciplines.items():
        if f'abs:"{phrase}"' in query:
            # Replace abs:"phrase" with (abs:term OR keyword:"phrase") collection:X
            new_part = f'(abs:{term} OR keyword:"{kw_phrase}") collection:{collection}'
            query = query.replace(f'abs:"{phrase}"', new_part)
            break

    return query


def fix_generic_compound(query: str, nl: str, index: int) -> str:
    """Split generic compound topics with AND."""
    compounds = {
        1040: ('abs:"stellar evolution"', 'abs:(stellar AND evolution)'),
        1240: ('abs:"radial velocity"', 'abs:(radial AND velocity)'),
        1520: ('abs:"galaxy formation"', 'abs:(galaxy AND formation)'),
        4420: ('abs:"galaxy clusters"', 'abs:(galaxy AND cluster*)'),  # clusters -> cluster* for flexibility
        4140: ('abs:"detector sensitivity"', 'abs:(detector AND sensitivity)'),
        4280: ('abs:"radiative transfer"', 'abs:(radiative AND transfer)'),
    }

    if index in compounds:
        old, new = compounds[index]
        query = query.replace(old, new)

    return query


def fix_single_word(query: str, index: int) -> str:
    """Remove unnecessary quotes from single-word terms."""
    fixes = {
        3020: ('abs:"exoplanets"', 'abs:exoplanets'),
        4020: ('abs:"photometry"', 'abs:photometry'),
        1460: ('abs:"magnetar"', 'abs:magnetar'),  # Keep X-ray quoted
    }

    if index in fixes:
        old, new = fixes[index]
        query = query.replace(old, new)

    return query


def fix_special_cases(query: str, index: int) -> str:
    """Handle special cases."""
    if index == 4440:  # redshift surveys
        # Change to reviews(abs:redshift) since "surveys" is about survey papers
        # which are better found via the reviews() operator
        query = 'reviews(abs:redshift) doctype:article'
    elif index == 2400:  # Chandra X-ray
        # Keep X-ray quoted (compound with hyphen), but Chandra is a proper noun
        # Combine properly: abs:(Chandra AND "X-ray")
        query = query.replace('abs:"Chandra X-ray"', 'abs:(Chandra AND "X-ray")')
        # Also fix unnecessary bibstem quoting
        query = query.replace('bibstem:"Science"', 'bibstem:Science')

    return query


def apply_fixes(examples: list, review_items: list) -> tuple[list, int, list]:
    """Apply categorized fixes to examples by matching NL+query content."""
    # Build lookup by (nl, query) tuple
    review_by_content = {}
    for item in review_items:
        key = (item["nl"], item["query"])
        review_by_content[key] = item

    fixed_count = 0
    kept_count = 0
    changes = []

    for i, ex in enumerate(examples):
        nl = ex.get("natural_language", "")
        query = ex.get("ads_query", "")
        key = (nl, query)

        if key not in review_by_content:
            continue

        item = review_by_content[key]
        original_index = item["index"]  # Original index for category lookup
        old_query = query
        new_query = old_query
        action = "keep"

        if original_index in ESTABLISHED_TERMS_INDICES:
            # Keep as-is - these are established scientific terms
            action = "keep_established"
            kept_count += 1
        elif original_index in SUBDISCIPLINE_INDICES:
            new_query = fix_subdiscipline(old_query, nl)
            action = "fix_subdiscipline"
        elif original_index in GENERIC_COMPOUND_INDICES:
            new_query = fix_generic_compound(old_query, nl, original_index)
            action = "fix_compound"
        elif original_index in SINGLE_WORD_INDICES:
            new_query = fix_single_word(old_query, original_index)
            action = "fix_single_word"
        elif original_index in SPECIAL_INDICES:
            new_query = fix_special_cases(old_query, original_index)
            action = "fix_special"
        else:
            # Default: keep as-is
            action = "keep_unclassified"
            kept_count += 1

        if new_query != old_query:
            ex["ads_query"] = new_query
            fixed_count += 1
            changes.append({
                "index": i,
                "original_index": original_index,
                "nl": nl,
                "old": old_query,
                "new": new_query,
                "action": action
            })
        else:
            changes.append({
                "index": i,
                "original_index": original_index,
                "nl": nl,
                "old": old_query,
                "new": new_query,
                "action": action
            })

    return examples, fixed_count, changes


def main():
    # Load data
    gold_path = Path("data/datasets/raw/gold_examples.json")
    review_path = Path("/tmp/for_review_v5.json")

    with open(gold_path) as f:
        examples = json.load(f)

    with open(review_path) as f:
        review_items = json.load(f)

    print(f"Loaded {len(examples)} examples")
    print(f"Review items: {len(review_items)}")

    # Categorize review items
    print(f"\nCategorization:")
    print(f"  Established terms (keep): {len(ESTABLISHED_TERMS_INDICES)}")
    print(f"  Subdisciplines (fix): {len(SUBDISCIPLINE_INDICES)}")
    print(f"  Generic compounds (fix): {len(GENERIC_COMPOUND_INDICES)}")
    print(f"  Single-word (fix): {len(SINGLE_WORD_INDICES)}")
    print(f"  Special cases (fix): {len(SPECIAL_INDICES)}")

    # Apply fixes
    examples, fixed_count, changes = apply_fixes(examples, review_items)

    print(f"\nApplied {fixed_count} fixes")

    # Show changes
    print("\n" + "=" * 60)
    print("CHANGES APPLIED:")
    print("=" * 60)

    for change in changes:
        print(f"\n[{change['action']}] Index {change['index']}")
        print(f"  NL: {change['nl']}")
        print(f"  OLD: {change['old']}")
        print(f"  NEW: {change['new']}")

    # Save
    with open(gold_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"\n✓ Saved {len(examples)} examples to {gold_path}")

    # Summary
    kept = len(review_items) - fixed_count
    print(f"\nSummary:")
    print(f"  Total review items: {len(review_items)}")
    print(f"  Fixed: {fixed_count}")
    print(f"  Kept as-is: {kept}")


if __name__ == "__main__":
    main()
