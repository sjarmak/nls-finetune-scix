#!/usr/bin/env python3
"""Curate SWEET vocabulary by removing common English words.

US-002: Filters SWEET's 12,986 entries to remove common English words
so auto-annotation produces meaningful spans instead of noise.

Filtering criteria:
1. Remove entries where canonical label is fewer than 4 characters
2. Remove entries where canonical label is in English stopword list (500+)
3. Remove entries where canonical label is in high-frequency English words (top 5,000)
4. Remove entries where canonical label is a single common English word

Outputs:
- data/vocabularies/sweet_curated.jsonl         (surviving entries)
- data/vocabularies/sweet_removed.jsonl          (removed entries + reason)
- data/vocabularies/sweet_curation_report.json   (statistics)
- data/vocabularies/sweet_borderline.jsonl       (4-6 char scientific terms)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from nltk.corpus import stopwords as nltk_stopwords
from wordfreq import top_n_list, word_frequency

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SWEET_CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "agent_runs"
    / "run_20260127_174306_999adfdd"
    / "normalized"
    / "topic_catalog_sweet.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "vocabularies"

# ──────────────────────────────────────────────
# Stopword list (500+ words)
# ──────────────────────────────────────────────

EXTENDED_STOPWORDS: frozenset[str] = frozenset(
    list(nltk_stopwords.words("english"))
    + [
        "able", "about", "above", "according", "across", "actual", "actually",
        "added", "additional", "adult", "after", "again", "against", "ago",
        "ahead", "air", "alive", "almost", "along", "already", "also",
        "although", "always", "among", "amount", "analysis", "ancient",
        "angle", "animal", "annual", "another", "appear", "appears", "area",
        "around", "arrival", "artificial", "ask", "asked", "assessment",
        "attitude", "author", "autumn", "available", "average", "away",
        "b", "back", "background", "bag", "balance", "bank", "bar", "base",
        "based", "basic", "battery", "bay", "became", "become", "becomes",
        "becoming", "began", "begin", "beginning", "behind", "beside",
        "besides", "best", "better", "beyond", "big", "bit", "body", "both",
        "bottom", "bring", "brings", "brought", "call", "called", "came",
        "can", "cannot", "case", "cases", "cause", "caused", "causes",
        "center", "certain", "certainly", "change", "changes", "clear",
        "clearly", "close", "closely", "come", "comes", "coming", "common",
        "completely", "consider", "considered", "contain", "contains",
        "continue", "continued", "correct", "could", "course", "current",
        "currently", "d", "data", "day", "days", "deal", "describe",
        "described", "despite", "determine", "determined", "develop",
        "developed", "did", "different", "direct", "directly", "done", "down",
        "due", "early", "effect", "effective", "either", "end", "enough",
        "entire", "especially", "essentially", "establish", "established",
        "etc", "even", "every", "everyone", "everything", "evidence",
        "exactly", "example", "except", "expect", "expected", "experience",
        "explain", "expressed", "fact", "fairly", "fall", "far", "fast",
        "feature", "feel", "few", "figure", "final", "finally", "find",
        "first", "focus", "follow", "following", "form", "former", "forward",
        "found", "four", "free", "front", "full", "fully", "further", "gain",
        "gave", "general", "generally", "get", "gets", "getting", "give",
        "given", "gives", "go", "goes", "going", "gone", "good", "got",
        "great", "ground", "group", "grow", "growing", "growth", "half",
        "hand", "handle", "happen", "happens", "hard", "having", "head",
        "hear", "heard", "held", "help", "hence", "here", "herself",
        "high", "himself", "hold", "home", "hour", "hours", "however",
        "huge", "hundred", "idea", "identify", "immediately", "impact",
        "important", "improve", "include", "included", "includes",
        "including", "increase", "increased", "increasingly", "indeed",
        "indicate", "individual", "information", "initial", "inside",
        "instead", "interest", "interested", "interesting", "involve",
        "involved", "involves", "issue", "issues", "itself", "join", "just",
        "keep", "key", "kind", "knew", "know", "knowledge", "known", "lack",
        "large", "largely", "last", "late", "later", "latter", "lead",
        "leads", "least", "leave", "led", "left", "less", "let", "level",
        "levels", "light", "like", "likely", "limit", "limited", "line",
        "list", "little", "local", "location", "long", "longer", "look",
        "looks", "loss", "lost", "lot", "low", "lower", "made", "main",
        "mainly", "make", "makes", "making", "many", "mark", "matter",
        "may", "maybe", "mean", "means", "measure", "meet", "member",
        "method", "methods", "middle", "might", "million", "mind", "minor",
        "miss", "model", "modern", "moment", "month", "months", "move",
        "movement", "moving", "much", "must", "name", "named", "natural",
        "nature", "near", "nearly", "necessary", "need", "needed", "needs",
        "negative", "neither", "network", "never", "nevertheless", "new",
        "next", "non", "none", "normal", "note", "noted", "nothing",
        "notice", "now", "number", "obtain", "obtained", "occur", "occurs",
        "offer", "offered", "official", "often", "old", "one", "ones",
        "onto", "open", "operate", "operation", "opposite", "order",
        "original", "others", "otherwise", "outside", "overall", "own",
        "pair", "paper", "part", "particular", "particularly", "pass",
        "passed", "past", "pattern", "per", "percent", "performance",
        "perhaps", "period", "person", "physical", "piece", "place", "plan",
        "play", "plus", "point", "positive", "possible", "potential",
        "power", "practice", "present", "previous", "previously", "primary",
        "problem", "probably", "process", "produce", "produced", "product",
        "program", "project", "proper", "properly", "prove", "provide",
        "provides", "pull", "purpose", "push", "put", "question", "quite",
        "range", "rate", "rather", "reach", "read", "real", "really",
        "reason", "receive", "received", "recent", "recently", "record",
        "reduce", "reduced", "reflect", "regard", "region", "regular",
        "related", "release", "remain", "remains", "remove", "report",
        "reported", "represent", "require", "required", "research",
        "respect", "response", "rest", "result", "results", "return",
        "returned", "reveal", "right", "rise", "role", "room", "round",
        "rule", "run", "said", "same", "sample", "say", "says", "scale",
        "search", "second", "section", "see", "seem", "seems", "seen",
        "select", "sense", "separate", "serve", "service", "set", "several",
        "shall", "shape", "share", "shift", "short", "show", "showed",
        "shown", "shows", "side", "significant", "similar", "simply",
        "since", "single", "situation", "size", "slightly", "small",
        "something", "sometimes", "soon", "sort", "source", "space",
        "speak", "special", "specific", "spread", "stage", "stand",
        "standard", "start", "started", "state", "step", "still", "stop",
        "straight", "strike", "strong", "structure", "study", "subject",
        "success", "successful", "suggest", "support", "sure", "surface",
        "system", "take", "taken", "takes", "tell", "tend", "term", "terms",
        "test", "text", "theory", "therefore", "thing", "things", "think",
        "third", "though", "thought", "thousand", "three", "through",
        "throughout", "thus", "time", "title", "today", "together", "told",
        "took", "top", "total", "touch", "toward", "towards", "track",
        "trade", "traditional", "transfer", "true", "truth", "try", "turn",
        "turned", "turns", "two", "type", "types", "understand",
        "understanding", "unit", "unless", "unlikely", "until", "up",
        "upon", "upper", "use", "used", "uses", "using", "usual", "usually",
        "value", "various", "version", "via", "view", "volume", "want",
        "watch", "water", "way", "week", "weeks", "well", "went", "whatever",
        "whether", "whole", "whose", "wide", "within", "without", "wonder",
        "word", "work", "works", "world", "worth", "would", "write",
        "wrong", "wrote", "year", "years", "yes", "yet",
    ]
)


def load_sweet_catalog(path: Path) -> list[dict[str, Any]]:
    """Load SWEET topic catalog from JSONL."""
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_high_frequency_set(n: int = 5000) -> frozenset[str]:
    """Return top N high-frequency English words from wordfreq."""
    return frozenset(top_n_list("en", n))


def is_single_common_english_word(label: str) -> bool:
    """Check if a single-word label is a common English word.

    Uses word frequency > 1e-6 as threshold: this captures words
    that appear roughly once per million words in English text.
    """
    words = label.split()
    if len(words) != 1:
        return False
    freq = word_frequency(label, "en")
    return freq > 1e-6


def classify_entry(
    entry: dict[str, Any],
    high_freq_words: frozenset[str],
) -> tuple[str | None, str]:
    """Classify a SWEET entry as kept or removed.

    Returns:
        (removal_reason, category) where removal_reason is None if kept.
        category is one of: 'kept', 'short', 'stopword', 'high_frequency',
        'common_english'.
    """
    label = entry["label"].lower().strip()

    if len(label) < 4:
        return "fewer_than_4_characters", "short"

    if label in EXTENDED_STOPWORDS:
        return "in_stopword_list", "stopword"

    if label in high_freq_words:
        return "in_high_frequency_english_top_5000", "high_frequency"

    if is_single_common_english_word(label):
        return "single_common_english_word", "common_english"

    return None, "kept"


def is_borderline_scientific(entry: dict[str, Any]) -> bool:
    """Check if an entry is a borderline case: 4-6 char scientific term."""
    label = entry["label"].strip()
    return 4 <= len(label) <= 6


def curate_sweet_vocabulary(
    input_path: Path = SWEET_CATALOG_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the full SWEET vocabulary curation pipeline.

    Returns the curation report as a dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_sweet_catalog(input_path)
    high_freq_words = build_high_frequency_set(5000)

    curated: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    borderline: list[dict[str, Any]] = []
    removal_reason_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    for entry in entries:
        reason, category = classify_entry(entry, high_freq_words)
        category_counts[category] += 1

        if reason is not None:
            removed_entry = {**entry, "removal_reason": reason}
            removed.append(removed_entry)
            removal_reason_counts[reason] += 1
        else:
            curated.append(entry)
            if is_borderline_scientific(entry):
                borderline.append(entry)

    # Write outputs
    curated_path = output_dir / "sweet_curated.jsonl"
    removed_path = output_dir / "sweet_removed.jsonl"
    borderline_path = output_dir / "sweet_borderline.jsonl"
    report_path = output_dir / "sweet_curation_report.json"

    _write_jsonl(curated_path, curated)
    _write_jsonl(removed_path, removed)
    _write_jsonl(borderline_path, borderline)

    report = _build_report(
        total=len(entries),
        curated_count=len(curated),
        removed_count=len(removed),
        borderline_count=len(borderline),
        removal_reason_counts=dict(removal_reason_counts),
        category_counts=dict(category_counts),
        stopword_list_size=len(EXTENDED_STOPWORDS),
        high_freq_list_size=5000,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_summary(report, curated_path, removed_path, borderline_path, report_path)

    return report


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_report(
    *,
    total: int,
    curated_count: int,
    removed_count: int,
    borderline_count: int,
    removal_reason_counts: dict[str, int],
    category_counts: dict[str, int],
    stopword_list_size: int,
    high_freq_list_size: int,
) -> dict[str, Any]:
    """Build the curation report."""
    return {
        "summary": {
            "total_input_entries": total,
            "curated_entries": curated_count,
            "removed_entries": removed_count,
            "borderline_entries": borderline_count,
            "retention_rate": round(curated_count / total, 4) if total > 0 else 0,
            "removal_rate": round(removed_count / total, 4) if total > 0 else 0,
        },
        "removal_reasons": removal_reason_counts,
        "category_breakdown": category_counts,
        "filtering_config": {
            "min_label_length": 4,
            "stopword_list_size": stopword_list_size,
            "high_frequency_list_size": high_freq_list_size,
            "common_word_frequency_threshold": 1e-6,
        },
    }


def _print_summary(
    report: dict[str, Any],
    curated_path: Path,
    removed_path: Path,
    borderline_path: Path,
    report_path: Path,
) -> None:
    """Print human-readable summary to stdout."""
    s = report["summary"]
    print("=" * 60)
    print("SWEET Vocabulary Curation Report")
    print("=" * 60)
    print(f"Input entries:     {s['total_input_entries']:>6}")
    print(f"Curated (kept):    {s['curated_entries']:>6}")
    print(f"Removed:           {s['removed_entries']:>6}")
    print(f"Borderline (4-6c): {s['borderline_entries']:>6}")
    print(f"Retention rate:    {s['retention_rate']:>6.1%}")
    print()
    print("Removal breakdown:")
    for reason, count in sorted(
        report["removal_reasons"].items(), key=lambda x: -x[1]
    ):
        print(f"  {reason:50s} {count:>5}")
    print()
    print("Output files:")
    print(f"  Curated:    {curated_path}")
    print(f"  Removed:    {removed_path}")
    print(f"  Borderline: {borderline_path}")
    print(f"  Report:     {report_path}")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Curate SWEET vocabulary by removing common English words"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SWEET_CATALOG_PATH,
        help="Path to topic_catalog_sweet.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for curated files",
    )
    args = parser.parse_args()
    curate_sweet_vocabulary(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
