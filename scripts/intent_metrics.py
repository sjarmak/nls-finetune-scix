"""Pure metric functions for the intent-classifier experiment.

Every function takes plain prediction/label rows and returns numbers, so the
runner (evaluate_intent_classifiers.py) stays IO-only and the maths is
unit-testable without any model call.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import quantiles

OPERATOR_CLASSES = ("none", "citations", "references", "similar", "trending", "useful", "reviews")
SELECTIVE_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.5 .. 0.95
ECE_BINS = 10


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """P/R/F1; F1 is None (undefined) when neither gold nor prediction has a positive."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if tp + fp + fn == 0:
        f1 = None
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def operator_report(pairs: list[tuple[str, str]]) -> dict:
    """Per-class P/R/F1, macro-F1 and accuracy for (gold, predicted) operator pairs.

    Macro-F1 averages over classes present in gold or predictions, so a
    dataset with no 'useful' items is not penalised for that class.
    """
    per_class: dict[str, dict] = {}
    present = {g for g, _ in pairs} | {p for _, p in pairs}
    for cls in OPERATOR_CLASSES:
        if cls not in present:
            continue
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        fp = sum(1 for g, p in pairs if g != cls and p == cls)
        fn = sum(1 for g, p in pairs if g == cls and p != cls)
        per_class[cls] = _prf(tp, fp, fn)
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    accuracy = sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0
    return {"n": len(pairs), "accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


def false_positive_rate(pairs: list[tuple[str, str]]) -> dict:
    """Share of gold-'none' items where an operator was predicted."""
    negatives = [(g, p) for g, p in pairs if g == "none"]
    fps = sum(1 for _, p in negatives if p != "none")
    rate = fps / len(negatives) if negatives else 0.0
    return {"n": len(negatives), "false_positives": fps, "rate": rate}


def set_f1(pairs: list[tuple[set[str], set[str]]]) -> dict:
    """Micro-averaged set F1 over (gold, predicted) value sets."""
    tp = fp = fn = 0
    exact = 0
    for gold, pred in pairs:
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += gold == pred
    out = _prf(tp, fp, fn)
    out["n"] = len(pairs)
    out["exact_match"] = exact / len(pairs) if pairs else 0.0
    return out


def selective_curve(rows: list[tuple[float, bool]], thresholds=SELECTIVE_THRESHOLDS) -> list[dict]:
    """Coverage and accuracy at each confidence threshold.

    rows are (confidence, correct). An item is answered when confidence >= t.
    """
    curve = []
    for t in thresholds:
        answered = [c for conf, c in rows if conf >= t]
        curve.append(
            {
                "threshold": t,
                "coverage": len(answered) / len(rows) if rows else 0.0,
                "accuracy": sum(answered) / len(answered) if answered else None,
                "n_answered": len(answered),
            }
        )
    return curve


def accuracy_at_coverage(rows: list[tuple[float, bool]], target_coverage: float) -> dict:
    """Accuracy on the most-confident ``target_coverage`` share of items.

    Thresholds the confidence at the value that keeps at least the target
    share, which is what a router tuned to 90% coverage would do.
    """
    if not rows:
        return {"coverage": 0.0, "accuracy": None, "threshold": None}
    ordered = sorted(rows, key=lambda r: -r[0])
    keep = max(1, round(target_coverage * len(rows)))
    kept = ordered[:keep]
    threshold = kept[-1][0]
    kept = [r for r in ordered if r[0] >= threshold]
    return {
        "coverage": len(kept) / len(rows),
        "accuracy": sum(c for _, c in kept) / len(kept),
        "threshold": threshold,
    }


def expected_calibration_error(rows: list[tuple[float, bool]], bins: int = ECE_BINS) -> dict:
    """ECE with equal-width bins over [0, 1] plus the reliability table."""
    table = []
    ece = 0.0
    n = len(rows)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [r for r in rows if lo <= r[0] < hi or (b == bins - 1 and r[0] == 1.0)]
        if not members:
            table.append({"bin": [lo, hi], "n": 0, "confidence": None, "accuracy": None})
            continue
        conf = sum(c for c, _ in members) / len(members)
        acc = sum(ok for _, ok in members) / len(members)
        ece += len(members) / n * abs(conf - acc)
        table.append({"bin": [lo, hi], "n": len(members), "confidence": conf, "accuracy": acc})
    return {"ece": ece if n else None, "n": n, "bins": table}


def stability(rows: list[dict], fields: tuple[str, ...]) -> dict:
    """Fraction of items whose answer changed across repeats, per field.

    rows carry item_id, repeat and one value per field (hashable).
    """
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    multi = {k: v for k, v in by_item.items() if len(v) > 1}
    out = {"n_items": len(multi), "repeats": max((len(v) for v in multi.values()), default=0)}
    for field in fields:
        changed = sum(1 for v in multi.values() if len({repr(r[field]) for r in v}) > 1)
        out[field] = changed / len(multi) if multi else None
    return out


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None}
    if len(values) == 1:
        return {"n": 1, "p50": values[0], "p95": values[0], "mean": values[0]}
    cuts = quantiles(values, n=20, method="inclusive")  # 19 cut points, 5% steps
    return {"n": len(values), "p50": cuts[9], "p95": cuts[18], "mean": sum(values) / len(values)}


def cost_summary(rows: list[dict], rate_in_per_mtok: float, rate_out_per_mtok: float = 0.0) -> dict:
    """Mean and total cost in USD from per-row token counts (0 when not called)."""
    n = len(rows)
    total_in = sum(r.get("input_tokens") or 0 for r in rows)
    total_out = sum(r.get("output_tokens") or 0 for r in rows)
    total = total_in / 1e6 * rate_in_per_mtok + total_out / 1e6 * rate_out_per_mtok
    return {
        "n": n,
        "mean_input_tokens": total_in / n if n else 0.0,
        "mean_output_tokens": total_out / n if n else 0.0,
        "mean_usd_per_query": total / n if n else 0.0,
        "total_usd": total,
        "classifier_call_rate": (
            sum(1 for r in rows if r.get("classifier_called")) / n if n else 0.0
        ),
    }


def class_counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def topic_words(text: str) -> frozenset[str]:
    """Lower-cased words without boolean keywords; used for gold and predictions alike."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(w for w in words if w not in ("and", "or", "not"))


def extraction_metrics(rows: list[dict]) -> dict | None:
    """Year range, first-author, citation-floor and topic-word agreement with gold.

    Rows need gold labels from ``derive_intent_labels`` (with ``topic_tokens``);
    returns None when none have them (hand-labelled sets). Each field is a
    ``set_f1`` over one-element sets, so ``exact_match`` is the accuracy.
    First author is scored only where gold names an author.
    """
    rows = [r for r in rows if "topic_tokens" in r["labels"]]
    if not rows:
        return None

    def year(d: dict) -> set[str]:
        if d["year_from"] is None and d["year_to"] is None:
            return set()
        return {f"{d['year_from']}-{d['year_to']}"}

    def flag(value: bool) -> set[str]:
        return {"yes"} if value else set()

    with_year = [r for r in rows if year(r["labels"])]
    with_author = [r for r in rows if r["labels"]["has_author"]]
    return {
        "year": set_f1([(year(r["labels"]), year(r["prediction"])) for r in rows]),
        "year_when_gold_has_one": set_f1(
            [(year(r["labels"]), year(r["prediction"])) for r in with_year]
        ),
        "first_author": set_f1(
            [
                (flag(r["labels"]["first_author"]), flag(r["prediction"]["first_author"]))
                for r in with_author
            ]
        ),
        "citation_floor": set_f1(
            [
                (
                    flag(r["labels"]["min_citations"] is not None),
                    flag(r["prediction"]["min_citations"] is not None),
                )
                for r in rows
            ]
        ),
        "topic_tokens": set_f1(
            [
                (
                    set(r["labels"]["topic_tokens"]),
                    topic_words(
                        " ".join(r["prediction"]["free_text_terms"] + r["prediction"]["or_terms"])
                    ),
                )
                for r in rows
            ]
        ),
    }
