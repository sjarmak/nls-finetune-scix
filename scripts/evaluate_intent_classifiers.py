#!/usr/bin/env python3
"""Run the Jev intent-classifier experiment arms A to E and score them.

Arms (docs/research/jev-intent-classifier-experiment.md):
    A  regex `extract_intent` as shipped
    B  Jev for operator, enums and gates; regex for names, years, topics
    C  Claude Haiku 4.5 structured output over the same question set
    D  as B with the regex IntentSpec sent to Jev as extra state
    E  regex first; Jev only when regex sets no operator or confidence < 0.5

Datasets:
    benchmark  data/datasets/benchmark/benchmark_queries.json (contaminated)
    val        data/datasets/processed/val.jsonl (synthetic NL, derived labels)
    heldout    data/datasets/benchmark/heldout_paraphrases.json; scored only
               when every item carries review.status == "approved"

One JSONL row per (arm, item, repeat) goes to
data/datasets/evaluations/intent_classifiers_<dataset>_<date>.jsonl and the
metric tables to the matching _metrics.json. Model calls are cached by
request fingerprint (jev_intent / llm_intent); repeats beyond the first
bypass the cache so they measure real run-to-run stability.

Usage:
    set -a; . ~/projects/omni-experiments/.env; . ~/projects/scix_experiments/.env; set +a
    uv run python scripts/evaluate_intent_classifiers.py --dataset benchmark --arms A B C D E
    uv run python scripts/evaluate_intent_classifiers.py --dataset val --arms A B E
    uv run python scripts/evaluate_intent_classifiers.py --dataset benchmark --arms B C D \\
        --stability-repeats 3 --stability-subset 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages/finetune/src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_heldout_paraphrases import load_heldout  # noqa: E402
from derive_intent_labels import (  # noqa: E402
    derive_labels,
    load_benchmark_items,
    load_val_items,
)
from intent_metrics import (  # noqa: E402
    accuracy_at_coverage,
    class_counts,
    cost_summary,
    expected_calibration_error,
    false_positive_rate,
    latency_summary,
    operator_report,
    selective_curve,
    set_f1,
    stability,
)

from finetune.domains.scix.intent_spec import IntentSpec  # noqa: E402
from finetune.domains.scix.jev_intent import (  # noqa: E402
    NONE_OPTION,
    PROPERTY_BOOLEANS,
    JevClient,
    classify_and_extract,
)
from finetune.domains.scix.llm_intent import (  # noqa: E402
    UNKNOWN,
    LlmClient,
    classify_and_extract_llm,
)
from finetune.domains.scix.ner import extract_intent  # noqa: E402
from finetune.domains.scix.pipeline import (  # noqa: E402
    GATED_CONFIDENCE_THRESHOLD,
    compute_pipeline_confidence,
)

ARMS = {
    "A": "regex",
    "B": "jev",
    "C": "llm_haiku_4_5",
    "D": "jev_with_regex_state",
    "E": "jev_gated",
}
DATASETS = ("benchmark", "val", "heldout")
OUTPUT_DIR = REPO_ROOT / "data/datasets/evaluations"
JEV_USD_PER_MTOK_IN = 0.042
HAIKU_USD_PER_MTOK_IN = 1.0
HAIKU_USD_PER_MTOK_OUT = 5.0
ENUM_FIELDS = ("property", "doctype", "bibgroup", "collection")
SCORED_PROPERTIES = frozenset(PROPERTY_BOOLEANS)
GATE_FIELDS = ("search_kind", "needs_clarification", "refers_to_specific_paper")
WORKERS = 4


# --------------------------------------------------------------------------- items


def load_items(dataset: str) -> list[dict]:
    """Items as {id, nl, labels, meta}. Labels are derived (benchmark, val) or hand-written."""
    if dataset == "benchmark":
        return [_derived(item) for item in load_benchmark_items()]
    if dataset == "val":
        return [_derived(item) for item in load_val_items()]
    doc = load_heldout()
    status = {i["id"]: i.get("review", {}).get("status") for i in doc["items"]}
    unreviewed = [i for i, s in status.items() if s not in ("approved", "rejected")]
    if unreviewed:
        raise SystemExit(
            f"heldout: {len(unreviewed)} of {len(doc['items'])} items are not reviewed by a "
            "human; the plan forbids scoring the set before review. Nothing was run. "
            "Record the review with scripts/approve_heldout_paraphrases.py."
        )
    return [
        {"id": i["id"], "nl": i["nl"], "labels": i["labels"], "meta": {"stratum": i["stratum"]}}
        for i in doc["items"]
        if status[i["id"]] == "approved"
    ]


def _derived(item: dict) -> dict:
    labels = derive_labels(item["gold_query"]).to_dict()
    meta = {**item.get("meta", {}), "category": item.get("category"), "source": item["source"]}
    return {"id": item["id"], "nl": item["nl"], "labels": labels, "meta": meta}


# --------------------------------------------------------------------------- arms


def _intent_fields(intent: IntentSpec) -> dict:
    return {
        "operator": intent.operator or NONE_OPTION,
        "property": sorted(intent.property),
        "doctype": sorted(intent.doctype),
        "bibgroup": sorted(intent.bibgroup),
        "collection": sorted(intent.collection),
        "authors": list(intent.authors),
        "year_from": intent.year_from,
        "year_to": intent.year_to,
        "free_text_terms": list(intent.free_text_terms),
    }


def run_regex(nl: str) -> dict:
    started = time.perf_counter()
    intent = extract_intent(nl)
    latency = (time.perf_counter() - started) * 1000
    return {
        **_intent_fields(intent),
        # The regex has no calibrated signal: 0.95 when a pattern fires, nothing otherwise.
        "confidence": {"operator": intent.confidence.get("operator")},
        "search_kind": None,
        "needs_clarification": None,
        "refers_to_specific_paper": None,
        "classifier_called": False,
        "latency_ms": latency,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached": False,
        "model": "regex",
    }


def run_jev(nl: str, client: JevClient, include_regex_state: bool, use_cache: bool) -> dict:
    started = time.perf_counter()
    intent, answers = classify_and_extract(
        nl, client, include_regex_state=include_regex_state, use_cache=use_cache
    )
    local_latency = (time.perf_counter() - started) * 1000
    if answers is None:  # ADS passthrough; Jev not called
        row = run_regex(nl)
        return row
    return {
        **_intent_fields(intent),
        "confidence": {k: v for k, v in intent.confidence.items() if not k.startswith("regex_")},
        "operator_probabilities": answers.choices["operator"].probabilities,
        "search_kind": answers.choices["search_kind"].choice,
        "needs_clarification": answers.booleans["needs_clarification"],
        "refers_to_specific_paper": answers.booleans["refers_to_specific_paper"],
        "classifier_called": True,
        "latency_ms": answers.latency_ms if answers.cached else local_latency,
        "input_tokens": answers.input_tokens,
        "output_tokens": 0,
        "cached": answers.cached,
        "model": answers.model,
    }


def run_llm(nl: str, client: LlmClient, use_cache: bool) -> dict:
    started = time.perf_counter()
    intent, answers = classify_and_extract_llm(nl, client, use_cache=use_cache)
    local_latency = (time.perf_counter() - started) * 1000
    if answers is None:
        return run_regex(nl)
    v = answers.values
    return {
        **_intent_fields(intent),
        "confidence": intent.confidence,
        "unknown_fields": sorted(k for k, val in v.items() if val == UNKNOWN),
        "search_kind": None if v["search_kind"] == UNKNOWN else v["search_kind"],
        "needs_clarification": _tri(v["needs_clarification"]),
        "refers_to_specific_paper": _tri(v["refers_to_specific_paper"]),
        "classifier_called": True,
        "latency_ms": answers.latency_ms if answers.cached else local_latency,
        "input_tokens": answers.input_tokens,
        "output_tokens": answers.output_tokens,
        "cached": answers.cached,
        "model": answers.model,
    }


def _tri(value: str) -> float | None:
    return None if value == UNKNOWN else (1.0 if value == "true" else 0.0)


def run_gated(nl: str, client: JevClient, use_cache: bool) -> dict:
    """Arm E, mirroring pipeline.extract_intent_with_backend('jev_gated')."""
    regex_row = run_regex(nl)
    intent = extract_intent(nl)
    if intent.confidence.get("ads_passthrough"):
        return regex_row
    confidence, _ = compute_pipeline_confidence(intent)
    if intent.operator is not None and confidence >= GATED_CONFIDENCE_THRESHOLD:
        return regex_row
    jev_row = run_jev(nl, client, include_regex_state=False, use_cache=use_cache)
    jev_row["latency_ms"] += regex_row["latency_ms"]
    return jev_row


class ArmRunner:
    def __init__(
        self,
        arms: list[str],
        jev_cache: Path | None,
        llm_cache: Path | None,
        llm_transport: str = "sdk",
    ) -> None:
        self.jev = None
        self.llm = None
        if any(a in arms for a in "BDE"):
            key = os.environ.get("TYPESAFE_API_KEY", "")
            if not key:
                raise SystemExit("TYPESAFE_API_KEY is not set; arms B, D and E need it")
            self.jev = JevClient(api_key=key, cache_path=jev_cache)
        if "C" in arms:
            if llm_transport == "sdk" and not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit(
                    "ANTHROPIC_API_KEY is not set; arm C needs it (or --llm-transport cli)"
                )
            self.llm = LlmClient(cache_path=llm_cache, transport=llm_transport)

    def run(self, arm: str, nl: str, use_cache: bool) -> dict:
        if arm == "A":
            return run_regex(nl)
        if arm == "B":
            return run_jev(nl, self.jev, False, use_cache)
        if arm == "C":
            return run_llm(nl, self.llm, use_cache)
        if arm == "D":
            return run_jev(nl, self.jev, True, use_cache)
        if arm == "E":
            return run_gated(nl, self.jev, use_cache)
        raise ValueError(f"unknown arm {arm!r}")

    def close(self) -> None:
        if self.jev is not None:
            self.jev.close()


def score_rows(
    runner: ArmRunner, arm: str, items: list[dict], repeat: int, dataset: str
) -> list[dict]:
    """One row per item. Repeat 1 may hit the cache; later repeats never do."""

    def one(item: dict) -> dict:
        try:
            pred = runner.run(arm, item["nl"], use_cache=repeat == 1)
            error = None
        except Exception as exc:  # recorded, not swallowed: the row carries the error
            pred, error = {}, f"{type(exc).__name__}: {exc}"
        return {
            "dataset": dataset,
            "arm": arm,
            "arm_name": ARMS[arm],
            "item_id": item["id"],
            "repeat": repeat,
            "nl": item["nl"],
            "labels": item["labels"],
            "meta": item["meta"],
            "prediction": pred,
            "error": error,
        }

    workers = 1 if arm == "A" else WORKERS
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


# --------------------------------------------------------------------------- metrics


def _conf_rows(rows: list[dict]) -> list[tuple[float, bool]]:
    out = []
    for r in rows:
        p = r["prediction"]
        conf = p["confidence"].get("operator")
        if conf is None:
            continue
        out.append((float(conf), p["operator"] == r["labels"]["operator"]))
    return out


def arm_metrics(rows: list[dict]) -> dict:
    """Metric tables for the repeat-1 rows of one arm on one dataset."""
    first = [r for r in rows if r["repeat"] == 1 and r["error"] is None]
    errors = [r for r in rows if r["error"]]
    supported = [r for r in first if not r["labels"].get("unsupported_operator")]
    pairs = [(r["labels"]["operator"], r["prediction"]["operator"]) for r in supported]
    unsupported = [r for r in first if r["labels"].get("unsupported_operator")]
    conf = _conf_rows(supported)
    out: dict = {
        "n_items": len(first),
        "n_errors": len(errors),
        "n_unsupported_operator": len(unsupported),
        "operator": operator_report(pairs),
        "operator_fp_on_gold_none": false_positive_rate(pairs),
        "operator_pred_on_unsupported": class_counts(
            [r["prediction"]["operator"] for r in unsupported]
        ),
        "selective": selective_curve(conf),
        "accuracy_at_90_coverage": accuracy_at_coverage(conf, 0.9),
        "calibration": expected_calibration_error(conf),
        "latency_ms": latency_summary([r["prediction"]["latency_ms"] for r in first]),
        "latency_ms_uncached": latency_summary(
            [
                r["prediction"]["latency_ms"]
                for r in first
                if r["prediction"]["classifier_called"] and not r["prediction"]["cached"]
            ]
        ),
    }
    for field in ENUM_FIELDS:
        gold_pred = []
        for r in first:
            gold = set(r["labels"][field])
            pred = set(r["prediction"][field])
            if field == "property":
                gold &= SCORED_PROPERTIES
                pred &= SCORED_PROPERTIES
            gold_pred.append((gold, pred))
        out[f"enum_{field}"] = set_f1(gold_pred)
    strata = {r["meta"].get("stratum") for r in first} - {None}
    if strata:
        out["by_stratum"] = {
            s: operator_report(
                [
                    (r["labels"]["operator"], r["prediction"]["operator"])
                    for r in first
                    if r["meta"].get("stratum") == s
                ]
            )
            for s in sorted(strata)
        }
        neg = [
            (r["labels"]["operator"], r["prediction"]["operator"])
            for r in first
            if r["meta"].get("stratum") == "operator_negative"
        ]
        out["operator_negative_fp"] = false_positive_rate(neg)
        out["gates"] = _gate_metrics(first)
    preds = [r["prediction"] for r in first]
    arm = rows[0]["arm"] if rows else None
    if arm == "C":
        out["cost"] = cost_summary(preds, HAIKU_USD_PER_MTOK_IN, HAIKU_USD_PER_MTOK_OUT)
        out["unknown_rate"] = (
            {
                f: sum(1 for p in preds if f in p.get("unknown_fields", [])) / len(preds)
                for f in ("operator", *ENUM_FIELDS[1:], *GATE_FIELDS)
            }
            if preds
            else {}
        )
    else:
        out["cost"] = cost_summary(preds, JEV_USD_PER_MTOK_IN)
    repeated = [r for r in rows if r["error"] is None]
    if any(r["repeat"] > 1 for r in repeated):
        out["stability"] = stability(
            [
                {
                    "item_id": r["item_id"],
                    "repeat": r["repeat"],
                    **{f: r["prediction"][f] for f in ("operator", *ENUM_FIELDS, "search_kind")},
                }
                for r in repeated
            ],
            ("operator", *ENUM_FIELDS, "search_kind"),
        )
    return out


def _gate_metrics(first: list[dict]) -> dict:
    """search_kind accuracy and boolean-gate accuracy where the arm answers them."""
    out: dict = {}
    sk = [
        (r["labels"]["search_kind"], r["prediction"]["search_kind"])
        for r in first
        if r["prediction"]["search_kind"] is not None
    ]
    out["search_kind"] = {
        "n": len(sk),
        "accuracy": sum(g == p for g, p in sk) / len(sk) if sk else None,
    }
    for field in ("needs_clarification", "refers_to_specific_paper"):
        pairs = [
            (bool(r["labels"][field]), r["prediction"][field] >= 0.5)
            for r in first
            if r["prediction"][field] is not None
        ]
        tp = sum(1 for g, p in pairs if g and p)
        fp = sum(1 for g, p in pairs if not g and p)
        fn = sum(1 for g, p in pairs if g and not p)
        out[field] = {
            "n": len(pairs),
            "accuracy": sum(g == p for g, p in pairs) / len(pairs) if pairs else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    return out


def _fmt(value: float | None, fmt: str = "{:.3f}") -> str:
    return "n/a" if value is None else fmt.format(value)


def summary_table(metrics: dict[str, dict]) -> str:
    head = (
        "arm",
        "n",
        "op macro-F1",
        "op acc",
        "FP@none",
        "acc@90cov",
        "ECE",
        "prop F1",
        "doctype F1",
        "bibgroup F1",
        "coll F1",
        "p50 ms",
        "p95 ms",
        "$/query",
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for arm, m in metrics.items():
        a90 = m["accuracy_at_90_coverage"]["accuracy"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{arm} {ARMS[arm]}",
                    str(m["n_items"]),
                    f"{m['operator']['macro_f1']:.3f}",
                    f"{m['operator']['accuracy']:.3f}",
                    f"{m['operator_fp_on_gold_none']['rate']:.3f}",
                    _fmt(a90),
                    _fmt(m["calibration"]["ece"]),
                    *[_fmt(m[f"enum_{f}"]["f1"]) for f in ENUM_FIELDS],
                    _fmt(m["latency_ms"]["p50"], "{:.0f}"),
                    _fmt(m["latency_ms"]["p95"], "{:.0f}"),
                    _fmt(m["cost"]["mean_usd_per_query"], "{:.6f}"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stability-repeats", type=int, default=1)
    parser.add_argument(
        "--stability-subset",
        type=int,
        default=100,
        help="Items (from the start of the dataset) that get repeats",
    )
    parser.add_argument("--jev-cache", type=Path, default=Path("data/cache/jev_systemone.jsonl"))
    parser.add_argument("--llm-cache", type=Path, default=Path("data/cache/llm_intent.jsonl"))
    parser.add_argument(
        "--llm-transport",
        choices=("sdk", "cli"),
        default="sdk",
        help="Arm C via the Anthropic SDK (API key) or `claude -p` (logged-in account)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    items = load_items(args.dataset)
    if args.limit:
        items = items[: args.limit]
    label = args.label or args.dataset
    stamp = datetime.now(UTC).date().isoformat()
    rows_path = args.output_dir / f"intent_classifiers_{label}_{stamp}.jsonl"
    metrics_path = args.output_dir / f"intent_classifiers_{label}_{stamp}_metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runner = ArmRunner(args.arms, args.jev_cache, args.llm_cache, args.llm_transport)
    all_rows: list[dict] = []
    metrics: dict[str, dict] = {}
    try:
        for arm in args.arms:
            started = time.perf_counter()
            rows = score_rows(runner, arm, items, 1, args.dataset)
            if args.stability_repeats > 1 and arm != "A":
                subset = items[: args.stability_subset]
                for repeat in range(2, args.stability_repeats + 1):
                    rows.extend(score_rows(runner, arm, subset, repeat, args.dataset))
            all_rows.extend(rows)
            metrics[arm] = arm_metrics(rows)
            errors = metrics[arm]["n_errors"]
            print(
                f"arm {arm} ({ARMS[arm]}): {len(rows)} rows, {errors} errors, "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
            for r in rows:
                if r["error"]:
                    print(f"  ERROR {r['item_id']} repeat {r['repeat']}: {r['error']}")
    finally:
        runner.close()
        with rows_path.open("w", encoding="utf-8") as handle:
            for r in all_rows:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    payload = {
        "dataset": args.dataset,
        "label": label,
        "n_items": len(items),
        "arms": {a: ARMS[a] for a in args.arms},
        "llm_transport": args.llm_transport,
        "generated_at": datetime.now(UTC).isoformat(),
        "stability_repeats": args.stability_repeats,
        "stability_subset": args.stability_subset,
        "rows_path": str(rows_path),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(summary_table(metrics))
    print(f"\nrows: {rows_path}\nmetrics: {metrics_path}")
    return 1 if any(m["n_errors"] for m in metrics.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
