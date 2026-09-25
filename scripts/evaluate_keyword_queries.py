#!/usr/bin/env python3
"""Score the pipeline on terse Google Scholar style keyword queries.

Dataset: data/datasets/benchmark/keyword_queries.json, unfielded queries such
as "accomazzi europa" whose difficulty is telling surnames from missions,
eponymous terms, objects, facilities, years and topic phrases.

Each item runs through process_query with the regex backend and, when
TYPESAFE_API_KEY is set, the jev backend (no paper lookup). Fields are scored
against gold:

    authors      set match after normalising case, comma order and initials
    topic        exact set match of the abs: phrases, plus word-level F1
    bibgroup     set match, case-insensitive
    years        (year_from, year_to) match
    operator     match ("none" when no operator)
    first_author match
    exact        all of the above

and the produced query is sent to ADS (rows=0) for numFound, so the share of
queries that find anything is reported next to the field scores. Objects in
gold are metadata only; gold writes them as abs: terms because the ADS search
API has no object: field.

Rows go to data/datasets/evaluations/keyword_queries_<backend>_<date>.jsonl and
metrics to the matching _metrics.json.

Usage:
    set -a; . ~/projects/omni-experiments/.env; . ~/projects/scix_experiments/.env; set +a
    uv run python scripts/evaluate_keyword_queries.py --backends regex jev
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages/finetune/src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from intent_metrics import topic_words  # noqa: E402

DATASET = REPO_ROOT / "data/datasets/benchmark/keyword_queries.json"
OUTPUT_DIR = REPO_ROOT / "data/datasets/evaluations"
ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
ADS_MIN_INTERVAL_S = 0.3
BACKENDS = ("regex", "jev")
REFERENCE_YEAR = 2025
WORKERS = 4
FIELDS = ("authors", "topic", "bibgroup", "years", "operator", "first_author")


# --------------------------------------------------------------------------- scoring (pure)


def author_key(name: str) -> tuple[str, str | None]:
    """(surname, first initial or None) for an ADS-style or natural-order name.

    "Seager, Sara", "Seager, S." and "Sara Seager" all give ("seager", "s");
    "Seager" gives ("seager", None). Case, accents, hyphens, spaces and the
    first-author caret are ignored in the surname.
    """
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = text.replace("^", "").strip()
    if "," in text:
        last, first = (part.strip() for part in text.split(",", 1))
    else:
        parts = text.split()
        last, first = (parts[-1], " ".join(parts[:-1])) if parts else ("", "")
    initial = re.search(r"[A-Za-z]", first)
    return re.sub(r"[^a-z]", "", last.lower()), initial.group().lower() if initial else None


def _same_author(gold: tuple[str, str | None], pred: tuple[str, str | None]) -> bool:
    """Surnames equal, and initials equal when both sides give one."""
    return gold[0] == pred[0] and (gold[1] is None or pred[1] is None or gold[1] == pred[1])


def authors_match(gold: list[str], pred: list[str]) -> bool:
    """Every gold author paired with a distinct predicted one and nothing left over."""
    if len(gold) != len(pred):
        return False
    remaining = [author_key(p) for p in pred]
    for g in map(author_key, gold):
        hit = next((i for i, p in enumerate(remaining) if _same_author(g, p)), None)
        if hit is None:
            return False
        remaining.pop(hit)
    return True


def term_key(term: str) -> str:
    """A topic phrase as lower-case words, so "X-ray" and "x ray" compare equal."""
    return " ".join(re.findall(r"[a-z0-9]+", term.lower()))


def topics_match(gold: list[str], pred: list[str]) -> bool:
    """Same set of phrases, grouping included: ["a b", "c"] differs from ["a", "b c"]."""
    return {term_key(t) for t in gold} - {""} == {term_key(t) for t in pred} - {""}


def word_prf(gold: list[str], pred: list[str]) -> dict:
    """Word-level tp/fp/fn and F1 of the topic phrases, grouping ignored."""
    g, p = topic_words(" ".join(gold)), topic_words(" ".join(pred))
    tp, fp, fn = len(g & p), len(p - g), len(g - p)
    if tp + fp + fn == 0:
        f1 = 1.0
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}


def score_item(gold: dict, pred: dict) -> dict:
    """Per-field booleans, word F1 and overall exact match for one item."""
    fields = {
        "authors": authors_match(gold["authors"], pred["authors"]),
        "topic": topics_match(gold["topic_terms"], pred["topic_terms"]),
        "bibgroup": {b.lower() for b in gold["bibgroup"]} == {b.lower() for b in pred["bibgroup"]},
        "years": (gold["year_from"], gold["year_to"]) == (pred["year_from"], pred["year_to"]),
        "operator": gold["operator"] == pred["operator"],
        "first_author": gold["first_author"] == pred["first_author"],
    }
    return {
        **fields,
        "exact": all(fields.values()),
        "topic_words": word_prf(gold["topic_terms"], pred["topic_terms"]),
    }


def summarize(rows: list[dict]) -> dict:
    """Field accuracies, micro word F1 and ADS hit rates over scored rows."""
    n = len(rows)
    if not n:
        return {"n": 0}
    out: dict = {"n": n}
    for field in (*FIELDS, "exact"):
        out[f"{field}_accuracy"] = sum(r["score"][field] for r in rows) / n
    tp = sum(r["score"]["topic_words"]["tp"] for r in rows)
    fp = sum(r["score"]["topic_words"]["fp"] for r in rows)
    fn = sum(r["score"]["topic_words"]["fn"] for r in rows)
    out["topic_word_f1_micro"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    out["topic_word_f1_mean"] = sum(r["score"]["topic_words"]["f1"] for r in rows) / n
    out["hits_gt0_rate"] = sum(1 for r in rows if (r["ads_hits"] or 0) > 0) / n
    out["ads_error_rate"] = sum(1 for r in rows if r["ads_error"]) / n
    gold_found = [r for r in rows if r["gold_hits"]]
    out["hits_gt0_rate_where_gold_finds"] = (
        sum(1 for r in gold_found if (r["ads_hits"] or 0) > 0) / len(gold_found)
        if gold_found
        else None
    )
    return out


def metrics_by_stratum(rows: list[dict]) -> dict:
    strata = sorted({r["stratum"] for r in rows})
    return {
        "overall": summarize(rows),
        "by_stratum": {s: summarize([r for r in rows if r["stratum"] == s]) for s in strata},
    }


def prediction_of(result) -> dict:
    """The scored fields of a PipelineResult, in gold's shape."""
    intent = result.intent
    return {
        "authors": list(intent.authors),
        "topic_terms": list(intent.free_text_terms) + list(intent.or_terms),
        "or_terms": list(intent.or_terms),
        "objects": list(intent.objects),
        "bibgroup": sorted(intent.bibgroup),
        "year_from": intent.year_from,
        "year_to": intent.year_to,
        "operator": intent.operator or "none",
        "first_author": intent.first_author,
        "final_query": result.final_query,
        "classifier_called": result.debug_info.classifier_called,
        "classifier_error": result.debug_info.classifier_error,
    }


# --------------------------------------------------------------------------- IO


class AdsCounter:
    """numFound for a query (rows=0), memoised and spaced out to stay polite."""

    def __init__(self, api_key: str) -> None:
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=httpx.Timeout(30.0)
        )
        self._cache: dict[str, tuple[int | None, str | None]] = {}
        self._lock = threading.Lock()
        self._last = 0.0

    def count(self, query: str) -> tuple[int | None, str | None]:
        """(numFound, None), or (None, reason) for an empty query or an ADS error."""
        if not query.strip():
            return None, "empty query"
        with self._lock:
            if query in self._cache:
                return self._cache[query]
            wait = ADS_MIN_INTERVAL_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self._http.get(ADS_SEARCH_URL, params={"q": query, "rows": 0})
                self._last = time.monotonic()
                response.raise_for_status()
                answer = (int(response.json()["response"]["numFound"]), None)
            except httpx.HTTPStatusError as error:
                body = error.response.text
                reason = re.search(r'"msg":"([^"]*)"', body)
                detail = reason.group(1) if reason else body[:200]
                answer = (None, f"HTTP {error.response.status_code}: {detail}")
            self._cache[query] = answer
            return answer

    def close(self) -> None:
        self._http.close()


def load_items(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc["items"]


def run_backend(backend: str, items: list[dict], jev_client, ads: AdsCounter) -> list[dict]:
    from finetune.domains.scix.pipeline import process_query

    def one(item: dict) -> dict:
        result = process_query(item["query"], backend, jev_client, REFERENCE_YEAR)
        pred = prediction_of(result)
        hits, error = ads.count(pred["final_query"])
        return {
            "backend": backend,
            "item_id": item["id"],
            "stratum": item["stratum"],
            "query": item["query"],
            "gold": item["gold"],
            "gold_query": item["gold_query"],
            "gold_hits": item["gold_hits"],
            "ambiguous": item["ambiguous"],
            "prediction": pred,
            "ads_hits": hits,
            "ads_error": error,
            "score": score_item(item["gold"], pred),
        }

    workers = 1 if backend == "regex" else WORKERS
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


def summary_table(metrics: dict[str, dict]) -> str:
    head = (
        "backend",
        "stratum",
        "n",
        "author",
        "topic",
        "topic wF1",
        "bibgroup",
        "years",
        "exact",
        "hits>0",
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for backend, m in metrics.items():
        for name, s in [("overall", m["overall"]), *m["by_stratum"].items()]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        backend,
                        name,
                        str(s["n"]),
                        f"{s['authors_accuracy']:.2f}",
                        f"{s['topic_accuracy']:.2f}",
                        f"{s['topic_word_f1_micro']:.2f}",
                        f"{s['bibgroup_accuracy']:.2f}",
                        f"{s['years_accuracy']:.2f}",
                        f"{s['exact_accuracy']:.2f}",
                        f"{s['hits_gt0_rate']:.2f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--date", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jev-cache", type=Path, default=Path("data/cache/jev_systemone.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    ads_key = os.environ.get("ADS_API_KEY", "")
    if not ads_key:
        raise SystemExit("ADS_API_KEY must be set to count the hits of produced queries")
    items = load_items(args.dataset)[: args.limit]
    backends = list(args.backends)
    jev_client = None
    if "jev" in backends:
        jev_key = os.environ.get("TYPESAFE_API_KEY", "")
        if jev_key:
            from finetune.domains.scix.jev_intent import JevClient

            jev_client = JevClient(api_key=jev_key, cache_path=args.jev_cache)
        else:
            print("TYPESAFE_API_KEY is not set; skipping the jev backend", file=sys.stderr)
            backends.remove("jev")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ads = AdsCounter(ads_key)
    metrics: dict[str, dict] = {}
    try:
        for backend in backends:
            started = time.perf_counter()
            rows = run_backend(backend, items, jev_client, ads)
            stem = args.output_dir / f"keyword_queries_{backend}_{args.date}"
            with stem.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics[backend] = metrics_by_stratum(rows)
            jev_errors = sum(1 for r in rows if r["prediction"]["classifier_error"])
            payload = {
                "dataset": str(args.dataset.relative_to(REPO_ROOT)),
                "backend": backend,
                "n_items": len(rows),
                "classifier_errors": jev_errors,
                "reference_year": REFERENCE_YEAR,
                "generated_at": datetime.now(UTC).isoformat(),
                "rows_path": str(stem.with_suffix(".jsonl").relative_to(REPO_ROOT)),
                "metrics": metrics[backend],
            }
            Path(f"{stem}_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                f"{backend}: {len(rows)} rows, {jev_errors} classifier errors, "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    finally:
        ads.close()
        if jev_client is not None:
            jev_client.close()
    print()
    print(summary_table(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
