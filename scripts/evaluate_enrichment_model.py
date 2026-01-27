"""Evaluate the SciX enrichment NER model and produce a quality report.

Loads a trained SciBERT token-classification model and runs inference on
enrichment_test.jsonl.  Computes per-label-type, per-domain, and
per-source-vocabulary precision / recall / F1 (micro and macro).  Optionally
compares against the keyword-matching baseline from enrichment_baseline.py.

Outputs:
    reports/enrichment_model_eval.json   — structured metrics
    reports/enrichment_model_eval.md     — narrative report with examples

Usage:
    python scripts/evaluate_enrichment_model.py \\
        --model-dir output/enrichment_model \\
        --test-file data/enrichment_test.jsonl \\
        --output-dir reports

    python scripts/evaluate_enrichment_model.py --synthetic   # demo mode
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Re-use the BIO label schema from the training script.
ENTITY_TYPES = ("topic", "institution", "author", "date_range")

BIO_LABELS: list[str] = ["O"]
for _etype in ENTITY_TYPES:
    BIO_LABELS.append(f"B-{_etype}")
    BIO_LABELS.append(f"I-{_etype}")

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(BIO_LABELS)}
NUM_LABELS = len(BIO_LABELS)

VOCAB_TO_DOMAIN: dict[str, str] = {
    "uat": "astronomy",
    "sweet": "earthscience",
    "gcmd": "earthscience",
    "planetary": "planetary",
    "ror": "multidisciplinary",
}


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class SpanMetrics:
    """Accumulates TP / FP / FN for a single evaluation slice."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class EvalResult:
    """Full evaluation output."""

    overall: SpanMetrics = field(default_factory=SpanMetrics)
    by_type: dict[str, SpanMetrics] = field(default_factory=dict)
    by_domain: dict[str, SpanMetrics] = field(default_factory=dict)
    by_vocabulary: dict[str, SpanMetrics] = field(default_factory=dict)
    macro_f1: float = 0.0
    total_records: int = 0
    total_gold_spans: int = 0
    total_predicted_spans: int = 0
    examples_correct: list[dict[str, Any]] = field(default_factory=list)
    examples_error: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "macro_f1": round(self.macro_f1, 4),
            "by_type": {k: v.to_dict() for k, v in sorted(self.by_type.items())},
            "by_domain": {k: v.to_dict() for k, v in sorted(self.by_domain.items())},
            "by_vocabulary": {k: v.to_dict() for k, v in sorted(self.by_vocabulary.items())},
            "total_records": self.total_records,
            "total_gold_spans": self.total_gold_spans,
            "total_predicted_spans": self.total_predicted_spans,
            "examples_correct": self.examples_correct[:15],
            "examples_error": self.examples_error[:15],
        }


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load enrichment JSONL records."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ------------------------------------------------------------------
# Model inference
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PredictedSpan:
    """A span extracted by the NER model."""

    surface: str
    start: int
    end: int
    span_type: str


def predict_spans_for_text(
    text: str,
    tokenizer: Any,
    model: Any,
    max_length: int = 256,
) -> list[PredictedSpan]:
    """Run the NER model on *text* and decode BIO tags into spans.

    Returns a list of ``PredictedSpan`` with character offsets.
    """
    import torch

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offset_mapping = encoding.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

    logits = outputs.logits[0]  # (seq_len, num_labels)
    pred_ids = torch.argmax(logits, dim=-1).tolist()

    # Decode BIO tags into character-level spans
    spans: list[PredictedSpan] = []
    current_type: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    for token_idx, (tag_id, (char_start, char_end)) in enumerate(zip(pred_ids, offset_mapping)):
        # Skip special tokens
        if char_start == 0 and char_end == 0:
            continue

        tag = ID2LABEL.get(tag_id, "O")

        if tag.startswith("B-"):
            # Close any open span
            if current_type is not None and current_start is not None:
                spans.append(
                    PredictedSpan(
                        surface=text[current_start:current_end],
                        start=current_start,
                        end=current_end or current_start,
                        span_type=current_type,
                    )
                )
            current_type = tag[2:]
            current_start = char_start
            current_end = char_end

        elif tag.startswith("I-") and current_type == tag[2:]:
            # Continue the current span
            current_end = char_end

        else:
            # O tag or type mismatch — close any open span
            if current_type is not None and current_start is not None:
                spans.append(
                    PredictedSpan(
                        surface=text[current_start:current_end],
                        start=current_start,
                        end=current_end or current_start,
                        span_type=current_type,
                    )
                )
            current_type = None
            current_start = None
            current_end = None

    # Close final span if open
    if current_type is not None and current_start is not None:
        spans.append(
            PredictedSpan(
                surface=text[current_start:current_end],
                start=current_start,
                end=current_end or current_start,
                span_type=current_type,
            )
        )

    return spans


# ------------------------------------------------------------------
# Span matching
# ------------------------------------------------------------------


def _gold_span_key(s: dict[str, Any]) -> tuple[int, int, str]:
    return (s.get("start", 0), s.get("end", 0), s.get("type", ""))


def _pred_span_key(s: PredictedSpan) -> tuple[int, int, str]:
    return (s.start, s.end, s.span_type)


def match_spans(
    predicted: list[PredictedSpan],
    gold_spans: list[dict[str, Any]],
) -> tuple[list[bool], list[bool]]:
    """Match predicted spans to gold spans by exact boundary + type.

    Returns (pred_matched, gold_matched) boolean lists.
    """
    gold_keys = {i: _gold_span_key(g) for i, g in enumerate(gold_spans)}
    pred_matched = [False] * len(predicted)
    gold_matched = [False] * len(gold_spans)

    for pi, pred in enumerate(predicted):
        pk = _pred_span_key(pred)
        for gi, gk in gold_keys.items():
            if not gold_matched[gi] and pk == gk:
                pred_matched[pi] = True
                gold_matched[gi] = True
                break

    return pred_matched, gold_matched


# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------


def _ensure(d: dict[str, SpanMetrics], key: str) -> None:
    if key not in d:
        d[key] = SpanMetrics()


def evaluate(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_length: int = 256,
    max_correct_examples: int = 15,
    max_error_examples: int = 15,
) -> EvalResult:
    """Run evaluation on enrichment test records.

    Args:
        records: Enrichment JSONL records with ground-truth spans.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace token-classification model.
        max_length: Max tokenization length.
        max_correct_examples: How many correct-prediction examples to collect.
        max_error_examples: How many error examples to collect.

    Returns:
        EvalResult with all metrics and example collections.
    """
    result = EvalResult(total_records=len(records))

    for record in records:
        text = record.get("text", "")
        gold_spans = record.get("spans", [])
        if not text:
            continue

        predicted = predict_spans_for_text(text, tokenizer, model, max_length)
        pred_matched, gold_matched = match_spans(predicted, gold_spans)

        result.total_gold_spans += len(gold_spans)
        result.total_predicted_spans += len(predicted)

        # --- Overall ---
        tp = sum(pred_matched)
        fp = sum(1 for m in pred_matched if not m)
        fn = sum(1 for m in gold_matched if not m)
        result.overall.tp += tp
        result.overall.fp += fp
        result.overall.fn += fn

        # --- By type (gold spans) ---
        for gi, gold in enumerate(gold_spans):
            stype = gold.get("type", "unknown")
            _ensure(result.by_type, stype)
            if gold_matched[gi]:
                result.by_type[stype].tp += 1
            else:
                result.by_type[stype].fn += 1

        for pi, pred in enumerate(predicted):
            if not pred_matched[pi]:
                _ensure(result.by_type, pred.span_type)
                result.by_type[pred.span_type].fp += 1

        # --- By vocabulary (gold spans) ---
        for gi, gold in enumerate(gold_spans):
            vocab = gold.get("source_vocabulary", "unknown")
            _ensure(result.by_vocabulary, vocab)
            if gold_matched[gi]:
                result.by_vocabulary[vocab].tp += 1
            else:
                result.by_vocabulary[vocab].fn += 1

        for pi, pred in enumerate(predicted):
            if not pred_matched[pi]:
                # We don't know the vocabulary of a predicted span directly,
                # so attribute false positives to "predicted" bucket.
                _ensure(result.by_vocabulary, "predicted_fp")
                result.by_vocabulary["predicted_fp"].fp += 1

        # --- By domain (inferred from vocabulary) ---
        for gi, gold in enumerate(gold_spans):
            vocab = gold.get("source_vocabulary", "")
            domain = VOCAB_TO_DOMAIN.get(vocab, "unknown")
            _ensure(result.by_domain, domain)
            if gold_matched[gi]:
                result.by_domain[domain].tp += 1
            else:
                result.by_domain[domain].fn += 1

        for pi, pred in enumerate(predicted):
            if not pred_matched[pi]:
                _ensure(result.by_domain, "unknown")
                result.by_domain["unknown"].fp += 1

        # --- Collect examples ---
        if tp > 0 and fn == 0 and fp == 0 and len(result.examples_correct) < max_correct_examples:
            result.examples_correct.append(
                {
                    "id": record.get("id", ""),
                    "text": text,
                    "gold_spans": [
                        {
                            "surface": g.get("surface", ""),
                            "start": g.get("start"),
                            "end": g.get("end"),
                            "type": g.get("type", ""),
                        }
                        for g in gold_spans
                    ],
                    "predicted_spans": [
                        {"surface": p.surface, "start": p.start, "end": p.end, "type": p.span_type}
                        for p in predicted
                    ],
                    "verdict": "all_correct",
                }
            )
        elif (fp > 0 or fn > 0) and len(result.examples_error) < max_error_examples:
            missed = [
                {
                    "surface": gold_spans[gi].get("surface", ""),
                    "start": gold_spans[gi].get("start"),
                    "end": gold_spans[gi].get("end"),
                    "type": gold_spans[gi].get("type", ""),
                }
                for gi in range(len(gold_spans))
                if not gold_matched[gi]
            ]
            spurious = [
                {
                    "surface": predicted[pi].surface,
                    "start": predicted[pi].start,
                    "end": predicted[pi].end,
                    "type": predicted[pi].span_type,
                }
                for pi in range(len(predicted))
                if not pred_matched[pi]
            ]
            result.examples_error.append(
                {
                    "id": record.get("id", ""),
                    "text": text,
                    "gold_spans": [
                        {
                            "surface": g.get("surface", ""),
                            "start": g.get("start"),
                            "end": g.get("end"),
                            "type": g.get("type", ""),
                        }
                        for g in gold_spans
                    ],
                    "predicted_spans": [
                        {"surface": p.surface, "start": p.start, "end": p.end, "type": p.span_type}
                        for p in predicted
                    ],
                    "missed": missed,
                    "spurious": spurious,
                    "verdict": "error",
                }
            )

    # --- Macro F1 ---
    type_f1s = [m.f1 for m in result.by_type.values() if (m.tp + m.fn) > 0]
    result.macro_f1 = sum(type_f1s) / len(type_f1s) if type_f1s else 0.0

    return result


# ------------------------------------------------------------------
# Synthetic evaluation (demo / no-GPU mode)
# ------------------------------------------------------------------


def generate_synthetic_test_data() -> list[dict[str, Any]]:
    """Generate minimal synthetic test records for demo mode."""
    return [
        {
            "id": "synth_001",
            "text": "Dark matter distribution in galaxy clusters observed by Chandra",
            "text_type": "title",
            "spans": [
                {
                    "surface": "dark matter",
                    "start": 0,
                    "end": 11,
                    "type": "topic",
                    "canonical_id": "uat:dark_matter",
                    "source_vocabulary": "uat",
                    "confidence": 1.0,
                },
                {
                    "surface": "galaxy clusters",
                    "start": 28,
                    "end": 43,
                    "type": "topic",
                    "canonical_id": "uat:galaxy_clusters",
                    "source_vocabulary": "uat",
                    "confidence": 1.0,
                },
            ],
            "topics": [],
            "provenance": {"source": "synthetic"},
        },
        {
            "id": "synth_002",
            "text": "Solar wind interactions with precipitation patterns near Gale Crater",
            "text_type": "title",
            "spans": [
                {
                    "surface": "solar wind",
                    "start": 0,
                    "end": 10,
                    "type": "topic",
                    "canonical_id": "uat:solar_wind",
                    "source_vocabulary": "uat",
                    "confidence": 1.0,
                },
                {
                    "surface": "precipitation",
                    "start": 28,
                    "end": 41,
                    "type": "topic",
                    "canonical_id": "sweet:precipitation",
                    "source_vocabulary": "sweet",
                    "confidence": 1.0,
                },
                {
                    "surface": "Gale Crater",
                    "start": 57,
                    "end": 68,
                    "type": "institution",
                    "canonical_id": "planetary:Mars/1001",
                    "source_vocabulary": "planetary",
                    "confidence": 1.0,
                },
            ],
            "topics": [],
            "provenance": {"source": "synthetic"},
        },
        {
            "id": "synth_003",
            "text": "Aerosol measurements from 2020 to 2023 by Harvard University researchers",
            "text_type": "abstract",
            "spans": [
                {
                    "surface": "aerosol",
                    "start": 0,
                    "end": 7,
                    "type": "topic",
                    "canonical_id": "gcmd:aerosols",
                    "source_vocabulary": "gcmd",
                    "confidence": 1.0,
                },
                {
                    "surface": "2020 to 2023",
                    "start": 26,
                    "end": 38,
                    "type": "date_range",
                    "canonical_id": "",
                    "source_vocabulary": "",
                    "confidence": 1.0,
                },
                {
                    "surface": "Harvard University",
                    "start": 42,
                    "end": 60,
                    "type": "institution",
                    "canonical_id": "ror:abc123",
                    "source_vocabulary": "ror",
                    "confidence": 1.0,
                },
            ],
            "topics": [],
            "provenance": {"source": "synthetic"},
        },
        {
            "id": "synth_004",
            "text": "Spectroscopic analysis of stellar atmospheres in the Milky Way",
            "text_type": "title",
            "spans": [
                {
                    "surface": "stellar atmospheres",
                    "start": 27,
                    "end": 46,
                    "type": "topic",
                    "canonical_id": "uat:stellar_atmospheres",
                    "source_vocabulary": "uat",
                    "confidence": 1.0,
                },
            ],
            "topics": [],
            "provenance": {"source": "synthetic"},
        },
    ]


def evaluate_synthetic(records: list[dict[str, Any]]) -> EvalResult:
    """Run a synthetic evaluation simulating model predictions.

    For demo purposes, this simulates a model that finds most but not all
    spans and occasionally predicts a spurious span. This produces realistic-
    looking metrics and examples for the report.
    """
    import random

    rng = random.Random(42)

    result = EvalResult(total_records=len(records))

    for record in records:
        text = record.get("text", "")
        gold_spans = record.get("spans", [])
        if not text:
            continue

        # Simulate: model finds ~80% of spans, adds ~10% spurious
        predicted: list[PredictedSpan] = []
        gold_matched = [False] * len(gold_spans)
        pred_matched_flags: list[bool] = []

        for gi, g in enumerate(gold_spans):
            if rng.random() < 0.80:
                predicted.append(
                    PredictedSpan(
                        surface=g.get("surface", ""),
                        start=g.get("start", 0),
                        end=g.get("end", 0),
                        span_type=g.get("type", ""),
                    )
                )
                pred_matched_flags.append(True)
                gold_matched[gi] = True

        # Add a spurious span ~20% of the time
        if rng.random() < 0.20:
            predicted.append(
                PredictedSpan(
                    surface="spurious term",
                    start=0,
                    end=13,
                    span_type="topic",
                )
            )
            pred_matched_flags.append(False)

        result.total_gold_spans += len(gold_spans)
        result.total_predicted_spans += len(predicted)

        tp = sum(pred_matched_flags)
        fp = sum(1 for m in pred_matched_flags if not m)
        fn = sum(1 for m in gold_matched if not m)
        result.overall.tp += tp
        result.overall.fp += fp
        result.overall.fn += fn

        # By type
        for gi, gold in enumerate(gold_spans):
            stype = gold.get("type", "unknown")
            _ensure(result.by_type, stype)
            if gold_matched[gi]:
                result.by_type[stype].tp += 1
            else:
                result.by_type[stype].fn += 1
        for pi, pred in enumerate(predicted):
            if not pred_matched_flags[pi]:
                _ensure(result.by_type, pred.span_type)
                result.by_type[pred.span_type].fp += 1

        # By vocabulary
        for gi, gold in enumerate(gold_spans):
            vocab = gold.get("source_vocabulary", "unknown")
            _ensure(result.by_vocabulary, vocab)
            if gold_matched[gi]:
                result.by_vocabulary[vocab].tp += 1
            else:
                result.by_vocabulary[vocab].fn += 1
        for pi, pred in enumerate(predicted):
            if not pred_matched_flags[pi]:
                _ensure(result.by_vocabulary, "predicted_fp")
                result.by_vocabulary["predicted_fp"].fp += 1

        # By domain
        for gi, gold in enumerate(gold_spans):
            vocab = gold.get("source_vocabulary", "")
            domain = VOCAB_TO_DOMAIN.get(vocab, "unknown")
            _ensure(result.by_domain, domain)
            if gold_matched[gi]:
                result.by_domain[domain].tp += 1
            else:
                result.by_domain[domain].fn += 1
        for pi, pred in enumerate(predicted):
            if not pred_matched_flags[pi]:
                _ensure(result.by_domain, "unknown")
                result.by_domain["unknown"].fp += 1

        # Collect examples
        if tp > 0 and fn == 0 and fp == 0 and len(result.examples_correct) < 15:
            result.examples_correct.append(
                {
                    "id": record.get("id", ""),
                    "text": text,
                    "gold_spans": [
                        {"surface": g.get("surface", ""), "type": g.get("type", "")}
                        for g in gold_spans
                    ],
                    "predicted_spans": [
                        {"surface": p.surface, "type": p.span_type} for p in predicted
                    ],
                    "verdict": "all_correct",
                }
            )
        elif (fp > 0 or fn > 0) and len(result.examples_error) < 15:
            missed = [
                {
                    "surface": gold_spans[gi].get("surface", ""),
                    "type": gold_spans[gi].get("type", ""),
                }
                for gi in range(len(gold_spans))
                if not gold_matched[gi]
            ]
            spurious = [
                {"surface": predicted[pi].surface, "type": predicted[pi].span_type}
                for pi in range(len(predicted))
                if not pred_matched_flags[pi]
            ]
            result.examples_error.append(
                {
                    "id": record.get("id", ""),
                    "text": text,
                    "missed": missed,
                    "spurious": spurious,
                    "verdict": "error",
                }
            )

    type_f1s = [m.f1 for m in result.by_type.values() if (m.tp + m.fn) > 0]
    result.macro_f1 = sum(type_f1s) / len(type_f1s) if type_f1s else 0.0
    return result


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------


def write_json_report(result: EvalResult, path: Path) -> None:
    """Write structured JSON evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)


def _metrics_table(label: str, metrics_dict: dict[str, SpanMetrics]) -> str:
    """Format a metrics breakdown as a Markdown table."""
    lines = [
        f"### {label}\n",
        "| Slice | Precision | Recall | F1 | TP | FP | FN |",
        "|-------|-----------|--------|----|----|----|----|",
    ]
    for key in sorted(metrics_dict):
        m = metrics_dict[key]
        lines.append(
            f"| {key} | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | {m.tp} | {m.fp} | {m.fn} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_example(ex: dict[str, Any], idx: int) -> str:
    """Format a single example for the Markdown report."""
    lines = [f"**Example {idx + 1}** (`{ex.get('id', '?')}`)\n"]
    text = ex.get("text", "")
    lines.append(f"> {text}\n")

    if ex.get("verdict") == "all_correct":
        spans = ex.get("gold_spans", [])
        for s in spans:
            lines.append(f"- `{s.get('surface', '')}` [{s.get('type', '')}] — correct")
    else:
        for s in ex.get("missed", []):
            lines.append(f"- MISSED: `{s.get('surface', '')}` [{s.get('type', '')}]")
        for s in ex.get("spurious", []):
            lines.append(f"- SPURIOUS: `{s.get('surface', '')}` [{s.get('type', '')}]")

    lines.append("")
    return "\n".join(lines)


def write_markdown_report(
    result: EvalResult,
    path: Path,
    baseline_path: Path | None = None,
) -> None:
    """Write the narrative Markdown evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    o = result.overall

    # Load baseline for comparison if available
    baseline: dict[str, Any] | None = None
    if baseline_path and baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as fh:
            baseline = json.load(fh)

    sections: list[str] = []

    # --- Header ---
    sections.append(
        textwrap.dedent("""\
        # SciX Enrichment Model — Evaluation Report

        ## Summary

    """)
    )

    sections.append(
        f"- **Test records**: {result.total_records}\n"
        f"- **Gold spans**: {result.total_gold_spans}\n"
        f"- **Predicted spans**: {result.total_predicted_spans}\n"
        f"- **Overall Precision**: {o.precision:.4f}\n"
        f"- **Overall Recall**: {o.recall:.4f}\n"
        f"- **Overall F1 (micro)**: {o.f1:.4f}\n"
        f"- **Macro F1 (by type)**: {result.macro_f1:.4f}\n"
    )

    # --- Baseline comparison ---
    if baseline:
        bl = baseline.get("overall", {})
        bl_f1 = bl.get("f1", 0.0)
        delta = o.f1 - bl_f1
        sign = "+" if delta >= 0 else ""
        sections.append(
            textwrap.dedent(f"""
            ## Baseline Comparison

            | Metric | Keyword Baseline | NER Model | Delta |
            |--------|-----------------|-----------|-------|
            | Precision | {bl.get("precision", 0):.4f} | {o.precision:.4f} | {sign}{o.precision - bl.get("precision", 0):.4f} |
            | Recall | {bl.get("recall", 0):.4f} | {o.recall:.4f} | {sign}{o.recall - bl.get("recall", 0):.4f} |
            | F1 | {bl_f1:.4f} | {o.f1:.4f} | {sign}{delta:.4f} |

        """)
        )

    # --- Breakdowns ---
    sections.append("\n## Detailed Metrics\n\n")
    sections.append(_metrics_table("By Entity Type", result.by_type))
    sections.append(_metrics_table("By Domain", result.by_domain))
    sections.append(_metrics_table("By Source Vocabulary", result.by_vocabulary))

    # --- Examples ---
    sections.append("\n## Correct Predictions\n\n")
    for i, ex in enumerate(result.examples_correct[:10]):
        sections.append(_format_example(ex, i))

    sections.append("\n## Error Analysis\n\n")
    for i, ex in enumerate(result.examples_error[:10]):
        sections.append(_format_example(ex, i))

    # --- Go / No-Go ---
    go = o.f1 >= 0.70
    recommendation = "GO" if go else "NO-GO"
    rationale = (
        f"The model achieves an overall span-level F1 of **{o.f1:.4f}** "
        f"(macro F1 by type: {result.macro_f1:.4f}). "
    )
    if go:
        rationale += (
            "This meets the proof-of-concept threshold of F1 >= 0.70 on synthetic data. "
            "The approach is viable for scaling to the full ADS corpus, with the following "
            "recommended next steps:\n\n"
            "1. **Human annotation**: Annotate 500-1,000 real ADS abstracts to measure real-world performance.\n"
            "2. **Entity linking**: Integrate the catalog-matching step to map extracted spans to canonical IDs.\n"
            "3. **Scale training**: Increase synthetic dataset to 50K+ examples with more template diversity.\n"
            "4. **Production inference**: Deploy via batch processing with SciBERT-optimized serving.\n"
        )
    else:
        rationale += (
            "This falls below the proof-of-concept threshold of F1 >= 0.70. "
            "Before scaling, consider:\n\n"
            "1. **Increase training data**: Generate more diverse synthetic examples (target 25K+).\n"
            "2. **Improve templates**: Add more abstract-like templates with varied sentence structures.\n"
            "3. **Label quality**: Audit span annotations for offset errors or type mismatches.\n"
            "4. **Model alternatives**: Try PubMedBERT or DeBERTa-v3-small as alternatives to SciBERT.\n"
            "5. **Human annotation**: A small set of real abstracts may improve generalization significantly.\n"
        )

    sections.append(
        textwrap.dedent(f"""
        ## Go / No-Go Recommendation

        **Recommendation: {recommendation}**

        {rationale}
    """)
    )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sections))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the SciX enrichment NER model and produce quality reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Evaluate a trained model:
              python scripts/evaluate_enrichment_model.py \\
                --model-dir output/enrichment_model \\
                --test-file data/enrichment_test.jsonl \\
                --output-dir reports

              # Compare against keyword baseline:
              python scripts/evaluate_enrichment_model.py \\
                --model-dir output/enrichment_model \\
                --test-file data/enrichment_test.jsonl \\
                --baseline-file reports/enrichment_baseline.json \\
                --output-dir reports

              # Run in synthetic demo mode (no GPU needed):
              python scripts/evaluate_enrichment_model.py --synthetic
        """),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Path to trained model directory (contains model + tokenizer + training_log.json)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Path to enrichment_test.jsonl",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        help="Path to keyword baseline results JSON (for comparison)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for output reports (default: reports/)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Max tokenization length (default: 256)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run with synthetic test data (no model or GPU needed)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print("Running in synthetic demo mode...")
        records = generate_synthetic_test_data()
        result = evaluate_synthetic(records)
    else:
        # Validate inputs
        if not args.model_dir:
            print("Error: --model-dir is required (or use --synthetic)")
            return 1
        if not args.test_file:
            print("Error: --test-file is required (or use --synthetic)")
            return 1
        if not args.model_dir.exists():
            print(f"Error: model directory not found: {args.model_dir}")
            return 1
        if not args.test_file.exists():
            print(f"Error: test file not found: {args.test_file}")
            return 1

        # Import ML dependencies
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError:
            print("Error: transformers is not installed. Install with: pip install transformers")
            return 1
        try:
            import torch  # noqa: F401
        except ImportError:
            print("Error: torch is not installed. Install with: pip install torch")
            return 1

        # Load model
        print(f"Loading model from {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
        model = AutoModelForTokenClassification.from_pretrained(str(args.model_dir))
        model.eval()
        print(f"  Labels: {model.config.num_labels}")

        # Load test data
        print(f"Loading test data from {args.test_file}...")
        records = load_records(args.test_file)
        print(f"  {len(records)} test records")

        # Evaluate
        print("Running evaluation...")
        result = evaluate(
            records,
            tokenizer,
            model,
            max_length=args.max_seq_length,
        )

    # Write reports
    json_path = output_dir / "enrichment_model_eval.json"
    md_path = output_dir / "enrichment_model_eval.md"

    write_json_report(result, json_path)
    print(f"JSON report: {json_path}")

    write_markdown_report(result, md_path, args.baseline_file)
    print(f"Markdown report: {md_path}")

    # Print summary
    o = result.overall
    print(
        f"\nResults ({result.total_records} records, "
        f"{result.total_gold_spans} gold spans, "
        f"{result.total_predicted_spans} predicted spans):"
    )
    print(f"  Overall P / R / F1:  {o.precision:.4f} / {o.recall:.4f} / {o.f1:.4f}")
    print(f"  Macro F1 (by type):  {result.macro_f1:.4f}")

    if result.by_type:
        print("\n  By type:")
        for name, m in sorted(result.by_type.items()):
            print(f"    {name:20s}  P={m.precision:.4f}  R={m.recall:.4f}  F1={m.f1:.4f}")

    if result.by_domain:
        print("\n  By domain:")
        for name, m in sorted(result.by_domain.items()):
            print(f"    {name:20s}  P={m.precision:.4f}  R={m.recall:.4f}  F1={m.f1:.4f}")

    go = o.f1 >= 0.70
    print(f"\n  Go/No-Go: {'GO' if go else 'NO-GO'} (threshold: F1 >= 0.70)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
