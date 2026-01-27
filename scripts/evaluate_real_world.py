"""Evaluate NER model performance on real ADS abstracts.

Runs the trained NER model on annotated real ADS abstracts and compares
predictions to catalog-based annotations. Computes P/R/F1 and documents
the synthetic-to-real performance gap (US-011).

The key insight: the NER model was trained on synthetic snippets generated
from the same catalogs. Real ADS abstracts contain natural language that
is structurally different from the training data. This evaluation measures
how well the model generalizes from synthetic to real text.

Usage:
    python scripts/evaluate_real_world.py \
        --model-dir output/enrichment_model \
        --annotated-file data/evaluation/ads_sample_annotated.jsonl \
        --synthetic-eval reports/enrichment_model_eval.json \
        --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# BIO label schema (must match the training script)
# ---------------------------------------------------------------------------

ENTITY_TYPES = ("topic", "institution", "author", "date_range")

BIO_LABELS: list[str] = ["O"]
for _etype in ENTITY_TYPES:
    BIO_LABELS.append(f"B-{_etype}")
    BIO_LABELS.append(f"I-{_etype}")

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(BIO_LABELS)}

# Map BIO entity types back to enrichment dataset span types.
BIO_TO_SPAN_TYPE: dict[str, str] = {
    "institution": "entity",
}

VOCAB_TO_DOMAIN: dict[str, str] = {
    "uat": "astronomy",
    "sweet": "earthscience",
    "gcmd": "earthscience",
    "planetary": "planetary",
    "ror": "multidisciplinary",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


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
class RealWorldEvalResult:
    """Full real-world evaluation result."""

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


# ---------------------------------------------------------------------------
# NER inference
# ---------------------------------------------------------------------------


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
    max_length: int = 512,
) -> list[PredictedSpan]:
    """Run the NER model on text and decode BIO tags into spans."""
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

    logits = outputs.logits[0]
    pred_ids = torch.argmax(logits, dim=-1).tolist()

    spans: list[PredictedSpan] = []
    current_type: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    for tag_id, (char_start, char_end) in zip(pred_ids, offset_mapping):
        if char_start == 0 and char_end == 0:
            continue

        tag = ID2LABEL.get(tag_id, "O")

        if tag.startswith("B-"):
            if current_type is not None and current_start is not None:
                spans.append(PredictedSpan(
                    surface=text[current_start:current_end],
                    start=current_start,
                    end=current_end or current_start,
                    span_type=current_type,
                ))
            raw_type = tag[2:]
            current_type = BIO_TO_SPAN_TYPE.get(raw_type, raw_type)
            current_start = char_start
            current_end = char_end

        elif tag.startswith("I-") and current_type == BIO_TO_SPAN_TYPE.get(tag[2:], tag[2:]):
            current_end = char_end

        else:
            if current_type is not None and current_start is not None:
                spans.append(PredictedSpan(
                    surface=text[current_start:current_end],
                    start=current_start,
                    end=current_end or current_start,
                    span_type=current_type,
                ))
            current_type = None
            current_start = None
            current_end = None

    if current_type is not None and current_start is not None:
        spans.append(PredictedSpan(
            surface=text[current_start:current_end],
            start=current_start,
            end=current_end or current_start,
            span_type=current_type,
        ))

    return spans


# ---------------------------------------------------------------------------
# Span matching — relaxed matching for real-world evaluation
# ---------------------------------------------------------------------------


def _gold_key(s: dict[str, Any]) -> tuple[int, int, str]:
    return (s.get("start", 0), s.get("end", 0), s.get("type", ""))


def _pred_key(s: PredictedSpan) -> tuple[int, int, str]:
    return (s.start, s.end, s.span_type)


def match_spans_exact(
    predicted: list[PredictedSpan],
    gold_spans: list[dict[str, Any]],
) -> tuple[list[bool], list[bool]]:
    """Exact boundary + type matching."""
    gold_keys = {i: _gold_key(g) for i, g in enumerate(gold_spans)}
    pred_matched = [False] * len(predicted)
    gold_matched = [False] * len(gold_spans)

    for pi, pred in enumerate(predicted):
        pk = _pred_key(pred)
        for gi, gk in gold_keys.items():
            if not gold_matched[gi] and pk == gk:
                pred_matched[pi] = True
                gold_matched[gi] = True
                break

    return pred_matched, gold_matched


def match_spans_partial(
    predicted: list[PredictedSpan],
    gold_spans: list[dict[str, Any]],
    overlap_threshold: float = 0.5,
) -> tuple[list[bool], list[bool]]:
    """Partial overlap matching — more forgiving for real-world evaluation.

    A match is counted if:
    - Types are the same
    - The overlap ratio (intersection / union) >= threshold
    """
    pred_matched = [False] * len(predicted)
    gold_matched = [False] * len(gold_spans)

    for pi, pred in enumerate(predicted):
        for gi, gold in enumerate(gold_spans):
            if gold_matched[gi]:
                continue

            gold_type = gold.get("type", "")
            if pred.span_type != gold_type:
                continue

            gold_start = gold.get("start", 0)
            gold_end = gold.get("end", 0)

            # Calculate overlap
            overlap_start = max(pred.start, gold_start)
            overlap_end = min(pred.end, gold_end)
            overlap = max(0, overlap_end - overlap_start)

            union = (pred.end - pred.start) + (gold_end - gold_start) - overlap

            if union > 0 and overlap / union >= overlap_threshold:
                pred_matched[pi] = True
                gold_matched[gi] = True
                break

    return pred_matched, gold_matched


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _ensure(d: dict[str, SpanMetrics], key: str) -> None:
    if key not in d:
        d[key] = SpanMetrics()


def evaluate_real_world(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_length: int = 512,
    match_mode: str = "exact",
    max_examples: int = 15,
) -> RealWorldEvalResult:
    """Run real-world evaluation on annotated ADS abstracts.

    Args:
        records: Annotated abstract records with gold span annotations.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace token-classification model.
        max_length: Max tokenization length.
        match_mode: "exact" or "partial" span matching.
        max_examples: Max examples to collect for report.

    Returns:
        RealWorldEvalResult with all metrics.
    """
    result = RealWorldEvalResult(total_records=len(records))
    match_fn = match_spans_exact if match_mode == "exact" else match_spans_partial

    for i, record in enumerate(records):
        text = record.get("abstract", "")
        gold_spans = record.get("spans", [])
        if not text:
            continue

        predicted = predict_spans_for_text(text, tokenizer, model, max_length)
        pred_matched, gold_matched = match_fn(predicted, gold_spans)

        result.total_gold_spans += len(gold_spans)
        result.total_predicted_spans += len(predicted)

        tp = sum(pred_matched)
        fp = sum(1 for m in pred_matched if not m)
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
            if not pred_matched[pi]:
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
            if not pred_matched[pi]:
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
            if not pred_matched[pi]:
                _ensure(result.by_domain, "unknown")
                result.by_domain["unknown"].fp += 1

        # Collect examples
        bibcode = record.get("bibcode", f"record_{i}")
        if tp > 0 and fn == 0 and fp == 0 and len(result.examples_correct) < max_examples:
            result.examples_correct.append({
                "bibcode": bibcode,
                "text_snippet": text[:200],
                "gold_count": len(gold_spans),
                "pred_count": len(predicted),
                "gold_spans": [
                    {"surface": g.get("surface", ""), "type": g.get("type", "")}
                    for g in gold_spans[:5]
                ],
                "verdict": "all_correct",
            })
        elif (fp > 0 or fn > 0) and len(result.examples_error) < max_examples:
            missed = [
                {"surface": gold_spans[gi].get("surface", ""), "type": gold_spans[gi].get("type", "")}
                for gi in range(len(gold_spans))
                if not gold_matched[gi]
            ][:5]
            spurious = [
                {"surface": predicted[pi].surface, "type": predicted[pi].span_type}
                for pi in range(len(predicted))
                if not pred_matched[pi]
            ][:5]
            result.examples_error.append({
                "bibcode": bibcode,
                "text_snippet": text[:200],
                "gold_count": len(gold_spans),
                "pred_count": len(predicted),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "missed": missed,
                "spurious": spurious,
                "verdict": "error",
            })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(records)} abstracts...")

    # Macro F1
    type_f1s = [m.f1 for m in result.by_type.values() if (m.tp + m.fn) > 0]
    result.macro_f1 = sum(type_f1s) / len(type_f1s) if type_f1s else 0.0

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_json_report(result: RealWorldEvalResult, path: Path) -> None:
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


def write_markdown_report(
    result: RealWorldEvalResult,
    path: Path,
    synthetic_eval: dict[str, Any] | None = None,
) -> None:
    """Write narrative Markdown report with synthetic-to-real comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)

    o = result.overall
    sections: list[str] = []

    # Header
    sections.append(textwrap.dedent("""\
        # SciX Enrichment Model — Real-World Evaluation Report

        ## Summary

    """))

    sections.append(
        f"- **Test abstracts**: {result.total_records}\n"
        f"- **Gold spans**: {result.total_gold_spans}\n"
        f"- **Predicted spans**: {result.total_predicted_spans}\n"
        f"- **Overall Precision**: {o.precision:.4f}\n"
        f"- **Overall Recall**: {o.recall:.4f}\n"
        f"- **Overall F1 (micro)**: {o.f1:.4f}\n"
        f"- **Macro F1 (by type)**: {result.macro_f1:.4f}\n"
    )

    # Synthetic-to-real comparison
    if synthetic_eval:
        synth_overall = synthetic_eval.get("overall", {})
        synth_f1 = synth_overall.get("f1", 0.0)
        synth_p = synth_overall.get("precision", 0.0)
        synth_r = synth_overall.get("recall", 0.0)

        delta_f1 = o.f1 - synth_f1
        delta_p = o.precision - synth_p
        delta_r = o.recall - synth_r

        sections.append(textwrap.dedent(f"""
        ## Synthetic-to-Real Performance Gap

        | Metric | Synthetic Test | Real ADS Abstracts | Delta |
        |--------|---------------|-------------------|-------|
        | Precision | {synth_p:.4f} | {o.precision:.4f} | {delta_p:+.4f} |
        | Recall | {synth_r:.4f} | {o.recall:.4f} | {delta_r:+.4f} |
        | F1 | {synth_f1:.4f} | {o.f1:.4f} | {delta_f1:+.4f} |

        **Performance gap**: F1 dropped by **{abs(delta_f1):.4f}** from synthetic to real data.

        """))

        # Per-type comparison
        synth_by_type = synthetic_eval.get("by_type", {})
        if synth_by_type:
            sections.append("### Per-Type Gap\n\n")
            sections.append("| Type | Synthetic F1 | Real F1 | Delta |\n")
            sections.append("|------|-------------|---------|-------|\n")
            all_types = sorted(set(list(synth_by_type.keys()) + list(result.by_type.keys())))
            for t in all_types:
                sf1 = synth_by_type.get(t, {}).get("f1", 0.0)
                rf1 = result.by_type.get(t, SpanMetrics()).f1
                sections.append(f"| {t} | {sf1:.4f} | {rf1:.4f} | {rf1 - sf1:+.4f} |\n")
            sections.append("\n")

    # Detailed metrics
    sections.append("\n## Detailed Metrics\n\n")
    sections.append(_metrics_table("By Entity Type", result.by_type))
    sections.append(_metrics_table("By Domain", result.by_domain))
    sections.append(_metrics_table("By Source Vocabulary", result.by_vocabulary))

    # Examples
    sections.append("\n## Correct Predictions\n\n")
    for i, ex in enumerate(result.examples_correct[:10]):
        sections.append(f"**Example {i + 1}** (`{ex.get('bibcode', '?')}`)\n")
        sections.append(f"> {ex.get('text_snippet', '')[:150]}...\n")
        for s in ex.get("gold_spans", []):
            sections.append(f"- `{s.get('surface', '')}` [{s.get('type', '')}] — correct\n")
        sections.append("\n")

    sections.append("\n## Error Analysis\n\n")
    for i, ex in enumerate(result.examples_error[:10]):
        sections.append(f"**Example {i + 1}** (`{ex.get('bibcode', '?')}`)\n")
        sections.append(f"> {ex.get('text_snippet', '')[:150]}...\n")
        sections.append(f"- TP={ex.get('tp', 0)}, FP={ex.get('fp', 0)}, FN={ex.get('fn', 0)}\n")
        for s in ex.get("missed", []):
            sections.append(f"- MISSED: `{s.get('surface', '')}` [{s.get('type', '')}]\n")
        for s in ex.get("spurious", []):
            sections.append(f"- SPURIOUS: `{s.get('surface', '')}` [{s.get('type', '')}]\n")
        sections.append("\n")

    # Analysis
    sections.append(textwrap.dedent(f"""
    ## Performance Gap Analysis

    The synthetic-to-real gap is expected because:

    1. **Training data distribution**: The model was trained on template-generated snippets
       with catalog entries inserted deterministically. Real abstracts use natural language
       that may reference the same concepts with different phrasing.

    2. **Text length**: Real abstracts are much longer (avg ~1,500 chars) than synthetic
       snippets (~50-200 chars). The SciBERT tokenizer truncates to {512} tokens,
       which may miss spans in later portions of long abstracts.

    3. **Vocabulary mismatch**: Catalog labels are canonical forms (e.g., "dark matter")
       while abstracts may use abbreviations, acronyms, or alternative phrasings
       (e.g., "DM", "non-baryonic matter").

    4. **Context effects**: In synthetic data, catalog entries appear in predictable
       template positions. In real text, the same terms appear in varied syntactic
       contexts that the model hasn't seen during training.

    5. **Annotation methodology**: The "gold" annotations on real text are derived from
       catalog keyword matching, which has known limitations (false positives from
       short common terms, inability to handle abbreviations). This means some
       "errors" may actually be correct model behavior.

    """))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sections))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate NER model on real ADS abstracts.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to trained NER model directory",
    )
    parser.add_argument(
        "--annotated-file",
        type=Path,
        required=True,
        help="Path to ads_sample_annotated.jsonl",
    )
    parser.add_argument(
        "--synthetic-eval",
        type=Path,
        default=None,
        help="Path to synthetic evaluation JSON (for gap comparison)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max tokenization length (default: 512, longer for abstracts)",
    )
    parser.add_argument(
        "--match-mode",
        choices=["exact", "partial"],
        default="exact",
        help="Span matching mode (default: exact)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.model_dir.exists():
        print(f"Error: model directory not found: {args.model_dir}")
        return 1
    if not args.annotated_file.exists():
        print(f"Error: annotated file not found: {args.annotated_file}")
        return 1

    # Import ML dependencies
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        print("Error: transformers is not installed.")
        return 1
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Error: torch is not installed.")
        return 1

    # Load model
    print(f"Loading NER model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(args.model_dir))
    model.eval()
    print(f"  Labels: {model.config.num_labels}")

    # Load annotated abstracts
    print(f"Loading annotated abstracts from {args.annotated_file}...")
    records: list[dict[str, Any]] = []
    with open(args.annotated_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  {len(records)} annotated abstracts")

    # Load synthetic eval for comparison
    synthetic_eval: dict[str, Any] | None = None
    if args.synthetic_eval and args.synthetic_eval.exists():
        print(f"Loading synthetic evaluation from {args.synthetic_eval}...")
        with open(args.synthetic_eval, encoding="utf-8") as fh:
            synthetic_eval = json.load(fh)

    # Run evaluation
    print(f"Running real-world evaluation (match_mode={args.match_mode})...")
    result = evaluate_real_world(
        records,
        tokenizer,
        model,
        max_length=args.max_seq_length,
        match_mode=args.match_mode,
    )

    # Write reports
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "enrichment_real_world_eval.json"
    write_json_report(result, json_path)
    print(f"JSON report: {json_path}")

    md_path = args.output_dir / "enrichment_real_world_eval.md"
    write_markdown_report(result, md_path, synthetic_eval)
    print(f"Markdown report: {md_path}")

    # Print summary
    o = result.overall
    print(f"\nReal-World Results ({result.total_records} abstracts, "
          f"{result.total_gold_spans} gold spans, "
          f"{result.total_predicted_spans} predicted):")
    print(f"  Overall P / R / F1:  {o.precision:.4f} / {o.recall:.4f} / {o.f1:.4f}")
    print(f"  Macro F1 (by type):  {result.macro_f1:.4f}")

    if result.by_type:
        print("\n  By type:")
        for name, m in sorted(result.by_type.items()):
            print(f"    {name:20s}  P={m.precision:.4f}  R={m.recall:.4f}  F1={m.f1:.4f}")

    # Synthetic comparison
    if synthetic_eval:
        synth_f1 = synthetic_eval.get("overall", {}).get("f1", 0.0)
        delta = o.f1 - synth_f1
        print(f"\n  Synthetic-to-Real Gap:")
        print(f"    Synthetic F1: {synth_f1:.4f}")
        print(f"    Real F1:      {o.f1:.4f}")
        print(f"    Delta:        {delta:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
