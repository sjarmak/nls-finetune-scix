"""Training script for the SciX enrichment NER model.

Fine-tunes SciBERT (allenai/scibert_scivocab_uncased) on enrichment_labels
for token-classification NER with BIO tags. Converts character-level span
annotations from enrichment_train.jsonl / enrichment_val.jsonl into
BIO-tagged token sequences, then fine-tunes with HuggingFace Trainer.

BIO label set (9 tags):
    B-topic, I-topic, B-institution, I-institution,
    B-author, I-author, B-date_range, I-date_range, O

Usage:
    python scripts/train_enrichment_model.py \
        --train-file data/enrichment_train.jsonl \
        --val-file data/enrichment_val.jsonl \
        --output-dir output/enrichment_model

    python scripts/train_enrichment_model.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# BIO label schema
# ---------------------------------------------------------------------------

ENTITY_TYPES = ("topic", "institution", "author", "date_range")

# Map enrichment dataset span types to BIO entity types.
# The enrichment dataset uses "entity" but BIO labels use "institution".
SPAN_TYPE_TO_BIO: dict[str, str] = {
    "entity": "institution",
}

BIO_LABELS: list[str] = ["O"]
for _etype in ENTITY_TYPES:
    BIO_LABELS.append(f"B-{_etype}")
    BIO_LABELS.append(f"I-{_etype}")

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(BIO_LABELS)}
NUM_LABELS: int = len(BIO_LABELS)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and paths for enrichment model training."""

    model_name: str = "allenai/scibert_scivocab_uncased"
    train_file: str = "data/enrichment_train.jsonl"
    val_file: str = "data/enrichment_val.jsonl"
    output_dir: str = "output/enrichment_model"
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 256
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 3
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    log_format: str = "json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_enrichment_records(path: Path) -> list[dict[str, Any]]:
    """Load enrichment records from a JSONL file.

    Each record has:
        {id, text, text_type, spans: [{surface, start, end, type, ...}], ...}
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Character-span to BIO-token conversion
# ---------------------------------------------------------------------------


def _char_labels(text: str, spans: list[dict[str, Any]]) -> list[str]:
    """Assign a BIO label to each character in *text*.

    Spans are expected to have {start, end, type} fields. Overlapping spans
    are resolved by first-in-list priority (earlier spans take precedence).
    """
    char_tags = ["O"] * len(text)

    for span in spans:
        start = span.get("start", 0)
        end = span.get("end", 0)
        entity_type = span.get("type", "")
        entity_type = SPAN_TYPE_TO_BIO.get(entity_type, entity_type)
        if entity_type not in ENTITY_TYPES:
            continue
        if start >= end or start >= len(text):
            continue

        end = min(end, len(text))

        # Only assign if positions are still untagged (first-in-list wins)
        if char_tags[start] != "O":
            continue

        char_tags[start] = f"B-{entity_type}"
        for ci in range(start + 1, end):
            if char_tags[ci] == "O":
                char_tags[ci] = f"I-{entity_type}"

    return char_tags


def align_labels_to_tokens(
    text: str,
    spans: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 256,
) -> dict[str, Any]:
    """Tokenize *text* and produce aligned BIO label IDs.

    Uses the tokenizer's ``offset_mapping`` to map character-level BIO
    labels to sub-word tokens. Special tokens ([CLS], [SEP], PAD) receive
    label id -100 (ignored by CrossEntropyLoss).

    Returns a dict with keys: ``input_ids``, ``attention_mask``, ``labels``.
    """
    char_tags = _char_labels(text, spans)

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    offsets = encoding.pop("offset_mapping")
    labels: list[int] = []

    for token_idx, (start, end) in enumerate(offsets):
        # Special tokens have (0, 0) offsets
        if start == 0 and end == 0:
            labels.append(-100)
            continue

        # Use the label of the first character in the token span
        tag = char_tags[start] if start < len(char_tags) else "O"
        labels.append(LABEL2ID.get(tag, LABEL2ID["O"]))

    encoding["labels"] = labels
    return encoding


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_dataset(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 256,
) -> Any:
    """Convert enrichment records into a HuggingFace Dataset for NER training."""
    try:
        from datasets import Dataset
    except ImportError:
        _exit_missing("datasets")
        raise  # unreachable

    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    all_labels: list[list[int]] = []

    for record in records:
        text = record.get("text", "")
        spans = record.get("spans", [])
        if not text:
            continue

        encoded = align_labels_to_tokens(text, spans, tokenizer, max_length)
        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])
        all_labels.append(encoded["labels"])

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def build_compute_metrics(id2label: dict[int, str]) -> Any:
    """Return a compute_metrics function for the HuggingFace Trainer.

    Computes per-label and micro-average precision, recall, F1.
    """
    import numpy as np

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        predictions, label_ids = eval_pred
        preds = np.argmax(predictions, axis=-1)

        # Flatten, ignoring -100 padding
        true_labels: list[str] = []
        pred_labels: list[str] = []

        for seq_true, seq_pred in zip(label_ids, preds):
            for t, p in zip(seq_true, seq_pred):
                if t == -100:
                    continue
                true_labels.append(id2label.get(int(t), "O"))
                pred_labels.append(id2label.get(int(p), "O"))

        # Micro-average (excluding O)
        entity_labels = {lbl for lbl in set(true_labels) | set(pred_labels) if lbl != "O"}

        tp = 0
        fp = 0
        fn = 0
        per_label: dict[str, dict[str, int]] = {}

        for label in entity_labels:
            per_label[label] = {"tp": 0, "fp": 0, "fn": 0}

        for true, pred in zip(true_labels, pred_labels):
            if true == pred and true != "O":
                tp += 1
                if true in per_label:
                    per_label[true]["tp"] += 1
            elif pred != "O" and true != pred:
                fp += 1
                if pred in per_label:
                    per_label[pred]["fp"] += 1
                if true != "O" and true in per_label:
                    per_label[true]["fn"] += 1
            elif true != "O" and pred != true:
                fn += 1
                if true in per_label:
                    per_label[true]["fn"] += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics: dict[str, float] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

        # Per-label F1
        for label in sorted(entity_labels):
            ltp = per_label[label]["tp"]
            lfp = per_label[label]["fp"]
            lfn = per_label[label]["fn"]
            lp = ltp / (ltp + lfp) if (ltp + lfp) > 0 else 0.0
            lr = ltp / (ltp + lfn) if (ltp + lfn) > 0 else 0.0
            lf1 = 2 * lp * lr / (lp + lr) if (lp + lr) > 0 else 0.0
            metrics[f"{label}_f1"] = round(lf1, 4)

        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Training log persistence
# ---------------------------------------------------------------------------


def save_training_log(
    output_dir: Path,
    config: TrainConfig,
    train_result: Any,
    eval_metrics: dict[str, float] | None,
    elapsed_seconds: float,
) -> None:
    """Write a JSON training log alongside the model checkpoint."""
    log: dict[str, Any] = {
        "model_name": config.model_name,
        "num_labels": NUM_LABELS,
        "bio_labels": BIO_LABELS,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "hyperparameters": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "max_seq_length": config.max_seq_length,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "seed": config.seed,
            "fp16": config.fp16,
            "bf16": config.bf16,
        },
        "training": {},
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    if train_result is not None:
        metrics = getattr(train_result, "metrics", {})
        log["training"] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in metrics.items()
        }

    if eval_metrics is not None:
        log["eval"] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in eval_metrics.items()
        }

    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exit_missing(package: str) -> None:
    """Print an error and exit if a required package is missing."""
    print(f"Error: {package} is not installed.")
    print(f"Install with: pip install {package}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(config: TrainConfig) -> int:
    """Run the full training pipeline.

    Returns 0 on success, non-zero on failure.
    """
    start_time = time.time()

    # --- Import ML dependencies (defer to fail fast if missing) ---
    try:
        import torch  # noqa: F401
    except ImportError:
        _exit_missing("torch")
        return 1

    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        _exit_missing("transformers")
        return 1

    try:
        import numpy  # noqa: F401
    except ImportError:
        _exit_missing("numpy")
        return 1

    # --- Validate input files ---
    train_path = Path(config.train_file)
    val_path = Path(config.val_file)

    if not train_path.exists():
        print(f"Error: training file not found: {train_path}")
        return 1
    if not val_path.exists():
        print(f"Error: validation file not found: {val_path}")
        return 1

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading training data from {train_path}...")
    train_records = load_enrichment_records(train_path)
    print(f"  {len(train_records)} training records")

    print(f"Loading validation data from {val_path}...")
    val_records = load_enrichment_records(val_path)
    print(f"  {len(val_records)} validation records")

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        model_max_length=config.max_seq_length,
    )

    # --- Prepare datasets ---
    print("Tokenizing and aligning BIO labels (train)...")
    train_dataset = prepare_dataset(train_records, tokenizer, config.max_seq_length)
    print(f"  {len(train_dataset)} training examples")

    print("Tokenizing and aligning BIO labels (val)...")
    val_dataset = prepare_dataset(val_records, tokenizer, config.max_seq_length)
    print(f"  {len(val_dataset)} validation examples")

    # --- Load model ---
    print(f"Loading model: {config.model_name}...")
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {trainable_params:,} trainable / {total_params:,} total parameters")

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=config.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # --- Callbacks ---
    callbacks = []
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )

    # --- Trainer ---
    compute_metrics_fn = build_compute_metrics(ID2LABEL)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    # --- Train ---
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Model:          {config.model_name}")
    print(f"  Labels:         {NUM_LABELS} BIO tags")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples:   {len(val_dataset)}")
    print(f"  Epochs:         {config.num_epochs}")
    print(f"  Batch size:     {config.batch_size}")
    print(f"  Learning rate:  {config.learning_rate}")
    print(f"  Early stopping: patience={config.early_stopping_patience}")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    # --- Evaluate ---
    print("\nRunning final evaluation...")
    eval_metrics = trainer.evaluate()
    for key, value in sorted(eval_metrics.items()):
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key}: {formatted}")

    # --- Save ---
    print(f"\nSaving model and tokenizer to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    elapsed = time.time() - start_time
    save_training_log(output_dir, config, train_result, eval_metrics, elapsed)
    print(f"Training log saved to {output_dir / 'training_log.json'}")

    print(f"\nDone! Elapsed: {elapsed:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> TrainConfig:
    """Parse CLI arguments into a TrainConfig."""
    parser = argparse.ArgumentParser(
        description="Train the SciX enrichment NER model (SciBERT token classification).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Train on generated enrichment data:\n"
            "  python scripts/train_enrichment_model.py \\\n"
            "    --train-file data/enrichment_train.jsonl \\\n"
            "    --val-file data/enrichment_val.jsonl \\\n"
            "    --output-dir output/enrichment_model\n"
            "\n"
            "  # Custom hyperparameters:\n"
            "  python scripts/train_enrichment_model.py \\\n"
            "    --train-file data/enrichment_train.jsonl \\\n"
            "    --val-file data/enrichment_val.jsonl \\\n"
            "    --output-dir output/enrichment_model \\\n"
            "    --learning-rate 3e-5 \\\n"
            "    --batch-size 8 \\\n"
            "    --num-epochs 5 \\\n"
            "    --max-seq-length 512\n"
            "\n"
            "  # With mixed precision on GPU:\n"
            "  python scripts/train_enrichment_model.py \\\n"
            "    --train-file data/enrichment_train.jsonl \\\n"
            "    --val-file data/enrichment_val.jsonl \\\n"
            "    --output-dir output/enrichment_model \\\n"
            "    --fp16\n"
            "\n"
            "BIO label set (9 tags):\n"
            "  O, B-topic, I-topic, B-institution, I-institution,\n"
            "  B-author, I-author, B-date_range, I-date_range\n"
            "\n"
            "Model architecture:\n"
            "  SciBERT (allenai/scibert_scivocab_uncased) with a token\n"
            "  classification head. See docs/enrichment-model-selection.md\n"
            "  for the full architecture analysis.\n"
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="HuggingFace model name or path (default: allenai/scibert_scivocab_uncased)",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to enrichment_train.jsonl",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        required=True,
        help="Path to enrichment_val.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/enrichment_model",
        help="Directory to save model checkpoint and training log "
        "(default: output/enrichment_model)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Peak learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device batch size (default: 16)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio of total steps (default: 0.1)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW (default: 0.01)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Maximum sequence length in tokens (default: 256, use 512 for abstracts)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience in eval steps (default: 3, 0 to disable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 mixed precision (requires CUDA)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 mixed precision (requires Ampere+ GPU)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log training metrics every N steps (default: 50)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Evaluate every N steps (default: 200)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps (default: 200)",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Max checkpoints to keep (default: 3)",
    )
    parser.add_argument(
        "--log-format",
        choices=["json", "text"],
        default="json",
        help="Training log format: json or text (default: json)",
    )

    args = parser.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        log_format=args.log_format,
    )


def main() -> int:
    """CLI entry point."""
    config = parse_args()
    return train(config)


if __name__ == "__main__":
    sys.exit(main())
