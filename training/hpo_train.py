#!/usr/bin/env python3
"""Hyperparameter optimization training script for SciX NLS fine-tuning.

Uses Optuna to search over LoRA rank, learning rate, L2 regularization,
and epoch count. Compatible with accelerate launch for multi-GPU training.

Dependencies (add to docker/requirements.txt if not present):
    optuna>=3.6.0
    click>=8.1.0

Usage:
    # Single GPU (Unsloth active when available):
    python scripts/training/hpo_train.py --n-trials 20 --storage sqlite:///hpo.db

    # Multi-GPU data parallel (Unsloth disabled, standard PEFT):
    accelerate launch --num_processes 2 scripts/training/hpo_train.py --n-trials 20

    # Resume an interrupted run (requires --storage):
    python scripts/training/hpo_train.py --n-trials 20 --storage sqlite:///hpo.db

After finding best params, run train_standalone.py with the reported values.
"""

import gc
import json
import math
import os
from pathlib import Path

import click
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Maps canonical HuggingFace model names to Unsloth-prefixed equivalents.
# Unsloth requires its own model zoo; unlisted models fall back to HF.
UNSLOTH_MODEL_MAP = {
    "Qwen/Qwen3-1.7B":               "unsloth/Qwen3-1.7B",
    "Qwen/Qwen3-4B":                 "unsloth/Qwen3-4B",
    "Qwen/Qwen3-7B":                 "unsloth/Qwen3-7B",
    "Qwen/Qwen2.5-1.5B-Instruct":   "unsloth/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct":     "unsloth/Qwen2.5-3B-Instruct",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_chat_dataset(path: str, tokenizer):
    """Load a JSONL file of {"messages": [...]} records and apply chat template."""
    import datasets as hf_datasets

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    raw = hf_datasets.Dataset.from_list(records)

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return raw.map(format_example, remove_columns=raw.column_names)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(
    model_name: str,
    lora_r: int,
    max_seq_length: int,
    use_unsloth: bool,
):
    """Load base model and apply LoRA adapters.

    Uses Unsloth when use_unsloth=True (single-GPU CUDA only).
    Falls back to standard transformers + PEFT otherwise.
    """
    lora_alpha = lora_r * 2  # convention: alpha = 2 * r (matches train_standalone.py)

    if use_unsloth:
        from unsloth import FastLanguageModel  # noqa: PLC0415

        unsloth_name = UNSLOTH_MODEL_MAP.get(model_name, model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=unsloth_name,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            max_seq_length=max_seq_length,
        )
    else:
        from peft import LoraConfig, TaskType, get_peft_model  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.gradient_checkpointing_enable()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Optuna pruning callback
# ---------------------------------------------------------------------------

class OptunaPruningCallback:
    """TrainerCallback that reports eval_loss to Optuna and prunes bad trials.

    On multi-GPU runs, only rank 0 holds a live trial object. The pruning
    decision is broadcast to all ranks via accelerate so that every process
    stops training at the same time.
    """

    def __init__(self, trial, accelerator):
        # trial is an optuna.Trial on rank 0; None on other ranks.
        self.trial = trial
        self.accelerator = accelerator
        self.pruned = False

    def __call__(self):
        # Return self as a TrainerCallback-compatible object.
        # We implement the callback protocol directly to avoid a hard
        # dependency on transformers at class-definition time.
        from transformers import TrainerCallback  # noqa: PLC0415

        outer = self

        class _Callback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics is None:
                    return
                eval_loss = metrics.get("eval_loss")
                if eval_loss is None:
                    return

                from accelerate.utils import broadcast_object_list  # noqa: PLC0415

                should_prune = [False]
                if outer.accelerator.is_main_process:
                    outer.trial.report(eval_loss, step=int(state.epoch))
                    should_prune = [outer.trial.should_prune()]

                broadcast_object_list(should_prune, from_process=0)

                if should_prune[0]:
                    outer.pruned = True
                    control.should_training_stop = True

        return _Callback()


# ---------------------------------------------------------------------------
# Single-trial training
# ---------------------------------------------------------------------------

def run_trial(
    trial,
    params: dict,
    accelerator,
    model_name: str,
    train_path: str,
    val_path: str,
    output_dir: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
) -> "float | None":
    """Run one training trial. Returns eval_loss, or None if pruned."""
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments  # noqa: PLC0415

    # Unsloth: only on single-GPU CUDA runs (no DDP support)
    use_unsloth = (accelerator.num_processes == 1) and torch.cuda.is_available()
    if use_unsloth:
        try:
            import unsloth  # noqa: F401, PLC0415
        except ImportError:
            use_unsloth = False

    trial_dir = Path(output_dir) / f"trial_{params['trial_number']}"

    model, tokenizer = build_model_and_tokenizer(
        model_name=model_name,
        lora_r=params["lora_r"],
        max_seq_length=max_seq_length,
        use_unsloth=use_unsloth,
    )

    train_dataset = load_chat_dataset(train_path, tokenizer)
    val_dataset = load_chat_dataset(val_path, tokenizer)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Determine optimizer — adamw_8bit requires bitsandbytes
    optim = "adamw_torch"
    try:
        import bitsandbytes  # noqa: F401, PLC0415
        if torch.cuda.is_available():
            optim = "adamw_8bit"
    except ImportError:
        pass

    pruning_holder = OptunaPruningCallback(trial=trial, accelerator=accelerator)
    pruning_cb = pruning_holder()

    training_args = TrainingArguments(
        output_dir=str(trial_dir),
        num_train_epochs=params["num_epochs"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available(),
        optim=optim,
        eval_strategy="epoch",    # pruning callback fires after each epoch
        save_strategy="no",       # no checkpoints during HPO
        logging_steps=20,
        seed=42,
        report_to="none",
        dataloader_num_workers=0, # required for DDP (fork-unsafe CUDA)
        ddp_find_unused_parameters=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[pruning_cb],
    )

    trainer.train()

    eval_loss = None
    if not pruning_holder.pruned:
        results = trainer.evaluate()
        eval_loss = results.get("eval_loss")

    # Release GPU memory before the next trial
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if pruning_holder.pruned:
        return None
    return eval_loss


# ---------------------------------------------------------------------------
# Main HPO loop
# ---------------------------------------------------------------------------

def hpo_main(
    model_name: str,
    train_path: str,
    val_path: str,
    output_dir: str,
    n_trials: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
    study_name: str,
    storage: "str | None",
):
    import optuna  # noqa: PLC0415
    from accelerate import Accelerator  # noqa: PLC0415
    from accelerate.utils import broadcast_object_list  # noqa: PLC0415
    from optuna.trial import TrialState  # noqa: PLC0415

    accelerator = Accelerator()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        accelerator.print(f"Model:  {model_name}")
        accelerator.print(f"Train:  {train_path}")
        accelerator.print(f"Val:    {val_path}")
        accelerator.print(f"Trials: {n_trials}")
        accelerator.print(f"GPUs:   {accelerator.num_processes}")

    # Only rank 0 manages the Optuna study.
    study = None
    if accelerator.is_main_process:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            load_if_exists=True,
        )
        completed_so_far = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        remaining = max(0, n_trials - completed_so_far)
        accelerator.print(
            f"Study '{study_name}': {completed_so_far} completed, {remaining} remaining"
        )

    for trial_idx in range(n_trials):
        # --- Rank 0: sample hyperparameters and put them in a broadcast container ---
        params = [None]
        trial = None  # None on all non-zero ranks

        if accelerator.is_main_process:
            # Skip trials that were already completed when resuming.
            completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
            if len(completed) >= n_trials:
                accelerator.print("All trials already completed. Exiting.")
                break

            trial = study.ask()
            params = [{
                "trial_number": trial.number,
                "lora_r":        trial.suggest_categorical("lora_r", [8, 16, 32, 64]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "weight_decay":  trial.suggest_float("weight_decay", 0.0, 0.1),
                "num_epochs":    trial.suggest_int("num_epochs", 1, 4),
            }]
            accelerator.print(
                f"\n[Trial {trial.number}] params: "
                f"lora_r={params[0]['lora_r']} "
                f"lr={params[0]['learning_rate']:.2e} "
                f"wd={params[0]['weight_decay']:.4f} "
                f"epochs={params[0]['num_epochs']}"
            )

        # --- Broadcast params to all ranks ---
        broadcast_object_list(params, from_process=0)
        current_params = params[0]

        if current_params is None:
            # Rank 0 broke out early (all trials done); workers follow.
            break

        # --- All ranks participate in training ---
        eval_loss = run_trial(
            trial=trial,
            params=current_params,
            accelerator=accelerator,
            model_name=model_name,
            train_path=train_path,
            val_path=val_path,
            output_dir=output_dir,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
        )

        # --- Rank 0: report result to Optuna ---
        if accelerator.is_main_process:
            if eval_loss is None:
                study.tell(trial, state=TrialState.PRUNED)
                accelerator.print(f"[Trial {trial.number}] PRUNED")
            elif math.isnan(eval_loss):
                study.tell(trial, state=TrialState.FAIL)
                accelerator.print(f"[Trial {trial.number}] FAILED (NaN loss)")
            else:
                study.tell(trial, eval_loss)
                accelerator.print(
                    f"[Trial {trial.number}] eval_loss={eval_loss:.4f}"
                )

    # --- Rank 0: save best params ---
    if accelerator.is_main_process and study is not None:
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            accelerator.print("\nNo trials completed successfully.")
            return

        best = study.best_trial
        best_info = {
            "trial_number": best.number,
            "eval_loss": best.value,
            "params": best.params,
        }
        out_path = Path(output_dir) / "best_params.json"
        out_path.write_text(json.dumps(best_info, indent=2))

        accelerator.print("\n" + "=" * 50)
        accelerator.print("HPO complete")
        accelerator.print("=" * 50)
        accelerator.print(f"Best trial: {best.number}")
        accelerator.print(f"Best eval loss: {best.value:.4f}")
        accelerator.print(f"Best params: {best.params}")
        accelerator.print(f"\nSaved to: {out_path}")
        accelerator.print(
            "\nTo train a final model with these params, run:\n"
            f"  python scripts/train_standalone.py \\\n"
            f"    --lora-r {best.params['lora_r']} \\\n"
            f"    --learning-rate {best.params['learning_rate']:.2e} \\\n"
            f"    --epochs {best.params['num_epochs']}"
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@click.command()
@click.option("--model-name",                  default="Qwen/Qwen3-1.7B",                      show_default=True,
              help="HuggingFace model name (Unsloth variant used if available on single-GPU).")
@click.option("--train-path",                  default="data/datasets/processed/train.jsonl",  show_default=True,
              help="Path to training JSONL (chat message format).")
@click.option("--val-path",                    default="data/datasets/processed/val.jsonl",    show_default=True,
              help="Path to validation JSONL.")
@click.option("--output-dir",                  default="./hpo_output",                         show_default=True,
              help="Directory for trial logs and best_params.json.")
@click.option("--n-trials",         type=int,  default=20,                                     show_default=True,
              help="Number of Optuna trials to run.")
@click.option("--batch-size",       type=int,  default=4,                                      show_default=True,
              help="Per-device batch size.")
@click.option("--gradient-accumulation-steps",
                                    type=int,  default=4,                                      show_default=True,
              help="Gradient accumulation steps.")
@click.option("--max-seq-length",   type=int,  default=512,                                    show_default=True,
              help="Maximum tokenized sequence length.")
@click.option("--study-name",                  default="scix-nls-hpo",                         show_default=True,
              help="Optuna study name (used for storage namespacing).")
@click.option("--storage",                     default=None,
              help="Optuna storage URL for persistence/resume (e.g. sqlite:///hpo.db).")
def cli(**kwargs):
    """Hyperparameter optimization for SciX NLS fine-tuning.

    Searches over LoRA rank, learning rate, weight decay, and epoch count
    using Optuna with MedianPruner. Compatible with accelerate launch.
    """
    hpo_main(**kwargs)


if __name__ == "__main__":
    cli()

