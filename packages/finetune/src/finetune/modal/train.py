"""Modal training for SciX/ADS fine-tuning with TRL."""

import modal

# Custom training image with latest packages (no DeepSpeed for single GPU)
training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.45.0",
    "accelerate>=0.30.0",
    "datasets>=2.20.0",
    "peft>=0.12.0",
    "trl>=0.9.0",
    "huggingface_hub>=0.24.0",
    "sentencepiece>=0.2.0",
    "scipy>=1.12.0",
    "einops>=0.8.0",
)

# Volume for training data and runs
data_volume = modal.Volume.from_name("scix-finetune-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("scix-finetune-runs", create_if_missing=True)

app = modal.App("scix-finetune-train")


@app.function(
    image=training_image,
    gpu="H100",  # Single H100 (80GB) is plenty for 1.7B model with LoRA
    timeout=3600,  # 1 hour max
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run_training(run_name: str, use_wandb: bool = False) -> dict:
    """Run full training with TRL SFTTrainer."""
    import json
    from pathlib import Path

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # GPU info
    num_gpus = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"✓ GPUs initialized: {num_gpus}x {device_name}")

    # Verify data exists
    data_path = Path("/data/train.jsonl")
    if not data_path.exists():
        raise FileNotFoundError("Training data not found. Run 'scix-finetune upload-data' first.")

    # Load data
    train_data = []
    with open(data_path) as f:
        for line in f:
            train_data.append(json.loads(line))

    dataset = Dataset.from_list(train_data)
    print(f"✓ Training data: {len(train_data)} examples")

    # Create output directory
    output_dir = Path(f"/runs/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded (vocab_size: {tokenizer.vocab_size})")

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"✓ LoRA applied: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)"
    )

    # Training config
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="none" if not use_wandb else "wandb",
    )

    # Formatter for chat messages
    def formatting_func(example):
        messages = example["messages"]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    train_result = trainer.train()

    # Save final model
    trainer.save_model()
    print(f"\n✓ Model saved to {output_dir}")

    # Extract metrics
    metrics = train_result.metrics
    final_loss = metrics.get("train_loss", 0)

    # Find checkpoints
    checkpoints = list(output_dir.glob("checkpoint-*"))
    latest_checkpoint = (
        max(checkpoints, key=lambda p: int(p.name.split("-")[1])) if checkpoints else None
    )

    # Commit volumes to persist data
    data_volume.commit()
    runs_volume.commit()

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)

    return {
        "run_name": run_name,
        "num_gpus": num_gpus,
        "gpu_name": device_name,
        "train_examples": len(train_data),
        "output_dir": str(output_dir),
        "checkpoint": str(latest_checkpoint) if latest_checkpoint else str(output_dir),
        "final_loss": final_loss,
    }
