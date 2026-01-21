"""Modal training using Unsloth for faster fine-tuning with fixed pad token."""

import modal

# Unsloth training image - exact versions from official Modal example
# https://modal.com/docs/examples/unsloth_finetune
unsloth_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
    )
    .env({"HF_HOME": "/model_cache"})
)

# Volumes for data and runs
data_volume = modal.Volume.from_name("nls-query-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("nls-query-runs", create_if_missing=True)

app = modal.App("nls-finetune-train-unsloth")


@app.function(
    image=unsloth_image,
    gpu="H100",
    timeout=3600,  # 1 hour max
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run_training_unsloth(run_name: str, use_wandb: bool = False) -> dict:
    """Run training with Unsloth for 2x faster fine-tuning.

    Benefits of Unsloth:
    - 2x faster training with 70% less VRAM
    - Fixed pad_token handling (prevents infinite generation)
    - Better inference compatibility with vLLM
    """
    import json
    from pathlib import Path

    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

    # GPU info
    num_gpus = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"✓ GPUs: {num_gpus}x {device_name}")

    # Verify data exists
    data_path = Path("/data/train.jsonl")
    if not data_path.exists():
        raise FileNotFoundError("Training data not found. Run 'nls-finetune upload-data' first.")

    # Load training data
    train_data = []
    with open(data_path) as f:
        for line in f:
            train_data.append(json.loads(line))
    print(f"✓ Training data: {len(train_data)} examples")

    # Create output directory
    output_dir = Path(f"/runs/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    # Configuration
    max_seq_length = 512  # Short sequences for Sourcegraph queries

    # Load model with Unsloth - use their optimized Qwen3-1.7B
    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B",
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,  # 4-bit for faster training, merge to 16-bit later
    )
    print("✓ Model loaded: unsloth/Qwen3-1.7B (4-bit)")
    print(f"✓ Tokenizer vocab: {tokenizer.vocab_size}, pad_token: {tokenizer.pad_token}")
    print(f"✓ Tokenizer eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")

    # Preprocess data - apply chat template to each example
    def format_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(format_chat_template, remove_columns=dataset.column_names)
    print(f"✓ Preprocessed dataset: {len(dataset)} examples")

    # Apply LoRA with Unsloth optimizations
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,  # 0 is optimized for Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=42,
        max_seq_length=max_seq_length,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")

    # Training config using UnslothTrainer to avoid TRL patching issues
    training_args = UnslothTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Larger batch with 4-bit
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
        optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
        max_seq_length=max_seq_length,
        dataset_text_field="text",  # Use preprocessed text field
        report_to="none" if not use_wandb else "wandb",
        seed=42,
    )

    # Create trainer using UnslothTrainer
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting Unsloth training...")
    print("=" * 50 + "\n")

    train_result = trainer.train()

    # Save LoRA adapter
    trainer.save_model()
    print(f"\n✓ LoRA adapter saved to {output_dir}")

    # Extract metrics
    metrics = train_result.metrics
    final_loss = metrics.get("train_loss", 0)

    # Commit volumes
    data_volume.commit()
    runs_volume.commit()

    print("\n" + "=" * 50)
    print("Unsloth training complete!")
    print("=" * 50)

    return {
        "run_name": run_name,
        "num_gpus": num_gpus,
        "gpu_name": device_name,
        "train_examples": len(train_data),
        "output_dir": str(output_dir),
        "final_loss": final_loss,
        "framework": "unsloth",
    }


@app.function(
    image=unsloth_image,
    gpu="H100",
    timeout=1800,  # 30 min for merge
    volumes={
        "/runs": runs_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def merge_unsloth_model(run_name: str) -> dict:
    """Merge LoRA adapter into full model for deployment.

    Saves in 16-bit format for optimal vLLM inference.
    """
    from pathlib import Path

    import torch
    from unsloth import FastLanguageModel

    adapter_dir = Path(f"/runs/{run_name}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_name}")

    merged_dir = adapter_dir / "merged"
    merged_dir.mkdir(exist_ok=True)

    print(f"Loading adapter from {adapter_dir}...")

    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Merge and save in 16-bit for vLLM
    print("Merging LoRA weights into base model...")
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    # Commit volume
    runs_volume.commit()

    print(f"\n✓ Merged model saved to {merged_dir}")

    return {
        "run_name": run_name,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
    }
