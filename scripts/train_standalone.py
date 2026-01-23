#!/usr/bin/env python3
"""Standalone training script for ADS/SciX NL query translator.

This script can run on any machine with a CUDA GPU (Colab, AWS, Lambda, etc.)
No Modal dependency required.

Usage:
    # Install dependencies first:
    pip install torch transformers datasets peft accelerate unsloth trl

    # Run training:
    python scripts/train_standalone.py --output-dir ./output

    # Or with custom settings:
    python scripts/train_standalone.py --output-dir ./output --epochs 3 --batch-size 8

After training, the merged model will be in ./output/merged/ ready for HuggingFace upload.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train ADS/SciX NL query translator")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory for model")
    parser.add_argument("--data-path", type=str, default=None, help="Path to train.jsonl (auto-detected if not set)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merging LoRA into base model")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HuggingFace repo to push to (e.g., adsabs/scix-nls-translator)")
    args = parser.parse_args()

    # Check for CUDA
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a CUDA GPU.")
        print("Options:")
        print("  - Use Google Colab (free T4 GPU)")
        print("  - Use AWS EC2 g5.xlarge (A10G GPU)")
        print("  - Use Lambda Labs or other GPU cloud")
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"✓ CUDA available: {device_name}")

    # Find training data
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Auto-detect from common locations
        candidates = [
            Path("data/datasets/processed/train.jsonl"),
            Path("../data/datasets/processed/train.jsonl"),
            Path("train.jsonl"),
        ]
        data_path = None
        for candidate in candidates:
            if candidate.exists():
                data_path = candidate
                break
        if data_path is None:
            print("ERROR: Training data not found. Specify --data-path or ensure train.jsonl exists.")
            return 1

    print(f"✓ Training data: {data_path}")

    # Load training data
    train_data = []
    with open(data_path) as f:
        for line in f:
            train_data.append(json.loads(line))
    print(f"✓ Loaded {len(train_data)} training examples")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    # Import ML libraries (after CUDA check to fail fast)
    print("Loading Unsloth...")
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
    from datasets import Dataset

    # Load model with Unsloth
    print("Loading base model: Qwen/Qwen3-1.7B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B",
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    print(f"✓ Model loaded, vocab size: {tokenizer.vocab_size}")

    # Apply chat template
    def format_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(format_chat_template, remove_columns=dataset.column_names)
    print(f"✓ Preprocessed {len(dataset)} examples")

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=args.max_seq_length,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA applied: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.2f}%)")

    # Training arguments
    training_args = UnslothTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    # Create trainer
    trainer = UnslothTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    trainer.train()
    print("✓ Training complete!")

    # Save LoRA adapter
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"✓ LoRA adapter saved to {adapter_dir}")

    # Merge LoRA into base model
    if not args.skip_merge:
        print("\nMerging LoRA adapter into base model...")
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(exist_ok=True)

        # Reload in float16 for merging
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print("  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

        print("  Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))

        print("  Merging...")
        merged_model = model.merge_and_unload()

        print(f"  Saving to {merged_dir}...")
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        base_tokenizer.save_pretrained(str(merged_dir))

        print(f"✓ Merged model saved to {merged_dir}")

        # Push to HuggingFace if requested
        if args.push_to_hub:
            print(f"\nPushing to HuggingFace: {args.push_to_hub}...")
            merged_model.push_to_hub(args.push_to_hub, safe_serialization=True)
            base_tokenizer.push_to_hub(args.push_to_hub)
            print(f"✓ Pushed to https://huggingface.co/{args.push_to_hub}")

    print("\n" + "="*50)
    print("DONE!")
    print("="*50)
    print(f"\nModel files are in: {output_dir / 'merged'}")
    print("\nTo upload to HuggingFace manually:")
    print(f"  huggingface-cli upload adsabs/scix-nls-translator {output_dir / 'merged'} .")
    print("\nTo test locally with vLLM:")
    print(f"  vllm serve {output_dir / 'merged'} --max-model-len 512")

    return 0


if __name__ == "__main__":
    exit(main())
