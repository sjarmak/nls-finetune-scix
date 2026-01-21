"""Modal dry-run functions for testing model loading and training."""

import modal

# Base image with ML dependencies
base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.45.0",
    "accelerate>=0.30.0",
    "huggingface_hub>=0.24.0",
)

# Training image with additional dependencies
training_image = base_image.pip_install(
    "datasets>=2.20.0",
    "peft>=0.12.0",
    "trl>=0.9.0",
)

# Model dry-run app
app = modal.App("nls-finetune-dry-run")


@app.function(
    image=base_image,
    gpu="H100",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_model_loading() -> dict:
    """Test loading Qwen3-1.7B on a single H100."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # GPU info
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"✓ GPU: {device_name}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    print(f"✓ Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Memory usage
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"✓ Model loaded in bf16 ({mem_gb:.1f} GB VRAM)")

    # Test forward pass
    print("Testing forward pass...")
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    print("✓ Forward pass: success")

    return {
        "gpu": device_name,
        "vocab_size": tokenizer.vocab_size,
        "memory_gb": mem_gb,
        "forward_pass": "success",
    }


# Training dry-run app
volume = modal.Volume.from_name("nls-query-data", create_if_missing=True)
app_train = modal.App("nls-finetune-dry-run-train")


@app_train.function(
    image=training_image,
    gpu="H100:4",
    timeout=1800,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_training(num_steps: int) -> dict:
    """Test training pipeline with a few steps on 4x H100."""
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

    # Load data
    data_path = Path("/data/train.jsonl")
    if not data_path.exists():
        raise FileNotFoundError("Training data not found. Run 'nls-finetune upload-data' first.")

    train_data = []
    with open(data_path) as f:
        for line in f:
            train_data.append(json.loads(line))

    # Use small subset for dry run
    train_data = train_data[:50]
    dataset = Dataset.from_list(train_data)
    print(f"✓ Dataset loaded: {len(train_data)} examples (subset for dry-run)")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    print("✓ LoRA applied (r=16, alpha=32)")

    # Training config
    training_args = SFTConfig(
        output_dir="/tmp/dry-run-output",
        max_steps=num_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    # Trainer
    def formatting_func(example):
        messages = example["messages"]
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|{role}|>\n{content}\n"
        return text

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    print(f"\nTraining {num_steps} steps...")

    # Use standard training loop
    train_result = trainer.train()

    # Extract losses from log history
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]

    if losses:
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_decreased = final_loss < initial_loss if len(losses) > 1 else True
        for i, loss in enumerate(losses):
            print(f"  Step {i + 1}/{len(losses)}: loss={loss:.3f}")
    else:
        initial_loss = train_result.training_loss
        final_loss = train_result.training_loss
        loss_decreased = True
        print(f"  Final loss: {final_loss:.3f}")

    return {
        "num_gpus": num_gpus,
        "gpu_name": device_name,
        "steps": num_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": loss_decreased,
    }
