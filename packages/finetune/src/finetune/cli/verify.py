"""Verification commands for NLS Query fine-tuning CLI."""

import json
import subprocess
import sys
from pathlib import Path

import typer

verify_app = typer.Typer(help="Run verification checks", no_args_is_help=True)

# Project root detection
PROJECT_ROOT = Path(__file__).parent.parent.parent


def print_check(name: str, passed: bool, detail: str = "") -> bool:
    """Print a check result."""
    symbol = "✓" if passed else "✗"
    status = f"{symbol} {name}"
    if detail:
        status += f": {detail}"
    print(status)
    return passed


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{title}")
    print("-" * len(title))


@verify_app.command("env")
def verify_env() -> None:
    """Check Modal setup and dependencies."""
    print_section("Verifying environment")

    all_passed = True

    # Check Modal CLI
    try:
        result = subprocess.run(
            ["modal", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        modal_version = result.stdout.strip() if result.returncode == 0 else None
        all_passed &= print_check(
            "Modal CLI", modal_version is not None, modal_version or "not found"
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        all_passed &= print_check("Modal CLI", False, "not installed")

    # Check Modal profile/workspace
    try:
        result = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            workspace = result.stdout.strip()
            all_passed &= print_check("Modal workspace", True, workspace)
        else:
            all_passed &= print_check("Modal workspace", False, "not configured")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        all_passed &= print_check("Modal workspace", False, "modal command failed")

    # Check HuggingFace secret
    try:
        result = subprocess.run(
            ["modal", "secret", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        has_hf = "huggingface" in result.stdout.lower() if result.returncode == 0 else False
        all_passed &= print_check(
            "Secret 'huggingface-secret'", has_hf, "found" if has_hf else "not found"
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        all_passed &= print_check("Secret 'huggingface-secret'", False, "could not check")

    # Check W&B secret (optional)
    try:
        result = subprocess.run(
            ["modal", "secret", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        has_wandb = "wandb" in result.stdout.lower() if result.returncode == 0 else False
        print_check(
            "Secret 'wandb-secret'", has_wandb, "found" if has_wandb else "not found (optional)"
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_check("Secret 'wandb-secret'", False, "could not check (optional)")

    # Summary
    print()
    if all_passed:
        print("All environment checks passed")
    else:
        print("Some environment checks failed")
        sys.exit(1)


@verify_app.command("data")
def verify_data(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show sample entries"),
) -> None:
    """Validate training data format and content."""
    print_section("Verifying training data")

    all_passed = True
    data_dir = PROJECT_ROOT / "data" / "datasets" / "processed"

    for filename in ["train.jsonl", "val.jsonl"]:
        filepath = data_dir / filename
        print(f"\nChecking {filepath.relative_to(PROJECT_ROOT)}...")

        if not filepath.exists():
            all_passed &= print_check(f"{filename} exists", False, "file not found")
            continue

        all_passed &= print_check(f"{filename} exists", True)

        # Parse and validate entries
        entries = []
        errors = []
        try:
            with open(filepath) as f:
                for i, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)

                        # Validate schema
                        if "messages" not in entry:
                            errors.append(f"Line {i}: missing 'messages' field")
                        else:
                            for j, msg in enumerate(entry["messages"]):
                                if "role" not in msg:
                                    errors.append(f"Line {i}, message {j}: missing 'role'")
                                if "content" not in msg:
                                    errors.append(f"Line {i}, message {j}: missing 'content'")
                                elif not msg["content"]:
                                    errors.append(f"Line {i}, message {j}: empty content")
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i}: invalid JSON - {e}")
        except Exception as e:
            all_passed &= print_check("File readable", False, str(e))
            continue

        all_passed &= print_check("Valid JSONL", len(errors) == 0, f"{len(entries)} entries")

        if errors:
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")

        # Role statistics
        if entries:
            roles = {}
            for entry in entries:
                for msg in entry.get("messages", []):
                    role = msg.get("role", "unknown")
                    roles[role] = roles.get(role, 0) + 1

            role_str = ", ".join(f"{r} ({c})" for r, c in sorted(roles.items()))
            all_passed &= print_check("Schema validation", len(errors) == 0, f"roles: {role_str}")

            # Content statistics
            user_lens = []
            asst_lens = []
            for entry in entries:
                for msg in entry.get("messages", []):
                    content = msg.get("content", "")
                    if msg.get("role") == "user":
                        user_lens.append(len(content))
                    elif msg.get("role") == "assistant":
                        asst_lens.append(len(content))

            if user_lens:
                avg_user = sum(user_lens) // len(user_lens)
                print(f"  Avg user message: {avg_user} chars")
            if asst_lens:
                avg_asst = sum(asst_lens) // len(asst_lens)
                print(f"  Avg assistant message: {avg_asst} chars")

        # Show sample entries if verbose
        if verbose and entries:
            print(f"\n  Sample entries from {filename}:")
            for entry in entries[:2]:
                print(f"    {json.dumps(entry, ensure_ascii=False)[:100]}...")

    # Summary
    print()
    if all_passed:
        print("All data checks passed")
    else:
        print("Some data checks failed")
        sys.exit(1)


@verify_app.command("config")
def verify_config(
    show: bool = typer.Option(False, "--show", help="Show training configuration details"),
) -> None:
    """Verify Modal training scripts are valid."""
    print_section("Verifying training configuration")

    all_passed = True
    modal_dir = PROJECT_ROOT / "finetune" / "modal"

    # Check core Modal training scripts
    training_scripts = [
        "train.py",
        "train_unsloth.py",
        "merge.py",
    ]

    for script in training_scripts:
        script_path = modal_dir / script
        if not script_path.exists():
            print_check(f"{script}", False, "not found")
            continue

        # Syntax check
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(script_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            all_passed &= print_check(f"{script}", result.returncode == 0, "valid syntax")
        except Exception as e:
            all_passed &= print_check(f"{script}", False, str(e))

    if show:
        print("\n--- Training Configuration (TRL SFTTrainer) ---")
        print("  Framework: TRL SFTTrainer with PEFT/LoRA")
        print("  Base model: Qwen/Qwen3-1.7B")
        print("  LoRA config: r=16, alpha=32, target=all-linear")
        print("  Training: 3 epochs, batch_size=4, gradient_accum=2")
        print("  Precision: BF16, single H100 GPU")

    # Summary
    print()
    if all_passed:
        print("All config checks passed")
    else:
        print("Some config checks failed")
        sys.exit(1)


@verify_app.command("volumes")
def verify_volumes() -> None:
    """Check Modal volume access."""
    print_section("Verifying Modal volumes")

    all_passed = True
    volumes = ["nls-query-runs", "nls-query-data"]

    for volume_name in volumes:
        try:
            result = subprocess.run(
                ["modal", "volume", "list"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                exists = volume_name in result.stdout
                if exists:
                    all_passed &= print_check(f"Volume '{volume_name}'", True, "accessible")
                else:
                    # Volume doesn't exist yet, but can be created
                    print_check(f"Volume '{volume_name}'", True, "will be created on first use")
            else:
                all_passed &= print_check(
                    f"Volume '{volume_name}'", False, "could not list volumes"
                )
        except subprocess.TimeoutExpired:
            all_passed &= print_check(f"Volume '{volume_name}'", False, "timeout")
        except FileNotFoundError:
            all_passed &= print_check(f"Volume '{volume_name}'", False, "modal CLI not found")

    # Summary
    print()
    if all_passed:
        print("All volume checks passed")
    else:
        print("Some volume checks failed")
        sys.exit(1)


@verify_app.command("all")
def verify_all() -> None:
    """Run all verification checks."""
    print("Running all verification checks...\n")

    try:
        verify_env()
    except SystemExit:
        pass

    try:
        verify_data()
    except SystemExit:
        pass

    try:
        verify_config()
    except SystemExit:
        pass

    try:
        verify_volumes()
    except SystemExit:
        pass

    print("\n" + "=" * 40)
    print("Verification complete")
