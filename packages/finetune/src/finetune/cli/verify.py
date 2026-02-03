"""Verification commands for NLS Query fine-tuning CLI."""

import json
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


@verify_app.command("all")
def verify_all() -> None:
    """Run all verification checks."""
    print("Running all verification checks...\n")

    try:
        verify_data()
    except SystemExit:
        pass

    print("\n" + "=" * 40)
    print("Verification complete")
