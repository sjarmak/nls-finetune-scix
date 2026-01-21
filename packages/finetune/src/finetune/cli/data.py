"""Data management commands for NLS Query fine-tuning CLI."""

import json
from pathlib import Path

import typer

data_app = typer.Typer(help="Data management commands")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent


def upload_data_impl(dry_run: bool = False) -> None:
    """Upload training data to Modal volume."""
    data_dir = PROJECT_ROOT / "data" / "datasets" / "processed"
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    # Count entries
    train_count = 0
    val_count = 0

    if train_path.exists():
        with open(train_path) as f:
            train_count = sum(1 for _ in f)

    if val_path.exists():
        with open(val_path) as f:
            val_count = sum(1 for _ in f)

    train_size = train_path.stat().st_size if train_path.exists() else 0
    val_size = val_path.stat().st_size if val_path.exists() else 0

    if dry_run:
        print("Upload preview (dry-run):")
        print(f"  train.jsonl: {train_count} entries, {train_size // 1024} KB")
        print(f"  val.jsonl: {val_count} entries, {val_size // 1024} KB")
        print("\nRun without --dry-run to upload.")
        return

    print("Uploading training data to Modal...")

    # Import Modal and run upload
    try:
        import modal

        volume = modal.Volume.from_name("scix-finetune-data", create_if_missing=True)

        # Read data
        train_data = []
        val_data = []

        if train_path.exists():
            with open(train_path) as f:
                train_data = [json.loads(line) for line in f]

        if val_path.exists():
            with open(val_path) as f:
                val_data = [json.loads(line) for line in f]

        # Create a Modal function to write data
        app = modal.App("nls-finetune-upload")

        @app.function(volumes={"/data": volume}, timeout=300, serialized=True)
        def write_data(train: list, val: list) -> dict:
            import json
            from pathlib import Path

            data_dir = Path("/data")
            data_dir.mkdir(exist_ok=True)

            # Write train data
            with open(data_dir / "train.jsonl", "w") as f:
                for item in train:
                    f.write(json.dumps(item) + "\n")

            # Write val data
            with open(data_dir / "val.jsonl", "w") as f:
                for item in val:
                    f.write(json.dumps(item) + "\n")

            return {"train": len(train), "val": len(val)}

        # Run the upload
        with app.run():
            result = write_data.remote(train_data, val_data)

        print(f"✓ Uploaded {result['train']} train, {result['val']} val examples")
        print("Upload complete")

    except ImportError:
        print("✗ Modal not installed. Run: pip install modal")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        raise typer.Exit(1)
