"""Dataset management service."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from api.models.dataset import DatasetExample, DatasetStats

# Path to data directory (relative to repo root)
DATA_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "datasets"


class DatasetService:
    """Service for managing training datasets."""

    def __init__(self):
        self.raw_dir = DATA_DIR / "raw"
        self.processed_dir = DATA_DIR / "processed"

    def get_stats(self) -> DatasetStats:
        """Get dataset statistics."""
        # Load gold examples
        gold_examples = self._load_gold_examples()

        # Load generated examples from valid_pairs.json
        generated_examples = self._load_generated_examples()

        # Count by category from generated examples
        by_category: dict[str, int] = {}
        for ex in generated_examples:
            cat = ex.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1

        # Count train/val split
        train_count = len(self._load_jsonl(self.processed_dir / "train.jsonl"))
        val_count = len(self._load_jsonl(self.processed_dir / "val.jsonl"))

        by_source = {
            "gold": len(gold_examples),
            "generated": len(generated_examples),
        }

        return DatasetStats(
            total_examples=len(gold_examples) + len(generated_examples),
            train_examples=train_count,
            val_examples=val_count,
            by_type=by_category,  # Now shows categories
            by_source=by_source,
        )

    def list_examples(
        self,
        split: str = "all",
        source: str | None = None,
        category: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DatasetExample]:
        """List examples with filtering."""
        examples: list[DatasetExample] = []

        # Load based on split/source
        if split in ("all", "gold") and source in (None, "gold"):
            examples.extend(self._load_gold_examples_as_dataset())

        if split in ("all", "generated") and source in (None, "generated"):
            examples.extend(self._load_generated_examples_as_dataset())

        # Filter by category
        if category:
            examples = [e for e in examples if getattr(e, "category", None) == category]

        return examples[offset : offset + limit]

    def get_example(self, example_id: str) -> DatasetExample | None:
        """Get a specific example."""
        examples = self._load_all_examples()
        for ex in examples:
            if ex.id == example_id:
                return ex
        return None

    def create_example(self, example: DatasetExample) -> DatasetExample:
        """Create a new example."""
        if not example.id:
            example.id = str(uuid.uuid4())[:8]
        example.created_at = datetime.now()

        # Save to raw/manual.json
        manual_file = self.raw_dir / "manual.json"
        existing = []
        if manual_file.exists():
            existing = json.loads(manual_file.read_text())

        existing.append(example.model_dump())
        manual_file.write_text(json.dumps(existing, indent=2, default=str))

        return example

    def update_example(self, example_id: str, example: DatasetExample) -> DatasetExample | None:
        """Update an existing example."""
        # Implementation depends on storage format
        # For now, just return the example
        example.id = example_id
        return example

    def delete_example(self, example_id: str) -> bool:
        """Delete an example."""
        # Implementation depends on storage format
        return True

    def _load_gold_examples(self) -> list[dict]:
        """Load gold examples as raw dicts."""
        gold_file = self.raw_dir / "gold_examples.json"
        if gold_file.exists():
            return json.loads(gold_file.read_text())
        return []

    def _load_generated_examples(self) -> list[dict]:
        """Load generated examples from valid_pairs.json as raw dicts."""
        valid_pairs = self.processed_dir / "valid_pairs.json"
        if valid_pairs.exists():
            return json.loads(valid_pairs.read_text())
        return []

    def _load_gold_examples_as_dataset(self) -> list[DatasetExample]:
        """Load gold examples as DatasetExample objects."""
        examples = []
        for i, item in enumerate(self._load_gold_examples()):
            examples.append(
                DatasetExample(
                    id=f"gold-{i}",
                    user_query=item["user_query"],
                    date=item["date"],
                    expected_output=item["expected_output"],
                    source="gold",
                )
            )
        return examples

    def _load_generated_examples_as_dataset(self) -> list[DatasetExample]:
        """Load generated examples from valid_pairs.json as DatasetExample objects."""
        examples = []
        for i, item in enumerate(self._load_generated_examples()):
            examples.append(
                DatasetExample(
                    id=f"gen-{i}",
                    user_query=item["natural_language"],
                    date="2025-12-15",
                    expected_output={"query": item["sourcegraph_query"]},
                    source="generated",
                    category=item.get("category", "unknown"),
                )
            )
        return examples

    def _load_all_examples(self) -> list[DatasetExample]:
        """Load all examples (gold + generated)."""
        return self._load_gold_examples_as_dataset() + self._load_generated_examples_as_dataset()

    def _load_jsonl(self, path: Path) -> list[dict]:
        """Load JSONL file."""
        if not path.exists():
            return []
        lines = path.read_text().strip().split("\n")
        return [json.loads(line) for line in lines if line]
