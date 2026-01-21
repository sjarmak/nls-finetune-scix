"""Dataset models."""

from datetime import datetime

from pydantic import BaseModel


class DatasetExample(BaseModel):
    """A single training/validation example."""

    id: str
    user_query: str
    date: str
    expected_output: dict  # {"query": "..."}
    created_at: datetime | None = None
    source: str = "manual"  # "manual", "generated", "gold"
    category: str | None = None  # e.g., "repo_scoped", "commit_search", etc.


class DatasetStats(BaseModel):
    """Dataset statistics."""

    total_examples: int
    train_examples: int
    val_examples: int
    by_type: dict[str, int]  # {"code_search": 100, "commit": 50, ...}
    by_source: dict[str, int]  # {"manual": 10, "generated": 490, ...}
