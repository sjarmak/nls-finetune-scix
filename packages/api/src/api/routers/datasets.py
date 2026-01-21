"""Dataset API routes."""

from fastapi import APIRouter, HTTPException

from api.models.dataset import DatasetExample, DatasetStats
from api.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])
dataset_service = DatasetService()


@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats() -> DatasetStats:
    """Get dataset statistics."""
    return dataset_service.get_stats()


@router.get("/examples", response_model=list[DatasetExample])
async def list_examples(
    split: str = "all",  # "gold", "generated", "all"
    source: str | None = None,
    category: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[DatasetExample]:
    """List dataset examples with optional filtering."""
    return dataset_service.list_examples(
        split=split, source=source, category=category, limit=limit, offset=offset
    )


@router.get("/examples/{example_id}", response_model=DatasetExample)
async def get_example(example_id: str) -> DatasetExample:
    """Get a specific example by ID."""
    example = dataset_service.get_example(example_id)
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")
    return example


@router.post("/examples", response_model=DatasetExample)
async def create_example(example: DatasetExample) -> DatasetExample:
    """Create a new example."""
    return dataset_service.create_example(example)


@router.put("/examples/{example_id}", response_model=DatasetExample)
async def update_example(example_id: str, example: DatasetExample) -> DatasetExample:
    """Update an existing example."""
    updated = dataset_service.update_example(example_id, example)
    if not updated:
        raise HTTPException(status_code=404, detail="Example not found")
    return updated


@router.delete("/examples/{example_id}")
async def delete_example(example_id: str) -> dict:
    """Delete an example."""
    success = dataset_service.delete_example(example_id)
    if not success:
        raise HTTPException(status_code=404, detail="Example not found")
    return {"status": "deleted"}
