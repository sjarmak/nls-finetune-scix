"""Evaluation API routes - reads CLI-generated evaluation results."""

from fastapi import APIRouter, HTTPException

from api.models.evaluation import EvaluationRun, EvaluationRunSummary
from api.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
evaluation_service = EvaluationService()


@router.get("/runs", response_model=list[EvaluationRunSummary])
async def list_evaluation_runs(limit: int = 10) -> list[EvaluationRunSummary]:
    """List past evaluation runs.

    Returns summaries of CLI-generated evaluation runs.
    Run evaluations via: `nls-finetune eval run`
    """
    return evaluation_service.list_runs(limit=limit)


@router.get("/runs/latest", response_model=EvaluationRun)
async def get_latest_evaluation_run() -> EvaluationRun:
    """Get the most recent evaluation run with full results."""
    run = evaluation_service.get_latest_run()
    if not run:
        raise HTTPException(
            status_code=404,
            detail="No evaluation runs found. Run 'nls-finetune eval run' first.",
        )
    return run


@router.get("/runs/{run_id}", response_model=EvaluationRun)
async def get_evaluation_run(run_id: str) -> EvaluationRun:
    """Get a specific evaluation run with full results."""
    run = evaluation_service.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation run '{run_id}' not found.",
        )
    return run
