"""Evaluation models matching CLI output format."""

from datetime import datetime

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    """Metrics for a model (fine-tuned or baseline)."""

    syntax_valid: int
    semantic_match: int
    avg_latency_ms: float
    total: int | None = None  # Only present in baseline


class EvalSummary(BaseModel):
    """Summary of evaluation results."""

    total: int
    fine_tuned: ModelMetrics
    baseline: ModelMetrics | None = None


class ModelOutput(BaseModel):
    """Output from a single model for one example."""

    output: str
    syntax_valid: bool
    hallucinations: list[str]
    semantic_match: bool
    latency_ms: float


class EvaluationResult(BaseModel):
    """Result of evaluating a single example."""

    id: str
    input: str
    expected: str
    fine_tuned: ModelOutput
    baseline: ModelOutput | None = None
    verdict: str  # fine_tuned_better, baseline_better, tie, both_wrong


class EvaluationRun(BaseModel):
    """Complete evaluation run from CLI."""

    id: str
    timestamp: datetime | str
    models: dict[str, str]  # {"fine_tuned": "...", "baseline": "..."}
    summary: EvalSummary
    results: list[EvaluationResult]


class EvaluationRunSummary(BaseModel):
    """Summary of an evaluation run for list view."""

    id: str
    timestamp: datetime | str
    model: str
    total: int
    syntax_valid_pct: float
    semantic_match_pct: float
    avg_latency_ms: float
    status: str  # "pass", "fail", "needs_improvement"
