"""Evaluation service - reads CLI-generated evaluation results."""

import json
from pathlib import Path

from api.models.evaluation import (
    EvaluationRun,
    EvaluationRunSummary,
)

EVAL_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "datasets" / "evaluations"
)

# Success criteria thresholds
SYNTAX_VALID_TARGET = 95  # ≥95%
SEMANTIC_MATCH_TARGET = 70  # ≥70%
LATENCY_TARGET_MS = 100  # ≤100ms


class EvaluationService:
    """Service for reading CLI-generated evaluation results.

    Evaluations are run via CLI: `nls-finetune eval run`
    This service only reads the JSON results for display in the web UI.
    """

    def list_runs(self, limit: int = 10) -> list[EvaluationRunSummary]:
        """List past evaluation runs as summaries."""
        summaries = []
        if not EVAL_DIR.exists():
            return summaries

        # Only read eval-*.json files (not baseline files)
        eval_files = sorted(EVAL_DIR.glob("eval-*.json"), reverse=True)[:limit]

        for file in eval_files:
            try:
                data = json.loads(file.read_text())
                run = EvaluationRun(**data)
                summaries.append(self._to_summary(run))
            except (json.JSONDecodeError, ValueError):
                continue

        return summaries

    def get_run(self, run_id: str) -> EvaluationRun | None:
        """Get a specific evaluation run with full results."""
        # Handle "latest" as special case
        if run_id == "latest":
            return self.get_latest_run()

        file = EVAL_DIR / f"{run_id}.json"
        if not file.exists():
            return None

        try:
            data = json.loads(file.read_text())
            return EvaluationRun(**data)
        except (json.JSONDecodeError, ValueError):
            return None

    def get_latest_run(self) -> EvaluationRun | None:
        """Get the most recent evaluation run."""
        if not EVAL_DIR.exists():
            return None

        eval_files = sorted(EVAL_DIR.glob("eval-*.json"), reverse=True)
        if not eval_files:
            return None

        try:
            data = json.loads(eval_files[0].read_text())
            return EvaluationRun(**data)
        except (json.JSONDecodeError, ValueError):
            return None

    def _to_summary(self, run: EvaluationRun) -> EvaluationRunSummary:
        """Convert full run to summary for list view."""
        total = run.summary.total
        ft = run.summary.fine_tuned

        syntax_pct = (ft.syntax_valid / total * 100) if total > 0 else 0
        semantic_pct = (ft.semantic_match / total * 100) if total > 0 else 0

        # Determine status based on targets
        if (
            syntax_pct >= SYNTAX_VALID_TARGET
            and semantic_pct >= SEMANTIC_MATCH_TARGET
            and ft.avg_latency_ms <= LATENCY_TARGET_MS
        ):
            status = "pass"
        elif syntax_pct >= SYNTAX_VALID_TARGET and semantic_pct >= SEMANTIC_MATCH_TARGET:
            status = "needs_improvement"  # Latency issue only
        else:
            status = "fail"

        return EvaluationRunSummary(
            id=run.id,
            timestamp=run.timestamp,
            model=run.models.get("fine_tuned", "unknown"),
            total=total,
            syntax_valid_pct=round(syntax_pct, 1),
            semantic_match_pct=round(semantic_pct, 1),
            avg_latency_ms=round(ft.avg_latency_ms, 1),
            status=status,
        )
