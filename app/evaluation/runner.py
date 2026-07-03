from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

from app.core import WorkflowOrchestrator
from app.models import TaskRequest


@dataclass(frozen=True)
class EvaluationCase:
    """One question used to evaluate a workflow."""

    case_id: str
    question: str
    max_revisions: int = 1

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty")

        if not self.question.strip():
            raise ValueError("question cannot be empty")

        if not 0 <= self.max_revisions <= 3:
            raise ValueError(
                "max_revisions must be between 0 and 3"
            )


@dataclass(frozen=True)
class EvaluationResult:
    """Recorded outcome for one evaluation case."""

    case_id: str
    question: str
    approved: bool
    revision_count: int
    runtime_seconds: float
    final_answer: str
    issues: tuple[str, ...]
    event_count: int


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics across evaluation results."""

    case_count: int
    approved_count: int
    approval_rate: float
    average_runtime_seconds: float
    total_revisions: int


def evaluate_cases(
    *,
    orchestrator: WorkflowOrchestrator,
    cases: Sequence[EvaluationCase],
    clock: Callable[[], float] = perf_counter,
) -> list[EvaluationResult]:
    """Run every evaluation case through one orchestrator."""

    results: list[EvaluationResult] = []

    for case in cases:
        start_time = clock()

        task_result = orchestrator.run(
            TaskRequest(
                question=case.question,
                max_revisions=case.max_revisions,
            )
        )

        elapsed_seconds = max(
            0.0,
            clock() - start_time,
        )

        results.append(
            EvaluationResult(
                case_id=case.case_id,
                question=case.question,
                approved=task_result.review.approved,
                revision_count=task_result.revision_count,
                runtime_seconds=elapsed_seconds,
                final_answer=task_result.final_answer,
                issues=tuple(task_result.review.issues),
                event_count=len(task_result.events),
            )
        )

    return results


def summarize_results(
    results: Sequence[EvaluationResult],
) -> EvaluationSummary:
    """Calculate aggregate evaluation metrics."""

    case_count = len(results)

    if case_count == 0:
        return EvaluationSummary(
            case_count=0,
            approved_count=0,
            approval_rate=0.0,
            average_runtime_seconds=0.0,
            total_revisions=0,
        )

    approved_count = sum(
        result.approved
        for result in results
    )

    total_runtime = sum(
        result.runtime_seconds
        for result in results
    )

    total_revisions = sum(
        result.revision_count
        for result in results
    )

    return EvaluationSummary(
        case_count=case_count,
        approved_count=approved_count,
        approval_rate=approved_count / case_count,
        average_runtime_seconds=(
            total_runtime / case_count
        ),
        total_revisions=total_revisions,
    )
