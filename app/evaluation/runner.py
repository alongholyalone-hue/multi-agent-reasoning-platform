from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

from app.core import WorkflowOrchestrator
from app.evaluation.scoring import (
    AnswerRubric,
    score_answer,
)
from app.models import TaskRequest


@dataclass(frozen=True)
class EvaluationCase:
    """One question used to evaluate a workflow."""

    case_id: str
    question: str
    max_revisions: int = 1
    rubric: AnswerRubric | None = None

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
    benchmark_passed: bool | None = None
    missing_concepts: tuple[str, ...] = ()
    matched_forbidden_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics across evaluation results."""

    case_count: int
    approved_count: int
    approval_rate: float
    average_runtime_seconds: float
    total_revisions: int
    benchmark_scored_count: int
    benchmark_passed_count: int
    benchmark_pass_rate: float
    reviewer_false_positive_count: int


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

        benchmark_score = (
            score_answer(
                answer=task_result.final_answer,
                rubric=case.rubric,
            )
            if case.rubric is not None
            else None
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
                benchmark_passed=(
                    benchmark_score.passed
                    if benchmark_score is not None
                    else None
                ),
                missing_concepts=(
                    benchmark_score.missing_concepts
                    if benchmark_score is not None
                    else ()
                ),
                matched_forbidden_claims=(
                    benchmark_score.matched_forbidden_claims
                    if benchmark_score is not None
                    else ()
                ),
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
            benchmark_scored_count=0,
            benchmark_passed_count=0,
            benchmark_pass_rate=0.0,
            reviewer_false_positive_count=0,
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

    scored_results = [
        result
        for result in results
        if result.benchmark_passed is not None
    ]

    benchmark_scored_count = len(scored_results)

    benchmark_passed_count = sum(
        result.benchmark_passed is True
        for result in scored_results
    )

    benchmark_pass_rate = (
        benchmark_passed_count / benchmark_scored_count
        if benchmark_scored_count
        else 0.0
    )

    reviewer_false_positive_count = sum(
        result.approved
        and result.benchmark_passed is False
        for result in scored_results
    )

    return EvaluationSummary(
        case_count=case_count,
        approved_count=approved_count,
        approval_rate=approved_count / case_count,
        average_runtime_seconds=(
            total_runtime / case_count
        ),
        total_revisions=total_revisions,
        benchmark_scored_count=benchmark_scored_count,
        benchmark_passed_count=benchmark_passed_count,
        benchmark_pass_rate=benchmark_pass_rate,
        reviewer_false_positive_count=(
            reviewer_false_positive_count
        ),
    )
