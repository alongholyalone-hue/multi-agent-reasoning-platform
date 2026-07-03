import pytest

from app.core import WorkflowOrchestrator
from app.evaluation import (
    EvaluationCase,
    EvaluationResult,
    evaluate_cases,
    summarize_results,
)


def test_evaluate_cases_records_workflow_result() -> None:
    times = iter([10.0, 12.5])

    results = evaluate_cases(
        orchestrator=WorkflowOrchestrator(),
        cases=[
            EvaluationCase(
                case_id="gravity",
                question=(
                    "Explain gravitational acceleration."
                ),
                max_revisions=1,
            )
        ],
        clock=lambda: next(times),
    )

    assert len(results) == 1

    result = results[0]

    assert result.case_id == "gravity"
    assert result.approved is True
    assert result.revision_count == 0
    assert result.runtime_seconds == 2.5
    assert result.event_count == 4
    assert result.final_answer


def test_summarize_results_calculates_metrics() -> None:
    results = [
        EvaluationResult(
            case_id="first",
            question="First question.",
            approved=True,
            revision_count=0,
            runtime_seconds=2.0,
            final_answer="First answer.",
            issues=(),
            event_count=4,
        ),
        EvaluationResult(
            case_id="second",
            question="Second question.",
            approved=False,
            revision_count=1,
            runtime_seconds=4.0,
            final_answer="Safe refusal.",
            issues=("Incorrect answer.",),
            event_count=6,
        ),
    ]

    summary = summarize_results(results)

    assert summary.case_count == 2
    assert summary.approved_count == 1
    assert summary.approval_rate == 0.5
    assert summary.average_runtime_seconds == 3.0
    assert summary.total_revisions == 1


def test_summarize_empty_results() -> None:
    summary = summarize_results([])

    assert summary.case_count == 0
    assert summary.approved_count == 0
    assert summary.approval_rate == 0.0
    assert summary.average_runtime_seconds == 0.0
    assert summary.total_revisions == 0


def test_evaluation_case_rejects_blank_question() -> None:
    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        EvaluationCase(
            case_id="invalid",
            question="   ",
        )
