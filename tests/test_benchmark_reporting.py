from app.evaluation import (
    DEFAULT_EVALUATION_CASES,
    EvaluationResult,
    summarize_results,
)


def test_all_default_cases_have_rubrics() -> None:
    assert len(DEFAULT_EVALUATION_CASES) == 5

    assert all(
        case.rubric is not None
        for case in DEFAULT_EVALUATION_CASES
    )


def test_summary_reports_reviewer_false_positive() -> None:
    results = [
        EvaluationResult(
            case_id="false-positive",
            question="Question one.",
            approved=True,
            revision_count=0,
            runtime_seconds=1.0,
            final_answer="Incorrect answer.",
            issues=(),
            event_count=4,
            benchmark_passed=False,
            missing_concepts=("required concept",),
        ),
        EvaluationResult(
            case_id="benchmark-pass",
            question="Question two.",
            approved=False,
            revision_count=1,
            runtime_seconds=2.0,
            final_answer="Correct benchmark answer.",
            issues=("Reviewer rejected it.",),
            event_count=6,
            benchmark_passed=True,
        ),
    ]

    summary = summarize_results(results)

    assert summary.benchmark_scored_count == 2
    assert summary.benchmark_passed_count == 1
    assert summary.benchmark_pass_rate == 0.5
    assert summary.reviewer_false_positive_count == 1
