import pytest

from app.agents import FinalizerAgent
from app.models import (
    DraftAnswer,
    FinalAnswer,
    ReviewResult,
)


def create_draft(
    revision_number: int = 0,
) -> DraftAnswer:
    return DraftAnswer(
        content=(
            "Draft answer for: Explain orbital velocity.\n\n"
            "Orbital velocity depends on gravitational attraction "
            "and orbital radius. A larger orbital radius generally "
            "corresponds to a lower orbital velocity."
        ),
        reasoning_steps=[
            "Identify the orbital relationship.",
            "Explain how radius affects velocity.",
            "State the conclusion.",
        ],
        revision_number=revision_number,
    )


def test_finalizer_returns_approved_final_answer() -> None:
    finalizer = FinalizerAgent()
    draft = create_draft()

    review = ReviewResult(
        approved=True,
        issues=[],
        revision_instructions=[],
    )

    answer = finalizer.run(
        question="Explain orbital velocity.",
        draft=draft,
        review=review,
    )

    assert isinstance(answer, FinalAnswer)
    assert answer.approved is True
    assert answer.unresolved_issues == []
    assert answer.content.startswith(
    "Final answer for: Explain orbital velocity."
    )
    assert "Draft answer for:" not in answer.content


def test_finalizer_preserves_unresolved_issues() -> None:
    finalizer = FinalizerAgent()
    draft = create_draft()

    review = ReviewResult(
        approved=False,
        issues=[
            "The answer needs a governing equation.",
        ],
        revision_instructions=[
            "Include the governing equation.",
        ],
    )

    answer = finalizer.run(
        question="Explain orbital velocity.",
        draft=draft,
        review=review,
    )

    assert answer.approved is False
    assert answer.unresolved_issues == [
        "The answer needs a governing equation.",
    ]
    assert answer.content.startswith(
        "Unable to provide a reliable answer for: "
        "Explain orbital velocity."
    )


def test_finalizer_preserves_revision_number() -> None:
    finalizer = FinalizerAgent()
    draft = create_draft(revision_number=2)

    review = ReviewResult(
        approved=True,
    )

    answer = finalizer.run(
        question="Explain orbital velocity.",
        draft=draft,
        review=review,
    )

    assert answer.revision_number == 2


def test_finalizer_normalizes_question_whitespace() -> None:
    finalizer = FinalizerAgent()
    draft = create_draft()

    review = ReviewResult(
        approved=True,
    )

    answer = finalizer.run(
        question="  Explain   orbital   velocity.  ",
        draft=draft,
        review=review,
    )

    assert answer.content.startswith(
        "Final answer for: Explain orbital velocity."
    )


def test_finalizer_rejects_blank_question() -> None:
    finalizer = FinalizerAgent()
    draft = create_draft()

    review = ReviewResult(
        approved=True,
    )

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        finalizer.run(
            question="   ",
            draft=draft,
            review=review,
        )