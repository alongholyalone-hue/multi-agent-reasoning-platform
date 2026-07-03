import pytest

from app.agents import ReviewerAgent
from app.models import DraftAnswer, ReviewResult


def create_complete_draft() -> DraftAnswer:
    return DraftAnswer(
        content=(
            "Orbital velocity is determined by the balance between "
            "gravity and circular motion. For a circular orbit, the "
            "relationship v = sqrt(GM / r) shows that increasing the "
            "orbital radius increases the denominator and lowers the "
            "required velocity. Therefore, an object farther from the "
            "same central mass moves more slowly in its circular orbit."
        ),
        reasoning_steps=[
            "Identify the governing orbital equation.",
            "Explain how orbital radius affects velocity.",
            "State the final conclusion clearly.",
        ],
        requested_tools=[],
        revision_number=0,
    )


def test_reviewer_approves_complete_draft() -> None:
    reviewer = ReviewerAgent()
    draft = create_complete_draft()

    review = reviewer.run(
        question="Explain orbital velocity.",
        draft=draft,
    )

    assert isinstance(review, ReviewResult)
    assert review.approved is True
    assert review.issues == []
    assert review.revision_instructions == []


def test_reviewer_rejects_brief_draft() -> None:
    reviewer = ReviewerAgent()

    draft = DraftAnswer(
        content="A very brief answer.",
        reasoning_steps=[
            "Identify the concept.",
            "Explain the relationship.",
            "Provide the conclusion.",
        ],
    )

    review = reviewer.run(
        question="Explain orbital velocity.",
        draft=draft,
    )

    assert review.approved is False
    assert "The draft answer is too brief." in review.issues
    assert (
        "Expand the answer to at least three complete "
        "sentences with a clear explanation."
        in review.revision_instructions
    )


def test_reviewer_rejects_too_few_reasoning_steps() -> None:
    reviewer = ReviewerAgent()

    draft = DraftAnswer(
        content=(
            "This draft contains enough words to avoid the short-answer "
            "check, but it does not contain enough structured reasoning "
            "steps to demonstrate a complete solution process."
        ),
        reasoning_steps=[
            "Provide a conclusion."
        ],
    )

    review = reviewer.run(
        question="Explain orbital velocity.",
        draft=draft,
    )

    assert review.approved is False
    assert (
        "The draft contains too few reasoning steps."
        in review.issues
    )


def test_reviewer_detects_placeholder_text() -> None:
    reviewer = ReviewerAgent()

    draft = DraftAnswer(
        content=(
            "This response explains the main scientific concept and "
            "provides a structured discussion of the relevant factors. "
            "TODO: insert answer before submitting the final response."
        ),
        reasoning_steps=[
            "Identify the relevant concept.",
            "Develop the explanation.",
            "State the conclusion.",
        ],
    )

    review = reviewer.run(
        question="Explain orbital velocity.",
        draft=draft,
    )

    assert review.approved is False
    assert (
        "The draft contains unfinished placeholder text."
        in review.issues
    )


def test_reviewer_detects_missing_requested_tool() -> None:
    reviewer = ReviewerAgent()

    draft = DraftAnswer(
        content=(
            "This response describes the relevant orbital principles "
            "and explains how the variables relate to each other before "
            "presenting a final conclusion."
        ),
        reasoning_steps=[
            "Identify the equation.",
            "Substitute the values.",
            "Explain the result.",
        ],
        requested_tools=["calculator"],
    )

    review = reviewer.run(
        question="Calculate orbital velocity.",
        draft=draft,
    )

    assert review.approved is False
    assert (
        "The draft does not acknowledge all requested tools."
        in review.issues
    )
    assert (
        "Explain how the following tools should be used: calculator."
        in review.revision_instructions
    )


def test_reviewer_accepts_acknowledged_requested_tool() -> None:
    reviewer = ReviewerAgent()

    draft = DraftAnswer(
        content=(
            "The solution first identifies the governing orbital "
            "equation and the values supplied by the problem. It uses "
            "a calculator to evaluate the numerical expression and "
            "checks that the resulting units represent velocity. "
            "Finally, it explains what the calculated value means in "
            "the physical context of the orbit."
        ),
        reasoning_steps=[
            "Identify the equation.",
            "Use the calculator.",
            "Interpret the result.",
        ],
        requested_tools=["calculator"],
    )

    review = reviewer.run(
        question="Calculate orbital velocity.",
        draft=draft,
    )

    assert review.approved is True


def test_reviewer_rejects_blank_question() -> None:
    reviewer = ReviewerAgent()
    draft = create_complete_draft()

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        reviewer.run(
            question="   ",
            draft=draft,
        )


def test_reviewer_rejects_shallow_repetitive_answer() -> None:
    reviewer = ReviewerAgent()

    draft = DraftAnswer(
        content=(
            "The orbital velocity decreases as the orbital "
            "radius increases. This decreases the velocity of "
            "the object at the same time."
        ),
        reasoning_steps=[
            "Identify the relevant concepts.",
            "Explain the relationship.",
            "State the conclusion.",
        ],
    )

    review = reviewer.run(
        question=(
            "Why does orbital velocity decrease "
            "as orbital radius increases?"
        ),
        draft=draft,
    )

    assert review.approved is False

    assert (
        "The draft answer is too brief."
        in review.issues
    )

    assert (
        "The draft contains too few complete sentences."
        in review.issues
    )

    assert (
        "Expand the answer to at least three complete "
        "sentences with a clear explanation."
        in review.revision_instructions
    )

    assert (
        "Write at least three complete explanatory sentences."
        in review.revision_instructions
    )