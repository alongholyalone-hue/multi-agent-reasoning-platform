import pytest

from app.agents import SolverAgent
from app.models import DraftAnswer, Plan


def create_test_plan(
    required_tools: list[str] | None = None,
) -> Plan:
    return Plan(
        objective="Explain the relationship between radius and velocity.",
        steps=[
            "Identify the relevant orbital equation.",
            "Explain how radius affects velocity.",
            "Produce a clear conclusion.",
        ],
        required_tools=required_tools or [],
    )


def test_solver_returns_structured_draft() -> None:
    solver = SolverAgent()
    plan = create_test_plan()

    draft = solver.run(
        question=(
            "Why does orbital velocity decrease "
            "as radius increases?"
        ),
        plan=plan,
    )

    assert isinstance(draft, DraftAnswer)
    assert "orbital velocity" in draft.content
    assert draft.reasoning_steps == plan.steps
    assert draft.revision_number == 0


def test_solver_preserves_requested_tools() -> None:
    solver = SolverAgent()
    plan = create_test_plan(
        required_tools=["calculator"]
    )

    draft = solver.run(
        question="Calculate the orbital velocity.",
        plan=plan,
    )

    assert draft.requested_tools == ["calculator"]
    assert "Requested tools: calculator" in draft.content


def test_solver_applies_revision_instructions() -> None:
    solver = SolverAgent()
    plan = create_test_plan()

    draft = solver.run(
        question="Explain orbital velocity.",
        plan=plan,
        revision_instructions=[
            "Include the governing equation.",
            "Clarify the final conclusion.",
        ],
        revision_number=1,
    )

    assert draft.revision_number == 1
    assert draft.applied_revision_instructions == [
        "Include the governing equation.",
        "Clarify the final conclusion.",
    ]
    assert (
        "Apply reviewer instruction: "
        "Include the governing equation."
    ) in draft.reasoning_steps


def test_solver_normalizes_question_whitespace() -> None:
    solver = SolverAgent()
    plan = create_test_plan()

    draft = solver.run(
        question="  Explain   orbital   velocity.  ",
        plan=plan,
    )

    assert draft.content.startswith(
        "Draft answer for: Explain orbital velocity."
    )


def test_solver_rejects_blank_question() -> None:
    solver = SolverAgent()
    plan = create_test_plan()

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        solver.run(
            question="   ",
            plan=plan,
        )


def test_solver_rejects_negative_revision_number() -> None:
    solver = SolverAgent()
    plan = create_test_plan()

    with pytest.raises(
        ValueError,
        match="revision_number cannot be negative",
    ):
        solver.run(
            question="Explain orbital velocity.",
            plan=plan,
            revision_number=-1,
        )