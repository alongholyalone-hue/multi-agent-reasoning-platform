import pytest

from app.agents import PlannerAgent
from app.models import Plan


def test_planner_returns_structured_plan() -> None:
    planner = PlannerAgent()

    plan = planner.run(
        "Why does orbital velocity decrease as radius increases?"
    )

    assert isinstance(plan, Plan)
    assert "orbital velocity" in plan.objective
    assert len(plan.steps) == 4
    assert plan.required_tools == []


def test_planner_detects_numerical_question() -> None:
    planner = PlannerAgent()

    plan = planner.run(
        "Calculate the orbital speed at a radius of 7000 km."
    )

    assert plan.required_tools == ["calculator"]


def test_planner_detects_calculation_keyword_without_number() -> None:
    planner = PlannerAgent()

    plan = planner.run(
        "How much energy is required to change the orbit?"
    )

    assert plan.required_tools == ["calculator"]


def test_planner_normalizes_question_whitespace() -> None:
    planner = PlannerAgent()

    plan = planner.run(
        "  Explain   gravitational   acceleration.  "
    )

    assert plan.objective.endswith(
        "Explain gravitational acceleration."
    )


def test_planner_rejects_blank_question() -> None:
    planner = PlannerAgent()

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        planner.run("   ")