import pytest
from pydantic import ValidationError

from app.models import (
    AgentEvent,
    AgentName,
    Plan,
    ReviewResult,
    TaskRequest,
    TaskResult,
)


def test_task_request_uses_default_revision_limit() -> None:
    request = TaskRequest(
        question="Why does orbital velocity decrease with radius?"
    )

    assert request.question == (
        "Why does orbital velocity decrease with radius?"
    )
    assert request.max_revisions == 1


def test_task_request_strips_surrounding_whitespace() -> None:
    request = TaskRequest(
        question="   Explain gravitational acceleration.   "
    )

    assert request.question == "Explain gravitational acceleration."


def test_task_request_rejects_short_question() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(question="?")


def test_task_request_rejects_excessive_revision_limit() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(
            question="Explain the orbit.",
            max_revisions=10,
        )


def test_task_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(
            question="Explain the orbit.",
            unknown_setting=True,
        )


def test_agent_event_rejects_invalid_sequence() -> None:
    with pytest.raises(ValidationError):
        AgentEvent(
            sequence=0,
            agent_name=AgentName.PLANNER,
            action="create_plan",
            input_summary="A physics question",
            output_summary="A four-step plan",
        )


def test_task_result_supports_nested_workflow_models() -> None:
    plan = Plan(
        objective="Explain orbital velocity.",
        steps=[
            "Identify the governing equation.",
            "Explain the relationship between radius and velocity.",
        ],
        required_tools=[],
    )

    review = ReviewResult(
        approved=True,
        issues=[],
        revision_instructions=[],
    )

    event = AgentEvent(
        sequence=1,
        agent_name=AgentName.PLANNER,
        action="create_plan",
        input_summary="Question about orbital velocity",
        output_summary="Created a two-step solution plan",
    )

    result = TaskResult(
        task_id="task-001",
        final_answer=(
            "Orbital velocity decreases as orbital radius increases."
        ),
        plan=plan,
        review=review,
        events=[event],
        revision_count=0,
        completed=True,
    )

    assert result.completed is True
    assert result.plan.objective == "Explain orbital velocity."
    assert result.review.approved is True
    assert result.events[0].agent_name == AgentName.PLANNER