from app.agents import ReviewerAgent, SolverAgent
from app.core import WorkflowOrchestrator
from app.models import (
    AgentName,
    DraftAnswer,
    Plan,
    ReviewResult,
    TaskRequest,
    TaskResult,
)


class RejectOnceReviewer:
    """Reject the first draft and approve the first revision."""

    def __init__(self) -> None:
        self.call_count = 0

    def run(
        self,
        question: str,
        draft: DraftAnswer,
    ) -> ReviewResult:
        self.call_count += 1

        if self.call_count == 1:
            return ReviewResult(
                approved=False,
                issues=[
                    "The draft requires additional detail."
                ],
                revision_instructions=[
                    "Add a clearer explanation."
                ],
            )

        return ReviewResult(
            approved=True,
            issues=[],
            revision_instructions=[],
        )


class AlwaysRejectReviewer:
    """Reject every submitted draft."""

    def run(
        self,
        question: str,
        draft: DraftAnswer,
    ) -> ReviewResult:
        return ReviewResult(
            approved=False,
            issues=[
                "The draft remains incomplete."
            ],
            revision_instructions=[
                "Add more supporting detail."
            ],
        )


class RecordingSolver(SolverAgent):
    """Record every solver invocation for testing."""

    def __init__(self) -> None:
        self.revision_numbers: list[int] = []
        self.received_instructions: list[list[str]] = []

    def run(
        self,
        question: str,
        plan: Plan,
        revision_instructions: list[str] | None = None,
        revision_number: int = 0,
    ) -> DraftAnswer:
        self.revision_numbers.append(revision_number)
        self.received_instructions.append(
            list(revision_instructions or [])
        )

        return super().run(
            question=question,
            plan=plan,
            revision_instructions=revision_instructions,
            revision_number=revision_number,
        )


def test_orchestrator_runs_complete_workflow() -> None:
    orchestrator = WorkflowOrchestrator()

    result = orchestrator.run(
        TaskRequest(
            question=(
                "Why does orbital velocity decrease "
                "as radius increases?"
            )
        )
    )

    assert isinstance(result, TaskResult)
    assert result.completed is True
    assert result.review.approved is True
    assert result.revision_count == 0
    assert result.final_answer


def test_orchestrator_records_agents_in_order() -> None:
    orchestrator = WorkflowOrchestrator()

    result = orchestrator.run(
        TaskRequest(
            question="Explain gravitational acceleration."
        )
    )

    assert [
        event.agent_name
        for event in result.events
    ] == [
        AgentName.PLANNER,
        AgentName.SOLVER,
        AgentName.REVIEWER,
        AgentName.FINALIZER,
    ]

    assert [
        event.sequence
        for event in result.events
    ] == [1, 2, 3, 4]


def test_orchestrator_performs_requested_revision() -> None:
    solver = RecordingSolver()
    reviewer = RejectOnceReviewer()

    orchestrator = WorkflowOrchestrator(
        solver=solver,
        reviewer=reviewer,
    )

    result = orchestrator.run(
        TaskRequest(
            question="Explain orbital velocity.",
            max_revisions=1,
        )
    )

    assert result.review.approved is True
    assert result.revision_count == 1
    assert reviewer.call_count == 2
    assert solver.revision_numbers == [0, 1]
    assert solver.received_instructions[1] == [
        "Add a clearer explanation."
    ]


def test_orchestrator_respects_zero_revision_limit() -> None:
    orchestrator = WorkflowOrchestrator(
        reviewer=AlwaysRejectReviewer(),
    )

    result = orchestrator.run(
        TaskRequest(
            question="Explain orbital velocity.",
            max_revisions=0,
        )
    )

    assert result.review.approved is False
    assert result.revision_count == 0
    assert len(result.events) == 4
    assert result.final_answer.startswith(
        "Best available answer for:"
    )


def test_orchestrator_stops_at_maximum_revisions() -> None:
    solver = RecordingSolver()

    orchestrator = WorkflowOrchestrator(
        solver=solver,
        reviewer=AlwaysRejectReviewer(),
    )

    result = orchestrator.run(
        TaskRequest(
            question="Explain orbital velocity.",
            max_revisions=2,
        )
    )

    assert result.review.approved is False
    assert result.revision_count == 2
    assert solver.revision_numbers == [0, 1, 2]

    assert [
        event.agent_name
        for event in result.events
    ] == [
        AgentName.PLANNER,
        AgentName.SOLVER,
        AgentName.REVIEWER,
        AgentName.SOLVER,
        AgentName.REVIEWER,
        AgentName.SOLVER,
        AgentName.REVIEWER,
        AgentName.FINALIZER,
    ]


def test_orchestrator_generates_unique_task_ids() -> None:
    orchestrator = WorkflowOrchestrator()

    first_result = orchestrator.run(
        TaskRequest(
            question="Explain orbital velocity."
        )
    )

    second_result = orchestrator.run(
        TaskRequest(
            question="Explain gravitational acceleration."
        )
    )

    assert first_result.task_id
    assert second_result.task_id
    assert first_result.task_id != second_result.task_id