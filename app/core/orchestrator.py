from uuid import uuid4

from app.agents import (
    FinalizerAgent,
    PlannerAgent,
    ReviewerAgent,
    SolverAgent,
)
from app.models import (
    AgentEvent,
    AgentName,
    TaskRequest,
    TaskResult,
)


class WorkflowOrchestrator:
    """Coordinate the complete multi-agent workflow."""

    def __init__(
        self,
        planner: PlannerAgent | None = None,
        solver: SolverAgent | None = None,
        reviewer: ReviewerAgent | None = None,
        finalizer: FinalizerAgent | None = None,
    ) -> None:
        self.planner = planner or PlannerAgent()
        self.solver = solver or SolverAgent()
        self.reviewer = reviewer or ReviewerAgent()
        self.finalizer = finalizer or FinalizerAgent()

    def run(
        self,
        request: TaskRequest,
    ) -> TaskResult:
        """Run all agents and return the complete workflow result."""

        events: list[AgentEvent] = []
        sequence = 1

        plan = self.planner.run(request.question)

        events.append(
            AgentEvent(
                sequence=sequence,
                agent_name=AgentName.PLANNER,
                action="create_plan",
                input_summary=(
                    f"Received task: {request.question}"
                ),
                output_summary=(
                    f"Created a plan with "
                    f"{len(plan.steps)} steps."
                ),
            )
        )
        sequence += 1

        revision_count = 0

        draft = self.solver.run(
            question=request.question,
            plan=plan,
            revision_number=revision_count,
        )

        events.append(
            AgentEvent(
                sequence=sequence,
                agent_name=AgentName.SOLVER,
                action="create_draft",
                input_summary=(
                    f"Received a plan with "
                    f"{len(plan.steps)} steps."
                ),
                output_summary=(
                    f"Produced revision "
                    f"{draft.revision_number}."
                ),
            )
        )
        sequence += 1

        review = self.reviewer.run(
            question=request.question,
            draft=draft,
        )

        events.append(
            AgentEvent(
                sequence=sequence,
                agent_name=AgentName.REVIEWER,
                action="review_draft",
                input_summary=(
                    f"Reviewed revision "
                    f"{draft.revision_number}."
                ),
                output_summary=(
                    "Approved the draft."
                    if review.approved
                    else (
                        f"Found {len(review.issues)} "
                        "issue(s)."
                    )
                ),
            )
        )
        sequence += 1

        while (
            not review.approved
            and revision_count < request.max_revisions
        ):
            revision_count += 1

            draft = self.solver.run(
                question=request.question,
                plan=plan,
                revision_instructions=(
                    review.revision_instructions
                ),
                revision_number=revision_count,
            )

            events.append(
                AgentEvent(
                    sequence=sequence,
                    agent_name=AgentName.SOLVER,
                    action="revise_draft",
                    input_summary=(
                        f"Received "
                        f"{len(review.revision_instructions)} "
                        "revision instruction(s)."
                    ),
                    output_summary=(
                        f"Produced revision "
                        f"{draft.revision_number}."
                    ),
                )
            )
            sequence += 1

            review = self.reviewer.run(
                question=request.question,
                draft=draft,
            )

            events.append(
                AgentEvent(
                    sequence=sequence,
                    agent_name=AgentName.REVIEWER,
                    action="review_revision",
                    input_summary=(
                        f"Reviewed revision "
                        f"{draft.revision_number}."
                    ),
                    output_summary=(
                        "Approved the revised draft."
                        if review.approved
                        else (
                            f"Found {len(review.issues)} "
                            "remaining issue(s)."
                        )
                    ),
                )
            )
            sequence += 1

        final_answer = self.finalizer.run(
            question=request.question,
            draft=draft,
            review=review,
        )

        events.append(
            AgentEvent(
                sequence=sequence,
                agent_name=AgentName.FINALIZER,
                action="finalize_answer",
                input_summary=(
                    f"Received revision "
                    f"{draft.revision_number}."
                ),
                output_summary=(
                    "Produced an approved final answer."
                    if final_answer.approved
                    else (
                        "Produced a safe refusal with unresolved issues."
                    )
                ),
            )
        )

        return TaskResult(
            task_id=str(uuid4()),
            final_answer=final_answer.content,
            plan=plan,
            review=review,
            events=events,
            revision_count=revision_count,
            completed=True,
        )