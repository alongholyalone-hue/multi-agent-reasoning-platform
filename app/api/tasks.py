from fastapi import APIRouter

from app.core import WorkflowOrchestrator
from app.models import TaskRequest, TaskResult


router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
)

orchestrator = WorkflowOrchestrator()


@router.post(
    "/solve",
    response_model=TaskResult,
    summary="Run the multi-agent reasoning workflow",
)
def solve_task(request: TaskRequest) -> TaskResult:
    """
    Run the planner, solver, reviewer, optional revision loop,
    and finalizer for a submitted technical question.
    """

    return orchestrator.run(request)