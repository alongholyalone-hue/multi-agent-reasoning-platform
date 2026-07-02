from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends

from app.core import (
    WorkflowOrchestrator,
    create_workflow_orchestrator,
)
from app.models import TaskRequest, TaskResult


router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
)


@lru_cache(maxsize=1)
def get_orchestrator() -> WorkflowOrchestrator:
    """
    Create and cache the configured workflow orchestrator.

    The Hugging Face model still loads lazily during the first
    generation request.
    """

    return create_workflow_orchestrator()


@router.post(
    "/solve",
    response_model=TaskResult,
    summary="Run the multi-agent reasoning workflow",
)
def solve_task(
    request: TaskRequest,
    orchestrator: Annotated[
        WorkflowOrchestrator,
        Depends(get_orchestrator),
    ],
) -> TaskResult:
    """Run the complete configured multi-agent workflow."""

    return orchestrator.run(request)