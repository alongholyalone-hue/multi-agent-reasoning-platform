from app.agents import SolverAgent
from app.core.orchestrator import WorkflowOrchestrator
from app.providers import create_model_provider


def create_workflow_orchestrator(
    provider_mode: str | None = None,
) -> WorkflowOrchestrator:
    """Create an orchestrator using the configured model provider."""

    provider = create_model_provider(
        mode=provider_mode
    )

    solver = SolverAgent(
        provider=provider
    )

    return WorkflowOrchestrator(
        solver=solver
    )