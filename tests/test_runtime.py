from app.core import create_workflow_orchestrator
from app.providers import (
    HuggingFaceCausalProvider,
    HuggingFaceText2TextProvider,
)


def test_runtime_uses_scaffold_solver() -> None:
    orchestrator = create_workflow_orchestrator(
        provider_mode="scaffold"
    )

    assert orchestrator.solver.provider is None


def test_runtime_injects_huggingface_provider() -> None:
    orchestrator = create_workflow_orchestrator(
        provider_mode="huggingface"
    )

    assert (
    orchestrator.reviewer.provider
    is orchestrator.solver.provider
    )

    assert isinstance(
        orchestrator.solver.provider,
        HuggingFaceText2TextProvider,
    )


def test_runtime_injects_causal_provider() -> None:
    orchestrator = create_workflow_orchestrator(
        provider_mode="causal"
    )

    assert (
    orchestrator.reviewer.provider
    is orchestrator.solver.provider
    )

    assert isinstance(
        orchestrator.solver.provider,
        HuggingFaceCausalProvider,
    )