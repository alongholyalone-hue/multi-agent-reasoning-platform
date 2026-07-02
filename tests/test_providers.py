import pytest

from app.providers import (
    DeterministicModelProvider,
    GenerationCall,
    ModelProvider,
)


def test_deterministic_provider_matches_protocol() -> None:
    provider = DeterministicModelProvider()

    assert isinstance(provider, ModelProvider)


def test_deterministic_provider_returns_configured_response() -> None:
    provider = DeterministicModelProvider(
        response="A predictable answer."
    )

    result = provider.generate(
        system_prompt="You are a scientific reasoning agent.",
        user_prompt="Explain orbital velocity.",
    )

    assert result == "A predictable answer."


def test_deterministic_provider_records_generation_call() -> None:
    provider = DeterministicModelProvider(
        response="Generated answer."
    )

    provider.generate(
        system_prompt="  You are a solver.  ",
        user_prompt="  Explain gravity.  ",
    )

    assert provider.calls == [
        GenerationCall(
            system_prompt="You are a solver.",
            user_prompt="Explain gravity.",
        )
    ]


def test_deterministic_provider_rejects_blank_system_prompt() -> None:
    provider = DeterministicModelProvider()

    with pytest.raises(
        ValueError,
        match="system_prompt cannot be empty",
    ):
        provider.generate(
            system_prompt="   ",
            user_prompt="Explain gravity.",
        )


def test_deterministic_provider_rejects_blank_user_prompt() -> None:
    provider = DeterministicModelProvider()

    with pytest.raises(
        ValueError,
        match="user_prompt cannot be empty",
    ):
        provider.generate(
            system_prompt="You are a solver.",
            user_prompt="   ",
        )


def test_deterministic_provider_rejects_blank_response() -> None:
    provider = DeterministicModelProvider(
        response="   "
    )

    with pytest.raises(
        ValueError,
        match="provider response cannot be empty",
    ):
        provider.generate(
            system_prompt="You are a solver.",
            user_prompt="Explain gravity.",
        )