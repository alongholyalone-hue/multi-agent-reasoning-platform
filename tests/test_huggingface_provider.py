import pytest

from app.providers import (
    HuggingFaceText2TextProvider,
    ModelProvider,
)


class FakeGenerator:
    """Simulate a Hugging Face generation pipeline."""

    def __init__(
        self,
        generated_text: str = "Generated local answer.",
    ) -> None:
        self.generated_text = generated_text
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        prompt: str,
        **kwargs: object,
    ) -> list[dict[str, str]]:
        self.calls.append(
            {
                "prompt": prompt,
                **kwargs,
            }
        )

        return [
            {
                "generated_text": self.generated_text,
            }
        ]


class RecordingPipelineFactory:
    """Record pipeline creation without loading a real model."""

    def __init__(
        self,
        generator: object,
    ) -> None:
        self.generator = generator
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        **kwargs: object,
    ) -> object:
        self.calls.append(dict(kwargs))
        return self.generator


def test_huggingface_provider_matches_protocol() -> None:
    provider = HuggingFaceText2TextProvider(
        pipeline_factory=RecordingPipelineFactory(
            FakeGenerator()
        )
    )

    assert isinstance(provider, ModelProvider)


def test_huggingface_provider_returns_generated_text() -> None:
    generator = FakeGenerator(
        generated_text="Orbital velocity decreases with radius."
    )

    provider = HuggingFaceText2TextProvider(
        pipeline_factory=RecordingPipelineFactory(
            generator
        )
    )

    result = provider.generate(
        system_prompt="You are a scientific solver.",
        user_prompt="Explain orbital velocity.",
    )

    assert result == (
        "Orbital velocity decreases with radius."
    )


def test_huggingface_provider_loads_pipeline_lazily() -> None:
    generator = FakeGenerator()
    factory = RecordingPipelineFactory(generator)

    provider = HuggingFaceText2TextProvider(
        model_name="test-model",
        pipeline_factory=factory,
    )

    assert factory.calls == []

    provider.generate(
        system_prompt="You are a solver.",
        user_prompt="Explain gravity.",
    )

    assert factory.calls == [
        {
            "task": "text2text-generation",
            "model": "test-model",
            "device": -1,
        }
    ]


def test_huggingface_provider_reuses_loaded_pipeline() -> None:
    generator = FakeGenerator()
    factory = RecordingPipelineFactory(generator)

    provider = HuggingFaceText2TextProvider(
        pipeline_factory=factory,
    )

    provider.generate(
        system_prompt="You are a solver.",
        user_prompt="Explain gravity.",
    )

    provider.generate(
        system_prompt="You are a solver.",
        user_prompt="Explain orbital velocity.",
    )

    assert len(factory.calls) == 1
    assert len(generator.calls) == 2


def test_huggingface_provider_combines_prompts() -> None:
    generator = FakeGenerator()

    provider = HuggingFaceText2TextProvider(
        pipeline_factory=RecordingPipelineFactory(
            generator
        )
    )

    provider.generate(
        system_prompt="You are a scientific solver.",
        user_prompt="Explain gravity.",
    )

    prompt = generator.calls[0]["prompt"]

    assert isinstance(prompt, str)
    assert "Instruction:" in prompt
    assert "You are a scientific solver." in prompt
    assert "Task:" in prompt
    assert "Explain gravity." in prompt


def test_huggingface_provider_uses_generation_settings() -> None:
    generator = FakeGenerator()

    provider = HuggingFaceText2TextProvider(
        max_new_tokens=128,
        pipeline_factory=RecordingPipelineFactory(
            generator
        ),
    )

    provider.generate(
        system_prompt="You are a solver.",
        user_prompt="Explain gravity.",
    )

    assert generator.calls[0]["max_new_tokens"] == 128
    assert generator.calls[0]["do_sample"] is False


def test_huggingface_provider_rejects_blank_prompt() -> None:
    provider = HuggingFaceText2TextProvider(
        pipeline_factory=RecordingPipelineFactory(
            FakeGenerator()
        )
    )

    with pytest.raises(
        ValueError,
        match="system_prompt cannot be empty",
    ):
        provider.generate(
            system_prompt="   ",
            user_prompt="Explain gravity.",
        )


def test_huggingface_provider_rejects_invalid_output() -> None:
    def invalid_generator(
        prompt: str,
        **kwargs: object,
    ) -> list[dict[str, str]]:
        return []

    provider = HuggingFaceText2TextProvider(
        pipeline_factory=RecordingPipelineFactory(
            invalid_generator
        )
    )

    with pytest.raises(
        RuntimeError,
        match="local model returned an invalid response",
    ):
        provider.generate(
            system_prompt="You are a solver.",
            user_prompt="Explain gravity.",
        )