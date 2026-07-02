import pytest

from app.providers import (
    HuggingFaceText2TextProvider,
    ModelProvider,
)


class FakeBatch(dict):
    """Simulate tokenized model inputs."""

    def __init__(self) -> None:
        super().__init__(
            input_ids="fake-input-ids",
            attention_mask="fake-attention-mask",
        )
        self.device: str | None = None

    def to(self, device: str) -> "FakeBatch":
        self.device = device
        return self


class FakeTokenizer:
    """Simulate a Hugging Face tokenizer."""

    def __init__(
        self,
        decoded_text: str = "Generated local answer.",
    ) -> None:
        self.decoded_text = decoded_text
        self.calls: list[dict[str, object]] = []
        self.decode_calls: list[dict[str, object]] = []
        self.last_batch: FakeBatch | None = None

    def __call__(
        self,
        prompt: str,
        **kwargs: object,
    ) -> FakeBatch:
        self.calls.append(
            {
                "prompt": prompt,
                **kwargs,
            }
        )

        self.last_batch = FakeBatch()
        return self.last_batch

    def decode(
        self,
        token_ids: object,
        *,
        skip_special_tokens: bool,
    ) -> str:
        self.decode_calls.append(
            {
                "token_ids": token_ids,
                "skip_special_tokens": skip_special_tokens,
            }
        )

        return self.decoded_text


class FakeModel:
    """Simulate a Hugging Face sequence-to-sequence model."""

    def __init__(self) -> None:
        self.device_calls: list[str] = []
        self.eval_call_count = 0
        self.generate_calls: list[dict[str, object]] = []

    def to(self, device: str) -> "FakeModel":
        self.device_calls.append(device)
        return self

    def eval(self) -> "FakeModel":
        self.eval_call_count += 1
        return self

    def generate(
        self,
        **kwargs: object,
    ) -> list[list[int]]:
        self.generate_calls.append(dict(kwargs))
        return [[101, 102, 103]]


class RecordingFactory:
    """Return a predefined component and record model names."""

    def __init__(self, component: object) -> None:
        self.component = component
        self.calls: list[str] = []

    def __call__(self, model_name: str) -> object:
        self.calls.append(model_name)
        return self.component


def create_provider(
    *,
    decoded_text: str = "Generated local answer.",
    max_new_tokens: int = 256,
) -> tuple[
    HuggingFaceText2TextProvider,
    FakeTokenizer,
    FakeModel,
    RecordingFactory,
    RecordingFactory,
]:
    tokenizer = FakeTokenizer(
        decoded_text=decoded_text
    )
    model = FakeModel()

    tokenizer_factory = RecordingFactory(tokenizer)
    model_factory = RecordingFactory(model)

    provider = HuggingFaceText2TextProvider(
        model_name="test-model",
        max_new_tokens=max_new_tokens,
        device=-1,
        tokenizer_factory=tokenizer_factory,
        model_factory=model_factory,
    )

    return (
        provider,
        tokenizer,
        model,
        tokenizer_factory,
        model_factory,
    )


def test_huggingface_provider_matches_protocol() -> None:
    provider, _, _, _, _ = create_provider()

    assert isinstance(provider, ModelProvider)


def test_huggingface_provider_returns_decoded_text() -> None:
    provider, _, _, _, _ = create_provider(
        decoded_text=(
            "Orbital velocity decreases with radius."
        )
    )

    result = provider.generate(
        system_prompt="You are a scientific solver.",
        user_prompt="Explain orbital velocity.",
    )

    assert result == (
        "Orbital velocity decreases with radius."
    )


def test_huggingface_provider_loads_components_lazily() -> None:
    (
        provider,
        _,
        _,
        tokenizer_factory,
        model_factory,
    ) = create_provider()

    assert tokenizer_factory.calls == []
    assert model_factory.calls == []

    provider.generate(
        system_prompt="You are a solver.",
        user_prompt="Explain gravity.",
    )

    assert tokenizer_factory.calls == ["test-model"]
    assert model_factory.calls == ["test-model"]


def test_huggingface_provider_reuses_loaded_components() -> None:
    (
        provider,
        _,
        model,
        tokenizer_factory,
        model_factory,
    ) = create_provider()

    provider.generate(
        system_prompt="You are a solver.",
        user_prompt="Explain gravity.",
    )

    provider.generate(
        system_prompt="You are a solver.",
        user_prompt="Explain orbital velocity.",
    )

    assert len(tokenizer_factory.calls) == 1
    assert len(model_factory.calls) == 1
    assert model.eval_call_count == 1
    assert len(model.generate_calls) == 2


def test_huggingface_provider_combines_prompts() -> None:
    provider, tokenizer, _, _, _ = create_provider()

    provider.generate(
        system_prompt="You are a scientific solver.",
        user_prompt="Explain gravity.",
    )

    prompt = tokenizer.calls[0]["prompt"]

    assert isinstance(prompt, str)
    assert "Instruction:" in prompt
    assert "You are a scientific solver." in prompt
    assert "Task:" in prompt
    assert "Explain gravity." in prompt


def test_huggingface_provider_uses_generation_settings() -> None:
    provider, tokenizer, model, _, _ = create_provider(
        max_new_tokens=128
    )

    provider.generate(
        system_prompt="You are a solver.",
        user_prompt="Explain gravity.",
    )

    generation_call = model.generate_calls[0]

    assert generation_call["max_new_tokens"] == 128
    assert generation_call["min_new_tokens"] == 24
    assert generation_call["num_beams"] == 4
    assert generation_call["do_sample"] is False
    assert generation_call["early_stopping"] is True
    assert generation_call["no_repeat_ngram_size"] == 3

    assert tokenizer.last_batch is not None
    assert tokenizer.last_batch.device == "cpu"
    assert model.device_calls == ["cpu"]


def test_huggingface_provider_rejects_blank_prompt() -> None:
    provider, _, _, _, _ = create_provider()

    with pytest.raises(
        ValueError,
        match="system_prompt cannot be empty",
    ):
        provider.generate(
            system_prompt="   ",
            user_prompt="Explain gravity.",
        )


def test_huggingface_provider_rejects_empty_output() -> None:
    provider, _, _, _, _ = create_provider(
        decoded_text="   "
    )

    with pytest.raises(
        RuntimeError,
        match="local model returned an empty response",
    ):
        provider.generate(
            system_prompt="You are a solver.",
            user_prompt="Explain gravity.",
        )