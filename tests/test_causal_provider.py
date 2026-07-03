import pytest
import torch

from app.providers import (
    HuggingFaceCausalProvider,
    ModelProvider,
)


class FakeTokenizer:
    """Simulate a tokenizer with a chat template."""

    eos_token_id = 99

    def __init__(
        self,
        decoded_text: str = "Generated causal answer.",
    ) -> None:
        self.decoded_text = decoded_text
        self.template_calls: list[dict[str, object]] = []
        self.tokenizer_calls: list[dict[str, object]] = []
        self.decode_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": (
                    add_generation_prompt
                ),
            }
        )

        return "formatted chat prompt"

    def __call__(
        self,
        prompt: str,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        self.tokenizer_calls.append(
            {
                "prompt": prompt,
                **kwargs,
            }
        )

        return {
            "input_ids": torch.tensor(
                [[11, 12, 13]]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1]]
            ),
        }

    def decode(
        self,
        token_ids: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> str:
        self.decode_calls.append(
            {
                "token_ids": token_ids.tolist(),
                "skip_special_tokens": (
                    skip_special_tokens
                ),
            }
        )

        return self.decoded_text


class FakeModel:
    """Simulate a causal language model."""

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
    ) -> torch.Tensor:
        self.generate_calls.append(dict(kwargs))

        return torch.tensor(
            [[11, 12, 13, 201, 202, 203]]
        )


class RecordingFactory:
    """Return a component and record model names."""

    def __init__(self, component: object) -> None:
        self.component = component
        self.calls: list[str] = []

    def __call__(self, model_name: str) -> object:
        self.calls.append(model_name)
        return self.component


def create_provider(
    *,
    decoded_text: str = "Generated causal answer.",
    max_new_tokens: int = 128,
) -> tuple[
    HuggingFaceCausalProvider,
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

    provider = HuggingFaceCausalProvider(
        model_name="test-causal-model",
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


def test_causal_provider_matches_protocol() -> None:
    provider, _, _, _, _ = create_provider()

    assert isinstance(provider, ModelProvider)


def test_causal_provider_returns_generated_text() -> None:
    provider, _, _, _, _ = create_provider(
        decoded_text=(
            "Orbital velocity follows from gravity."
        )
    )

    result = provider.generate(
        system_prompt="You are a physics assistant.",
        user_prompt="Explain orbital velocity.",
    )

    assert result == (
        "Orbital velocity follows from gravity."
    )


def test_causal_provider_loads_components_lazily() -> None:
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
        system_prompt="You are an assistant.",
        user_prompt="Explain gravity.",
    )

    assert tokenizer_factory.calls == [
        "test-causal-model"
    ]
    assert model_factory.calls == [
        "test-causal-model"
    ]


def test_causal_provider_reuses_components() -> None:
    (
        provider,
        _,
        model,
        tokenizer_factory,
        model_factory,
    ) = create_provider()

    provider.generate(
        system_prompt="You are an assistant.",
        user_prompt="Explain gravity.",
    )

    provider.generate(
        system_prompt="You are an assistant.",
        user_prompt="Explain velocity.",
    )

    assert len(tokenizer_factory.calls) == 1
    assert len(model_factory.calls) == 1
    assert model.eval_call_count == 1
    assert len(model.generate_calls) == 2


def test_causal_provider_builds_chat_messages() -> None:
    provider, tokenizer, _, _, _ = create_provider()

    provider.generate(
        system_prompt="You are a physics assistant.",
        user_prompt="Explain orbital velocity.",
    )

    template_call = tokenizer.template_calls[0]
    messages = template_call["messages"]

    assert messages == [
        {
            "role": "system",
            "content": "You are a physics assistant.",
        },
        {
            "role": "user",
            "content": "Explain orbital velocity.",
        },
    ]

    assert template_call["tokenize"] is False
    assert (
        template_call["add_generation_prompt"]
        is True
    )


def test_causal_provider_uses_generation_settings() -> None:
    provider, tokenizer, model, _, _ = create_provider(
        max_new_tokens=96
    )

    provider.generate(
        system_prompt="You are an assistant.",
        user_prompt="Explain gravity.",
    )

    generation_call = model.generate_calls[0]

    assert generation_call["max_new_tokens"] == 96
    assert generation_call["do_sample"] is False
    assert generation_call["pad_token_id"] == 99
    assert model.device_calls == ["cpu"]

    assert tokenizer.decode_calls[0]["token_ids"] == [
        201,
        202,
        203,
    ]


def test_causal_provider_rejects_blank_prompt() -> None:
    provider, _, _, _, _ = create_provider()

    with pytest.raises(
        ValueError,
        match="system_prompt cannot be empty",
    ):
        provider.generate(
            system_prompt="   ",
            user_prompt="Explain gravity.",
        )


def test_causal_provider_rejects_empty_output() -> None:
    provider, _, _, _, _ = create_provider(
        decoded_text="   "
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "local causal model returned "
            "an empty response"
        ),
    ):
        provider.generate(
            system_prompt="You are an assistant.",
            user_prompt="Explain gravity.",
        )