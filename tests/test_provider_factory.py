import pytest

from app.providers import (
    HuggingFaceCausalProvider,
    HuggingFaceText2TextProvider,
    create_model_provider,
)


def test_provider_factory_uses_scaffold_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "MODEL_PROVIDER",
        raising=False,
    )

    provider = create_model_provider()

    assert provider is None


def test_provider_factory_creates_huggingface_provider() -> None:
    provider = create_model_provider(
        mode="huggingface"
    )

    assert isinstance(
        provider,
        HuggingFaceText2TextProvider,
    )


def test_provider_factory_creates_causal_provider() -> None:
    provider = create_model_provider(
        mode="causal"
    )

    assert isinstance(
        provider,
        HuggingFaceCausalProvider,
    )


def test_provider_factory_normalizes_mode() -> None:
    provider = create_model_provider(
        mode="  CAUSAL  "
    )

    assert isinstance(
        provider,
        HuggingFaceCausalProvider,
    )


def test_provider_factory_reads_huggingface_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MODEL_PROVIDER",
        "huggingface",
    )
    monkeypatch.setenv(
        "HF_MODEL_NAME",
        "custom-text-model",
    )
    monkeypatch.setenv(
        "HF_MAX_NEW_TOKENS",
        "77",
    )
    monkeypatch.setenv(
        "HF_DEVICE",
        "cpu",
    )

    provider = create_model_provider()

    assert isinstance(
        provider,
        HuggingFaceText2TextProvider,
    )
    assert provider.model_name == "custom-text-model"
    assert provider.max_new_tokens == 77
    assert provider.device == "cpu"


def test_provider_factory_reads_causal_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MODEL_PROVIDER",
        "causal",
    )
    monkeypatch.setenv(
        "CAUSAL_MODEL_NAME",
        "custom-causal-model",
    )
    monkeypatch.setenv(
        "CAUSAL_MAX_NEW_TOKENS",
        "96",
    )
    monkeypatch.setenv(
        "CAUSAL_DEVICE",
        "cpu",
    )

    provider = create_model_provider()

    assert isinstance(
        provider,
        HuggingFaceCausalProvider,
    )
    assert provider.model_name == "custom-causal-model"
    assert provider.max_new_tokens == 96
    assert provider.device == "cpu"


def test_provider_factory_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="MODEL_PROVIDER must be one of",
    ):
        create_model_provider(
            mode="unknown"
        )


def test_provider_factory_rejects_invalid_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HF_MAX_NEW_TOKENS",
        "zero",
    )

    with pytest.raises(
        ValueError,
        match="HF_MAX_NEW_TOKENS must be an integer",
    ):
        create_model_provider(
            mode="huggingface"
        )