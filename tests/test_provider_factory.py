import pytest

from app.providers import (
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


def test_provider_factory_normalizes_mode() -> None:
    provider = create_model_provider(
        mode="  HUGGINGFACE  "
    )

    assert isinstance(
        provider,
        HuggingFaceText2TextProvider,
    )


def test_provider_factory_reads_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MODEL_PROVIDER",
        "huggingface",
    )
    monkeypatch.setenv(
        "HF_MODEL_NAME",
        "custom-test-model",
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
    assert provider.model_name == "custom-test-model"
    assert provider.max_new_tokens == 77
    assert provider.device == "cpu"


def test_provider_factory_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "MODEL_PROVIDER must be either "
            "'scaffold' or 'huggingface'"
        ),
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