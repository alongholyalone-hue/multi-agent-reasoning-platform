import os

from app.providers.base import ModelProvider
from app.providers.huggingface import (
    DEFAULT_MODEL_NAME,
    HuggingFaceText2TextProvider,
)


DEFAULT_PROVIDER_MODE = "scaffold"


def create_model_provider(
    mode: str | None = None,
) -> ModelProvider | None:
    """
    Create the configured model provider.

    scaffold:
        Use the Solver Agent's deterministic development output.

    huggingface:
        Use a local Hugging Face sequence-to-sequence model.
    """

    selected_mode = (
        mode
        if mode is not None
        else os.getenv(
            "MODEL_PROVIDER",
            DEFAULT_PROVIDER_MODE,
        )
    )

    normalized_mode = selected_mode.strip().lower()

    if normalized_mode == "scaffold":
        return None

    if normalized_mode == "huggingface":
        model_name = os.getenv(
            "HF_MODEL_NAME",
            DEFAULT_MODEL_NAME,
        ).strip()

        max_new_tokens = _read_positive_integer(
            environment_name="HF_MAX_NEW_TOKENS",
            default=128,
        )

        device = _read_device()

        return HuggingFaceText2TextProvider(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    raise ValueError(
        "MODEL_PROVIDER must be either "
        "'scaffold' or 'huggingface'"
    )


def _read_positive_integer(
    *,
    environment_name: str,
    default: int,
) -> int:
    """Read and validate a positive integer environment variable."""

    raw_value = os.getenv(
        environment_name,
        str(default),
    ).strip()

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{environment_name} must be an integer"
        ) from error

    if value <= 0:
        raise ValueError(
            f"{environment_name} must be greater than zero"
        )

    return value


def _read_device() -> int | str:
    """Read a pipeline-style integer or a torch device string."""

    raw_device = os.getenv(
        "HF_DEVICE",
        "-1",
    ).strip()

    if not raw_device:
        raise ValueError("HF_DEVICE cannot be empty")

    try:
        return int(raw_device)
    except ValueError:
        return raw_device