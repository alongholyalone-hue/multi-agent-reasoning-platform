import os

from app.providers.base import ModelProvider
from app.providers.causal import (
    DEFAULT_CAUSAL_MODEL_NAME,
    HuggingFaceCausalProvider,
)
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
        Use a local sequence-to-sequence Hugging Face model.

    causal:
        Use a local causal instruction model.
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
        model_name = _read_nonempty_string(
            environment_name="HF_MODEL_NAME",
            default=DEFAULT_MODEL_NAME,
        )

        max_new_tokens = _read_positive_integer(
            environment_name="HF_MAX_NEW_TOKENS",
            default=128,
        )

        device = _read_device(
            environment_name="HF_DEVICE",
        )

        return HuggingFaceText2TextProvider(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    if normalized_mode == "causal":
        model_name = _read_nonempty_string(
            environment_name="CAUSAL_MODEL_NAME",
            default=DEFAULT_CAUSAL_MODEL_NAME,
        )

        max_new_tokens = _read_positive_integer(
            environment_name="CAUSAL_MAX_NEW_TOKENS",
            default=128,
        )

        device = _read_device(
            environment_name="CAUSAL_DEVICE",
        )

        return HuggingFaceCausalProvider(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    raise ValueError(
        "MODEL_PROVIDER must be one of "
        "'scaffold', 'huggingface', or 'causal'"
    )


def _read_nonempty_string(
    *,
    environment_name: str,
    default: str,
) -> str:
    """Read and validate a nonempty string environment variable."""

    value = os.getenv(
        environment_name,
        default,
    ).strip()

    if not value:
        raise ValueError(
            f"{environment_name} cannot be empty"
        )

    return value


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


def _read_device(
    *,
    environment_name: str,
) -> int | str:
    """Read an integer index or a torch device string."""

    raw_device = os.getenv(
        environment_name,
        "-1",
    ).strip()

    if not raw_device:
        raise ValueError(
            f"{environment_name} cannot be empty"
        )

    try:
        return int(raw_device)
    except ValueError:
        return raw_device