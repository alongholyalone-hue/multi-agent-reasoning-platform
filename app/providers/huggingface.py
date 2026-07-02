from collections.abc import Callable
from typing import Any

import torch


DEFAULT_MODEL_NAME = "google/flan-t5-small"


class HuggingFaceText2TextProvider:
    """
    Generate responses using a local Hugging Face sequence-to-sequence model.

    The tokenizer and model are loaded lazily during the first generation
    request so importing the application does not immediately download
    or load the model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = 256,
        device: int | str = -1,
        tokenizer_factory: Callable[[str], Any] | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        cleaned_model_name = model_name.strip()

        if not cleaned_model_name:
            raise ValueError("model_name cannot be empty")

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero"
            )

        self.model_name = cleaned_model_name
        self.max_new_tokens = max_new_tokens
        self.device = self._resolve_device(device)

        self._tokenizer_factory = tokenizer_factory
        self._model_factory = model_factory

        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response using the local model."""

        cleaned_system_prompt = system_prompt.strip()
        cleaned_user_prompt = user_prompt.strip()

        if not cleaned_system_prompt:
            raise ValueError(
                "system_prompt cannot be empty"
            )

        if not cleaned_user_prompt:
            raise ValueError(
                "user_prompt cannot be empty"
            )

        tokenizer, model = self._get_components()

        combined_prompt = (
            "Instruction:\n"
            f"{cleaned_system_prompt}\n\n"
            "Task:\n"
            f"{cleaned_user_prompt}"
        )

        model_inputs = tokenizer(
            combined_prompt,
            return_tensors="pt",
            truncation=True,
        )

        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(self.device)
        else:
            model_inputs = {
                name: tensor.to(self.device)
                for name, tensor in model_inputs.items()
            }

        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        if not generated_text:
            raise RuntimeError(
                "local model returned an empty response"
            )

        return generated_text

    def _get_components(self) -> tuple[Any, Any]:
        """Load and cache the tokenizer and model."""

        if (
            self._tokenizer is not None
            and self._model is not None
        ):
            return self._tokenizer, self._model

        tokenizer_factory = self._tokenizer_factory
        model_factory = self._model_factory

        if tokenizer_factory is None or model_factory is None:
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )

            tokenizer_factory = (
                tokenizer_factory
                or AutoTokenizer.from_pretrained
            )

            model_factory = (
                model_factory
                or AutoModelForSeq2SeqLM.from_pretrained
            )

        self._tokenizer = tokenizer_factory(
            self.model_name
        )

        self._model = model_factory(
            self.model_name
        )

        self._model.to(self.device)
        self._model.eval()

        return self._tokenizer, self._model

    @staticmethod
    def _resolve_device(
        device: int | str,
    ) -> str:
        """Convert pipeline-style device settings into torch devices."""

        if isinstance(device, int):
            if device < 0:
                return "cpu"

            return f"cuda:{device}"

        cleaned_device = device.strip()

        if not cleaned_device:
            raise ValueError("device cannot be empty")

        return cleaned_device