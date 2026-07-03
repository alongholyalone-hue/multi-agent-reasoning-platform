from collections.abc import Callable
from typing import Any

import torch


DEFAULT_CAUSAL_MODEL_NAME = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)


class HuggingFaceCausalProvider:
    """
    Generate answers using a local causal instruction model.

    The tokenizer and model are loaded lazily during the first
    generation request.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CAUSAL_MODEL_NAME,
        max_new_tokens: int = 128,
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
        """Generate an answer from a causal instruction model."""

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

        messages = [
            {
                "role": "system",
                "content": cleaned_system_prompt,
            },
            {
                "role": "user",
                "content": cleaned_user_prompt,
            },
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if not isinstance(formatted_prompt, str):
            raise RuntimeError(
                "tokenizer returned an invalid chat prompt"
            )

        model_inputs = tokenizer(
            formatted_prompt,
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

        input_length = int(
            model_inputs["input_ids"].shape[-1]
        )

        generation_settings: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }

        eos_token_id = getattr(
            tokenizer,
            "eos_token_id",
            None,
        )

        if eos_token_id is not None:
            generation_settings["pad_token_id"] = (
                eos_token_id
            )

        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs,
                **generation_settings,
            )

        generated_ids = output_ids[0][input_length:]

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        if not generated_text:
            raise RuntimeError(
                "local causal model returned an empty response"
            )

        return generated_text

    def _get_components(self) -> tuple[Any, Any]:
        """Load and cache the tokenizer and causal model."""

        if (
            self._tokenizer is not None
            and self._model is not None
        ):
            return self._tokenizer, self._model

        tokenizer_factory = self._tokenizer_factory
        model_factory = self._model_factory

        if tokenizer_factory is None or model_factory is None:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            tokenizer_factory = (
                tokenizer_factory
                or AutoTokenizer.from_pretrained
            )

            model_factory = (
                model_factory
                or AutoModelForCausalLM.from_pretrained
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
        """Convert an integer or string into a torch device name."""

        if isinstance(device, int):
            if device < 0:
                return "cpu"

            return f"cuda:{device}"

        cleaned_device = device.strip()

        if not cleaned_device:
            raise ValueError("device cannot be empty")

        return cleaned_device