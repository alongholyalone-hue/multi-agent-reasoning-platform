from collections.abc import Callable
from typing import Any


DEFAULT_MODEL_NAME = "google/flan-t5-small"


class HuggingFaceText2TextProvider:
    """
    Generate responses using a local Hugging Face text-to-text model.

    The model is loaded lazily during the first generation request so
    importing the application does not immediately download or load it.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = 256,
        device: int = -1,
        pipeline_factory: Callable[..., Any] | None = None,
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
        self.device = device
        self._pipeline_factory = pipeline_factory
        self._generator: Any | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response from the local model."""

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

        generator = self._get_generator()

        combined_prompt = (
            "Instruction:\n"
            f"{cleaned_system_prompt}\n\n"
            "Task:\n"
            f"{cleaned_user_prompt}"
        )

        outputs = generator(
            combined_prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        generated_text = self._extract_generated_text(
            outputs
        )

        if not generated_text:
            raise RuntimeError(
                "local model returned an empty response"
            )

        return generated_text

    def _get_generator(self) -> Any:
        """Load the Hugging Face pipeline only when first needed."""

        if self._generator is not None:
            return self._generator

        pipeline_factory = self._pipeline_factory

        if pipeline_factory is None:
            from transformers import pipeline

            pipeline_factory = pipeline

        self._generator = pipeline_factory(
            task="text2text-generation",
            model=self.model_name,
            device=self.device,
        )

        return self._generator

    @staticmethod
    def _extract_generated_text(
        outputs: Any,
    ) -> str:
        """Validate and extract generated text from pipeline output."""

        if not isinstance(outputs, list) or not outputs:
            raise RuntimeError(
                "local model returned an invalid response"
            )

        first_output = outputs[0]

        if not isinstance(first_output, dict):
            raise RuntimeError(
                "local model returned an invalid response"
            )

        generated_text = first_output.get(
            "generated_text"
        )

        if not isinstance(generated_text, str):
            raise RuntimeError(
                "local model response did not contain generated_text"
            )

        return generated_text.strip()