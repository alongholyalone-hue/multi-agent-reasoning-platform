from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelProvider(Protocol):
    """Common interface implemented by all model providers."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate text from system and user prompts."""

        ...