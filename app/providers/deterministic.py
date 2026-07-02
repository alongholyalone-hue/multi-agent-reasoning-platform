from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationCall:
    """One recorded model-generation request."""

    system_prompt: str
    user_prompt: str


@dataclass
class DeterministicModelProvider:
    """
    Return a predictable response for tests and local development.

    This provider does not use an AI model. It allows the agent workflow
    to be tested without network access, model downloads, or randomness.
    """

    response: str = "Deterministic model response."

    calls: list[GenerationCall] = field(
        default_factory=list,
        init=False,
    )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Record the prompts and return the configured response."""

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

        cleaned_response = self.response.strip()

        if not cleaned_response:
            raise ValueError(
                "provider response cannot be empty"
            )

        self.calls.append(
            GenerationCall(
                system_prompt=cleaned_system_prompt,
                user_prompt=cleaned_user_prompt,
            )
        )

        return cleaned_response