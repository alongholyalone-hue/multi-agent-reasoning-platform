from app.models import DraftAnswer, Plan
from app.providers import ModelProvider


class SolverAgent:
    """Produce a structured draft from a plan."""

    name = "solver"

    # This class-level default keeps subclasses compatible even when
    # their constructors do not call super().__init__().
    provider: ModelProvider | None = None

    def __init__(
        self,
        provider: ModelProvider | None = None,
    ) -> None:
        self.provider = provider

    def run(
        self,
        question: str,
        plan: Plan,
        revision_instructions: list[str] | None = None,
        revision_number: int = 0,
    ) -> DraftAnswer:
        """Create a structured draft answer."""

        cleaned_question = " ".join(question.split())

        if not cleaned_question:
            raise ValueError("question cannot be empty")

        if revision_number < 0:
            raise ValueError(
                "revision_number cannot be negative"
            )

        cleaned_instructions = [
            " ".join(instruction.split())
            for instruction in (revision_instructions or [])
            if instruction.strip()
        ]

        reasoning_steps = list(plan.steps)

        for instruction in cleaned_instructions:
            reasoning_steps.append(
                f"Apply reviewer instruction: {instruction}"
            )

        if self.provider is None:
            content = self._build_deterministic_content(
                question=cleaned_question,
                plan=plan,
                reasoning_steps=reasoning_steps,
                revision_instructions=cleaned_instructions,
            )
        else:
            content = self.provider.generate(
                system_prompt=self._build_system_prompt(),
                user_prompt=self._build_user_prompt(
                    question=cleaned_question,
                    plan=plan,
                    revision_instructions=(
                        cleaned_instructions
                    ),
                    revision_number=revision_number,
                ),
            )

        return DraftAnswer(
            content=content,
            reasoning_steps=reasoning_steps,
            requested_tools=list(plan.required_tools),
            applied_revision_instructions=(
                cleaned_instructions
            ),
            revision_number=revision_number,
        )

    @staticmethod
    def _build_system_prompt() -> str:
        """Return concise instructions for answer generation."""

        return (
            "Answer the technical question accurately and directly. "
            "Write one coherent paragraph using several complete sentences. "
            "Explain the cause or mechanism instead of stating only the "
            "conclusion. Do not use headings, numbered lists, bullet points, "
            "or unfinished placeholders."
        )

    @staticmethod
    def _build_user_prompt(
        *,
        question: str,
        plan: Plan,
        revision_instructions: list[str],
        revision_number: int,
    ) -> str:
        """Build a compact prompt for the local instruction model."""

        guidance = " ".join(plan.steps)

        feedback = (
            " ".join(revision_instructions)
            if revision_instructions
            else "No reviewer feedback."
        )

        tool_guidance = (
            "Available tools: "
            + ", ".join(plan.required_tools)
            + "."
            if plan.required_tools
            else ""
        )

        return (
            f"Question: {question}\n\n"
            f"Reasoning guidance: {guidance}\n\n"
            f"Reviewer feedback: {feedback}\n\n"
            f"{tool_guidance}\n\n"
            "Provide the complete explanatory answer now."
        )   

    @staticmethod
    def _build_deterministic_content(
        *,
        question: str,
        plan: Plan,
        reasoning_steps: list[str],
        revision_instructions: list[str],
    ) -> str:
        """Build the original deterministic development response."""

        content_lines = [
            f"Draft answer for: {question}",
            "",
            f"Objective: {plan.objective}",
            "",
            "Reasoning approach:",
        ]

        for step_number, step in enumerate(
            reasoning_steps,
            start=1,
        ):
            content_lines.append(
                f"{step_number}. {step}"
            )

        if plan.required_tools:
            content_lines.extend(
                [
                    "",
                    (
                        "Requested tools: "
                        + ", ".join(plan.required_tools)
                    ),
                ]
            )

        if revision_instructions:
            content_lines.extend(
                [
                    "",
                    "Applied revision instructions:",
                ]
            )

            for instruction in revision_instructions:
                content_lines.append(
                    f"- {instruction}"
                )

        return "\n".join(content_lines)