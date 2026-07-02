from app.models import DraftAnswer, Plan


class SolverAgent:
    """Produce a deterministic draft from a structured plan."""

    name = "solver"

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
            raise ValueError("revision_number cannot be negative")

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

        content_lines = [
            f"Draft answer for: {cleaned_question}",
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

        if cleaned_instructions:
            content_lines.extend(
                [
                    "",
                    "Applied revision instructions:",
                ]
            )

            for instruction in cleaned_instructions:
                content_lines.append(
                    f"- {instruction}"
                )

        return DraftAnswer(
            content="\n".join(content_lines),
            reasoning_steps=reasoning_steps,
            requested_tools=list(plan.required_tools),
            applied_revision_instructions=cleaned_instructions,
            revision_number=revision_number,
        )