from app.models import DraftAnswer, ReviewResult


PLACEHOLDER_MARKERS = (
    "todo",
    "tbd",
    "placeholder",
    "insert answer",
    "finish later",
)


class ReviewerAgent:
    """Evaluate a draft answer using deterministic quality checks."""

    name = "reviewer"

    def run(
        self,
        question: str,
        draft: DraftAnswer,
    ) -> ReviewResult:
        """Review a draft and return issues and revision instructions."""

        cleaned_question = " ".join(question.split())

        if not cleaned_question:
            raise ValueError("question cannot be empty")

        issues: list[str] = []
        revision_instructions: list[str] = []

        word_count = len(draft.content.split())

        if word_count < 20:
            issues.append(
                "The draft answer is too brief."
            )
            revision_instructions.append(
                "Expand the answer with a clearer explanation."
            )

        if len(draft.reasoning_steps) < 3:
            issues.append(
                "The draft contains too few reasoning steps."
            )
            revision_instructions.append(
                "Include at least three logical reasoning steps."
            )

        lowered_content = draft.content.lower()

        if any(
            marker in lowered_content
            for marker in PLACEHOLDER_MARKERS
        ):
            issues.append(
                "The draft contains unfinished placeholder text."
            )
            revision_instructions.append(
                "Replace all placeholder text with complete content."
            )

        missing_tools = [
            tool
            for tool in draft.requested_tools
            if tool.lower() not in lowered_content
        ]

        if missing_tools:
            issues.append(
                "The draft does not acknowledge all requested tools."
            )
            revision_instructions.append(
                "Explain how the following tools should be used: "
                + ", ".join(missing_tools)
                + "."
            )

        return ReviewResult(
            approved=not issues,
            issues=issues,
            revision_instructions=revision_instructions,
        )