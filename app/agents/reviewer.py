import re

from app.models import DraftAnswer, ReviewResult


MIN_WORD_COUNT = 35
MIN_SENTENCE_COUNT = 3

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

        cleaned_content = draft.content.strip()
        lowered_content = cleaned_content.lower()

        word_count = len(cleaned_content.split())

        if word_count < MIN_WORD_COUNT:
            issues.append(
                "The draft answer is too brief."
            )
            revision_instructions.append(
                "Expand the answer to at least three complete "
                "sentences with a clear explanation."
            )

        sentence_count = len(
            re.findall(
                r"[.!?](?:\s|$)",
                cleaned_content,
            )
        )

        if sentence_count < MIN_SENTENCE_COUNT:
            issues.append(
                "The draft contains too few complete sentences."
            )
            revision_instructions.append(
                "Write at least three complete explanatory sentences."
            )

        if len(draft.reasoning_steps) < 3:
            issues.append(
                "The draft contains too few reasoning steps."
            )
            revision_instructions.append(
                "Include at least three logical reasoning steps."
            )

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