from app.models import (
    DraftAnswer,
    FinalAnswer,
    ReviewResult,
)


class FinalizerAgent:
    """Convert a reviewed draft into a final response."""

    name = "finalizer"

    def run(
        self,
        question: str,
        draft: DraftAnswer,
        review: ReviewResult,
    ) -> FinalAnswer:
        """Return a finalized answer and its review status."""

        cleaned_question = " ".join(question.split())

        if not cleaned_question:
            raise ValueError("question cannot be empty")

        cleaned_draft = self._remove_draft_heading(
            draft.content
        )

        if review.approved:
            heading = (
                f"Final answer for: {cleaned_question}"
            )
            unresolved_issues: list[str] = []
        else:
            heading = (
                f"Best available answer for: "
                f"{cleaned_question}"
            )
            unresolved_issues = list(review.issues)

        return FinalAnswer(
            content=f"{heading}\n\n{cleaned_draft}",
            approved=review.approved,
            revision_number=draft.revision_number,
            unresolved_issues=unresolved_issues,
        )

    @staticmethod
    def _remove_draft_heading(content: str) -> str:
        """Remove the solver's temporary draft heading."""

        lines = content.splitlines()

        if (
            lines
            and lines[0]
            .strip()
            .lower()
            .startswith("draft answer for:")
        ):
            lines = lines[1:]

        while lines and not lines[0].strip():
            lines.pop(0)

        cleaned_content = "\n".join(lines).strip()

        if cleaned_content:
            return cleaned_content

        return content.strip()