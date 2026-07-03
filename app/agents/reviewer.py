import json
import re
from typing import Any

from app.models import DraftAnswer, ReviewResult
from app.providers import ModelProvider


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
    """Evaluate draft answers using deterministic and semantic checks."""

    name = "reviewer"

    # Keeps subclasses compatible when they do not call super().__init__().
    provider: ModelProvider | None = None

    def __init__(
        self,
        provider: ModelProvider | None = None,
    ) -> None:
        self.provider = provider

    def run(
        self,
        question: str,
        draft: DraftAnswer,
    ) -> ReviewResult:
        """Review a draft and return issues and revision instructions."""

        cleaned_question = " ".join(question.split())

        if not cleaned_question:
            raise ValueError("question cannot be empty")

        cleaned_content = draft.content.strip()
        lowered_content = cleaned_content.lower()

        issues: list[str] = []
        revision_instructions: list[str] = []

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

        # Avoid spending model time reviewing an answer that already
        # failed basic deterministic quality checks.
        if issues or self.provider is None:
            return ReviewResult(
                approved=not issues,
                issues=issues,
                revision_instructions=revision_instructions,
            )

        return self._run_semantic_review(
            question=cleaned_question,
            content=cleaned_content,
        )

    def _run_semantic_review(
        self,
        *,
        question: str,
        content: str,
    ) -> ReviewResult:
        """Ask the configured model to review factual and logical quality."""

        if self.provider is None:
            raise RuntimeError(
                "semantic review requires a model provider"
            )

        response = self.provider.generate(
            system_prompt=self._build_semantic_system_prompt(),
            user_prompt=self._build_semantic_user_prompt(
                question=question,
                content=content,
            ),
        )

        try:
            payload = self._parse_json_object(response)
            return self._validate_semantic_payload(payload)
        except (ValueError, json.JSONDecodeError):
            return ReviewResult(
                approved=False,
                issues=[
                    "The semantic reviewer returned an invalid response."
                ],
                revision_instructions=[
                    "Rewrite the answer with clear factual reasoning "
                    "and no internal contradictions."
                ],
            )

    @staticmethod
    def _build_semantic_system_prompt() -> str:
        return (
            "You are an independent answer reviewer. Evaluate whether "
            "the draft accurately and directly answers the question. "
            "Check factual correctness, causal direction, internal "
            "contradictions, unsupported claims, and logical coherence. "
            "Do not approve an answer merely because it is long. "
            "Return only valid JSON with exactly these keys: "
            '"approved", "issues", and "revision_instructions". '
            '"approved" must be a boolean. The other two values must '
            "be arrays of strings."
        )

    @staticmethod
    def _build_semantic_user_prompt(
        *,
        question: str,
        content: str,
    ) -> str:
        return (
            f"Question:\n{question}\n\n"
            f"Draft answer:\n{content}\n\n"
            "Review the draft now. Return only the JSON object."
        )

    @staticmethod
    def _parse_json_object(
        response: str,
    ) -> dict[str, Any]:
        """Extract a JSON object even when surrounded by extra text."""

        cleaned_response = response.strip()

        start_index = cleaned_response.find("{")
        end_index = cleaned_response.rfind("}")

        if (
            start_index == -1
            or end_index == -1
            or end_index < start_index
        ):
            raise ValueError(
                "semantic review did not contain a JSON object"
            )

        payload = json.loads(
            cleaned_response[start_index:end_index + 1]
        )

        if not isinstance(payload, dict):
            raise ValueError(
                "semantic review JSON must be an object"
            )

        return payload

    @staticmethod
    def _validate_semantic_payload(
        payload: dict[str, Any],
    ) -> ReviewResult:
        """Validate and normalize the semantic review response."""

        approved = payload.get("approved")
        issues = payload.get("issues")
        revision_instructions = payload.get(
            "revision_instructions"
        )

        if not isinstance(approved, bool):
            raise ValueError(
                "approved must be a boolean"
            )

        if not ReviewerAgent._is_string_list(issues):
            raise ValueError(
                "issues must be a list of strings"
            )

        if not ReviewerAgent._is_string_list(
            revision_instructions
        ):
            raise ValueError(
                "revision_instructions must be a list of strings"
            )

        cleaned_issues = [
            item.strip()
            for item in issues
            if item.strip()
        ]

        cleaned_instructions = [
            item.strip()
            for item in revision_instructions
            if item.strip()
        ]

        # An answer cannot be approved while unresolved issues remain.
        normalized_approved = (
            approved and not cleaned_issues
        )

        if not normalized_approved and not cleaned_issues:
            cleaned_issues.append(
                "The semantic reviewer rejected the draft."
            )

        if (
            not normalized_approved
            and not cleaned_instructions
        ):
            cleaned_instructions.append(
                "Correct the factual or logical problems in the answer."
            )

        return ReviewResult(
            approved=normalized_approved,
            issues=cleaned_issues,
            revision_instructions=cleaned_instructions,
        )

    @staticmethod
    def _is_string_list(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(
                isinstance(item, str)
                for item in value
            )
        )