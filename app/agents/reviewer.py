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

DIRECTIONAL_QUESTION_PATTERN = re.compile(
    r"why\s+does\s+"
    r"(?P<subject>.+?)\s+"
    r"(?P<subject_direction>increase|decrease)\s+"
    r"as\s+"
    r"(?P<condition>.+?)\s+"
    r"(?P<condition_direction>increases?|decreases?)"
    r"\??$",
    re.IGNORECASE,
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

        directional_conflict = (
            self._find_directional_conflict(
                question=cleaned_question,
                content=cleaned_content,
            )
        )

        if directional_conflict is not None:
            issues.append(
                "The draft contradicts the directional "
                "relationship stated in the question."
            )
            revision_instructions.append(
                "Remove statements that reverse or contradict "
                "the relationship described in the question."
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
            '"approved" must be a boolean. "issues" and '
            '"revision_instructions" must always be arrays of strings. '
            "Never return a single string instead of an array. "
            "For an approved answer, return exactly this structure: "
            '{"approved": true, "issues": [], '
            '"revision_instructions": []}.'
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
    def _find_directional_conflict(
        *,
        question: str,
        content: str,
    ) -> str | None:
        """
        Detect when the answer directly reverses the direction
        of the subject relationship stated in the question.
        """

        match = DIRECTIONAL_QUESTION_PATTERN.search(
            question.strip()
        )

        if match is None:
            return None

        subject = match.group("subject").strip()

        expected_direction = match.group(
            "subject_direction"
        ).lower()

        subject_tokens = re.findall(
            r"[a-z0-9]+",
            subject.lower(),
        )

        if not subject_tokens:
            return None

        subject_pattern = r"\s+".join(
            re.escape(token)
            for token in subject_tokens
        )

        if expected_direction == "decrease":
            expected_words = (
                r"decrease|decreases|decreased|decreasing"
            )
            conflicting_words = (
                r"increase|increases|increased|increasing"
            )
        else:
            expected_words = (
                r"increase|increases|increased|increasing"
            )
            conflicting_words = (
                r"decrease|decreases|decreased|decreasing"
            )

        expected_pattern = re.compile(
            rf"\b(?:{expected_words})\b",
            re.IGNORECASE,
        )

        relationship_pattern = re.compile(
            rf"\b{subject_pattern}\b"
            rf"(?P<middle>.{{0,80}}?)"
            rf"\b(?P<conflict>{conflicting_words})\b",
            re.IGNORECASE,
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            content.strip(),
        )

        for sentence in sentences:
            relationship_match = (
                relationship_pattern.search(sentence)
            )

            if relationship_match is None:
                continue

            middle_text = relationship_match.group(
                "middle"
            )

            # Do not reject an echoed relationship such as:
            # "orbital velocity decreases as radius increases."
            if expected_pattern.search(middle_text):
                continue

            preceding_text = middle_text[-30:]

            if ReviewerAgent._contains_negation(
                preceding_text
            ):
                continue

            return sentence.strip()

        return None

    @staticmethod
    def _contains_negation(text: str) -> bool:
        """Return whether nearby text negates a directional verb."""

        return bool(
            re.search(
                r"\b(?:not|never|cannot|can't|"
                r"doesn't|does\s+not)\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_string_list(value: Any) -> bool:
        """Return whether a value is a list containing only strings."""

        return (
            isinstance(value, list)
            and all(
                isinstance(item, str)
                for item in value
            )
        )