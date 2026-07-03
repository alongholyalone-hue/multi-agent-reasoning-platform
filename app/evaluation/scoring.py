import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptRequirement:
    """A required concept and acceptable ways to express it."""

    label: str
    alternatives: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError(
                "concept label cannot be empty"
            )

        if not self.alternatives:
            raise ValueError(
                "concept alternatives cannot be empty"
            )

        if any(
            not alternative.strip()
            for alternative in self.alternatives
        ):
            raise ValueError(
                "concept alternatives cannot contain blank text"
            )


@dataclass(frozen=True)
class AnswerRubric:
    """Deterministic criteria for one benchmark answer."""

    required_concepts: tuple[
        ConceptRequirement,
        ...,
    ] = ()

    forbidden_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not claim.strip()
            for claim in self.forbidden_claims
        ):
            raise ValueError(
                "forbidden claims cannot contain blank text"
            )


@dataclass(frozen=True)
class BenchmarkScore:
    """Ground-truth-oriented result for one answer."""

    passed: bool
    missing_concepts: tuple[str, ...]
    matched_forbidden_claims: tuple[str, ...]


def score_answer(
    *,
    answer: str,
    rubric: AnswerRubric,
) -> BenchmarkScore:
    """Score an answer against deterministic benchmark criteria."""

    normalized_answer = _normalize_text(answer)

    missing_concepts = tuple(
        requirement.label
        for requirement in rubric.required_concepts
        if not any(
            _contains_phrase(
                normalized_answer,
                alternative,
            )
            for alternative in requirement.alternatives
        )
    )

    matched_forbidden_claims = tuple(
        claim
        for claim in rubric.forbidden_claims
        if _contains_phrase(
            normalized_answer,
            claim,
        )
    )

    return BenchmarkScore(
        passed=(
            not missing_concepts
            and not matched_forbidden_claims
        ),
        missing_concepts=missing_concepts,
        matched_forbidden_claims=(
            matched_forbidden_claims
        ),
    )


def _contains_phrase(
    normalized_text: str,
    phrase: str,
) -> bool:
    """Check for a complete normalized phrase."""

    normalized_phrase = _normalize_text(phrase)

    if not normalized_phrase:
        return False

    padded_text = f" {normalized_text} "
    padded_phrase = f" {normalized_phrase} "

    return padded_phrase in padded_text


def _normalize_text(text: str) -> str:
    """Normalize case, punctuation, and whitespace."""

    tokens = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    return " ".join(tokens)
