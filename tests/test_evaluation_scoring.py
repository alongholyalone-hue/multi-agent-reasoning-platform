import pytest

from app.evaluation import (
    AnswerRubric,
    ConceptRequirement,
    score_answer,
)


def create_binary_search_rubric() -> AnswerRubric:
    return AnswerRubric(
        required_concepts=(
            ConceptRequirement(
                label="compare with the middle element",
                alternatives=(
                    "compare with the middle element",
                    "compare the target to the midpoint",
                    "compares the target to the midpoint",
                ),
            ),
            ConceptRequirement(
                label="discard half of the search space",
                alternatives=(
                    "discard half",
                    "eliminate half",
                    "eliminates half",
                    "reduce the search interval by half",
                ),
            ),
        ),
        forbidden_claims=(
            "examines every element",
            "explores every possible position",
        ),
    )


def test_score_answer_passes_when_requirements_are_met() -> None:
    score = score_answer(
        answer=(
            "Binary search compares the target to the midpoint. "
            "It then eliminates half of the remaining interval."
        ),
        rubric=create_binary_search_rubric(),
    )

    assert score.passed is True
    assert score.missing_concepts == ()
    assert score.matched_forbidden_claims == ()


def test_score_answer_reports_missing_concepts() -> None:
    score = score_answer(
        answer=(
            "Binary search is used on a sorted list and is "
            "usually faster than linear search."
        ),
        rubric=create_binary_search_rubric(),
    )

    assert score.passed is False

    assert score.missing_concepts == (
        "compare with the middle element",
        "discard half of the search space",
    )


def test_score_answer_reports_forbidden_claims() -> None:
    score = score_answer(
        answer=(
            "Binary search compares the target to the midpoint "
            "and eliminates half of the interval, but it also "
            "explores every possible position."
        ),
        rubric=create_binary_search_rubric(),
    )

    assert score.passed is False

    assert score.matched_forbidden_claims == (
        "explores every possible position",
    )


def test_concept_requirement_rejects_blank_alternative() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "concept alternatives cannot contain blank text"
        ),
    ):
        ConceptRequirement(
            label="required idea",
            alternatives=("valid phrase", "   "),
        )
