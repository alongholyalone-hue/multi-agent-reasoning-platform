from app.evaluation.runner import EvaluationCase
from app.evaluation.scoring import (
    AnswerRubric,
    ConceptRequirement,
)


DEFAULT_EVALUATION_CASES = [
    EvaluationCase(
        case_id="orbital-velocity",
        question=(
            "Why does orbital velocity decrease "
            "as orbital radius increases?"
        ),
        rubric=AnswerRubric(
            required_concepts=(
                ConceptRequirement(
                    label="orbital velocity equation",
                    alternatives=(
                        "v sqrt gm r",
                        "inverse square root of radius",
                        "inversely proportional to the square root",
                    ),
                ),
                ConceptRequirement(
                    label="larger radius means lower velocity",
                    alternatives=(
                        "radius increases velocity decreases",
                        "larger radius lowers velocity",
                        "farther from the same central mass moves more slowly",
                    ),
                ),
            ),
            forbidden_claims=(
                "gravity becomes stronger farther",
                "velocity must be increased",
                "orbital velocity increases as orbital radius increases",
            ),
        ),
    ),
    EvaluationCase(
        case_id="electrical-resistance",
        question=(
            "Why does increasing electrical resistance "
            "decrease current when voltage remains constant?"
        ),
        rubric=AnswerRubric(
            required_concepts=(
                ConceptRequirement(
                    label="Ohm's law",
                    alternatives=(
                        "ohm s law",
                        "current equals voltage divided by resistance",
                        "i v r",
                    ),
                ),
                ConceptRequirement(
                    label="voltage remains constant",
                    alternatives=(
                        "voltage remains constant",
                        "voltage is held constant",
                        "at constant voltage",
                    ),
                ),
                ConceptRequirement(
                    label="higher resistance means lower current",
                    alternatives=(
                        "increasing resistance decreases current",
                        "as resistance increases current decreases",
                        "current is inversely proportional to resistance",
                        "larger resistance produces a smaller current",
                    ),
                ),
            ),
            forbidden_claims=(
                "resistance increases current",
                "product r times v",
                "r times v increases",
            ),
        ),
    ),
    EvaluationCase(
        case_id="machine-learning-types",
        question=(
            "What is the difference between supervised "
            "and unsupervised machine learning?"
        ),
        rubric=AnswerRubric(
            required_concepts=(
                ConceptRequirement(
                    label="supervised learning uses labeled data",
                    alternatives=(
                        "supervised learning uses labeled data",
                        "supervised learning uses labelled data",
                        "labeled training data",
                        "labelled training data",
                    ),
                ),
                ConceptRequirement(
                    label="unsupervised learning uses unlabeled data",
                    alternatives=(
                        "unsupervised learning uses unlabeled data",
                        "unsupervised learning uses unlabelled data",
                        "without explicit labeling",
                        "without explicit labelling",
                        "without any explicit labeling",
                    ),
                ),
                ConceptRequirement(
                    label="unsupervised learning discovers patterns",
                    alternatives=(
                        "discovering hidden structures or patterns",
                        "discovers hidden patterns",
                        "finds patterns or clusters",
                        "look for clusters associations or distributions",
                    ),
                ),
            ),
        ),
    ),
    EvaluationCase(
        case_id="overfitting",
        question=(
            "Why can overfitting reduce a machine learning "
            "model's performance on unseen data?"
        ),
        rubric=AnswerRubric(
            required_concepts=(
                ConceptRequirement(
                    label="fits noise or training-specific details",
                    alternatives=(
                        "capturing noise and irrelevant details",
                        "fits noise",
                        "memorizes the training data",
                        "learns the training data too well",
                    ),
                ),
                ConceptRequirement(
                    label="poor generalization to unseen data",
                    alternatives=(
                        "poor generalization",
                        "does not generalize",
                        "do not generalize",
                        "inability to generalize",
                        "fails to generalize",
                    ),
                ),
            ),
            forbidden_claims=(
                "high variance the tendency to have small errors",
                "low variance the ability to generalize well",
            ),
        ),
    ),
    EvaluationCase(
        case_id="binary-search",
        question=(
            "Explain how binary search reduces the search "
            "space when finding an item in a sorted list."
        ),
        rubric=AnswerRubric(
            required_concepts=(
                ConceptRequirement(
                    label="requires a sorted list",
                    alternatives=(
                        "sorted list",
                        "sorted array",
                    ),
                ),
                ConceptRequirement(
                    label="compare with the middle element",
                    alternatives=(
                        "compare with the middle element",
                        "compares with the middle element",
                        "compare the target to the midpoint",
                        "compares the target to the midpoint",
                        "middle value",
                    ),
                ),
                ConceptRequirement(
                    label="discard half of the search space",
                    alternatives=(
                        "discard half",
                        "eliminate half",
                        "eliminates half",
                        "halves the number of elements",
                        "dividing the search interval in half",
                        "reduce the search space by half",
                    ),
                ),
            ),
            forbidden_claims=(
                "explores every possible position",
                "every possible position is explored",
                "requires examining all n elements",
                "start with an unsorted list",
            ),
        ),
    ),
]
