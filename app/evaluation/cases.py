from app.evaluation.runner import EvaluationCase


DEFAULT_EVALUATION_CASES = [
    EvaluationCase(
        case_id="orbital-velocity",
        question=(
            "Why does orbital velocity decrease "
            "as orbital radius increases?"
        ),
    ),
    EvaluationCase(
        case_id="electrical-resistance",
        question=(
            "Why does increasing electrical resistance "
            "decrease current when voltage remains constant?"
        ),
    ),
    EvaluationCase(
        case_id="machine-learning-types",
        question=(
            "What is the difference between supervised "
            "and unsupervised machine learning?"
        ),
    ),
    EvaluationCase(
        case_id="overfitting",
        question=(
            "Why can overfitting reduce a machine learning "
            "model's performance on unseen data?"
        ),
    ),
    EvaluationCase(
        case_id="binary-search",
        question=(
            "Explain how binary search reduces the search "
            "space when finding an item in a sorted list."
        ),
    ),
]
