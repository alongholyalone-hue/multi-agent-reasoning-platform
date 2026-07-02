from app.models import Plan


CALCULATION_KEYWORDS = (
    "calculate",
    "compute",
    "evaluate",
    "solve for",
    "how many",
    "how much",
    "what is the value",
)


class PlannerAgent:
    """Create a structured plan for a submitted technical question."""

    name = "planner"

    def run(self, question: str) -> Plan:
        """Return a deterministic plan for solving the question."""

        cleaned_question = " ".join(question.split())

        if not cleaned_question:
            raise ValueError("question cannot be empty")

        required_tools: list[str] = []

        if self._requires_calculator(cleaned_question):
            required_tools.append("calculator")

        return Plan(
            objective=(
                "Produce a clear and verifiable answer to: "
                f"{cleaned_question}"
            ),
            steps=[
                "Identify the main task and relevant concepts.",
                "Break the problem into logical reasoning steps.",
                "Develop a complete draft answer.",
                "Check the answer for accuracy and clarity.",
            ],
            required_tools=required_tools,
        )

    @staticmethod
    def _requires_calculator(question: str) -> bool:
        """Determine whether the question may require calculation."""

        lowered_question = question.lower()

        contains_number = any(
            character.isdigit()
            for character in question
        )

        contains_calculation_keyword = any(
            keyword in lowered_question
            for keyword in CALCULATION_KEYWORDS
        )

        return (
            contains_number
            or contains_calculation_keyword
        )