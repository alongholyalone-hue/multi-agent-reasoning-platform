from pydantic import BaseModel, ConfigDict, Field


class TaskRequest(BaseModel):
    """A user request submitted to the multi-agent workflow."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    question: str = Field(
        min_length=3,
        max_length=2000,
        description="The technical question for the agents to solve.",
    )

    max_revisions: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Maximum number of solver-review revision cycles.",
    )