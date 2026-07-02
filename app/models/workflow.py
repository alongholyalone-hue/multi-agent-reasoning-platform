from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AgentName(str, Enum):
    """Names of the agents participating in the workflow."""

    PLANNER = "planner"
    SOLVER = "solver"
    REVIEWER = "reviewer"
    FINALIZER = "finalizer"


class Plan(BaseModel):
    """A structured plan created by the planner agent."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    objective: str = Field(
        min_length=1,
        description="The main objective of the task.",
    )

    steps: list[str] = Field(
        min_length=1,
        description="Ordered steps required to complete the task.",
    )

    required_tools: list[str] = Field(
        default_factory=list,
        description="Tools the solver may need to use.",
    )


class DraftAnswer(BaseModel):
    """A structured draft produced by the solver agent."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    content: str = Field(
        min_length=1,
        description="The solver's current draft answer.",
    )

    reasoning_steps: list[str] = Field(
        min_length=1,
        description="The reasoning steps used to produce the draft.",
    )

    requested_tools: list[str] = Field(
        default_factory=list,
        description="Tools requested by the plan.",
    )

    applied_revision_instructions: list[str] = Field(
        default_factory=list,
        description="Reviewer instructions applied to this draft.",
    )

    revision_number: int = Field(
        default=0,
        ge=0,
        description="The revision number of this draft.",
    )


class ReviewResult(BaseModel):
    """The reviewer agent's evaluation of a draft answer."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    approved: bool

    issues: list[str] = Field(
        default_factory=list,
        description="Problems found in the draft answer.",
    )

    revision_instructions: list[str] = Field(
        default_factory=list,
        description="Instructions for improving the draft.",
    )


class AgentEvent(BaseModel):
    """One traceable event produced during workflow execution."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    sequence: int = Field(
        ge=1,
        description="The event's position in the workflow trace.",
    )

    agent_name: AgentName

    action: str = Field(
        min_length=1,
        description="The action performed by the agent.",
    )

    input_summary: str = Field(
        min_length=1,
        description="A short summary of the agent's input.",
    )

    output_summary: str = Field(
        min_length=1,
        description="A short summary of the agent's output.",
    )

    success: bool = True


class TaskResult(BaseModel):
    """The complete result returned by the multi-agent workflow."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        min_length=1,
        description="Unique identifier for the workflow execution.",
    )

    final_answer: str = Field(
        min_length=1,
        description="The final answer returned to the user.",
    )

    plan: Plan
    review: ReviewResult
    events: list[AgentEvent]

    revision_count: int = Field(
        ge=0,
        description="Number of revision cycles performed.",
    )

    completed: bool