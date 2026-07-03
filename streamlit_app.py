import os
from time import perf_counter
from typing import Any

import streamlit as st

from app.core import create_workflow_orchestrator
from app.models import TaskRequest


st.set_page_config(
    page_title="Multi-Agent Reasoning Platform",
    page_icon="🧠",
    layout="wide",
)


st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #8b949e;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .status-approved {
            padding: 0.75rem 1rem;
            border-radius: 0.6rem;
            background-color: rgba(46, 160, 67, 0.15);
            border: 1px solid rgba(46, 160, 67, 0.45);
            font-weight: 600;
        }

        .status-rejected {
            padding: 0.75rem 1rem;
            border-radius: 0.6rem;
            background-color: rgba(248, 81, 73, 0.15);
            border: 1px solid rgba(248, 81, 73, 0.45);
            font-weight: 600;
        }

        .agent-step {
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
            border-radius: 0.6rem;
            border: 1px solid rgba(139, 148, 158, 0.35);
            background-color: rgba(110, 118, 129, 0.08);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(139, 148, 158, 0.3);
            padding: 0.8rem;
            border-radius: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_orchestrator(provider_mode: str):
    """Create and cache one orchestrator for each provider mode."""

    return create_workflow_orchestrator(
        provider_mode=provider_mode
    )


def get_model_name(orchestrator: Any) -> str:
    """Return a readable name for the active provider."""

    provider = orchestrator.solver.provider

    if provider is None:
        return "Deterministic scaffold"

    return getattr(
        provider,
        "model_name",
        type(provider).__name__,
    )


def display_result(
    result: dict[str, Any],
    runtime_seconds: float,
    provider_mode: str,
    model_name: str,
) -> None:
    """Display one completed workflow result."""

    approved = result["review"]["approved"]

    if approved:
        st.markdown(
            '<div class="status-approved">'
            "✅ Answer approved by the Reviewer"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-rejected">'
            "⚠️ Answer rejected — safe refusal returned"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    metric_one, metric_two, metric_three, metric_four = (
        st.columns(4)
    )

    metric_one.metric(
        "Status",
        "Approved" if approved else "Rejected",
    )
    metric_two.metric(
        "Revisions",
        result["revision_count"],
    )
    metric_three.metric(
        "Runtime",
        f"{runtime_seconds:.2f}s",
    )
    metric_four.metric(
        "Events",
        len(result["events"]),
    )

    st.caption(
        f"Provider: `{provider_mode}` · "
        f"Model: `{model_name}` · "
        f"Task ID: `{result['task_id']}`"
    )

    (
        answer_tab,
        plan_tab,
        review_tab,
        trace_tab,
        raw_tab,
    ) = st.tabs(
        [
            "💬 Final Answer",
            "📋 Plan",
            "🔍 Review",
            "🧭 Execution Trace",
            "🧾 Raw Response",
        ]
    )

    with answer_tab:
        st.subheader("Final Answer")
        st.markdown(result["final_answer"])

        if not approved:
            st.info(
                "The system returned a safe refusal because the "
                "generated answer did not pass review."
            )

    with plan_tab:
        st.subheader("Objective")
        st.write(result["plan"]["objective"])

        st.subheader("Reasoning Plan")

        for number, step in enumerate(
            result["plan"]["steps"],
            start=1,
        ):
            st.markdown(f"**{number}.** {step}")

        required_tools = result["plan"]["required_tools"]

        if required_tools:
            st.subheader("Required Tools")
            for tool in required_tools:
                st.markdown(f"- {tool}")
        else:
            st.caption(
                "No external tools were requested."
            )

    with review_tab:
        st.subheader("Reviewer Decision")

        st.write(
            "**Approved:**",
            "Yes" if approved else "No",
        )

        issues = result["review"]["issues"]
        instructions = result["review"][
            "revision_instructions"
        ]

        st.subheader("Issues")

        if issues:
            for issue in issues:
                st.error(issue)
        else:
            st.success(
                "The Reviewer reported no unresolved issues."
            )

        st.subheader("Revision Instructions")

        if instructions:
            for instruction in instructions:
                st.warning(instruction)
        else:
            st.caption(
                "No revision instructions were returned."
            )

    with trace_tab:
        st.subheader("Agent Execution Trace")

        for event in result["events"]:
            icon = "✅" if event["success"] else "❌"
            agent = event["agent_name"].title()
            action = event["action"].replace(
                "_",
                " ",
            ).title()

            st.markdown(
                f"""
                <div class="agent-step">
                    <strong>
                        {icon} {event["sequence"]}.
                        {agent} — {action}
                    </strong>
                    <br>
                    <small>
                        Input: {event["input_summary"]}
                    </small>
                    <br>
                    <small>
                        Output: {event["output_summary"]}
                    </small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with raw_tab:
        st.subheader("Complete API-Compatible Response")
        st.json(result)


st.markdown(
    '<div class="main-title">'
    "🧠 Multi-Agent Reasoning Platform"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Plan → Solve → Review → Revise → Finalize"
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "Submit a focused technical question and inspect how "
    "the specialized agents process, evaluate, and finalize it."
)


with st.sidebar:
    st.header("Configuration")

    provider_options = [
        "scaffold",
        "causal",
        "huggingface",
    ]

    environment_provider = os.getenv(
        "MODEL_PROVIDER",
        "scaffold",
    ).lower()

    default_index = (
        provider_options.index(environment_provider)
        if environment_provider in provider_options
        else 0
    )

    provider_mode = st.selectbox(
        "Model provider",
        options=provider_options,
        index=default_index,
    )

    max_revisions = st.slider(
        "Maximum revisions",
        min_value=0,
        max_value=3,
        value=1,
    )

    if provider_mode == "scaffold":
        st.warning(
            "Scaffold mode tests the workflow but does not "
            "generate a genuine answer."
        )
    elif provider_mode == "causal":
        st.info(
            "Causal mode uses the configured local instruction model."
        )
    else:
        st.info(
            "Hugging Face mode uses the configured "
            "sequence-to-sequence model."
        )

    st.divider()

    st.markdown(
        "[View the project on GitHub]"
        "(https://github.com/alongholyalone-hue/"
        "multi-agent-reasoning-platform)"
    )

    st.caption("Portfolio release v1.0.0")


question = st.text_area(
    "Question",
    value=(
        "What is an API, and how does a client "
        "communicate with a server?"
    ),
    height=130,
    max_chars=2000,
    help=(
        "Focused, self-contained technical questions work best "
        "with small local models."
    ),
)

submitted = st.button(
    "Run Multi-Agent Workflow",
    type="primary",
    use_container_width=True,
)


if submitted:
    cleaned_question = question.strip()

    if len(cleaned_question) < 3:
        st.error(
            "Please enter a question containing at least "
            "three characters."
        )
    else:
        try:
            with st.status(
                "Loading the provider and running the agents...",
                expanded=True,
            ) as workflow_status:
                st.write(
                    f"Provider selected: `{provider_mode}`"
                )
                st.write(
                    "Creating the Planner, Solver, Reviewer, "
                    "and Finalizer workflow..."
                )

                orchestrator = get_orchestrator(
                    provider_mode
                )

                model_name = get_model_name(
                    orchestrator
                )

                st.write(
                    f"Running model: `{model_name}`"
                )

                start_time = perf_counter()

                task_result = orchestrator.run(
                    TaskRequest(
                        question=cleaned_question,
                        max_revisions=max_revisions,
                    )
                )

                runtime_seconds = (
                    perf_counter() - start_time
                )

                result_data = task_result.model_dump(
                    mode="json"
                )

                workflow_status.update(
                    label="Workflow completed",
                    state="complete",
                    expanded=False,
                )

            st.session_state["last_result"] = (
                result_data
            )
            st.session_state["last_runtime"] = (
                runtime_seconds
            )
            st.session_state["last_provider"] = (
                provider_mode
            )
            st.session_state["last_model"] = (
                model_name
            )

        except Exception as error:
            st.error(
                "The workflow could not be completed."
            )
            st.exception(error)


if "last_result" in st.session_state:
    st.divider()

    display_result(
        result=st.session_state["last_result"],
        runtime_seconds=st.session_state[
            "last_runtime"
        ],
        provider_mode=st.session_state[
            "last_provider"
        ],
        model_name=st.session_state[
            "last_model"
        ],
    )
else:
    st.divider()
    st.info(
        "Enter a question and run the workflow to see "
        "the final answer, plan, review, and execution trace."
    )