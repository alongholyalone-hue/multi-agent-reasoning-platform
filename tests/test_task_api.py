import pytest
from fastapi.testclient import TestClient

from app.api.tasks import get_orchestrator
from app.main import app


@pytest.fixture(autouse=True)
def use_scaffold_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Keep API unit tests deterministic regardless of shell configuration.
    """

    monkeypatch.setenv(
        "MODEL_PROVIDER",
        "scaffold",
    )

    get_orchestrator.cache_clear()

    yield

    get_orchestrator.cache_clear()


client = TestClient(app)


def test_solve_task_returns_complete_workflow() -> None:
    response = client.post(
        "/tasks/solve",
        json={
            "question": (
                "Why does orbital velocity decrease "
                "as radius increases?"
            ),
            "max_revisions": 1,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["completed"] is True
    assert data["task_id"]
    assert data["final_answer"]
    assert data["revision_count"] == 0
    assert data["review"]["approved"] is True
    assert len(data["plan"]["steps"]) == 4


def test_solve_task_returns_agent_trace_in_order() -> None:
    response = client.post(
        "/tasks/solve",
        json={
            "question": "Explain gravitational acceleration.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    agent_names = [
        event["agent_name"]
        for event in data["events"]
    ]

    assert agent_names == [
        "planner",
        "solver",
        "reviewer",
        "finalizer",
    ]

    sequences = [
        event["sequence"]
        for event in data["events"]
    ]

    assert sequences == [1, 2, 3, 4]


def test_solve_task_detects_calculator_requirement() -> None:
    response = client.post(
        "/tasks/solve",
        json={
            "question": (
                "Calculate the orbital speed at "
                "a radius of 7000 km."
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["plan"]["required_tools"] == [
        "calculator"
    ]


def test_solve_task_rejects_question_that_is_too_short() -> None:
    response = client.post(
        "/tasks/solve",
        json={
            "question": "?",
        },
    )

    assert response.status_code == 422


def test_solve_task_rejects_excessive_revision_limit() -> None:
    response = client.post(
        "/tasks/solve",
        json={
            "question": "Explain orbital velocity.",
            "max_revisions": 10,
        },
    )

    assert response.status_code == 422


def test_solve_task_rejects_unknown_request_fields() -> None:
    response = client.post(
        "/tasks/solve",
        json={
            "question": "Explain orbital velocity.",
            "unknown_setting": True,
        },
    )

    assert response.status_code == 422