from fastapi import FastAPI


app = FastAPI(
    title="Multi-Agent Reasoning Platform",
    description=(
        "A traceable multi-agent platform for planning, solving, "
        "reviewing, and revising technical problems."
    ),
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the current API health status."""

    return {
        "status": "ok",
        "service": "multi-agent-reasoning-platform",
    }
