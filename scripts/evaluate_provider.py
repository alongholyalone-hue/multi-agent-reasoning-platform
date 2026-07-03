import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.core import create_workflow_orchestrator
from app.evaluation import (
    DEFAULT_EVALUATION_CASES,
    evaluate_cases,
    summarize_results,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a configured model provider "
            "across a small reasoning benchmark."
        )
    )

    parser.add_argument(
        "--provider",
        choices=[
            "scaffold",
            "huggingface",
            "causal",
        ],
        default="scaffold",
        help="Provider mode to evaluate.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to run.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation_results.json"
        ),
        help="Path for the JSON evaluation report.",
    )

    return parser


def main() -> None:
    """Run provider evaluation and save a JSON report."""

    parser = build_parser()
    arguments = parser.parse_args()

    if (
        arguments.limit is not None
        and arguments.limit <= 0
    ):
        parser.error(
            "--limit must be greater than zero"
        )

    cases = DEFAULT_EVALUATION_CASES

    if arguments.limit is not None:
        cases = cases[:arguments.limit]

    orchestrator = create_workflow_orchestrator(
        provider_mode=arguments.provider
    )

    provider = orchestrator.solver.provider

    results = evaluate_cases(
        orchestrator=orchestrator,
        cases=cases,
    )

    summary = summarize_results(results)

    payload = {
        "provider_mode": arguments.provider,
        "provider_class": (
            type(provider).__name__
            if provider is not None
            else "ScaffoldProvider"
        ),
        "model_name": getattr(
            provider,
            "model_name",
            None,
        ),
        "summary": asdict(summary),
        "results": [
            asdict(result)
            for result in results
        ],
    }

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Provider: {payload['provider_class']}"
    )
    print(
        f"Model: {payload['model_name']}"
    )
    print(
        f"Cases: {summary.case_count}"
    )
    print(
        f"Approved: "
        f"{summary.approved_count}/"
        f"{summary.case_count}"
    )
    print(
        f"Approval rate: "
        f"{summary.approval_rate:.1%}"
    )
    print(
        f"Average runtime: "
        f"{summary.average_runtime_seconds:.2f} seconds"
    )
    print(
        f"Total revisions: "
        f"{summary.total_revisions}"
    )
    print(
        f"Report: {arguments.output}"
    )


if __name__ == "__main__":
    main()
