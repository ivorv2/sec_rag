"""Reporting and persistence for the evaluation pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "eval/eval_results.json"


def print_report(
    metrics: dict[str, Any],
    results: list[dict[str, Any]],
    ragas_metrics: dict[str, Any] | None = None,
) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 70)
    print("SEC RAG EVALUATION REPORT")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 45)
    print(f"{'Total questions':<30} {metrics['total_questions']:>15}")
    print(f"{'Successful queries':<30} {metrics['successful_queries']:>15}")
    print(f"{'Errors':<30} {metrics['error_count']:>15}")
    print(f"{'Response rate':<30} {metrics['response_rate']:>14.1%}")
    print(f"{'Average confidence':<30} {metrics['average_confidence']:>15.4f}")
    print(f"{'Citation rate':<30} {metrics['citation_rate']:>14.1%}")
    print(f"{'Total obligations found':<30} {metrics['total_obligations']:>15}")
    print(f"{'Average latency (s)':<30} {metrics['average_latency_seconds']:>15.2f}")

    sections = metrics.get("section_types_seen", [])
    print(f"\n{'Section types in citations':<30} ({len(sections)} types)")
    for section in sections:
        print(f"  - {section}")

    # ragas metrics
    if ragas_metrics:
        print("\n" + "-" * 45)
        print("LLM-AS-JUDGE METRICS (ragas)")
        print("-" * 45)
        for name, value in ragas_metrics.items():
            if value is not None:
                print(f"{'  ' + name:<30} {value:>15.4f}")
            else:
                print(f"{'  ' + name:<30} {'FAILED':>15}")

    # Per-question summary
    print("\n" + "-" * 70)
    print("PER-QUESTION SUMMARY")
    print("-" * 70)

    for i, r in enumerate(results):
        question = r.get("question", f"Q{i + 1}")
        success = r.get("success", False)
        latency = r.get("latency_seconds", 0)
        result = r.get("result")

        status = "OK" if success else "FAIL"
        n_obligations = len(result.get("obligations", [])) if result else 0
        confidence = result.get("confidence", 0) if result else 0

        # Truncate question for display
        q_display = question[:55] + "..." if len(question) > 58 else question
        print(
            f"  [{status:>4}] {q_display:<58} "
            f"oblig={n_obligations} conf={confidence:.2f} {latency:.1f}s"
        )

    # Errors detail
    errors = [r for r in results if not r.get("success")]
    if errors:
        print(f"\n{'ERRORS':}")
        for r in errors:
            q = r.get("question", "?")[:60]
            err = r.get("error", "unknown")
            print(f"  - {q}: {err}")

    print("\n" + "=" * 70)


def save_results(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_path: str,
    ragas_metrics: dict[str, Any] | None = None,
) -> None:
    """Save the full evaluation results and metrics to a JSON file."""
    output: dict[str, Any] = {
        "metrics": metrics,
        "per_question_results": results,
    }
    if ragas_metrics:
        output["ragas_metrics"] = ragas_metrics

    filepath = Path(output_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", filepath.resolve())
