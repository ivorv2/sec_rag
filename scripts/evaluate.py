"""Evaluation script for the SEC RAG pipeline.

Loads a golden set of question-answer pairs, queries the API, and computes
retrieval and generation quality metrics.

Usage:
    # Start the API first:
    uvicorn sec_rag.api.main:app --host 0.0.0.0 --port 8000

    # Then run evaluation:
    python scripts/evaluate.py
    python scripts/evaluate.py --api-url http://localhost:8000
    python scripts/evaluate.py --golden-set eval/golden_set.json
    python scripts/evaluate.py --with-ragas  # adds LLM-as-judge metrics
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

# Ensure sibling modules are importable when run from project root (m09)
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from eval_api import (  # noqa: E402
    DEFAULT_API_URL,
    DEFAULT_GOLDEN_SET,
    check_api_health,
    load_golden_set,
    query_api,
)
from eval_metrics import compute_metrics, compute_ragas_metrics  # noqa: E402
from eval_report import DEFAULT_OUTPUT, print_report, save_results  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_evaluation(
    api_url: str,
    golden_set_path: str,
    output_path: str,
    with_ragas: bool = False,
) -> dict[str, Any]:
    """Execute the full evaluation pipeline.

    Returns the metrics dict.
    """
    golden_set = load_golden_set(golden_set_path)

    if not check_api_health(api_url):
        logger.error("API is not healthy. Aborting evaluation.")
        sys.exit(1)

    results: list[dict[str, Any]] = []
    for i, item in enumerate(golden_set):
        question = item["question"]
        logger.info("[%d/%d] Querying: %s", i + 1, len(golden_set), question[:80])

        response = query_api(api_url, question, include_contexts=with_ragas)
        # Attach metadata from the golden set for later analysis
        response["question"] = question
        response["ground_truth"] = item.get("ground_truth", "")
        response["context_keywords"] = item.get("context_keywords", [])
        results.append(response)

    metrics = compute_metrics(results)

    ragas_metrics = None
    if with_ragas:
        ragas_metrics = compute_ragas_metrics(results)

    print_report(metrics, results, ragas_metrics)
    save_results(results, metrics, output_path, ragas_metrics)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the SEC RAG pipeline against a golden set"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"Base URL of the running API (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--golden-set",
        type=str,
        default=DEFAULT_GOLDEN_SET,
        help=f"Path to golden set JSON (default: {DEFAULT_GOLDEN_SET})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to save evaluation results (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        default=False,
        help="Run LLM-as-judge metrics (faithfulness, relevancy, correctness) via ragas",
    )
    args = parser.parse_args()

    run_evaluation(
        api_url=args.api_url,
        golden_set_path=args.golden_set,
        output_path=args.output,
        with_ragas=args.with_ragas,
    )


if __name__ == "__main__":
    main()
