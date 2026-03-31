"""Metric computation for the evaluation pipeline."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute evaluation metrics from a list of per-question results.

    Metrics:
        response_rate: fraction of queries that returned at least one obligation.
        average_confidence: mean confidence score across successful responses.
        citation_rate: fraction of obligations that have at least one citation.
        average_latency: mean response time in seconds.
        section_coverage: set of unique section types that appear in citations.
        total_obligations: total number of obligations extracted across all queries.
        error_count: number of queries that failed.
    """
    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}

    successful = [r for r in results if r.get("success")]
    error_count = total - len(successful)

    # Response rate: queries where at least one obligation was returned
    responses_with_obligations = 0
    confidence_scores: list[float] = []
    total_obligations = 0
    obligations_with_citations = 0
    section_types_seen: set[str] = set()
    latencies: list[float] = []

    for r in results:
        latency = r.get("latency_seconds")
        if latency is not None:
            latencies.append(latency)

        result = r.get("result")
        if result is None:
            continue

        obligations = result.get("obligations", [])
        if len(obligations) > 0:
            responses_with_obligations += 1

        confidence = result.get("confidence")
        if confidence is not None:
            confidence_scores.append(confidence)

        for obligation in obligations:
            total_obligations += 1
            citations = obligation.get("citations", [])
            if len(citations) > 0:
                obligations_with_citations += 1
            for citation in citations:
                section = citation.get("section_type")
                if section:
                    section_types_seen.add(section)

    response_rate = responses_with_obligations / total if total > 0 else 0.0
    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    )
    citation_rate = (
        obligations_with_citations / total_obligations if total_obligations > 0 else 0.0
    )
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "total_questions": total,
        "successful_queries": len(successful),
        "error_count": error_count,
        "response_rate": round(response_rate, 4),
        "average_confidence": round(avg_confidence, 4),
        "citation_rate": round(citation_rate, 4),
        "total_obligations": total_obligations,
        "section_types_seen": sorted(section_types_seen),
        "average_latency_seconds": round(avg_latency, 2),
    }


def compute_ragas_metrics(results: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute LLM-as-judge metrics using ragas.

    Filters results to those with success=True, non-None contexts, and
    non-empty ground_truth. Returns an empty dict if ragas is not installed
    or no valid samples exist.
    """
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            Faithfulness,
        )
    except ImportError:
        logger.warning("ragas not installed — skipping LLM-judge metrics")
        return {}

    # Filter to valid samples only
    valid = [
        r
        for r in results
        if r.get("success")
        and r.get("contexts") is not None
        and r.get("ground_truth")
    ]

    if not valid:
        logger.warning("No valid samples for ragas (need success + contexts + ground_truth)")
        return {}

    n_calls = 3 * len(valid)  # Faithfulness + AnswerRelevancy + AnswerCorrectness
    logger.info(
        "ragas will make ~%d LLM judge calls for %d samples "
        "(estimated cost: $%.2f–$%.2f depending on model)",
        n_calls,
        len(valid),
        n_calls * 0.003,  # low estimate (~$0.003/call for GPT-4o-mini)
        n_calls * 0.03,  # high estimate (~$0.03/call for Claude Sonnet)
    )

    # Build ragas LLM judge — tries Anthropic first, falls back to OpenAI
    import os

    ragas_llm = None
    try:
        anthropic_key = os.environ.get(
            "SEC_RAG_ANTHROPIC_API_KEY"
        ) or os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if anthropic_key:
            from anthropic import Anthropic

            anthropic_client = Anthropic(api_key=anthropic_key)
            ragas_llm = llm_factory(
                "claude-sonnet-4-20250514", provider="anthropic", client=anthropic_client
            )
        elif openai_key:
            from openai import OpenAI

            openai_client = OpenAI(api_key=openai_key)
            ragas_llm = llm_factory("gpt-4o-mini", provider="openai", client=openai_client)
        else:
            logger.warning(
                "No LLM API key for ragas judge "
                "(set SEC_RAG_ANTHROPIC_API_KEY or OPENAI_API_KEY)"
            )
            return {}
    except Exception:
        logger.exception("Failed to create ragas LLM judge")
        return {}

    # HuggingFace embeddings for AnswerRelevancy (no API key needed)
    ragas_embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    logger.info("Building ragas dataset from %d valid samples", len(valid))
    samples = []
    for r in valid:
        summary = ""
        result_data = r.get("result")
        if result_data:
            summary = result_data.get("summary", "")
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r["contexts"],
                response=summary,
                reference=r["ground_truth"],
            )
        )

    dataset = EvaluationDataset(samples=samples)  # type: ignore[arg-type]
    metrics_to_run = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        AnswerCorrectness(llm=ragas_llm),
    ]

    try:
        logger.info("Running ragas evaluate (this calls the LLM judge)...")
        eval_result = evaluate(dataset=dataset, metrics=metrics_to_run)  # type: ignore[arg-type]
        # Use to_pandas() public API instead of private _repr_dict (m10)
        df = eval_result.to_pandas()  # type: ignore[union-attr]
        scores: dict[str, float | None] = {}
        for metric_name in ["faithfulness", "answer_relevancy", "answer_correctness"]:
            if metric_name in df.columns:
                scores[metric_name] = float(df[metric_name].mean())
            else:
                scores[metric_name] = None
        return scores
    except Exception:
        logger.exception("ragas evaluate failed")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "answer_correctness": None,
        }
