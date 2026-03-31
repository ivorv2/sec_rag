"""LangGraph node functions and conditional-edge helpers.

Every node takes ``QueryState`` and returns a partial state update ``dict``.
External dependencies (LLM, retrieval pipeline) are bound via
:func:`functools.partial` in ``graph.py``.
"""

from __future__ import annotations

import json
import re
from string import Template
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from sec_rag.agent.prompts import (
    EVALUATE_RELEVANCE_PROMPT,
    EXTRACT_OBLIGATIONS_PROMPT,
    QUERY_DELIM_END,
    QUERY_DELIM_START,
    REWRITE_QUERY_PROMPT,
    ROUTE_SYSTEM_PROMPT,
)
from sec_rag.models.analysis import AnalysisResult, RelevanceGrade
from sec_rag.models.retrieval import RetrievalResult, ScoredChunk
from sec_rag.models.state import QueryState
from sec_rag.retrieval.pipeline import RetrievalPipeline

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Override LLM's is_relevant=True when the numeric score contradicts it.
RELEVANCE_SCORE_THRESHOLD = 0.3

# Cap confidence on best-effort generation (retries exhausted, chunks not relevant).
BEST_EFFORT_CONFIDENCE_CAP = 0.3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_chunk_text(text: str) -> str:
    """Strip prompt delimiter strings from chunk text to prevent injection.

    SEC filings ingested from EDGAR could contain adversarial text that mimics
    the query delimiters, breaking the prompt boundary and allowing document-
    sourced instructions to reach the LLM outside the sandboxed context block.
    """
    return text.replace(QUERY_DELIM_START, "").replace(QUERY_DELIM_END, "")


def _format_chunks(scored_chunks: list[ScoredChunk]) -> str:
    """Build a context string from scored chunks for the LLM prompt."""
    parts: list[str] = []
    for chunk in scored_chunks:
        header = (
            f"[chunk_id: {chunk.chunk_id}] "
            f"[company: {chunk.metadata.company_name}] "
            f"[section: {chunk.metadata.section_type}]"
        )
        parts.append(f"{header}\n{_sanitize_chunk_text(chunk.text)}\n---")
    return "\n".join(parts)


_FENCED_CODE_RE = re.compile(r"^```\w*\n?(.*?)```$", re.DOTALL)


def _extract_text_content(content: str | list[Any], fallback: str = "") -> str:
    """Normalize LangChain multi-block content to a plain text string.

    Anthropic returns ``[{"type": "text", "text": "..."}]`` while OpenAI
    returns a plain ``str``. This helper unifies both.
    """
    if isinstance(content, list):
        return content[0].get("text", fallback) if content else fallback
    return content if content else fallback


def _parse_json(text: str | list[Any]) -> dict[str, Any]:
    """Extract the first JSON object from *text*.

    Handles cases where the LLM wraps JSON in markdown fences, including
    single-line fences with a language tag (e.g. triple-backtick json).

    Also handles LangChain multi-block content (``list[dict]``) by extracting
    the text from the first block via :func:`_extract_text_content`.
    """
    text = _extract_text_content(text)
    cleaned = text.strip()
    fence_match = _FENCED_CODE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    result: dict[str, Any] = json.loads(cleaned)
    return result


# ---------------------------------------------------------------------------
# Node functions
#
# State access convention: use state["key"] for fields guaranteed by graph
# topology (set before this node runs). Use state.get("key") for fields that
# depend on upstream success and may be None.
# ---------------------------------------------------------------------------

def route_node(state: QueryState, *, llm: BaseChatModel) -> dict[str, Any]:
    """Classify the query as *extraction* or *general*.

    On LLM or parse failure the node defaults to ``"extraction"`` so the
    pipeline always attempts retrieval rather than silently dropping a valid
    query.
    """
    original_query: str = state["original_query"]
    try:
        delimited_query = f"{QUERY_DELIM_START}\n{original_query}\n{QUERY_DELIM_END}"
        response = llm.invoke([
            SystemMessage(content=ROUTE_SYSTEM_PROMPT),
            HumanMessage(content=delimited_query),
        ])
        parsed = _parse_json(response.content)
        query_type = parsed.get("query_type", "extraction")
        if query_type not in ("extraction", "general"):
            query_type = "extraction"
        return {"query_type": query_type, "current_query": original_query}
    except json.JSONDecodeError:
        logger.warning("route_node_json_parse_failed", query=original_query[:80])
        return {"query_type": "extraction", "current_query": original_query}
    except Exception:
        logger.exception("route_node_llm_error")
        return {"query_type": "extraction", "current_query": original_query}


def retrieve_node(state: QueryState, *, pipeline: RetrievalPipeline) -> dict[str, Any]:
    """Execute the retrieval pipeline against the current query.

    On failure, returns ``retrieval_result=None`` WITHOUT setting ``error``.
    This allows the evaluate→rewrite retry loop to handle transient failures
    instead of immediately terminating the pipeline.
    """
    current_query: str = state["current_query"]
    try:
        result: RetrievalResult = pipeline.retrieve(current_query)
        return {"retrieval_result": result}
    except Exception:
        logger.exception("retrieve_node_pipeline_error")
        return {"retrieval_result": None}


def evaluate_node(state: QueryState, *, llm: BaseChatModel) -> dict[str, Any]:
    """CRAG-style relevance check on the retrieved chunks."""
    retrieval_result: RetrievalResult | None = state.get("retrieval_result")

    # No retrieval result or empty chunks → not relevant
    if retrieval_result is None or not retrieval_result.scored_chunks:
        return {
            "relevance_grade": RelevanceGrade(
                is_relevant=False,
                reasoning="No chunks retrieved.",
                score=0.0,
            ),
        }

    context = _format_chunks(retrieval_result.scored_chunks)
    current_query: str = state["current_query"]
    prompt = Template(EVALUATE_RELEVANCE_PROMPT).substitute(
        query=current_query,
        context=context,
        query_delim_start=QUERY_DELIM_START,
        query_delim_end=QUERY_DELIM_END,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json(response.content)
        raw_score = float(parsed.get("score", 0.0))
        clamped_score = max(0.0, min(1.0, raw_score))
        is_relevant = bool(parsed.get("is_relevant", False))
        # Cross-validate: override is_relevant if score contradicts it
        if is_relevant and clamped_score < RELEVANCE_SCORE_THRESHOLD:
            is_relevant = False
        elif not is_relevant and clamped_score >= RELEVANCE_SCORE_THRESHOLD:
            is_relevant = True
        grade = RelevanceGrade(
            is_relevant=is_relevant,
            reasoning=str(parsed.get("reasoning", "")),
            score=clamped_score,
        )
        return {"relevance_grade": grade}
    except json.JSONDecodeError:
        logger.warning("evaluate_node_json_parse_failed")
        return {
            "relevance_grade": RelevanceGrade(
                is_relevant=False,
                reasoning="Evaluation failed; defaulting to not relevant.",
                score=0.0,
            ),
        }
    except Exception:
        logger.exception("evaluate_node_llm_error")
        return {
            "relevance_grade": RelevanceGrade(
                is_relevant=False,
                reasoning="Evaluation failed; defaulting to not relevant.",
                score=0.0,
            ),
        }


def _attempt_structured_output(
    llm: BaseChatModel, prompt: str, original_query: str, source_count: int,
) -> AnalysisResult | None:
    """Try LangChain structured output. Return AnalysisResult or None on failure."""
    try:
        structured_llm = llm.with_structured_output(AnalysisResult)
        raw_result = structured_llm.invoke([HumanMessage(content=prompt)])
        result = raw_result if isinstance(raw_result, AnalysisResult) else None
        if result is not None:
            result = result.model_copy(
                update={"query": original_query, "source_count": source_count}
            )
        return result
    except Exception:
        logger.warning("generate_node_structured_output_failed")
        return None


def _attempt_manual_parse(
    llm: BaseChatModel, prompt: str, original_query: str, source_count: int,
) -> AnalysisResult | None:
    """Fallback: invoke LLM, parse JSON manually, validate as AnalysisResult.

    Returns None on any failure (JSONDecodeError or other exception).
    Caller handles the error dict.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json(response.content)
        parsed["query"] = original_query
        parsed["source_count"] = source_count
        if "confidence" in parsed:
            parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
        return AnalysisResult.model_validate(parsed)
    except json.JSONDecodeError:
        logger.warning("generate_node_manual_parse_failed")
        return None
    except Exception:
        logger.exception("generate_node_llm_error")
        return None


def _apply_best_effort_cap(result: AnalysisResult, is_best_effort: bool) -> AnalysisResult:
    """Cap confidence and prepend warning if best-effort generation."""
    if not is_best_effort:
        return result
    capped_confidence = min(result.confidence, BEST_EFFORT_CONFIDENCE_CAP)
    warning = (
        "Warning: The retrieved documents may not be directly relevant to your "
        "query. Results should be verified against the original contract text."
    )
    summary = f"{warning} {result.summary}" if result.summary else warning
    return result.model_copy(
        update={"confidence": capped_confidence, "summary": summary}
    )


def generate_node(state: QueryState, *, llm: BaseChatModel) -> dict[str, Any]:
    """Extract obligations from retrieved chunks using structured output.

    When the relevance grade indicates chunks are NOT relevant (best-effort
    generation after max retries), the result is capped at low confidence and
    the summary warns the user.
    """
    retrieval_result: RetrievalResult | None = state.get("retrieval_result")
    original_query: str = state["original_query"]

    if retrieval_result is None:
        # retrieval_result=None means every retrieval attempt raised an exception
        # (e.g., Qdrant unreachable).  Distinct from "Qdrant responded but returned
        # no matching chunks" which is scored_chunks=[].
        return {
            "analysis_result": AnalysisResult(
                query=original_query,
                obligations=[],
                summary=(
                    "Unable to retrieve contract data. The retrieval service may "
                    "be temporarily unavailable. Please try again later."
                ),
                confidence=0.0,
                source_count=0,
            ),
        }
    if not retrieval_result.scored_chunks:
        return {
            "analysis_result": AnalysisResult(
                query=original_query,
                obligations=[],
                summary="No relevant contract data found for this query.",
                confidence=0.0,
                source_count=0,
            ),
        }

    relevance_grade = state.get("relevance_grade")
    is_best_effort = relevance_grade is not None and not relevance_grade.is_relevant

    context = _format_chunks(retrieval_result.scored_chunks)
    source_count = len(retrieval_result.scored_chunks)
    prompt = Template(EXTRACT_OBLIGATIONS_PROMPT).substitute(
        query=original_query,
        context=context,
        query_delim_start=QUERY_DELIM_START,
        query_delim_end=QUERY_DELIM_END,
    )

    # Try structured output first, fall back to manual JSON parse.
    result = _attempt_structured_output(llm, prompt, original_query, source_count)
    if result is None:
        result = _attempt_manual_parse(llm, prompt, original_query, source_count)
    if result is None:
        return {
            "analysis_result": None,
            "error": "Obligation extraction failed. Please try again.",
        }

    return {"analysis_result": _apply_best_effort_cap(result, is_best_effort)}


def rewrite_node(state: QueryState, *, llm: BaseChatModel) -> dict[str, Any]:
    """Reformulate the current query for better retrieval."""
    current_query: str = state["current_query"]
    retry_count: int = state.get("retry_count", 0)

    try:
        prompt = Template(REWRITE_QUERY_PROMPT).substitute(
            query=current_query,
            query_delim_start=QUERY_DELIM_START,
            query_delim_end=QUERY_DELIM_END,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        new_query = _extract_text_content(response.content, fallback=current_query)
        new_query = new_query.strip()
        if not new_query:
            new_query = current_query  # guard against empty LLM response
        return {"current_query": new_query, "retry_count": retry_count + 1}
    except Exception:
        logger.exception("rewrite_node_llm_error")
        return {"current_query": current_query, "retry_count": retry_count + 1}


# ---------------------------------------------------------------------------
# Conditional edge functions
#
# Error signaling pattern: only generate_node sets the ``error`` state
# field because it's the terminal extraction node — there is nothing left to
# retry.  Upstream nodes (route, retrieve, evaluate, rewrite) gracefully
# degrade to safe defaults instead of setting ``error``, keeping the retry
# loop alive.  The ``state.get("error")`` checks below are defensive guards
# for robustness — they are not triggered by the current node implementations.
# ---------------------------------------------------------------------------

def general_response_node(state: QueryState) -> dict[str, Any]:
    """Return an informative response for general (non-extraction) queries."""
    return {
        "analysis_result": AnalysisResult(
            query=state["original_query"],
            obligations=[],
            summary=(
                "This system analyzes specific provisions in employment contracts "
                "filed with SEC EDGAR. Try asking about specific terms like "
                "non-compete obligations, compensation terms, or termination conditions."
            ),
            confidence=0.0,
            source_count=0,
        ),
    }


def should_retrieve(state: QueryState) -> str:
    """After ROUTE: route to retrieve, general_response, or end."""
    if state.get("error"):
        return "end"
    if state["query_type"] == "extraction":
        return "retrieve"
    return "general_response"


def should_retry_or_generate(state: QueryState, *, max_retries: int) -> str:
    """After EVALUATE: decide whether to generate, rewrite, or end."""
    if state.get("error"):
        return "end"
    grade = state.get("relevance_grade")
    if grade and grade.is_relevant:
        return "generate"
    if state.get("retry_count", 0) >= max_retries:
        return "generate"
    return "rewrite"
