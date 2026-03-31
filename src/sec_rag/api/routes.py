"""API route handlers for the SEC RAG query and health endpoints."""

import concurrent.futures
import contextvars
import hmac
import threading
import time
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import Counter, Histogram

from sec_rag.agent.guardrails import validate_input
from sec_rag.api.schemas import HealthResponse, QueryRequest, QueryResponse

logger = structlog.get_logger(__name__)
router = APIRouter()

# Global concurrency limit for graph invocations
_GRAPH_SEMAPHORE = threading.Semaphore(20)

# Custom Prometheus metrics
CACHE_HITS = Counter("sec_rag_cache_hits_total", "Cache hit count")
CACHE_MISSES = Counter("sec_rag_cache_misses_total", "Cache miss count")
GRAPH_TIMEOUTS = Counter("sec_rag_graph_timeouts_total", "Graph invocation timeout count")
QUERY_LATENCY = Histogram(
    "sec_rag_query_latency_seconds", "Query processing latency", ["stage"]
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(
    request: Request, api_key: str | None = Depends(_api_key_header)
) -> None:
    """Verify the API key if one is configured in settings."""
    api_key_setting = getattr(request.app.state.settings, "api_key", None)
    configured_key = api_key_setting.get_secret_value() if api_key_setting else ""
    if configured_key:
        if not api_key or not hmac.compare_digest(api_key, configured_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _check_cache(
    cache: Any, cache_key: str, skip_cache: bool, start_time: float, query: str,
) -> QueryResponse | None:
    """Return cached QueryResponse on hit, else None. Handles metrics."""
    if cache is None or skip_cache:
        return None
    cached = cache.get(cache_key)
    if cached is not None:
        try:
            cached["cached"] = True
            response = QueryResponse.model_validate(cached)
            CACHE_HITS.inc()
            latency_ms = round((time.monotonic() - start_time) * 1000)
            QUERY_LATENCY.labels(stage="cache_hit").observe(latency_ms / 1000)
            logger.info(
                "query_completed",
                success=True,
                cached=True,
                latency_ms=latency_ms,
                query=query[:50],
            )
            return response
        except Exception:
            logger.warning("cache_deserialization_failed", query=query[:50])
    CACHE_MISSES.inc()
    return None


def _run_graph(
    graph: Any, initial_state: dict[str, Any], config: dict[str, Any], timeout_seconds: int,
) -> dict[str, Any]:
    """Invoke graph in a thread with contextvars propagation.

    Creates and shuts down a single-use ThreadPoolExecutor internally.
    Raises concurrent.futures.TimeoutError or Exception on failure.
    """
    ctx = contextvars.copy_context()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(ctx.run, graph.invoke, initial_state, config)
        result: dict[str, Any] = future.result(timeout=timeout_seconds)
        return result
    finally:
        # wait=False: the running thread may continue briefly after timeout, but
        # it is bounded by the per-call LLM timeout (~40s) and the global
        # semaphore (max 20 concurrent).  Using wait=True would block the HTTP
        # response, defeating the timeout's purpose.
        executor.shutdown(wait=False, cancel_futures=True)


def _build_response(
    final_state: dict[str, Any], include_contexts: bool,
) -> QueryResponse:
    """Construct QueryResponse from graph final state."""
    contexts = None
    if include_contexts:
        retrieval_result = final_state.get("retrieval_result")
        if retrieval_result is not None:
            contexts = [chunk.text for chunk in retrieval_result.scored_chunks]

    if final_state.get("error"):
        return QueryResponse(
            success=False,
            error=final_state["error"],
            contexts=contexts,
        )
    if final_state.get("analysis_result") is None:
        return QueryResponse(
            success=False,
            error="Analysis could not be completed. Please try again.",
            contexts=contexts,
        )
    return QueryResponse(
        success=True,
        result=final_state["analysis_result"],
        contexts=contexts,
    )


@router.post("/query", dependencies=[Depends(_verify_api_key)])
def query_contracts(
    request: Request, body: QueryRequest, background_tasks: BackgroundTasks
) -> QueryResponse:
    """Run the agentic RAG pipeline on a user query about employment contracts.

    This is a sync ``def`` endpoint so FastAPI runs it in a thread pool,
    preventing the blocking ``graph.invoke()`` call from stalling the event loop.
    """
    start_time = time.monotonic()

    validation = validate_input(body.query)
    if not validation.is_valid:
        return QueryResponse(success=False, error=validation.reason)

    cache = getattr(request.app.state, "cache", None)
    cache_key = f"{body.query}:ctx={body.include_contexts}"
    cached_response = _check_cache(cache, cache_key, body.skip_cache, start_time, body.query)
    if cached_response is not None:
        return cached_response

    graph = request.app.state.graph
    handler = None
    if getattr(request.app.state, "langfuse_enabled", False):
        try:
            from sec_rag.observability.langfuse_setup import create_langfuse_handler

            handler = create_langfuse_handler(request.app.state.settings)
        except Exception:
            logger.warning("langfuse_handler_creation_failed")

    initial_state: dict[str, Any] = {
        "original_query": body.query,
        "query_type": "",
        "current_query": "",
        "retrieval_result": None,
        "relevance_grade": None,
        "analysis_result": None,
        "retry_count": 0,
        "error": None,
    }
    config: dict[str, Any] = {"recursion_limit": 25}
    if handler:
        config["callbacks"] = [handler]

    timeout_seconds = getattr(request.app.state.settings, "query_timeout_seconds", 120)

    # Semaphore: caller owns acquire/release
    if not _GRAPH_SEMAPHORE.acquire(blocking=False):
        return QueryResponse(success=False, error="Server busy. Please try again shortly.")
    try:
        final_state = _run_graph(graph, initial_state, config, timeout_seconds)
    except concurrent.futures.TimeoutError:
        GRAPH_TIMEOUTS.inc()
        logger.error("graph_invocation_timed_out", timeout_seconds=timeout_seconds)
        return QueryResponse(success=False, error="Request timed out. Try a simpler query.")
    except Exception:
        logger.exception("graph_invocation_failed")
        return QueryResponse(success=False, error="An internal error occurred. Please try again.")
    finally:
        _GRAPH_SEMAPHORE.release()
        if handler is not None:
            try:
                handler.flush()
            except Exception:
                logger.warning("langfuse_flush_failed")

    response = _build_response(final_state, body.include_contexts)

    if cache is not None and response.success:
        response_dict = response.model_dump(mode="json")

        def _safe_cache_set() -> None:
            try:
                cache.set(cache_key, response_dict)
            except Exception:
                logger.warning("background_cache_write_failed")

        background_tasks.add_task(_safe_cache_set)

    latency_ms = round((time.monotonic() - start_time) * 1000)
    QUERY_LATENCY.labels(stage="full").observe(latency_ms / 1000)
    logger.info(
        "query_completed",
        success=response.success,
        cached=False,
        latency_ms=latency_ms,
        query=body.query[:50],
    )
    return response


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> JSONResponse:
    """Check system health: Qdrant connection and collection status.

    Returns 200 when healthy, 503 when degraded (Qdrant unreachable).
    The 503 ensures Docker healthchecks and load balancers correctly
    detect an unhealthy instance.

    Sync ``def`` so FastAPI runs it in a thread pool, preventing the
    blocking ``qdrant.get_collection()`` call from stalling the event loop.
    """
    settings = request.app.state.settings
    qdrant = request.app.state.qdrant_client

    cache = getattr(request.app.state, "cache", None)
    cache_connected = cache is not None and cache.ping()

    try:
        collection = qdrant.get_collection(settings.qdrant_collection)
        payload = HealthResponse(
            status="healthy",
            qdrant_connected=True,
            collection_count=collection.points_count or 0,
            cache_connected=cache_connected,
        )
        return JSONResponse(content=payload.model_dump(), status_code=200)
    except Exception:
        logger.warning("health_check_failed", exc_info=True)
        payload = HealthResponse(
            status="degraded",
            qdrant_connected=False,
            collection_count=0,
            cache_connected=cache_connected,
        )
        return JSONResponse(content=payload.model_dump(), status_code=503)
