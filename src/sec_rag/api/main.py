"""FastAPI application entrypoint with lifespan-managed dependencies."""

import os
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from sec_rag.api.routes import router
from sec_rag.observability.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
)

logger = structlog.get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared resources on startup, cleanup on shutdown."""
    from qdrant_client import QdrantClient

    from sec_rag.agent.graph import build_graph
    from sec_rag.agent.llm import create_llm
    from sec_rag.config import Settings
    from sec_rag.embedding import ChunkEmbedder, SparseEncoder
    from sec_rag.retrieval.pipeline import RetrievalPipeline
    from sec_rag.retrieval.reranker import Reranker

    settings = Settings()

    configure_logging(settings.log_format, settings.log_level)

    # Early config validation — before resource acquisition.
    if not settings.api_key.get_secret_value():
        if settings.require_api_key:
            raise RuntimeError(
                "SEC_RAG_REQUIRE_API_KEY is True but SEC_RAG_API_KEY is not set. "
                "Refusing to start without authentication."
            )
        logger.warning(
            "NO_API_KEY_CONFIGURED",
            msg="API running WITHOUT authentication. Set SEC_RAG_API_KEY for production.",
        )

    # Build components — cleanup on partial initialization failure.
    qdrant_client: QdrantClient | None = None
    cache = None
    try:
        logger.info("Creating LLM client", llm_provider=settings.llm_provider)
        llm = create_llm(settings)
        logger.info("Connecting to Qdrant", qdrant_url=settings.qdrant_url)
        qdrant_kwargs: dict[str, Any] = {"url": settings.qdrant_url, "timeout": 5}
        if settings.qdrant_api_key.get_secret_value():
            qdrant_kwargs["api_key"] = settings.qdrant_api_key.get_secret_value()
        qdrant_client = QdrantClient(**qdrant_kwargs)
        logger.info("Loading embedding model", model=settings.embedding_model)
        embedder = ChunkEmbedder(model_name=settings.embedding_model)
        logger.info("Loading sparse encoder")
        sparse_encoder = SparseEncoder(model_name=settings.sparse_model)
        logger.info("Loading reranker model", model=settings.rerank_model)
        reranker = Reranker(model_name=settings.rerank_model)

        retrieval_pipeline = RetrievalPipeline(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
            reranker=reranker,
            dense_limit=settings.retrieval_dense_limit,
            sparse_limit=settings.retrieval_sparse_limit,
            rrf_top_k=settings.retrieval_rrf_top_k,
            rerank_top_k=settings.rerank_top_k,
        )

        graph = build_graph(
            llm=llm, retrieval_pipeline=retrieval_pipeline, max_retries=settings.max_retries
        )
        # Store whether Langfuse is configured (handler created per-request for thread safety)
        langfuse_enabled = bool(
            settings.langfuse_public_key and settings.langfuse_secret_key.get_secret_value()
        )
        if langfuse_enabled:
            logger.info("langfuse_enabled")

        # Cache (optional — disabled if redis_url is empty)
        if settings.redis_url and settings.cache_enabled:
            from sec_rag.cache import QueryCache

            cache = QueryCache(settings.redis_url, settings.cache_ttl_seconds)
            # QueryCache.__init__ already pings Redis and sets _available — no need
            # to ping again here. Just log the connection status.
            if cache.is_available:
                logger.info("cache_connected", redis_url=settings.redis_url.split("@")[-1])
            else:
                logger.warning("cache_unavailable", msg="Operating without Redis cache")
        else:
            logger.info("cache_disabled", msg="No Redis URL configured")
    except Exception:
        # Clean up resources already acquired before re-raising.
        if cache is not None:
            try:
                cache.close()
            except Exception:
                logger.warning("cleanup_cache_failed_during_init_rollback")
        if qdrant_client is not None:
            try:
                qdrant_client.close()
            except Exception:
                logger.warning("cleanup_qdrant_failed_during_init_rollback")
        raise

    # Store on app.state — accessed by route handlers via request.app.state.
    # Contract: settings, graph, qdrant_client always present after lifespan.
    # langfuse_enabled (bool) and cache (QueryCache | None) are optional.
    app.state.settings = settings
    app.state.graph = graph
    app.state.langfuse_enabled = langfuse_enabled
    app.state.qdrant_client = qdrant_client
    app.state.cache = cache

    logger.info("Startup complete")

    yield

    # Cleanup — independent try/except per resource.
    try:
        if cache is not None:
            cache.close()
    except Exception:
        logger.warning("cache_close_failed")
    try:
        qdrant_client.close()
    except Exception:
        logger.warning("qdrant_close_failed")


app = FastAPI(
    title="SEC RAG -- Employment Contract Analyzer",
    description="Agentic RAG for analyzing employment agreements from SEC EDGAR filings",
    version="0.1.0",
    lifespan=lifespan,
)

# Prometheus metrics. Instrument always; expose /metrics conditionally.
# When exposed, restrict at reverse proxy or network layer in production.
_instrumentator = Instrumentator().instrument(app)
if os.getenv("SEC_RAG_EXPOSE_METRICS", "true").lower() in ("true", "1", "yes"):
    _instrumentator.expose(app, endpoint="/metrics")

# In-memory rate limiter: max 10 requests per minute per client IP.
# NOTE: This state is per-process and resets on restart. For horizontal scaling
# (multiple app instances behind a load balancer), replace with a Redis-backed
# rate limiter (e.g., redis sliding window) so limits are shared across instances.
_rate_limit_store: OrderedDict[str, list[float]] = OrderedDict()
_RATE_LIMIT_MAX = 10
_RATE_LIMIT_WINDOW = 60.0  # seconds
_RATE_LIMIT_MAX_IPS = 10_000  # cap tracked IPs to prevent memory exhaustion

_MAX_BODY_BYTES = 1_048_576  # 1 MB request body limit


def _get_client_ip(request: Request) -> str:
    """Extract client IP for rate limiting.

    Only reads X-Forwarded-For when ``settings.behind_proxy`` is True.
    When False (default), the header is ignored — preventing spoofed IPs from
    bypassing rate limits when the app is exposed directly.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings and getattr(settings, "behind_proxy", False):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def body_size_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Reject requests with Content-Length exceeding the body size limit."""
    content_length = request.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_BODY_BYTES:
                logger.warning("request_body_too_large", content_length=content_length)
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large. Max 1 MB."},
                )
        except ValueError:
            pass  # Malformed header — let downstream handle it
    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Reject requests exceeding the per-IP rate limit on /query and /health."""
    rate_limit = _RATE_LIMIT_MAX if request.url.path == "/query" else 60
    if request.url.path in ("/query", "/health"):
        client_ip = _get_client_ip(request)
        now = time.monotonic()
        timestamps = _rate_limit_store.get(client_ip, [])
        # Prune old entries and cap to rate_limit to bound memory per IP
        timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
        timestamps = timestamps[-rate_limit:]

        if len(timestamps) >= rate_limit:
            _rate_limit_store[client_ip] = timestamps
            logger.warning("rate_limit_exceeded", client_ip=client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Max {rate_limit} requests per minute."
                },
            )

        # Record timestamp BEFORE yielding to call_next to close the TOCTOU
        # window — concurrent requests from the same IP will see this timestamp
        # immediately rather than after the response completes.
        timestamps.append(time.monotonic())
        if client_ip not in _rate_limit_store and len(_rate_limit_store) >= _RATE_LIMIT_MAX_IPS:
            _rate_limit_store.popitem(last=False)  # O(1) eviction of oldest entry
        _rate_limit_store[client_ip] = timestamps

        return await call_next(request)
    return await call_next(request)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Attach a unique request ID to every request for log correlation."""
    request_id = request.headers.get("X-Request-ID", "")
    if not _UUID_RE.match(request_id):
        request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    bind_request_context(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        clear_request_context()


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Set security headers on all responses, including error responses from inner middleware."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


app.include_router(router)
