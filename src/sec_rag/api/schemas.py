"""Pydantic schemas for the FastAPI request/response layer."""

from pydantic import BaseModel, Field

from sec_rag.models.analysis import AnalysisResult


class QueryRequest(BaseModel):
    """Incoming query payload."""

    query: str = Field(min_length=10, max_length=500)
    skip_cache: bool = False
    include_contexts: bool = False


class QueryResponse(BaseModel):
    """Unified response for query endpoints."""

    success: bool
    result: AnalysisResult | None = None
    error: str | None = None
    contexts: list[str] | None = None
    cached: bool = False


class HealthResponse(BaseModel):
    """Health check response.

    ``llm_provider`` removed to prevent information disclosure —
    revealing the LLM backend helps attackers target provider-specific
    prompt injection techniques.
    """

    status: str
    qdrant_connected: bool
    collection_count: int
    cache_connected: bool = False
