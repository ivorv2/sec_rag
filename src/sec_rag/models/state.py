from typing import TypedDict

from sec_rag.models.analysis import AnalysisResult, RelevanceGrade
from sec_rag.models.retrieval import RetrievalResult


class QueryState(TypedDict):
    """LangGraph state schema. Each field is read/written by specific nodes."""

    # Input (set once at entry)
    original_query: str

    # Router output
    query_type: str  # "extraction" for Phase 1
    current_query: str  # may be rewritten by evaluator

    # Retriever output
    retrieval_result: RetrievalResult | None

    # Evaluator output
    relevance_grade: RelevanceGrade | None

    # Generator output
    analysis_result: AnalysisResult | None

    # Control flow
    retry_count: int  # starts at 0, max 2
    error: str | None  # populated on unrecoverable failure
