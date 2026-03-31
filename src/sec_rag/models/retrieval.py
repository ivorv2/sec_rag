from pydantic import BaseModel

from sec_rag.models.documents import ChunkMetadata


class ScoredChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class RetrievalResult(BaseModel):
    query: str
    scored_chunks: list[ScoredChunk]
    total_candidates_before_rerank: int = 0
