"""Retrieval pipeline: hybrid search then cross-encoder rerank."""

from __future__ import annotations

import structlog
from qdrant_client import QdrantClient, models

from sec_rag.embedding import ChunkEmbedder, SparseEncoder
from sec_rag.models.documents import ChunkMetadata
from sec_rag.models.retrieval import RetrievalResult, ScoredChunk
from sec_rag.retrieval.hybrid_search import hybrid_query
from sec_rag.retrieval.reranker import Reranker

logger = structlog.get_logger(__name__)


class RetrievalPipeline:
    """Orchestrates hybrid_search followed by cross-encoder reranking."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        embedder: ChunkEmbedder,
        sparse_encoder: SparseEncoder,
        reranker: Reranker,
        dense_limit: int = 50,
        sparse_limit: int = 50,
        rrf_top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        self._client = qdrant_client
        self._collection_name = collection_name
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._reranker = reranker
        self._dense_limit = dense_limit
        self._sparse_limit = sparse_limit
        self._rrf_top_k = rrf_top_k
        self._rerank_top_k = rerank_top_k

    def retrieve(
        self,
        query: str,
        section_filter: str | None = None,
        company_filter: str | None = None,
    ) -> RetrievalResult:
        """Run the full retrieval pipeline: embed, search, convert, rerank.

        Steps:
            1. Embed query into dense vector.
            2. Encode query into sparse vector.
            3. Run hybrid_query() to get top RRF candidates.
            4. Convert ScoredPoints to ScoredChunks via payload reconstruction.
            5. Rerank with cross-encoder.
            6. Return RetrievalResult.

        Args:
            query: The user's search query.
            section_filter: If set, restrict to this section_type value.
            company_filter: If set, restrict to this company_name value.

        Returns:
            RetrievalResult with scored_chunks and total_candidates_before_rerank.

        Raises:
            ConnectionError: If Qdrant is unreachable.
        """
        # 1. Dense embedding
        dense_array = self._embedder.embed_texts([query])
        dense_vector = dense_array[0].tolist()

        # 2. Sparse encoding
        sparse_vectors = self._sparse_encoder.encode([query])
        sparse_vector = sparse_vectors[0]

        # 3. Hybrid search
        scored_points = hybrid_query(
            client=self._client,
            collection_name=self._collection_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            dense_limit=self._dense_limit,
            sparse_limit=self._sparse_limit,
            rrf_top_k=self._rrf_top_k,
            section_filter=section_filter,
            company_filter=company_filter,
        )

        # 4. Convert ScoredPoints to ScoredChunks (skip corrupted points)
        scored_chunks = []
        for sp in scored_points:
            try:
                scored_chunks.append(_scored_point_to_chunk(sp))
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "skipping_corrupted_point",
                    point_id=sp.id,
                    error=str(exc),
                    payload_keys=sorted(sp.payload.keys()) if sp.payload else None,
                )
        total_candidates = len(scored_chunks)

        # 5. Rerank
        reranked = self._reranker.rerank(query, scored_chunks, top_k=self._rerank_top_k)

        # 6. Build result
        return RetrievalResult(
            query=query,
            scored_chunks=reranked,
            total_candidates_before_rerank=total_candidates,
        )


def _scored_point_to_chunk(sp: models.ScoredPoint) -> ScoredChunk:
    """Convert a Qdrant ScoredPoint to a ScoredChunk by reading its payload.

    Args:
        sp: A qdrant_client.models.ScoredPoint with payload dict.

    Returns:
        ScoredChunk with metadata reconstructed from payload fields.
    """
    payload = sp.payload
    if payload is None:
        raise ValueError(f"ScoredPoint {sp.id} has no payload")
    metadata = ChunkMetadata.from_qdrant_payload(payload)
    return ScoredChunk(
        chunk_id=payload["chunk_id"],
        text=payload["text"],
        metadata=metadata,
        rrf_score=float(sp.score),
    )
