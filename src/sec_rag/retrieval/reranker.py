"""Cross-encoder reranker for retrieval pipeline."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from sec_rag.models.retrieval import ScoredChunk


class Reranker:
    """Score and reorder chunks using a cross-encoder model."""

    def __init__(self, model_name: str):
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_k: int = 5
    ) -> list[ScoredChunk]:
        """Score each chunk against query and return top_k sorted by rerank score.

        Creates (query, chunk.text) pairs, scores them with the cross-encoder,
        and returns new ScoredChunk copies with ``rerank_score`` set.

        Args:
            query: The user's search query.
            chunks: Candidate chunks to rerank.
            top_k: Maximum number of chunks to return.

        Returns:
            Top-k chunks sorted by rerank_score descending. If chunks is empty,
            returns []. If len(chunks) <= top_k, returns all chunks (still
            scored and sorted).
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)

        scored = [
            chunk.model_copy(update={"rerank_score": float(score)})
            for chunk, score in zip(chunks, scores)
        ]
        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        return scored[:top_k]
