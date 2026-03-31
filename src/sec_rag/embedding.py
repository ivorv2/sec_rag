"""Dense and sparse embedding encoders for SEC contract chunks."""

import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from sec_rag.models.documents import Chunk


class ChunkEmbedder:
    """Dense embedding using sentence-transformers."""

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return shape (n, dim) float32 array of L2-normalised embeddings."""
        return self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True
        )

    def embed_chunks(
        self, chunks: list[Chunk], batch_size: int = 64
    ) -> list[tuple[Chunk, np.ndarray]]:
        """Convenience: return list of (chunk, embedding_vector) tuples."""
        texts = [c.text for c in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        return list(zip(chunks, embeddings))

    @property
    def dimension(self) -> int:
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("Model did not report embedding dimension")
        return dim


class SparseEncoder:
    """Sparse BM25 encoding using fastembed for Qdrant compatibility."""

    def __init__(self, model_name: str):
        self._model = SparseTextEmbedding(model_name=model_name)

    def encode(self, texts: list[str], batch_size: int = 256) -> list[models.SparseVector]:
        """Return list of SparseVector (indices, values) for each text.

        Processes in batches to limit peak memory usage.
        """
        result: list[models.SparseVector] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings = list(self._model.embed(batch))
            result.extend(
                models.SparseVector(
                    indices=e.indices.tolist(),
                    values=e.values.tolist(),
                )
                for e in embeddings
            )
        return result
