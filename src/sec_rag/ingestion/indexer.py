"""Qdrant collection lifecycle and chunk indexing."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from sec_rag.models.documents import Chunk


class QdrantIndexer:
    """Manages Qdrant collection lifecycle and chunk indexing."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_dim: int,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._dense_dim = dense_dim

    @classmethod
    def from_url(
        cls, url: str, collection_name: str, dense_dim: int, api_key: str = "",
    ) -> QdrantIndexer:
        """Create an indexer connected to a Qdrant server."""
        kwargs: dict[str, Any] = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key
        return cls(QdrantClient(**kwargs), collection_name, dense_dim)

    @classmethod
    def from_memory(
        cls, collection_name: str = "test", dense_dim: int = 384
    ) -> QdrantIndexer:
        """Create an in-memory indexer for testing."""
        return cls(QdrantClient(":memory:"), collection_name, dense_dim)

    def ensure_collection(self) -> None:
        """Create collection if it does not exist. Idempotent.

        Config:
        - Named vector "dense": size=dense_dim, COSINE distance
        - Named sparse vector "sparse": for BM25 term vectors
        - Payload indexes on: company_name (keyword), section_type (keyword),
          filing_date (datetime)

        Raises:
            ConnectionError: If Qdrant is unreachable.
        """
        if self._client.collection_exists(self._collection):
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._dense_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )

        self._client.create_payload_index(
            self._collection, "company_name", models.PayloadSchemaType.KEYWORD
        )
        self._client.create_payload_index(
            self._collection, "section_type", models.PayloadSchemaType.KEYWORD
        )
        self._client.create_payload_index(
            self._collection, "filing_date", models.PayloadSchemaType.DATETIME
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense_vectors: np.ndarray,
        sparse_vectors: list[models.SparseVector],
        batch_size: int = 100,
    ) -> int:
        """Upsert points with both dense and sparse vectors plus full metadata payload.

        Point ID is a deterministic UUID5 from chunk.chunk_id, converted to string.

        Args:
            chunks: Chunk objects to index.
            dense_vectors: Dense embeddings array of shape (len(chunks), dense_dim).
            sparse_vectors: Sparse BM25 vectors, one per chunk.
            batch_size: Number of points per upsert batch.

        Returns:
            Count of upserted points.

        Raises:
            ValueError: If input lengths do not match.
            qdrant_client.http.exceptions.UnexpectedResponse: On Qdrant error.
        """
        n = len(chunks)
        if dense_vectors.shape[0] != n or len(sparse_vectors) != n:
            msg = (
                f"Length mismatch: {n} chunks, "
                f"{dense_vectors.shape[0]} dense vectors, "
                f"{len(sparse_vectors)} sparse vectors"
            )
            raise ValueError(msg)

        if n == 0:
            return 0

        dense_lists = dense_vectors.tolist()  # batch convert once
        points: list[models.PointStruct] = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            payload = {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                **chunk.metadata.to_qdrant_payload(),
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_lists[i],
                        "sparse": sparse_vectors[i],
                    },
                    payload=payload,
                )
            )

        for start in range(0, n, batch_size):
            batch = points[start : start + batch_size]
            self._client.upsert(collection_name=self._collection, points=batch)

        return n

    def collection_info(self) -> dict[str, Any]:
        """Return collection status summary.

        Returns:
            Dict with keys ``point_count`` (int) and ``status`` (str).
        """
        info = self._client.get_collection(self._collection)
        return {
            "point_count": info.points_count,
            "status": str(info.status.value),
        }

    def close(self) -> None:
        """Close the Qdrant client connection."""
        self._client.close()

    @property
    def client(self) -> QdrantClient:
        """Expose client for retrieval pipeline use."""
        return self._client

    @property
    def collection_name(self) -> str:
        return self._collection
