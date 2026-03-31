"""Tests for QdrantIndexer — in-memory mode, no Docker required."""

from __future__ import annotations

import uuid
from datetime import date

import numpy as np
import pytest
from qdrant_client import models

from sec_rag.ingestion.indexer import QdrantIndexer
from sec_rag.models.documents import Chunk, ChunkMetadata, SectionType

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_chunk(index: int) -> Chunk:
    """Create a Chunk with minimal valid metadata for testing."""
    return Chunk(
        chunk_id=f"test-idx-{index:03d}",
        text=f"This is test chunk number {index} for indexer testing.",
        metadata=ChunkMetadata(
            company_name="Acme Inc",
            cik="0001234567",
            filing_date=date(2024, 6, 15),
            exhibit_number="EX-10.1",
            accession_number="0001234567-24-000042",
            section_type=SectionType.COMPENSATION,
            chunk_index=index,
            source_url="https://www.sec.gov/Archives/edgar/data/1234567/ex10-1.htm",
        ),
    )


def _make_chunks(n: int) -> list[Chunk]:
    return [_make_chunk(i) for i in range(n)]


def _random_dense(n: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _random_sparse(n: int) -> list[models.SparseVector]:
    rng = np.random.default_rng(42)
    result: list[models.SparseVector] = []
    for _ in range(n):
        k = rng.integers(3, 10)
        indices = sorted(rng.choice(1000, size=k, replace=False).tolist())
        values = rng.uniform(0.1, 1.0, size=k).tolist()
        result.append(models.SparseVector(indices=indices, values=values))
    return result


@pytest.fixture
def indexer() -> QdrantIndexer:
    """In-memory indexer with collection already ensured."""
    idx = QdrantIndexer.from_memory(collection_name="test_collection", dense_dim=384)
    idx.ensure_collection()
    return idx


# ── ensure_collection ────────────────────────────────────────────────────────


class TestEnsureCollection:
    def test_creates_collection_with_correct_vector_config(self) -> None:
        """Dense vector size=384 and sparse vector both exist."""
        idx = QdrantIndexer.from_memory(collection_name="vec_check", dense_dim=384)
        idx.ensure_collection()

        info = idx.client.get_collection("vec_check")
        # Dense vector config
        dense_config = info.config.params.vectors["dense"]
        assert dense_config.size == 384
        assert dense_config.distance == models.Distance.COSINE
        # Sparse vector config exists
        sparse_config = info.config.params.sparse_vectors
        assert "sparse" in sparse_config

    def test_idempotent_no_error_on_second_call(self) -> None:
        """Calling ensure_collection twice does not raise."""
        idx = QdrantIndexer.from_memory(collection_name="idem_check", dense_dim=384)
        idx.ensure_collection()
        idx.ensure_collection()  # no error

        assert idx.client.collection_exists("idem_check")


# ── upsert_chunks ────────────────────────────────────────────────────────────


class TestUpsertChunks:
    def test_upsert_five_chunks_point_count(self, indexer: QdrantIndexer) -> None:
        """Upserting 5 chunks results in point_count=5."""
        chunks = _make_chunks(5)
        dense = _random_dense(5)
        sparse = _random_sparse(5)

        count = indexer.upsert_chunks(chunks, dense, sparse)

        assert count == 5
        info = indexer.collection_info()
        assert info["point_count"] == 5

    def test_upsert_idempotent_deterministic_uuids(
        self, indexer: QdrantIndexer
    ) -> None:
        """Upserting the same 5 chunks twice keeps point_count=5."""
        chunks = _make_chunks(5)
        dense = _random_dense(5)
        sparse = _random_sparse(5)

        indexer.upsert_chunks(chunks, dense, sparse)
        indexer.upsert_chunks(chunks, dense, sparse)

        info = indexer.collection_info()
        assert info["point_count"] == 5

    def test_payload_fields_match_chunk_metadata(self, indexer: QdrantIndexer) -> None:
        """All payload fields match the original chunk's metadata."""
        chunk = _make_chunk(0)
        dense = _random_dense(1)
        sparse = _random_sparse(1)

        indexer.upsert_chunks([chunk], dense, sparse)

        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
        results = indexer.client.retrieve(
            indexer.collection_name, ids=[point_id], with_payload=True
        )
        assert len(results) == 1
        payload = results[0].payload

        assert payload["text"] == chunk.text
        assert payload["chunk_id"] == chunk.chunk_id
        assert payload["company_name"] == chunk.metadata.company_name
        assert payload["cik"] == chunk.metadata.cik
        assert payload["filing_date"] == chunk.metadata.filing_date.isoformat()
        assert payload["exhibit_number"] == chunk.metadata.exhibit_number
        assert payload["accession_number"] == chunk.metadata.accession_number
        assert payload["section_type"] == chunk.metadata.section_type.value
        assert payload["chunk_index"] == chunk.metadata.chunk_index
        assert payload["source_url"] == chunk.metadata.source_url

    def test_batch_size_smaller_than_total(self, indexer: QdrantIndexer) -> None:
        """batch_size=2 with 5 chunks still upserts all points."""
        chunks = _make_chunks(5)
        dense = _random_dense(5)
        sparse = _random_sparse(5)

        count = indexer.upsert_chunks(chunks, dense, sparse, batch_size=2)

        assert count == 5
        info = indexer.collection_info()
        assert info["point_count"] == 5

    def test_empty_chunks_returns_zero(self, indexer: QdrantIndexer) -> None:
        """Upserting zero chunks returns 0 and does not error."""
        count = indexer.upsert_chunks([], np.empty((0, 384), dtype=np.float32), [])
        assert count == 0

    def test_mismatched_lengths_raises_value_error(
        self, indexer: QdrantIndexer
    ) -> None:
        """Mismatched input lengths raise ValueError."""
        chunks = _make_chunks(3)
        dense = _random_dense(2)  # wrong size
        sparse = _random_sparse(3)

        with pytest.raises(ValueError, match="Length mismatch"):
            indexer.upsert_chunks(chunks, dense, sparse)


# ── collection_info ──────────────────────────────────────────────────────────


class TestCollectionInfo:
    def test_returns_point_count_and_status(self, indexer: QdrantIndexer) -> None:
        """collection_info returns dict with point_count and status keys."""
        info = indexer.collection_info()

        assert "point_count" in info
        assert "status" in info
        assert isinstance(info["point_count"], int)
        assert isinstance(info["status"], str)
        assert info["point_count"] == 0
        assert info["status"] == "green"
