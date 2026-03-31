"""Tests for dense and sparse embedding encoders."""

from datetime import date

import numpy as np
import pytest
from qdrant_client import models

from sec_rag.embedding import ChunkEmbedder, SparseEncoder
from sec_rag.models.documents import Chunk, ChunkMetadata, SectionType


def _make_chunk(text: str, chunk_index: int = 0) -> Chunk:
    """Create a Chunk with minimal valid metadata for testing."""
    return Chunk(
        chunk_id=f"test-embed-{chunk_index}",
        text=text,
        metadata=ChunkMetadata(
            company_name="Acme Inc",
            cik="0001234567",
            filing_date=date(2024, 6, 1),
            exhibit_number="EX-10.1",
            accession_number="0001234567-24-000042",
            section_type=SectionType.COMPENSATION,
            chunk_index=chunk_index,
            source_url="https://www.sec.gov/Archives/edgar/data/1234567/ex10-1.htm",
        ),
    )


@pytest.fixture(scope="module")
def dense_embedder() -> ChunkEmbedder:
    """Module-scoped to avoid reloading the model per test."""
    return ChunkEmbedder(model_name="all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def sparse_encoder() -> SparseEncoder:
    """Module-scoped to avoid reloading the model per test."""
    return SparseEncoder(model_name="Qdrant/bm25")


# ── ChunkEmbedder.embed_texts ──────────────────────────────────────────────


class TestEmbedTextsShape:
    def test_output_shape_and_dtype(self, dense_embedder: ChunkEmbedder) -> None:
        texts = [
            "Base salary of one hundred fifty thousand dollars",
            "Non-compete clause applies for twelve months",
            "Employee is entitled to four weeks paid vacation",
            "Stock options vest over a four year period",
            "Severance equals six months of base pay",
        ]
        result = dense_embedder.embed_texts(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 384)
        assert result.dtype == np.float32


class TestEmbedTextsNormalization:
    def test_embeddings_are_l2_normalized(self, dense_embedder: ChunkEmbedder) -> None:
        texts = [
            "Annual bonus target is thirty percent of base salary",
            "Governing law shall be the State of Delaware",
            "Confidentiality obligations survive termination",
        ]
        result = dense_embedder.embed_texts(texts)

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ── ChunkEmbedder.embed_chunks ─────────────────────────────────────────────


class TestEmbedChunks:
    def test_returns_chunk_embedding_tuples(
        self, dense_embedder: ChunkEmbedder
    ) -> None:
        chunks = [
            _make_chunk("Employee duties include managing the engineering team", 0),
            _make_chunk("Termination for cause requires thirty days notice", 1),
            _make_chunk("Equity grant of ten thousand restricted stock units", 2),
        ]
        result = dense_embedder.embed_chunks(chunks)

        assert len(result) == 3
        for i, (chunk, vec) in enumerate(result):
            assert isinstance(chunk, Chunk)
            assert chunk is chunks[i]
            assert isinstance(vec, np.ndarray)
            assert vec.shape == (384,)
            assert vec.dtype == np.float32


# ── ChunkEmbedder.dimension ────────────────────────────────────────────────


class TestDimension:
    def test_dimension_is_384(self, dense_embedder: ChunkEmbedder) -> None:
        assert dense_embedder.dimension == 384


# ── SparseEncoder.encode ───────────────────────────────────────────────────


class TestSparseEncode:
    def test_returns_sparse_vectors(self, sparse_encoder: SparseEncoder) -> None:
        texts = [
            "Base salary compensation package",
            "Non-compete restrictive covenant agreement",
            "Change of control acquisition merger",
        ]
        result = sparse_encoder.encode(texts)

        assert len(result) == 3
        for sv in result:
            assert isinstance(sv, models.SparseVector)
            assert len(sv.indices) > 0
            assert len(sv.values) > 0
            assert len(sv.indices) == len(sv.values)
            assert all(isinstance(idx, int) for idx in sv.indices)
            assert all(isinstance(val, float) for val in sv.values)


# ── Semantic similarity ────────────────────────────────────────────────────


class TestSemanticSimilarity:
    def test_similar_texts_high_cosine_dissimilar_texts_low_cosine(
        self, dense_embedder: ChunkEmbedder
    ) -> None:
        texts = [
            "employee salary compensation",
            "base pay annual",
            "employee salary",
            "chocolate recipe",
        ]
        embeddings = dense_embedder.embed_texts(texts)

        # Cosine similarity (vectors are already L2-normalised, so dot product = cosine)
        sim_similar = float(np.dot(embeddings[0], embeddings[1]))
        sim_dissimilar = float(np.dot(embeddings[2], embeddings[3]))

        assert sim_similar > 0.5, (
            f"Expected cosine similarity > 0.5 for related texts, got {sim_similar}"
        )
        assert sim_dissimilar < 0.3, (
            f"Expected cosine similarity < 0.3 for unrelated texts, got {sim_dissimilar}"
        )
