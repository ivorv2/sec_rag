"""Tests for the retrieval pipeline — hybrid search, reranker, and full pipeline.

Uses Qdrant in-memory mode with real embedding models (all-MiniLM-L6-v2,
Qdrant/bm25, cross-encoder/ms-marco-MiniLM-L6-v2). Marked slow because
model loading takes several seconds.
"""

from __future__ import annotations

from datetime import date

import pytest
from qdrant_client import models

from sec_rag.embedding import ChunkEmbedder, SparseEncoder
from sec_rag.ingestion.indexer import QdrantIndexer
from sec_rag.models.documents import Chunk, ChunkMetadata, SectionType
from sec_rag.models.retrieval import RetrievalResult, ScoredChunk
from sec_rag.retrieval.hybrid_search import hybrid_query
from sec_rag.retrieval.pipeline import RetrievalPipeline
from sec_rag.retrieval.reranker import Reranker

pytestmark = pytest.mark.slow

# ── Sample data ─────────────────────────────────────────────────────────────

SAMPLE_CHUNKS_DATA: list[tuple[str, SectionType, str]] = [
    (
        "The Employee shall receive a base salary of $250,000 per annum, "
        "payable in bi-weekly installments in accordance with the Company's "
        "standard payroll practices.",
        SectionType.COMPENSATION,
        "Acme Corp",
    ),
    (
        "Employee shall serve as Chief Technology Officer and shall report "
        "directly to the Chief Executive Officer. Employee's duties shall "
        "include managing all technology operations and strategy.",
        SectionType.DUTIES,
        "Acme Corp",
    ),
    (
        "Either party may terminate this Agreement upon 30 days written notice. "
        "In the event of termination without cause, Employee shall receive "
        "12 months of base salary as severance payment.",
        SectionType.TERMINATION,
        "Acme Corp",
    ),
    (
        "For a period of 12 months following termination, Employee shall not "
        "directly or indirectly engage in any business that competes with the "
        "Company within the United States. This non-compete restriction applies "
        "to employment, consulting, and advisory roles.",
        SectionType.NON_COMPETE,
        "Acme Corp",
    ),
    (
        "Employee shall be granted 50,000 stock options under the Company's "
        "2024 Equity Incentive Plan, vesting over four years with a one-year "
        "cliff. The exercise price shall be the fair market value on the date "
        "of grant.",
        SectionType.EQUITY,
        "Acme Corp",
    ),
    (
        "Employee shall be entitled to participate in all health, dental, and "
        "vision insurance plans offered by the Company. The Company shall pay "
        "100% of premiums for Employee and dependents.",
        SectionType.BENEFITS,
        "Acme Corp",
    ),
    (
        "Employee shall be entitled to 25 days of paid vacation per calendar "
        "year, accruing on a monthly basis. Unused vacation days may be carried "
        "over to the following year up to a maximum of 10 days.",
        SectionType.VACATION,
        "Acme Corp",
    ),
    (
        "Employee shall be eligible for an annual performance bonus of up to "
        "50% of base salary, based on individual and company performance "
        "metrics as determined by the Board of Directors.",
        SectionType.BONUS,
        "Acme Corp",
    ),
    (
        "In the event of a Change of Control, all unvested equity awards shall "
        "immediately vest and become exercisable. Employee shall receive a lump "
        "sum payment equal to 24 months of base salary.",
        SectionType.CHANGE_OF_CONTROL,
        "Acme Corp",
    ),
    (
        "Employee agrees to maintain confidentiality of all proprietary "
        "information, trade secrets, and business strategies. This obligation "
        "survives termination of this Agreement indefinitely.",
        SectionType.CONFIDENTIALITY,
        "Beta Inc",
    ),
]


def _make_chunks() -> list[Chunk]:
    """Build sample Chunk objects from SAMPLE_CHUNKS_DATA."""
    chunks: list[Chunk] = []
    for i, (text, section, company) in enumerate(SAMPLE_CHUNKS_DATA):
        chunks.append(
            Chunk(
                chunk_id=f"test-retrieval-{i:03d}",
                text=text,
                metadata=ChunkMetadata(
                    company_name=company,
                    cik="0001234567" if company == "Acme Corp" else "0009876543",
                    filing_date=date(2024, 6, 15),
                    exhibit_number="EX-10.1",
                    accession_number="0001234567-24-000042",
                    section_type=section,
                    chunk_index=i,
                    source_url=(
                        "https://www.sec.gov/Archives/edgar/data/1234567/ex10-1.htm"
                    ),
                ),
            )
        )
    return chunks


# ── Module-scoped fixtures (avoid reloading models per test) ────────────────


@pytest.fixture(scope="module")
def embedder() -> ChunkEmbedder:
    return ChunkEmbedder(model_name="all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def sparse_encoder() -> SparseEncoder:
    return SparseEncoder(model_name="Qdrant/bm25")


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return Reranker(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def indexed_env(
    embedder: ChunkEmbedder, sparse_encoder: SparseEncoder
) -> tuple[QdrantIndexer, list[Chunk]]:
    """Create an in-memory Qdrant collection with 10 embedded chunks."""
    chunks = _make_chunks()
    indexer = QdrantIndexer.from_memory(
        collection_name="retrieval_test", dense_dim=embedder.dimension
    )
    indexer.ensure_collection()

    texts = [c.text for c in chunks]
    dense_vectors = embedder.embed_texts(texts)
    sparse_vectors = sparse_encoder.encode(texts)
    indexer.upsert_chunks(chunks, dense_vectors, sparse_vectors)

    return indexer, chunks


@pytest.fixture(scope="module")
def pipeline(
    indexed_env: tuple[QdrantIndexer, list[Chunk]],
    embedder: ChunkEmbedder,
    sparse_encoder: SparseEncoder,
    reranker: Reranker,
) -> RetrievalPipeline:
    """Create a RetrievalPipeline backed by the indexed in-memory collection."""
    indexer, _ = indexed_env
    return RetrievalPipeline(
        qdrant_client=indexer.client,
        collection_name=indexer.collection_name,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        reranker=reranker,
        dense_limit=50,
        sparse_limit=50,
        rrf_top_k=20,
        rerank_top_k=5,
    )


# ── Empty collection fixture (function-scoped, isolated) ───────────────────


@pytest.fixture
def empty_indexer(embedder: ChunkEmbedder) -> QdrantIndexer:
    """An in-memory indexer with an empty collection."""
    indexer = QdrantIndexer.from_memory(
        collection_name="empty_test", dense_dim=embedder.dimension
    )
    indexer.ensure_collection()
    return indexer


# ── Tests: hybrid_query ─────────────────────────────────────────────────────


class TestHybridQuery:
    def test_returns_results_for_indexed_data(
        self,
        indexed_env: tuple[QdrantIndexer, list[Chunk]],
        embedder: ChunkEmbedder,
        sparse_encoder: SparseEncoder,
    ) -> None:
        """hybrid_query returns non-empty results when querying indexed data."""
        indexer, _ = indexed_env
        query = "what is the employee salary"
        dense = embedder.embed_texts([query])[0].tolist()
        sparse = sparse_encoder.encode([query])[0]

        results = hybrid_query(
            client=indexer.client,
            collection_name=indexer.collection_name,
            dense_vector=dense,
            sparse_vector=sparse,
            rrf_top_k=10,
        )

        assert len(results) > 0
        assert all(isinstance(r, models.ScoredPoint) for r in results)
        assert all(r.payload is not None for r in results)
        assert all("text" in r.payload for r in results)

    def test_section_filter_returns_only_matching_types(
        self,
        indexed_env: tuple[QdrantIndexer, list[Chunk]],
        embedder: ChunkEmbedder,
        sparse_encoder: SparseEncoder,
    ) -> None:
        """hybrid_query with section_filter='non_compete' returns only non-compete chunks."""
        indexer, _ = indexed_env
        query = "employment agreement terms"
        dense = embedder.embed_texts([query])[0].tolist()
        sparse = sparse_encoder.encode([query])[0]

        results = hybrid_query(
            client=indexer.client,
            collection_name=indexer.collection_name,
            dense_vector=dense,
            sparse_vector=sparse,
            section_filter="non_compete",
            rrf_top_k=10,
        )

        assert len(results) >= 1
        for r in results:
            assert r.payload["section_type"] == "non_compete"

    def test_empty_collection_returns_empty(
        self,
        empty_indexer: QdrantIndexer,
        embedder: ChunkEmbedder,
        sparse_encoder: SparseEncoder,
    ) -> None:
        """Querying an empty collection returns an empty list without error."""
        query = "salary compensation"
        dense = embedder.embed_texts([query])[0].tolist()
        sparse = sparse_encoder.encode([query])[0]

        results = hybrid_query(
            client=empty_indexer.client,
            collection_name=empty_indexer.collection_name,
            dense_vector=dense,
            sparse_vector=sparse,
            rrf_top_k=10,
        )

        assert results == []


# ── Tests: Reranker ─────────────────────────────────────────────────────────


class TestReranker:
    def test_rerank_returns_top_k_sorted_by_score(
        self, reranker: Reranker
    ) -> None:
        """Reranker returns top_k chunks sorted by rerank_score descending."""
        meta = ChunkMetadata(
            company_name="Test",
            cik="0001234567",
            filing_date=date(2024, 1, 1),
            exhibit_number="EX-10.1",
            accession_number="0001234567-24-000001",
            section_type=SectionType.GENERAL,
            chunk_index=0,
            source_url="https://example.com",
        )
        chunks = [
            ScoredChunk(
                chunk_id=f"rerank-{i}",
                text=text,
                metadata=meta.model_copy(update={"chunk_index": i}),
            )
            for i, text in enumerate(
                [
                    "The weather is sunny today with clear skies.",
                    "Employee shall receive a base salary of $200,000 per year.",
                    "The cat sat on the mat in the living room.",
                    "Annual compensation includes base pay, bonus, and equity grants.",
                    "Python is a popular programming language.",
                ]
            )
        ]

        result = reranker.rerank("what is the employee compensation", chunks, top_k=3)

        assert len(result) == 3
        # All returned chunks should have rerank_score set (not default 0.0)
        for chunk in result:
            assert chunk.rerank_score != 0.0
        # Scores should be in descending order
        scores = [c.rerank_score for c in result]
        assert scores == sorted(scores, reverse=True)
        # The compensation-related chunks should score higher than irrelevant ones
        top_texts = {c.text for c in result}
        assert any("salary" in t or "compensation" in t for t in top_texts)

    def test_rerank_empty_returns_empty(self, reranker: Reranker) -> None:
        """Reranking an empty list returns an empty list."""
        result = reranker.rerank("any query", [], top_k=5)
        assert result == []

    def test_rerank_fewer_than_top_k_returns_all(
        self, reranker: Reranker
    ) -> None:
        """When len(chunks) < top_k, all chunks are returned scored and sorted."""
        meta = ChunkMetadata(
            company_name="Test",
            cik="0001234567",
            filing_date=date(2024, 1, 1),
            exhibit_number="EX-10.1",
            accession_number="0001234567-24-000001",
            section_type=SectionType.GENERAL,
            chunk_index=0,
            source_url="https://example.com",
        )
        chunks = [
            ScoredChunk(chunk_id="a", text="Employee salary is $100,000.", metadata=meta),
            ScoredChunk(
                chunk_id="b",
                text="The sky is blue.",
                metadata=meta.model_copy(update={"chunk_index": 1}),
            ),
        ]

        result = reranker.rerank("salary", chunks, top_k=10)

        assert len(result) == 2
        scores = [c.rerank_score for c in result]
        assert scores == sorted(scores, reverse=True)


# ── Tests: RetrievalPipeline ───────────────────────────────────────────────


class TestRetrievalPipeline:
    def test_retrieve_returns_retrieval_result_with_correct_length(
        self, pipeline: RetrievalPipeline
    ) -> None:
        """retrieve() returns a RetrievalResult with scored_chunks up to rerank_top_k."""
        result = pipeline.retrieve("what is the base salary")

        assert isinstance(result, RetrievalResult)
        assert len(result.scored_chunks) <= 5
        assert len(result.scored_chunks) > 0
        assert result.total_candidates_before_rerank > 0

    def test_retrieve_chunks_have_valid_metadata(
        self, pipeline: RetrievalPipeline
    ) -> None:
        """Each scored chunk contains fully populated ChunkMetadata."""
        result = pipeline.retrieve("employee duties and responsibilities")

        for chunk in result.scored_chunks:
            meta = chunk.metadata
            assert meta.company_name in ("Acme Corp", "Beta Inc")
            assert len(meta.cik) > 0
            assert isinstance(meta.filing_date, date)
            assert meta.filing_date == date(2024, 6, 15)
            assert meta.exhibit_number == "EX-10.1"
            assert len(meta.accession_number) > 0
            assert isinstance(meta.section_type, SectionType)
            assert isinstance(meta.chunk_index, int)
            assert meta.source_url.startswith("https://")
            assert len(chunk.chunk_id) > 0
            assert len(chunk.text) > 0

    def test_retrieve_non_compete_query_ranks_relevant_chunks_highly(
        self, pipeline: RetrievalPipeline
    ) -> None:
        """A query about non-compete should rank non-compete chunks near the top."""
        result = pipeline.retrieve(
            "non-compete clause restrictions after termination"
        )

        # The non-compete chunk should appear in the top results
        section_types = [c.metadata.section_type for c in result.scored_chunks]
        assert SectionType.NON_COMPETE in section_types
        # Verify it's ranked in the top 3
        non_compete_positions = [
            i for i, st in enumerate(section_types) if st == SectionType.NON_COMPETE
        ]
        assert any(pos < 3 for pos in non_compete_positions)

    def test_retrieve_query_field(self, pipeline: RetrievalPipeline) -> None:
        """RetrievalResult.query matches the input query string."""
        query = "equity stock options vesting schedule"
        result = pipeline.retrieve(query)
        assert result.query == query
