"""Segment 1 verification: all types import and instantiate correctly."""

from datetime import date

from sec_rag.config import Settings
from sec_rag.models import (
    AnalysisResult,
    Chunk,
    ChunkMetadata,
    Citation,
    Document,
    InputValidation,
    Obligation,
    QueryState,
    RelevanceGrade,
    RetrievalResult,
    ScoredChunk,
    SectionType,
)


def test_section_type_values():
    assert SectionType.COMPENSATION.value == "compensation"
    assert SectionType.NON_COMPETE.value == "non_compete"
    assert len(SectionType) == 17


def test_chunk_metadata(sample_metadata: ChunkMetadata):
    assert sample_metadata.company_name == "Test Corp"
    assert sample_metadata.section_type == SectionType.COMPENSATION


def test_chunk_auto_char_count(sample_metadata: ChunkMetadata):
    chunk = Chunk(
        chunk_id="test-001",
        text="Hello world",
        metadata=sample_metadata,
    )
    assert chunk.char_count == 11


def test_document_auto_char_count():
    doc = Document(
        accession_number="0001234567-24-000001",
        exhibit_number="EX-10.1",
        company_name="Test Corp",
        cik="1234567",
        filing_date=date(2024, 1, 15),
        source_url="https://example.com",
        raw_html="<html></html>",
        extracted_text="Some extracted text here",
    )
    assert doc.char_count == len("Some extracted text here")


def test_scored_chunk(sample_metadata: ChunkMetadata):
    sc = ScoredChunk(
        chunk_id="test-001",
        text="test",
        metadata=sample_metadata,
        rrf_score=0.85,
    )
    assert sc.rrf_score == 0.85
    assert sc.rerank_score == 0.0


def test_retrieval_result(sample_metadata: ChunkMetadata):
    rr = RetrievalResult(
        query="test query",
        scored_chunks=[
            ScoredChunk(chunk_id="1", text="t", metadata=sample_metadata),
        ],
    )
    assert len(rr.scored_chunks) == 1
    assert rr.total_candidates_before_rerank == 0


def test_citation():
    c = Citation(
        chunk_id="c1",
        company_name="Acme",
        section_type="compensation",
        excerpt="Base salary of $100k",
    )
    assert c.excerpt == "Base salary of $100k"


def test_obligation():
    ob = Obligation(
        obligation_type="compensation",
        party="employer",
        description="Pay base salary",
        citations=[
            Citation(
                chunk_id="c1",
                company_name="Acme",
                section_type="compensation",
                excerpt="shall pay",
            )
        ],
    )
    assert ob.party == "employer"
    assert ob.conditions is None


def test_analysis_result():
    ar = AnalysisResult(
        query="What are the compensation terms?",
        obligations=[],
        summary="No obligations found.",
        confidence=0.0,
        source_count=0,
    )
    assert ar.confidence == 0.0


def test_relevance_grade():
    rg = RelevanceGrade(is_relevant=True, reasoning="Context matches query", score=0.9)
    assert rg.is_relevant is True


def test_input_validation():
    iv = InputValidation(is_valid=False, reason="Off-topic query")
    assert iv.is_valid is False


def test_query_state_structure():
    state: QueryState = {
        "original_query": "What is the salary?",
        "query_type": "extraction",
        "current_query": "What is the salary?",
        "retrieval_result": None,
        "relevance_grade": None,
        "analysis_result": None,
        "retry_count": 0,
        "error": None,
    }
    assert state["retry_count"] == 0
    assert state["error"] is None


def test_settings_defaults():
    s = Settings(anthropic_api_key="test-key")
    assert s.llm_provider == "anthropic"
    assert s.max_retries == 2
    assert s.max_chunk_chars == 2_000
