from datetime import date

import pytest

from sec_rag.models.documents import Chunk, ChunkMetadata, Document, SectionType


@pytest.fixture
def sample_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        company_name="Test Corp",
        cik="1234567",
        filing_date=date(2024, 1, 15),
        exhibit_number="EX-10.1",
        accession_number="0001234567-24-000001",
        section_type=SectionType.COMPENSATION,
        chunk_index=0,
        source_url="https://www.sec.gov/Archives/edgar/data/1234567/example.htm",
    )


@pytest.fixture
def sample_chunk(sample_metadata: ChunkMetadata) -> Chunk:
    return Chunk(
        chunk_id="test-chunk-001",
        text="Employee shall receive a base salary of $150,000 per annum.",
        metadata=sample_metadata,
    )


@pytest.fixture
def sample_document() -> Document:
    return Document(
        accession_number="0001234567-24-000001",
        exhibit_number="EX-10.1",
        company_name="Test Corp",
        cik="1234567",
        filing_date=date(2024, 1, 15),
        source_url="https://www.sec.gov/Archives/edgar/data/1234567/example.htm",
        raw_html="<html><body><p>Employment Agreement</p></body></html>",
        extracted_text="Employment Agreement\nBase Salary: $150,000",
    )
