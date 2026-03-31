from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SectionType(StrEnum):
    COMPENSATION = "compensation"
    DUTIES = "duties"
    TERMINATION = "termination"
    NON_COMPETE = "non_compete"
    EQUITY = "equity"
    BENEFITS = "benefits"
    VACATION = "vacation"
    BONUS = "bonus"
    SEVERANCE = "severance"
    CONFIDENTIALITY = "confidentiality"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    GOVERNING_LAW = "governing_law"
    TERM_AND_RENEWAL = "term_and_renewal"
    CHANGE_OF_CONTROL = "change_of_control"
    ARBITRATION = "arbitration"
    INDEMNIFICATION = "indemnification"
    GENERAL = "general"


class ChunkMetadata(BaseModel):
    company_name: str
    cik: str
    filing_date: date
    exhibit_number: str
    accession_number: str
    section_type: SectionType
    chunk_index: int
    source_url: str

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Serialize to Qdrant payload dict. Single source of truth for the
        field schema shared between indexer (write) and pipeline (read)."""
        return {
            "company_name": self.company_name,
            "cik": self.cik,
            "filing_date": self.filing_date.isoformat(),
            "exhibit_number": self.exhibit_number,
            "accession_number": self.accession_number,
            "section_type": self.section_type.value,
            "chunk_index": self.chunk_index,
            "source_url": self.source_url,
        }

    @classmethod
    def from_qdrant_payload(cls, payload: dict[str, Any]) -> "ChunkMetadata":
        """Deserialize from Qdrant payload dict."""
        return cls(
            company_name=payload["company_name"],
            cik=payload["cik"],
            filing_date=date.fromisoformat(payload["filing_date"]),
            exhibit_number=payload["exhibit_number"],
            accession_number=payload["accession_number"],
            section_type=SectionType(payload["section_type"]),
            chunk_index=int(payload["chunk_index"]),
            source_url=payload["source_url"],
        )


class Chunk(BaseModel):
    chunk_id: str = Field(description="Deterministic ID: UUID5 from accession_exhibit_chunkIdx")
    text: str
    metadata: ChunkMetadata
    char_count: int = Field(default=0)

    def model_post_init(self, __context: object) -> None:
        if self.char_count == 0:
            self.char_count = len(self.text)


class Document(BaseModel):
    accession_number: str
    exhibit_number: str
    company_name: str
    cik: str
    filing_date: date
    source_url: str
    raw_html: str
    extracted_text: str
    char_count: int = Field(default=0)
    is_full_agreement: bool = False

    def model_post_init(self, __context: object) -> None:
        if self.char_count == 0:
            self.char_count = len(self.extracted_text)
