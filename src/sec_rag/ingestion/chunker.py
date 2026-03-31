"""Keyword-based chunker for SEC EDGAR employment agreements.

Splits extracted text into clause-level chunks with SectionType labels.
Handles oversized chunks by splitting at paragraph then sentence boundaries.
"""

import re
import uuid

from sec_rag.ingestion.keywords import KEYWORD_TO_SECTION
from sec_rag.models.documents import Chunk, ChunkMetadata, Document, SectionType

# A line starts with numbering if it matches patterns like:
# "3.1 ", "3.1. ", "(a) ", "SECTION 4 ", "SECTION 4. ", "1) ", "1. ", "(iv) "
# Uses alternation to avoid matching plain words like "the" or "salary".
_NUMBERING_RE = re.compile(
    r"^(?:"
    r"\([\da-zA-Z]{1,4}\)"          # (a), (iv), (1), (A)
    r"|\d[\d.]*[\.\)]?"             # 1, 1., 1.1, 3.1., 1)
    r"|[Ss][Ee][Cc][Tt][Ii][Oo][Nn]\s+\d"  # SECTION 4, Section 4
    r"|[A-Z]\."                     # A., B.
    r")\s"
)

# Sentence boundary: split AFTER ". " when followed by an uppercase letter.
# Uses lookbehind so the period stays attached to the preceding sentence.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=\.) (?=[A-Z])")


def _detect_section_type(line: str) -> SectionType | None:
    """Check if a line is a section header. Return SectionType if yes, None if no.

    A line is a section header if ALL of:
    1. It contains a keyword from KEYWORD_TO_SECTION (case-insensitive)
    2. It is shorter than 80 characters
    3. It starts with numbering OR more than 40% of its characters are uppercase
    """
    stripped = line.strip()
    if not stripped or len(stripped) >= 80:
        return None

    # Check condition 3: numbering or >40% uppercase
    has_numbering = bool(_NUMBERING_RE.match(stripped))
    if not has_numbering:
        alpha_chars = [c for c in stripped if c.isalpha()]
        if not alpha_chars:
            return None
        uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if uppercase_ratio <= 0.4:
            return None

    # Check condition 1: keyword match
    lower_line = stripped.lower()
    for keyword, section_type in KEYWORD_TO_SECTION:
        if keyword in lower_line:
            return section_type

    return None


def _split_paragraph_by_sentences(para: str, max_chars: int) -> list[str]:
    """Split a single oversized paragraph at sentence boundaries with hard-split fallback."""
    sentences = _SENTENCE_BOUNDARY_RE.split(para)
    chunks: list[str] = []
    current_buf = ""
    for sentence in sentences:
        candidate = sentence if not current_buf else current_buf + " " + sentence
        if len(candidate) <= max_chars:
            current_buf = candidate
        else:
            if current_buf:
                chunks.append(current_buf)
            if len(sentence) > max_chars:
                while sentence:
                    chunks.append(sentence[:max_chars])
                    sentence = sentence[max_chars:]
                current_buf = ""
            else:
                current_buf = sentence
    if current_buf:
        chunks.append(current_buf)
    return chunks


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split text exceeding max_chars at paragraph boundaries, then sentence boundaries.

    1. Split at paragraph boundaries (double newline).
    2. If any paragraph still exceeds max_chars, split at sentence boundaries.
    3. If a sentence still exceeds max_chars, hard-split at max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    result: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            result.append(para)
        else:
            result.extend(_split_paragraph_by_sentences(para, max_chars))

    return result if result else [text]


def chunk_document(document: Document, max_chunk_chars: int = 2000) -> list[Chunk]:
    """Split a document's extracted_text into clause-level chunks with section labels.

    Params:
        document: Document with extracted_text to chunk.
        max_chunk_chars: Maximum character count per chunk (default 2000).

    Returns:
        list[Chunk] -- always non-empty (minimum one GENERAL chunk).
    """
    text = document.extracted_text
    lines = text.split("\n")

    # Walk lines, accumulating sections
    sections: list[tuple[SectionType, list[str]]] = []
    current_type = SectionType.GENERAL
    current_lines: list[str] = []

    for line in lines:
        detected = _detect_section_type(line)
        if detected is not None:
            # Save accumulated lines as a section
            sections.append((current_type, current_lines))
            current_type = detected
            current_lines = [line]
        else:
            current_lines.append(line)

    # Don't forget the last section
    sections.append((current_type, current_lines))

    # Build raw chunks (section_type, text) -- filter out empty sections
    raw_chunks: list[tuple[SectionType, str]] = []
    for section_type, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if section_text:
            raw_chunks.append((section_type, section_text))

    # Split oversized chunks
    final_chunks: list[tuple[SectionType, str]] = []
    for section_type, section_text in raw_chunks:
        if len(section_text) <= max_chunk_chars:
            final_chunks.append((section_type, section_text))
        else:
            for sub_text in _split_oversized(section_text, max_chunk_chars):
                final_chunks.append((section_type, sub_text))

    # Guarantee at least one chunk — use the raw text (trimmed) so downstream
    # consumers never receive an empty-text chunk that corrupts LLM context.
    if not final_chunks:
        stripped = text.strip()
        if stripped:
            fallback_text = stripped[:max_chunk_chars]
        else:
            fallback_text = f"[No extractable text for {document.company_name}]"
        final_chunks.append((SectionType.GENERAL, fallback_text))

    # Build Chunk objects with deterministic IDs
    result: list[Chunk] = []
    for idx, (section_type, chunk_text) in enumerate(final_chunks):
        chunk_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{document.accession_number}_{document.exhibit_number}_{idx}",
            )
        )
        metadata = ChunkMetadata(
            company_name=document.company_name,
            cik=document.cik,
            filing_date=document.filing_date,
            exhibit_number=document.exhibit_number,
            accession_number=document.accession_number,
            section_type=section_type,
            chunk_index=idx,
            source_url=document.source_url,
        )
        result.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=metadata,
            )
        )

    return result
