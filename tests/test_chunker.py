"""Tests for the keyword-based employment agreement chunker."""

from datetime import date

import pytest

from sec_rag.ingestion.chunker import (
    _detect_section_type,
    _split_oversized,
    chunk_document,
)
from sec_rag.models.documents import Document, SectionType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_document() -> Document:
    """Minimal document for simple tests."""
    return Document(
        accession_number="0001234567-24-000001",
        exhibit_number="EX-10.1",
        company_name="Acme Corp",
        cik="1234567",
        filing_date=date(2024, 6, 15),
        source_url="https://www.sec.gov/Archives/edgar/data/1234567/example.htm",
        raw_html="<html></html>",
        extracted_text="Some general text.",
    )


def _make_document(text: str) -> Document:
    """Helper to create a document with custom extracted_text."""
    return Document(
        accession_number="0001234567-24-000001",
        exhibit_number="EX-10.1",
        company_name="Acme Corp",
        cik="1234567",
        filing_date=date(2024, 6, 15),
        source_url="https://www.sec.gov/Archives/edgar/data/1234567/example.htm",
        raw_html="<html></html>",
        extracted_text=text,
    )


def _build_full_agreement() -> str:
    """Build a realistic ~12K char employment agreement with multiple sections."""
    sections = []

    # Preamble (GENERAL)
    sections.append(
        "EMPLOYMENT AGREEMENT\n\n"
        "This Employment Agreement (this \"Agreement\") is entered into as of January 1, 2024, "
        "by and between Acme Corporation, a Delaware corporation (the \"Company\"), and "
        "John Smith (the \"Executive\").\n\n"
        "WHEREAS, the Company desires to employ the Executive, and the Executive desires to "
        "accept such employment, on the terms and conditions set forth herein.\n\n"
        "NOW, THEREFORE, in consideration of the mutual covenants and agreements set forth "
        "herein, and for other good and valuable consideration, the receipt and sufficiency of "
        "which are hereby acknowledged, the parties agree as follows:"
    )

    # Section 1: Term
    sections.append(
        "1. TERM OF EMPLOYMENT\n\n"
        "The term of this Agreement shall commence on January 1, 2024 (the \"Commencement "
        "Date\") and shall continue for a period of three (3) years, unless earlier terminated "
        "in accordance with Section 5 hereof. The term shall automatically renew for successive "
        "one-year periods unless either party provides written notice of non-renewal at least "
        "ninety (90) days prior to the end of the then-current term."
    )

    # Section 2: Duties
    sections.append(
        "2. DUTIES AND RESPONSIBILITIES\n\n"
        "The Executive shall serve as Chief Technology Officer of the Company and shall report "
        "directly to the Chief Executive Officer. The Executive shall devote substantially all "
        "of the Executive's business time, attention, skill and efforts to the performance of "
        "the Executive's duties hereunder. The Executive shall perform such duties as are "
        "customarily associated with the position of Chief Technology Officer, including but "
        "not limited to:\n\n"
        "(a) Overseeing all technology operations and evaluating them according to established "
        "goals;\n\n"
        "(b) Devising and establishing the Company's IT policies and systems;\n\n"
        "(c) Analyzing the business requirements of all departments to determine their "
        "technology needs;\n\n"
        "(d) Managing the Company's technology budget and ensuring cost-effectiveness;\n\n"
        "(e) Leading the engineering team and fostering a culture of innovation."
    )

    # Section 3: Compensation (long section to test splitting)
    comp_paras = []
    comp_paras.append(
        "3. COMPENSATION\n\n"
        "3.1 Base Salary. The Company shall pay the Executive an annual base salary of Three "
        "Hundred Fifty Thousand Dollars ($350,000) (the \"Base Salary\"), payable in accordance "
        "with the Company's standard payroll practices. The Base Salary shall be reviewed "
        "annually by the Board of Directors and may be increased (but not decreased) in the "
        "Board's sole discretion."
    )
    comp_paras.append(
        "3.2 Annual Bonus. The Executive shall be eligible to receive an annual performance "
        "bonus (the \"Annual Bonus\") with a target amount equal to fifty percent (50%) of "
        "the Executive's then-current Base Salary. The actual amount of the Annual Bonus "
        "shall be determined by the Board based on the achievement of individual and company "
        "performance objectives established by the Board. The Annual Bonus, if earned, shall "
        "be paid within sixty (60) days following the end of the fiscal year to which it "
        "relates, subject to the Executive's continued employment through the payment date."
    )
    comp_paras.append(
        "3.3 Equity Awards. Subject to approval by the Board, the Executive shall be granted "
        "an initial equity award of 100,000 restricted stock units (RSUs) under the Company's "
        "2024 Equity Incentive Plan. The RSUs shall vest in equal annual installments over a "
        "four-year period beginning on the first anniversary of the Commencement Date, subject "
        "to the Executive's continued employment. Additional equity awards may be granted from "
        "time to time in the Board's discretion."
    )
    sections.append("\n\n".join(comp_paras))

    # Section 4: Benefits
    sections.append(
        "4. BENEFITS\n\n"
        "4.1 Health Benefits. The Executive shall be entitled to participate in all employee "
        "benefit plans, programs and arrangements made available generally to the Company's "
        "senior executives, including medical, dental, vision, life insurance and disability "
        "insurance plans, subject to the terms and conditions of such plans.\n\n"
        "4.2 Retirement Benefits. The Executive shall be eligible to participate in the "
        "Company's 401(k) plan and any other retirement plans available to senior executives.\n\n"
        "4.3 Vacation. The Executive shall be entitled to four (4) weeks of paid vacation per "
        "calendar year, in addition to Company holidays, to be taken at times mutually "
        "convenient to the Executive and the Company.\n\n"
        "4.4 Expense Reimbursement. The Company shall reimburse the Executive for all "
        "reasonable and documented business expenses incurred in the performance of the "
        "Executive's duties, subject to the Company's expense reimbursement policies."
    )

    # Section 5: Termination
    sections.append(
        "5. TERMINATION\n\n"
        "5.1 Termination by the Company for Cause. The Company may terminate the Executive's "
        "employment for Cause at any time upon written notice specifying in reasonable detail "
        "the basis for such termination. For purposes of this Agreement, \"Cause\" shall mean: "
        "(a) the Executive's material breach of this Agreement; (b) the Executive's conviction "
        "of a felony or crime involving moral turpitude; (c) the Executive's gross negligence "
        "or willful misconduct in the performance of duties; or (d) the Executive's material "
        "violation of any Company policy.\n\n"
        "5.2 Termination Without Cause. The Company may terminate the Executive's employment "
        "without Cause at any time upon thirty (30) days' prior written notice.\n\n"
        "5.3 Resignation. The Executive may resign from employment at any time upon thirty "
        "(30) days' prior written notice to the Company.\n\n"
        "5.4 Severance. In the event of termination without Cause, the Executive shall be "
        "entitled to receive severance pay equal to twelve (12) months of Base Salary, "
        "payable in accordance with the Company's standard payroll practices."
    )

    # Section 6: Non-Competition
    sections.append(
        "6. NON-COMPETITION AND NON-SOLICITATION\n\n"
        "6.1 Non-Competition. During the term of employment and for a period of twelve (12) "
        "months following the termination of employment for any reason, the Executive shall "
        "not, directly or indirectly, engage in or assist any business that competes with the "
        "Company's business within a fifty (50) mile radius of any office of the Company.\n\n"
        "6.2 Non-Solicitation. During the same restricted period, the Executive shall not "
        "solicit any employee, consultant, or contractor of the Company to leave the Company's "
        "service, or solicit any customer or client of the Company for the purpose of providing "
        "products or services competitive with those of the Company."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Test 1: Full agreement with multiple sections
# ---------------------------------------------------------------------------


class TestFullAgreement:
    def test_produces_multiple_chunks_with_correct_section_types(self):
        doc = _make_document(_build_full_agreement())
        chunks = chunk_document(doc)

        assert len(chunks) >= 5, f"Expected at least 5 chunks, got {len(chunks)}"

        section_types = [c.metadata.section_type for c in chunks]

        # The preamble should be GENERAL
        assert section_types[0] == SectionType.GENERAL

        # Must find these section types somewhere in the output
        assert SectionType.TERM_AND_RENEWAL in section_types
        assert SectionType.DUTIES in section_types
        assert SectionType.COMPENSATION in section_types
        assert SectionType.BENEFITS in section_types
        assert SectionType.TERMINATION in section_types
        assert SectionType.NON_COMPETE in section_types

    def test_chunk_text_is_nonempty(self):
        doc = _make_document(_build_full_agreement())
        chunks = chunk_document(doc)

        for chunk in chunks:
            assert len(chunk.text.strip()) > 0, (
                f"Chunk {chunk.metadata.chunk_index} has empty text"
            )

    def test_chunk_indices_are_sequential(self):
        doc = _make_document(_build_full_agreement())
        chunks = chunk_document(doc)

        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Test 2: Oversized chunk splitting
# ---------------------------------------------------------------------------


class TestOversizedChunkSplitting:
    def test_split_at_paragraph_boundaries(self):
        # Build a section with 3000+ chars using multiple paragraphs
        para = "The Executive acknowledges and agrees that this paragraph is important. " * 15
        long_text = (
            "1. COMPENSATION\n\n"
            + para.strip()
            + "\n\n"
            + para.strip()
            + "\n\n"
            + para.strip()
            + "\n\n"
            + para.strip()
        )
        doc = _make_document(long_text)
        assert len(long_text) > 3000

        chunks = chunk_document(doc, max_chunk_chars=2000)

        # Should produce multiple chunks
        assert len(chunks) >= 2

        # None should exceed max size
        for chunk in chunks:
            assert chunk.char_count <= 2000, (
                f"Chunk {chunk.metadata.chunk_index} has {chunk.char_count} chars"
            )

    def test_split_at_sentence_boundaries_when_paragraph_too_large(self):
        # Single huge paragraph (no double newlines)
        sentence = "The Executive shall comply with all applicable rules and regulations. "
        single_para = "1. COMPENSATION\n\n" + sentence * 50  # ~3400 chars
        doc = _make_document(single_para)

        chunks = chunk_document(doc, max_chunk_chars=1000)
        assert len(chunks) >= 2

        for chunk in chunks:
            assert chunk.char_count <= 1000

    def test_split_oversized_function_directly(self):
        text = "A" * 5000
        parts = _split_oversized(text, 2000)
        assert len(parts) >= 3
        for part in parts:
            assert len(part) <= 2000

    def test_split_oversized_preserves_content(self):
        para1 = "First paragraph content here."
        para2 = "Second paragraph content here."
        text = para1 + "\n\n" + para2
        parts = _split_oversized(text, 5000)
        assert len(parts) == 1
        assert parts[0] == text

    def test_split_oversized_returns_original_when_within_limit(self):
        text = "Short text."
        assert _split_oversized(text, 100) == ["Short text."]


# ---------------------------------------------------------------------------
# Test 3: Header detection — various formats
# ---------------------------------------------------------------------------


class TestHeaderDetection:
    def test_numbered_with_keyword(self):
        assert _detect_section_type("3.1 Base Salary") == SectionType.COMPENSATION

    def test_all_caps_keyword(self):
        assert _detect_section_type("TERMINATION") == SectionType.TERMINATION

    def test_parenthesized_number_with_keyword(self):
        assert _detect_section_type("(a) Non-Competition") == SectionType.NON_COMPETE

    def test_section_prefix_with_keyword(self):
        assert (
            _detect_section_type("SECTION 5. CONFIDENTIALITY")
            == SectionType.CONFIDENTIALITY
        )

    def test_numbered_duties(self):
        assert (
            _detect_section_type("2. DUTIES AND RESPONSIBILITIES")
            == SectionType.DUTIES
        )

    def test_mixed_case_with_numbering(self):
        assert _detect_section_type("4.1 Benefits") == SectionType.BENEFITS

    def test_governing_law_header(self):
        assert _detect_section_type("9. GOVERNING LAW") == SectionType.GOVERNING_LAW

    def test_change_of_control_header(self):
        assert (
            _detect_section_type("7. CHANGE IN CONTROL")
            == SectionType.CHANGE_OF_CONTROL
        )

    def test_indemnification_header(self):
        assert (
            _detect_section_type("10. INDEMNIFICATION")
            == SectionType.INDEMNIFICATION
        )

    def test_arbitration_header(self):
        assert (
            _detect_section_type("8. DISPUTE RESOLUTION")
            == SectionType.ARBITRATION
        )

    def test_equity_header(self):
        assert _detect_section_type("3.3 Equity Awards") == SectionType.EQUITY

    def test_intellectual_property_header(self):
        assert (
            _detect_section_type("11. INTELLECTUAL PROPERTY")
            == SectionType.INTELLECTUAL_PROPERTY
        )


# ---------------------------------------------------------------------------
# Test 4: Non-header lines should NOT be detected as headers
# ---------------------------------------------------------------------------


class TestNonHeaderDetection:
    def test_long_line_with_keyword_not_detected(self):
        long_line = (
            "The Executive's salary shall be determined by the Board of Directors in its sole "
            "discretion based on the Executive's performance and market conditions."
        )
        assert len(long_line) >= 80
        assert _detect_section_type(long_line) is None

    def test_body_text_mentioning_keyword_not_detected(self):
        # A line under 80 chars but no numbering and not mostly uppercase
        body = "the salary will be reviewed annually by management"
        assert _detect_section_type(body) is None

    def test_empty_line_not_detected(self):
        assert _detect_section_type("") is None

    def test_whitespace_only_not_detected(self):
        assert _detect_section_type("   ") is None

    def test_short_line_no_keyword(self):
        assert _detect_section_type("1. Introduction") is None

    def test_keyword_in_lowercase_no_numbering(self):
        # Under 80 chars, has keyword, but not numbered and not >40% uppercase
        line = "the compensation package includes several components"
        assert _detect_section_type(line) is None


# ---------------------------------------------------------------------------
# Test 5: Empty / minimal document
# ---------------------------------------------------------------------------


class TestEmptyMinimalDocument:
    def test_empty_text_returns_one_general_chunk_with_marker(self):
        """m06/m12: Empty text fallback uses unambiguous marker, not raw company name."""
        doc = _make_document("")
        chunks = chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.section_type == SectionType.GENERAL
        assert chunks[0].text == f"[No extractable text for {doc.company_name}]"

    def test_whitespace_only_returns_one_general_chunk_with_marker(self):
        """m06/m12: Whitespace-only text fallback uses unambiguous marker."""
        doc = _make_document("   \n\n   ")
        chunks = chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.section_type == SectionType.GENERAL
        assert chunks[0].text == f"[No extractable text for {doc.company_name}]"

    def test_minimal_text_no_headers(self):
        doc = _make_document("This is a short agreement.")
        chunks = chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.section_type == SectionType.GENERAL
        assert chunks[0].text == "This is a short agreement."


# ---------------------------------------------------------------------------
# Test 6: Deterministic chunk IDs
# ---------------------------------------------------------------------------


class TestDeterministicChunkIds:
    def test_same_document_produces_same_ids(self):
        text = _build_full_agreement()
        doc1 = _make_document(text)
        doc2 = _make_document(text)

        chunks1 = chunk_document(doc1)
        chunks2 = chunk_document(doc2)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id

    def test_different_accession_produces_different_ids(self):
        text = "1. COMPENSATION\n\nBase salary is $100,000."
        doc1 = _make_document(text)
        doc2 = Document(
            accession_number="9999999999-24-999999",
            exhibit_number="EX-10.1",
            company_name="Acme Corp",
            cik="1234567",
            filing_date=date(2024, 6, 15),
            source_url="https://www.sec.gov/example.htm",
            raw_html="<html></html>",
            extracted_text=text,
        )

        chunks1 = chunk_document(doc1)
        chunks2 = chunk_document(doc2)

        # Same number of chunks but different IDs
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id != c2.chunk_id

    def test_chunk_id_is_valid_uuid(self):
        import uuid

        doc = _make_document("1. COMPENSATION\n\nSalary details here.")
        chunks = chunk_document(doc)

        for chunk in chunks:
            parsed = uuid.UUID(chunk.chunk_id)
            assert parsed.version == 5


# ---------------------------------------------------------------------------
# Test 7: Chunk metadata correctness
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_all_chunks_carry_document_metadata(self):
        doc = _make_document(_build_full_agreement())
        chunks = chunk_document(doc)

        for chunk in chunks:
            assert chunk.metadata.company_name == "Acme Corp"
            assert chunk.metadata.cik == "1234567"
            assert chunk.metadata.filing_date == date(2024, 6, 15)
            assert chunk.metadata.exhibit_number == "EX-10.1"
            assert chunk.metadata.accession_number == "0001234567-24-000001"
            assert chunk.metadata.source_url == (
                "https://www.sec.gov/Archives/edgar/data/1234567/example.htm"
            )

    def test_char_count_matches_text_length(self):
        doc = _make_document(_build_full_agreement())
        chunks = chunk_document(doc)

        for chunk in chunks:
            assert chunk.char_count == len(chunk.text)

    def test_section_type_in_metadata_matches_expected(self):
        text = "Preamble text.\n\n5. TERMINATION\n\nYou can be fired."
        doc = _make_document(text)
        chunks = chunk_document(doc)

        assert chunks[0].metadata.section_type == SectionType.GENERAL
        assert chunks[1].metadata.section_type == SectionType.TERMINATION
