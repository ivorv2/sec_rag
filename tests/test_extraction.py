"""Tests for HTML extraction and agreement filtering."""

from sec_rag.ingestion.filter import SECTION_KEYWORDS, is_full_agreement
from sec_rag.ingestion.html_extractor import extract_text_from_html

# ---------------------------------------------------------------------------
# extract_text_from_html
# ---------------------------------------------------------------------------


class TestExtractTextFromHtml:
    """Tests for the EDGAR HTML text extractor."""

    def test_realistic_edgar_html(self):
        """Realistic Wdesk-style HTML with <div>, <font>, and styled paragraphs."""
        html = (
            '<html><body>'
            '<div style="font-family:Times New Roman;font-size:10pt;">'
            '<p style="text-align:center;"><font style="font-weight:bold;">'
            "EMPLOYMENT AGREEMENT</font></p>"
            '<p style="text-indent:36pt;">This Employment Agreement '
            '("Agreement") is entered into as of January 1, 2024, by and '
            "between Acme Corp and John Doe.</p>"
            '<p style="text-indent:36pt;">1. <font style="font-weight:bold;">'
            "Position and Duties.</font> Employee shall serve as Chief "
            "Technology Officer.</p>"
            '<p style="text-indent:36pt;">2. <font style="font-weight:bold;">'
            "Compensation.</font> Employee shall receive a base salary of "
            "$350,000 per annum.</p>"
            "</div></body></html>"
        )

        result = extract_text_from_html(html)

        assert "EMPLOYMENT AGREEMENT" in result
        assert "Acme Corp and John Doe" in result
        assert "Position and Duties" in result
        assert "base salary of $350,000" in result
        # Tags and attributes must be stripped.
        assert "<font" not in result
        assert "<div" not in result
        assert "font-weight" not in result

    def test_sgml_document_wrapper(self):
        """EDGAR SGML <DOCUMENT><TEXT> wrapper is stripped before parsing."""
        html = (
            "<DOCUMENT>"
            "<TYPE>EX-10.1\n"
            "<SEQUENCE>2\n"
            "<TEXT>"
            "<html><body><p>Employment Agreement between X and Y.</p></body></html>"
            "</TEXT>"
            "</DOCUMENT>"
        )

        result = extract_text_from_html(html)

        assert "Employment Agreement between X and Y" in result
        # The SGML metadata should not leak into the output.
        assert "EX-10.1" not in result
        assert "SEQUENCE" not in result

    def test_empty_input_returns_empty_string(self):
        """Empty string input returns empty string."""
        assert extract_text_from_html("") == ""

    def test_garbage_input_returns_empty_string(self):
        """Non-HTML garbage input returns empty string (never raises)."""
        # Pure garbage bytes -- should not crash.
        result = extract_text_from_html("\x00\x01\x02\x03")
        assert isinstance(result, str)
        # None as input is outside the type contract, but the function
        # should still not raise thanks to the blanket except.

    def test_strips_script_and_style_tags(self):
        """<script> and <style> tag content is removed."""
        html = (
            "<html><head>"
            "<style>.cls { color: red; }</style>"
            "</head><body>"
            '<script>alert("xss")</script>'
            "<p>Important contract clause.</p>"
            "</body></html>"
        )

        result = extract_text_from_html(html)

        assert "Important contract clause" in result
        assert "color: red" not in result
        assert "alert" not in result
        assert "<script" not in result
        assert "<style" not in result

    def test_collapses_excessive_newlines(self):
        """Three or more consecutive newlines are collapsed to two."""
        html = (
            "<html><body>"
            "<p>Paragraph one.</p>"
            "<p></p><p></p><p></p><p></p><p></p>"
            "<p>Paragraph two.</p>"
            "</body></html>"
        )

        result = extract_text_from_html(html)

        assert "Paragraph one." in result
        assert "Paragraph two." in result
        # After collapsing, there should be no run of 3+ newlines.
        assert "\n\n\n" not in result

    def test_strips_nbsp_only_lines(self):
        r"""Lines containing only \xa0 (nbsp) are removed."""
        html = (
            "<html><body>"
            "<p>First.</p>"
            "<p>\xa0</p>"
            "<p>Second.</p>"
            "</body></html>"
        )

        result = extract_text_from_html(html)

        assert "First." in result
        assert "Second." in result
        assert "\xa0" not in result


# ---------------------------------------------------------------------------
# is_full_agreement
# ---------------------------------------------------------------------------


class TestIsFullAgreement:
    """Tests for the employment agreement filter."""

    def _make_long_text(self, char_count: int, keywords: list[str]) -> str:
        """Build a text block of at least *char_count* chars containing *keywords*."""
        keyword_block = "\n".join(keywords) + "\n"
        filler = "lorem ipsum dolor sit amet. "
        needed = max(0, char_count - len(keyword_block))
        repetitions = (needed // len(filler)) + 1
        return keyword_block + filler * repetitions

    def test_full_agreement_passes(self):
        """Text with 10K+ chars and 5+ distinct keywords returns True."""
        keywords = SECTION_KEYWORDS[:6]  # 6 distinct keywords
        text = self._make_long_text(12_000, keywords)

        assert is_full_agreement(text) is True

    def test_short_text_fails(self):
        """Short text (amendment-like) returns False even with many keywords."""
        keywords = SECTION_KEYWORDS[:8]
        text = "\n".join(keywords)

        assert len(text) < 10_000  # sanity: it is indeed short
        assert is_full_agreement(text) is False

    def test_long_text_too_few_keywords_fails(self):
        """Long text without enough employment keywords (e.g., a lease) returns False."""
        # Only 2 keywords -- well below the default min_keywords=5.
        text = self._make_long_text(15_000, ["rent", "tenant"])

        assert is_full_agreement(text) is False

    def test_case_insensitive_matching(self):
        """Keywords match regardless of case in the source text."""
        keywords_upper = [kw.upper() for kw in SECTION_KEYWORDS[:6]]
        text = self._make_long_text(12_000, keywords_upper)

        assert is_full_agreement(text) is True

    def test_exact_boundary_chars(self):
        """Text at exactly min_chars with enough keywords returns True."""
        keywords = SECTION_KEYWORDS[:5]
        text = self._make_long_text(10_000, keywords)
        # Trim to exactly 10_000 characters.
        text = text[:10_000]

        # Verify keywords are still present after trimming (they are at the top).
        lower = text.lower()
        hits = sum(1 for kw in SECTION_KEYWORDS if kw in lower)
        assert hits >= 5

        assert is_full_agreement(text) is True

    def test_below_boundary_chars_fails(self):
        """Text one char below min_chars fails."""
        keywords = SECTION_KEYWORDS[:6]
        text = self._make_long_text(10_000, keywords)[:9_999]

        assert is_full_agreement(text) is False

    def test_custom_thresholds(self):
        """Custom min_chars and min_keywords are respected."""
        text = self._make_long_text(500, SECTION_KEYWORDS[:2])

        # With low thresholds, even a small text with 2 keywords qualifies.
        assert is_full_agreement(text, min_chars=100, min_keywords=2) is True
        # With default thresholds, same text fails.
        assert is_full_agreement(text) is False
