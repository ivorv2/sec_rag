"""Extract clean text from SEC EDGAR HTML filings."""

from __future__ import annotations

import re

import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger(__name__)

_SGML_TEXT_RE = re.compile(
    r"<TEXT>(.*?)</TEXT>",
    re.DOTALL | re.IGNORECASE,
)

_COLLAPSE_NEWLINES_RE = re.compile(r"\n{3,}")


def extract_text_from_html(raw_html: str) -> str:
    """Return cleaned text from an EDGAR HTML exhibit.

    Strips SGML wrappers, removes script/style tags, normalises whitespace,
    and preserves paragraph boundaries as double newlines.

    Never raises -- returns an empty string on garbage / empty input.
    """
    try:
        html = raw_html

        # 1. Strip SGML <DOCUMENT>/<TEXT> wrapper if present.
        match = _SGML_TEXT_RE.search(html)
        if match:
            html = match.group(1)

        # 2. Parse with BeautifulSoup + lxml.
        soup = BeautifulSoup(html, "lxml")

        # 3. Remove <script> and <style> elements.
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()

        # 4. Get text with newline separator.
        text = soup.get_text(separator="\n")

        # 5. Strip each line, drop lines that are only whitespace / nbsp.
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line and line != "\xa0"]

        text = "\n".join(lines)

        # 6. Collapse 3+ consecutive newlines to 2.
        text = _COLLAPSE_NEWLINES_RE.sub("\n\n", text)

        return text.strip()
    except Exception:
        logger.debug("HTML extraction failed", exc_info=True)
        return ""
