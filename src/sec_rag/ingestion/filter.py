"""Filter employment agreements from SEC EDGAR exhibits."""

from __future__ import annotations

from sec_rag.ingestion.keywords import SECTION_KEYWORDS


def is_full_agreement(
    text: str,
    min_chars: int = 10_000,
    min_keywords: int = 5,
) -> bool:
    """Return True if *text* is a full employment agreement.

    A full agreement must meet BOTH criteria:
    - Length >= *min_chars* characters.
    - At least *min_keywords* distinct section keywords appear (case-insensitive).
    """
    if len(text) < min_chars:
        return False

    lower_text = text.lower()
    keyword_hits = sum(1 for kw in SECTION_KEYWORDS if kw in lower_text)

    return keyword_hits >= min_keywords
