"""Input validation guardrails for employment contract queries."""

import re

from sec_rag.agent.prompts import QUERY_DELIM_END, QUERY_DELIM_START
from sec_rag.models.analysis import InputValidation

# Topics that are clearly off-topic for employment contract analysis.
# Multi-word phrases use substring matching; single words use word-boundary matching
# to avoid false positives (e.g., "game" blocking "game plan").
_OFF_TOPIC_PHRASES: list[str] = [
    "sports score",
    "movie review",
    "tell me a joke",
    "write a poem",
    "write a story",
    "write a song",
    "cooking recipe",
]
_OFF_TOPIC_WORDS_RE = re.compile(
    r"\b(?:recipe|cooking|weather forecast)\b",
    re.IGNORECASE,
)
# Fuzzy delimiter detection: catches case/whitespace variations of the prompt
# boundary markers that exact-match would miss.
_DELIMITER_LIKE_RE = re.compile(
    r"-{2,}\s*(BEGIN|END)\s+USER\s+QUERY",
    re.IGNORECASE,
)


def validate_input(query: str) -> InputValidation:
    """Validate a user query for the employment contract analysis pipeline.

    Checks (applied in order, first failure wins):
        1. Query is not empty or only whitespace.
        2. Query does not contain prompt delimiter strings.
        3. Query does not match obvious off-topic patterns (case-insensitive
           substring match against ``_OFF_TOPIC_PHRASES`` and ``_OFF_TOPIC_WORDS_RE``).

    Note: Length validation (10-500 chars) is handled by Pydantic in
    ``schemas.py``, not here.

    Returns:
        InputValidation with ``is_valid=True`` if all checks pass,
        otherwise ``is_valid=False`` with an explanatory ``reason``.
    """
    stripped = query.strip()

    if not stripped:
        return InputValidation(is_valid=False, reason="Query is empty")

    # Reject queries containing prompt delimiter patterns to prevent delimiter
    # breakout injection attacks.
    # Fuzzy match: catches exact delimiters plus whitespace/case variations.
    if QUERY_DELIM_END in stripped or QUERY_DELIM_START in stripped:
        return InputValidation(
            is_valid=False,
            reason="Query contains reserved characters",
        )
    if _DELIMITER_LIKE_RE.search(stripped):
        return InputValidation(
            is_valid=False,
            reason="Query contains reserved characters",
        )

    # Length validation is handled by Pydantic in schemas.py (min_length=10, max_length=500).
    # Guardrails only check semantic validity (off-topic detection).

    lower_query = stripped.lower()
    for phrase in _OFF_TOPIC_PHRASES:
        if phrase in lower_query:
            return InputValidation(
                is_valid=False,
                reason="Query appears off-topic for employment contract analysis",
            )
    if _OFF_TOPIC_WORDS_RE.search(stripped):
        return InputValidation(
            is_valid=False,
            reason="Query appears off-topic for employment contract analysis",
        )

    return InputValidation(is_valid=True)
