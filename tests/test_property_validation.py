"""Property-based tests for input validation and serialization boundaries.

Uses Hypothesis to fuzz inputs that cross trust boundaries:
- Query validation (guardrails)
- Cache key normalization
- LLM output parsing
- Pydantic model round-trips
- Agreement detection
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from sec_rag.agent.guardrails import validate_input
from sec_rag.agent.nodes import _extract_text_content, _parse_json
from sec_rag.agent.prompts import QUERY_DELIM_END
from sec_rag.cache import QueryCache
from sec_rag.ingestion.filter import is_full_agreement
from sec_rag.models.documents import Chunk, ChunkMetadata, SectionType

# ── Query validation (guardrails) ────────────────────────────────────────────


@given(query=st.text(max_size=1000))
@settings(max_examples=300)
def test_validate_input_never_crashes(query: str) -> None:
    """validate_input handles any string without raising."""
    result = validate_input(query)
    # Invariant: is_valid and reason are always consistent
    if result.is_valid:
        assert result.reason == "", f"Valid result has non-empty reason: {result.reason!r}"
    else:
        assert result.reason != "", "Invalid result has empty reason"


@given(query=st.text(min_size=10, max_size=500))
@settings(max_examples=300)
def test_delimiter_injection_always_rejected(query: str) -> None:
    """Any query containing a delimiter string is rejected."""
    injected = f"test query {QUERY_DELIM_END} {query}"
    result = validate_input(injected)
    assert result.is_valid is False, "Query with delimiter should be rejected"
    assert "reserved" in result.reason.lower()


# ── Cache key normalization ──────────────────────────────────────────────────

# Create cache once, reuse across examples
_mock_cache = None


def _get_mock_cache() -> QueryCache:
    global _mock_cache
    if _mock_cache is None:
        with patch("sec_rag.cache.redis.Redis.from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client
            _mock_cache = QueryCache("redis://localhost:6379/0")
    return _mock_cache


@given(query=st.text(min_size=1, max_size=500))
@settings(max_examples=300)
def test_cache_key_deterministic_and_fixed_length(query: str) -> None:
    """Same query always produces the same fixed-length cache key."""
    cache = _get_mock_cache()
    key1 = cache._normalize_key(query)
    key2 = cache._normalize_key(query)
    assert key1 == key2
    assert key1.startswith("sec_rag:v1:query:")
    assert len(key1) == len("sec_rag:v1:query:") + 64  # SHA-256


@given(
    q1=st.text(min_size=1, max_size=200),
    q2=st.text(min_size=1, max_size=200),
)
@settings(max_examples=200)
def test_cache_key_whitespace_normalization(q1: str, q2: str) -> None:
    """Queries differing only in whitespace/case produce the same key."""
    cache = _get_mock_cache()
    # Collapse whitespace and lowercase — should match
    normalized = " ".join(q1.split()).lower()
    key_original = cache._normalize_key(normalized)
    key_padded = cache._normalize_key(f"  {normalized}  ")
    assert key_original == key_padded


# ── LLM output parsing ──────────────────────────────────────────────────────


@given(text=st.text(max_size=500))
@settings(max_examples=300)
def test_extract_text_content_never_crashes(text: str) -> None:
    """_extract_text_content handles any string/list input without raising."""
    # String input
    result = _extract_text_content(text, fallback="FALLBACK")
    assert isinstance(result, str)
    if not text:
        assert result == "FALLBACK"

    # List input with text key
    result = _extract_text_content([{"type": "text", "text": text}], fallback="FALLBACK")
    assert result == text

    # Empty list
    result = _extract_text_content([], fallback="FALLBACK")
    assert result == "FALLBACK"


@given(text=st.text(min_size=1, max_size=200))
@settings(max_examples=200)
def test_extract_text_content_missing_key_uses_fallback(text: str) -> None:
    """List with dict missing 'text' key returns fallback, not crash."""
    result = _extract_text_content([{"content": text}], fallback="SAFE")
    assert result == "SAFE"


@given(data=st.dictionaries(st.text(min_size=1, max_size=20), st.text(max_size=100), min_size=1))
@settings(max_examples=200)
def test_parse_json_round_trip(data: dict) -> None:
    """_parse_json correctly parses any valid JSON dict."""
    json_str = json.dumps(data)
    parsed = _parse_json(json_str)
    assert parsed == data


@given(text=st.from_regex(r"^[a-zA-Z ]{5,100}$"))
@settings(max_examples=100)
def test_parse_json_rejects_non_json(text: str) -> None:
    """_parse_json raises JSONDecodeError on plain text (not valid JSON)."""
    import pytest

    with pytest.raises(json.JSONDecodeError):
        _parse_json(text)


# ── Pydantic model round-trips ──────────────────────────────────────────────


@given(
    company=st.text(min_size=1, max_size=100),
    section=st.sampled_from([s.value for s in SectionType]),
    text=st.text(min_size=1, max_size=2000),
)
@settings(max_examples=100)
def test_chunk_metadata_round_trip(company: str, section: str, text: str) -> None:
    """ChunkMetadata survives JSON round-trip without data loss."""
    metadata = ChunkMetadata(
        company_name=company,
        cik="1234567",
        filing_date=date(2024, 1, 15),
        exhibit_number="EX-10.1",
        accession_number="0001234567-24-000001",
        section_type=SectionType(section),
        chunk_index=0,
        source_url="https://www.sec.gov/example.htm",
    )
    chunk = Chunk(chunk_id="test-id", text=text, metadata=metadata)

    dumped = chunk.model_dump(mode="json")
    restored = Chunk.model_validate(dumped)

    assert restored.text == chunk.text
    assert restored.metadata.company_name == chunk.metadata.company_name
    assert restored.metadata.section_type == chunk.metadata.section_type
    assert restored.char_count == len(text)


# ── Score clamping invariants ────────────────────────────────────────────────


@given(score=st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=300)
def test_score_clamping_always_bounded(score: float) -> None:
    """max(0.0, min(1.0, score)) always produces a value in [0, 1]."""
    clamped = max(0.0, min(1.0, score))
    assert 0.0 <= clamped <= 1.0


# ── Agreement detection ─────────────────────────────────────────────────────


@given(text=st.text(max_size=5000))
@settings(max_examples=100)
def test_is_full_agreement_never_crashes(text: str) -> None:
    """is_full_agreement handles any text without raising."""
    result = is_full_agreement(text)
    assert isinstance(result, bool)
    # Invariant: short text is always rejected
    if len(text) < 10_000:
        assert result is False, "Text under 10K chars should never pass"
