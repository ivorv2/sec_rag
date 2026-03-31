"""Tests for the EDGAR EFTS client module."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from sec_rag.ingestion.edgar_client import (
    EDGAR_ARCHIVE_BASE,
    EdgarHit,
    _parse_hit,
    fetch_agreements,
    search_employment_agreements,
)

# ---------------------------------------------------------------------------
# Fixtures: realistic EFTS response data
# ---------------------------------------------------------------------------

MOCK_HIT_1 = {
    "_id": "0001234567-24-000001:ex10-1.htm",
    "_source": {
        "adsh": "0001234567-24-000001",
        "ciks": ["0001234567"],
        "display_names": ["Acme Corp (ACME)"],
        "file_date": "2024-03-15",
    },
}

MOCK_HIT_2 = {
    "_id": "0009876543-23-000042:exhibit10_2.htm",
    "_source": {
        "adsh": "0009876543-23-000042",
        "ciks": ["0009876543"],
        "display_names": ["Beta Inc"],
        "file_date": "2023-11-20",
    },
}


def _make_efts_response(
    hits: list[dict],
    total: int | None = None,
) -> dict:
    """Build a mock EFTS JSON response envelope."""
    if total is None:
        total = len(hits)
    return {
        "hits": {
            "total": {"value": total, "relation": "eq"},
            "hits": hits,
        }
    }


# ---------------------------------------------------------------------------
# search_employment_agreements
# ---------------------------------------------------------------------------


class TestSearchEmploymentAgreements:
    async def test_parses_single_page(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_efts_response([MOCK_HIT_1, MOCK_HIT_2])
        mock_response.raise_for_status = MagicMock()

        with patch("sec_rag.ingestion.edgar_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            results = await search_employment_agreements(
                start_date="2023-01-01",
                end_date="2024-12-31",
                max_results=10,
                user_agent="test-agent",
            )

        assert len(results) == 2

        r1 = results[0]
        assert isinstance(r1, EdgarHit)
        assert r1.accession_number == "0001234567-24-000001"
        assert r1.cik == "1234567"
        assert r1.company_name == "Acme Corp"
        assert r1.filing_date_raw == "2024-03-15"
        assert r1.filename == "ex10-1.htm"
        assert "000123456724000001" in r1.exhibit_url
        assert r1.exhibit_url.endswith("/ex10-1.htm")

        r2 = results[1]
        assert r2.accession_number == "0009876543-23-000042"
        assert r2.cik == "9876543"
        assert r2.company_name == "Beta Inc"

    async def test_paginates_until_max_results(self) -> None:
        """When max_results > page size, multiple pages are fetched."""
        page1 = _make_efts_response([MOCK_HIT_1], total=20)
        page2 = _make_efts_response([MOCK_HIT_2], total=20)

        mock_resp_1 = MagicMock()
        mock_resp_1.json.return_value = page1
        mock_resp_1.raise_for_status = MagicMock()

        mock_resp_2 = MagicMock()
        mock_resp_2.json.return_value = page2
        mock_resp_2.raise_for_status = MagicMock()

        with patch("sec_rag.ingestion.edgar_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = [mock_resp_1, mock_resp_2]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            results = await search_employment_agreements(
                start_date="2023-01-01",
                end_date="2024-12-31",
                max_results=2,
                user_agent="test-agent",
            )

        assert len(results) == 2
        assert mock_client.get.call_count == 2

    async def test_raises_on_http_error(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with patch("sec_rag.ingestion.edgar_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await search_employment_agreements(
                    start_date="2023-01-01",
                    end_date="2024-12-31",
                    user_agent="test-agent",
                )

    async def test_empty_response_returns_empty_list(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = _make_efts_response([], total=0)
        mock_response.raise_for_status = MagicMock()

        with patch("sec_rag.ingestion.edgar_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            results = await search_employment_agreements(
                start_date="2023-01-01",
                end_date="2024-12-31",
                user_agent="test-agent",
            )

        assert results == []


# ---------------------------------------------------------------------------
# _parse_hit (URL construction + malformed hit handling)
# ---------------------------------------------------------------------------


class TestParseHit:
    def test_url_construction(self) -> None:
        result = _parse_hit(MOCK_HIT_1)
        assert result is not None
        expected_url = (
            f"{EDGAR_ARCHIVE_BASE}/1234567/000123456724000001/ex10-1.htm"
        )
        assert result.exhibit_url == expected_url

    def test_cik_strips_leading_zeros(self) -> None:
        result = _parse_hit(MOCK_HIT_1)
        assert result is not None
        assert result.cik == "1234567"

    def test_company_name_strips_ticker(self) -> None:
        result = _parse_hit(MOCK_HIT_1)
        assert result is not None
        assert result.company_name == "Acme Corp"

    def test_company_name_without_ticker(self) -> None:
        result = _parse_hit(MOCK_HIT_2)
        assert result is not None
        assert result.company_name == "Beta Inc"

    def test_missing_display_names_fallback(self) -> None:
        hit = {
            "_id": "0001234567-24-000001:ex10-1.htm",
            "_source": {
                "adsh": "0001234567-24-000001",
                "ciks": ["0001234567"],
                "display_names": [],
                "file_date": "2024-03-15",
            },
        }
        result = _parse_hit(hit)
        assert result is not None
        assert result.company_name == "Unknown"

    def test_missing_display_names_key_fallback(self) -> None:
        hit = {
            "_id": "0001234567-24-000001:ex10-1.htm",
            "_source": {
                "adsh": "0001234567-24-000001",
                "ciks": ["0001234567"],
                "file_date": "2024-03-15",
            },
        }
        result = _parse_hit(hit)
        assert result is not None
        assert result.company_name == "Unknown"

    def test_id_without_colon_returns_none(self) -> None:
        hit = {
            "_id": "no-colon-here",
            "_source": {
                "adsh": "0001234567-24-000001",
                "ciks": ["0001234567"],
                "display_names": ["Acme Corp"],
                "file_date": "2024-03-15",
            },
        }
        result = _parse_hit(hit)
        assert result is None

    def test_empty_id_returns_none(self) -> None:
        hit = {
            "_id": "",
            "_source": {},
        }
        result = _parse_hit(hit)
        assert result is None

    def test_missing_ciks_uses_zero(self) -> None:
        hit = {
            "_id": "0001234567-24-000001:ex10-1.htm",
            "_source": {
                "adsh": "0001234567-24-000001",
                "ciks": [],
                "display_names": ["Acme Corp"],
                "file_date": "2024-03-15",
            },
        }
        result = _parse_hit(hit)
        assert result is not None
        assert result.cik == "0"


# ---------------------------------------------------------------------------
# fetch_agreements (shared client, bounded concurrency)
# ---------------------------------------------------------------------------


class TestFetchAgreements:
    async def test_combines_search_and_download(self) -> None:
        search_hits = [
            EdgarHit(
                accession_number="0001234567-24-000001",
                cik="1234567",
                company_name="Acme Corp",
                filing_date_raw="2024-03-15",
                exhibit_url="https://www.sec.gov/Archives/edgar/data/1234567/ex.htm",
                filename="ex.htm",
            ),
        ]

        mock_response = MagicMock()
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status = MagicMock()

        with (
            patch(
                "sec_rag.ingestion.edgar_client.search_employment_agreements",
                new_callable=AsyncMock,
                return_value=search_hits,
            ),
            patch("sec_rag.ingestion.edgar_client.httpx.AsyncClient") as mock_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            results = await fetch_agreements(user_agent="test-agent", max_results=1)

        assert len(results) == 1
        assert results[0].raw_html == "<html>content</html>"
        assert results[0].accession_number == "0001234567-24-000001"

    async def test_skips_failed_downloads(self) -> None:
        search_hits = [
            EdgarHit(
                accession_number="acc-1",
                cik="111",
                company_name="Good Corp",
                filing_date_raw="2024-01-01",
                exhibit_url="https://sec.gov/good",
                filename="good.htm",
            ),
            EdgarHit(
                accession_number="acc-2",
                cik="222",
                company_name="Bad Corp",
                filing_date_raw="2024-01-02",
                exhibit_url="https://sec.gov/bad",
                filename="bad.htm",
            ),
        ]

        good_response = MagicMock()
        good_response.text = "<html>ok</html>"
        good_response.raise_for_status = MagicMock()

        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        async def _mock_get(url: str, **kwargs) -> MagicMock:
            if "bad" in url:
                return bad_response
            return good_response

        with (
            patch(
                "sec_rag.ingestion.edgar_client.search_employment_agreements",
                new_callable=AsyncMock,
                return_value=search_hits,
            ),
            patch("sec_rag.ingestion.edgar_client.httpx.AsyncClient") as mock_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = _mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            results = await fetch_agreements(user_agent="test-agent", max_results=2)

        assert len(results) == 1
        assert results[0].company_name == "Good Corp"


# ---------------------------------------------------------------------------
# Integration test (requires network, opt-in via env var)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="Needs network -- set RUN_INTEGRATION=1",
)
class TestEdgarIntegration:
    async def test_search_returns_edgar_hits(self) -> None:
        results = await search_employment_agreements(
            start_date="2024-01-01",
            end_date="2024-12-31",
            max_results=2,
            user_agent="test-agent",
        )
        assert len(results) > 0
        for r in results:
            assert isinstance(r, EdgarHit)
            assert r.cik.isdigit()
            assert r.exhibit_url.startswith("https://www.sec.gov/Archives/")
