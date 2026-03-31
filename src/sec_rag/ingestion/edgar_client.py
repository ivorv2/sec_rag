"""EDGAR EFTS client for searching and downloading employment agreements."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
_RATE_LIMIT_SECONDS = 0.15
_PAGE_SIZE = 10  # EFTS default page size


class EdgarHit(BaseModel):
    """Typed representation of an EDGAR search result."""

    accession_number: str
    cik: str
    company_name: str
    filing_date_raw: str
    exhibit_url: str
    filename: str
    raw_html: str = ""


async def search_employment_agreements(
    start_date: str,
    end_date: str,
    user_agent: str,
    max_results: int = 200,
) -> list[EdgarHit]:
    """Search EDGAR EFTS for employment agreement exhibits in 10-K filings.

    Paginates through results until ``max_results`` are collected or no more
    hits remain.

    Returns a list of EdgarHit objects.

    Raises:
        httpx.HTTPStatusError: If the EFTS API returns a non-2xx status.
    """
    results: list[EdgarHit] = []
    offset = 0

    async with httpx.AsyncClient(
        timeout=30, headers={"User-Agent": user_agent}
    ) as client:
        while len(results) < max_results:
            params: dict[str, str | int] = {
                "q": '"employment agreement"',
                "forms": "10-K",
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
                "from": offset,
            }

            response = await client.get(EFTS_SEARCH_URL, params=params)
            response.raise_for_status()

            data = response.json()
            hits = data.get("hits", {}).get("hits", [])

            if not hits:
                break

            for hit in hits:
                if len(results) >= max_results:
                    break

                parsed = _parse_hit(hit)
                if parsed is not None:
                    results.append(parsed)

            total = data.get("hits", {}).get("total", {}).get("value", 0)
            offset += _PAGE_SIZE

            if offset >= total:
                break

            await asyncio.sleep(_RATE_LIMIT_SECONDS)

    return results


def _parse_hit(hit: dict[str, Any]) -> EdgarHit | None:
    """Parse a single EFTS hit into an EdgarHit.

    Returns None if the hit is malformed (e.g. ``_id`` has no ':' separator).
    """
    hit_id: str = hit.get("_id", "")
    if ":" not in hit_id:
        return None

    accession_for_url, filename = hit_id.split(":", 1)

    source = hit.get("_source", {})

    accession_number = source.get("adsh", "")

    ciks = source.get("ciks", [])
    cik = str(int(ciks[0])) if ciks else "0"

    display_names = source.get("display_names", [])
    if display_names:
        company_name = display_names[0].split("(")[0].strip()
    else:
        company_name = "Unknown"

    filing_date = source.get("file_date", "")

    accession_no_dashes = accession_for_url.replace("-", "")
    exhibit_url = f"{EDGAR_ARCHIVE_BASE}/{cik}/{accession_no_dashes}/{filename}"

    return EdgarHit(
        accession_number=accession_number,
        cik=cik,
        company_name=company_name,
        filing_date_raw=filing_date,
        exhibit_url=exhibit_url,
        filename=filename,
    )


async def _download_exhibit(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Download a single exhibit HTML file from EDGAR using a shared client.

    Returns the raw HTML string.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx status.
        httpx.TimeoutException: If the request exceeds 30 seconds.
    """
    async with semaphore:
        await asyncio.sleep(_RATE_LIMIT_SECONDS)
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def fetch_agreements(
    user_agent: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    max_results: int = 100,
) -> list[EdgarHit]:
    """Search for employment agreements and download their exhibit HTML.

    This is the main entry point for the ingestion pipeline.  It combines
    :func:`search_employment_agreements` and :func:`_download_exhibit`,
    using bounded concurrency (5 concurrent downloads) with rate-limiting
    to respect EDGAR limits.

    Failed downloads are logged as warnings and skipped.

    Returns a list of EdgarHit objects with ``raw_html`` populated.
    """
    search_results = await search_employment_agreements(
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
        user_agent=user_agent,
    )

    semaphore = asyncio.Semaphore(5)
    results_with_html: list[EdgarHit] = []

    async with httpx.AsyncClient(
        timeout=30, headers={"User-Agent": user_agent}
    ) as client:

        async def _fetch_one(hit: EdgarHit) -> EdgarHit | None:
            try:
                html = await _download_exhibit(client, hit.exhibit_url, semaphore)
                return hit.model_copy(update={"raw_html": html})
            except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                logger.warning(
                    "failed_to_download_exhibit", url=hit.exhibit_url, error=str(exc)
                )
                return None

        # Semaphore caps concurrency at 5. For >500 results, consider batched
        # gather to reduce peak coroutine memory. Acceptable for typical volumes.
        downloaded = await asyncio.gather(*[_fetch_one(hit) for hit in search_results])
        results_with_html = [r for r in downloaded if r is not None]

    return results_with_html
