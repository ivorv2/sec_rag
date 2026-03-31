"""API interaction helpers for the evaluation pipeline."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_GOLDEN_SET = "eval/golden_set.json"
REQUEST_TIMEOUT_SECONDS = 120

def _api_headers() -> dict[str, str]:
    """Build request headers, including API key if configured."""
    import os

    headers: dict[str, str] = {}
    api_key = os.environ.get("SEC_RAG_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def load_golden_set(path: str) -> list[dict[str, Any]]:
    """Load the golden-set JSON file and return a list of question dicts."""
    filepath = Path(path)
    if not filepath.exists():
        logger.error("Golden set not found at %s", filepath.resolve())
        sys.exit(1)

    with filepath.open() as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        logger.error("Golden set must be a non-empty JSON array")
        sys.exit(1)

    logger.info("Loaded %d golden-set questions from %s", len(data), path)
    return data


def check_api_health(api_url: str) -> bool:
    """Verify the API is reachable and healthy."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=10)
        if resp.status_code == 200:
            health = resp.json()
            logger.info(
                "API healthy: qdrant=%s, points=%d",
                health.get("qdrant_connected"),
                health.get("collection_count", 0),
            )
            return True
        logger.error("Health check returned status %d", resp.status_code)
        return False
    except requests.ConnectionError:
        logger.error(
            "Cannot connect to API at %s. Is the server running?",
            api_url,
        )
        return False
    except requests.RequestException as exc:
        logger.error("Health check failed: %s", exc)
        return False


def query_api(api_url: str, question: str, include_contexts: bool = False) -> dict[str, Any]:
    """Send a single question to the /query endpoint and return the response.

    Returns a dict with keys: success, result (or None), error (or None),
    and latency_seconds.
    """
    start = time.monotonic()
    try:
        body: dict[str, Any] = {"query": question, "skip_cache": True}
        if include_contexts:
            body["include_contexts"] = True

        resp = requests.post(
            f"{api_url}/query",
            json=body,
            headers=_api_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        latency = time.monotonic() - start
        if resp.status_code == 200:
            resp_body: dict[str, Any] = resp.json()
            resp_body["latency_seconds"] = round(latency, 2)
            return resp_body
        return _error_response(
            f"HTTP {resp.status_code}: {resp.text[:200]}", latency
        )
    except requests.ConnectionError:
        return _error_response("Connection refused", time.monotonic() - start)
    except requests.Timeout:
        return _error_response(
            f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s",
            REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return _error_response(str(exc), time.monotonic() - start)


def _error_response(error: str, latency: float) -> dict[str, Any]:
    """Build a standardized error response dict (m28)."""
    return {
        "success": False,
        "result": None,
        "error": error,
        "latency_seconds": round(latency, 2),
    }
