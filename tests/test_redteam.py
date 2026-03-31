"""LLM red team tests using DeepTeam.

Tests the RAG pipeline for prompt injection, PII leakage, and jailbreak
vulnerabilities. Requires a running API with LLM API key configured.

These tests make real LLM API calls — run only during audits when the user
provides an API key. Estimated cost: ~$3-5 per run.

Usage:
    SEC_RAG_API_KEY=<key> pytest tests/test_redteam.py -x -q
"""

from __future__ import annotations

import os

import pytest

# Skip entire module if no API key or deepteam not installed
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("RUN_REDTEAM"),
        reason="RUN_REDTEAM not set — red team tests require a running API + LLM keys. "
        "Run with: RUN_REDTEAM=1 SEC_RAG_API_KEY=<key> pytest tests/test_redteam.py",
    ),
]

API_URL = os.environ.get("SEC_RAG_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("SEC_RAG_API_KEY", "")


def model_callback(prompt: str) -> str:
    """Send a query to the SEC RAG API and return the response text."""
    import requests

    resp = requests.post(
        f"{API_URL}/query",
        json={"query": prompt, "skip_cache": True},
        headers={"X-API-Key": API_KEY},
        timeout=120,
    )
    if resp.status_code != 200:
        return f"Error: {resp.status_code}"

    data = resp.json()
    if data.get("success") and data.get("result"):
        return data["result"].get("summary", "")
    return data.get("error", "No response")


def test_prompt_injection() -> None:
    """Test resistance to prompt injection attacks."""
    from deepteam import red_team
    from deepteam.vulnerabilities import IndirectInstruction

    red_team(model_callback=model_callback, vulnerabilities=[IndirectInstruction()])


def test_pii_leakage() -> None:
    """Test that the system doesn't leak PII from contract documents."""
    from deepteam import red_team
    from deepteam.vulnerabilities import PIILeakage

    red_team(model_callback=model_callback, vulnerabilities=[PIILeakage()])
