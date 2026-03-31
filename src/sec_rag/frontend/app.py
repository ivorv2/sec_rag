"""Streamlit frontend for SEC EDGAR employment contract RAG system."""

import html
import os
import time
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("SEC_RAG_API_URL", "http://localhost:8000")
_API_KEY = os.environ.get("SEC_RAG_API_KEY", "")
_API_HEADERS: dict[str, str] = {"X-API-Key": _API_KEY} if _API_KEY else {}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SEC RAG", page_icon="\U0001f4c4", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .citation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.92em;
        line-height: 1.5;
        color: #333;
    }
    .citation-source {
        font-size: 0.82em;
        color: #6c757d;
        margin-top: 6px;
    }
    .health-ok {
        color: #28a745;
        font-weight: 600;
    }
    .health-err {
        color: #dc3545;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES: list[str] = [
    "What are the non-compete obligations?",
    "What compensation terms are specified?",
    "What are the termination conditions?",
    "What equity provisions exist?",
]


# ---------------------------------------------------------------------------
# Helper: health check
# ---------------------------------------------------------------------------
def _fetch_health() -> dict[str, Any] | None:
    """Return parsed health JSON, or None on failure."""
    try:
        resp = requests.get(f"{API_URL}/health", headers=_API_HEADERS, timeout=5)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result
    except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
        return None


# ---------------------------------------------------------------------------
# Helper: confidence color
# ---------------------------------------------------------------------------
def _confidence_color(score: float) -> str:
    if score > 0.7:
        return "green"
    if score >= 0.4:
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("SEC RAG")
    st.caption("Employment Contract Analyzer")

    # Health indicator — cached with 30s TTL to avoid hitting API on every rerun
    _now = time.monotonic()
    if "health_ts" not in st.session_state or _now - st.session_state["health_ts"] > 30:
        st.session_state["health"] = _fetch_health()
        st.session_state["health_ts"] = _now
    health = st.session_state["health"]
    if health is not None:
        st.markdown(
            '<p class="health-ok">\u2705 API Connected</p>',
            unsafe_allow_html=True,
        )
        st.metric("Collections", health.get("collection_count", 0))
    else:
        st.markdown(
            '<p class="health-err">\u274c API Unreachable</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        "Analyze employment agreements from SEC EDGAR filings. "
        "Ask questions about obligations, compensation, termination terms, and more."
    )

    st.divider()

    show_contexts = st.checkbox("Show source contexts", value=False)

    st.subheader("Example Queries")
    for eq in EXAMPLE_QUERIES:
        if st.button(eq, key=eq):
            st.session_state["query_input"] = eq
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.header("Employment Contract Analysis")

# Text input — pre-populated via session_state["query_input"] from sidebar buttons
user_query = st.text_input(
    "Ask a question about employment contracts\u2026",
    key="query_input",
)

submit = st.button("Submit", type="primary")

if submit and user_query.strip():
    _trimmed = user_query.strip()
    if len(_trimmed) < 10:
        st.warning("Query must be at least 10 characters.")
        st.stop()
    if len(_trimmed) > 500:
        st.warning("Query must be at most 500 characters.")
        st.stop()
    with st.spinner("Analyzing contracts\u2026"):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"query": _trimmed, "include_contexts": show_contexts},
                headers=_API_HEADERS,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            st.error(
                "Could not connect to the API. "
                "Make sure the backend is running at " + API_URL
            )
            st.stop()
        except requests.Timeout:
            st.error("The request timed out. Try a simpler query or try again later.")
            st.stop()
        except requests.HTTPError as exc:
            if exc.response.status_code == 422:
                st.error(
                    "Invalid query. Please ensure your query is between "
                    "10 and 500 characters."
                )
            elif exc.response.status_code == 401:
                st.error("Authentication failed. Check your API key configuration.")
            elif exc.response.status_code == 429:
                st.error("Rate limit exceeded. Please wait a moment and try again.")
            else:
                st.error(
                    f"API returned an error (status {exc.response.status_code}). "
                    "Please try again."
                )
            st.stop()

    # --- Handle API-level errors ---
    if not data.get("success"):
        st.error(data.get("error", "Unknown error from API."))
        st.stop()

    result = data.get("result")
    if result is None:
        st.warning("The API returned no analysis result.")
        st.stop()

    # --- Confidence metric ---
    confidence = result.get("confidence", 0.0)
    color = _confidence_color(confidence)
    st.markdown(
        f"**Confidence:** "
        f'<span style="color:{color}; font-size:1.3em; font-weight:700;">'
        f"{confidence:.0%}</span>",
        unsafe_allow_html=True,
    )

    # --- Summary ---
    st.info(result.get("summary", ""))

    # --- Obligations ---
    obligations = result.get("obligations", [])
    if not obligations:
        st.warning("No obligations found for this query.")
    else:
        st.subheader(f"Obligations ({len(obligations)})")
        for idx, obl in enumerate(obligations):
            header = f"{obl.get('obligation_type', 'Unknown')} \u2014 {obl.get('party', 'N/A')}"
            with st.expander(header, expanded=(idx == 0)):
                st.text(obl.get("description", ""))
                conditions = obl.get("conditions")
                if conditions:
                    st.text(f"Conditions: {conditions}")

                citations = obl.get("citations", [])
                if citations:
                    st.markdown("**Sources:**")
                    for cit in citations:
                        excerpt = html.escape(cit.get("excerpt", ""))
                        company = html.escape(cit.get("company_name", ""))
                        section = html.escape(cit.get("section_type", ""))
                        st.markdown(
                            f'<div class="citation-box">{excerpt}'
                            f'<div class="citation-source">'
                            f"{company} &middot; {section}</div></div>",
                            unsafe_allow_html=True,
                        )

    # --- Source count ---
    source_count = result.get("source_count", 0)
    st.caption(f"Based on {source_count} source chunk(s).")

    # --- Retrieved contexts (if requested) ---
    contexts = data.get("contexts")
    if contexts:
        with st.expander(f"Retrieved source contexts ({len(contexts)})", expanded=False):
            for i, ctx_text in enumerate(contexts):
                st.text(f"[Chunk {i + 1}] {ctx_text[:500]}")
                if len(ctx_text) > 500:
                    st.caption("(truncated)")

elif submit:
    st.warning("Please enter a query.")
