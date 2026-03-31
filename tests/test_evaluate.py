"""Tests for the evaluation script — compute_ragas_metrics, query_api, run_evaluation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts/ to path for importing evaluate.py
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Skip this module entirely if the scripts dir doesn't exist (e.g., mutmut worktree)
try:
    from eval_api import query_api  # noqa: E402
    from eval_metrics import compute_ragas_metrics  # noqa: E402
except ModuleNotFoundError:
    import pytest

    pytest.skip("evaluate module not available (scripts/ not found)", allow_module_level=True)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_successful_result(
    question: str = "What is the base salary?",
    ground_truth: str = "Base salary is $150,000",
    contexts: list[str] | None = None,
) -> dict:
    """Build a mock per-question result dict."""
    return {
        "success": True,
        "result": {
            "obligations": [{"obligation_type": "compensation", "description": "test"}],
            "summary": "The base salary is $150,000 per year.",
            "confidence": 0.9,
            "source_count": 3,
        },
        "contexts": contexts or ["The base salary shall be $150,000 per year."],
        "question": question,
        "ground_truth": ground_truth,
        "latency_seconds": 2.5,
    }


def _make_failed_result(question: str = "What is the bonus?") -> dict:
    return {
        "success": False,
        "result": None,
        "error": "Connection refused",
        "contexts": None,
        "question": question,
        "ground_truth": "Annual bonus of 20%",
        "latency_seconds": 0.1,
    }


# ── Tests: query_api ────────────────────────────────────────────────────────


class TestQueryApi:
    """Tests for the modified query_api function."""

    @patch("eval_api.requests.post")
    def test_sends_skip_cache_true(self, mock_post: MagicMock) -> None:
        """query_api always sends skip_cache=True in the request body."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": True, "result": None}
        mock_post.return_value = mock_resp

        query_api("http://localhost:8000", "What is the salary?")

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["skip_cache"] is True

    @patch("eval_api.requests.post")
    def test_sends_include_contexts_when_requested(self, mock_post: MagicMock) -> None:
        """query_api sends include_contexts=True when include_contexts=True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": True, "result": None}
        mock_post.return_value = mock_resp

        query_api("http://localhost:8000", "What is the salary?", include_contexts=True)

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["include_contexts"] is True

    @patch("eval_api.requests.post")
    def test_does_not_send_include_contexts_by_default(self, mock_post: MagicMock) -> None:
        """query_api does not send include_contexts when not requested."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": True, "result": None}
        mock_post.return_value = mock_resp

        query_api("http://localhost:8000", "What is the salary?")

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "include_contexts" not in body


# ── Tests: compute_ragas_metrics ────────────────────────────────────────────


class TestComputeRagasMetrics:
    """Tests for compute_ragas_metrics."""

    def test_returns_empty_dict_when_ragas_not_installed(self) -> None:
        """L3 #18: ImportError → returns empty dict."""

        import eval_metrics as eval_mod  # noqa: E402

        try:
            with patch.dict("sys.modules", {"ragas": None}):
                importlib.reload(eval_mod)
                result = eval_mod.compute_ragas_metrics([_make_successful_result()])
        finally:
            importlib.reload(eval_mod)  # always restore, even on assertion failure
        assert result == {}

    def test_empty_results_returns_empty_dict(self) -> None:
        """L3 #24: empty results → returns {} without calling ragas."""
        result = compute_ragas_metrics([])
        assert result == {}

    def test_all_failed_results_returns_empty_dict(self) -> None:
        """L3 #20: all failed results → no valid samples → returns {}."""
        results = [_make_failed_result(), _make_failed_result()]
        result = compute_ragas_metrics(results)
        assert result == {}

    def test_filters_results_without_contexts(self) -> None:
        """Results with contexts=None are excluded from ragas dataset."""
        result_no_ctx = _make_successful_result()
        result_no_ctx["contexts"] = None

        result = compute_ragas_metrics([result_no_ctx])
        assert result == {}

    def test_filters_results_without_ground_truth(self) -> None:
        """Results with empty ground_truth are excluded from ragas dataset."""
        result_no_gt = _make_successful_result()
        result_no_gt["ground_truth"] = ""

        result = compute_ragas_metrics([result_no_gt])
        assert result == {}

    @patch.dict("os.environ", {"SEC_RAG_ANTHROPIC_API_KEY": "test-key"})
    @patch("anthropic.Anthropic")
    @patch("ragas.llms.llm_factory")
    @patch("ragas.embeddings.HuggingFaceEmbeddings")
    @patch(
        "ragas.metrics.collections.faithfulness.metric.Faithfulness.__init__",
        return_value=None,
    )
    @patch(
        "ragas.metrics.collections.answer_relevancy.AnswerRelevancy.__init__",
        return_value=None,
    )
    @patch(
        "ragas.metrics.collections.answer_correctness.AnswerCorrectness.__init__",
        return_value=None,
    )
    @patch("ragas.evaluate")
    def test_builds_correct_samples(
        self,
        mock_eval: MagicMock,
        _mock_ac: MagicMock,
        _mock_ar: MagicMock,
        _mock_f: MagicMock,
        _mock_hf: MagicMock,
        mock_llm_factory: MagicMock,
        mock_anthropic: MagicMock,
    ) -> None:
        """Verify ragas evaluate is called and scores are returned via to_pandas()."""
        import pandas as pd

        mock_llm_factory.return_value = MagicMock()
        # Simulate EvaluationResult with to_pandas() returning a DataFrame
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = pd.DataFrame(
            [
                {
                    "faithfulness": 0.85,
                    "answer_relevancy": 0.90,
                    "answer_correctness": 0.78,
                }
            ]
        )
        mock_eval.return_value = mock_result
        results = [_make_successful_result()]
        scores = compute_ragas_metrics(results)

        assert scores["faithfulness"] == 0.85
        assert scores["answer_relevancy"] == 0.90
        assert scores["answer_correctness"] == 0.78
        mock_eval.assert_called_once()

    @patch.dict("os.environ", {"SEC_RAG_ANTHROPIC_API_KEY": "test-key"})
    @patch("anthropic.Anthropic")
    @patch("ragas.llms.llm_factory")
    @patch("ragas.embeddings.HuggingFaceEmbeddings")
    @patch(
        "ragas.metrics.collections.faithfulness.metric.Faithfulness.__init__",
        return_value=None,
    )
    @patch(
        "ragas.metrics.collections.answer_relevancy.AnswerRelevancy.__init__",
        return_value=None,
    )
    @patch(
        "ragas.metrics.collections.answer_correctness.AnswerCorrectness.__init__",
        return_value=None,
    )
    @patch("ragas.evaluate")
    def test_evaluate_exception_returns_none_values(
        self,
        mock_eval: MagicMock,
        _mock_ac: MagicMock,
        _mock_ar: MagicMock,
        _mock_f: MagicMock,
        _mock_hf: MagicMock,
        mock_llm_factory: MagicMock,
        mock_anthropic: MagicMock,
    ) -> None:
        """L3 #19: ragas evaluate() raises → returns dict with None values."""
        mock_llm_factory.return_value = MagicMock()
        mock_eval.side_effect = RuntimeError("LLM API key invalid")
        results = [_make_successful_result()]
        scores = compute_ragas_metrics(results)

        assert scores["faithfulness"] is None
        assert scores["answer_relevancy"] is None
        assert scores["answer_correctness"] is None

    @patch.dict(
        "os.environ",
        {"SEC_RAG_ANTHROPIC_API_KEY": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""},
        clear=False,
    )
    def test_no_api_key_returns_empty_dict(self) -> None:
        """No LLM API key available → returns {} with warning."""
        results = [_make_successful_result()]
        scores = compute_ragas_metrics(results)
        assert scores == {}
