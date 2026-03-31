"""Tests for QueryCache — Redis-backed query response cache."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import redis

from sec_rag.cache import QueryCache

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_redis(ping_ok: bool = True) -> MagicMock:
    """Create a mock Redis client that optionally fails ping."""
    mock_client = MagicMock(spec=redis.Redis)
    if ping_ok:
        mock_client.ping.return_value = True
    else:
        mock_client.ping.side_effect = redis.ConnectionError("Connection refused")
    return mock_client


# ── __init__ ─────────────────────────────────────────────────────────────────


class TestInit:
    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_creates_instance_with_valid_params(self, mock_from_url: MagicMock) -> None:
        """Happy path: valid URL and positive TTL creates a working cache."""
        mock_client = _make_mock_redis(ping_ok=True)
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0", ttl_seconds=600)

        assert cache._available is True
        assert cache._ttl == 600
        mock_from_url.assert_called_once_with(
            "redis://localhost:6379/0",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

    def test_ttl_zero_raises_value_error(self) -> None:
        """L1 error: ttl_seconds <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            QueryCache("redis://localhost:6379/0", ttl_seconds=0)

    def test_ttl_negative_raises_value_error(self) -> None:
        """L1 error: negative ttl_seconds raises ValueError."""
        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            QueryCache("redis://localhost:6379/0", ttl_seconds=-10)

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_redis_down_sets_available_false(self, mock_from_url: MagicMock) -> None:
        """L3 #6: Redis server down during init sets _available=False."""
        mock_client = _make_mock_redis(ping_ok=False)
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")

        assert cache._available is False

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_from_url_exception_sets_available_false(
        self, mock_from_url: MagicMock
    ) -> None:
        """L1 error: Exception during from_url sets _available=False."""
        mock_from_url.side_effect = ValueError("Malformed URL")

        cache = QueryCache("not-a-valid-url://bad")

        assert cache._available is False


# ── _normalize_key ───────────────────────────────────────────────────────────


class TestNormalizeKey:
    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_deterministic_key_format(self, mock_from_url: MagicMock) -> None:
        """Happy path: produces sec_rag:v1:query:<sha256> format."""
        mock_from_url.return_value = _make_mock_redis()
        cache = QueryCache("redis://localhost:6379/0")

        key = cache._normalize_key("what is the base salary?")

        assert key.startswith("sec_rag:v1:query:")
        hex_part = key.removeprefix("sec_rag:v1:query:")
        assert len(hex_part) == 64  # SHA-256 hex digest length

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_case_and_whitespace_normalization(self, mock_from_url: MagicMock) -> None:
        """L3 #8: different casing/whitespace produces same key."""
        mock_from_url.return_value = _make_mock_redis()
        cache = QueryCache("redis://localhost:6379/0")

        key1 = cache._normalize_key("  What is the BASE salary?  ")
        key2 = cache._normalize_key("what is the base salary?")

        assert key1 == key2

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_multiple_whitespace_collapsed(self, mock_from_url: MagicMock) -> None:
        """Tabs, newlines, and multiple spaces collapse to single space."""
        mock_from_url.return_value = _make_mock_redis()
        cache = QueryCache("redis://localhost:6379/0")

        key1 = cache._normalize_key("what\t\nis   the   salary")
        key2 = cache._normalize_key("what is the salary")

        assert key1 == key2


# ── get ──────────────────────────────────────────────────────────────────────


class TestGet:
    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_cache_miss_returns_none(self, mock_from_url: MagicMock) -> None:
        """Happy path: query not in cache returns None."""
        mock_client = _make_mock_redis()
        mock_client.get.return_value = None
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        result = cache.get("unknown query")

        assert result is None

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_cache_hit_returns_dict(self, mock_from_url: MagicMock) -> None:
        """Happy path: cached JSON dict is returned correctly."""
        expected = {"success": True, "answer": "42", "sources": []}
        mock_client = _make_mock_redis()
        mock_client.get.return_value = json.dumps(expected)
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        result = cache.get("some query")

        assert result == expected

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_missing_success_key_returns_none(self, mock_from_url: MagicMock) -> None:
        """m07: dict without 'success' key is rejected as invalid cache data."""
        mock_client = _make_mock_redis()
        mock_client.get.return_value = json.dumps({"answer": "42"})
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        result = cache.get("some query")

        assert result is None

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_unavailable_returns_none_immediately(
        self, mock_from_url: MagicMock
    ) -> None:
        """L1 error: _available=False returns None without touching Redis."""
        mock_client = _make_mock_redis(ping_ok=False)
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        result = cache.get("any query")

        assert result is None
        mock_client.get.assert_not_called()

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_redis_error_returns_none(self, mock_from_url: MagicMock) -> None:
        """L1/L3 #10: RedisError during get is caught, returns None."""
        mock_client = _make_mock_redis()
        mock_client.get.side_effect = redis.RedisError("connection lost")
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        result = cache.get("some query")

        assert result is None

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_json_decode_error_returns_none(self, mock_from_url: MagicMock) -> None:
        """L3 #22: corrupted JSON in Redis returns None."""
        mock_client = _make_mock_redis()
        mock_client.get.return_value = "not valid json {{{{"
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        result = cache.get("some query")

        assert result is None

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_json_list_not_dict_returns_none(self, mock_from_url: MagicMock) -> None:
        """L3 #28: valid JSON but a list, not a dict, returns None."""
        mock_client = _make_mock_redis()
        mock_client.get.return_value = json.dumps([1, 2, 3])
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        result = cache.get("some query")

        assert result is None


# ── set ──────────────────────────────────────────────────────────────────────


class TestSet:
    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_stores_json_with_ttl(self, mock_from_url: MagicMock) -> None:
        """Happy path: set stores JSON with SETEX and correct TTL."""
        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0", ttl_seconds=300)
        response = {"answer": "42"}
        cache.set("some query", response)

        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][0].startswith("sec_rag:v1:query:")
        assert call_args[0][1] == 300
        assert json.loads(call_args[0][2]) == response

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_unavailable_is_noop(self, mock_from_url: MagicMock) -> None:
        """L1 error: _available=False skips set without touching Redis."""
        mock_client = _make_mock_redis(ping_ok=False)
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        cache.set("any query", {"data": "value"})

        mock_client.setex.assert_not_called()

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_redis_error_is_noop(self, mock_from_url: MagicMock) -> None:
        """L1/L3 #10: RedisError during set is caught, no-op."""
        mock_client = _make_mock_redis()
        mock_client.setex.side_effect = redis.RedisError("write failed")
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        # Should not raise
        cache.set("some query", {"answer": "42"})

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_non_serializable_response_is_noop(
        self, mock_from_url: MagicMock
    ) -> None:
        """L3 #21: non-JSON-serializable response is caught, no-op."""
        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        cache.set("some query", {"bad": object()})  # type: ignore[dict-item]

        mock_client.setex.assert_not_called()


# ── close ────────────────────────────────────────────────────────────────────


class TestClose:
    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_close_sets_available_false(self, mock_from_url: MagicMock) -> None:
        """Happy path: close sets _available=False and calls client.close()."""
        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        assert cache._available is True

        cache.close()

        assert cache._available is False
        mock_client.close.assert_called_once()

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_close_redis_error_is_noop(self, mock_from_url: MagicMock) -> None:
        """L1 error: RedisError during close is caught, still sets _available=False."""
        mock_client = _make_mock_redis()
        mock_client.close.side_effect = redis.RedisError("close failed")
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        cache.close()

        assert cache._available is False

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_get_after_close_returns_none(self, mock_from_url: MagicMock) -> None:
        """L3 #26: get after close returns None (no Redis call)."""
        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        cache.close()
        result = cache.get("some query")

        assert result is None

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_set_after_close_is_noop(self, mock_from_url: MagicMock) -> None:
        """L3 #26: set after close is a no-op (no Redis call)."""
        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        cache.close()
        cache.set("some query", {"data": "val"})

        # setex should not be called after close
        mock_client.setex.assert_not_called()


# ── ping ─────────────────────────────────────────────────────────────────────


class TestPing:
    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_ping_returns_true_when_healthy(self, mock_from_url: MagicMock) -> None:
        """Happy path: ping returns True when Redis responds."""
        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        assert cache.ping() is True

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_ping_returns_false_on_redis_error(
        self, mock_from_url: MagicMock
    ) -> None:
        """L1 error: RedisError during ping returns False."""
        mock_client = _make_mock_redis()
        # First ping succeeds (init), second fails
        mock_client.ping.side_effect = [True, redis.RedisError("down")]
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        assert cache.ping() is False

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_ping_returns_false_when_unavailable(
        self, mock_from_url: MagicMock
    ) -> None:
        """ping returns False when _available is False."""
        mock_client = _make_mock_redis(ping_ok=False)
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        assert cache.ping() is False


# ── Automatic reconnection ─────────────────────────────────────────────────


class TestAutoReconnect:
    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_get_reconnects_after_backoff(self, mock_from_url: MagicMock) -> None:
        """After a Redis error, get auto-reconnects once backoff elapses."""
        mock_client = _make_mock_redis()
        mock_client.get.return_value = json.dumps({"success": True, "answer": "ok"})
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        # Simulate a Redis error marking cache unavailable
        cache._available = False
        cache._retry_after = 0.0  # backoff already elapsed

        result = cache.get("some query")
        assert result is not None
        assert result["success"] is True

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_get_skips_reconnect_during_backoff(
        self, mock_from_url: MagicMock
    ) -> None:
        """During backoff, get returns None without attempting reconnection."""
        import time as time_mod

        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        cache._available = False
        cache._retry_after = time_mod.monotonic() + 9999  # far in the future

        result = cache.get("some query")
        assert result is None
        # ping should NOT be called for reconnection (only init ping)
        assert mock_client.ping.call_count == 1  # just the init call

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_set_reconnects_after_backoff(self, mock_from_url: MagicMock) -> None:
        """After backoff, set auto-reconnects and stores data."""
        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        cache._available = False
        cache._retry_after = 0.0  # backoff elapsed

        cache.set("some query", {"success": True})
        mock_client.setex.assert_called_once()

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_reconnect_failure_reschedules_backoff(
        self, mock_from_url: MagicMock
    ) -> None:
        """Failed reconnect sets a new retry_after in the future."""
        import time as time_mod

        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        cache._available = False
        cache._retry_after = 0.0

        # Make ping fail for reconnection attempt (init already consumed return_value)
        mock_client.ping.side_effect = redis.RedisError("still down")

        result = cache.get("some query")
        assert result is None
        assert cache._retry_after > time_mod.monotonic()

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_handle_redis_error_sets_retry_backoff(
        self, mock_from_url: MagicMock
    ) -> None:
        """_handle_redis_error sets retry_after to future timestamp."""
        import time as time_mod

        mock_client = _make_mock_redis()
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        before = time_mod.monotonic()
        cache._handle_redis_error("test_event", "test_key")

        assert cache._available is False
        assert cache._retry_after >= before + 29  # ~30s backoff


# ── L3 Integration-style: round-trip ────────────────────────────────────────


class TestRoundTrip:
    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_set_then_get_returns_cached_response(
        self, mock_from_url: MagicMock
    ) -> None:
        """L3 #7: set then get returns the cached dict."""
        stored_data: dict[str, str] = {}
        mock_client = _make_mock_redis()

        def fake_setex(key: str, ttl: int, value: str) -> None:
            stored_data[key] = value

        def fake_get(key: str) -> str | None:
            return stored_data.get(key)

        mock_client.setex.side_effect = fake_setex
        mock_client.get.side_effect = fake_get
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0")
        response = {
            "success": True,
            "answer": "the base salary is $100k",
            "sources": ["doc1"],
        }

        # First get: miss
        assert cache.get("what is the base salary?") is None

        # Set
        cache.set("what is the base salary?", response)

        # Second get: hit
        result = cache.get("what is the base salary?")
        assert result == response

    @patch("sec_rag.cache.redis.Redis.from_url")
    def test_ttl_expired_returns_none(self, mock_from_url: MagicMock) -> None:
        """L3 #11: after TTL expiry, get returns None."""
        mock_client = _make_mock_redis()
        # Simulate key expired — get returns None
        mock_client.get.return_value = None
        mock_from_url.return_value = mock_client

        cache = QueryCache("redis://localhost:6379/0", ttl_seconds=1)
        result = cache.get("some query")

        assert result is None


# ── Integration tests (require real Redis) ───────────────────────────────────


def _redis_available() -> bool:
    """Check if Redis is reachable on localhost:6379."""
    try:
        client = redis.Redis()
        client.ping()
        client.close()
        return True
    except Exception:
        return False


requires_redis = pytest.mark.skipif(not _redis_available(), reason="Redis not running")


@requires_redis
class TestQueryCacheIntegration:
    """Integration tests against a real Redis instance."""

    def setup_method(self):
        """Create cache with short TTL, flush test keys."""
        self.cache = QueryCache("redis://localhost:6379/0", ttl_seconds=5)
        # Clean up any leftover test keys
        self.cache._client.delete(
            self.cache._normalize_key("integration test query")
        )

    def teardown_method(self):
        """Clean up after test."""
        self.cache._client.delete(
            self.cache._normalize_key("integration test query")
        )
        self.cache.close()

    def test_real_round_trip(self):
        """Full set → get cycle with real Redis."""
        response = {"success": True, "result": {"summary": "Test answer"}, "cached": False}

        assert self.cache.get("integration test query") is None

        self.cache.set("integration test query", response)

        result = self.cache.get("integration test query")
        assert result is not None
        assert result["success"] is True
        assert result["result"]["summary"] == "Test answer"

    def test_real_ping(self):
        """ping returns True with real Redis."""
        assert self.cache.ping() is True

    def test_real_key_normalization_consistency(self):
        """Queries that should normalize to the same key hit the same cache entry."""
        response = {"success": True, "answer": "cached"}

        self.cache.set("  Integration  Test  QUERY  ", response)

        result = self.cache.get("integration test query")
        assert result is not None
        assert result["success"] is True
        assert result["answer"] == "cached"

    def test_real_close_then_get(self):
        """After close, get returns None without error."""
        self.cache.close()
        assert self.cache.get("integration test query") is None
        # Recreate for teardown
        self.cache = QueryCache("redis://localhost:6379/0", ttl_seconds=5)
