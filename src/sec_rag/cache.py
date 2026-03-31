"""Redis-backed query response cache with graceful degradation."""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import redis
import structlog

logger = structlog.get_logger(__name__)

_SOCKET_CONNECT_TIMEOUT = 5  # seconds — max wait for initial TCP handshake
_SOCKET_TIMEOUT = 5  # seconds — max wait for any Redis operation
_RETRY_INTERVAL = 30.0  # seconds — backoff before attempting reconnection


class QueryCache:
    """Caches full query responses in Redis to avoid repeat LLM calls.

    Degrades gracefully: if Redis is unavailable, all operations become no-ops.
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 3600) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

        self._ttl = ttl_seconds
        self._available = False
        self._client: redis.Redis | None = None
        self._retry_after: float = 0.0

        try:
            self._client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=_SOCKET_CONNECT_TIMEOUT,
                socket_timeout=_SOCKET_TIMEOUT,
            )
            self._client.ping()
            self._available = True
        except Exception:
            # Redact credentials from URL before logging
            safe_url = redis_url.split("@")[-1] if "@" in redis_url else redis_url
            logger.warning(
                "redis_unavailable",
                redis_host=safe_url,
                msg="Cache disabled — operating without Redis",
            )

    @property
    def is_available(self) -> bool:
        """Whether the cache is currently available."""
        return self._available

    def _maybe_reconnect(self) -> bool:
        """Attempt to reconnect if the retry backoff has elapsed.

        Returns True if the cache is now available, False otherwise.
        """
        if self._available:
            return True
        if time.monotonic() < self._retry_after:
            return False
        if self._client is None:
            return False
        try:
            self._client.ping()
            self._available = True
            logger.info("cache_reconnected")
            return True
        except redis.RedisError:
            self._retry_after = time.monotonic() + _RETRY_INTERVAL
            return False

    def get(self, query: str) -> dict[str, Any] | None:
        """Retrieve a cached response for the given query.

        The query is normalized (whitespace collapsed, lowercased, SHA-256
        hashed) before lookup. Returns None on cache miss, unavailability,
        or any error.
        """
        if not self._maybe_reconnect():
            return None

        key = self._normalize_key(query)
        try:
            raw: bytes | None = self._client.get(key)  # type: ignore[union-attr,assignment]
        except redis.RedisError:
            self._handle_redis_error("cache_get_redis_error", key)
            return None

        if raw is None:
            return None

        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("cache_get_json_decode_error", key=key)
            return None

        if not isinstance(decoded, dict):
            logger.warning(
                "cache_get_unexpected_type",
                key=key,
                actual_type=type(decoded).__name__,
            )
            return None

        # Defense-in-depth: reject dicts missing the required 'success' key.
        # Full Pydantic validation happens in the caller (routes.py) — this
        # catches obviously corrupt data early.
        if "success" not in decoded:
            logger.warning("cache_get_missing_required_key", key=key)
            return None

        return decoded

    def set(self, query: str, response: dict[str, Any]) -> None:
        """Cache a query response with TTL.

        The query is normalized (whitespace collapsed, lowercased, SHA-256
        hashed) before storage. No-op on any failure.
        """
        if not self._maybe_reconnect():
            return

        try:
            value = json.dumps(response)
        except (TypeError, ValueError):
            logger.warning("cache_set_json_encode_error", key=self._normalize_key(query))
            return

        try:
            key = self._normalize_key(query)
            self._client.setex(key, self._ttl, value)  # type: ignore[union-attr]
        except redis.RedisError:
            self._handle_redis_error("cache_set_redis_error", key)

    def _normalize_key(self, query: str) -> str:
        """Normalize query text and produce a deterministic Redis key.

        Steps: strip + lowercase, collapse whitespace, SHA-256 hash.
        """
        normalized = query.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        hex_digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"sec_rag:v1:query:{hex_digest}"

    def _handle_redis_error(self, event: str, key: str) -> None:
        """Log Redis error with key (not query text to avoid PII).

        If the connection fails, mark cache as unavailable and schedule a
        reconnection attempt after _RETRY_INTERVAL seconds.
        """
        logger.warning(event, key=key)
        self._available = False
        self._retry_after = time.monotonic() + _RETRY_INTERVAL

    def close(self) -> None:
        """Close the Redis connection and mark cache as permanently unavailable."""
        self._available = False
        client = self._client
        self._client = None  # prevent _maybe_reconnect from retrying
        if client is not None:
            try:
                client.close()
            except redis.RedisError:
                logger.warning("cache_close_redis_error")

    def ping(self) -> bool:
        """Check if the Redis connection is healthy.

        Re-enables the cache if Redis is reachable again after a failure.
        """
        if self._client is None:
            return False
        try:
            self._client.ping()
            self._available = True
            return True
        except redis.RedisError:
            self._available = False
            return False
