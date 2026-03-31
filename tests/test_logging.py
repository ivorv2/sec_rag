"""Tests for structured logging configuration."""

import io
import json
import logging

import structlog

from sec_rag.observability.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
)


class TestConfigureLogging:
    """Tests for configure_logging."""

    def setup_method(self):
        """Reset structlog and stdlib logging between tests."""
        structlog.reset_defaults()
        root = logging.getLogger()
        root.handlers.clear()
        # Reset the module-level _configured flag if it exists
        import sec_rag.observability.logging as log_mod

        if hasattr(log_mod, "_configured"):
            log_mod._configured = False

    def test_happy_path_json_format(self):
        """JSON format produces valid JSON output with required keys."""
        configure_logging("json", "INFO")

        stream = io.StringIO()
        logger = structlog.get_logger("test")
        # Capture output by adding a stream handler to stdlib root
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.root.handlers[0].formatter)
        logging.root.addHandler(handler)

        logger.info("test_event", key="value")

        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "event" in parsed
        assert parsed["event"] == "test_event"

    def test_happy_path_console_format(self):
        """Console format produces non-JSON human-readable output."""
        configure_logging("console", "INFO")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.root.handlers[0].formatter)
        logging.root.addHandler(handler)

        logger = structlog.get_logger("test")
        logger.info("test_event")

        output = stream.getvalue()
        assert "test_event" in output

    def test_error_invalid_log_format(self):
        """ValueError raised for unsupported log_format."""
        try:
            configure_logging("xml", "INFO")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "json" in str(e).lower() or "console" in str(e).lower()

    def test_error_invalid_log_level(self):
        """ValueError raised for invalid log_level name."""
        try:
            configure_logging("json", "TRACE")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "level" in str(e).lower() or "TRACE" in str(e).lower()

    def test_idempotent_double_call(self):
        """Second call is a no-op — no duplicate handlers."""
        configure_logging("json", "INFO")
        handler_count_after_first = len(logging.root.handlers)

        configure_logging("json", "INFO")
        handler_count_after_second = len(logging.root.handlers)

        assert handler_count_after_second == handler_count_after_first

    def test_stdlib_logging_captured(self):
        """stdlib logging calls are processed through structlog pipeline."""
        configure_logging("json", "INFO")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.root.handlers[0].formatter)
        logging.root.addHandler(handler)

        # Use stdlib logger directly
        stdlib_logger = logging.getLogger("third_party_lib")
        stdlib_logger.warning("stdlib warning message")

        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["event"] == "stdlib warning message"
        assert parsed["level"] == "warning"


class TestBindRequestContext:
    """Tests for bind_request_context."""

    def setup_method(self):
        structlog.reset_defaults()
        root = logging.getLogger()
        root.handlers.clear()
        import sec_rag.observability.logging as log_mod

        if hasattr(log_mod, "_configured"):
            log_mod._configured = False

    def test_happy_path_binds_request_id(self):
        """request_id appears in subsequent log output after binding."""
        configure_logging("json", "INFO")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.root.handlers[0].formatter)
        logging.root.addHandler(handler)

        bind_request_context("test-request-123")
        logger = structlog.get_logger("test")
        logger.info("bound_event")

        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["request_id"] == "test-request-123"

        clear_request_context()


class TestClearRequestContext:
    """Tests for clear_request_context."""

    def setup_method(self):
        structlog.reset_defaults()
        root = logging.getLogger()
        root.handlers.clear()
        import sec_rag.observability.logging as log_mod

        if hasattr(log_mod, "_configured"):
            log_mod._configured = False

    def test_happy_path_clears_context(self):
        """After clearing, request_id no longer appears in log output."""
        configure_logging("json", "INFO")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.root.handlers[0].formatter)
        logging.root.addHandler(handler)

        bind_request_context("to-be-cleared")
        clear_request_context()

        logger = structlog.get_logger("test")
        logger.info("after_clear")

        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert "request_id" not in parsed
