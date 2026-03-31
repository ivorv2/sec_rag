"""Structured logging configuration using structlog."""

import logging
import sys

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

_configured = False

_VALID_FORMATS = {"json", "console"}


def configure_logging(log_format: str, log_level: str) -> None:
    """Configure structlog for the application.

    Idempotent — no-op if already configured.

    Args:
        log_format: "json" or "console".
        log_level: Python log level name (e.g., "INFO", "DEBUG").

    Raises:
        ValueError: If log_format or log_level is invalid.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    if log_format not in _VALID_FORMATS:
        msg = f"log_format must be 'json' or 'console', got '{log_format}'"
        raise ValueError(msg)

    if not isinstance(log_level, str):
        msg = f"Invalid log level: '{log_level}'"
        raise ValueError(msg)

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        msg = f"Invalid log level: '{log_level}'"
        raise ValueError(msg)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    _configured = True


def bind_request_context(request_id: str) -> None:
    """Bind request_id to the structlog context for the current scope."""
    bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    """Clear all structlog context variables for the current scope."""
    clear_contextvars()
