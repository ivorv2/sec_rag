"""Langfuse observability integration for LangGraph tracing."""

from langfuse.callback import CallbackHandler

from sec_rag.config import Settings


def create_langfuse_handler(settings: Settings) -> CallbackHandler | None:
    """Return a Langfuse CallbackHandler if keys are configured, else None.

    Caller passes the handler to ``graph.invoke(config={"callbacks": [handler]})``.
    """
    if not settings.langfuse_public_key or not settings.langfuse_secret_key.get_secret_value():
        return None
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_base_url,
    )
