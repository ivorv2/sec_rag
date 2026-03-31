"""LLM factory: create a chat model from application settings."""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from sec_rag.config import Settings


def create_llm(settings: Settings) -> BaseChatModel:
    """Return an LLM instance based on ``settings.llm_provider``.

    Supported providers:
        - ``"anthropic"`` → :class:`ChatAnthropic`
        - ``"openai"``    → :class:`ChatOpenAI`

    Raises:
        ValueError: If ``settings.llm_provider`` is not recognised.
    """
    provider = settings.llm_provider.lower()

    # Per-call timeout prevents a single LLM call from blocking indefinitely.
    # The graph may make 3-5 calls (route, evaluate, rewrite, evaluate, generate),
    # so individual call timeout should be a fraction of the total budget.
    per_call_timeout = settings.query_timeout_seconds / 3

    if provider == "anthropic":
        return ChatAnthropic(  # type: ignore[call-arg]
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key.get_secret_value(),  # type: ignore[arg-type]
            timeout=per_call_timeout,
            max_tokens=settings.llm_max_tokens,
        )

    if provider == "openai":
        return ChatOpenAI(  # type: ignore[call-arg]
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[arg-type]
            request_timeout=per_call_timeout,
            max_tokens=settings.llm_max_tokens,
        )

    raise ValueError(
        f"Unknown LLM provider: '{settings.llm_provider}'. "
        f"Supported providers: 'anthropic', 'openai'."
    )
