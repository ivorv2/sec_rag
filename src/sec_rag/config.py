from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "SEC_RAG_", "env_file": ".env", "extra": "ignore"}

    # LLM — override llm_model via SEC_RAG_LLM_MODEL env var when the
    # pinned version is deprecated. Anthropic typically provides 3+ months
    # notice before sunsetting dated model IDs.
    llm_provider: str = "anthropic"
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    llm_model: str = "claude-sonnet-4-20250514"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "employment_contracts"
    qdrant_api_key: SecretStr = SecretStr("")

    # Retrieval
    retrieval_dense_limit: int = 50
    retrieval_sparse_limit: int = 50
    retrieval_rrf_top_k: int = 20
    rerank_top_k: int = 5
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # Ingestion
    edgar_user_agent: str = "SecRAG Research research@example.com"
    min_agreement_chars: int = 10_000
    min_section_keywords: int = 5
    max_chunk_chars: int = 2_000

    # Agent
    max_retries: int = 2

    # Langfuse
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    # API
    api_key: SecretStr = SecretStr("")
    require_api_key: bool = False  # set True in production to refuse startup without auth
    expose_metrics: bool = True  # set False to disable /metrics endpoint
    query_timeout_seconds: int = 120  # per-LLM-call timeout derived from this
    behind_proxy: bool = False  # set True when behind a reverse proxy that sets X-Forwarded-For

    # Logging
    log_format: str = "console"
    log_level: str = "INFO"

    # Redis / Cache
    redis_url: str = ""
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True
