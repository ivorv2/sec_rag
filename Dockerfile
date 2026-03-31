# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY requirements.lock .
COPY src/ src/

# Install from pinned lockfile for reproducible builds (m04).
# Hash pinning (--require-hashes) not feasible due to deep dependency tree;
# version pinning provides deterministic installs without supply-chain hash
# verification. To add hash pinning, use: uv pip compile --generate-hashes.
RUN pip install --no-cache-dir --prefix=/install -r requirements.lock \
    && pip install --no-cache-dir --prefix=/install --no-deps .

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app

# Copy scripts for optional ingestion
COPY scripts/ scripts/

# Set model cache directories
ENV PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence-transformers \
    FASTEMBED_CACHE_PATH=/app/models/fastembed \
    HF_HOME=/app/models/huggingface

# Pre-download models so container starts instantly (no internet needed at runtime)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')" \
    && python -c "\
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')" \
    && python -c "\
from fastembed import SparseTextEmbedding; \
list(SparseTextEmbedding(model_name='Qdrant/bm25').embed(['warmup']))"

# Set ownership and switch to non-root user
RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "sec_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
