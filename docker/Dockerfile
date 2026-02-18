# ==============================================================================
# Multi-stage Dockerfile for Korean Sentiment Analysis API
# Stage 1: Install dependencies
# Stage 2: Copy source and run — smaller final image
# ==============================================================================

# --- Stage 1: Builder ---
FROM python:3.13-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[api]"

# --- Stage 2: Runtime ---
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
