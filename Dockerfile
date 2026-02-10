FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -e "."

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels

# Install dependencies
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY src/ ./src/
COPY .env.example ./

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import rag_chatbot" || exit 1

# Default command
ENTRYPOINT ["python", "-m", "rag_chatbot"]
CMD ["--help"]
