# Base Dockerfile for ReleAF AI services
# UPGRADED: Python 3.11, multi-stage build, optimized layers

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better layer caching)
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Runtime stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models data

# Create non-root user for security
RUN useradd -m -u 1000 releaf && \
    chown -R releaf:releaf /app
USER releaf

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port (will be overridden by specific services)
EXPOSE 8000

# Default command (will be overridden by docker-compose)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

