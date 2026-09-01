FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# ============================================================================
# Build stage - Install dependencies
# ============================================================================

FROM base AS builder

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# For Docker: Install CPU-only PyTorch to reduce image size (use latest available CPU version)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Final stage - Runtime image
# ============================================================================

FROM base AS final

# Install runtime dependencies
# - curl: for healthcheck
# - libgomp1: required by scikit-learn, LightGBM, XGBoost (OpenMP)
# - libglib2.0-0: required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p \
    data/raw \
    data/processed \
    models \
    evaluation \
    mlflow_logs \
    monitoring_logs \
    retraining_logs \
    logs \
    && chmod -R 755 data models evaluation mlflow_logs monitoring_logs retraining_logs logs

# Expose ports
# 8000 for FastAPI
# 8501 for Streamlit
# 5000 for MLflow UI
EXPOSE 8000 8501 5000

# Health check for API service
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - can be overridden in docker-compose.yml
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
