# Multi-stage build: builder + runtime
FROM python:3.12-slim as builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PORT=8000

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Application code
COPY config ./config
COPY api ./api
COPY src ./src

# Optional: copy pre-trained models (or mount at runtime)
RUN mkdir -p models/churn models/anomaly data/feature_store data/raw data/processed

# Run with gunicorn + uvicorn workers
CMD ["gunicorn", "api.main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--timeout", "120"]
