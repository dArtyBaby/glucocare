# Dockerfile
# WHY multi-stage build:
#   Stage 1 (builder): Install ALL dependencies including build tools (gcc, etc.)
#   Stage 2 (runtime): Copy only what's needed. Final image has NO build tools.
#   Result: Production image is ~40% smaller and has fewer attack surfaces.

# ===== STAGE 1: Builder =====
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
# WHY: If requirements.txt doesn't change, Docker reuses the cached layer.
# This makes rebuilds 10x faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ===== STAGE 2: Runtime =====
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# WHY non-root user:
# Running as root inside a container is a security risk.
# If an attacker escapes the container, they'd have root on the host.
RUN useradd --create-home --shell /bin/bash glucocare
RUN chown -R glucocare:glucocare /app
USER glucocare

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# WHY gunicorn in production (not uvicorn directly):
# Gunicorn is a battle-tested process manager. It handles worker crashes,
# graceful restarts, and process supervision. Uvicorn workers handle async.
CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
