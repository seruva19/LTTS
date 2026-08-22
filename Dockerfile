# syntax=docker/dockerfile:1.7
# =============================================================================
# LTTS Docker Image
# =============================================================================
# Build:  docker build --build-context ltts_modules=../LTTS-modules -t ltts .
# Run:    docker run -p 8000:8000 -v ./user:/app/user ltts
# GPU:    docker run --gpus all -p 8000:8000 -v ./user:/app/user ltts

FROM alpine:3.22 AS modules
COPY --from=ltts_modules . /src
RUN find /src -type d \( \
        -name venv -o -name .venv -o -name __pycache__ -o -name .git \
    \) -prune -exec rm -rf '{}' +

FROM python:3.10-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
COPY requirements-modules.txt .
ARG USE_GPU=false
RUN python -m pip install --upgrade pip
RUN if [ "$USE_GPU" = "true" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        grep -v -E '^(--extra-index-url|torch==|torchvision==|torchaudio==)' requirements.txt > requirements.runtime.txt && \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir -r requirements.runtime.txt; \
    fi

# Common analysis-module runtimes.  The two heavyweight specialist modules
# (SAELens and repeng) remain explicitly optional and report SKIP when absent.
RUN pip install --no-cache-dir -r requirements-modules.txt

# Copy application
COPY . .

# Analysis modules are maintained in their own repository.  Compose supplies
# the required sibling checkout as this named build context.  For a direct
# build use: --build-context ltts_modules=../LTTS-modules.
COPY --from=modules /src /opt/LTTS-modules

# Build frontend (if node is available)
FROM node:22-slim AS frontend
WORKDIR /app
COPY package*.json ./
RUN npm ci --ignore-scripts
COPY . .
RUN npm run build-client

# Final image
FROM base AS final
COPY --from=frontend /app/dist /app/dist

# Ensure user config directory exists
RUN mkdir -p /app/user /app/models

# Environment defaults
ENV LTTS_HOST=0.0.0.0
ENV LTTS_PORT=8000
ENV LTTS_LOG_LEVEL=INFO
ENV LTTS_MODULES_DIR=/opt/LTTS-modules
# All deps are installed in the main environment, so skip per-module venvs.
ENV LTTS_SKIP_MODULE_VENVS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "uvicorn", "ltts:app", "--host", "0.0.0.0", "--port", "8000"]
