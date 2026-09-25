# Stage 1: GDAL/PROJ base
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.5 AS gdal-base

# Stage 2: Application image
FROM gdal-base AS app

ARG PYTHON_VERSION=3.11
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    pandoc \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /project

# Install Python dependencies
COPY environment.yml .
RUN pip install --upgrade pip && \
    pip install conda-lock || true

# Install from requirements derived from environment.yml
# Using pip directly for Docker compatibility
COPY requirements_docker.txt .
RUN pip install -r requirements_docker.txt

# Copy project source
COPY . .

# Validate project structure
RUN python scripts/p0_scaffold.py

# Default: run full pipeline
CMD ["make", "pipeline"]
