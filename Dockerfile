FROM python:3.12-slim-bookworm AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --python 3.12

COPY app ./app
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS worker

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libpython3.12 \
        python3.12 \
        python3-pip \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --break-system-packages --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra worker --python python3.12

COPY app ./app
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

CMD ["celery", "-A", "app.tasks.celery_app.celery_app", "worker", "--loglevel=INFO", "--concurrency=3"]
