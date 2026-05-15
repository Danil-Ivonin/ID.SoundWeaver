from collections.abc import Generator

import pytest

from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    get_settings.cache_clear()
    keys = [
        "DB_DRIVER",
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "REDIS_URL",
        "MINIO_ENDPOINT",
        "MINIO_PUBLIC_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET",
        "MINIO_SECURE",
        "HF_TOKEN",
        "PYANNOTE_MODEL",
        "GIGAAM_MODEL",
        "DEVICE",
        "MAX_AUDIO_DURATION_SEC",
        "TRANSCRIPTION_CHUNK_DURATION_SEC",
        "TRANSCRIPTION_CHUNK_STRIDE_SEC",
        "PRESIGNED_UPLOAD_TTL_SEC",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield
    get_settings.cache_clear()
