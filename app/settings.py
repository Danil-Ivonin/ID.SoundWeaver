from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://soundweaver:soundweaver@localhost:5432/soundweaver"
    redis_url: str = "redis://localhost:6379/0"

    minio_endpoint: str = "localhost:9000"
    minio_public_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "soundweaver"
    minio_secret_key: str = "soundweaver-secret"
    minio_bucket: str = "soundweaver-audio"
    minio_secure: bool = False

    hf_token: str = ""
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    gigaam_model: str = "e2e_rnnt"
    device: str = "cuda"

    max_upload_size_bytes: int = Field(default=104_857_600, ge=1)
    max_audio_duration_sec: int = Field(default=3600, ge=1)
    presigned_upload_ttl_sec: int = Field(default=900, ge=60)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
