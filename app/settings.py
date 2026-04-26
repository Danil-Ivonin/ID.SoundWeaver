from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_driver: str = "postgresql+asyncpg"
    db_user: str = "soundweaver"
    db_password: str = "soundweaver"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "soundweaver"
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

    max_audio_duration_sec: int = Field(default=3600, ge=1)
    presigned_upload_ttl_sec: int = Field(default=900, ge=60)

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"{self.db_driver}://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
