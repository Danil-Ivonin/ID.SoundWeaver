from app.settings import Settings


def test_settings_load_expected_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "http://localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "secret")
    monkeypatch.setenv("MINIO_BUCKET", "audio")
    monkeypatch.setenv("HF_TOKEN", "hf_token")

    settings = Settings()

    assert settings.database_url == "postgresql+psycopg://u:p@localhost:5432/db"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.minio_bucket == "audio"
    assert settings.device == "cuda"
    assert settings.max_upload_size_bytes == 104_857_600
    assert settings.max_audio_duration_sec == 300
    assert settings.presigned_upload_ttl_sec == 900


def test_settings_parses_minio_secure(monkeypatch):
    monkeypatch.setenv("MINIO_SECURE", "true")

    settings = Settings()

    assert settings.minio_secure is True
