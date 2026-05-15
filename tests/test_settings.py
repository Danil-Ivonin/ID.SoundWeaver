from app.settings import Settings


def test_settings_load_expected_env(monkeypatch):
    monkeypatch.setenv("DB_DRIVER", "postgresql+psycopg")
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "http://localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "secret")
    monkeypatch.setenv("MINIO_BUCKET", "audio")
    monkeypatch.setenv("HF_TOKEN", "hf_token")

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+psycopg://u:p@localhost:5432/db"
    assert settings.db_driver == "postgresql+psycopg"
    assert settings.db_host == "localhost"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.minio_bucket == "audio"
    assert settings.device == "cuda"
    assert settings.max_audio_duration_sec == 3600
    assert settings.transcription_chunk_duration_sec == 30
    assert settings.transcription_chunk_stride_sec == 25
    assert settings.presigned_upload_ttl_sec == 900


def test_settings_parses_minio_secure(monkeypatch):
    monkeypatch.setenv("MINIO_SECURE", "true")

    settings = Settings(_env_file=None)

    assert settings.minio_secure is True


def test_settings_use_default_db_parts():
    settings = Settings(_env_file=None)

    assert settings.db_driver == "postgresql+asyncpg"
    assert settings.db_user == "soundweaver"
    assert settings.db_password == "soundweaver"
    assert settings.db_host == "localhost"
    assert settings.db_port == 5432
    assert settings.db_name == "soundweaver"
    assert settings.database_url == "postgresql+asyncpg://soundweaver:soundweaver@localhost:5432/soundweaver"
    assert settings.max_audio_duration_sec == 3600
    assert settings.transcription_chunk_duration_sec == 30
    assert settings.transcription_chunk_stride_sec == 25


def test_settings_loads_transcription_chunk_window(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_CHUNK_DURATION_SEC", "20")
    monkeypatch.setenv("TRANSCRIPTION_CHUNK_STRIDE_SEC", "15")

    settings = Settings(_env_file=None)

    assert settings.transcription_chunk_duration_sec == 20
    assert settings.transcription_chunk_stride_sec == 15
