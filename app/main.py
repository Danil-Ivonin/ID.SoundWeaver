from datetime import datetime, timezone

from fastapi import FastAPI

from app.api import health, transcriptions, uploads
from app.db.repositories import create_job, create_upload, get_job, get_upload
from app.settings import get_settings
from app.storage.minio import build_object_key, create_minio_client, create_presigned_put_url


class RuntimeStorage:
    def __init__(self, settings):
        self.settings = settings

    def build_object_key(self, upload_id: str, filename: str) -> str:
        return build_object_key(upload_id, filename)

    def create_presigned_put_url(self, object_key: str) -> str:
        client = create_minio_client(self.settings)
        return create_presigned_put_url(
            client,
            self.settings.minio_bucket,
            object_key,
            self.settings.presigned_upload_ttl_sec,
        )

    def download_to_file(self, object_key: str, path) -> None:
        client = create_minio_client(self.settings)
        client.fget_object(self.settings.minio_bucket, object_key, str(path))


class RuntimeUploadRepo:
    def create_upload(self, **kwargs):
        from app.db.session import SessionLocal

        with SessionLocal() as session:
            return create_upload(session, **kwargs)


class RuntimeJobRepo:
    def get_upload(self, upload_id: str):
        from app.db.session import SessionLocal

        with SessionLocal() as session:
            return get_upload(session, upload_id)

    def create_job(self, **kwargs):
        from app.db.session import SessionLocal

        with SessionLocal() as session:
            return create_job(session, **kwargs)

    def get_job(self, job_id: str):
        from app.db.session import SessionLocal

        with SessionLocal() as session:
            return get_job(session, job_id)


class RuntimeQueue:
    def enqueue(self, job_id: str) -> None:
        from app.tasks.transcription import transcribe_audio

        transcribe_audio.delay(job_id)


def create_app() -> FastAPI:
    app = FastAPI(title="ID.SoundWeaver")
    settings = get_settings()
    app.state.storage = RuntimeStorage(settings)
    app.state.upload_repo = RuntimeUploadRepo()
    app.state.job_repo = RuntimeJobRepo()
    app.state.queue = RuntimeQueue()
    app.state.now = lambda: datetime.now(timezone.utc)
    app.include_router(health.router)
    app.include_router(uploads.router)
    app.include_router(transcriptions.router)
    return app


app = create_app()
