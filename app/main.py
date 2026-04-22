from datetime import datetime, timezone

from fastapi import FastAPI

from app.api import health, transcriptions, uploads
from app.db.repositories import create_upload, get_job, get_or_create_job, get_upload
from app.db.session import SessionLocal
from app.settings import get_settings
from app.storage.minio import AsyncS3Storage


class RuntimeUploadRepo:
    async def create_upload(self, **kwargs):
        async with SessionLocal() as session:
            return await create_upload(session, **kwargs)


class RuntimeJobRepo:
    async def get_upload(self, upload_id: str):
        async with SessionLocal() as session:
            return await get_upload(session, upload_id)

    async def get_or_create_job(self, **kwargs):
        async with SessionLocal() as session:
            return await get_or_create_job(session, **kwargs)

    async def get_job(self, job_id: str):
        async with SessionLocal() as session:
            return await get_job(session, job_id)


class RuntimeQueue:
    def enqueue(self, job_id: str) -> None:
        from app.tasks.transcription import prepare_transcription_job

        prepare_transcription_job.delay(job_id)


def create_app() -> FastAPI:
    app = FastAPI(title="ID.SoundWeaver")
    settings = get_settings()
    app.state.storage = AsyncS3Storage(settings)
    app.state.upload_repo = RuntimeUploadRepo()
    app.state.job_repo = RuntimeJobRepo()
    app.state.queue = RuntimeQueue()
    app.state.now = lambda: datetime.now(timezone.utc)
    app.include_router(health.router)
    app.include_router(uploads.router)
    app.include_router(transcriptions.router)
    return app


app = create_app()
