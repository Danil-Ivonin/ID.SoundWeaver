from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, transcriptions, uploads
from app.async_utils import maybe_await
from app.dependencies import (
    Clock,
    IdGenerator,
    JobRepo,
    Queue,
    Storage,
    UploadRepo,
    build_app_dependencies,
    storage_supports_ensure_bucket,
)
from app.settings import Settings


def create_app(
    *,
    settings: Settings | None = None,
    storage: Storage | None = None,
    upload_repo: UploadRepo | None = None,
    job_repo: JobRepo | None = None,
    queue: Queue | None = None,
    now: Clock | None = None,
    new_id: IdGenerator | None = None,
) -> FastAPI:
    deps = build_app_dependencies(
        settings=settings,
        storage=storage,
        upload_repo=upload_repo,
        job_repo=job_repo,
        queue=queue,
        now=now,
        new_id_generator=new_id,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ensure_bucket = storage_supports_ensure_bucket(deps.storage)
        if ensure_bucket is not None:
            await maybe_await(ensure_bucket())
        yield

    app = FastAPI(title="ID.SoundWeaver", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(uploads.build_uploads_router(deps))
    app.include_router(transcriptions.build_transcriptions_router(deps))
    return app


app = create_app()
