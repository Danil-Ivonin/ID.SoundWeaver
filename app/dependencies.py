from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from app.db.repositories import create_job, create_upload, get_job, get_or_create_job, get_upload
from app.db.session import SessionLocal
from app.settings import Settings, get_settings
from app.storage.minio import AsyncS3Storage
from app.tasks.celery_app import celery_app


class UploadRepo(Protocol):
    async def create_upload(self, **kwargs): ...


class JobRepo(Protocol):
    async def get_upload(self, upload_id: str): ...

    async def create_job(self, **kwargs): ...

    async def get_or_create_job(self, **kwargs): ...

    async def get_job(self, job_id: str): ...


class Storage(Protocol):
    def build_object_key(self, upload_id: str, filename: str) -> str: ...

    async def create_presigned_put_url(self, object_key: str) -> str: ...


class Queue(Protocol):
    def enqueue(self, job_id: str) -> None: ...


Clock = Callable[[], datetime]
IdGenerator = Callable[[], str]


class RuntimeUploadRepo:
    async def create_upload(self, **kwargs):
        async with SessionLocal() as session:
            return await create_upload(session, **kwargs)


class RuntimeJobRepo:
    async def get_upload(self, upload_id: str):
        async with SessionLocal() as session:
            return await get_upload(session, upload_id)

    async def create_job(self, **kwargs):
        async with SessionLocal() as session:
            return await create_job(session, **kwargs)

    async def get_or_create_job(self, **kwargs):
        async with SessionLocal() as session:
            return await get_or_create_job(session, **kwargs)

    async def get_job(self, job_id: str):
        async with SessionLocal() as session:
            return await get_job(session, job_id)


class RuntimeQueue:
    def enqueue(self, job_id: str) -> None:
        celery_app.send_task("prepare_transcription_job", args=[job_id])


def new_id() -> str:
    return uuid4().hex


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class AppDependencies:
    settings: Settings
    storage: Storage
    upload_repo: UploadRepo
    job_repo: JobRepo
    queue: Queue
    now: Clock = now_utc
    new_id: IdGenerator = new_id


def build_app_dependencies(
    *,
    settings: Settings | None = None,
    storage: Storage | None = None,
    upload_repo: UploadRepo | None = None,
    job_repo: JobRepo | None = None,
    queue: Queue | None = None,
    now: Clock | None = None,
    new_id_generator: IdGenerator | None = None,
) -> AppDependencies:
    app_settings = settings or get_settings()
    return AppDependencies(
        settings=app_settings,
        storage=storage or AsyncS3Storage(app_settings),
        upload_repo=upload_repo or RuntimeUploadRepo(),
        job_repo=job_repo or RuntimeJobRepo(),
        queue=queue or RuntimeQueue(),
        now=now or now_utc,
        new_id=new_id_generator or new_id,
    )


def storage_supports_object_exists(storage: Storage) -> Any:
    return getattr(storage, "object_exists", None)


def storage_supports_ensure_bucket(storage: Storage) -> Any:
    return getattr(storage, "ensure_bucket", None)
