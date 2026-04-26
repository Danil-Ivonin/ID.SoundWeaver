from datetime import datetime, timezone
from hashlib import sha256

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import TranscriptionJob, TranscriptionResult, TranscriptionTaskResult, Upload


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def build_transcription_request_key(
    *,
    upload_id: str,
    diarization: bool,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> str:
    raw = "|".join(
        [
            upload_id,
            str(diarization),
            str(num_speakers or ""),
            str(min_speakers or ""),
            str(max_speakers or ""),
        ]
    )
    return sha256(raw.encode("utf-8")).hexdigest()


async def create_upload(
    session: AsyncSession,
    *,
    upload_id: str,
    object_key: str,
    filename: str,
    content_type: str,
    size_bytes: int | None,
    expires_at: datetime,
) -> Upload:
    upload = Upload(
        id=upload_id,
        object_key=object_key,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        status="created",
        expires_at=expires_at,
    )
    session.add(upload)
    await session.commit()
    await session.refresh(upload)
    return upload


async def get_upload(session: AsyncSession, upload_id: str) -> Upload | None:
    return await session.get(Upload, upload_id)


async def create_job(
    session: AsyncSession,
    *,
    job_id: str,
    upload_id: str,
    diarization: bool,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> TranscriptionJob:
    job, _ = await get_or_create_job(
        session,
        job_id=job_id,
        upload_id=upload_id,
        diarization=diarization,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return job


async def get_or_create_job(
    session: AsyncSession,
    *,
    job_id: str,
    upload_id: str,
    diarization: bool,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> tuple[TranscriptionJob, bool]:
    request_key = build_transcription_request_key(
        upload_id=upload_id,
        diarization=diarization,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    existing = await get_job_by_request_key(session, request_key)
    if existing is not None:
        return existing, False

    job = TranscriptionJob(
        id=job_id,
        upload_id=upload_id,
        request_key=request_key,
        status="queued",
        diarization=diarization,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    session.add(job)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await get_job_by_request_key(session, request_key)
        if existing is not None:
            return existing, False
        raise
    await session.refresh(job)
    return job, True


async def get_job_by_request_key(session: AsyncSession, request_key: str) -> TranscriptionJob | None:
    result = await session.execute(
        select(TranscriptionJob)
        .options(selectinload(TranscriptionJob.upload), selectinload(TranscriptionJob.result))
        .where(TranscriptionJob.request_key == request_key)
    )
    return result.scalar_one_or_none()


async def get_job(session: AsyncSession, job_id: str) -> TranscriptionJob | None:
    result = await session.execute(
        select(TranscriptionJob)
        .options(
            selectinload(TranscriptionJob.upload),
            selectinload(TranscriptionJob.result),
            selectinload(TranscriptionJob.task_results),
        )
        .where(TranscriptionJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def claim_job_processing(session: AsyncSession, job_id: str) -> bool:
    now = now_utc()
    result = await session.execute(
        update(TranscriptionJob)
        .where(TranscriptionJob.id == job_id, TranscriptionJob.status == "queued")
        .values(status="processing", started_at=now, updated_at=now)
    )
    await session.commit()
    return result.rowcount == 1


async def mark_job_processing(session: AsyncSession, job_id: str) -> None:
    await claim_job_processing(session, job_id)


async def mark_job_completed(
    session: AsyncSession,
    *,
    job_id: str,
    duration_sec: float,
    text: str,
    utterances: list[dict],
    diagnostics: dict,
) -> None:
    job = await session.get(TranscriptionJob, job_id)
    if job is None or job.status == "failed":
        return

    now = now_utc()
    job.status = "completed"
    job.finished_at = now
    job.updated_at = now
    result = TranscriptionResult(
        job_id=job_id,
        duration_sec=duration_sec,
        text=text,
        utterances=utterances,
        diagnostics=diagnostics,
    )
    await session.merge(result)
    await session.commit()


async def mark_job_failed(session: AsyncSession, job_id: str, error_code: str, error_message: str) -> None:
    job = await session.get(TranscriptionJob, job_id)
    if job is None or job.status == "completed":
        return

    now = now_utc()
    job.status = "failed"
    job.error_code = error_code
    job.error_message = error_message
    job.finished_at = now
    job.updated_at = now
    await session.commit()


async def upsert_task_result(
    session: AsyncSession,
    *,
    job_id: str,
    task_type: str,
    status: str,
    payload: dict,
    exec_duration: float | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> TranscriptionTaskResult:
    now = now_utc()
    task_result = TranscriptionTaskResult(
        job_id=job_id,
        task_type=task_type,
        status=status,
        payload=payload,
        exec_duration=exec_duration,
        error_code=error_code,
        error_message=error_message,
        updated_at=now,
    )
    merged = await session.merge(task_result)
    await session.commit()
    return merged


async def get_task_results(session: AsyncSession, job_id: str) -> dict[str, TranscriptionTaskResult]:
    result = await session.execute(
        select(TranscriptionTaskResult).where(TranscriptionTaskResult.job_id == job_id)
    )
    return {task_result.task_type: task_result for task_result in result.scalars().all()}
