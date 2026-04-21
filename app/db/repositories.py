from datetime import datetime, timezone

from sqlalchemy.orm import Session, joinedload

from app.db.models import TranscriptionJob, TranscriptionResult, Upload


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def create_upload(
    session: Session,
    *,
    upload_id: str,
    object_key: str,
    filename: str,
    content_type: str,
    size_bytes: int,
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
    session.commit()
    session.refresh(upload)
    return upload


def get_upload(session: Session, upload_id: str) -> Upload | None:
    return session.get(Upload, upload_id)


def create_job(
    session: Session,
    *,
    job_id: str,
    upload_id: str,
    diarization: bool,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> TranscriptionJob:
    job = TranscriptionJob(
        id=job_id,
        upload_id=upload_id,
        status="queued",
        diarization=diarization,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_job(session: Session, job_id: str) -> TranscriptionJob | None:
    return (
        session.query(TranscriptionJob)
        .options(joinedload(TranscriptionJob.upload), joinedload(TranscriptionJob.result))
        .filter(TranscriptionJob.id == job_id)
        .one_or_none()
    )


def mark_job_processing(session: Session, job_id: str) -> None:
    job = session.get(TranscriptionJob, job_id)
    if job is None:
        return
    job.status = "processing"
    now = now_utc()
    job.started_at = now
    job.updated_at = now
    session.commit()


def mark_job_completed(
    session: Session,
    *,
    job_id: str,
    duration_sec: float,
    text: str,
    utterances: list[dict],
    diagnostics: dict,
) -> None:
    job = session.get(TranscriptionJob, job_id)
    if job is None:
        return
    now = now_utc()
    job.status = "completed"
    job.finished_at = now
    job.updated_at = now
    session.add(
        TranscriptionResult(
            job_id=job_id,
            duration_sec=duration_sec,
            text=text,
            utterances=utterances,
            diagnostics=diagnostics,
        )
    )
    session.commit()


def mark_job_failed(session: Session, job_id: str, error_code: str, error_message: str) -> None:
    job = session.get(TranscriptionJob, job_id)
    if job is None:
        return
    now = now_utc()
    job.status = "failed"
    job.error_code = error_code
    job.error_message = error_message
    job.finished_at = now
    job.updated_at = now
    session.commit()
