import asyncio
from datetime import timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import TranscriptionJob, TranscriptionResult, Upload, utc_now
from app.db.repositories import (
    claim_job_processing,
    create_job,
    create_upload,
    get_job,
    get_or_create_job,
    mark_job_completed,
    mark_job_failed,
    upsert_task_result,
)


async def make_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(bind=engine, expire_on_commit=False)


def test_upload_job_result_models_round_trip():
    async def run():
        engine, Session = await make_session()

        try:
            async with Session() as session:
                upload = Upload(
                    id="upload_1",
                    object_key="uploads/upload_1/audio.wav",
                    filename="audio.wav",
                    content_type="audio/wav",
                    size_bytes=1024,
                    status="created",
                    created_at=utc_now(),
                    expires_at=utc_now(),
                )
                job = TranscriptionJob(
                    id="job_1",
                    upload_id="upload_1",
                    status="queued",
                    diarization=True,
                    min_speakers=1,
                    max_speakers=5,
                )
                result = TranscriptionResult(
                    job_id="job_1",
                    duration_sec=3.5,
                    text="hello",
                    utterances=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "hello"}],
                    diagnostics={"model": "e2e_rnnt"},
                )

                session.add(upload)
                session.add(job)
                await session.flush()
                session.add(result)
                await session.commit()
                loaded = await session.get(TranscriptionResult, "job_1")
        finally:
            await engine.dispose()

        assert loaded is not None
        assert loaded.text == "hello"
        assert loaded.utterances[0]["speaker"] == "SPEAKER_00"

    asyncio.run(run())


def test_repository_creates_job_and_marks_completed():
    async def run():
        engine, Session = await make_session()

        try:
            async with Session() as session:
                upload = await create_upload(
                    session,
                    upload_id="upload_repo",
                    object_key="uploads/upload_repo/audio.wav",
                    filename="audio.wav",
                    content_type="audio/wav",
                    size_bytes=2048,
                    expires_at=utc_now() + timedelta(minutes=15),
                )
                job = await create_job(
                    session,
                    job_id="job_repo",
                    upload_id=upload.id,
                    diarization=False,
                    num_speakers=None,
                    min_speakers=None,
                    max_speakers=None,
                )
                await mark_job_completed(
                    session,
                    job_id=job.id,
                    duration_sec=1.5,
                    text="recognized",
                    utterances=[],
                    diagnostics={"device": "cuda"},
                )
                loaded = await get_job(session, "job_repo")
        finally:
            await engine.dispose()

        assert loaded is not None
        assert loaded.status == "completed"
        assert loaded.result is not None
        assert loaded.result.utterances == []

    asyncio.run(run())


def test_repository_marks_job_failed():
    async def run():
        engine, Session = await make_session()

        try:
            async with Session() as session:
                await create_upload(
                    session,
                    upload_id="upload_failed",
                    object_key="uploads/upload_failed/audio.wav",
                    filename="audio.wav",
                    content_type="audio/wav",
                    size_bytes=2048,
                    expires_at=utc_now() + timedelta(minutes=15),
                )
                await create_job(
                    session,
                    job_id="job_failed",
                    upload_id="upload_failed",
                    diarization=True,
                    num_speakers=None,
                    min_speakers=1,
                    max_speakers=5,
                )
                await mark_job_failed(session, "job_failed", "audio_decode_failed", "Could not decode audio file")
                loaded = await get_job(session, "job_failed")
        finally:
            await engine.dispose()

        assert loaded is not None
        assert loaded.status == "failed"
        assert loaded.error_code == "audio_decode_failed"

    asyncio.run(run())


def test_get_or_create_job_returns_existing_duplicate_request():
    async def run():
        engine, Session = await make_session()

        try:
            async with Session() as session:
                await create_upload(
                    session,
                    upload_id="upload_duplicate",
                    object_key="uploads/upload_duplicate/audio.wav",
                    filename="audio.wav",
                    content_type="audio/wav",
                    size_bytes=2048,
                    expires_at=utc_now() + timedelta(minutes=15),
                )
                first, first_created = await get_or_create_job(
                    session,
                    job_id="job_first",
                    upload_id="upload_duplicate",
                    diarization=True,
                    num_speakers=None,
                    min_speakers=1,
                    max_speakers=5,
                )
                second, second_created = await get_or_create_job(
                    session,
                    job_id="job_second",
                    upload_id="upload_duplicate",
                    diarization=True,
                    num_speakers=None,
                    min_speakers=1,
                    max_speakers=5,
                )
        finally:
            await engine.dispose()

        assert first_created is True
        assert second_created is False
        assert second.id == first.id

    asyncio.run(run())


def test_claim_job_processing_only_claims_queued_job_once():
    async def run():
        engine, Session = await make_session()

        try:
            async with Session() as session:
                await create_upload(
                    session,
                    upload_id="upload_claim",
                    object_key="uploads/upload_claim/audio.wav",
                    filename="audio.wav",
                    content_type="audio/wav",
                    size_bytes=2048,
                    expires_at=utc_now() + timedelta(minutes=15),
                )
                await create_job(
                    session,
                    job_id="job_claim",
                    upload_id="upload_claim",
                    diarization=False,
                    num_speakers=None,
                    min_speakers=None,
                    max_speakers=None,
                )

                first = await claim_job_processing(session, "job_claim")
                second = await claim_job_processing(session, "job_claim")
        finally:
            await engine.dispose()

        assert first is True
        assert second is False

    asyncio.run(run())


def test_upsert_task_result_replaces_existing_payload():
    async def run():
        engine, Session = await make_session()

        try:
            async with Session() as session:
                await create_upload(
                    session,
                    upload_id="upload_task",
                    object_key="uploads/upload_task/audio.wav",
                    filename="audio.wav",
                    content_type="audio/wav",
                    size_bytes=2048,
                    expires_at=utc_now() + timedelta(minutes=15),
                )
                await create_job(
                    session,
                    job_id="job_task",
                    upload_id="upload_task",
                    diarization=False,
                    num_speakers=None,
                    min_speakers=None,
                    max_speakers=None,
                )
                await upsert_task_result(
                    session,
                    job_id="job_task",
                    task_type="asr",
                    status="completed",
                    payload={"text": "first"},
                )
                updated = await upsert_task_result(
                    session,
                    job_id="job_task",
                    task_type="asr",
                    status="completed",
                    payload={"text": "second"},
                )
        finally:
            await engine.dispose()

        assert updated.payload == {"text": "second"}

    asyncio.run(run())
