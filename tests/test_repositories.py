from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import TranscriptionJob, TranscriptionResult, Upload, utc_now
from app.db.repositories import (
    create_job,
    create_upload,
    get_job,
    mark_job_completed,
    mark_job_failed,
)


def make_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_upload_job_result_models_round_trip():
    Session = make_session()

    with Session() as session:
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
        session.flush()
        session.add(result)
        session.commit()
        loaded = session.get(TranscriptionResult, "job_1")

    assert loaded is not None
    assert loaded.text == "hello"
    assert loaded.utterances[0]["speaker"] == "SPEAKER_00"


def test_repository_creates_job_and_marks_completed():
    Session = make_session()

    with Session() as session:
        upload = create_upload(
            session,
            upload_id="upload_repo",
            object_key="uploads/upload_repo/audio.wav",
            filename="audio.wav",
            content_type="audio/wav",
            size_bytes=2048,
            expires_at=utc_now() + timedelta(minutes=15),
        )
        job = create_job(
            session,
            job_id="job_repo",
            upload_id=upload.id,
            diarization=False,
            num_speakers=None,
            min_speakers=None,
            max_speakers=None,
        )
        mark_job_completed(
            session,
            job_id=job.id,
            duration_sec=1.5,
            text="recognized",
            utterances=[],
            diagnostics={"device": "cuda"},
        )
        loaded = get_job(session, "job_repo")

    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.result is not None
    assert loaded.result.utterances == []


def test_repository_marks_job_failed():
    Session = make_session()

    with Session() as session:
        create_upload(
            session,
            upload_id="upload_failed",
            object_key="uploads/upload_failed/audio.wav",
            filename="audio.wav",
            content_type="audio/wav",
            size_bytes=2048,
            expires_at=utc_now() + timedelta(minutes=15),
        )
        create_job(
            session,
            job_id="job_failed",
            upload_id="upload_failed",
            diarization=True,
            num_speakers=None,
            min_speakers=1,
            max_speakers=5,
        )
        mark_job_failed(session, "job_failed", "audio_decode_failed", "Could not decode audio file")
        loaded = get_job(session, "job_failed")

    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.error_code == "audio_decode_failed"
