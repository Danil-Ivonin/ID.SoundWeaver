from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


JsonList = MutableList.as_mutable(JSON().with_variant(JSONB, "postgresql"))
JsonDict = MutableDict.as_mutable(JSON().with_variant(JSONB, "postgresql"))


class Upload(Base):
    __tablename__ = "uploads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    object_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    jobs: Mapped[list["TranscriptionJob"]] = relationship(back_populates="upload")


class TranscriptionJob(Base):
    __tablename__ = "transcription_jobs"
    __table_args__ = (UniqueConstraint("request_key", name="uq_transcription_jobs_request_key"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    upload_id: Mapped[str] = mapped_column(ForeignKey("uploads.id"), nullable=False)
    request_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    diarization: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    num_speakers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_speakers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_speakers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    upload: Mapped[Upload] = relationship(back_populates="jobs")
    result: Mapped["TranscriptionResult | None"] = relationship(back_populates="job")
    task_results: Mapped[list["TranscriptionTaskResult"]] = relationship(back_populates="job")


class TranscriptionResult(Base):
    __tablename__ = "transcription_results"

    job_id: Mapped[str] = mapped_column(ForeignKey("transcription_jobs.id"), primary_key=True)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    utterances: Mapped[list[dict]] = mapped_column(JsonList, nullable=False, default=list)
    diagnostics: Mapped[dict] = mapped_column(JsonDict, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    job: Mapped[TranscriptionJob] = relationship(back_populates="result")


class TranscriptionTaskResult(Base):
    __tablename__ = "transcription_task_results"

    job_id: Mapped[str] = mapped_column(ForeignKey("transcription_jobs.id"), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JsonDict, nullable=False, default=dict)
    exec_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    job: Mapped[TranscriptionJob] = relationship(back_populates="task_results")
