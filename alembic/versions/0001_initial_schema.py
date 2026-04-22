from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

json_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "uploads",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("object_key", sa.String(length=512), nullable=False, unique=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "transcription_jobs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("upload_id", sa.String(length=64), sa.ForeignKey("uploads.id"), nullable=False),
        sa.Column("request_key", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("diarization", sa.Boolean(), nullable=False),
        sa.Column("num_speakers", sa.Integer(), nullable=True),
        sa.Column("min_speakers", sa.Integer(), nullable=True),
        sa.Column("max_speakers", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("request_key", name="uq_transcription_jobs_request_key"),
    )
    op.create_table(
        "transcription_results",
        sa.Column(
            "job_id",
            sa.String(length=64),
            sa.ForeignKey("transcription_jobs.id"),
            primary_key=True,
        ),
        sa.Column("duration_sec", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("utterances", json_type, nullable=False),
        sa.Column("diagnostics", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "transcription_task_results",
        sa.Column(
            "job_id",
            sa.String(length=64),
            sa.ForeignKey("transcription_jobs.id"),
            primary_key=True,
        ),
        sa.Column("task_type", sa.String(length=32), primary_key=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("payload", json_type, nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("transcription_task_results")
    op.drop_table("transcription_results")
    op.drop_table("transcription_jobs")
    op.drop_table("uploads")
