from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_task_duration"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("transcription_task_results", sa.Column("exec_duration", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("transcription_task_results", "exec_duration")
