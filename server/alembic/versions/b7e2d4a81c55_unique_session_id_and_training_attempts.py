"""unique session_id on uploads, and a record of crashed automatic training runs

Fails if two stored uploads already share a session_id; remove the duplicate upload
(and its review items) first, since those clips are in the queue twice.

Revision ID: b7e2d4a81c55
Revises: a3f1c9d27b40
Create Date: 2026-10-11 13:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b7e2d4a81c55'
down_revision: Union[str, None] = 'a3f1c9d27b40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index(op.f('ix_session_uploads_session_id'), table_name='session_uploads')
    op.create_index(op.f('ix_session_uploads_session_id'), 'session_uploads', ['session_id'], unique=True)
    op.create_table('training_attempts',
    sa.Column('language', sa.String(), nullable=False),
    sa.Column('attempted_at', sa.DateTime(), nullable=False),
    sa.Column('error', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('language')
    )


def downgrade() -> None:
    op.drop_table('training_attempts')
    op.drop_index(op.f('ix_session_uploads_session_id'), table_name='session_uploads')
    op.create_index(op.f('ix_session_uploads_session_id'), 'session_uploads', ['session_id'], unique=False)
