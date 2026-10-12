"""per-language kiosk model versions and training-run provenance

Revision ID: a3f1c9d27b40
Revises: 76adb2b2864e
Create Date: 2026-10-11 12:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a3f1c9d27b40'
down_revision: Union[str, None] = '76adb2b2864e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('kiosks', sa.Column('model_versions', sa.JSON(), nullable=False, server_default='{}'))
    op.add_column('model_versions', sa.Column('training_started_at', sa.DateTime(), nullable=True))
    op.add_column('model_versions', sa.Column('champion_version', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_versions', 'champion_version')
    op.drop_column('model_versions', 'training_started_at')
    op.drop_column('kiosks', 'model_versions')
