"""training lock, rollout targets table, per-language knowledge base, checkpoint hash

Rollout targets move out of a JSON column into a table; the existing lists are copied
across before the column is dropped, so no rollout loses its targets.

Revision ID: c4d8f1a6e932
Revises: b7e2d4a81c55
Create Date: 2026-10-12 10:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c4d8f1a6e932'
down_revision: Union[str, None] = 'b7e2d4a81c55'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('model_versions', sa.Column('checkpoint_sha256', sa.String(), nullable=True))

    op.create_table('training_locks',
    sa.Column('language', sa.String(), nullable=False),
    sa.Column('holder', sa.String(), nullable=False),
    sa.Column('acquired_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('language')
    )

    # Existing knowledge-base entries were all written for the original language.
    with op.batch_alter_table('kb_entries') as batch:
        batch.add_column(sa.Column('language', sa.String(), nullable=False, server_default='liu'))
        batch.create_index(op.f('ix_kb_entries_language'), ['language'], unique=False)

    op.create_table('rollout_targets',
    sa.Column('rollout_id', sa.String(), nullable=False),
    sa.Column('kiosk_id', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['rollout_id'], ['rollouts.id'], ),
    sa.PrimaryKeyConstraint('rollout_id', 'kiosk_id')
    )
    op.create_index(op.f('ix_rollout_targets_kiosk_id'), 'rollout_targets', ['kiosk_id'], unique=False)

    bind = op.get_bind()
    rollouts = sa.table('rollouts', sa.column('id', sa.String), sa.column('target_kiosk_ids', sa.JSON))
    targets = sa.table('rollout_targets', sa.column('rollout_id', sa.String), sa.column('kiosk_id', sa.String))
    for rollout_id, kiosk_ids in bind.execute(sa.select(rollouts.c.id, rollouts.c.target_kiosk_ids)).fetchall():
        for kiosk_id in dict.fromkeys(kiosk_ids or []):
            bind.execute(targets.insert().values(rollout_id=rollout_id, kiosk_id=kiosk_id))

    with op.batch_alter_table('rollouts') as batch:
        batch.drop_column('target_kiosk_ids')


def downgrade() -> None:
    with op.batch_alter_table('rollouts') as batch:
        batch.add_column(sa.Column('target_kiosk_ids', sa.JSON(), nullable=False, server_default='[]'))

    bind = op.get_bind()
    rollouts = sa.table('rollouts', sa.column('id', sa.String), sa.column('target_kiosk_ids', sa.JSON))
    targets = sa.table('rollout_targets', sa.column('rollout_id', sa.String), sa.column('kiosk_id', sa.String))
    grouped: dict[str, list[str]] = {}
    for rollout_id, kiosk_id in bind.execute(sa.select(targets.c.rollout_id, targets.c.kiosk_id)).fetchall():
        grouped.setdefault(rollout_id, []).append(kiosk_id)
    for rollout_id, kiosk_ids in grouped.items():
        bind.execute(rollouts.update().where(rollouts.c.id == rollout_id).values(target_kiosk_ids=kiosk_ids))

    op.drop_index(op.f('ix_rollout_targets_kiosk_id'), table_name='rollout_targets')
    op.drop_table('rollout_targets')
    with op.batch_alter_table('kb_entries') as batch:
        batch.drop_index(op.f('ix_kb_entries_language'))
        batch.drop_column('language')
    op.drop_table('training_locks')
    op.drop_column('model_versions', 'checkpoint_sha256')
