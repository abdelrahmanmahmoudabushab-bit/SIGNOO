"""SQLAlchemy engine/session setup.

Postgres in production (docker-compose); SQLite works unmodified for local
development where Docker is unavailable — see `config.py`. No model in this file
uses a Postgres-only type, so both backends see the same schema.
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from server.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates tables directly — used by tests and first-run dev setup.

    Real deployments migrate with Alembic (`server/alembic`); this is a shortcut for
    the dev-simulation / test-suite path, not a replacement for migrations.
    """
    import server.models  # noqa: F401  (registers models on Base.metadata)

    Base.metadata.create_all(engine)
