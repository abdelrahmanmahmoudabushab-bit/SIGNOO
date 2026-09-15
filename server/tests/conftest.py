"""Shared pytest fixtures: an isolated SQLite DB per test, plus a TestClient wired
to it via a `get_db` override (never the dev-machine signoo_dev.db)."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.db import Base, get_db
from server.main import app


@pytest.fixture()
def db_session(tmp_path: Path):
    db_path = tmp_path / f"test_{uuid.uuid4().hex}.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    import server.models  # noqa: F401
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    session = session_local()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def client(db_session):
    def _override():
        yield db_session

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def make_user(db_session):
    from server.auth import create_access_token, hash_password
    from server.models import User

    def _make(email: str, role: str, password: str = "password123") -> str:
        user = User(email=email, hashed_password=hash_password(password), role=role)
        db_session.add(user)
        db_session.commit()
        return create_access_token(subject=user.id, role=role)

    return _make
