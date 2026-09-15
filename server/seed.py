"""Creates the first admin user, for a fresh database. Run once.

    python -m server.seed admin@signoo.dev changeme123

Not exposed as an API endpoint on purpose — see routers/auth.py, whose /auth/users
endpoint requires an existing admin token, which does not exist yet on a fresh DB.
"""
from __future__ import annotations

import sys

from server.auth import hash_password
from server.db import SessionLocal, init_db
from server.models import User


def seed_admin(email: str, password: str) -> None:
    init_db()
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).one_or_none() is not None:
            print(f"user {email} already exists")
            return
        db.add(User(email=email, hashed_password=hash_password(password), role="admin"))
        db.commit()
        print(f"created admin user {email}")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m server.seed <email> <password>")
        raise SystemExit(1)
    seed_admin(sys.argv[1], sys.argv[2])
