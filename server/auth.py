"""JWT auth and role-based least-privilege enforcement (FR-W9, NFR-S4).

Roles: operator | reviewer | engineer | admin. Each router declares the minimal set
of roles allowed per endpoint via `require_role(...)` — never a blanket "any
authenticated user".
"""
from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from server.config import settings
from server.db import get_db
from server.models import User

ROLES = ("operator", "reviewer", "engineer", "admin")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_access_token(*, subject: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


class CurrentUser:
    def __init__(self, user_id: str, email: str, role: str):
        self.user_id = user_id
        self.email = email
        self.role = role


def get_current_user(
    token: str | None = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> CurrentUser:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise unauthorized
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise unauthorized
    user_id = payload.get("sub")
    role = payload.get("role")
    if user_id is None or role is None:
        raise unauthorized
    user = db.get(User, user_id)
    if user is None:
        raise unauthorized
    return CurrentUser(user_id=user.id, email=user.email, role=user.role)


def kiosk_token_for(kiosk_id: str) -> str:
    """The credential issued to one kiosk in `per_kiosk` mode: HMAC-SHA256 of its id under
    the server's master token. Derived, so the server stores nothing per kiosk and a
    token leaked from one kiosk opens nothing belonging to another."""
    return hmac.new(settings.kiosk_token.encode(), kiosk_id.encode(), hashlib.sha256).hexdigest()


def require_kiosk_token(x_kiosk_token: str | None = Header(default=None)) -> str:
    """Kiosk-facing endpoints (ingest, register, heartbeat, model download) accept only
    requests that carry a kiosk token. Returns the token it was given so a handler that
    knows the kiosk id can call `authorize_kiosk`.

    "shared" mode is fully decided here. "per_kiosk" mode can only be decided once the
    kiosk id is known, which for ingest and register is in the request body, so here it
    only insists a token is present. With no master token configured the check is off,
    which is only reachable in `dev`; production startup refuses a blank token.
    """
    expected = settings.kiosk_token
    presented = x_kiosk_token or ""
    if not expected:
        return presented
    if settings.kiosk_token_mode == "per_kiosk":
        if not presented:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid kiosk token")
    elif not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid kiosk token")
    return presented


def authorize_kiosk(presented: str, kiosk_id: str) -> None:
    """In `per_kiosk` mode, 401 unless `presented` is the token issued to `kiosk_id`."""
    if not settings.kiosk_token or settings.kiosk_token_mode != "per_kiosk":
        return
    if not hmac.compare_digest(presented, kiosk_token_for(kiosk_id)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"this token was not issued to kiosk '{kiosk_id}'")


def require_role(*allowed: str):
    """Dependency factory: 403s any role not in `allowed` (least privilege, FR-W9)."""

    def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role '{user.role}' cannot access this endpoint "
                       f"(requires one of {allowed})",
            )
        return user

    return _check
