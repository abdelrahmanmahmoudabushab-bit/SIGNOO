from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.auth import ROLES, create_access_token, hash_password, require_role, verify_password
from server.db import get_db
from server.models import User
from server.schemas import CreateUserIn, LoginIn, TokenOut, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

# Verified against when the email is unknown, so a miss costs the same bcrypt work as
# a wrong password and response time does not reveal which emails have accounts.
_DUMMY_HASH = hash_password("no-such-user")


@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, db: Session = Depends(get_db)) -> TokenOut:
    user = db.query(User).filter(User.email == payload.email).one_or_none()
    password_ok = verify_password(payload.password, user.hashed_password if user else _DUMMY_HASH)
    if user is None or not password_ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad credentials")
    token = create_access_token(subject=user.id, role=user.role)
    return TokenOut(access_token=token, role=user.role)


@router.post(
    "/users", response_model=UserOut, status_code=201,
    dependencies=[Depends(require_role("admin"))],
)
def create_user(payload: CreateUserIn, db: Session = Depends(get_db)) -> UserOut:
    """Admin-only user provisioning. First-run bootstrapping (creating the very
    first admin, before any admin token exists) is done by `server/seed.py`, not
    this endpoint — see server/README.md.

    Takes a JSON body, not query params: a query-string password would land in
    plaintext in server/proxy access logs.
    """
    if payload.role not in ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {ROLES}")
    if db.query(User).filter(User.email == payload.email).one_or_none() is not None:
        raise HTTPException(status_code=409, detail="user already exists")

    user = User(
        email=payload.email, hashed_password=hash_password(payload.password), role=payload.role,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # The pre-check above is a fast common-case path, not the source of truth:
        # two concurrent requests for the same email can both pass it before either
        # commits. The unique constraint on users.email is what actually enforces
        # this, so a violation here means the same race, not a server bug.
        db.rollback()
        raise HTTPException(status_code=409, detail="user already exists")
    db.refresh(user)
    return UserOut(id=user.id, email=user.email, role=user.role)
