"""B14 knowledge-base CRUD (FR-W7), read by kiosk B7 via the shared JSON file too."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.auth import CurrentUser, require_role
from server.db import get_db
from server.kb import load_clip_library, load_vocab_index, sync_kiosk_kb_file, unrenderable_words
from server.models import KBEntry
from server.schemas import KBEntryIn, KBEntryOut

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

EDIT_ROLES = ("engineer", "admin")
READ_ROLES = ("operator", "reviewer", "engineer", "admin")


def _to_out(
    entry: KBEntry,
    vocab_index: dict[str, str] | None = None,
    clip_library: dict | None = None,
) -> KBEntryOut:
    return KBEntryOut(
        id=entry.id, site_id=entry.site_id, keywords=entry.keywords,
        answer_ar=entry.answer_ar, answer_en=entry.answer_en,
        updated_at=entry.updated_at.isoformat(),
        unrenderable_words=unrenderable_words(
            entry.answer_ar, entry.answer_en, entry.keywords, vocab_index, clip_library,
        ),
    )


def _resync_file(db: Session) -> None:
    entries = list(db.scalars(select(KBEntry)))
    sync_kiosk_kb_file([
        {
            "id": e.id, "site_id": e.site_id, "keywords": e.keywords,
            "answer_ar": e.answer_ar, "answer_en": e.answer_en,
            "updated_at": e.updated_at.isoformat(),
        }
        for e in entries
    ])


@router.get("", response_model=list[KBEntryOut])
def list_entries(
    site_id: str | None = None,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*READ_ROLES)),
) -> list[KBEntryOut]:
    stmt = select(KBEntry)
    if site_id:
        stmt = stmt.where(KBEntry.site_id == site_id)
    vocab_index = load_vocab_index()
    clip_library = load_clip_library()
    return [_to_out(e, vocab_index, clip_library) for e in db.scalars(stmt)]


@router.get("/clip-library")
def get_clip_library(
    _: CurrentUser = Depends(require_role(*READ_ROLES)),
) -> dict:
    """Read-only, for the console's signed-rendering preview (FR-W7). Registered
    before POST/PUT/DELETE on `/{entry_id}` purely for file organisation — the
    fixed path `/clip-library` never collides with a path parameter on GET since
    no other GET takes one.
    """
    return load_clip_library()


@router.post("", response_model=KBEntryOut, status_code=201)
def create_entry(
    payload: KBEntryIn,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*EDIT_ROLES)),
) -> KBEntryOut:
    entry = KBEntry(**payload.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    _resync_file(db)
    return _to_out(entry)


@router.put("/{entry_id}", response_model=KBEntryOut)
def update_entry(
    entry_id: str,
    payload: KBEntryIn,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*EDIT_ROLES)),
) -> KBEntryOut:
    entry = db.get(KBEntry, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="kb entry not found")
    for field, value in payload.model_dump().items():
        setattr(entry, field, value)
    db.commit()
    db.refresh(entry)
    _resync_file(db)
    return _to_out(entry)


@router.delete("/{entry_id}", status_code=204)
def delete_entry(
    entry_id: str,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*EDIT_ROLES)),
) -> None:
    entry = db.get(KBEntry, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="kb entry not found")
    db.delete(entry)
    db.commit()
    _resync_file(db)
