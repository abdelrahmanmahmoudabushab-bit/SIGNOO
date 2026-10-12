"""B14 knowledge-base CRUD (FR-W7), read by kiosk B7 via the shared JSON file too."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.auth import CurrentUser, require_role
from server.db import get_db
from server.kb import (
    kb_path_for, load_clip_library, load_vocab_index, sync_kiosk_kb_file, unrenderable_words,
)
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
        id=entry.id, language=entry.language, site_id=entry.site_id, keywords=entry.keywords,
        answer_ar=entry.answer_ar, answer_en=entry.answer_en,
        updated_at=entry.updated_at.isoformat(),
        unrenderable_words=unrenderable_words(
            entry.answer_ar, entry.answer_en, entry.keywords, vocab_index, clip_library,
            language=entry.language,
        ),
    )


def _resync_file(db: Session, language: str) -> None:
    """Rewrites one language's knowledge-base file from its own rows, and only its own."""
    entries = list(db.scalars(select(KBEntry).where(KBEntry.language == language)))
    sync_kiosk_kb_file([
        {
            "id": e.id, "site_id": e.site_id, "keywords": e.keywords,
            "answer_ar": e.answer_ar, "answer_en": e.answer_en,
            "updated_at": e.updated_at.isoformat(),
        }
        for e in entries
    ], language)


def _require_kb_file(language: str) -> None:
    """Checked before anything is saved: an entry for a language with nowhere to publish
    it would sit in the database and never reach a kiosk."""
    if kb_path_for(language) is None:
        raise HTTPException(
            status_code=422,
            detail=f"language '{language}' has no knowledge_base file in core/languages.json",
        )


@router.get("", response_model=list[KBEntryOut])
def list_entries(
    site_id: str | None = None,
    language: str | None = None,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*READ_ROLES)),
) -> list[KBEntryOut]:
    stmt = select(KBEntry)
    if site_id:
        stmt = stmt.where(KBEntry.site_id == site_id)
    if language:
        stmt = stmt.where(KBEntry.language == language)
    # Each language is checked against its own vocabulary and clips; load each once.
    vocab_indexes: dict[str, dict] = {}
    clip_libraries: dict[str, dict] = {}
    out = []
    for e in db.scalars(stmt):
        if e.language not in vocab_indexes:
            vocab_indexes[e.language] = load_vocab_index(e.language)
            clip_libraries[e.language] = load_clip_library(e.language)
        out.append(_to_out(e, vocab_indexes[e.language], clip_libraries[e.language]))
    return out


@router.get("/clip-library")
def get_clip_library(
    language: str = "liu",
    _: CurrentUser = Depends(require_role(*READ_ROLES)),
) -> dict:
    """Read-only, for the console's signed-rendering preview (FR-W7). Registered
    before POST/PUT/DELETE on `/{entry_id}` purely for file organisation — the
    fixed path `/clip-library` never collides with a path parameter on GET since
    no other GET takes one.
    """
    return load_clip_library(language)


@router.post("", response_model=KBEntryOut, status_code=201)
def create_entry(
    payload: KBEntryIn,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*EDIT_ROLES)),
) -> KBEntryOut:
    _require_kb_file(payload.language)
    entry = KBEntry(**payload.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    _resync_file(db, entry.language)
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
    _require_kb_file(payload.language)
    previous_language = entry.language
    for field, value in payload.model_dump().items():
        setattr(entry, field, value)
    db.commit()
    db.refresh(entry)
    _resync_file(db, entry.language)
    if previous_language != entry.language:
        _resync_file(db, previous_language)   # the answer has left that language's file
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
    language = entry.language
    db.delete(entry)
    db.commit()
    _resync_file(db, language)
