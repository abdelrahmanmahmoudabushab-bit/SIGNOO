"""SIGNOO server: FastAPI app wiring B11-B16.

Run with:  uvicorn server.main:app --reload
See server/README.md for full setup (Docker Compose Postgres, migrations, seeding).
"""
from __future__ import annotations

import logging
import sys

from fastapi import FastAPI

from server.db import init_db
from server.routers import auth, fleet, ingest, kb, registry, review, training

# Windows console defaults to cp1252; Arabic KB content and logs would otherwise
# crash on print(). Reconfigure stdout/stderr to UTF-8 unconditionally.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="SIGNOO server",
    description="B11-B16: ingest/review, training, registry+gate, knowledge base, "
                "fleet control, console API.",
    version="0.1.0",
)

app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(review.router)
app.include_router(training.router)
app.include_router(registry.router)
app.include_router(kb.router)
app.include_router(fleet.router)


@app.on_event("startup")
def _on_startup() -> None:
    # Dev-simulation convenience only; a real deployment migrates with Alembic
    # (server/alembic) instead of create_all — see server/README.md.
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
