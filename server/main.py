"""SIGNOO server: FastAPI app wiring B11-B16.

Run with:  uvicorn server.main:app --reload
See server/README.md for full setup (Docker Compose Postgres, migrations, seeding).
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.config import settings
from server.db import init_db
from server.routers import auth, fleet, ingest, kb, registry, review, training
from server.scheduler import scheduler_loop

# Windows console defaults to cp1252; Arabic KB content and logs would otherwise
# crash on print(). Reconfigure stdout/stderr to UTF-8 unconditionally.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Dev-simulation convenience only; a real deployment migrates with Alembic
    # (server/alembic) instead of create_all — see server/README.md.
    init_db()
    task = asyncio.create_task(scheduler_loop()) if settings.auto_train_enabled else None
    try:
        yield
    finally:
        if task is not None:
            task.cancel()


app = FastAPI(
    title="SIGNOO server",
    description="B11-B16: ingest/review, training, registry+gate, knowledge base, "
                "fleet control, console API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(review.router)
app.include_router(training.router)
app.include_router(registry.router)
app.include_router(kb.router)
app.include_router(fleet.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
