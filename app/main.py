"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.context import router as context_router
from app.api.routes.deep_memory import router as deep_memory_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.process import router as process_router
from app.api.routes.snapshot import router as snapshot_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.runtime_migrations import maybe_run_startup_migrations

settings = get_settings()
configure_logging(settings.debug)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run startup migrations before serving requests."""
    maybe_run_startup_migrations(settings, wait_for_db=False)
    yield


app = FastAPI(title="aCMF", version="0.1.0", lifespan=lifespan)
app.include_router(context_router)
app.include_router(process_router)
app.include_router(jobs_router)
app.include_router(snapshot_router)
app.include_router(deep_memory_router)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Return a simple health payload."""
    return {"status": "ok"}
