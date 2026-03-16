"""Runtime Alembic migration helpers for service startup."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

import psycopg

from app.core.config import Settings

logger = logging.getLogger(__name__)

MIGRATION_LOCK_ID = 84214537
_migration_once_lock = threading.Lock()
_migration_completed = False


def normalize_database_url(url: str) -> str:
    """Convert a SQLAlchemy psycopg URL into a psycopg connection URL."""
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql://", 1)
    return url


def wait_for_database(database_url: str, timeout_seconds: float) -> None:
    """Wait until Postgres is reachable."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            with psycopg.connect(database_url):
                return
        except psycopg.Error:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for Postgres")
            time.sleep(1.0)


def run_migrations(database_url: str) -> None:
    """Run Alembic under a Postgres advisory lock."""
    with psycopg.connect(database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
        try:
            subprocess.run(["alembic", "upgrade", "head"], check=True)
        finally:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,))


def maybe_run_startup_migrations(settings: Settings, *, wait_for_db: bool) -> None:
    """Run migrations once per process unless startup auto-migration is disabled."""
    global _migration_completed
    if settings.env == "test" or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if not settings.auto_migrate_on_startup:
        logger.info("Startup migrations disabled.")
        return
    with _migration_once_lock:
        if _migration_completed:
            return
        database_url = normalize_database_url(settings.database_url)
        if wait_for_db:
            wait_for_database(database_url, settings.db_startup_timeout_seconds)
        logger.info("Running startup migrations.")
        run_migrations(database_url)
        _migration_completed = True
