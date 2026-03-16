"""Container entrypoint that waits for Postgres and runs Alembic safely."""

from __future__ import annotations

import os
import sys

from app.core.config import get_settings
from app.db.runtime_migrations import maybe_run_startup_migrations


def main() -> int:
    """Wait for DB, migrate, then launch the container command."""
    if len(sys.argv) < 2:
        raise SystemExit("No command provided to container entrypoint")
    if not os.environ.get("ACMF_DATABASE_URL"):
        raise SystemExit("ACMF_DATABASE_URL is required")
    maybe_run_startup_migrations(get_settings(), wait_for_db=True)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
