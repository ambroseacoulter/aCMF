"""Snapshot persistence operations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.time import utc_now
from app.db.models import Snapshot


class SnapshotRepository:
    """Repository for hourly user snapshots."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_latest(self, user_id: str) -> Snapshot | None:
        """Return the latest snapshot for a user."""
        stmt = select(Snapshot).where(Snapshot.user_id == user_id)
        return self.session.scalar(stmt)

    def upsert_latest(self, user_id: str, summary: str, memory_refs: list[str], health_note: str | None = None, version: int = 1) -> Snapshot:
        """Overwrite the latest snapshot for the user."""
        snapshot = self.get_latest(user_id)
        if snapshot is None:
            snapshot = Snapshot(
                user_id=user_id,
                summary=summary,
                memory_refs=memory_refs,
                health_note=health_note,
                version=version,
            )
            self.session.add(snapshot)
        else:
            snapshot.summary = summary
            snapshot.memory_refs = memory_refs
            snapshot.health_note = health_note
            snapshot.generated_at = utc_now()
            snapshot.version = version
        self.session.flush()
        return snapshot
