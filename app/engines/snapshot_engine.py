"""Snapshot orchestration."""

from __future__ import annotations

from app.api.schemas.snapshot import SnapshotPayload, SnapshotResponse
from app.storage.snapshot_repo import SnapshotRepository
from app.storage.user_repo import UserRepository


class SnapshotEngine:
    """Read hourly snapshots."""

    def __init__(self, user_repo: UserRepository, snapshot_repo: SnapshotRepository) -> None:
        self.user_repo = user_repo
        self.snapshot_repo = snapshot_repo

    def get_latest_snapshot(self, user_id: str) -> SnapshotResponse:
        """Return the latest snapshot payload for a user."""
        user = self.user_repo.get(user_id)
        if user is None:
            return SnapshotResponse(status="not_found", user_id=user_id)
        snapshot = self.snapshot_repo.get_latest(user_id)
        if snapshot is None:
            return SnapshotResponse(status="not_found", user_id=user_id)
        return SnapshotResponse(
            status="ok",
            user_id=user_id,
            generated_at=snapshot.generated_at,
            snapshot=SnapshotPayload(summary=snapshot.summary, memory_refs=snapshot.memory_refs, health_note=snapshot.health_note),
        )
