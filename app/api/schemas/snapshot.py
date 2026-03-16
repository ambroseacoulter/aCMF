"""Schemas for `/v1/snapshot`."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SnapshotPayload(BaseModel):
    """Snapshot contents."""

    summary: str
    memory_refs: list[str] = Field(default_factory=list)
    health_note: str | None = None


class SnapshotResponse(BaseModel):
    """Snapshot endpoint response."""

    status: str
    user_id: str
    generated_at: datetime | None = None
    snapshot: SnapshotPayload | None = None
