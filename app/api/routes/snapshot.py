"""Route for `/v1/snapshot/{user_id}`."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import build_snapshot_engine
from app.api.schemas.snapshot import SnapshotResponse
from app.db.session import get_session
from app.engines.snapshot_engine import SnapshotEngine

router = APIRouter(tags=["snapshot"])


def get_engine(session: Session = Depends(get_session)) -> SnapshotEngine:
    """Build the snapshot engine for the request."""
    return build_snapshot_engine(session)


@router.get("/v1/snapshot/{user_id}", response_model=SnapshotResponse)
def get_snapshot(
    user_id: str,
    engine: SnapshotEngine = Depends(get_engine),
    session: Session = Depends(get_session),
) -> SnapshotResponse:
    """Return the latest user snapshot."""
    response = engine.get_latest_snapshot(user_id)
    session.commit()
    return response
