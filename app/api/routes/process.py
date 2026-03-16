"""Route for `/v1/process`."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import build_process_engine
from app.api.schemas.process import ProcessRequest, ProcessResponse
from app.db.session import get_session
from app.engines.process_engine import ProcessEngine

router = APIRouter(tags=["process"])


def get_engine(session: Session = Depends(get_session)) -> ProcessEngine:
    """Build the process engine for the request."""
    return build_process_engine(session)


@router.post("/v1/process", response_model=ProcessResponse)
def process_turn(
    request: ProcessRequest,
    engine: ProcessEngine = Depends(get_engine),
    session: Session = Depends(get_session),
) -> ProcessResponse:
    """Queue post-turn processing."""
    response = engine.enqueue_turn(request)
    session.commit()
    return response
