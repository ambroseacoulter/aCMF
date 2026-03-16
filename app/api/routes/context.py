"""Route for `/v1/context`."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import build_context_engine
from app.api.schemas.context import ContextRequest, ContextResponse
from app.db.session import get_session
from app.engines.context_engine import ContextEngine

router = APIRouter(tags=["context"])


def get_engine(session: Session = Depends(get_session)) -> ContextEngine:
    """Build the context engine for the request."""
    return build_context_engine(session)


@router.post("/v1/context", response_model=ContextResponse)
def create_context(
    request: ContextRequest,
    engine: ContextEngine = Depends(get_engine),
    session: Session = Depends(get_session),
) -> ContextResponse:
    """Synthesize pre-turn context."""
    response = engine.build_context(request)
    session.commit()
    return response
