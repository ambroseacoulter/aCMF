"""Route for `/v1/deep-memory`."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import build_context_engine
from app.api.schemas.deep_memory import DeepMemoryRequest, DeepMemoryResponse
from app.db.session import get_session
from app.engines.context_engine import ContextEngine

router = APIRouter(tags=["deep-memory"])


def get_engine(session: Session = Depends(get_session)) -> ContextEngine:
    """Build the context engine for the request."""
    return build_context_engine(session)


@router.post("/v1/deep-memory", response_model=DeepMemoryResponse)
def deep_memory(
    request: DeepMemoryRequest,
    engine: ContextEngine = Depends(get_engine),
    session: Session = Depends(get_session),
) -> DeepMemoryResponse:
    """Answer a focused deep-memory query."""
    response = engine.answer_deep_memory(request)
    session.commit()
    return response
