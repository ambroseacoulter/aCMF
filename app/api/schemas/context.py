"""Schemas for `/v1/context`."""

from __future__ import annotations

from pydantic import BaseModel

from app.api.schemas.common import RetrievalDiagnostics, ScopeReadRequest


class ContextRequest(ScopeReadRequest):
    """Pre-turn context synthesis request."""

    message: str


class ContextResponse(BaseModel):
    """Context synthesis response."""

    status: str
    has_usable_context: bool
    context_enhancement: str
    abstained_reason: str | None = None
    diagnostics: RetrievalDiagnostics
