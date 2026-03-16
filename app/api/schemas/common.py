"""Shared API schema fragments for aCMF v1."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.core.enums import ReadMode, ScopeLevel


class RequestMetadata(BaseModel):
    """Metadata propagated from the host application."""

    app: str | None = None
    source: str | None = None
    model: str | None = None
    trace_id: str | None = None


class RetrievalBudgets(BaseModel):
    """Read budget controls."""

    max_output_tokens: int = Field(default=400, ge=1, le=4000)
    max_candidate_memories: int = Field(default=30, ge=1, le=200)


class ContainerHint(BaseModel):
    """Optional container metadata supplied by the host app."""

    id: str
    type: str | None = None


class RetrievalDiagnostics(BaseModel):
    """Diagnostics returned from context and deep-memory requests."""

    scope_applied: ScopeLevel
    read_mode: ReadMode
    user_found: bool
    candidate_count: int
    used_memory_count: int
    missing_containers: list[str] = Field(default_factory=list)
    source_breakdown: dict[str, int] = Field(default_factory=dict)
    evidence_strength: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class ScopeReadRequest(BaseModel):
    """Common request fields for scoped memory reads."""

    user_id: str
    containers: list[ContainerHint] = Field(default_factory=list)
    scope_level: ScopeLevel = ScopeLevel.USER_GLOBAL
    read_mode: ReadMode = ReadMode.BALANCED
    budgets: RetrievalBudgets = Field(default_factory=RetrievalBudgets)
    metadata: RequestMetadata = Field(default_factory=RequestMetadata)


class TurnEnvelope(BaseModel):
    """Turn payload used by process requests."""

    user_message: str
    assistant_response: str
    occurred_at: datetime | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None
