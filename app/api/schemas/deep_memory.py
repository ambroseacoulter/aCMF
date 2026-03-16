"""Schemas for `/v1/deep-memory`."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.api.schemas.common import RetrievalBudgets, RetrievalDiagnostics, ScopeReadRequest


class DeepMemoryRequest(ScopeReadRequest):
    """Focused memory query request."""

    query: str
    budgets: RetrievalBudgets = Field(
        default_factory=lambda: RetrievalBudgets(max_output_tokens=500, max_candidate_memories=40)
    )


class EvidenceItem(BaseModel):
    """Grounding evidence for a deep-memory answer."""

    memory_id: str
    scope_type: str
    bucket_id: str | None = None
    relevance: float
    support: float
    relation_refs: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)


class DeepMemoryResponse(BaseModel):
    """Deep memory answer payload."""

    status: str
    answer: str
    abstained: bool
    abstained_reason: str | None = None
    used_memory_count: int
    diagnostics: RetrievalDiagnostics
    evidence: list[EvidenceItem]
