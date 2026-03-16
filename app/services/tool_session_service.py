"""Shared staged-operation framework for tool-driven LLM workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from app.core.enums import ScopeType


class StagedMemoryCreate(BaseModel):
    """Staged memory creation payload."""

    memory_type: str
    content: str
    summary: str | None = None
    rationale: str | None = None
    evidence: list[str] = Field(default_factory=list)
    importance_score: float = 0.5
    confidence_score: float = 0.5
    novelty_score: float = 0.5
    initial_relevance_score: float = 0.5
    contradiction_risk: float = 0.0
    target_scopes: list[ScopeType] = Field(default_factory=list)
    bucket_id: str | None = None


class StagedMemoryUpdate(BaseModel):
    """Staged memory update payload."""

    existing_memory_id: str
    memory_type: str
    content: str
    summary: str | None = None
    rationale: str | None = None
    evidence: list[str] = Field(default_factory=list)
    importance_score: float = 0.5
    confidence_score: float = 0.5
    novelty_score: float = 0.5
    initial_relevance_score: float = 0.5
    contradiction_risk: float = 0.0


class StagedMemoryMerge(BaseModel):
    """Staged memory merge payload."""

    existing_memory_id: str
    memory_type: str
    content: str
    summary: str | None = None
    rationale: str | None = None
    evidence: list[str] = Field(default_factory=list)
    importance_score: float = 0.5
    confidence_score: float = 0.5
    novelty_score: float = 0.5
    initial_relevance_score: float = 0.5
    contradiction_risk: float = 0.0
    topic: str | None = None


class StagedContradictionUpdate(BaseModel):
    """Staged contradiction update payload."""

    existing_memory_id: str
    topic: str
    description: str | None = None
    secondary_memory_id: str | None = None
    new_memory: StagedMemoryCreate | None = None


class StagedGraphEntity(BaseModel):
    """Staged graph entity payload."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    entity_type: str = Field(validation_alias=AliasChoices("entity_type", "type", "kind"))
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    linked_memory_refs: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("linked_memory_refs", "linked_memories", "memory_refs"),
    )

    @model_validator(mode="before")
    @classmethod
    def _lift_nested_type(cls, value: object) -> object:
        """Accept provider outputs that bury entity type inside attributes."""
        if not isinstance(value, dict):
            return value
        if value.get("entity_type") or value.get("type") or value.get("kind"):
            return value
        attributes = value.get("attributes")
        if isinstance(attributes, dict):
            nested_type = attributes.get("entity_type") or attributes.get("type") or attributes.get("kind")
            if nested_type:
                return {**value, "entity_type": nested_type}
        return value


class StagedGraphRelation(BaseModel):
    """Staged graph relation payload."""

    from_entity_name: str
    to_entity_name: str
    relation_type: str
    confidence_score: float = 0.5
    evidence: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class StagedMemoryEntityLink(BaseModel):
    """Staged memory-entity link payload."""

    memory_ref: str
    entity_name: str
    link_type: str = "MENTIONS"


class StagedStatusUpdate(BaseModel):
    """Staged memory status update payload."""

    memory_id: str
    status: str
    archived_reason: str | None = None


class StagedLineageEvent(BaseModel):
    """Staged lineage payload."""

    source_memory_id: str
    target_memory_id: str
    event_type: str
    confidence_score: float = 0.5
    rationale: str | None = None


class SnapshotCandidateSet(BaseModel):
    """Snapshot selection override payload."""

    selected_memory_ids: list[str] = Field(default_factory=list)


class MaintenanceProposal(BaseModel):
    """Programmatic maintenance proposal."""

    proposal_id: str
    proposal_type: str
    memory_ids: list[str] = Field(default_factory=list)
    status: str | None = None
    archived_reason: str | None = None
    confidence_score: float = 0.5
    rationale: str | None = None
    topic: str | None = None
    target_memory_id: str | None = None


class MaintenanceProposalBundle(BaseModel):
    """Per-user maintenance proposal bundle."""

    user_id: str
    proposals: list[MaintenanceProposal] = Field(default_factory=list)
    snapshot_candidates: list[str] = Field(default_factory=list)
    analysis_notes: list[str] = Field(default_factory=list)


class StagedOperation(BaseModel):
    """One staged operation produced by an LLM tool session."""

    operation_id: str
    operation_type: str
    payload: dict[str, Any]


@dataclass
class ToolSession:
    """Session-local staged operation ledger."""

    session_id: str
    kind: str
    user_id: str
    operations: list[StagedOperation] = field(default_factory=list)
    finalized: bool = False
    reasoning_summary: str = ""

    def stage(self, operation_type: str, payload: BaseModel | dict[str, Any]) -> dict[str, str]:
        """Stage one validated operation."""
        op_id = "{0}_{1}".format(self.kind, len(self.operations) + 1)
        data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
        self.operations.append(StagedOperation(operation_id=op_id, operation_type=operation_type, payload=data))
        return {"operation_id": op_id, "operation_type": operation_type}

    def finalize(self, reasoning_summary: str) -> dict[str, Any]:
        """Mark the tool session as finalized."""
        self.finalized = True
        self.reasoning_summary = reasoning_summary
        return {"finalized": True, "operation_count": len(self.operations)}

    def get(self, operation_id: str) -> StagedOperation | None:
        """Return a staged operation by ID."""
        for operation in self.operations:
            if operation.operation_id == operation_id:
                return operation
        return None


class ToolSessionValidator:
    """Validate staged operations before commit."""

    def validate_adjudication(self, session: ToolSession, allowed_scopes: list[str], existing_memory_ids: set[str]) -> list[str]:
        """Validate an adjudication tool session."""
        errors: list[str] = []
        if not session.finalized:
            errors.append("tool session was not finalized")
        seen_exact_mutations: set[str] = set()
        seen_updates: set[str] = set()
        for operation in session.operations:
            mutation_key = "{0}:{1}".format(operation.operation_type, json.dumps(operation.payload, sort_keys=True))
            if mutation_key in seen_exact_mutations:
                errors.append("duplicate staged operation: {0}".format(operation.operation_id))
            seen_exact_mutations.add(mutation_key)
            if operation.operation_type == "create_memory":
                scopes = operation.payload.get("target_scopes", [])
                if not scopes:
                    errors.append("create_memory missing target scopes")
                if any(scope not in allowed_scopes for scope in scopes):
                    errors.append("create_memory used illegal scope")
            elif operation.operation_type in {"update_memory", "merge_memory"}:
                existing_memory_id = operation.payload.get("existing_memory_id")
                if existing_memory_id not in existing_memory_ids:
                    errors.append("{0} references unknown memory".format(operation.operation_type))
                if existing_memory_id in seen_updates:
                    errors.append("multiple staged mutations for memory {0}".format(existing_memory_id))
                seen_updates.add(existing_memory_id)
            elif operation.operation_type == "mark_contradiction":
                existing_memory_id = operation.payload.get("existing_memory_id")
                secondary_memory_id = operation.payload.get("secondary_memory_id")
                new_memory = operation.payload.get("new_memory")
                if existing_memory_id not in existing_memory_ids:
                    errors.append("mark_contradiction references unknown memory")
                if not secondary_memory_id and not new_memory:
                    errors.append("contradiction update must reference a second memory or new memory")
            elif operation.operation_type == "link_memory_entity":
                memory_ref = operation.payload.get("memory_ref")
                if memory_ref not in existing_memory_ids and not memory_ref.startswith("adjudication_"):
                    errors.append("link_memory_entity references unknown memory")
        return errors

    def validate_cortex_review(self, session: ToolSession, proposal_ids: set[str], existing_memory_ids: set[str]) -> list[str]:
        """Validate a cortex review tool session."""
        errors: list[str] = []
        if not session.finalized:
            errors.append("tool session was not finalized")
        for operation in session.operations:
            if operation.operation_type == "status_update":
                if operation.payload.get("memory_id") not in existing_memory_ids:
                    errors.append("status update references unknown memory")
            elif operation.operation_type == "lineage_event":
                if operation.payload.get("source_memory_id") not in existing_memory_ids or operation.payload.get("target_memory_id") not in existing_memory_ids:
                    errors.append("lineage event references unknown memory")
            elif operation.operation_type == "contradiction_update":
                if operation.payload.get("existing_memory_id") not in existing_memory_ids:
                    errors.append("contradiction update references unknown memory")
                secondary_memory_id = operation.payload.get("secondary_memory_id")
                if secondary_memory_id and secondary_memory_id not in existing_memory_ids:
                    errors.append("contradiction update references unknown secondary memory")
            elif operation.operation_type == "snapshot_selection_override":
                selected_ids = operation.payload.get("selected_memory_ids", [])
                if any(memory_id not in existing_memory_ids for memory_id in selected_ids):
                    errors.append("snapshot override references unknown memory")
        return errors
