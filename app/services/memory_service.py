"""Memory domain services."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.enums import (
    ContradictionRole,
    GraphEdgeType,
    GraphNodeType,
    LineageEventType,
    MemoryStatus,
    ScopeType,
    SourceType,
)
from app.db.models import Memory
from app.services.embedding_service import EmbeddingService
from app.services.relevance_service import RelevanceService
from app.services.tool_session_service import (
    MaintenanceProposalBundle,
    StagedContradictionUpdate,
    StagedLineageEvent,
    StagedMemoryCreate,
    StagedMemoryMerge,
    StagedMemoryUpdate,
    StagedOperation,
    StagedStatusUpdate,
    ToolSession,
)
from app.storage.memory_repo import MemoryRepository


@dataclass
class AdjudicationCommitResult:
    """Result of applying an adjudication tool session."""

    created_memories: list[Memory]
    touched_memories: list[Memory]
    contradiction_topics: list[str]
    graph_entities: list[dict]
    graph_relations: list[dict]
    memory_links: list[tuple[str, str, str]]
    graph_edges: list[dict]


@dataclass
class CortexCommitResult:
    """Result of applying a Cortex review tool session."""

    touched_memories: list[Memory]
    snapshot_memory_ids: list[str]
    applied_proposal_ids: list[str]
    graph_edges: list[dict]


class MemoryService:
    """Service for creating, updating, and touching memories."""

    def __init__(
        self,
        memory_repo: MemoryRepository,
        embedding_service: EmbeddingService,
        relevance_service: RelevanceService,
        embedding_model: str,
    ) -> None:
        self.memory_repo = memory_repo
        self.embedding_service = embedding_service
        self.relevance_service = relevance_service
        self.embedding_model = embedding_model

    def apply_adjudication_session(
        self,
        *,
        user_id: str,
        turn_record_id: str,
        tool_session: ToolSession,
        default_container_id: str | None,
    ) -> AdjudicationCommitResult:
        """Apply a validated adjudication session deterministically."""
        created: list[Memory] = []
        touched: list[Memory] = []
        contradiction_topics: list[str] = []
        graph_entities: list[dict] = []
        graph_relations: list[dict] = []
        memory_links: list[tuple[str, str, str]] = []
        graph_edges: list[dict] = []
        staged_memory_refs: dict[str, list[Memory]] = {}

        for operation in tool_session.operations:
            if operation.operation_type == "create_memory":
                payload = StagedMemoryCreate.model_validate(operation.payload)
                new_memories = self._create_staged_memory(
                    user_id=user_id,
                    turn_record_id=turn_record_id,
                    payload=payload,
                    default_container_id=default_container_id,
                )
                staged_memory_refs[operation.operation_id] = new_memories
                created.extend(new_memories)
                touched.extend(new_memories)
            elif operation.operation_type == "update_memory":
                payload = StagedMemoryUpdate.model_validate(operation.payload)
                memory = self.memory_repo.get(payload.existing_memory_id)
                if memory is None:
                    continue
                updated = self.memory_repo.update_memory(
                    memory,
                    self._memory_payload(
                        turn_record_id=turn_record_id,
                        memory_type=payload.memory_type,
                        content=payload.content,
                        summary=payload.summary,
                        rationale=payload.rationale,
                        evidence=payload.evidence,
                        importance_score=payload.importance_score,
                        confidence_score=payload.confidence_score,
                        novelty_score=payload.novelty_score,
                        initial_relevance_score=payload.initial_relevance_score,
                        contradiction_risk=payload.contradiction_risk,
                        scope_type=memory.scope_type,
                        bucket_id=memory.bucket_id,
                    ),
                )
                self._upsert_memory_embedding(updated)
                touched.append(updated)
                staged_memory_refs[operation.operation_id] = [updated]
            elif operation.operation_type == "merge_memory":
                payload = StagedMemoryMerge.model_validate(operation.payload)
                merged = self._apply_merge(user_id=user_id, turn_record_id=turn_record_id, payload=payload)
                created.extend(merged.created_memories)
                touched.extend(merged.touched_memories)
                contradiction_topics.extend(merged.contradiction_topics)
                graph_edges.extend(merged.graph_edges)
                if merged.created_memories:
                    staged_memory_refs[operation.operation_id] = merged.created_memories
                elif merged.touched_memories:
                    staged_memory_refs[operation.operation_id] = merged.touched_memories
            elif operation.operation_type == "mark_contradiction":
                payload = StagedContradictionUpdate.model_validate(operation.payload)
                contradicted = self._apply_contradiction(
                    user_id=user_id,
                    turn_record_id=turn_record_id,
                    payload=payload,
                    default_container_id=default_container_id,
                )
                created.extend(contradicted.created_memories)
                touched.extend(contradicted.touched_memories)
                contradiction_topics.extend(contradicted.contradiction_topics)
                graph_edges.extend(contradicted.graph_edges)
                if contradicted.created_memories:
                    staged_memory_refs[operation.operation_id] = contradicted.created_memories
            elif operation.operation_type == "create_entity":
                graph_entities.append(dict(operation.payload))
            elif operation.operation_type == "create_relation":
                graph_relations.append(dict(operation.payload))
            elif operation.operation_type == "link_memory_entity":
                memory_id = self._resolve_memory_ref(str(operation.payload["memory_ref"]), staged_memory_refs)
                if memory_id is None:
                    continue
                memory_links.append((memory_id, str(operation.payload["entity_name"]), str(operation.payload.get("link_type", "MENTIONS"))))

        return AdjudicationCommitResult(
            created_memories=created,
            touched_memories=touched,
            contradiction_topics=sorted(set(contradiction_topics)),
            graph_entities=graph_entities,
            graph_relations=graph_relations,
            memory_links=memory_links,
            graph_edges=graph_edges,
        )

    def apply_cortex_review(
        self,
        *,
        user_id: str,
        tool_session: ToolSession,
        bundle: MaintenanceProposalBundle,
    ) -> CortexCommitResult:
        """Apply a validated Cortex review session deterministically."""
        touched: list[Memory] = []
        snapshot_memory_ids = list(bundle.snapshot_candidates)
        applied_proposal_ids: list[str] = []
        graph_edges: list[dict] = []

        for operation in tool_session.operations:
            if operation.operation_type == "status_update":
                payload = StagedStatusUpdate.model_validate(operation.payload)
                memory = self.memory_repo.get(payload.memory_id)
                if memory is None:
                    continue
                memory.status = payload.status
                memory.archived_reason = payload.archived_reason
                touched.append(memory)
                applied_proposal_ids.extend(self._proposal_ids_for_memories(bundle, [memory.id], "status_update"))
            elif operation.operation_type == "lineage_event":
                payload = StagedLineageEvent.model_validate(operation.payload)
                self.memory_repo.record_lineage_event(
                    user_id=user_id,
                    source_memory_id=payload.source_memory_id,
                    target_memory_id=payload.target_memory_id,
                    event_type=payload.event_type,
                    confidence_score=payload.confidence_score,
                    rationale=payload.rationale,
                    metadata_json={},
                )
                source_memory = self.memory_repo.get(payload.source_memory_id)
                if source_memory is not None:
                    if payload.event_type == LineageEventType.DUPLICATE_OF.value:
                        source_memory.status = MemoryStatus.DUPLICATE.value
                    else:
                        source_memory.status = MemoryStatus.SUPERSEDED.value
                    source_memory.superseded_by_memory_id = payload.target_memory_id
                    touched.append(source_memory)
                edge_type = (
                    GraphEdgeType.DUPLICATE_OF.value
                    if payload.event_type == LineageEventType.DUPLICATE_OF.value
                    else GraphEdgeType.SUPERSEDED_BY.value
                    if payload.event_type == LineageEventType.SUPERSEDED_BY.value
                    else GraphEdgeType.MERGED_INTO.value
                )
                graph_edges.append(
                    {
                        "from_node_type": GraphNodeType.MEMORY.value,
                        "from_node_id": payload.source_memory_id,
                        "to_node_type": GraphNodeType.MEMORY.value,
                        "to_node_id": payload.target_memory_id,
                        "edge_type": edge_type,
                        "confidence_score": payload.confidence_score,
                        "attributes": {},
                        "source_type": "lineage_event",
                        "source_ref": payload.source_memory_id,
                    }
                )
                applied_proposal_ids.extend(self._proposal_ids_for_memories(bundle, [payload.source_memory_id, payload.target_memory_id], None))
            elif operation.operation_type == "contradiction_update":
                payload = StagedContradictionUpdate.model_validate(operation.payload)
                topic = payload.topic
                group = self.memory_repo.find_or_create_contradiction_group(user_id, topic, payload.description)
                existing = self.memory_repo.get(payload.existing_memory_id)
                secondary = self.memory_repo.get(payload.secondary_memory_id) if payload.secondary_memory_id else None
                if existing is not None:
                    existing.status = MemoryStatus.CONFLICTED.value
                    self.memory_repo.add_contradiction_item(
                        group.id,
                        existing.id,
                        ContradictionRole.CLAIM.value,
                        existing.confidence_score,
                    )
                    touched.append(existing)
                if secondary is not None:
                    secondary.status = MemoryStatus.CONFLICTED.value
                    self.memory_repo.add_contradiction_item(
                        group.id,
                        secondary.id,
                        ContradictionRole.COUNTER_CLAIM.value,
                        secondary.confidence_score,
                    )
                    touched.append(secondary)
                    graph_edges.append(
                        {
                            "from_node_type": GraphNodeType.MEMORY.value,
                            "from_node_id": existing.id if existing is not None else payload.existing_memory_id,
                            "to_node_type": GraphNodeType.MEMORY.value,
                            "to_node_id": secondary.id,
                            "edge_type": GraphEdgeType.CONTRADICTS.value,
                            "confidence_score": max((existing.confidence_score if existing is not None else 0.5), secondary.confidence_score),
                            "attributes": {"topic": topic},
                            "source_type": "contradiction_group",
                            "source_ref": group.id,
                        }
                    )
                applied_proposal_ids.extend(self._proposal_ids_for_memories(bundle, [memory_id for memory_id in [payload.existing_memory_id, payload.secondary_memory_id] if memory_id], "contradiction_candidate"))
            elif operation.operation_type == "snapshot_selection_override":
                snapshot_memory_ids = list(operation.payload.get("selected_memory_ids", []))

        return CortexCommitResult(
            touched_memories=touched,
            snapshot_memory_ids=snapshot_memory_ids,
            applied_proposal_ids=sorted(set(applied_proposal_ids)),
            graph_edges=graph_edges,
        )

    def touch_memories(
        self,
        memories: list[Memory],
        query_similarity_map: dict[str, float],
        graph_density_signal: float = 0.0,
    ) -> list[Memory]:
        """Update memory relevance for used memories."""
        for memory in memories:
            self.relevance_service.touch_memory(
                memory,
                query_similarity=query_similarity_map.get(memory.id, 0.5),
                graph_density_signal=graph_density_signal,
                stale_penalty=0.1 if memory.status == MemoryStatus.STALE.value else 0.0,
                contradiction_penalty=memory.contradiction_risk * 0.2,
                superseded_penalty=0.15 if memory.superseded_by_memory_id else 0.0,
            )
        return memories

    def _create_staged_memory(
        self,
        *,
        user_id: str,
        turn_record_id: str,
        payload: StagedMemoryCreate,
        default_container_id: str | None,
    ) -> list[Memory]:
        """Create one memory per approved scope."""
        created: list[Memory] = []
        texts: list[str] = []
        db_payloads: list[dict] = []
        for scope in payload.target_scopes:
            bucket_id = payload.bucket_id
            if scope == ScopeType.CONTAINER and not bucket_id:
                bucket_id = default_container_id
            db_payload = self._memory_payload(
                turn_record_id=turn_record_id,
                memory_type=payload.memory_type,
                content=payload.content,
                summary=payload.summary,
                rationale=payload.rationale,
                evidence=payload.evidence,
                importance_score=payload.importance_score,
                confidence_score=payload.confidence_score,
                novelty_score=payload.novelty_score,
                initial_relevance_score=payload.initial_relevance_score,
                contradiction_risk=payload.contradiction_risk,
                scope_type=scope.value,
                bucket_id=bucket_id,
            )
            db_payload["user_id"] = user_id
            db_payloads.append(db_payload)
            texts.append(payload.content)
        embeddings = self.embedding_service.embed_texts(texts) if texts else []
        for db_payload, embedding in zip(db_payloads, embeddings):
            memory = self.memory_repo.create_memory(db_payload)
            self.memory_repo.upsert_embedding(memory.id, embedding, self.embedding_model)
            created.append(memory)
        return created

    def _apply_merge(self, *, user_id: str, turn_record_id: str, payload: StagedMemoryMerge) -> AdjudicationCommitResult:
        """Apply a merge into an existing memory."""
        created: list[Memory] = []
        touched: list[Memory] = []
        existing = self.memory_repo.get(payload.existing_memory_id)
        if existing is None:
            return AdjudicationCommitResult(created, touched, [], [], [], [], [])
        updated = self.memory_repo.update_memory(
            existing,
            self._memory_payload(
                turn_record_id=turn_record_id,
                memory_type=payload.memory_type,
                content=payload.content,
                summary=payload.summary,
                rationale=payload.rationale,
                evidence=payload.evidence,
                importance_score=payload.importance_score,
                confidence_score=payload.confidence_score,
                novelty_score=payload.novelty_score,
                initial_relevance_score=payload.initial_relevance_score,
                contradiction_risk=payload.contradiction_risk,
                scope_type=existing.scope_type,
                bucket_id=existing.bucket_id,
            ),
        )
        self._upsert_memory_embedding(updated)
        duplicate = self.memory_repo.create_memory(
            {
                **self._memory_payload(
                    turn_record_id=turn_record_id,
                    memory_type=payload.memory_type,
                    content=payload.content,
                    summary=payload.summary,
                    rationale=payload.rationale,
                    evidence=payload.evidence,
                    importance_score=payload.importance_score,
                    confidence_score=payload.confidence_score,
                    novelty_score=payload.novelty_score,
                    initial_relevance_score=payload.initial_relevance_score,
                    contradiction_risk=payload.contradiction_risk,
                    scope_type=existing.scope_type,
                    bucket_id=existing.bucket_id,
                ),
                "user_id": user_id,
                "status": MemoryStatus.DUPLICATE.value,
                "superseded_by_memory_id": existing.id,
            }
        )
        self._upsert_memory_embedding(duplicate)
        self.memory_repo.record_lineage_event(
            user_id=user_id,
            source_memory_id=duplicate.id,
            target_memory_id=existing.id,
            event_type=LineageEventType.MERGED_INTO.value,
            confidence_score=payload.confidence_score,
            rationale=payload.rationale,
            metadata_json={"topic": payload.topic},
        )
        created.append(duplicate)
        touched.append(updated)
        return AdjudicationCommitResult(
            created,
            touched,
            [],
            [],
            [],
            [],
            [
                {
                    "from_node_type": GraphNodeType.MEMORY.value,
                    "from_node_id": duplicate.id,
                    "to_node_type": GraphNodeType.MEMORY.value,
                    "to_node_id": existing.id,
                    "edge_type": GraphEdgeType.MERGED_INTO.value,
                    "confidence_score": payload.confidence_score,
                    "attributes": {"topic": payload.topic or "", "memory_type": payload.memory_type},
                    "source_type": "lineage_event",
                    "source_ref": duplicate.id,
                },
                {
                    "from_node_type": GraphNodeType.MEMORY.value,
                    "from_node_id": duplicate.id,
                    "to_node_type": GraphNodeType.MEMORY.value,
                    "to_node_id": existing.id,
                    "edge_type": GraphEdgeType.DUPLICATE_OF.value,
                    "confidence_score": payload.confidence_score,
                    "attributes": {"topic": payload.topic or "", "memory_type": payload.memory_type},
                    "source_type": "lineage_event",
                    "source_ref": duplicate.id,
                },
            ],
        )

    def _apply_contradiction(
        self,
        *,
        user_id: str,
        turn_record_id: str,
        payload: StagedContradictionUpdate,
        default_container_id: str | None,
    ) -> AdjudicationCommitResult:
        """Apply a contradiction update."""
        created: list[Memory] = []
        touched: list[Memory] = []
        contradiction_topics = [payload.topic]
        existing = self.memory_repo.get(payload.existing_memory_id)
        if existing is None:
            return AdjudicationCommitResult(created, touched, contradiction_topics, [], [], [], [])
        existing.status = MemoryStatus.CONFLICTED.value
        touched.append(existing)
        graph_edges: list[dict] = []
        group = self.memory_repo.find_or_create_contradiction_group(user_id, payload.topic, payload.description)
        self.memory_repo.add_contradiction_item(
            group.id,
            existing.id,
            ContradictionRole.CLAIM.value,
            existing.confidence_score,
        )
        if payload.secondary_memory_id:
            secondary = self.memory_repo.get(payload.secondary_memory_id)
            if secondary is not None:
                secondary.status = MemoryStatus.CONFLICTED.value
                self.memory_repo.add_contradiction_item(
                    group.id,
                    secondary.id,
                    ContradictionRole.COUNTER_CLAIM.value,
                    secondary.confidence_score,
                )
                touched.append(secondary)
                graph_edges.append(
                    {
                        "from_node_type": GraphNodeType.MEMORY.value,
                        "from_node_id": existing.id,
                        "to_node_type": GraphNodeType.MEMORY.value,
                        "to_node_id": secondary.id,
                        "edge_type": GraphEdgeType.CONTRADICTS.value,
                        "confidence_score": max(existing.confidence_score, secondary.confidence_score),
                        "attributes": {"topic": payload.topic},
                        "source_type": "contradiction_group",
                        "source_ref": group.id,
                    }
                )
        elif payload.new_memory:
            new_memories = self._create_staged_memory(
                user_id=user_id,
                turn_record_id=turn_record_id,
                payload=payload.new_memory,
                default_container_id=default_container_id,
            )
            for memory in new_memories:
                memory.status = MemoryStatus.CONFLICTED.value
                self.memory_repo.add_contradiction_item(
                    group.id,
                    memory.id,
                    ContradictionRole.COUNTER_CLAIM.value,
                    memory.confidence_score,
                )
            created.extend(new_memories)
            touched.extend(new_memories)
            for memory in new_memories:
                graph_edges.append(
                    {
                        "from_node_type": GraphNodeType.MEMORY.value,
                        "from_node_id": existing.id,
                        "to_node_type": GraphNodeType.MEMORY.value,
                        "to_node_id": memory.id,
                        "edge_type": GraphEdgeType.CONTRADICTS.value,
                        "confidence_score": max(existing.confidence_score, memory.confidence_score),
                        "attributes": {"topic": payload.topic},
                        "source_type": "contradiction_group",
                        "source_ref": group.id,
                    }
                )
        return AdjudicationCommitResult(created, touched, contradiction_topics, [], [], [], graph_edges)

    def _upsert_memory_embedding(self, memory: Memory) -> None:
        """Refresh the embedding for a created or updated memory."""
        self.memory_repo.upsert_embedding(
            memory.id,
            self.embedding_service.embed_texts([memory.content])[0],
            self.embedding_model,
        )

    def _resolve_memory_ref(self, memory_ref: str, staged_memory_refs: dict[str, list[Memory]]) -> str | None:
        """Resolve a staged or canonical memory reference."""
        if memory_ref in staged_memory_refs and staged_memory_refs[memory_ref]:
            return staged_memory_refs[memory_ref][0].id
        memory = self.memory_repo.get(memory_ref)
        if memory is None:
            return None
        return memory.id

    @staticmethod
    def _memory_payload(
        *,
        turn_record_id: str,
        memory_type: str,
        content: str,
        summary: str | None,
        rationale: str | None,
        evidence: list[str],
        importance_score: float,
        confidence_score: float,
        novelty_score: float,
        initial_relevance_score: float,
        contradiction_risk: float,
        scope_type: str,
        bucket_id: str | None,
    ) -> dict:
        """Build a canonical memory payload."""
        return {
            "turn_record_id": turn_record_id,
            "scope_type": scope_type,
            "bucket_id": bucket_id,
            "memory_type": memory_type,
            "content": content,
            "summary": summary,
            "rationale": rationale,
            "evidence_json": {"items": evidence},
            "importance_score": importance_score,
            "confidence_score": confidence_score,
            "novelty_score": novelty_score,
            "initial_relevance_score": initial_relevance_score,
            "current_relevance_score": initial_relevance_score,
            "average_relevance_score": initial_relevance_score,
            "contradiction_risk": contradiction_risk,
            "recall_count": 0,
            "decay_score": 1.0,
            "status": MemoryStatus.ACTIVE.value,
            "source_type": SourceType.TURN.value,
            "source_ref": turn_record_id,
        }

    @staticmethod
    def _proposal_ids_for_memories(
        bundle: MaintenanceProposalBundle,
        memory_ids: list[str],
        proposal_type: str | None,
    ) -> list[str]:
        """Resolve maintenance proposals that touch a set of memory ids."""
        selected = []
        wanted = set(memory_ids)
        for proposal in bundle.proposals:
            if proposal_type and proposal.proposal_type != proposal_type:
                continue
            if wanted.intersection(proposal.memory_ids):
                selected.append(proposal.proposal_id)
        return selected
