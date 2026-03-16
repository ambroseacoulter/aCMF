"""Graph persistence operations."""

from __future__ import annotations

from sqlalchemy import Select, distinct, select, union_all
from sqlalchemy.orm import Session

from app.db.models import GraphEdge, GraphEntity, GraphProjectionOutbox, GraphRelation, Memory, MemoryEntityLink


class GraphRepository:
    """Repository for canonical graph tables and sync outbox."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def find_entity(self, user_id: str, canonical_name: str) -> GraphEntity | None:
        """Find an entity by canonical name."""
        stmt: Select[tuple[GraphEntity]] = select(GraphEntity).where(
            GraphEntity.user_id == user_id,
            GraphEntity.canonical_name == canonical_name,
        )
        return self.session.scalar(stmt)

    def upsert_entity(self, user_id: str, entity_type: str, canonical_name: str, aliases: list[str], attributes: dict) -> GraphEntity:
        """Create or update a graph entity."""
        entity = self.find_entity(user_id, canonical_name)
        if entity is None:
            entity = GraphEntity(
                user_id=user_id,
                entity_type=entity_type,
                canonical_name=canonical_name,
                aliases_json=aliases,
                attributes_json=attributes,
            )
            self.session.add(entity)
            self.session.flush()
        else:
            entity.entity_type = entity_type
            entity.aliases_json = sorted(set(entity.aliases_json + aliases))
            entity.attributes_json = {**entity.attributes_json, **attributes}
        return entity

    def upsert_relation(self, user_id: str, from_entity_id: str, to_entity_id: str, relation_type: str, confidence_score: float, evidence: dict, attributes: dict) -> GraphRelation:
        """Create a graph relation."""
        relation = GraphRelation(
            user_id=user_id,
            from_entity_id=from_entity_id,
            to_entity_id=to_entity_id,
            relation_type=relation_type,
            confidence_score=confidence_score,
            evidence_json=evidence,
            attributes_json=attributes,
        )
        self.session.add(relation)
        self.session.flush()
        return relation

    def link_memory_to_entity(self, memory_id: str, entity_id: str, link_type: str = "MENTIONS") -> MemoryEntityLink:
        """Create a memory-entity link if missing."""
        existing = self.session.get(MemoryEntityLink, {"memory_id": memory_id, "entity_id": entity_id})
        if existing is None:
            existing = MemoryEntityLink(memory_id=memory_id, entity_id=entity_id, link_type=link_type)
            self.session.add(existing)
            self.session.flush()
        return existing

    def upsert_graph_edge(
        self,
        *,
        user_id: str,
        from_node_type: str,
        from_node_id: str,
        to_node_type: str,
        to_node_id: str,
        edge_type: str,
        confidence_score: float,
        attributes: dict,
        source_type: str,
        source_ref: str | None = None,
    ) -> GraphEdge:
        """Create or update a canonical graph edge."""
        stmt = select(GraphEdge).where(
            GraphEdge.user_id == user_id,
            GraphEdge.from_node_type == from_node_type,
            GraphEdge.from_node_id == from_node_id,
            GraphEdge.to_node_type == to_node_type,
            GraphEdge.to_node_id == to_node_id,
            GraphEdge.edge_type == edge_type,
            GraphEdge.source_ref == source_ref,
        )
        edge = self.session.scalar(stmt)
        if edge is None:
            edge = GraphEdge(
                user_id=user_id,
                from_node_type=from_node_type,
                from_node_id=from_node_id,
                to_node_type=to_node_type,
                to_node_id=to_node_id,
                edge_type=edge_type,
                confidence_score=confidence_score,
                attributes_json=attributes,
                source_type=source_type,
                source_ref=source_ref,
            )
            self.session.add(edge)
            self.session.flush()
        else:
            edge.confidence_score = confidence_score
            edge.attributes_json = {**edge.attributes_json, **attributes}
            edge.source_type = source_type
        return edge

    def get_memory_projection_payloads(self, memory_ids: list[str]) -> list[dict]:
        """Return serialized memory payloads for graph projection."""
        if not memory_ids:
            return []
        stmt = select(Memory).where(Memory.id.in_(memory_ids))
        memories = list(self.session.scalars(stmt))
        return [
            {
                "id": memory.id,
                "user_id": memory.user_id,
                "scope_type": memory.scope_type,
                "bucket_id": memory.bucket_id,
                "memory_type": memory.memory_type,
                "status": memory.status,
                "importance_score": memory.importance_score,
                "confidence_score": memory.confidence_score,
                "novelty_score": memory.novelty_score,
                "current_relevance_score": memory.current_relevance_score,
                "average_relevance_score": memory.average_relevance_score,
                "contradiction_risk": memory.contradiction_risk,
                "recall_count": memory.recall_count,
                "decay_score": memory.decay_score,
                "summary": memory.summary,
                "content": memory.content,
                "rationale": memory.rationale,
                "evidence_json": memory.evidence_json,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            }
            for memory in memories
        ]

    def get_user_memory_projection_payloads(self, user_id: str) -> list[dict]:
        """Return all memory projection payloads for a user."""
        stmt = select(Memory.id).where(Memory.user_id == user_id)
        memory_ids = [memory_id for memory_id in self.session.scalars(stmt)]
        return self.get_memory_projection_payloads(memory_ids)

    def get_entity_projection_payloads(self, user_id: str) -> list[dict]:
        """Return serialized entity payloads for graph projection."""
        stmt = select(GraphEntity).where(GraphEntity.user_id == user_id)
        entities = list(self.session.scalars(stmt))
        return [
            {
                "id": entity.id,
                "entity_type": entity.entity_type,
                "canonical_name": entity.canonical_name,
                "aliases_json": entity.aliases_json,
                "attributes_json": entity.attributes_json,
            }
            for entity in entities
        ]

    def get_relation_projection_payloads(self, user_id: str) -> list[dict]:
        """Return serialized relation payloads for graph projection."""
        source_entity = GraphEntity.__table__.alias("source_entity")
        target_entity = GraphEntity.__table__.alias("target_entity")
        stmt = (
            select(
                GraphRelation,
                source_entity.c.canonical_name.label("from_entity_name"),
                target_entity.c.canonical_name.label("to_entity_name"),
            )
            .join(source_entity, GraphRelation.from_entity_id == source_entity.c.id)
            .join(target_entity, GraphRelation.to_entity_id == target_entity.c.id)
            .where(GraphRelation.user_id == user_id)
        )
        rows = self.session.execute(stmt).all()
        return [
            {
                "id": relation.id,
                "from_entity_id": relation.from_entity_id,
                "to_entity_id": relation.to_entity_id,
                "from_entity_name": from_entity_name,
                "to_entity_name": to_entity_name,
                "relation_type": relation.relation_type,
                "confidence_score": relation.confidence_score,
                "attributes_json": relation.attributes_json,
                "evidence_json": relation.evidence_json,
            }
            for relation, from_entity_name, to_entity_name in rows
        ]

    def get_memory_link_projection_payloads(self, user_id: str) -> list[dict]:
        """Return serialized memory-entity link payloads for graph projection."""
        entity = GraphEntity.__table__.alias("entity")
        memory = Memory.__table__.alias("memory")
        stmt = (
            select(
                MemoryEntityLink,
                entity.c.canonical_name.label("entity_name"),
                memory.c.summary.label("memory_summary"),
            )
            .join(memory, MemoryEntityLink.memory_id == memory.c.id)
            .join(entity, MemoryEntityLink.entity_id == entity.c.id)
            .where(memory.c.user_id == user_id)
        )
        rows = self.session.execute(stmt).all()
        return [
            {
                "id": "{0}:{1}:{2}".format(link.memory_id, link.entity_id, link.link_type),
                "memory_id": link.memory_id,
                "entity_id": link.entity_id,
                "link_type": link.link_type,
                "entity_name": entity_name,
                "memory_summary": memory_summary,
            }
            for link, entity_name, memory_summary in rows
        ]

    def get_graph_edge_projection_payloads(self, user_id: str) -> list[dict]:
        """Return serialized general edge payloads for graph projection."""
        stmt = select(GraphEdge).where(GraphEdge.user_id == user_id)
        edges = list(self.session.scalars(stmt))
        return [
            {
                "id": edge.id,
                "from_node_type": edge.from_node_type,
                "from_node_id": edge.from_node_id,
                "to_node_type": edge.to_node_type,
                "to_node_id": edge.to_node_id,
                "edge_type": edge.edge_type,
                "confidence_score": edge.confidence_score,
                "attributes_json": edge.attributes_json,
                "source_type": edge.source_type,
                "source_ref": edge.source_ref,
            }
            for edge in edges
        ]

    def enqueue_projection_event(self, event_type: str, user_id: str, payload: dict) -> GraphProjectionOutbox:
        """Create an outbox event for Neo4j sync."""
        event = GraphProjectionOutbox(event_type=event_type, user_id=user_id, payload_json=payload)
        self.session.add(event)
        self.session.flush()
        return event

    def get_pending_outbox_events(self, limit: int = 100) -> list[GraphProjectionOutbox]:
        """Return pending outbox events ordered by creation time."""
        stmt = (
            select(GraphProjectionOutbox)
            .where(GraphProjectionOutbox.status == "pending")
            .order_by(GraphProjectionOutbox.created_at)
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def get_outbox_events_by_ids(self, event_ids: list[str]) -> list[GraphProjectionOutbox]:
        """Return pending outbox events for the provided ids in creation order."""
        if not event_ids:
            return []
        stmt = (
            select(GraphProjectionOutbox)
            .where(
                GraphProjectionOutbox.id.in_(event_ids),
                GraphProjectionOutbox.status == "pending",
            )
            .order_by(GraphProjectionOutbox.created_at)
        )
        return list(self.session.scalars(stmt))

    def get_oldest_pending_outbox_created_at(self):
        """Return the oldest pending outbox timestamp if any."""
        stmt = (
            select(GraphProjectionOutbox.created_at)
            .where(GraphProjectionOutbox.status == "pending")
            .order_by(GraphProjectionOutbox.created_at)
            .limit(1)
        )
        return self.session.scalar(stmt)

    def list_projection_user_ids(self) -> list[str]:
        """Return user ids that have canonical graph state to project."""
        user_union = union_all(
            select(Memory.user_id),
            select(GraphEntity.user_id),
            select(GraphRelation.user_id),
            select(GraphEdge.user_id),
        ).subquery()
        stmt = select(distinct(user_union.c.user_id)).order_by(user_union.c.user_id)
        return [user_id for user_id in self.session.scalars(stmt) if user_id]
