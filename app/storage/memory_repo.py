"""Memory, contradiction, and lineage persistence operations."""

from __future__ import annotations

from sqlalchemy import Select, desc, func, or_, select
from sqlalchemy.orm import Session

from app.core.time import utc_now
from app.db.models import (
    Memory,
    MemoryContradictionGroup,
    MemoryContradictionItem,
    MemoryEmbedding,
    MemoryLineageEvent,
)


class MemoryRepository:
    """Repository for memory records and related maintenance state."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def base_scope_query(self, user_id: str, scope_types: list[str], bucket_ids: list[str]) -> Select[tuple[Memory]]:
        """Build a scope-filtered memory query."""
        stmt = select(Memory).where(Memory.user_id == user_id, Memory.scope_type.in_(scope_types))
        if "container" in scope_types and bucket_ids:
            stmt = stmt.where(or_(Memory.scope_type != "container", Memory.bucket_id.in_(bucket_ids)))
        elif "container" not in scope_types:
            stmt = stmt.where(Memory.scope_type != "container")
        return stmt

    def list_recent_candidates(self, user_id: str, scope_types: list[str], bucket_ids: list[str], limit: int) -> list[Memory]:
        """Return recent scoped candidate memories."""
        stmt = (
            self.base_scope_query(user_id, scope_types, bucket_ids)
            .order_by(desc(Memory.updated_at), desc(Memory.average_relevance_score))
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def list_high_signal_candidates(self, user_id: str, scope_types: list[str], bucket_ids: list[str], limit: int) -> list[Memory]:
        """Return high-signal candidates by metadata."""
        stmt = (
            self.base_scope_query(user_id, scope_types, bucket_ids)
            .order_by(
                desc(Memory.importance_score),
                desc(Memory.confidence_score),
                desc(Memory.average_relevance_score),
            )
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def search_by_metadata(
        self,
        user_id: str,
        scope_types: list[str],
        bucket_ids: list[str],
        *,
        statuses: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        """Search memories by metadata filters."""
        stmt = self.base_scope_query(user_id, scope_types, bucket_ids)
        if statuses:
            stmt = stmt.where(Memory.status.in_(statuses))
        if memory_types:
            stmt = stmt.where(Memory.memory_type.in_(memory_types))
        stmt = stmt.order_by(desc(Memory.updated_at)).limit(limit)
        return list(self.session.scalars(stmt))

    def similarity_search(self, user_id: str, scope_types: list[str], bucket_ids: list[str], query_vector: list[float], limit: int) -> list[Memory]:
        """Return vector-similar memories when pgvector is available."""
        return [memory for memory, _score in self.similarity_search_with_scores(user_id, scope_types, bucket_ids, query_vector, limit)]

    def similarity_search_with_scores(
        self,
        user_id: str,
        scope_types: list[str],
        bucket_ids: list[str],
        query_vector: list[float],
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Return vector-similar memories and normalized similarity scores."""
        try:
            stmt = (
                select(
                    Memory,
                    (1 - MemoryEmbedding.embedding.cosine_distance(query_vector)).label("similarity"),
                )
                .join(MemoryEmbedding, MemoryEmbedding.memory_id == Memory.id)
                .where(Memory.user_id == user_id, Memory.scope_type.in_(scope_types))
            )
            if "container" in scope_types and bucket_ids:
                stmt = stmt.where(or_(Memory.scope_type != "container", Memory.bucket_id.in_(bucket_ids)))
            elif "container" not in scope_types:
                stmt = stmt.where(Memory.scope_type != "container")
            stmt = stmt.order_by(MemoryEmbedding.embedding.cosine_distance(query_vector)).limit(limit)
            rows = self.session.execute(stmt).all()
            return [(memory, max(0.0, min(float(similarity or 0.0), 1.0))) for memory, similarity in rows]
        except Exception:
            return []

    def get(self, memory_id: str) -> Memory | None:
        """Return one memory by ID."""
        return self.session.get(Memory, memory_id)

    def get_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        """Return memories by ID."""
        if not memory_ids:
            return []
        stmt = select(Memory).where(Memory.id.in_(memory_ids))
        return list(self.session.scalars(stmt))

    def create_memory(self, payload: dict) -> Memory:
        """Persist a memory from a validated payload."""
        memory = Memory(**payload)
        self.session.add(memory)
        self.session.flush()
        return memory

    def update_memory(self, memory: Memory, payload: dict) -> Memory:
        """Update an existing memory record."""
        for key, value in payload.items():
            setattr(memory, key, value)
        self.session.flush()
        return memory

    def upsert_embedding(self, memory_id: str, embedding: list[float], embedding_model: str) -> None:
        """Insert or replace a memory embedding."""
        record = self.session.get(MemoryEmbedding, memory_id)
        if record is None:
            record = MemoryEmbedding(memory_id=memory_id, embedding=embedding, embedding_model=embedding_model)
            self.session.add(record)
        else:
            record.embedding = embedding
            record.embedding_model = embedding_model
            record.created_at = utc_now()

    def list_user_global_memories(self, user_id: str, limit: int | None = None) -> list[Memory]:
        """Return non-container memories for snapshots and cortex."""
        stmt = (
            select(Memory)
            .where(Memory.user_id == user_id, Memory.scope_type.in_(["user", "global"]))
            .order_by(desc(Memory.average_relevance_score), desc(Memory.updated_at))
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    def summarize_open_contradictions(self, user_id: str, limit: int = 10) -> list[str]:
        """Return short summaries of open contradiction groups."""
        stmt = (
            select(MemoryContradictionGroup)
            .where(MemoryContradictionGroup.user_id == user_id, MemoryContradictionGroup.status == "open")
            .order_by(desc(MemoryContradictionGroup.updated_at))
            .limit(limit)
        )
        groups = list(self.session.scalars(stmt))
        return ["{0}: {1}".format(group.topic, group.description or "open contradiction") for group in groups]

    def list_open_contradiction_groups(self, user_id: str, topic: str | None = None, limit: int = 10) -> list[MemoryContradictionGroup]:
        """Return open contradiction groups for a user."""
        stmt = select(MemoryContradictionGroup).where(
            MemoryContradictionGroup.user_id == user_id,
            MemoryContradictionGroup.status == "open",
        )
        if topic:
            stmt = stmt.where(MemoryContradictionGroup.topic.ilike("%{0}%".format(topic)))
        stmt = stmt.order_by(desc(MemoryContradictionGroup.updated_at)).limit(limit)
        return list(self.session.scalars(stmt))

    def summarize_lineage(self, user_id: str, limit: int = 10) -> list[str]:
        """Return short summaries of recent lineage events."""
        stmt = (
            select(MemoryLineageEvent)
            .where(MemoryLineageEvent.user_id == user_id)
            .order_by(desc(MemoryLineageEvent.updated_at))
            .limit(limit)
        )
        events = list(self.session.scalars(stmt))
        return [
            "{0} {1} {2}".format(event.source_memory_id, event.event_type, event.target_memory_id)
            for event in events
        ]

    def list_lineage_events(self, user_id: str, memory_id: str | None = None, limit: int = 10) -> list[MemoryLineageEvent]:
        """Return recent lineage events for a user or memory."""
        stmt = select(MemoryLineageEvent).where(MemoryLineageEvent.user_id == user_id)
        if memory_id:
            stmt = stmt.where(
                or_(
                    MemoryLineageEvent.source_memory_id == memory_id,
                    MemoryLineageEvent.target_memory_id == memory_id,
                )
            )
        stmt = stmt.order_by(desc(MemoryLineageEvent.updated_at)).limit(limit)
        return list(self.session.scalars(stmt))

    def find_or_create_contradiction_group(self, user_id: str, topic: str, description: str | None = None) -> MemoryContradictionGroup:
        """Fetch or create an open contradiction group by topic."""
        stmt = select(MemoryContradictionGroup).where(
            MemoryContradictionGroup.user_id == user_id,
            MemoryContradictionGroup.topic == topic,
            MemoryContradictionGroup.status == "open",
        )
        group = self.session.scalar(stmt)
        if group is None:
            group = MemoryContradictionGroup(user_id=user_id, topic=topic, description=description)
            self.session.add(group)
            self.session.flush()
        elif description and not group.description:
            group.description = description
        return group

    def add_contradiction_item(self, group_id: str, memory_id: str, role: str, confidence_score: float, notes: dict | None = None) -> MemoryContradictionItem:
        """Attach a memory to a contradiction group if not already attached."""
        stmt = select(MemoryContradictionItem).where(
            MemoryContradictionItem.group_id == group_id,
            MemoryContradictionItem.memory_id == memory_id,
        )
        existing = self.session.scalar(stmt)
        if existing is not None:
            return existing
        item = MemoryContradictionItem(
            group_id=group_id,
            memory_id=memory_id,
            role=role,
            confidence_score=confidence_score,
            notes=notes or {},
        )
        self.session.add(item)
        self.session.flush()
        return item

    def record_lineage_event(
        self,
        *,
        user_id: str,
        source_memory_id: str,
        target_memory_id: str,
        event_type: str,
        confidence_score: float,
        rationale: str | None,
        metadata_json: dict | None = None,
    ) -> MemoryLineageEvent:
        """Create a lineage event."""
        event = MemoryLineageEvent(
            user_id=user_id,
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            event_type=event_type,
            confidence_score=confidence_score,
            rationale=rationale,
            metadata_json=metadata_json or {},
        )
        self.session.add(event)
        self.session.flush()
        return event

    def count_user_memories(self, user_id: str) -> int:
        """Return total memory count for a user."""
        stmt = select(func.count()).select_from(Memory).where(Memory.user_id == user_id)
        return int(self.session.scalar(stmt) or 0)
