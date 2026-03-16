"""Primary SQLAlchemy models for aCMF v1."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.time import utc_now
from app.db.base import Base


class TimestampMixin:
    """Shared timestamp fields."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class User(Base, TimestampMixin):
    """Stored user identity."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)
    snapshot_dirty: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class Container(Base, TimestampMixin):
    """Container-scoped bucket for local memories."""

    __tablename__ = "containers"
    __table_args__ = (Index("ix_containers_user_status", "user_id", "status"),)

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    container_type: Mapped[str] = mapped_column(String(100), default="generic", nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)


class TurnRecord(Base, TimestampMixin):
    """Normalized turn payload and processing metadata."""

    __tablename__ = "turn_records"
    __table_args__ = (Index("ix_turn_records_user_created", "user_id", "created_at"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    job_id: Mapped[Optional[str]] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_response: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    user_message_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    assistant_message_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    container_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    processing_notes: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class Memory(Base, TimestampMixin):
    """Canonical persisted memory."""

    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memories_scope_lookup", "user_id", "scope_type", "bucket_id", "status"),
        Index("ix_memories_turn_scope", "turn_record_id", "scope_type"),
        Index("ix_memories_updated_at", "updated_at"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    turn_record_id: Mapped[Optional[str]] = mapped_column(ForeignKey("turn_records.id"), nullable=True)
    scope_type: Mapped[str] = mapped_column(String(50), nullable=False)
    bucket_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    novelty_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    initial_relevance_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_relevance_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    average_relevance_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    contradiction_risk: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    recall_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    decay_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)
    superseded_by_memory_id: Mapped[Optional[str]] = mapped_column(ForeignKey("memories.id"), nullable=True)
    archived_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_recalled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    source_type: Mapped[str] = mapped_column(String(50), default="turn", nullable=False)
    source_ref: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


class MemoryEmbedding(Base):
    """Vector embedding for a memory."""

    __tablename__ = "memory_embeddings"

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memories.id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MemoryContradictionGroup(Base, TimestampMixin):
    """Contradiction cluster for related memories."""

    __tablename__ = "memory_contradiction_groups"
    __table_args__ = (Index("ix_contradiction_groups_user_status", "user_id", "status"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), default="open", nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolved_memory_id: Mapped[Optional[str]] = mapped_column(ForeignKey("memories.id"), nullable=True)


class MemoryContradictionItem(Base, TimestampMixin):
    """Links memories into contradiction groups."""

    __tablename__ = "memory_contradiction_items"
    __table_args__ = (
        UniqueConstraint("group_id", "memory_id", name="uq_contradiction_group_memory"),
        Index("ix_contradiction_items_group_role", "group_id", "role"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    group_id: Mapped[str] = mapped_column(ForeignKey("memory_contradiction_groups.id", ondelete="CASCADE"), nullable=False)
    memory_id: Mapped[str] = mapped_column(ForeignKey("memories.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    notes: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class MemoryLineageEvent(Base, TimestampMixin):
    """Explicit memory lineage relationship."""

    __tablename__ = "memory_lineage_events"
    __table_args__ = (Index("ix_memory_lineage_source_target", "source_memory_id", "target_memory_id"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    source_memory_id: Mapped[str] = mapped_column(ForeignKey("memories.id", ondelete="CASCADE"), nullable=False)
    target_memory_id: Mapped[str] = mapped_column(ForeignKey("memories.id", ondelete="CASCADE"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class GraphEntity(Base, TimestampMixin):
    """Canonical graph entity record."""

    __tablename__ = "graph_entities"
    __table_args__ = (Index("ix_graph_entities_user_name", "user_id", "canonical_name"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    aliases_json: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    attributes_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class GraphRelation(Base, TimestampMixin):
    """Canonical graph relation record."""

    __tablename__ = "graph_relations"
    __table_args__ = (
        Index("ix_graph_relations_from_to", "from_entity_id", "to_entity_id"),
        Index("ix_graph_relations_user_type", "user_id", "relation_type"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    from_entity_id: Mapped[str] = mapped_column(ForeignKey("graph_entities.id"), nullable=False)
    to_entity_id: Mapped[str] = mapped_column(ForeignKey("graph_entities.id"), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    evidence_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    attributes_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class MemoryEntityLink(Base):
    """Join table between memories and graph entities."""

    __tablename__ = "memory_entity_links"
    __table_args__ = (UniqueConstraint("memory_id", "entity_id", name="uq_memory_entity_link"),)

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memories.id", ondelete="CASCADE"),
        primary_key=True,
    )
    entity_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        primary_key=True,
    )
    link_type: Mapped[str] = mapped_column(String(50), default="MENTIONS", nullable=False)


class GraphEdge(Base, TimestampMixin):
    """Canonical general graph edge between memory/entity nodes."""

    __tablename__ = "graph_edges"
    __table_args__ = (
        Index("ix_graph_edges_user_type", "user_id", "edge_type"),
        Index("ix_graph_edges_from_node", "from_node_type", "from_node_id"),
        Index("ix_graph_edges_to_node", "to_node_type", "to_node_id"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    from_node_type: Mapped[str] = mapped_column(String(20), nullable=False)
    from_node_id: Mapped[str] = mapped_column(String(255), nullable=False)
    to_node_type: Mapped[str] = mapped_column(String(20), nullable=False)
    to_node_id: Mapped[str] = mapped_column(String(255), nullable=False)
    edge_type: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    attributes_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, default="system")
    source_ref: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


class Snapshot(Base):
    """Latest user/global snapshot per user."""

    __tablename__ = "snapshots"
    __table_args__ = (UniqueConstraint("user_id", name="uq_snapshots_user_id"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    memory_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    health_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class Job(Base, TimestampMixin):
    """Background job record."""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_status_type_user", "status", "job_type", "user_id"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    payload_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class GraphProjectionOutbox(Base):
    """Outbox table used to sync canonical graph data into Neo4j."""

    __tablename__ = "graph_projection_outbox"
    __table_args__ = (Index("ix_graph_projection_outbox_status", "status", "created_at"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    payload_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
