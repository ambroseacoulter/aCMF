"""Initial v1 schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "20260316_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial tables."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", sa.String(length=255), primary_key=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="active"),
        sa.Column("snapshot_dirty", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("job_type", sa.String(length=50), nullable=False),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])
    op.create_index("ix_jobs_status_type_user", "jobs", ["status", "job_type", "user_id"])

    op.create_table(
        "containers",
        sa.Column("id", sa.String(length=255), primary_key=True),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("container_type", sa.String(length=100), nullable=False, server_default="generic"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_containers_user_id", "containers", ["user_id"])
    op.create_index("ix_containers_user_status", "containers", ["user_id", "status"])

    op.create_table(
        "turn_records",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("job_id", sa.UUID(), sa.ForeignKey("jobs.id"), nullable=True),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("assistant_response", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_message_id", sa.String(length=255), nullable=True),
        sa.Column("assistant_message_id", sa.String(length=255), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("container_ids", sa.JSON(), nullable=False),
        sa.Column("processing_status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("processing_notes", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_turn_records_user_id", "turn_records", ["user_id"])
    op.create_index("ix_turn_records_user_created", "turn_records", ["user_id", "created_at"])

    op.create_table(
        "memories",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("turn_record_id", sa.UUID(), sa.ForeignKey("turn_records.id"), nullable=True),
        sa.Column("scope_type", sa.String(length=50), nullable=False),
        sa.Column("bucket_id", sa.String(length=255), nullable=True),
        sa.Column("memory_type", sa.String(length=50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("importance_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("confidence_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("novelty_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("initial_relevance_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("current_relevance_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("average_relevance_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("contradiction_risk", sa.Float(), nullable=False, server_default="0"),
        sa.Column("recall_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("decay_score", sa.Float(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="active"),
        sa.Column("superseded_by_memory_id", sa.UUID(), sa.ForeignKey("memories.id"), nullable=True),
        sa.Column("archived_reason", sa.String(length=255), nullable=True),
        sa.Column("last_recalled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_type", sa.String(length=50), nullable=False, server_default="turn"),
        sa.Column("source_ref", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memories_user_id", "memories", ["user_id"])
    op.create_index("ix_memories_bucket_id", "memories", ["bucket_id"])
    op.create_index("ix_memories_turn_scope", "memories", ["turn_record_id", "scope_type"])
    op.create_index("ix_memories_scope_lookup", "memories", ["user_id", "scope_type", "bucket_id", "status"])
    op.create_index("ix_memories_updated_at", "memories", ["updated_at"])

    op.create_table(
        "memory_embeddings",
        sa.Column("memory_id", sa.UUID(), sa.ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "memory_contradiction_groups",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="open"),
        sa.Column("topic", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("resolved_memory_id", sa.UUID(), sa.ForeignKey("memories.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_contradiction_groups_user_status", "memory_contradiction_groups", ["user_id", "status"])

    op.create_table(
        "memory_contradiction_items",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("group_id", sa.UUID(), sa.ForeignKey("memory_contradiction_groups.id", ondelete="CASCADE"), nullable=False),
        sa.Column("memory_id", sa.UUID(), sa.ForeignKey("memories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("notes", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("group_id", "memory_id", name="uq_contradiction_group_memory"),
    )
    op.create_index("ix_contradiction_items_group_role", "memory_contradiction_items", ["group_id", "role"])

    op.create_table(
        "memory_lineage_events",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("source_memory_id", sa.UUID(), sa.ForeignKey("memories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_memory_id", sa.UUID(), sa.ForeignKey("memories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memory_lineage_source_target", "memory_lineage_events", ["source_memory_id", "target_memory_id"])

    op.create_table(
        "graph_entities",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("canonical_name", sa.String(length=255), nullable=False),
        sa.Column("aliases_json", sa.JSON(), nullable=False),
        sa.Column("attributes_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_graph_entities_user_id", "graph_entities", ["user_id"])
    op.create_index("ix_graph_entities_user_name", "graph_entities", ["user_id", "canonical_name"])

    op.create_table(
        "graph_relations",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("from_entity_id", sa.UUID(), sa.ForeignKey("graph_entities.id"), nullable=False),
        sa.Column("to_entity_id", sa.UUID(), sa.ForeignKey("graph_entities.id"), nullable=False),
        sa.Column("relation_type", sa.String(length=50), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("attributes_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_graph_relations_user_id", "graph_relations", ["user_id"])
    op.create_index("ix_graph_relations_from_to", "graph_relations", ["from_entity_id", "to_entity_id"])
    op.create_index("ix_graph_relations_user_type", "graph_relations", ["user_id", "relation_type"])

    op.create_table(
        "memory_entity_links",
        sa.Column("memory_id", sa.UUID(), sa.ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("entity_id", sa.UUID(), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("link_type", sa.String(length=50), nullable=False, server_default="MENTIONS"),
        sa.UniqueConstraint("memory_id", "entity_id", name="uq_memory_entity_link"),
    )

    op.create_table(
        "graph_edges",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("from_node_type", sa.String(length=20), nullable=False),
        sa.Column("from_node_id", sa.String(length=255), nullable=False),
        sa.Column("to_node_type", sa.String(length=20), nullable=False),
        sa.Column("to_node_id", sa.String(length=255), nullable=False),
        sa.Column("edge_type", sa.String(length=50), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("attributes_json", sa.JSON(), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False, server_default="system"),
        sa.Column("source_ref", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_graph_edges_user_id", "graph_edges", ["user_id"])
    op.create_index("ix_graph_edges_user_type", "graph_edges", ["user_id", "edge_type"])
    op.create_index("ix_graph_edges_from_node", "graph_edges", ["from_node_type", "from_node_id"])
    op.create_index("ix_graph_edges_to_node", "graph_edges", ["to_node_type", "to_node_id"])

    op.create_table(
        "snapshots",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("memory_refs", sa.JSON(), nullable=False),
        sa.Column("health_note", sa.Text(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.UniqueConstraint("user_id", name="uq_snapshots_user_id"),
    )
    op.create_index("ix_snapshots_user_id", "snapshots", ["user_id"])

    op.create_table(
        "graph_projection_outbox",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("user_id", sa.String(length=255), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_graph_projection_outbox_user_id", "graph_projection_outbox", ["user_id"])
    op.create_index("ix_graph_projection_outbox_status", "graph_projection_outbox", ["status", "created_at"])


def downgrade() -> None:
    """Drop initial tables."""
    op.drop_index("ix_graph_projection_outbox_status", table_name="graph_projection_outbox")
    op.drop_index("ix_graph_projection_outbox_user_id", table_name="graph_projection_outbox")
    op.drop_table("graph_projection_outbox")
    op.drop_index("ix_snapshots_user_id", table_name="snapshots")
    op.drop_table("snapshots")
    op.drop_index("ix_graph_edges_to_node", table_name="graph_edges")
    op.drop_index("ix_graph_edges_from_node", table_name="graph_edges")
    op.drop_index("ix_graph_edges_user_type", table_name="graph_edges")
    op.drop_index("ix_graph_edges_user_id", table_name="graph_edges")
    op.drop_table("graph_edges")
    op.drop_table("memory_entity_links")
    op.drop_index("ix_graph_relations_user_type", table_name="graph_relations")
    op.drop_index("ix_graph_relations_from_to", table_name="graph_relations")
    op.drop_index("ix_graph_relations_user_id", table_name="graph_relations")
    op.drop_table("graph_relations")
    op.drop_index("ix_graph_entities_user_name", table_name="graph_entities")
    op.drop_index("ix_graph_entities_user_id", table_name="graph_entities")
    op.drop_table("graph_entities")
    op.drop_index("ix_memory_lineage_source_target", table_name="memory_lineage_events")
    op.drop_table("memory_lineage_events")
    op.drop_index("ix_contradiction_items_group_role", table_name="memory_contradiction_items")
    op.drop_table("memory_contradiction_items")
    op.drop_index("ix_contradiction_groups_user_status", table_name="memory_contradiction_groups")
    op.drop_table("memory_contradiction_groups")
    op.drop_table("memory_embeddings")
    op.drop_index("ix_memories_updated_at", table_name="memories")
    op.drop_index("ix_memories_scope_lookup", table_name="memories")
    op.drop_index("ix_memories_turn_scope", table_name="memories")
    op.drop_index("ix_memories_bucket_id", table_name="memories")
    op.drop_index("ix_memories_user_id", table_name="memories")
    op.drop_table("memories")
    op.drop_index("ix_turn_records_user_created", table_name="turn_records")
    op.drop_index("ix_turn_records_user_id", table_name="turn_records")
    op.drop_table("turn_records")
    op.drop_index("ix_containers_user_status", table_name="containers")
    op.drop_index("ix_containers_user_id", table_name="containers")
    op.drop_table("containers")
    op.drop_index("ix_jobs_status_type_user", table_name="jobs")
    op.drop_index("ix_jobs_user_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_table("users")
