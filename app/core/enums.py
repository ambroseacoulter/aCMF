"""Shared enum definitions for the aCMF v1 domain."""

from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        """Compatibility shim for Python versions without ``StrEnum``."""


class ScopeType(StrEnum):
    """Where a memory is stored."""

    USER = "user"
    GLOBAL = "global"
    CONTAINER = "container"


class ScopeLevel(StrEnum):
    """Which scopes a read request includes."""

    USER = "user"
    USER_GLOBAL = "user_global"
    USER_GLOBAL_CONTAINER = "user_global_container"


class ReadMode(StrEnum):
    """Retrieval depth profiles."""

    SIMPLE = "simple"
    BALANCED = "balanced"
    DEEP = "deep"


class MemoryStatus(StrEnum):
    """Lifecycle status for a memory."""

    ACTIVE = "active"
    WARM = "warm"
    STALE = "stale"
    ARCHIVED = "archived"
    DUPLICATE = "duplicate"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"


class UserStatus(StrEnum):
    """User lifecycle status."""

    ACTIVE = "active"
    DISABLED = "disabled"


class ContainerStatus(StrEnum):
    """Container lifecycle status."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class ContainerType(StrEnum):
    """Known container classes."""

    THREAD = "thread"
    PROJECT = "project"
    CASE = "case"
    REPO = "repo"
    GENERIC = "generic"


class MemoryType(StrEnum):
    """Semantic memory types."""

    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    TASK = "task"
    CONSTRAINT = "constraint"
    RELATIONSHIP = "relationship"
    SUMMARY = "summary"
    IDENTITY = "identity"


class SourceType(StrEnum):
    """Memory provenance sources."""

    TURN = "turn"
    CORTEX = "cortex"
    MANUAL = "manual"


class JobType(StrEnum):
    """Background job types."""

    PROCESS_TURN = "process_turn"
    HOURLY_CORTEX = "hourly_cortex"
    SYNC_GRAPH_PROJECTION = "sync_graph_projection"


class JobStatus(StrEnum):
    """Background job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TurnStatus(StrEnum):
    """Processing state for normalized turn records."""

    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


class ProviderKind(StrEnum):
    """Supported provider types."""

    OPENAI_COMPATIBLE = "openai_compatible"
    STUB = "stub"


class MemoryAction(StrEnum):
    """Adjudicator memory actions."""

    DISCARD = "discard"
    CREATE = "create"
    UPDATE_EXISTING = "update_existing"
    MERGE_INTO_EXISTING = "merge_into_existing"
    MARK_CONTRADICTION = "mark_contradiction"


class ContradictionStatus(StrEnum):
    """Contradiction-group lifecycle states."""

    OPEN = "open"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class ContradictionRole(StrEnum):
    """A memory's role inside a contradiction group."""

    CLAIM = "claim"
    COUNTER_CLAIM = "counter_claim"
    REFERENCE = "reference"


class LineageEventType(StrEnum):
    """Explicit memory lineage event types."""

    DUPLICATE_OF = "duplicate_of"
    MERGED_INTO = "merged_into"
    SUPERSEDED_BY = "superseded_by"


class EntityType(StrEnum):
    """Supported graph entity types."""

    PERSON = "person"
    PROJECT = "project"
    ORGANIZATION = "organization"
    TOOL = "tool"
    THREAD = "thread"
    GOAL = "goal"
    TASK = "task"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    TOPIC = "topic"


class RelationType(StrEnum):
    """Supported graph relation types."""

    REPORTS_TO = "REPORTS_TO"
    INVOLVED_IN = "INVOLVED_IN"
    WORKS_ON = "WORKS_ON"
    PREFERS = "PREFERS"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    BLOCKED_BY = "BLOCKED_BY"
    SUPERSEDED_BY = "SUPERSEDED_BY"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"


class GraphNodeType(StrEnum):
    """Node types used in the projected graph."""

    MEMORY = "memory"
    ENTITY = "entity"


class GraphEdgeType(StrEnum):
    """General graph edge taxonomy."""

    MENTIONS = "MENTIONS"
    ABOUT = "ABOUT"
    PREFERS_ENTITY = "PREFERS_ENTITY"
    CONSTRAINS_ENTITY = "CONSTRAINS_ENTITY"
    TASK_FOR_ENTITY = "TASK_FOR_ENTITY"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDED_BY = "SUPERSEDED_BY"
    DUPLICATE_OF = "DUPLICATE_OF"
    MERGED_INTO = "MERGED_INTO"
    RELATED_TO = "RELATED_TO"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"


class ProjectionEventType(StrEnum):
    """Outbox event types for Neo4j projection."""

    MEMORY_NODES_UPSERTED = "memory_nodes_upserted"
    ENTITIES_UPSERTED = "entities_upserted"
    RELATIONS_UPSERTED = "relations_upserted"
    MEMORY_LINKS_UPSERTED = "memory_links_upserted"
    GRAPH_EDGES_UPSERTED = "graph_edges_upserted"


class OutboxStatus(StrEnum):
    """Graph projection outbox lifecycle."""

    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
