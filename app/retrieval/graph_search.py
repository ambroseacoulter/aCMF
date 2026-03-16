"""Graph retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphFact:
    """Structured graph fact returned from traversal."""

    entity_name: str
    relation_type: str
    related_entity_name: str
    confidence_score: float = 0.5


@dataclass
class GraphTraversalResult:
    """Result payload from graph traversal."""

    facts: list[GraphFact] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    graph_density_signal: float = 0.0
