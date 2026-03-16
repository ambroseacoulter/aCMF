"""Prompt rendering helpers for LLM roles."""

from __future__ import annotations

from dataclasses import dataclass

from app.db.models import Memory
from app.retrieval.graph_search import GraphFact


@dataclass
class PromptRenderContext:
    """Compact prompt render input."""

    user_id: str
    subject: str
    memories: list[Memory]
    graph_facts: list[GraphFact]
    contradiction_summaries: list[str]
    lineage_summaries: list[str]
    budgets: dict
    extras: dict


def format_memories(memories: list[Memory], limit: int = 20) -> str:
    """Render bounded memory summaries for prompts."""
    lines: list[str] = []
    for memory in memories[:limit]:
        lines.append(
            "- id={0} scope={1} bucket={2} type={3} status={4} importance={5:.2f} confidence={6:.2f} relevance={7:.2f} summary={8}".format(
                memory.id,
                memory.scope_type,
                memory.bucket_id or "none",
                memory.memory_type,
                memory.status,
                memory.importance_score,
                memory.confidence_score,
                memory.current_relevance_score,
                (memory.summary or memory.content)[:220].replace("\n", " "),
            )
        )
    return "\n".join(lines) if lines else "- none"


def format_graph_facts(graph_facts: list[GraphFact], limit: int = 20) -> str:
    """Render bounded graph facts."""
    lines = [
        "- entity={0} relation={1} related={2} confidence={3:.2f}".format(
            fact.entity_name,
            fact.relation_type,
            fact.related_entity_name,
            fact.confidence_score,
        )
        for fact in graph_facts[:limit]
    ]
    return "\n".join(lines) if lines else "- none"


def format_string_list(items: list[str], empty_label: str = "none", limit: int = 15) -> str:
    """Render a list of strings."""
    if not items:
        return "- {0}".format(empty_label)
    return "\n".join("- {0}".format(item) for item in items[:limit])
