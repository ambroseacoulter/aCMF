"""Query-relevance helpers for read-path retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.db.models import Memory
from app.retrieval.graph_search import GraphFact

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "do",
    "for",
    "how",
    "hows",
    "i",
    "in",
    "is",
    "it",
    "know",
    "of",
    "on",
    "the",
    "to",
    "what",
    "which",
}


@dataclass
class RetrievalCandidate:
    """One retrieved memory plus query relevance metadata."""

    memory: Memory
    sources: set[str] = field(default_factory=set)
    vector_similarity: float = 0.0
    lexical_similarity: float = 0.0
    graph_similarity: float = 0.0
    query_relevance: float = 0.0


def tokenize(text: str) -> set[str]:
    """Return normalized lexical tokens for lightweight overlap scoring."""
    return {
        token
        for token in _TOKEN_PATTERN.findall((text or "").lower())
        if len(token) > 1 and token not in _STOP_WORDS
    }


def memory_text(memory: Memory) -> str:
    """Return the searchable text for a memory."""
    parts = [memory.summary or "", memory.content or "", memory.rationale or ""]
    if memory.memory_type:
        parts.append(str(memory.memory_type))
    return " ".join(part for part in parts if part).strip()


def text_overlap_score(query: str, text: str) -> float:
    """Estimate lexical overlap between a query and text."""
    normalized_query = (query or "").strip().lower()
    normalized_text = (text or "").strip().lower()
    if not normalized_query or not normalized_text:
        return 0.0
    if normalized_query in normalized_text:
        return 1.0
    query_tokens = tokenize(normalized_query)
    text_tokens = tokenize(normalized_text)
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = query_tokens & text_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(query_tokens)


def graph_fact_score(query: str, graph_facts: list[GraphFact]) -> float:
    """Estimate graph relevance from facts returned by traversal."""
    if not graph_facts:
        return 0.0
    return max(
        text_overlap_score(
            query,
            "{0} {1} {2}".format(fact.entity_name, fact.relation_type, fact.related_entity_name),
        )
        for fact in graph_facts
    )
