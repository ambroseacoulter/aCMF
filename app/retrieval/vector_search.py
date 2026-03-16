"""Vector retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass

from app.db.models import Memory
from app.services.embedding_service import EmbeddingService
from app.storage.memory_repo import MemoryRepository


@dataclass
class VectorSearchHit:
    """One vector-search result with a normalized similarity score."""

    memory: Memory
    similarity: float


class VectorSearch:
    """Retrieve memories using embeddings and pgvector."""

    def __init__(self, memory_repo: MemoryRepository, embedding_service: EmbeddingService) -> None:
        self.memory_repo = memory_repo
        self.embedding_service = embedding_service

    def search(self, *, user_id: str, scope_types: list[str], bucket_ids: list[str], query: str, limit: int) -> list[Memory]:
        """Search for memories similar to the query text."""
        return [hit.memory for hit in self.search_with_scores(user_id=user_id, scope_types=scope_types, bucket_ids=bucket_ids, query=query, limit=limit)]

    def search_with_scores(
        self,
        *,
        user_id: str,
        scope_types: list[str],
        bucket_ids: list[str],
        query: str,
        limit: int,
    ) -> list[VectorSearchHit]:
        """Search for memories similar to the query text and return normalized similarity scores."""
        query_vector = self.embedding_service.embed_texts([query])[0]
        return [
            VectorSearchHit(memory=memory, similarity=similarity)
            for memory, similarity in self.memory_repo.similarity_search_with_scores(user_id, scope_types, bucket_ids, query_vector, limit)
        ]
