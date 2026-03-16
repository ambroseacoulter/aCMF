"""Embedding service abstraction."""

from __future__ import annotations

from app.llms.client import EmbeddingClient


class EmbeddingService:
    """Thin service wrapper around the embedding client."""

    def __init__(self, client: EmbeddingClient, embedding_model: str) -> None:
        self.client = client
        self.embedding_model = embedding_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given texts."""
        return self.client.embed_texts(texts)
