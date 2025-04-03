"""Fake embeddings for testing."""
from typing import List

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return fake embeddings for documents."""
        return [[1.0] * 9 + [float(i)] for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> List[float]:
        """Return fake embeddings for query."""
        return [1.0] * 9 + [0.0] 