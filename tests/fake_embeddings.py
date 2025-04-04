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


class ConsistentFakeEmbeddings(Embeddings):
    """Fake embeddings which remember all the texts seen so far to return consistent
    vectors for the same texts."""

    def __init__(self, dimensionality: int = 10) -> None:
        self.known_texts: List[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [float(1.0)] * (self.dimensionality - 1) + [
                float(self.known_texts.index(text))
            ]
            out_vectors.append(vector)
        return out_vectors

    def embed_query(self, text: str) -> List[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown."""
        return self.embed_documents([text])[0]