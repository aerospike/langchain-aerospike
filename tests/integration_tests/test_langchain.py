from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import Field

from langchain_aerospike_kv import AerospikeByteStore
from langchain_aerospike_kv._schema import CHUNK_SIZE

if TYPE_CHECKING:
    from tests._server import LiveStore

pytestmark = pytest.mark.integration


class CountingEmbeddings(DeterministicFakeEmbedding):
    document_calls: list[list[str]] = Field(default_factory=list)
    query_calls: list[str] = Field(default_factory=list)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return super().embed_query(text)


def embedding_cache(
    live: LiveStore, embeddings: CountingEmbeddings
) -> CacheBackedEmbeddings:
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        live.new_store(AerospikeByteStore),
        namespace="test-model/",
        query_embedding_cache=True,
        key_encoder="sha256",
    )


def test_embedding_cache_persists_across_store_instances(live_store: LiveStore) -> None:
    cold_embeddings = CountingEmbeddings(size=8)
    cold = embedding_cache(live_store, cold_embeddings)
    vectors = cold.embed_documents(["first", "second"])
    query = cold.embed_query("question")
    assert cold_embeddings.document_calls == [["first", "second"]]
    assert cold_embeddings.query_calls == ["question"]
    assert len(live_store.rows()) == 3

    warm_embeddings = CountingEmbeddings(size=8)
    warm = embedding_cache(live_store, warm_embeddings)
    assert warm.embed_documents(["second", "first", "second"]) == [
        vectors[1],
        vectors[0],
        vectors[1],
    ]
    assert warm.embed_query("question") == query
    assert warm_embeddings.document_calls == []
    assert warm_embeddings.query_calls == []
    assert warm.embed_documents(["first", "third"])[0] == vectors[0]
    assert warm_embeddings.document_calls == [["third"]]
    assert len(live_store.rows()) == 4


async def test_async_embedding_cache_persists_across_store_instances(
    live_store: LiveStore,
) -> None:
    cold_embeddings = CountingEmbeddings(size=8)
    cold = await asyncio.to_thread(embedding_cache, live_store, cold_embeddings)
    vectors = await cold.aembed_documents(["first", "second"])
    query = await cold.aembed_query("question")
    assert cold_embeddings.document_calls == [["first", "second"]]
    assert cold_embeddings.query_calls == ["question"]
    assert len(await asyncio.to_thread(live_store.rows)) == 3

    warm_embeddings = CountingEmbeddings(size=8)
    warm = await asyncio.to_thread(embedding_cache, live_store, warm_embeddings)
    assert await warm.aembed_documents(["second", "first", "second"]) == [
        vectors[1],
        vectors[0],
        vectors[1],
    ]
    assert await warm.aembed_query("question") == query
    assert warm_embeddings.document_calls == []
    assert warm_embeddings.query_calls == []
    assert (await warm.aembed_documents(["first", "third"]))[0] == vectors[0]
    assert warm_embeddings.document_calls == [["third"]]
    assert len(await asyncio.to_thread(live_store.rows)) == 4


def retriever(live: LiveStore, vectors: InMemoryVectorStore) -> MultiVectorRetriever:
    # LangChain's model validator supplies the otherwise-required docstore field.
    return MultiVectorRetriever(  # type: ignore[call-arg]
        vectorstore=vectors,
        byte_store=live.new_store(AerospikeByteStore),
        search_kwargs={"k": 4},
    )


def retrieval_documents() -> tuple[list[Document], list[tuple[str, Document]]]:
    children = [
        Document(page_content="Aerospike storage", metadata={"doc_id": "parent-a"}),
        Document(page_content="Another chunk", metadata={"doc_id": "parent-a"}),
        Document(page_content="LangChain retrieval", metadata={"doc_id": "parent-b"}),
        Document(page_content="Missing parent", metadata={"doc_id": "missing"}),
    ]
    parents = [
        (
            "parent-a",
            Document(
                page_content="Full Aerospike document " + "x" * CHUNK_SIZE,
                metadata={"source": "large-parent"},
            ),
        ),
        (
            "parent-b",
            Document(
                page_content="Full LangChain document", metadata={"source": "small"}
            ),
        ),
    ]
    return children, parents


def test_multi_vector_retriever_loads_persisted_parent_documents(
    live_store: LiveStore,
) -> None:
    vectors = InMemoryVectorStore(DeterministicFakeEmbedding(size=8))
    children, parents = retrieval_documents()
    vectors.add_documents(children)
    writer = retriever(live_store, vectors)
    writer.docstore.mset(parents)
    assert len(live_store.rows()) > len(parents)  # Serialized parent spans records.

    reader = retriever(live_store, vectors)
    assert reader.invoke("Aerospike storage") == [doc for _, doc in parents]
    reader.docstore.mdelete([key for key, _ in parents])
    assert reader.invoke("Aerospike storage") == []
    assert live_store.rows() == []


async def test_async_multi_vector_retriever_loads_persisted_parent_documents(
    live_store: LiveStore,
) -> None:
    vectors = InMemoryVectorStore(DeterministicFakeEmbedding(size=8))
    children, parents = retrieval_documents()
    await vectors.aadd_documents(children)
    writer = await asyncio.to_thread(retriever, live_store, vectors)
    await writer.docstore.amset(parents)
    assert len(await asyncio.to_thread(live_store.rows)) > len(parents)

    reader = await asyncio.to_thread(retriever, live_store, vectors)
    assert await reader.ainvoke("Aerospike storage") == [doc for _, doc in parents]
    await reader.docstore.amdelete([key for key, _ in parents])
    assert await reader.ainvoke("Aerospike storage") == []
    assert await asyncio.to_thread(live_store.rows) == []
