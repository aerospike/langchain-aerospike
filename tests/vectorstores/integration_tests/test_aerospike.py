"""Test Aerospike functionality."""

import inspect
import os
import subprocess
import time
from typing import Any, Generator

import pytest
from aerospike_vector_search import types, Client
from langchain_core.documents import Document

from langchain_aerospike.vectorstores import (
    Aerospike,
)
from ...fake_embeddings import ConsistentFakeEmbeddings

pytestmark = pytest.mark.requires("aerospike_vector_search")

TEST_INDEX_NAME = "test-index"
TEST_NAMESPACE = "test"
TEST_AEROSPIKE_HOST_PORT = ("localhost", 5002)
TEXT_KEY = "_text"
VECTOR_KEY = "_vector"
ID_KEY = "_id"
EUCLIDEAN_SCORE = 1.0
DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/dockercompose"
FEAT_KEY_PATH = DIR_PATH + "/features.conf"


def compose_up() -> None:
    subprocess.run(["docker", "compose", "up", "-d"], cwd=DIR_PATH)

    # Wait for the service to be ready
    max_retries = 60
    retry_interval = 1
    for _ in range(max_retries):
        try:
            # Try to connect to the Aerospike Vector Search service
            client = Client(seeds=types.HostPort(
                host=TEST_AEROSPIKE_HOST_PORT[0],
                port=TEST_AEROSPIKE_HOST_PORT[1]
            ))
            client.close()
            # If connection succeeds, service is ready
            break
        except Exception:
            time.sleep(retry_interval)
    else:
        raise TimeoutError("Aerospike service failed to start within the expected time")

    # a little extra time for the server to go from
    # connectable to ready to serve requests
    time.sleep(10)


def compose_down() -> None:
    subprocess.run(["docker", "compose", "down"], cwd=DIR_PATH)


@pytest.fixture(scope="class", autouse=True)
def docker_compose() -> Generator[None, None, None]:
    if not os.path.exists(FEAT_KEY_PATH):
        pytest.skip(
            "Aerospike feature key file not found at path {}".format(FEAT_KEY_PATH)
        )

    compose_up()
    yield
    compose_down()


@pytest.fixture(scope="class")
def seeds() -> Generator[Any, None, None]:
    yield types.HostPort(
        host=TEST_AEROSPIKE_HOST_PORT[0],
        port=TEST_AEROSPIKE_HOST_PORT[1],
    )


@pytest.fixture(scope="class")
@pytest.mark.requires("aerospike_vector_search")
def client(seeds: Any) -> Generator[Any, None, None]:
    with Client(seeds=seeds) as client:
        yield client


@pytest.fixture
def embedder() -> Any:
    return ConsistentFakeEmbeddings()


@pytest.fixture
def aerospike(
    client: Any, embedder: ConsistentFakeEmbeddings
) -> Generator[Aerospike, None, None]:
    yield Aerospike(
        client,
        embedder,
        TEST_NAMESPACE,
        vector_key=VECTOR_KEY,
        text_key=TEXT_KEY,
        id_key=ID_KEY,
    )


def get_func_name() -> str:
    """
    Used to get the name of the calling function. The name is used for the index
    and set name in Aerospike tests for debugging purposes.
    """
    return inspect.stack()[1].function


def wait_for_index_ready(client: Any, index_name: str) -> None:
    while True:
        index_status = client.index_get_status(namespace=TEST_NAMESPACE, name=index_name)
        if index_status.readiness == types.IndexReadiness.READY:
            break
        time.sleep(0.25)


"""
TODO: Add tests for delete()
"""


class TestAerospike:
    def test_from_text(
        self,
        client: Any,
        embedder: ConsistentFakeEmbeddings,
    ) -> None:
        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE,
        )
        aerospike = Aerospike.from_texts(
            ["foo", "bar", "baz", "bay", "bax", "baw", "bav"],
            embedder,
            client=client,
            namespace=TEST_NAMESPACE,
            index_name=index_name,
            ids=["1", "2", "3", "4", "5", "6", "7"],
            set_name=set_name,
        )

        expected = [
            Document(
                page_content="foo",
                metadata={
                    ID_KEY: "1",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                },
            ),
            Document(
                page_content="bar",
                metadata={
                    ID_KEY: "2",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
            ),
            Document(
                page_content="baz",
                metadata={
                    ID_KEY: "3",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
                },
            ),
        ]
        wait_for_index_ready(client, index_name)
        actual = aerospike.search(
            "foo", k=3, index_name=index_name, search_type="similarity"
        )

        assert actual == expected

    def test_from_documents(
        self,
        client: Any,
        embedder: ConsistentFakeEmbeddings,
    ) -> None:
        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )
        documents = [
            Document(
                page_content="foo",
                metadata={
                    ID_KEY: "1",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                },
            ),
            Document(
                page_content="bar",
                metadata={
                    ID_KEY: "2",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
            ),
            Document(
                page_content="baz",
                metadata={
                    ID_KEY: "3",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
                },
            ),
            Document(
                page_content="bay",
                metadata={
                    ID_KEY: "4",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0],
                },
            ),
            Document(
                page_content="bax",
                metadata={
                    ID_KEY: "5",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0],
                },
            ),
            Document(
                page_content="baw",
                metadata={
                    ID_KEY: "6",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
                },
            ),
            Document(
                page_content="bav",
                metadata={
                    ID_KEY: "7",
                    "_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0],
                },
            ),
        ]
        aerospike = Aerospike.from_documents(
            documents,
            embedder,
            client=client,
            namespace=TEST_NAMESPACE,
            index_name=index_name,
            ids=["1", "2", "3", "4", "5", "6", "7"],
            set_name=set_name,
        )

        wait_for_index_ready(client, index_name)
        actual = aerospike.search(
            "foo", k=3, index_name=index_name, search_type="similarity"
        )

        expected = documents[:3]

        assert actual == expected

    def test_delete(self, aerospike: Aerospike, client: Any) -> None:
        """Test end to end construction and search."""

        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )

        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )

        assert client.exists(namespace=TEST_NAMESPACE, set_name=set_name, key="1")
        assert client.exists(namespace=TEST_NAMESPACE, set_name=set_name, key="2")
        assert client.exists(namespace=TEST_NAMESPACE, set_name=set_name, key="3")

        aerospike.delete(["1", "2", "3"], set_name=set_name)

        assert not client.exists(namespace=TEST_NAMESPACE, set_name=set_name, key="1")
        assert not client.exists(namespace=TEST_NAMESPACE, set_name=set_name, key="2")
        assert not client.exists(namespace=TEST_NAMESPACE, set_name=set_name, key="3")

    def test_search_blocking(self, aerospike: Aerospike, client: Any) -> None:
        """Test end to end construction and search."""

        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )

        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # Blocks until all vectors are indexed
        expected = [Document(page_content="foo", metadata={ID_KEY: "1"})]
        wait_for_index_ready(client, index_name)
        actual = aerospike.search(
            "foo",
            k=1,
            index_name=index_name,
            search_type="similarity",
            metadata_keys=[ID_KEY],
        )

        assert actual == expected

    def test_similarity_search_with_score(
        self, aerospike: Aerospike, client: Any
    ) -> None:
        """Test end to end construction and search."""

        expected = [(Document(page_content="foo", metadata={ID_KEY: "1"}), 0.0)]
        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )
        wait_for_index_ready(client, index_name)
        actual = aerospike.similarity_search_with_score(
            "foo", k=1, index_name=index_name, metadata_keys=[ID_KEY]
        )

        assert actual == expected

    def test_similarity_search_by_vector_with_score(
        self,
        aerospike: Aerospike,
        client: Any,
        embedder: ConsistentFakeEmbeddings,
    ) -> None:
        """Test end to end construction and search."""

        expected = [
            (Document(page_content="foo", metadata={"a": "b", ID_KEY: "1"}), 0.0)
        ]
        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
            metadatas=[{"a": "b", "1": "2"}, {"a": "c"}, {"a": "d"}],
        )
        wait_for_index_ready(client, index_name)
        actual = aerospike.similarity_search_by_vector_with_score(
            embedder.embed_query("foo"),
            k=1,
            index_name=index_name,
            metadata_keys=["a", ID_KEY],
        )

        assert actual == expected

    def test_similarity_search_by_vector(
        self,
        aerospike: Aerospike,
        client: Any,
        embedder: ConsistentFakeEmbeddings,
    ) -> None:
        """Test end to end construction and search."""

        expected = [
            Document(page_content="foo", metadata={"a": "b", ID_KEY: "1"}),
            Document(page_content="bar", metadata={"a": "c", ID_KEY: "2"}),
        ]
        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
            metadatas=[{"a": "b", "1": "2"}, {"a": "c"}, {"a": "d"}],
        )
        wait_for_index_ready(client, index_name)
        actual = aerospike.similarity_search_by_vector(
            embedder.embed_query("foo"),
            k=2,
            index_name=index_name,
            metadata_keys=["a", ID_KEY],
        )

        assert actual == expected

    def test_similarity_search(self, aerospike: Aerospike, client: Any) -> None:
        """Test end to end construction and search."""

        expected = [
            Document(page_content="foo", metadata={ID_KEY: "1"}),
            Document(page_content="bar", metadata={ID_KEY: "2"}),
            Document(page_content="baz", metadata={ID_KEY: "3"}),
        ]
        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking
        wait_for_index_ready(client, index_name)
        actual = aerospike.similarity_search(
            "foo", k=3, index_name=index_name, metadata_keys=[ID_KEY]
        )

        assert actual == expected

    def test_max_marginal_relevance_search_by_vector(
        self,
        client: Any,
        embedder: ConsistentFakeEmbeddings,
    ) -> None:
        """Test max marginal relevance search."""

        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )
        aerospike = Aerospike.from_texts(
            ["foo", "bar", "baz", "bay", "bax", "baw", "bav"],
            embedder,
            client=client,
            namespace=TEST_NAMESPACE,
            index_name=index_name,
            ids=["1", "2", "3", "4", "5", "6", "7"],
            set_name=set_name,
        )

        wait_for_index_ready(client, index_name)
        mmr_output = aerospike.max_marginal_relevance_search_by_vector(
            embedder.embed_query("foo"), index_name=index_name, k=3, fetch_k=3
        )
        sim_output = aerospike.similarity_search("foo", index_name=index_name, k=3)

        assert len(mmr_output) == 3
        assert mmr_output == sim_output

        mmr_output = aerospike.max_marginal_relevance_search_by_vector(
            embedder.embed_query("foo"), index_name=index_name, k=2, fetch_k=3
        )

        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == "foo"
        assert mmr_output[1].page_content == "bar"

        mmr_output = aerospike.max_marginal_relevance_search_by_vector(
            embedder.embed_query("foo"),
            index_name=index_name,
            k=2,
            fetch_k=3,
            lambda_mult=0.1,  # more diversity
        )

        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == "foo"
        assert mmr_output[1].page_content == "baz"

        # if fetch_k < k, then the output will be less than k
        mmr_output = aerospike.max_marginal_relevance_search_by_vector(
            embedder.embed_query("foo"), index_name=index_name, k=3, fetch_k=2
        )
        assert len(mmr_output) == 2

    def test_max_marginal_relevance_search(
        self, aerospike: Aerospike, client: Any
    ) -> None:
        """Test max marginal relevance search."""

        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo", "bar", "baz", "bay", "bax", "baw", "bav"],
            ids=["1", "2", "3", "4", "5", "6", "7"],
            index_name=index_name,
            set_name=set_name,
        )
        wait_for_index_ready(client, index_name)
        mmr_output = aerospike.max_marginal_relevance_search(
            "foo", index_name=index_name, k=3, fetch_k=3
        )
        sim_output = aerospike.similarity_search("foo", index_name=index_name, k=3)

        assert len(mmr_output) == 3
        assert mmr_output == sim_output

        mmr_output = aerospike.max_marginal_relevance_search(
            "foo", index_name=index_name, k=2, fetch_k=3
        )

        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == "foo"
        assert mmr_output[1].page_content == "bar"

        mmr_output = aerospike.max_marginal_relevance_search(
            "foo",
            index_name=index_name,
            k=2,
            fetch_k=3,
            lambda_mult=0.1,  # more diversity
        )

        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == "foo"
        assert mmr_output[1].page_content == "baz"

        # if fetch_k < k, then the output will be less than k
        mmr_output = aerospike.max_marginal_relevance_search(
            "foo", index_name=index_name, k=3, fetch_k=2
        )
        assert len(mmr_output) == 2

    def test_cosine_distance(self, aerospike: Aerospike, client: Any) -> None:
        """Test cosine distance."""

        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.COSINE,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking
        wait_for_index_ready(client, index_name)
        """
        foo vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        far vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        cosine similarity ~= 0.71
        cosine distance ~= 1 - cosine similarity = 0.29
        """
        expected = pytest.approx(0.292, abs=0.002)
        output = aerospike.similarity_search_with_score(
            "far", index_name=index_name, k=3
        )

        _, actual_score = output[2]

        assert actual_score == expected

    def test_dot_product_distance(
        self, aerospike: Aerospike, client: Any
    ) -> None:
        """Test dot product distance."""

        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.DOT_PRODUCT,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking
        wait_for_index_ready(client, index_name)
        """
        foo vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        far vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        dot product = 9.0
        dot product distance = dot product * -1 = -9.0
        """
        expected = -9.0
        output = aerospike.similarity_search_with_score(
            "far", index_name=index_name, k=3
        )

        _, actual_score = output[2]

        assert actual_score == expected

    def test_euclidean_distance(self, aerospike: Aerospike, client: Any) -> None:
        """Test dot product distance."""

        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.SQUARED_EUCLIDEAN,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo", "bar", "baz"],
            ids=["1", "2", "3"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking
        wait_for_index_ready(client, index_name)
        """
        foo vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        far vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        euclidean distance = 9.0
        """
        expected = 9.0
        output = aerospike.similarity_search_with_score(
            "far", index_name=index_name, k=3
        )

        _, actual_score = output[2]

        assert actual_score == expected

    def test_as_retriever(self, aerospike: Aerospike, client: Any) -> None:
        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo", "foo", "foo", "foo", "bar"],
            ids=["1", "2", "3", "4", "5"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking
        wait_for_index_ready(client, index_name)
        aerospike._index_name = index_name
        retriever = aerospike.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        results = retriever.invoke("foo")
        assert len(results) == 3
        assert all([d.page_content == "foo" for d in results])

    def test_as_retriever_distance_threshold(
        self, aerospike: Aerospike, client: Any
    ) -> None:
        aerospike._distance_strategy = types.VectorDistanceMetric.COSINE
        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.COSINE,
            mode=types.IndexMode.STANDALONE
        )
        aerospike.add_texts(
            ["foo1", "foo2", "foo3", "bar4", "bar5", "bar6", "bar7", "bar8"],
            ids=["1", "2", "3", "4", "5", "6", "7", "8"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking

        wait_for_index_ready(client, index_name)
        aerospike._index_name = index_name
        retriever = aerospike.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 9, "score_threshold": 0.90},
        )
        results = retriever.invoke("foo1")

        assert all([d.page_content.startswith("foo") for d in results])
        assert len(results) == 3

    def test_as_retriever_add_documents(
        self, aerospike: Aerospike, client: Any
    ) -> None:

        aerospike._distance_strategy = types.VectorDistanceMetric.COSINE
        index_name = set_name = get_func_name()
        client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.COSINE,
            mode=types.IndexMode.STANDALONE
        )
        retriever = aerospike.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 9, "score_threshold": 0.90},
        )

        documents = [
            Document(
                page_content="foo1",
                metadata={
                    "a": 1,
                },
            ),
            Document(
                page_content="foo2",
                metadata={
                    "a": 2,
                },
            ),
            Document(
                page_content="foo3",
                metadata={
                    "a": 3,
                },
            ),
            Document(
                page_content="bar4",
                metadata={
                    "a": 4,
                },
            ),
            Document(
                page_content="bar5",
                metadata={
                    "a": 5,
                },
            ),
            Document(
                page_content="bar6",
                metadata={
                    "a": 6,
                },
            ),
            Document(
                page_content="bar7",
                metadata={
                    "a": 7,
                },
            ),
        ]
        retriever.add_documents(
            documents,
            ids=["1", "2", "3", "4", "5", "6", "7", "8"],
            index_name=index_name,
            set_name=set_name,
            wait_for_index=True,
        )

        wait_for_index_ready(client, index_name)
        aerospike._index_name = index_name
        results = retriever.invoke("foo1")

        assert all([d.page_content.startswith("foo") for d in results])
        assert len(results) == 3
