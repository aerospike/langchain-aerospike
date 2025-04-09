import sys
from typing import Any, Callable
from unittest.mock import MagicMock, Mock, call

from aerospike_vector_search.types import VectorDistanceMetric
import pytest
from langchain_core.documents import Document

from langchain_aerospike.vectorstores import Aerospike
from ...fake_embeddings import FakeEmbeddings

pytestmark = pytest.mark.requires("aerospike_vector_search") and pytest.mark.skipif(
    sys.version_info < (3, 9), reason="requires python3.9 or higher"
)


@pytest.fixture
def mock_client() -> None:
    try:
        from aerospike_vector_search import Client
    except ImportError:
        pytest.skip("aerospike_vector_search not installed")

    return MagicMock(Client)


def test_aerospike(mock_client: Any) -> None:
    """test that an AVS vector store can be created and used to search
    """
    from aerospike_vector_search import AVSError

    query_string = "foo"
    embedding = FakeEmbeddings()

    store = Aerospike(
        client=mock_client,
        embedding=embedding,
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=VectorDistanceMetric.COSINE,
    )

    mock_client.vector_search.side_effect = AVSError("Mocked error")

    with pytest.raises(AVSError):
        store.similarity_search_by_vector(embedding.embed_query(query_string))


def test_init_aerospike_distance(mock_client: Any) -> None:
    from aerospike_vector_search.types import VectorDistanceMetric

    embedding = FakeEmbeddings()
    aerospike = Aerospike(
        client=mock_client,
        embedding=embedding,
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=VectorDistanceMetric.COSINE,
    )

    assert aerospike._distance_strategy == VectorDistanceMetric.COSINE


def test_init_bad_embedding(mock_client: Any) -> None:
    def bad_embedding() -> None:
        return None

    with pytest.warns(
        UserWarning,
        match=(
            "Passing in `embedding` as a Callable is deprecated. Please pass"
            + " in an Embeddings object instead."
        ),
    ):
        Aerospike(
            client=mock_client,
            embedding=bad_embedding,
            text_key="text",
            vector_key="vector",
            index_name="dummy_index",
            namespace="test",
            set_name="testset",
            distance_strategy=VectorDistanceMetric.COSINE,
        )


def test_add_texts_returns_ids(mock_client: MagicMock) -> None:
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        namespace="test",
        set_name="testset",
        distance_strategy=VectorDistanceMetric.COSINE,
    )

    expected = ["0", "1"]
    actual = aerospike.add_texts(
        ["foo", "bar"],
        metadatas=[{"foo": 0}, {"bar": 1}],
        ids=["0", "1"],
        set_name="otherset",
        index_name="dummy_index",
        wait_for_index=True,
    )

    assert expected == actual
    mock_client.upsert.assert_has_calls(
        calls=[
            call(
                namespace="test",
                key="0",
                set_name="otherset",
                record_data={
                    "_id": "0",
                    "text": "foo",
                    "vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    "foo": 0,
                },
            ),
            call(
                namespace="test",
                key="1",
                set_name="otherset",
                record_data={
                    "_id": "1",
                    "text": "bar",
                    "vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "bar": 1,
                },
            ),
        ]
    )


def test_delete_returns_false(mock_client: MagicMock) -> None:
    from aerospike_vector_search import AVSServerError

    mock_client.delete.side_effect = Mock(side_effect=AVSServerError(rpc_error=""))
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        namespace="test",
        set_name="testset",
        distance_strategy=VectorDistanceMetric.COSINE,
    )

    assert not aerospike.delete(["foo", "bar"], set_name="testset")
    mock_client.delete.assert_called_once_with(
        namespace="test", key="foo", set_name="testset"
    )


def test_similarity_search_by_vector_with_score_missing_index_name(
    mock_client: Any,
) -> None:
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        # index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=VectorDistanceMetric.COSINE,
    )

    with pytest.raises(ValueError, match="index_name must be provided"):
        aerospike.similarity_search_by_vector_with_score([1.0, 2.0, 3.0])


def test_similarity_search_by_vector_with_score_filters_missing_text_key(
    mock_client: MagicMock,
) -> None:
    from aerospike_vector_search.types import Neighbor

    text_key = "text"
    mock_client.vector_search.return_value = [
        Neighbor(key="key1", fields={text_key: "1"}, distance=1.0),
        Neighbor(key="key2", fields={}, distance=0.0),
        Neighbor(key="key3", fields={text_key: "3"}, distance=3.0),
    ]
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key=text_key,
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=VectorDistanceMetric.COSINE,
    )

    actual = aerospike.similarity_search_by_vector_with_score(
        [1.0, 2.0, 3.0], k=10, metadata_keys=["foo"]
    )

    expected = [
        (Document(page_content="1"), 1.0),
        (Document(page_content="3"), 3.0),
    ]
    mock_client.vector_search.assert_called_once_with(
        index_name="dummy_index",
        namespace="test",
        query=[1.0, 2.0, 3.0],
        limit=10,
        include_fields=[text_key, "foo"],
    )

    assert expected == actual


def test_similarity_search_by_vector_with_score_overwrite_index_name(
    mock_client: MagicMock,
) -> None:
    mock_client.vector_search.return_value = []
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=VectorDistanceMetric.COSINE,
    )

    aerospike.similarity_search_by_vector_with_score(
        [1.0, 2.0, 3.0], index_name="other_index"
    )

    mock_client.vector_search.assert_called_once_with(
        index_name="other_index",
        namespace="test",
        query=[1.0, 2.0, 3.0],
        limit=4,
        include_fields=None,
    )


@pytest.mark.parametrize(
    "distance_strategy,expected_fn",
    [
        (VectorDistanceMetric.COSINE, Aerospike._cosine_relevance_score_fn),
        (VectorDistanceMetric.SQUARED_EUCLIDEAN, Aerospike._euclidean_relevance_score_fn),
        (VectorDistanceMetric.DOT_PRODUCT, Aerospike._max_inner_product_relevance_score_fn),
        (VectorDistanceMetric.HAMMING, ValueError),
    ],
)
def test_select_relevance_score_fn(
    mock_client: Any, distance_strategy: VectorDistanceMetric, expected_fn: Callable
) -> None:
    aerospike = Aerospike(
        client=mock_client,
        embedding=FakeEmbeddings(),
        text_key="text",
        vector_key="vector",
        index_name="dummy_index",
        namespace="test",
        set_name="testset",
        distance_strategy=distance_strategy,
    )

    if expected_fn is ValueError:
        with pytest.raises(ValueError):
            aerospike._select_relevance_score_fn()

    else:
        fn = aerospike._select_relevance_score_fn()

        assert fn == expected_fn
