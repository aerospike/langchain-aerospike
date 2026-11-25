from __future__ import annotations

from inspect import isgenerator
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from aerospike_sdk import (
    AerospikeError,
    Exp,
    RecordResult,
    ResultCode,
    TransactionError,
)
from aerospike_sdk.sync import TransactionalSession
from langchain_core.stores import BaseStore

from langchain_aerospike_kv import AerospikeByteStore
from langchain_aerospike_kv._schema import (
    CHUNK_SIZE,
    MAX_VALUE_SIZE,
    MUTATION_KEYS,
    READ_ATTEMPTS,
    chunk_ids,
    decode_root,
    encode_value,
    root_id,
)
from tests.unit_tests._helpers import DATASET, mock_session, row, stream

if TYPE_CHECKING:
    from langchain_aerospike_kv._schema import Bins


@pytest.fixture
def store() -> AerospikeByteStore:
    return AerospikeByteStore(mock_session(), namespace="test_sc", set_name="unit")


def test_inherits_executor_async_contract() -> None:
    for method in ("amget", "amset", "amdelete", "ayield_keys"):
        assert getattr(AerospikeByteStore, method) is getattr(BaseStore, method)


def test_mget_batches_unique_roots_preserving_order(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = encode_value("a", b"\0\xff")
    second, _ = encode_value("b", b"")
    read = MagicMock(return_value=[second, None, first])
    monkeypatch.setattr("langchain_aerospike_kv.stores.read_records", read)
    assert store.mget(["b", "missing", "a", "b"]) == [b"", None, b"\0\xff", b""]
    assert read.call_args_list[0].args[2] == [
        root_id("b"),
        root_id("missing"),
        root_id("a"),
    ]
    assert read.call_count == 1


def test_mget_numeric_chunks_beyond_ten(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"".join(bytes([i]) * CHUNK_SIZE for i in range(13))
    bins, chunks = encode_value("key", payload)
    read = MagicMock(side_effect=[[bins], chunks])
    monkeypatch.setattr("langchain_aerospike_kv.stores.read_records", read)
    assert store.mget(["key", "key"]) == [payload, payload]
    assert read.call_args_list[1].args[2] == chunk_ids("key", 0, len(chunks))


@pytest.mark.parametrize("replacement", [b"inline", None, b"x" * (CHUNK_SIZE + 1)])
def test_mget_retries_changed_root_not_a_snapshot(
    store: AerospikeByteStore,
    monkeypatch: pytest.MonkeyPatch,
    replacement: bytes | None,
) -> None:
    original, _ = encode_value("key", b"o" * (CHUNK_SIZE + 1))
    new, chunks = encode_value("key", replacement or b"")
    read = MagicMock(
        side_effect=[
            [original],
            [None, None],
            [new if replacement is not None else None],
            chunks,
        ]
    )
    monkeypatch.setattr("langchain_aerospike_kv.stores.read_records", read)
    assert store.mget(["key"]) == [replacement]
    assert read.call_count == (4 if chunks else 3)


def test_mget_unchanged_root_missing_chunk_is_corruption(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    original, _ = encode_value("key", b"o" * (CHUNK_SIZE + 1))
    read = MagicMock(side_effect=[[original], [None, None], [original]])
    monkeypatch.setattr("langchain_aerospike_kv.stores.read_records", read)
    with pytest.raises(ValueError, match="unchanged"):
        store.mget(["key"])
    assert read.call_count == 3


def test_mget_concurrent_retries_bounded(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    results: list[list[Bins | None]] = [
        [encode_value("key", b"x" * (CHUNK_SIZE + 1))[0]]
    ]
    for _ in range(READ_ATTEMPTS):
        results.extend(
            [[None, None], [encode_value("key", b"x" * (CHUNK_SIZE + 1))[0]]]
        )
    read = MagicMock(side_effect=results)
    monkeypatch.setattr("langchain_aerospike_kv.stores.read_records", read)
    with pytest.raises(RuntimeError, match="exhausted"):
        store.mget(["key"])
    assert read.call_count == 1 + READ_ATTEMPTS * 2


def test_mget_corrupt_root_not_miss(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "langchain_aerospike_kv.stores.read_records", MagicMock(return_value=[{}])
    )
    with pytest.raises(ValueError, match="kind/schema"):
        store.mget(["key"])


@pytest.mark.parametrize(
    "invalid",
    [
        [(cast("str", 1), b"x")],
        [("ok", b"x"), ("bad", cast("bytes", bytearray()))],
        [("ok", b"x"), ("\ud800", b"bad")],
        [("ok", b"x"), ("large", b"x" * (MAX_VALUE_SIZE + 1))],
    ],
)
def test_entire_mset_validated_before_any_transaction(
    store: AerospikeByteStore, invalid: list[tuple[str, bytes]]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        store.mset(invalid)
    cast("MagicMock", store._session).do_in_transaction.assert_not_called()


def test_mset_coalesces_duplicates_and_groups(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace = MagicMock()
    monkeypatch.setattr(store, "_mutate_batch", replace)
    session = cast("MagicMock", store._session)
    store.mset(
        [
            ("a", b"old"),
            *[(str(i), b"x") for i in range(MUTATION_KEYS)],
            ("a", b"new"),
        ]
    )
    assert replace.call_count == 2
    assert replace.call_args_list[0].args[0][0] == ("a", b"new")
    assert len(replace.call_args_list[0].args[0]) == MUTATION_KEYS
    session.do_in_transaction.assert_not_called()


def test_empty_operations_do_no_io(store: AerospikeByteStore) -> None:
    session = cast("MagicMock", store._session)
    session.reset_mock()
    assert store.mget([]) == []
    store.mset([])
    store.mdelete([])
    assert not session.mock_calls


def test_replace_shrink_cleanup_in_same_tx_and_never_expire(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_bins, old_chunks = encode_value("key", b"x" * (CHUNK_SIZE * 3))
    old = decode_root(old_bins, "key")
    roots = MagicMock(return_value=[old])
    monkeypatch.setattr(store, "_roots", roots)
    read = MagicMock(return_value=old_chunks)
    monkeypatch.setattr("langchain_aerospike_kv.stores.read_records", read)
    tx = MagicMock(spec=TransactionalSession)
    builder = tx.replace.return_value
    builder.put.return_value = builder
    builder.never_expire.return_value = builder
    builder.delete.return_value = builder
    builder.with_durable_delete.return_value = builder
    result_stream = stream([row(index=i) for i in range(4)])
    builder.execute.return_value = result_stream
    store._replace_group(tx, group=[("key", b"new")])
    roots.assert_called_once_with(tx, ["key"])
    assert read.call_args.args[0] is tx
    assert read.call_args.kwargs["bins"] == ["kind", "fmt", "version"]
    new_bins = builder.put.call_args.args[0]
    assert new_bins["data"] == b"new"
    assert new_bins["version"] != old.version
    builder.never_expire.assert_called_once()
    assert [key.value for key in builder.delete.call_args.args[0]] == chunk_ids(
        "key", 0, 3
    )
    builder.with_durable_delete.assert_called_once()
    result_stream.close.assert_called_once()


def test_missing_roots_skip_delete_even_tombstoned(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "_roots", MagicMock(return_value=[None]))
    tx = MagicMock(spec=TransactionalSession)
    store._delete_group(tx, keys=["missing"])
    tx.delete.assert_not_called()


def test_cleanup_missing_chunk_aborts_before_writes(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    bins, _ = encode_value("key", b"x" * (CHUNK_SIZE + 1))
    old = decode_root(bins, "key")
    monkeypatch.setattr(store, "_roots", MagicMock(return_value=[old]))
    monkeypatch.setattr(
        "langchain_aerospike_kv.stores.read_records",
        MagicMock(return_value=[None, None]),
    )
    tx = MagicMock(spec=TransactionalSession)
    with pytest.raises(ValueError, match="transactional cleanup"):
        store._replace_group(tx, group=[("key", b"new")])
    tx.replace.assert_not_called()
    with pytest.raises(ValueError, match="transactional cleanup"):
        store._delete_group(tx, keys=["key"])
    tx.delete.assert_not_called()


def test_cleanup_rechecks_root_in_same_transaction_for_retry(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    bins, _ = encode_value("key", b"x" * (CHUNK_SIZE + 1))
    old = decode_root(bins, "key")
    conflict = TransactionError(
        "root changed", result_code=ResultCode.MRT_VERSION_MISMATCH
    )
    roots = MagicMock(side_effect=[[old], conflict])
    monkeypatch.setattr(store, "_roots", roots)
    monkeypatch.setattr(
        "langchain_aerospike_kv.stores.read_records",
        MagicMock(return_value=[None, None]),
    )
    tx = MagicMock(spec=TransactionalSession)
    with pytest.raises(TransactionError, match="root changed"):
        store._replace_group(tx, group=[("key", b"new")])
    assert roots.call_args.args == (tx, ["key"])
    tx.replace.assert_not_called()


@pytest.mark.parametrize("early_exit", ["close", "failure", "exhaust"])
def test_yield_keys_is_lazy_generator_closing_stream(
    store: AerospikeByteStore, early_exit: str
) -> None:
    session = cast("MagicMock", store._session)
    error = AerospikeError("query failed")
    rows = [row({"key": "a[*]\0☃"})]
    if early_exit == "failure":
        rows.append(RecordResult(DATASET.id("b"), None, ResultCode.OK, exception=error))
    result_stream = stream(rows)
    query = session.query.return_value
    query.execute.return_value = result_stream
    iterator = store.yield_keys(prefix="a[")
    assert isgenerator(iterator)
    session.query.assert_not_called()
    assert next(iterator) == "a[*]\0☃"
    assert isinstance(query.where.call_args.args[0], Exp)
    query.bins.assert_called_once_with(["key"])
    assert query.with_hint.call_args.args[0].allow_scans_with_where
    if early_exit == "failure":
        with pytest.raises(AerospikeError, match="query failed"):
            next(iterator)
    elif early_exit == "exhaust":
        with pytest.raises(StopIteration):
            next(iterator)
    else:
        iterator.close()
    result_stream.close.assert_called_once()


def test_query_invalid_prefix_does_not_execute(store: AerospikeByteStore) -> None:
    with pytest.raises(UnicodeEncodeError):
        list(store.yield_keys(prefix="\ud800"))
    cast("MagicMock", store._session).query.return_value.execute.assert_not_called()


async def test_inherited_async_methods_execute_sync_paths(
    store: AerospikeByteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    get = MagicMock(return_value=[b"ok", None])
    put = MagicMock()
    delete = MagicMock()
    monkeypatch.setattr(store, "mget", get)
    monkeypatch.setattr(store, "mset", put)
    monkeypatch.setattr(store, "mdelete", delete)
    assert await store.amget(["a", "b"]) == [b"ok", None]
    await store.amset([("a", b"ok")])
    await store.amdelete(["a"])
    get.assert_called_once_with(["a", "b"])
    put.assert_called_once_with([("a", b"ok")])
    delete.assert_called_once_with(["a"])
