from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from aerospike_sdk import AerospikeError, ResultCode
from aerospike_sdk.sync import TransactionalSession

from langchain_aerospike_kv import AerospikeByteStore
from langchain_aerospike_kv._schema import (
    CHUNK_SIZE,
    MAX_CHUNKS,
    MAX_VALUE_SIZE,
    READ_KEYS,
    MutationBudgetExceededError,
    Root,
    check_mutation_budget,
    decode_root,
    encode_value,
    root_id,
)
from tests.unit_tests._helpers import missing, mock_session, row, stream

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [32, READ_KEYS, READ_KEYS + 1])
def test_root_dispatch_is_independent_of_chunk_window(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AerospikeByteStore(mock_session(), namespace="test_sc", set_name="unit")
    keys = [str(index) for index in range(size)]

    def roots(_: object, group: list[str]) -> list[Root]:
        return [decode_root(encode_value(key, key.encode())[0], key) for key in group]

    read = MagicMock(side_effect=roots)
    monkeypatch.setattr(store, "_roots", read)
    assert store.mget([*keys, keys[0]]) == [key.encode() for key in [*keys, keys[0]]]
    assert [len(call.args[1]) for call in read.call_args_list] == (
        [size] if size <= READ_KEYS else [READ_KEYS, 1]
    )


def test_chunk_reconstruction_keeps_small_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AerospikeByteStore(mock_session(), namespace="test_sc", set_name="unit")
    keys = [str(index) for index in range(32)]
    payload = b"x" * (CHUNK_SIZE + 1)
    encoded = [encode_value(key, payload) for key in keys]
    monkeypatch.setattr(
        store,
        "_roots",
        MagicMock(
            return_value=[
                decode_root(bins, key)
                for key, (bins, _) in zip(keys, encoded, strict=True)
            ]
        ),
    )
    read = MagicMock(
        side_effect=[
            [chunk for _, chunks in encoded[:16] for chunk in chunks],
            [chunk for _, chunks in encoded[16:] for chunk in chunks],
        ]
    )
    monkeypatch.setattr("langchain_aerospike_kv.stores.read_records", read)
    assert store.mget(keys) == [payload] * len(keys)
    assert [len(call.args[2]) for call in read.call_args_list] == [32, 32]


@pytest.mark.parametrize("deleting", [False, True])
def test_single_inline_mutation_is_conditional_and_not_transactional(
    *,
    deleting: bool,
) -> None:
    session = mock_session()
    store = AerospikeByteStore(session, namespace="test_sc", set_name="unit")
    bins, _ = encode_value("key", b"old")
    session.get.return_value = MagicMock(bins=bins, generation=17)
    builder = session.delete.return_value if deleting else session.replace.return_value
    for method in (
        "ensure_generation_is",
        "where",
        "put",
        "never_expire",
        "with_durable_delete",
        "with_txn",
        "include_missing_keys",
        "fail_on_filtered_out",
    ):
        getattr(builder, method).return_value = builder
    result = stream([row()])
    builder.execute.return_value = result
    if deleting:
        store.mdelete(["key", "key"])
        builder.with_durable_delete.assert_called_once()
    else:
        store.mset([("key", b"ignored"), ("key", b"new")])
        assert builder.put.call_args.args[0]["data"] == b"new"
        builder.never_expire.assert_called_once()
    session.get.assert_not_called()
    session.query.assert_not_called()
    builder.ensure_generation_is.assert_not_called()
    builder.where.assert_called_once()
    builder.with_txn.assert_called_once_with(None)
    builder.include_missing_keys.assert_called_once()
    builder.fail_on_filtered_out.assert_called_once()
    session.do_in_transaction.assert_not_called()
    result.close.assert_called_once()


@pytest.mark.parametrize("deleting", [False, True])
def test_missing_inline_root_is_created_or_ignored_without_a_read(
    *, deleting: bool
) -> None:
    session = mock_session()
    store = AerospikeByteStore(session, namespace="test_sc", set_name="unit")
    builder = session.delete.return_value if deleting else session.replace.return_value
    for method in (
        "put",
        "never_expire",
        "where",
        "with_durable_delete",
        "with_txn",
        "include_missing_keys",
        "fail_on_filtered_out",
    ):
        getattr(builder, method).return_value = builder
    builder.execute.return_value = stream([missing() if deleting else row()])
    if deleting:
        store.mdelete(["key"])
        session.insert.assert_not_called()
        session.delete.assert_called_once()
    else:
        store.mset([("key", b"new")])
        session.replace.assert_called_once_with(store._dataset.id(root_id("key")))
    session.get.assert_not_called()
    session.query.assert_not_called()
    session.do_in_transaction.assert_not_called()


@pytest.mark.parametrize("deleting", [False, True])
def test_chunked_old_root_falls_back_to_transaction(*, deleting: bool) -> None:
    session = mock_session()
    store = AerospikeByteStore(session, namespace="test_sc", set_name="unit")
    builder = session.delete.return_value if deleting else session.replace.return_value
    for method in (
        "put",
        "never_expire",
        "where",
        "with_durable_delete",
        "with_txn",
        "include_missing_keys",
        "fail_on_filtered_out",
    ):
        getattr(builder, method).return_value = builder
    failed = MagicMock(index=0)
    failed.or_raise.side_effect = AerospikeError(
        "chunked", result_code=ResultCode.FILTERED_OUT
    )
    builder.execute.return_value = stream([failed])
    if deleting:
        store.mdelete(["key"])
    else:
        store.mset([("key", b"new")])
    session.do_in_transaction.assert_called_once()
    session.get.assert_not_called()
    session.query.assert_not_called()


@pytest.mark.parametrize(
    "code",
    [
        ResultCode.GENERATION_ERROR,
        ResultCode.FILTERED_OUT,
        ResultCode.KEY_EXISTS_ERROR,
        ResultCode.KEY_NOT_FOUND_ERROR,
        ResultCode.MRT_BLOCKED,
    ],
)
def test_definite_conditional_conflict_falls_back_to_fresh_mrt(
    code: ResultCode,
) -> None:
    session = mock_session()
    store = AerospikeByteStore(session, namespace="test_sc", set_name="unit")
    session.get.return_value = MagicMock(
        bins=encode_value("key", b"old")[0], generation=1
    )
    builder = session.replace.return_value
    for method in (
        "ensure_generation_is",
        "where",
        "put",
        "never_expire",
        "with_txn",
        "include_missing_keys",
        "fail_on_filtered_out",
    ):
        getattr(builder, method).return_value = builder
    failed = MagicMock(index=0)
    failed.or_raise.side_effect = AerospikeError("conflict", result_code=code)
    result = stream([failed])
    builder.execute.return_value = result
    store.mset([("key", b"new")])
    session.do_in_transaction.assert_called_once()
    result.close.assert_called_once()


@pytest.mark.parametrize(
    ("code", "in_doubt"),
    [
        (ResultCode.GENERATION_ERROR, True),
        (None, False),
        (ResultCode.SERVER_ERROR, False),
    ],
)
def test_uncertain_or_infrastructure_error_is_not_retried(
    code: ResultCode | None, *, in_doubt: bool
) -> None:
    session = mock_session()
    store = AerospikeByteStore(session, namespace="test_sc", set_name="unit")
    builder = session.replace.return_value
    for method in (
        "put",
        "never_expire",
        "where",
        "with_txn",
        "include_missing_keys",
        "fail_on_filtered_out",
    ):
        getattr(builder, method).return_value = builder
    failure = AerospikeError("failed", result_code=code, in_doubt=in_doubt)
    builder.execute.side_effect = failure
    with pytest.raises(AerospikeError, match="failed"):
        store.mset([("key", b"new")])
    session.do_in_transaction.assert_not_called()


def test_budget_counts_old_cleanup_and_overlapping_positions() -> None:
    roots = [Root(str(i), "unused", MAX_CHUNKS, b"") for i in range(32)]
    check_mutation_budget(roots[:31], [b"small"] * 31)
    check_mutation_budget(roots[:31])
    for values in ([b"small"] * 32, None):
        with pytest.raises(MutationBudgetExceededError):
            check_mutation_budget(roots, values)
    # Overlapping old/new positions are each written only once.
    check_mutation_budget(roots[:31], [b"x" * MAX_VALUE_SIZE] * 31)
    exact = [*roots[:31], Root("last", "unused", 96, b"")]
    check_mutation_budget(exact)
    check_mutation_budget(exact, [b"small"] * 32)
    with pytest.raises(MutationBudgetExceededError):
        check_mutation_budget([*roots[:31], Root("last", "unused", 97, b"")])
    check_mutation_budget([None] * 128, [b"x"] * 128)
    check_mutation_budget([None] * 128)


def test_missing_transactional_roots_are_create_only_not_blind_upserts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AerospikeByteStore(mock_session(), namespace="test_sc", set_name="unit")
    monkeypatch.setattr(store, "_roots", MagicMock(return_value=[None, None]))
    tx = MagicMock(spec=TransactionalSession)
    builder = tx.insert.return_value
    for method in ("put", "never_expire", "replace", "insert"):
        getattr(builder, method).return_value = builder
    result = stream([row(index=index) for index in range(4)])
    builder.execute.return_value = result
    store._replace_group(
        tx, group=[("key", b"x" * (CHUNK_SIZE + 1)), ("next", b"inline")]
    )
    tx.insert.assert_called_once_with(store._dataset.id(root_id("key")))
    builder.insert.assert_called_once_with(store._dataset.id(root_id("next")))
    assert (
        len(builder.replace.call_args_list) == 2
    )  # Existing chunk positions are reused.
    tx.replace.assert_not_called()
    result.close.assert_called_once()


@pytest.mark.parametrize("deleting", [False, True])
def test_oversized_transaction_splits_before_writes(
    monkeypatch: pytest.MonkeyPatch, *, deleting: bool
) -> None:
    session = mock_session()
    store = AerospikeByteStore(session, namespace="test_sc", set_name="unit")
    transactions: list[MagicMock] = []

    def transaction(
        operation: Callable[[TransactionalSession], None], **_: object
    ) -> None:
        tx = MagicMock(spec=TransactionalSession)
        transactions.append(tx)
        operation(tx)

    def roots(_: object, keys: list[str]) -> list[Root]:
        return [Root(key, "unused", MAX_CHUNKS, b"") for key in keys]

    session.do_in_transaction.side_effect = transaction
    monkeypatch.setattr(store, "_roots", roots)
    # Stop only the fitting groups, after the production budget check.
    cleanup = MagicMock(side_effect=RuntimeError("reached writes"))
    monkeypatch.setattr(store, "_verify_cleanup", cleanup)
    keys = [str(i) for i in range(128)]
    action: Callable[[], None] = (
        (lambda: store._remove_group(keys))
        if deleting
        else (lambda: store._set_group([(key, b"small") for key in keys]))
    )
    with pytest.raises(RuntimeError, match="reached writes"):
        action()
    assert len(transactions) == 4  # 128 -> 64 -> 32 -> 16
    for tx in transactions[:3]:
        tx.replace.assert_not_called()
        tx.delete.assert_not_called()
    cleanup.assert_called_once()
