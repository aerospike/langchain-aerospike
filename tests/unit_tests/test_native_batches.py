from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from aerospike_sdk import AerospikeError, RecordResult, ResultCode

from langchain_aerospike_kv import AerospikeByteStore
from langchain_aerospike_kv._io import conditional_conflicts
from langchain_aerospike_kv._schema import CHUNK_SIZE
from tests.unit_tests._helpers import DATASET, mock_session, row, stream


def batch_store() -> tuple[AerospikeByteStore, MagicMock, MagicMock]:
    session = mock_session()
    store = AerospikeByteStore(session, namespace="test_sc", set_name="unit")
    builder = MagicMock()
    for method in ("insert", "replace", "delete"):
        getattr(session, method).return_value = builder
    for method in (
        "insert",
        "replace",
        "delete",
        "put",
        "never_expire",
        "where",
        "with_durable_delete",
        "with_txn",
        "include_missing_keys",
        "fail_on_filtered_out",
    ):
        getattr(builder, method).return_value = builder
    return store, session, builder


def failure(code: ResultCode, *, index: int, in_doubt: bool = False) -> RecordResult:
    return RecordResult(
        key=DATASET.id(str(index)),
        record=None,
        result_code=code,
        index=index,
        in_doubt=in_doubt,
    )


@pytest.mark.parametrize("deleting", [False, True])
def test_inline_batches_are_native_and_each_existing_root_is_guarded(
    *, deleting: bool
) -> None:
    store, session, builder = batch_store()
    count = 3
    result = stream([row(index=i) for i in reversed(range(count))])
    builder.execute.return_value = result
    if deleting:
        store.mdelete(["a", "b", "c", "a"])
        assert builder.with_durable_delete.call_count == 3
    else:
        store.mset([("a", b"new"), ("b", b"created"), ("c", b"new")])
        builder.insert.assert_not_called()
        assert builder.put.call_count == 3
    assert builder.where.call_count == 3
    builder.with_txn.assert_called_once_with(None)
    builder.include_missing_keys.assert_called_once()
    builder.fail_on_filtered_out.assert_called_once()
    session.get.assert_not_called()
    session.query.assert_not_called()
    session.do_in_transaction.assert_not_called()
    result.close.assert_called_once()


def test_only_chunked_keys_and_new_large_values_enter_mrt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large = b"x" * (CHUNK_SIZE + 1)
    store, session, builder = batch_store()
    fallback = MagicMock()
    monkeypatch.setattr(store, "_set_group", fallback)
    builder.execute.return_value = stream(
        [
            row(index=2),
            failure(ResultCode.FILTERED_OUT, index=1),
            row(index=0),
        ]
    )
    store.mset(
        [
            ("inline", b"new"),
            ("chunked", b"shrunk"),
            ("missing", b"created"),
            ("large", large),
        ]
    )
    fallback.assert_called_once_with([("chunked", b"shrunk"), ("large", large)])
    session.do_in_transaction.assert_not_called()


@pytest.mark.parametrize("deleting", [False, True])
def test_successful_keys_are_not_replayed_when_one_condition_loses(
    monkeypatch: pytest.MonkeyPatch, *, deleting: bool
) -> None:
    store, session, builder = batch_store()
    fallback = MagicMock()
    monkeypatch.setattr(store, "_remove_group" if deleting else "_set_group", fallback)
    builder.execute.return_value = stream(
        [
            row(index=2),
            failure(ResultCode.FILTERED_OUT, index=1),
            row(index=0),
        ]
    )
    if deleting:
        store.mdelete(["a", "b", "c"])
        fallback.assert_called_once_with(["b"])
    else:
        store.mset([(key, b"new") for key in ("a", "b", "c")])
        fallback.assert_called_once_with([("b", b"new")])
    session.do_in_transaction.assert_not_called()


@pytest.mark.parametrize(
    ("code", "in_doubt"),
    [(ResultCode.SERVER_ERROR, False), (ResultCode.FILTERED_OUT, True)],
)
def test_failed_or_uncertain_row_is_not_retried_and_later_results_are_drained(
    code: ResultCode, *, in_doubt: bool
) -> None:
    store, session, builder = batch_store()
    last = MagicMock(index=2)
    result = stream([row(index=0), failure(code, index=1, in_doubt=in_doubt), last])
    builder.execute.return_value = result
    with pytest.raises(AerospikeError) as caught:
        store.mset([(key, b"new") for key in ("a", "b", "c")])
    assert caught.value.result_code == code
    assert caught.value.in_doubt == in_doubt
    last.or_raise.assert_called_once()
    session.do_in_transaction.assert_not_called()
    result.close.assert_called_once()


@pytest.mark.parametrize("rows", [[], [row(index=0), row(index=0)], [row(index=7)]])
def test_bad_conditional_batch_result_indexes_never_look_successful(
    rows: list[RecordResult],
) -> None:
    result = stream(rows)
    with pytest.raises(RuntimeError):
        conditional_conflicts(result, 2)
    result.close.assert_called_once()


def test_uncertain_outcome_is_not_masked_by_an_earlier_definite_failure() -> None:
    result = stream(
        [
            failure(ResultCode.SERVER_ERROR, index=0),
            failure(ResultCode.FILTERED_OUT, index=1, in_doubt=True),
        ]
    )
    with pytest.raises(AerospikeError) as caught:
        conditional_conflicts(result, 2)
    assert caught.value.in_doubt
    assert caught.value.result_code == ResultCode.FILTERED_OUT
    result.close.assert_called_once()
