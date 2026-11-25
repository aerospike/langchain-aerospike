from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from aerospike_sdk import AerospikeError, ErrorStrategy, RecordResult, ResultCode

from langchain_aerospike_kv._io import check_write_results, read_records
from langchain_aerospike_kv._schema import BATCH_RECORDS
from tests.unit_tests._helpers import DATASET, missing, mock_session, row, stream


def test_read_order_misses_and_explicit_missing_policy() -> None:
    session = mock_session()
    result_stream = stream(
        [row({"data": b"b"}, index=2), missing(index=1), row({}, index=0)]
    )
    query = session.query.return_value
    query.include_missing_keys.return_value = query
    query.execute.return_value = result_stream
    assert read_records(session, DATASET, ["a", "missing", "b"]) == [
        {},
        None,
        {"data": b"b"},
    ]
    query.include_missing_keys.assert_called_once()
    query.execute.assert_called_once_with(on_error=ErrorStrategy.IN_STREAM)
    result_stream.close.assert_called_once()


def test_read_batches_and_projects_cleanup_metadata() -> None:
    session = mock_session()
    query = session.query.return_value
    query.include_missing_keys.return_value = query
    query.bins.return_value = query
    first = stream([missing(index=i) for i in range(BATCH_RECORDS)])
    second = stream([missing()])
    query.execute.side_effect = [first, second]
    identifiers = [str(i) for i in range(BATCH_RECORDS + 1)]
    assert read_records(session, DATASET, identifiers, bins=["kind"]) == [None] * len(
        identifiers
    )
    assert session.query.call_count == 2
    assert query.bins.call_count == 2
    first.close.assert_called_once()
    second.close.assert_called_once()


@pytest.mark.parametrize("index", [-1, 0])
def test_single_record_index_variants(index: int) -> None:
    session = mock_session()
    query = session.query.return_value.include_missing_keys.return_value
    query.execute.return_value = stream([row({}, index=index)])
    assert read_records(session, DATASET, ["key"]) == [{}]
    check_write_results(stream([row(index=index)]), 1)


@pytest.mark.parametrize("operation", ["read", "write"])
def test_per_row_failures_propagate_and_close(operation: str) -> None:
    error = AerospikeError("transport failed")
    # A client-side exception can coexist with ResultCode.OK.
    failed = RecordResult(
        DATASET.id("a"), None, ResultCode.OK, index=0, exception=error
    )
    result_stream = stream([failed])
    if operation == "write":
        with pytest.raises(AerospikeError, match="transport failed"):
            check_write_results(result_stream, 1)
    else:
        session = mock_session()
        query = session.query.return_value.include_missing_keys.return_value
        query.execute.return_value = result_stream
        with pytest.raises(AerospikeError, match="transport failed"):
            read_records(session, DATASET, ["a"])
    result_stream.close.assert_called_once()


@pytest.mark.parametrize(
    "rows", [[], [row({}, index=0), row({}, index=0)], [row({}, index=9)]]
)
def test_invalid_or_missing_result_positions_are_errors(
    rows: list[RecordResult],
) -> None:
    result_stream = stream(rows)
    with pytest.raises(RuntimeError):
        check_write_results(result_stream, 2)
    result_stream.close.assert_called_once()
    session = mock_session()
    result_stream = stream(rows)
    query = session.query.return_value.include_missing_keys.return_value
    query.execute.return_value = result_stream
    with pytest.raises(RuntimeError):
        read_records(session, DATASET, ["a", "b"])
    result_stream.close.assert_called_once()


def test_iteration_error_closes_stream() -> None:
    result_stream = MagicMock()
    result_stream.__iter__.side_effect = OSError("iteration failed")
    with pytest.raises(OSError, match="iteration failed"):
        check_write_results(result_stream, 1)
    result_stream.close.assert_called_once()
