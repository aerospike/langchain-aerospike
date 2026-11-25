"""Bounded SDK batch I/O with explicit per-row and stream-lifecycle checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aerospike_sdk import AerospikeError, ErrorStrategy, RecordNotFoundError, ResultCode

from langchain_aerospike_kv._schema import BATCH_RECORDS, Bins

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aerospike_sdk import DataSet
    from aerospike_sdk.sync import Session
    from aerospike_sdk.sync.record_stream import RecordStream


def read_records(
    session: Session,
    dataset: DataSet,
    identifiers: Sequence[str],
    *,
    bins: list[str] | None = None,
) -> list[Bins | None]:
    records: list[Bins | None] = []
    for offset in range(0, len(identifiers), BATCH_RECORDS):
        batch = identifiers[offset : offset + BATCH_RECORDS]
        query = session.query(dataset.ids(*batch)).include_missing_keys()
        if bins is not None:
            query = query.bins(bins)
        stream = query.execute(on_error=ErrorStrategy.IN_STREAM)
        batch_records: list[Bins | None] = [None] * len(batch)
        seen: set[int] = set()
        try:
            for row in stream:
                # SDK single-record success rows use the unspecified index
                # sentinel (-1); single-record errors may instead use zero.
                index = 0 if len(batch) == 1 and row.index == -1 else row.index
                if not 0 <= index < len(batch) or index in seen:
                    msg = "Invalid or duplicate SDK batch result index."
                    raise RuntimeError(msg)
                seen.add(index)
                try:
                    record = row.record_or_raise()
                except RecordNotFoundError:
                    continue
                batch_records[index] = record.bins
            if len(seen) != len(batch):
                msg = "SDK batch omitted one or more read results."
                raise RuntimeError(msg)
        finally:
            stream.close()
        records.extend(batch_records)
    return records


def check_write_results(stream: RecordStream, expected: int) -> None:
    seen: set[int] = set()
    try:
        for row in stream:
            row.or_raise()
            index = 0 if expected == 1 and row.index == -1 else row.index
            if not 0 <= index < expected or index in seen:
                msg = "Invalid or duplicate SDK mutation result index."
                raise RuntimeError(msg)
            seen.add(index)
        if len(seen) != expected:
            msg = "SDK batch omitted one or more mutation results."
            raise RuntimeError(msg)
    finally:
        stream.close()


def conditional_conflicts(
    stream: RecordStream, expected: int, *, missing_ok: bool = False
) -> set[int]:
    """Drain a native batch, returning only definite, safe-to-retry conflicts."""
    conflicts: set[int] = set()
    seen: set[int] = set()
    failure: AerospikeError | None = None
    try:
        for row in stream:
            index = 0 if expected == 1 and row.index == -1 else row.index
            if not 0 <= index < expected or index in seen:
                msg = "Invalid or duplicate SDK conditional mutation result index."
                raise RuntimeError(msg)
            seen.add(index)
            try:
                row.or_raise()
            except AerospikeError as exc:
                if (
                    missing_ok
                    and not exc.in_doubt
                    and exc.result_code == ResultCode.KEY_NOT_FOUND_ERROR
                ):
                    continue
                if not exc.in_doubt and exc.result_code in {
                    ResultCode.KEY_EXISTS_ERROR,
                    ResultCode.KEY_NOT_FOUND_ERROR,
                    ResultCode.GENERATION_ERROR,
                    ResultCode.FILTERED_OUT,
                    ResultCode.MRT_BLOCKED,
                }:
                    conflicts.add(index)
                elif failure is None or (exc.in_doubt and not failure.in_doubt):
                    failure = exc
        if len(seen) != expected:
            msg = "SDK batch omitted one or more conditional mutation results."
            raise RuntimeError(msg)
        if failure is not None:
            raise failure
    finally:
        stream.close()
    return conflicts
