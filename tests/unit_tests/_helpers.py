from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from aerospike_sdk import DataSet, RecordResult, ResultCode
from aerospike_sdk.sync import Session
from aerospike_sdk.sync.record_stream import RecordStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_aerospike_kv._schema import Bins

DATASET = DataSet.of("test_sc", "unit")
NAMESPACE_INFO = {
    "strong-consistency": "true",
    "disable-mrt-writes": "false",
    "unavailable_partitions": "0",
    "dead_partitions": "0",
    "stop_writes": "false",
    "clock_skew_stop_writes": "false",
    "max-record-size": "1048576",
    "mrt-duration": "10",
}


def mock_session(fields: dict[str, str] | None = None) -> MagicMock:
    session = MagicMock(spec=Session)
    namespace_info = NAMESPACE_INFO | (fields or {})

    def info_response(command: str) -> dict[str, dict[str, str]]:
        body = (
            "8.1.3.0"
            if command == "build"
            else ";".join(f"{key}={value}" for key, value in namespace_info.items())
        )
        return {"node": {command: body}}

    session.info.return_value.info_on_all_nodes.side_effect = info_response
    return session


def row(bins: Bins | None = None, *, index: int = 0) -> RecordResult:
    return RecordResult(
        key=DATASET.id(str(index)),
        record=MagicMock(bins=bins) if bins is not None else None,
        result_code=ResultCode.OK,
        index=index,
    )


def missing(*, index: int = 0) -> RecordResult:
    return RecordResult(
        key=DATASET.id(str(index)),
        record=None,
        result_code=ResultCode.KEY_NOT_FOUND_ERROR,
        index=index,
    )


def stream(rows: Sequence[RecordResult]) -> MagicMock:
    result = MagicMock(spec=RecordStream)
    result.__iter__.return_value = iter(rows)
    return result
