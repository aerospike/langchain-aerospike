from __future__ import annotations

from typing import cast

import pytest
from aerospike_sdk.sync import Session, TransactionalSession

from langchain_aerospike_kv import AerospikeByteStore
from langchain_aerospike_kv._config import namespace_record_limit, validate_names
from tests.unit_tests._helpers import mock_session


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strong-consistency", "false"),
        ("disable-mrt-writes", "true"),
        ("unavailable_partitions", "1"),
        ("dead_partitions", "1"),
        ("stop_writes", "true"),
        ("clock_skew_stop_writes", "true"),
        ("max-record-size", "65536"),
        ("max-record-size", "invalid"),
        ("mrt-duration", "0"),
    ],
)
def test_unusable_namespace_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="Node"):
        AerospikeByteStore(
            mock_session({field: value}), namespace="test_sc", set_name="unit"
        )


@pytest.mark.parametrize("build", ["7.2.0", "unknown"])
def test_unverified_build_rejected(build: str) -> None:
    session = mock_session()
    session.info.return_value.info_on_all_nodes.side_effect = [
        {"node": {"namespace/test_sc": ""}},
        {"node": {"build": build}},
    ]
    with pytest.raises(ValueError, match="operational SC/MRT"):
        namespace_record_limit(session, "test_sc")


def test_all_nodes_must_report_capability() -> None:
    session = mock_session()
    session.info.return_value.info_on_all_nodes.side_effect = [
        {"node": {}},
        {"node": {"build": "8.1.3"}, "other": {"build": "8.1.3"}},
    ]
    with pytest.raises(ValueError, match="every cluster node"):
        namespace_record_limit(session, "test_sc")


@pytest.mark.parametrize("namespace", ["", "x;y", "x/y", "x\n", "\0", "x=y"])
def test_namespace_delimiters_rejected(namespace: str) -> None:
    with pytest.raises(ValueError, match=r"nonempty|delimiter"):
        validate_names(namespace, "unit")


def test_constructor_retains_caller_session_without_connect_or_close() -> None:
    session = mock_session()
    store = AerospikeByteStore(session, namespace="test_sc", set_name="unit")
    assert store._session is session
    assert store._record_limit == 1_048_576
    session.info.assert_called_once_with()
    assert all(call[0].startswith("info") for call in session.mock_calls)


def test_configuration_types() -> None:
    with pytest.raises(TypeError):
        validate_names(cast("str", 1), "unit")
    with pytest.raises(ValueError, match="nonempty"):
        validate_names("test_sc", "")
    with pytest.raises(TypeError, match="TransactionalSession"):
        AerospikeByteStore(
            cast("Session", TransactionalSession.__new__(TransactionalSession)),
            namespace="test_sc",
            set_name="unit",
        )
