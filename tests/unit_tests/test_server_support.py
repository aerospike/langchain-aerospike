from __future__ import annotations

import errno
import socket
from unittest.mock import MagicMock

import pytest

from tests._server import ServerSettings, reachable


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionRefusedError(errno.ECONNREFUSED, "refused"),
        TimeoutError(),
        OSError(errno.EHOSTUNREACH, "unreachable"),
    ],
)
def test_only_unreachable_transport_allows_skip(
    monkeypatch: pytest.MonkeyPatch, failure: OSError
) -> None:
    connect = MagicMock(side_effect=failure)
    monkeypatch.setattr(socket, "create_connection", connect)
    assert not reachable(ServerSettings())


def test_invalid_address_is_not_silently_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = MagicMock(side_effect=socket.gaierror("invalid hostname"))
    monkeypatch.setattr(socket, "create_connection", connect)
    with pytest.raises(socket.gaierror):
        reachable(ServerSettings())


def test_reachability_socket_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    connect = MagicMock()
    monkeypatch.setattr(socket, "create_connection", connect)
    assert reachable(ServerSettings())
    connect.return_value.__exit__.assert_called_once()


def test_test_layer_environment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEROSPIKE_HOST", "test-host")
    monkeypatch.setenv("AEROSPIKE_PORT", "3100")
    monkeypatch.setenv("AEROSPIKE_NAMESPACE", "another_sc")
    assert ServerSettings.from_env() == ServerSettings("test-host", 3100, "another_sc")
