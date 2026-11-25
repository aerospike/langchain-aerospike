from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from aerospike_sdk import InvalidNodeError, RecordNotFoundError

from tests import local_server
from tests._server import ServerSettings

ClientConnectionError = local_server.ClientConnectionError

if TYPE_CHECKING:
    from pathlib import Path


def test_stack_verification_rejects_wrong_sdk_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = MagicMock(return_value="main")
    monkeypatch.setattr(local_server, "run", run)
    with pytest.raises(RuntimeError, match="Required local SDK branch"):
        local_server.verify_stack(tmp_path)
    assert run.call_count == 1  # No fallback install or image pull.


def test_main_uses_adjacent_sdk_from_standalone_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify = MagicMock(side_effect=RuntimeError("stop before connecting"))
    monkeypatch.setattr(local_server, "verify_stack", verify)
    monkeypatch.setattr("sys.argv", ["tests.local_server"])
    with pytest.raises(RuntimeError, match="stop before connecting"):
        local_server.main()
    verify.assert_called_once_with(
        local_server.PACKAGE.parent / "aerospike-client-python-sdk"
    )


@pytest.mark.parametrize(
    "labels", [{}, None, {local_server.OWNER_LABEL: "another-owner"}]
)
def test_cleanup_never_removes_an_unrelated_container(
    monkeypatch: pytest.MonkeyPatch, labels: dict[str, str] | None
) -> None:
    inspect = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps([{"Id": "unrelated", "Config": {"Labels": labels}}]),
        )
    )
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", inspect)
    monkeypatch.setattr(local_server, "run", run)
    with pytest.raises(RuntimeError, match="ownership token"):
        local_server.remove_owned_container("anything", "ours")
    run.assert_not_called()


def test_owned_cleanup_uses_verified_immutable_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "Id": "owned-id",
                        "Config": {"Labels": {local_server.OWNER_LABEL: "ours"}},
                    }
                ]
            ),
        )
    )
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", inspect)
    monkeypatch.setattr(local_server, "run", run)
    local_server.remove_owned_container("friendly-name", "ours")
    assert [call.args[0] for call in run.call_args_list] == [
        ["docker", "stop", "--time", "10", "owned-id"],
        ["docker", "rm", "owned-id"],
    ]


def test_adoption_never_stops_or_rosters_existing_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = MagicMock()
    roster = MagicMock()
    monkeypatch.setattr(local_server, "run", run)
    monkeypatch.setattr(local_server, "initialize_roster", roster)
    monkeypatch.setattr(local_server, "reachable", lambda _: True)
    monkeypatch.setattr(local_server, "namespace_record_limit", lambda *_: 1_048_576)
    cluster = MagicMock()
    connect = MagicMock()
    connect.return_value.connect.return_value = cluster
    monkeypatch.setattr(local_server, "ClusterDefinition", connect)
    with local_server.server_lifecycle(ServerSettings(), created=False):
        assert local_server.connect_ready(ServerSettings(), created=False) is cluster
    roster.assert_not_called()
    run.assert_not_called()
    cluster.close.assert_not_called()


def test_owned_server_refuses_an_occupied_port(monkeypatch: pytest.MonkeyPatch) -> None:
    run = MagicMock()
    monkeypatch.setattr(local_server, "reachable", lambda _: True)
    monkeypatch.setattr(local_server, "run", run)
    with (
        pytest.raises(RuntimeError, match="occupied"),
        local_server.owned_server(ServerSettings()),
    ):
        pytest.fail("Must not replace a running server.")
    run.assert_not_called()


def test_child_failure_still_cleans_only_created_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = MagicMock()
    cleanup = MagicMock()
    monkeypatch.setattr(local_server, "reachable", lambda _: False)
    monkeypatch.setattr(local_server, "run", run)
    monkeypatch.setattr(local_server, "remove_owned_container", cleanup)
    message = "child failed"
    with (
        pytest.raises(RuntimeError, match="child failed"),
        local_server.owned_server(ServerSettings(port=3100)),
    ):
        raise RuntimeError(message)
    command = run.call_args_list[0].args[0]
    assert command[:4] == ["docker", "run", "-d", "--pull=never"]
    assert command[-1] == local_server.IMAGE
    assert "127.0.0.1:3100:3000" in command
    assert "nofile=65536:65536" in command
    assert not any("--device" in arg or "--privileged" in arg for arg in command)
    cleanup.assert_called_once()
    name, token = cleanup.call_args.args
    assert name == f"langchain-aero-c-{token}"
    assert f"{local_server.OWNER_LABEL}={token}" in command


@pytest.mark.parametrize("created", [True, False])
def test_tcp_ready_but_server_starting(
    monkeypatch: pytest.MonkeyPatch, *, created: bool
) -> None:
    cluster = MagicMock()
    definition = MagicMock()
    definition.return_value.connect.side_effect = [
        ClientConnectionError("server starting"),
        cluster,
    ]
    monkeypatch.setattr(local_server, "ClusterDefinition", definition)
    monkeypatch.setattr(local_server, "reachable", lambda _: True)
    sleep = MagicMock()
    monkeypatch.setattr("tests.local_server.time.sleep", sleep)
    if created:
        assert (
            local_server.connect_transport(
                ServerSettings(), created=True, deadline=float("inf")
            )
            is cluster
        )
        sleep.assert_called_once()
    else:
        with pytest.raises(ClientConnectionError):
            local_server.connect_transport(
                ServerSettings(), created=False, deadline=float("inf")
            )
        sleep.assert_not_called()


@pytest.mark.parametrize("created", [True, False])
def test_roster_ready_but_partition_map_still_stale(
    monkeypatch: pytest.MonkeyPatch, *, created: bool
) -> None:
    cluster = MagicMock()
    session = cluster.create_session.return_value
    session.get.side_effect = [
        InvalidNodeError("partition map is stale"),
        RecordNotFoundError("ready miss"),
    ]
    monkeypatch.setattr(
        local_server, "connect_transport", MagicMock(return_value=cluster)
    )
    monkeypatch.setattr(local_server, "initialize_roster", MagicMock())
    monkeypatch.setattr(local_server, "namespace_record_limit", MagicMock())
    sleep = MagicMock()
    monkeypatch.setattr("tests.local_server.time.sleep", sleep)
    if created:
        assert local_server.connect_ready(ServerSettings(), created=True) is cluster
        assert session.get.call_count == 2
        sleep.assert_called_once()
    else:
        with pytest.raises(InvalidNodeError):
            local_server.connect_ready(ServerSettings(), created=False)
        sleep.assert_not_called()
        cluster.close.assert_called_once()
