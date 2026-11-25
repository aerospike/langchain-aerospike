from __future__ import annotations

from inspect import unwrap
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from tests._server import ServerSettings
from tests.integration_tests import conftest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from aerospike_sdk.sync import Session


def session_fixture(config: pytest.Config) -> Iterator[Session]:
    factory = cast(
        "Callable[[ServerSettings, pytest.Config], Iterator[Session]]",
        unwrap(conftest.live_session),
    )
    return factory(ServerSettings(), config)


@pytest.mark.parametrize("required", [True, False])
def test_only_unreachable_server_can_skip(
    monkeypatch: pytest.MonkeyPatch, *, required: bool
) -> None:
    config = MagicMock(spec=pytest.Config)
    config.getoption.return_value = required
    monkeypatch.setattr(conftest, "reachable", lambda _: False)
    error = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(error, match="unreachable"):
        next(session_fixture(config))


@pytest.mark.parametrize("failure_point", ["connect", "capability"])
def test_reachable_but_invalid_environment_always_fails(
    monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    config = MagicMock(spec=pytest.Config)
    config.getoption.return_value = False
    monkeypatch.setattr(conftest, "reachable", lambda _: True)
    connection = MagicMock()
    validate = MagicMock()
    error = ValueError("reachable but invalid")
    if failure_point == "connect":
        connection.return_value.connect.side_effect = error
    else:
        validate.side_effect = error
    monkeypatch.setattr(conftest, "ClusterDefinition", connection)
    monkeypatch.setattr(conftest, "namespace_record_limit", validate)
    with pytest.raises(ValueError, match="reachable but invalid"):
        next(session_fixture(config))
