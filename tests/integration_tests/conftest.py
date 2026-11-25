from __future__ import annotations

import os
from contextlib import ExitStack
from typing import TYPE_CHECKING

import pytest
from aerospike_sdk.sync import ClusterDefinition

from langchain_aerospike_kv._config import namespace_record_limit
from tests._server import LiveStore, ServerSettings, reachable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from aerospike_sdk.sync import Session


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-live",
        action="store_true",
        default=os.environ.get("AEROSPIKE_REQUIRE_LIVE") == "1",
        help="Fail, rather than skip, when the real Aerospike server is unreachable.",
    )


@pytest.fixture(scope="session")
def server_settings() -> ServerSettings:
    return ServerSettings.from_env()


@pytest.fixture(scope="session")
def live_session(
    server_settings: ServerSettings, pytestconfig: pytest.Config
) -> Iterator[Session]:
    if not reachable(server_settings):
        message = (
            f"Aerospike unreachable at {server_settings.host}:{server_settings.port}."
        )
        if pytestconfig.getoption("--require-live"):
            pytest.fail(message)
        pytest.skip(message)
    # A reachable but invalid service, AP namespace, inactive roster, disabled
    # MRT, auth error, or SDK/implementation error MUST fail, not become a skip.
    with ClusterDefinition(
        server_settings.host, server_settings.port
    ).connect() as cluster:
        session = cluster.create_session()
        namespace_record_limit(session, server_settings.namespace)
        yield session


@pytest.fixture
def store_factory(
    live_session: Session, server_settings: ServerSettings
) -> Iterator[Callable[[], LiveStore]]:
    with ExitStack() as cleanup:

        def create() -> LiveStore:
            live = LiveStore(live_session, server_settings.namespace)
            cleanup.callback(live.clear)
            assert live.rows() == []  # Fresh and EMPTY for every contract test.
            return live

        yield create


@pytest.fixture
def live_store(store_factory: Callable[[], LiveStore]) -> LiveStore:
    return store_factory()
