"""Development-only local test runner; adopts by default, never pulls an image."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from aerospike_sdk import AerospikeError, DataSet, RecordNotFoundError
from aerospike_sdk import ConnectionError as SdkConnectionError
from aerospike_sdk.sync import ClusterDefinition

from langchain_aerospike_kv._config import namespace_record_limit
from langchain_aerospike_kv._io import check_write_results
from langchain_aerospike_kv._schema import CHUNK_SIZE, root_id
from tests._server import LiveStore, ServerSettings, reachable

# SDK releases renamed their low-level client; local builds may use the old name.
try:
    from aerospike_native.exceptions import (
        ConnectionError as _ClientConnectionError,
    )
except ImportError:
    from aerospike_async.exceptions import (
        ConnectionError as _ClientConnectionError,
    )

if TYPE_CHECKING:
    from collections.abc import Iterator

    from aerospike_sdk.sync import Cluster, Session

ClientConnectionError = _ClientConnectionError
PACKAGE = Path(__file__).resolve().parents[1]
IMAGE = "aerospike-server:dev"
OWNER_LABEL = "langchain.aerospike-kv.test-owner"
logger = logging.getLogger(__name__)


class ProbeAbortError(Exception):
    """Deliberately abort a real MRT after its writes."""


def run(command: list[str], *, diagnostics: bool = False) -> str:
    """Run a fixed argument vector without a shell, surfacing failures."""
    result = subprocess.run(  # noqa: S603 - explicit local development commands
        command, check=True, capture_output=True, text=True
    )
    return (result.stdout + (result.stderr if diagnostics else "")).strip()


def verify_stack(sdk_path: Path) -> None:
    """Require the local SDK branch and already-present local server image."""
    branch = run(["git", "-C", str(sdk_path), "branch", "--show-current"])
    if branch != "aie/vector":
        msg = f"Required local SDK branch is aie/vector, found {branch!r}."
        raise RuntimeError(msg)
    image_id = run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])
    logger.info("Verified SDK branch %s, local image %s", branch, image_id)


def remove_owned_container(name: str, token: str) -> None:
    """Stop/remove only the immutable ID bearing this invocation's ownership token."""
    result = subprocess.run(  # noqa: S603 - fixed docker inspect, no shell
        ["docker", "container", "inspect", name],  # noqa: S607 - development tool
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return
    details = json.loads(result.stdout)[0]
    if (details["Config"]["Labels"] or {}).get(OWNER_LABEL) != token:
        msg = "Refusing cleanup: container ownership token does not match."
        raise RuntimeError(msg)
    identifier = details["Id"]
    run(["docker", "stop", "--time", "10", identifier])
    run(["docker", "rm", identifier])


@contextmanager
def owned_server(settings: ServerSettings) -> Iterator[None]:
    """Start an isolated file-backed local server and clean up only that server."""
    if settings.host != "127.0.0.1" or settings.namespace != "test_sc":
        msg = "--create requires 127.0.0.1 and test_sc; the host port may differ."
        raise ValueError(msg)
    if reachable(settings):
        msg = f"Port {settings.port} is occupied; omit --create to adopt that server."
        raise RuntimeError(msg)
    token = uuid4().hex
    name = f"langchain-aero-c-{token}"
    template = PACKAGE / "tests" / "integration_tests" / "aerospike.conf"
    with tempfile.TemporaryDirectory(prefix="lc-aero-c-") as directory:
        config = Path(directory) / "aerospike.conf"
        config.write_text(
            template.read_text().replace(
                "access-port 3000", f"access-port {settings.port}"
            )
        )
        try:
            run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--pull=never",
                    "--name",
                    name,
                    "--label",
                    f"{OWNER_LABEL}={token}",
                    "--ulimit",
                    "nofile=65536:65536",
                    "-p",
                    f"127.0.0.1:{settings.port}:3000",
                    "-v",
                    f"{config}:/etc/aerospike/aerospike.conf:ro",
                    IMAGE,
                ]
            )
            yield
        except BaseException:
            with suppress(subprocess.CalledProcessError):
                logger.exception(
                    "Created-server logs:\n%s",
                    run(["docker", "logs", name], diagnostics=True),
                )
            raise
        finally:
            # Also handles a failed docker run that allocated a stopped container.
            remove_owned_container(name, token)


def initialize_roster(session: Session, namespace: str) -> None:
    """Initialize ONLY a newly created isolated single-node server."""
    nodes = session.info().info("node")
    if len(nodes) != 1:
        msg = "Refusing to roster anything except the newly created single node."
        raise RuntimeError(msg)
    node = next(iter(nodes.values()))
    for command in (f"roster-set:namespace={namespace};nodes={node}", "recluster:"):
        response = session.info().info(command)
        if not response or any(value != "ok" for value in response.values()):
            msg = f"Local roster initialization failed: {response!r}"
            raise RuntimeError(msg)


def connect_transport(
    settings: ServerSettings, *, created: bool, deadline: float
) -> Cluster:
    """A published Docker TCP port is not yet an Aerospike-ready server."""
    while True:
        if reachable(settings):
            try:
                return ClusterDefinition(settings.host, settings.port).connect()
            except (SdkConnectionError, ClientConnectionError):
                if not created or time.monotonic() >= deadline:
                    raise
        elif not created or time.monotonic() >= deadline:
            msg = f"Aerospike is unreachable at {settings.host}:{settings.port}."
            raise RuntimeError(msg)
        time.sleep(0.1)


def connect_ready(settings: ServerSettings, *, created: bool) -> Cluster:
    """Poll actual readiness; never repair adopted servers."""
    deadline = time.monotonic() + 60
    cluster = connect_transport(settings, created=created, deadline=deadline)
    try:
        session = cluster.create_session()
        if created:
            # Administrative mutation is permitted ONLY on the new isolated
            # server created by this invocation, never on an adopted cluster.
            initialize_roster(session, settings.namespace)
        while True:
            try:
                namespace_record_limit(session, settings.namespace)
                # Info can report an active roster before the client's tend
                # thread has learned its new partition map. A real point read
                # (including a confirmed miss) proves data-plane readiness.
                with suppress(RecordNotFoundError):
                    session.get(
                        DataSet.of(settings.namespace, "lc_c_readiness").id(uuid4().hex)
                    )
            except (ValueError, AerospikeError):
                if not created or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
            else:
                return cluster
    except BaseException:
        cluster.close()
        raise


def capability_probe(session: Session, settings: ServerSettings) -> None:
    """Prove commit, complete reconstruction, abort and cleanup in an owned set."""
    live = LiveStore(session, settings.namespace)
    try:
        value = b"\0\xff" * CHUNK_SIZE
        live.store.mset([("chunked", value), ("inline", b"old")])
        if live.store.mget(["chunked", "inline"]) != [value, b"old"]:
            msg = "Real MRT commit/read probe failed."
            raise RuntimeError(msg)
        before = live.snapshot()
        try:
            with session.transaction() as tx:
                bins = session.get(live.dataset.id(root_id("inline"))).bins
                bins["data"] = b"must roll back"
                check_write_results(
                    tx.replace(live.dataset.id(root_id("inline")))
                    .put(bins)
                    .never_expire()
                    .execute(),
                    1,
                )
                raise ProbeAbortError  # noqa: TRY301 - intentional real transaction abort
        except ProbeAbortError:
            pass
        if live.snapshot() != before:
            msg = "Real MRT abort probe failed."
            raise RuntimeError(msg)
        live.store.mdelete(["chunked", "inline"])
        if live.rows():
            msg = "Real durable-delete cleanup probe failed."
            raise RuntimeError(msg)
        logger.info(
            "Verified SC/MRT commit/abort/cleanup; build=%s", session.info().build()
        )
    finally:
        live.clear()


def main() -> int:
    """Adopt an existing server or temporarily create one, then run strict tests."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true")
    parser.add_argument(
        "--sdk-path",
        type=Path,
        default=PACKAGE.parent / "aerospike-client-python-sdk",
    )
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    verify_stack(args.sdk_path)
    settings = ServerSettings.from_env()
    command = args.command or [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration_tests",
        "--require-live",
    ]
    with server_lifecycle(settings, created=args.create):
        with connect_ready(settings, created=args.create) as cluster:
            capability_probe(cluster.create_session(), settings)
        return subprocess.run(  # noqa: S603 - intentional caller-supplied test command
            command, cwd=PACKAGE, check=False
        ).returncode


@contextmanager
def server_lifecycle(settings: ServerSettings, *, created: bool) -> Iterator[None]:
    """Keep adopted resources alive and scope ownership of created resources."""
    if created:
        with owned_server(settings):
            yield
    else:
        yield


if __name__ == "__main__":
    raise SystemExit(main())
