"""Validate the caller's existing cluster, without owning its lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_aerospike_kv._schema import MIN_RECORD_SIZE

if TYPE_CHECKING:
    from aerospike_sdk.sync import Session

MRT_SERVER_MAJOR = 8


def validate_names(namespace: str, set_name: str) -> None:
    for name in (namespace, set_name):
        if not isinstance(name, str):
            msg = "Namespace and set_name must be strings."
            raise TypeError(msg)
        if not name or "\0" in name:
            msg = "Namespace and set_name must be nonempty and NUL-free."
            raise ValueError(msg)
        name.encode("utf-8")
    # The namespace is interpolated into read-only info commands.
    if any(character in namespace for character in ";:=/\r\n"):
        msg = "Namespace contains an info-command delimiter."
        raise ValueError(msg)


def namespace_record_limit(session: Session, namespace: str) -> int:
    """Fail closed on absent capability/configuration evidence on any node."""
    info = session.info()
    command = f"namespace/{namespace}"
    responses = info.info_on_all_nodes(command)
    builds = info.info_on_all_nodes("build")
    if not responses or responses.keys() != builds.keys():
        msg = "Cannot verify namespace and MRT capability on every cluster node."
        raise ValueError(msg)
    limits: list[int] = []
    for node, response in responses.items():
        fields = dict(
            part.split("=", 1)
            for part in response.get(command, "").split(";")
            if "=" in part
        )
        build = builds[node].get("build", "")
        major = build.split(".", 1)[0]
        if (
            not major.isdecimal()
            or int(major) < MRT_SERVER_MAJOR
            or fields.get("strong-consistency") != "true"
            or fields.get("disable-mrt-writes") != "false"
            or fields.get("unavailable_partitions") != "0"
            or fields.get("dead_partitions") != "0"
            or fields.get("stop_writes") != "false"
            or fields.get("clock_skew_stop_writes") != "false"
        ):
            msg = f"Node {node!r} requires an operational SC/MRT namespace."
            raise ValueError(msg)
        try:
            limit = int(fields["max-record-size"])
            duration = int(fields["mrt-duration"])
        except (KeyError, ValueError) as exc:
            msg = f"Node {node!r} does not report record-size/MRT limits."
            raise ValueError(msg) from exc
        if limit < MIN_RECORD_SIZE or duration <= 0:
            msg = f"Node {node!r} has insufficient record-size/MRT headroom."
            raise ValueError(msg)
        limits.append(limit)
    return min(limits)
