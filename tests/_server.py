"""Real-server test support; no connection configuration belongs in the library."""

from __future__ import annotations

import errno
import os
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar, cast
from uuid import uuid4

from aerospike_sdk import DataSet, ErrorStrategy

from langchain_aerospike_kv import AerospikeByteStore

if TYPE_CHECKING:
    from aerospike_sdk import RecordResult
    from aerospike_sdk.sync import Session

    from langchain_aerospike_kv._schema import Bins

StoreT = TypeVar("StoreT", bound=AerospikeByteStore)
UNREACHABLE = {
    errno.ECONNREFUSED,
    errno.ETIMEDOUT,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
}


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 3000
    namespace: str = "test_sc"

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            msg = "AEROSPIKE_PORT must be in 1..65535."
            raise ValueError(msg)

    @classmethod
    def from_env(cls) -> ServerSettings:
        return cls(
            host=os.environ.get("AEROSPIKE_HOST", "127.0.0.1"),
            port=int(os.environ.get("AEROSPIKE_PORT", "3000")),
            namespace=os.environ.get("AEROSPIKE_NAMESPACE", "test_sc"),
        )


def reachable(settings: ServerSettings) -> bool:
    """Only actual transport unreachability permits skipping a live test."""
    try:
        with socket.create_connection((settings.host, settings.port), timeout=1):
            return True
    except TimeoutError:
        return False
    except OSError as exc:
        if exc.errno in UNREACHABLE:
            return False
        raise


@dataclass
class LiveStore:
    """An exclusively owned set, including raw inspection/repair/cleanup access."""

    session: Session
    namespace: str
    set_name: str = field(default_factory=lambda: f"lc_c_{uuid4().hex}")
    store: AerospikeByteStore = field(init=False)

    def __post_init__(self) -> None:
        self.store = self.new_store(AerospikeByteStore)

    @property
    def dataset(self) -> DataSet:
        return DataSet.of(self.namespace, self.set_name)

    def new_store(self, cls: type[StoreT]) -> StoreT:
        return cls(self.session, namespace=self.namespace, set_name=self.set_name)

    def rows(self) -> list[RecordResult]:
        stream = self.session.query(self.dataset).execute(
            on_error=ErrorStrategy.IN_STREAM
        )
        try:
            rows = list(stream)
            for row in rows:
                row.record_or_raise()
            return rows
        finally:
            stream.close()

    def snapshot(self) -> dict[bytes, Bins]:
        return {
            row.key.digest: cast("Bins", row.record_or_raise().bins)
            for row in self.rows()
        }

    def put(self, identifier: str, bins: Bins) -> None:
        stream = (
            self.session.replace(self.dataset.id(identifier))
            .put(bins)
            .never_expire()
            .execute()
        )
        try:
            for row in stream:
                row.or_raise()
        finally:
            stream.close()

    def remove(self, identifier: str) -> None:
        stream = (
            self.session.delete(self.dataset.id(identifier))
            .with_durable_delete()
            .with_txn(None)
            .execute()
        )
        try:
            for row in stream:
                row.or_raise()
        finally:
            stream.close()

    def clear(self) -> None:
        # Raw digest keys also clean intentionally malformed roots and orphan
        # fragments. Never truncate, and never query/delete another test's set.
        keys = [row.key for row in self.rows()]
        if keys:
            stream = (
                self.session.delete(keys)
                .with_durable_delete()
                .with_txn(None)
                .execute(on_error=ErrorStrategy.IN_STREAM)
            )
            try:
                for row in stream:
                    row.or_raise()
            finally:
                stream.close()
        assert self.rows() == []
