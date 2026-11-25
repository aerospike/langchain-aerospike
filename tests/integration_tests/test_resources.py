from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aerospike_sdk.sync.record_stream import RecordStream

from langchain_aerospike_kv._schema import CHUNK_SIZE

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests._server import LiveStore

pytestmark = pytest.mark.integration


def test_early_enumeration_closes_actual_stream(
    live_store: LiveStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_store.store.mset([(str(index), b"value") for index in range(35)])
    closed: list[RecordStream] = []
    original = RecordStream.close

    def observe(stream: RecordStream) -> None:
        closed.append(stream)
        original(stream)

    # Monitor the real SDK close, without replacing query execution or rows.
    with monkeypatch.context() as patch:
        patch.setattr(RecordStream, "close", observe)
        iterator = live_store.store.yield_keys()
        assert next(iterator) in {str(index) for index in range(35)}
        previously_closed = len(closed)
        iterator.close()
        assert len(closed) > previously_closed
        assert list(iterator) == []
    assert len(list(live_store.store.yield_keys())) == 35
    assert live_store.session.info().build()


def test_raw_cleanup_handles_corrupt_and_orphan_records(
    store_factory: Callable[[], LiveStore],
) -> None:
    first, second = store_factory(), store_factory()
    first.store.mset([("key", b"x" * (CHUNK_SIZE + 1))])
    first.put("deliberately-malformed", {"fmt": "wrong"})
    second.store.mset([("key", b"unrelated")])
    first.clear()
    assert first.rows() == []
    assert second.store.mget(["key"]) == [b"unrelated"]
    # Repeated teardown must also handle empty/tombstoned sets safely.
    first.clear()
