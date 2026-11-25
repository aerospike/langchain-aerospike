from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING

import pytest
from aerospike_sdk import TransactionError

from langchain_aerospike_kv import AerospikeByteStore
from langchain_aerospike_kv._io import check_write_results
from langchain_aerospike_kv._schema import CHUNK_SIZE, root_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests._server import LiveStore

pytestmark = pytest.mark.integration


class BatchRaceStore(AerospikeByteStore):
    rival: AerospikeByteStore
    raced = False
    recreate = False
    retry_keys: list[str]

    def _write_inline_batch(
        self, native: Sequence[tuple[str, bytes | None]]
    ) -> set[int]:
        if native and not self.raced:
            self.raced = True
            if self.recreate:
                self.rival.mdelete(["key"])
            self.rival.mset([("key", b"r" * (CHUNK_SIZE * 3))])
        return super()._write_inline_batch(native)

    def _set_group(self, group: Sequence[tuple[str, bytes]]) -> None:
        self.retry_keys.extend(key for key, _ in group)
        # A successful native key must not be replayed over this later writer.
        self.rival.mset([("other", b"later")])
        super()._set_group(group)


@pytest.mark.parametrize("recreate", [False, True])
@pytest.mark.parametrize("deleting", [False, True])
def test_inline_batch_growth_races_never_orphan_chunks_or_replay_successes(
    live_store: LiveStore, *, deleting: bool, recreate: bool
) -> None:
    live_store.store.mset([("key", b"old"), ("other", b"old")])
    racing = live_store.new_store(BatchRaceStore)
    racing.rival = live_store.store
    racing.recreate = recreate
    racing.retry_keys = []
    if deleting:
        racing.mdelete(["key", "other"])
        assert live_store.store.mget(["key", "other"]) == [None, None]
        assert live_store.rows() == []
    else:
        racing.mset([("key", b"winner"), ("other", b"native")])
        assert racing.retry_keys == ["key"]
        assert live_store.store.mget(["key", "other"]) == [b"winner", b"later"]
        assert len(live_store.rows()) == 2
    assert racing.raced


def test_inline_batch_create_race_retries_only_the_conflicting_key(
    live_store: LiveStore,
) -> None:
    racing = live_store.new_store(BatchRaceStore)
    racing.rival = live_store.store
    racing.retry_keys = []
    racing.mset([("key", b"winner"), ("other", b"native")])
    assert racing.raced
    assert racing.retry_keys == ["key"]
    assert live_store.store.mget(["key", "other"]) == [b"winner", b"later"]
    assert len(live_store.rows()) == 2


@pytest.mark.parametrize("deleting", [False, True])
def test_inline_batch_failure_does_not_roll_back_other_successful_keys(
    live_store: LiveStore, *, deleting: bool
) -> None:
    store = live_store.store
    store.mset([("first", b"old"), ("locked", b"old")])
    locked = live_store.dataset.id(root_id("locked"))
    bins = live_store.session.get(locked).bins
    with live_store.session.transaction() as held:
        check_write_results(held.replace(locked).put(bins).never_expire().execute(), 1)
        action = (
            partial(store.mdelete, ["first", "locked"])
            if deleting
            else partial(store.mset, [("first", b"committed"), ("locked", b"blocked")])
        )
        with pytest.raises(TransactionError) as caught:
            action()
        assert not caught.value.in_doubt
        first = None if deleting else b"committed"
        assert store.mget(["first", "locked"]) == [first, b"old"]
        assert len(live_store.rows()) == (1 if deleting else 2)
        held.rollback()
    store.mset([("locked", b"after")])
    assert store.mget(["first", "locked"]) == [first, b"after"]


async def test_native_batches_keep_executor_async_contract(
    live_store: LiveStore,
) -> None:
    store = live_store.store
    await store.amset([("a", b"first"), ("b", b"second")])
    await store.amset([("a", b"replaced"), ("b", b"updated")])
    assert await store.amget(["b", "a", "b"]) == [b"updated", b"replaced", b"updated"]
    await store.amdelete(["a", "missing", "b"])
    assert await store.amget(["a", "b"]) == [None, None]
    assert await asyncio.to_thread(live_store.rows) == []
