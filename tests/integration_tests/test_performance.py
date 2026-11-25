from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aerospike_sdk import RecordExistsError
from aerospike_sdk.sync import TransactionalSession

from langchain_aerospike_kv import AerospikeByteStore
from langchain_aerospike_kv._schema import CHUNK_SIZE, MAX_VALUE_SIZE, root_id
from tests.integration_tests.test_transactions import (
    AbortingStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from aerospike_sdk.sync import Session

    from langchain_aerospike_kv._schema import Root
    from tests._server import LiveStore

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("deleting", [False, True])
def test_inline_aba_race_cannot_orphan_new_chunks(
    live_store: LiveStore, monkeypatch: pytest.MonkeyPatch, *, deleting: bool
) -> None:
    store = live_store.store
    store.mset([("key", b"old")])
    identifier = live_store.dataset.id(root_id("key"))
    before = live_store.session.get(identifier)
    original = store._write_inline_batch
    raced = False

    def mutate(native: Sequence[tuple[str, bytes | None]]) -> set[int]:
        nonlocal raced
        if native and not raced:
            raced = True
            live_store.remove(root_id("key"))
            store.mset([("key", b"x" * 131_073)])
            replacement = live_store.session.get(identifier)
            # SC durable deletion advances generations. Drive the real counter
            # through wrap to exercise ABA, without fabricating read results.
            for _ in range(2_048):
                if replacement.generation == before.generation:
                    break
                live_store.session.put(identifier, replacement.bins)
                replacement = live_store.session.get(identifier)
            else:
                pytest.fail("SC generation did not wrap within the expected bound.")
        return original(native)

    monkeypatch.setattr(store, "_write_inline_batch", mutate)
    if deleting:
        store.mdelete(["key"])
        assert store.mget(["key"]) == [None]
        assert live_store.rows() == []
    else:
        store.mset([("key", b"winner")])
        assert store.mget(["key"]) == [b"winner"]
        assert len(live_store.rows()) == 1
    assert raced


def test_create_race_falls_back_without_orphaning_chunks(
    live_store: LiveStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = live_store.store
    original = store._write_inline_batch
    raced = False

    def mutate(native: Sequence[tuple[str, bytes | None]]) -> set[int]:
        nonlocal raced
        if native and not raced:
            raced = True
            store.mset([("key", b"x" * 131_073)])
        return original(native)

    monkeypatch.setattr(store, "_write_inline_batch", mutate)
    store.mset([("key", b"winner")])
    assert raced
    assert store.mget(["key"]) == [b"winner"]
    assert len(live_store.rows()) == 1


@pytest.mark.parametrize("deleting", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_larger_inline_group_does_not_enter_transaction(
    live_store: LiveStore, *, deleting: bool, existing: bool
) -> None:
    keys = [str(i) for i in range(32)]
    if existing:
        live_store.store.mset([(key, b"old") for key in keys])
    aborting = live_store.new_store(AbortingStore)
    action: Callable[[], None] = (
        (lambda: aborting.mdelete(keys))
        if deleting
        else (lambda: aborting.mset([(key, b"new") for key in keys]))
    )
    action()
    assert aborting.writes_completed == 0
    assert live_store.store.mget(keys) == (
        [None] * len(keys) if deleting else [b"new"] * len(keys)
    )
    assert len(live_store.rows()) == (0 if deleting else len(keys))


class GrowingBudgetStore(AerospikeByteStore):
    rival: AerospikeByteStore
    grew = False
    group_sizes: list[int]

    def _roots(self, session: Session, keys: Sequence[str]) -> list[Root | None]:
        roots = super()._roots(session, keys)
        if isinstance(session, TransactionalSession) and not self.grew:
            self.grew = True
            payload = b"x" * MAX_VALUE_SIZE
            self.rival.mset([(key, payload) for key in keys])
        return roots

    def _replace_group(
        self, tx: TransactionalSession, *, group: Sequence[tuple[str, bytes]]
    ) -> None:
        self.group_sizes.append(len(group))
        super()._replace_group(tx, group=group)

    def _delete_group(self, tx: TransactionalSession, *, keys: Sequence[str]) -> None:
        self.group_sizes.append(len(keys))
        super()._delete_group(tx, keys=keys)


@pytest.mark.parametrize("deleting", [False, True])
def test_concurrent_growth_rebudgets_before_chunk_cleanup(
    live_store: LiveStore, *, deleting: bool
) -> None:
    keys = [str(i) for i in range(32)]
    live_store.store.mset([(key, b"o" * (CHUNK_SIZE + 1)) for key in keys])
    racing = live_store.new_store(GrowingBudgetStore)
    racing.rival = live_store.store
    racing.group_sizes = []
    if deleting:
        racing.mdelete(keys)
        assert live_store.store.mget(keys) == [None] * len(keys)
        assert live_store.rows() == []
    else:
        racing.mset([(key, b"winner") for key in keys])
        assert live_store.store.mget(keys) == [b"winner"] * len(keys)
        assert len(live_store.rows()) == len(keys)
    assert racing.grew
    assert racing.group_sizes == [32, 32, 16, 16]


class MissingRootRaceStore(AerospikeByteStore):
    rival: AerospikeByteStore
    raced = False

    def _roots(self, session: Session, keys: Sequence[str]) -> list[Root | None]:
        roots = super()._roots(session, keys)
        if isinstance(session, TransactionalSession) and not self.raced:
            self.raced = True
            assert roots[0] is None
            self.rival.mset([("key", b"r" * (CHUNK_SIZE * 3))])
        return roots


@pytest.mark.parametrize(
    ("replacement", "extra_key"),
    [
        (b"inline", True),
        (b"x" * (CHUNK_SIZE + 1), True),
        (b"x" * (CHUNK_SIZE + 1), False),
    ],
)
def test_transactional_missing_root_create_race_cannot_orphan_chunks(
    live_store: LiveStore, replacement: bytes, *, extra_key: bool
) -> None:
    racing = live_store.new_store(MissingRootRaceStore)
    racing.rival = live_store.store
    pairs = [("key", replacement)]
    if extra_key:
        pairs.append(("other", b"other"))
    try:
        racing._set_group(pairs)
    except RecordExistsError:
        assert racing.raced
        assert live_store.store.mget(["key", "other"]) == [
            b"r" * (CHUNK_SIZE * 3),
            None,
        ]
        assert len(live_store.rows()) == 4
        return
    assert racing.raced
    assert live_store.store.mget([key for key, _ in pairs]) == [
        value for _, value in pairs
    ]
    assert len(live_store.rows()) == (
        (1 if len(replacement) <= CHUNK_SIZE else 3) + int(extra_key)
    )
