from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

import pytest
from aerospike_sdk import CommitError, RecordResult, ResultCode, TransactionError

from langchain_aerospike_kv import AerospikeByteStore
from langchain_aerospike_kv._io import check_write_results
from langchain_aerospike_kv._schema import CHUNK_SIZE, root_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aerospike_sdk.sync import Session, TransactionalSession

    from langchain_aerospike_kv._schema import Root
    from tests._server import LiveStore

pytestmark = pytest.mark.integration


class InjectedFailureError(Exception):
    """Application callback failure, injected only AFTER real transaction writes."""


class AbortingStore(AerospikeByteStore):
    writes_completed = 0

    def _replace_group(
        self, tx: TransactionalSession, *, group: Sequence[tuple[str, bytes]]
    ) -> None:
        super()._replace_group(tx, group=group)
        assert tx.active
        self.writes_completed += 1
        raise InjectedFailureError

    def _delete_group(self, tx: TransactionalSession, *, keys: Sequence[str]) -> None:
        super()._delete_group(tx, keys=keys)
        assert tx.active
        self.writes_completed += 1
        raise InjectedFailureError


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (b"old", b"g" * (CHUNK_SIZE * 13 + 1)),
        (b"s" * (CHUNK_SIZE * 13 + 1), b"small"),
        (b"d" * (CHUNK_SIZE * 13 + 1), None),
    ],
)
def test_exception_after_real_writes_aborts_entire_group(
    live_store: LiveStore, old: bytes, new: bytes | None
) -> None:
    other = b"o" * (CHUNK_SIZE + 1)
    live_store.store.mset([("key", old), ("other", other)])
    before = live_store.snapshot()
    aborting = live_store.new_store(AbortingStore)
    if new is None:
        with pytest.raises(InjectedFailureError):
            aborting.mdelete(["key", "other"])
    else:
        with pytest.raises(InjectedFailureError):
            aborting.mset([("key", new), ("other", other + b"changed")])
    assert aborting.writes_completed == 1
    assert live_store.store.mget(["key", "other"]) == [old, other]
    assert live_store.snapshot() == before
    # A successful subsequent write proves the abort released record locks.
    live_store.store.mset([("key", b"after-abort")])
    assert live_store.store.mget(["key"]) == [b"after-abort"]


class CommitConflictStore(AerospikeByteStore):
    rival: AerospikeByteStore
    attempts = 0

    def _replace_group(
        self, tx: TransactionalSession, *, group: Sequence[tuple[str, bytes]]
    ) -> None:
        self.attempts += 1
        tx.get(self._dataset.id(root_id("guard")))  # Real MRT read set.
        super()._replace_group(tx, group=group)
        # Independent, real committed write after this MRT's writes. On callback
        # return, server commit verification MUST fail on the stale guard read.
        self.rival.mset([("guard", str(self.attempts).encode())])


def test_failed_server_commit_rolls_back_all_attempts(live_store: LiveStore) -> None:
    old = b"o" * (CHUNK_SIZE * 4 + 1)
    live_store.store.mset([("key", old), ("guard", b"initial")])
    conflict = live_store.new_store(CommitConflictStore)
    conflict.rival = live_store.store
    before = live_store.snapshot()
    with pytest.raises(CommitError) as failure:
        conflict.mset([("key", b"must roll back")])
    assert conflict.attempts == 5
    assert failure.value.result_code == ResultCode.MRT_VERSION_MISMATCH
    assert not failure.value.in_doubt
    assert live_store.store.mget(["key", "guard"]) == [old, b"5"]
    guard_digest = live_store.dataset.id(root_id("guard")).digest
    assert {
        key: bins for key, bins in live_store.snapshot().items() if key != guard_digest
    } == {key: bins for key, bins in before.items() if key != guard_digest}
    live_store.store.mdelete(["key"])
    assert live_store.store.mget(["key"]) == [None]


def test_real_per_row_error_aborts_successful_rows(
    live_store: LiveStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = live_store.store
    store.mset([("first", b"old-first"), ("locked", b"old-locked")])
    before = live_store.snapshot()
    observed: list[ResultCode] = []
    original = RecordResult.or_raise

    def observe(row: RecordResult) -> RecordResult:
        # Observe, then invoke the unmodified SDK assertion. No fake result,
        # protocol substitution, or replacement transaction is involved.
        observed.append(row.result_code)
        return original(row)

    locked_key = live_store.dataset.id(root_id("locked"))
    locked_bins = live_store.session.get(locked_key).bins
    with live_store.session.transaction() as held:
        check_write_results(
            held.replace(locked_key).put(locked_bins).never_expire().execute(), 1
        )
        with monkeypatch.context() as patch:
            patch.setattr(RecordResult, "or_raise", observe)
            with pytest.raises(TransactionError) as failure:
                store.mset(
                    [
                        ("first", b"f" * (CHUNK_SIZE * 2 + 1)),
                        ("locked", b"l" * (CHUNK_SIZE + 1)),
                    ]
                )
        assert failure.value.result_code == ResultCode.MRT_BLOCKED
        assert not failure.value.in_doubt
        assert ResultCode.OK in observed
        assert observed.count(ResultCode.MRT_BLOCKED) == 5
        assert store.mget(["first", "locked"]) == [b"old-first", b"old-locked"]
        assert live_store.snapshot() == before
        held.rollback()
    store.mset([("locked", b"released")])
    assert store.mget(["locked"]) == [b"released"]


class PausedReader(AerospikeByteStore):
    roots_read: Event
    resume: Event
    reads = 0

    def _roots(self, session: Session, keys: Sequence[str]) -> list[Root | None]:
        roots = super()._roots(session, keys)
        self.reads += 1
        if self.reads == 1:
            self.roots_read.set()
            assert self.resume.wait(timeout=15), "Writer failed to release reader."
        return roots


@pytest.mark.parametrize("replacement", [b"inline", b"n" * (CHUNK_SIZE * 2 + 7), None])
def test_synchronized_reader_writer_never_returns_torn_value(
    live_store: LiveStore, replacement: bytes | None
) -> None:
    store = live_store.store
    store.mset([("key", b"o" * (CHUNK_SIZE * 13 + 1))])
    reader = live_store.new_store(PausedReader)
    reader.roots_read, reader.resume = Event(), Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(reader.mget, ["key", "key", "missing"])
        try:
            assert reader.roots_read.wait(timeout=15), "Reader never fetched roots."
            if replacement is None:
                store.mdelete(["key"])
            else:
                store.mset([("key", replacement)])
        finally:
            reader.resume.set()
        assert future.result(timeout=15) == [replacement, replacement, None]
    assert reader.reads == 2  # The stale root MUST be retried.
    assert store.mget(["key"]) == [replacement]


class CleanupRaceStore(AerospikeByteStore):
    rival: AerospikeByteStore
    attempts = 0

    def _verify_cleanup(
        self, tx: Session, roots_and_ids: Sequence[tuple[Root, list[str]]]
    ) -> list[str]:
        self.attempts += 1
        if self.attempts == 1:
            # Deterministic replacement between the real transactional root and
            # chunk-metadata reads, not a scheduler-dependent stress race.
            self.rival.mset([("key", b"rival")])
        return super()._verify_cleanup(tx, roots_and_ids)


def test_cleanup_race_is_retryable_not_spurious_corruption(
    live_store: LiveStore,
) -> None:
    live_store.store.mset([("key", b"o" * (CHUNK_SIZE * 4))])
    racing = live_store.new_store(CleanupRaceStore)
    racing.rival = live_store.store
    racing.mset([("key", b"winner")])
    assert racing.attempts == 2
    assert live_store.store.mget(["key"]) == [b"winner"]
    assert len(live_store.rows()) == 1
