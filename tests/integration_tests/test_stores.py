from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from aerospike_sdk import AerospikeError, ResultCode
from aerospike_sdk.sync import ClusterDefinition
from langchain_classic.storage import create_kv_docstore
from langchain_core.documents import Document

from langchain_aerospike_kv import AerospikeByteStore
from langchain_aerospike_kv._schema import (
    CHUNK_SIZE,
    MAX_VALUE_SIZE,
    chunk_ids,
    root_id,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from aerospike_sdk.sync import TransactionalSession

    from tests._server import LiveStore, ServerSettings

pytestmark = pytest.mark.integration

LITERAL_KEYS = [
    "",
    "a",
    "a\0",
    "a\0b",
    "a\0c",
    "ab",
    "a[*]",
    "a.*$",
    "a\\x",
    "\0",
    "\0x",
    "é",
    "e\u0301",
    "é☃",
    "🙂",
    "\ud7ff",
    "\ue000",
    "\uffff",
    "\U00010000",
    "\U0010ffff",
    "\U0010ffffx",
    "\U0010ffff\U0010ffff",
    "a\U0010ffff",
    "a\U0010ffffx",
]


class RecordingStore(AerospikeByteStore):
    groups: list[list[tuple[str, int]]]

    def _replace_group(
        self, tx: TransactionalSession, *, group: Sequence[tuple[str, bytes]]
    ) -> None:
        assert tx.active
        self.groups.append([(key, len(value)) for key, value in group])
        super()._replace_group(tx, group=group)


def test_maximum_value_and_prewrite_oversize_rejection(live_store: LiveStore) -> None:
    store = live_store.store
    maximum = bytes(range(256)) * (8 * 1024 * 1024 // 256)
    store.mset([("max", maximum)])
    assert store.mget(["max", "missing", "max"]) == [maximum, None, maximum]
    rows = live_store.rows()
    assert len(rows) == 129
    assert sum(row.record_or_raise().bins["kind"] == 1 for row in rows) == 128
    original = live_store.snapshot()
    # The invalid entry is beyond the first key-count group. No earlier group
    # may land, and the old maximum-size value must remain byte-for-byte intact.
    pairs = [(str(index), b"ok") for index in range(129)]
    with pytest.raises(ValueError, match="cannot exceed"):
        store.mset([*pairs, ("max", maximum + b"x")])
    assert live_store.snapshot() == original
    store.mset([("max", b"")])
    assert store.mget(["max"]) == [b""]
    assert len(live_store.rows()) == 1
    store.mdelete(["max"])
    assert live_store.rows() == []


def test_multiple_key_and_payload_bounded_groups(live_store: LiveStore) -> None:
    store = live_store.new_store(RecordingStore)
    store.groups = []
    small = [(f"k{index}", bytes([index])) for index in range(35)]
    store.mset(small)
    assert store.groups == []  # Inline values no longer open an MRT.
    requested = ["k34", "missing", "k0", "k34", *reversed([k for k, _ in small])]
    expected = dict(small)
    assert store.mget(requested) == [expected.get(key) for key in requested]
    store.groups = []
    # Two 4-MiB values fill the byte budget exactly, a third forces a new MRT
    # even though the key-count limit is nowhere near reached.
    large = [(f"large{i}", bytes([i]) * (MAX_VALUE_SIZE // 2)) for i in range(3)]
    store.mset([*large, ("tail", b"t")])
    assert [sum(size for _, size in group) for group in store.groups] == [
        MAX_VALUE_SIZE,
        MAX_VALUE_SIZE // 2,
    ]
    assert store.mget([key for key, _ in large]) == [value for _, value in large]
    store.mdelete([key for key, _ in small + large] + ["missing", "tail", "k0"])
    assert live_store.rows() == []


def test_duplicates_empty_and_binary(live_store: LiveStore) -> None:
    store = live_store.store
    store.mset([])
    store.mdelete([])
    assert store.mget([]) == []
    assert list(store.yield_keys()) == []
    binary = bytes(range(256)) * 513
    store.mset([("", b"first"), ("", binary), ("empty", b"")])
    assert store.mget(["", "missing", "empty", "", "missing"]) == [
        binary,
        None,
        b"",
        binary,
        None,
    ]
    assert set(store.yield_keys()) == {"", "empty"}
    store.mdelete(["", "", "empty", "absent"])
    store.mdelete(["", "absent"])
    assert live_store.rows() == []


@pytest.mark.parametrize(
    "prefix",
    [
        None,
        "",
        "a",
        "a\0",
        "a\0b",
        "a[",
        "a.*",
        "a\\",
        "\0",
        "é",
        "e\u0301",
        "🙂",
        "\ud7ff",
        "\ue000",
        "\uffff",
        "\U00010000",
        "\U0010ffff",
        "\U0010ffff\U0010ffff",
        "a\U0010ffff",
        "not present",
    ],
)
def test_literal_unicode_prefixes(live_store: LiveStore, prefix: str | None) -> None:
    store = live_store.store
    store.mset([(key, key.encode()) for key in LITERAL_KEYS])
    # Include chunks in the queried set: only roots may appear in enumeration.
    store.mset([("a chunked", b"x" * (CHUNK_SIZE + 1))])
    keys = [*LITERAL_KEYS, "a chunked"]
    assert store.mget(LITERAL_KEYS) == [key.encode() for key in LITERAL_KEYS]
    assert set(store.yield_keys(prefix=prefix)) == {
        key for key in keys if prefix is None or key.startswith(prefix)
    }


def test_grow_shrink_delete_numeric_order_and_ttl(live_store: LiveStore) -> None:
    store = live_store.store
    values = [
        b"",
        b"\0\xff",
        b"a" * CHUNK_SIZE,
        b"b" * (CHUNK_SIZE + 1),
        b"".join(bytes([index]) * CHUNK_SIZE for index in range(13)),
        b"c" * (CHUNK_SIZE * 3 + 9),
        b"small",
    ]
    for value in values:
        store.mset([("key", value)])
        assert store.mget(["key"]) == [value]
        rows = live_store.rows()
        chunks = (
            0
            if len(value) <= CHUNK_SIZE
            else (len(value) + CHUNK_SIZE - 1) // CHUNK_SIZE
        )
        assert len(rows) == 1 + chunks
        assert {row.key.digest for row in rows} == {
            live_store.dataset.id(identifier).digest
            for identifier in [root_id("key"), *chunk_ids("key", 0, chunks)]
        }
        # The public SDK maps the never-expire sentinel to None.
        assert all(row.record_or_raise().ttl is None for row in rows)
    store.mset([("key", values[4])])
    store.mdelete(["key", "missing", "key"])
    store.mdelete(["key"])
    assert live_store.rows() == []


def test_isolation_sharing_and_caller_session_lifecycle(
    live_store: LiveStore, store_factory: Callable[[], LiveStore]
) -> None:
    other = store_factory()
    live_store.store.mset([("key", b"first")])
    other.store.mset([("key", b"second")])
    shared = live_store.new_store(AerospikeByteStore)
    assert shared.mget(["key"]) == [b"first"]
    live_store.store.mdelete(["key"])
    assert shared.mget(["key"]) == [None]
    assert other.store.mget(["key"]) == [b"second"]
    del shared
    # The store must never close or replace its caller's shared session.
    live_store.store.mset([("key", b"still connected")])
    assert live_store.session.info().build()


def test_real_document_adapter_sync(live_store: LiveStore) -> None:
    store = create_kv_docstore(live_store.store)
    first = Document(page_content="Unicode ☃\0", metadata={"source": "real SDK"})
    # Exercise chunking through actual host serialization, not a toy adapter.
    second = Document(page_content="large" * CHUNK_SIZE, metadata={"index": 2})
    store.mset([("doc/1", first)])
    store.mset([("doc/2", second)])
    assert store.mget(["doc/2", "absent", "doc/1"]) == [second, None, first]
    assert set(store.yield_keys(prefix="doc/")) == {"doc/1", "doc/2"}
    store.mdelete(["doc/1"])
    store.mdelete(["doc/2"])
    assert live_store.rows() == []


async def test_real_document_adapter_async(live_store: LiveStore) -> None:
    store = create_kv_docstore(live_store.store)
    document = Document(page_content="large ☃\0" * CHUNK_SIZE, metadata={"index": 2})
    await store.amset([("doc/1", document)])
    assert await store.amget(["doc/1", "absent", "doc/1"]) == [document, None, document]
    assert {key async for key in store.ayield_keys(prefix="doc/")} == {"doc/1"}
    await store.amdelete(["doc/1"])
    assert await store.amget(["doc/1"]) == [None]


@pytest.mark.parametrize(
    "invalid",
    [
        [("valid", b"x"), ("\ud800", b"bad")],
        [("duplicate", bytearray(b"bad")), ("duplicate", b"good")],
        [("valid", b"x"), (42, b"bad")],
    ],
)
def test_invalid_inputs_never_write(live_store: LiveStore, invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        live_store.store.mset(cast("list[tuple[str, bytes]]", invalid))
    assert live_store.rows() == []


def test_missing_namespace_is_not_a_cache_miss(live_store: LiveStore) -> None:
    with pytest.raises(ValueError, match="SC/MRT"):
        AerospikeByteStore(
            live_store.session, namespace="lc_c_absent", set_name=live_store.set_name
        )


def test_closed_caller_cluster_is_an_error(
    live_store: LiveStore, server_settings: ServerSettings
) -> None:
    cluster = ClusterDefinition(server_settings.host, server_settings.port).connect()
    try:
        store = AerospikeByteStore(
            cluster.create_session(),
            namespace=live_store.namespace,
            set_name=live_store.set_name,
        )
    finally:
        cluster.close()
    with pytest.raises(AerospikeError) as failure:
        store.mget(["absent"])
    assert failure.value.result_code != ResultCode.KEY_NOT_FOUND_ERROR
