from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from langchain_aerospike_kv._schema import CHUNK_SIZE, chunk_ids, root_id

if TYPE_CHECKING:
    from langchain_aerospike_kv._schema import Bins
    from tests._server import LiveStore

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fmt", 2),
        ("fmt", True),
        ("kind", "0"),
        ("kind", False),
        ("key", "wrong-key"),
        ("count", 1),
        ("count", False),
    ],
)
def test_invalid_root_schema_raises_not_a_miss(
    live_store: LiveStore, field: str, value: object
) -> None:
    store = live_store.store
    store.mset([("key", b"old")])
    bins: Bins = dict(
        live_store.session.get(live_store.dataset.id(root_id("key"))).bins
    )
    bins[field] = value
    live_store.put(root_id("key"), bins)
    before = live_store.snapshot()
    with pytest.raises(ValueError, match=r"[Bb]yte-store"):
        store.mget(["key"])
    with pytest.raises(ValueError, match=r"[Bb]yte-store"):
        store.mset([("key", b"must-not-land")])
    with pytest.raises(ValueError, match=r"[Bb]yte-store"):
        store.mdelete(["key"])
    assert live_store.snapshot() == before


@pytest.mark.parametrize("deleting", [False, True])
@pytest.mark.parametrize(
    ("field", "value"), [("data", "not a blob"), ("version", "not-a-uuid")]
)
def test_inline_payload_corruption_can_be_repaired_without_reading_old_value(
    live_store: LiveStore, field: str, value: object, *, deleting: bool
) -> None:
    store = live_store.store
    store.mset([("key", b"old")])
    bins: Bins = dict(
        live_store.session.get(live_store.dataset.id(root_id("key"))).bins
    )
    bins[field] = value
    live_store.put(root_id("key"), bins)
    with pytest.raises(ValueError, match=r"[Bb]yte-store"):
        store.mget(["key"])
    if deleting:
        store.mdelete(["key"])
        assert store.mget(["key"]) == [None]
        assert live_store.rows() == []
    else:
        store.mset([("key", b"repaired")])
        assert store.mget(["key"]) == [b"repaired"]
        assert len(live_store.rows()) == 1


@pytest.mark.parametrize("damage", ["missing", "version", "fmt", "short", "type"])
def test_stable_chunk_corruption_raises_and_does_not_publish(
    live_store: LiveStore, damage: str
) -> None:
    store = live_store.store
    store.mset([("key", b"x" * (CHUNK_SIZE * 3 + 7))])
    identifier = chunk_ids("key", 0, 1)[0]
    bins: Bins = dict(live_store.session.get(live_store.dataset.id(identifier)).bins)
    if damage == "missing":
        live_store.remove(identifier)
    else:
        field, value = {
            "version": ("version", uuid4().hex),
            "fmt": ("fmt", 2),
            "short": ("data", b"short"),
            "type": ("data", "not a blob"),
        }[damage]
        bins[field] = value
        live_store.put(identifier, bins)
    before = live_store.snapshot()
    with pytest.raises(ValueError, match="unchanged"):
        store.mget(["key", "absent"])
    if damage in {"missing", "version", "fmt"}:
        # Ownership validation in cleanup must prevent partial deletion or
        # replacement when root and fragment metadata disagree.
        with pytest.raises(ValueError, match="cleanup"):
            store.mset([("key", b"must-not-land")])
        with pytest.raises(ValueError, match="cleanup"):
            store.mdelete(["key"])
    assert live_store.snapshot() == before
