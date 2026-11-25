from __future__ import annotations

from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest

from langchain_aerospike_kv._schema import (
    CHUNK_SIZE,
    MAX_VALUE_SIZE,
    MIN_RECORD_SIZE,
    MUTATION_KEYS,
    chunk_ids,
    decode_chunks,
    decode_root,
    encode_value,
    key_hash,
    mutation_groups,
    prefix_successor,
    prepare_pairs,
    root_id,
)

if TYPE_CHECKING:
    from langchain_aerospike_kv._schema import Bins


@pytest.mark.parametrize("size", [0, 1, CHUNK_SIZE, CHUNK_SIZE + 1, MAX_VALUE_SIZE])
def test_encoding_boundaries_and_numeric_order(size: int) -> None:
    data = (bytes(range(256)) * (size // 256 + 1))[:size]
    bins, chunks = encode_value("☃\0x", data)
    root = decode_root(bins, "☃\0x")
    assert UUID(root.version).version == 4
    if size <= CHUNK_SIZE:
        assert not chunks
        assert root.count == 0
        assert root.data == data
    else:
        assert root.data == b""
        assert decode_chunks(root, chunks) == data
        assert root.count == len(chunks)
        identifiers = chunk_ids(root.key, 0, root.count)
        assert [
            int(identifier.rsplit(":", 1)[1]) for identifier in identifiers
        ] == list(range(root.count))


def test_stable_disjoint_physical_identity() -> None:
    assert key_hash("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert root_id("a").startswith("r:")
    assert len(root_id("a")) == 66
    assert chunk_ids("a", 10, 12) == [f"c:{key_hash('a')}:10", f"c:{key_hash('a')}:11"]
    assert key_hash("a\0") != key_hash("a")
    assert key_hash("é") != key_hash("e\u0301")
    assert (
        encode_value("a", b"x")[0]["version"] != encode_value("a", b"x")[0]["version"]
    )


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("", None),
        ("a", "b"),
        ("a\0", "a\1"),
        ("☃", "☄"),
        ("\ud7ff", "\ue000"),
        ("\U0010ffff", None),
        ("\U0010ffff\U0010ffff", None),
        ("a\U0010ffff", "b"),
        ("🙂", "🙃"),
        ("a[*]", "a[*^"),
    ],
)
def test_scalar_prefix_successor(prefix: str, expected: str | None) -> None:
    assert prefix_successor(prefix) == expected


@pytest.mark.parametrize("key", ["\ud800", "a\udfff"])
def test_surrogates_rejected(key: str) -> None:
    with pytest.raises(UnicodeEncodeError):
        key_hash(key)
    with pytest.raises(UnicodeEncodeError):
        prefix_successor(key)


def test_prepare_pairs_validates_shadowed_entries_and_last_wins() -> None:
    assert prepare_pairs([("k", b"a"), ("k", b"b")], MIN_RECORD_SIZE) == [("k", b"b")]
    with pytest.raises(TypeError, match="values must be bytes"):
        prepare_pairs([("k", cast("bytes", "invalid")), ("k", b"ok")], MIN_RECORD_SIZE)
    with pytest.raises(TypeError, match="keys must be strings"):
        prepare_pairs([(cast("str", 1), b"x")], MIN_RECORD_SIZE)
    with pytest.raises(ValueError, match="cannot exceed"):
        prepare_pairs([("k", b"x" * (MAX_VALUE_SIZE + 1))], MIN_RECORD_SIZE)
    with pytest.raises(ValueError, match="record-size budget"):
        prepare_pairs([("k" * MIN_RECORD_SIZE, b"")], MIN_RECORD_SIZE)
    # Chunked roots have no inline payload; budget their actual shape.
    assert prepare_pairs([("k" * CHUNK_SIZE, b"x" * (CHUNK_SIZE + 1))], MIN_RECORD_SIZE)


def test_transaction_groups_bound_keys_and_bytes() -> None:
    pairs = [(str(i), b"") for i in range(MUTATION_KEYS + 1)]
    assert [len(group) for group in mutation_groups(pairs)] == [MUTATION_KEYS, 1]
    assert list(mutation_groups([])) == []
    large = b"x" * MAX_VALUE_SIZE
    groups = list(mutation_groups([("a", large), ("b", b"x"), ("c", b"")]))
    assert groups == [[("a", large)], [("b", b"x"), ("c", b"")]]
    assert list(mutation_groups([("a", large), ("b", b"")])) == [
        [("a", large), ("b", b"")]
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", True),
        ("kind", 1),
        ("fmt", None),
        ("fmt", 2),
        ("key", "other"),
        ("key", 7),
        ("version", "bad"),
        ("version", "0" * 32),
        ("count", True),
        ("count", -1),
        ("count", 1),
        ("count", 129),
        ("data", None),
        ("data", bytearray(b"a")),
        ("data", b"a" * (CHUNK_SIZE + 1)),
    ],
)
def test_corrupt_roots_are_errors(field: str, value: object) -> None:
    bins, _ = encode_value("key", b"hello")
    bins[field] = value
    with pytest.raises(ValueError, match=r"(?i)byte-store"):
        decode_root(bins, "key")


@pytest.mark.parametrize("field", ["kind", "fmt", "key", "version", "count", "data"])
def test_missing_root_bins_are_errors(field: str) -> None:
    bins, _ = encode_value("key", b"")
    del bins[field]
    with pytest.raises(ValueError, match=r"(?i)byte-store"):
        decode_root(bins, "key")


@pytest.mark.parametrize(
    "failure", ["missing", "version", "fmt", "length", "empty", "type"]
)
def test_corrupt_chunks_are_errors(failure: str) -> None:
    bins, chunks = encode_value("k", b"x" * (CHUNK_SIZE + 1))
    root = decode_root(bins, "k")
    records: list[Bins | None] = list(chunks)
    if failure == "missing":
        records[0] = None
    elif failure == "version":
        chunks[0]["version"] = encode_value("k", b"x")[0]["version"]
    elif failure == "fmt":
        chunks[0]["fmt"] = 2
    elif failure == "length":
        chunks[0]["data"] = b"x"
    elif failure == "empty":
        chunks[-1]["data"] = b""
    else:
        chunks[-1]["data"] = "not bytes"
    with pytest.raises(ValueError, match=r"(?i)byte-store"):
        decode_chunks(root, records)
