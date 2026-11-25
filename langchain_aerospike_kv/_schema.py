"""Private, versioned byte-store record contract and bounded grouping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aerospike_sdk import Exp

CHUNK_SIZE = 65_536
MAX_CHUNKS = 128
MAX_VALUE_SIZE = CHUNK_SIZE * MAX_CHUNKS
GROUP_KEYS = 16
READ_KEYS = 128
MUTATION_KEYS = 128
TRANSACTION_RECORDS = 4_096
BATCH_RECORDS = 128
RECORD_HEADROOM = 1_024
MIN_RECORD_SIZE = CHUNK_SIZE + 4_096
READ_ATTEMPTS = 3
UUID_HEX_LENGTH = 32
UUID_VERSION = 4
MAX_SCALAR = 0x10FFFF
SURROGATE_START = 0xD800
SURROGATE_STOP = 0xE000

Bins = dict[str, object]

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


@dataclass(frozen=True)
class Root:
    """Validated root metadata, including the inline payload."""

    key: str
    version: str
    count: int
    data: bytes


class MutationBudgetExceededError(Exception):
    """A read-only transaction must be split before any application writes."""


def key_hash(key: str) -> str:
    """Validate a logical key without limiting its literal alphabet."""
    if not isinstance(key, str):
        msg = "Byte-store keys must be strings."
        raise TypeError(msg)
    return sha256(key.encode("utf-8")).hexdigest()


def root_id(key: str) -> str:
    return f"r:{key_hash(key)}"


def chunk_ids(key: str, start: int, stop: int) -> list[str]:
    digest = key_hash(key)
    return [f"c:{digest}:{index}" for index in range(start, stop)]


def prepare_pairs(
    pairs: Sequence[tuple[str, bytes]], record_limit: int
) -> list[tuple[str, bytes]]:
    """Validate even shadowed duplicates before coalescing (last value wins)."""
    unique: dict[str, bytes] = {}
    for key, value in pairs:
        key_hash(key)
        if not isinstance(value, bytes):
            msg = "Byte-store values must be bytes."
            raise TypeError(msg)
        if len(value) > MAX_VALUE_SIZE:
            msg = f"Byte-store values cannot exceed {MAX_VALUE_SIZE} bytes."
            raise ValueError(msg)
        root_payload_size = len(value) if len(value) <= CHUNK_SIZE else 0
        if len(key.encode("utf-8")) + root_payload_size > (
            record_limit - RECORD_HEADROOM
        ):
            msg = "The original key and root payload exceed the record-size budget."
            raise ValueError(msg)
        unique[key] = value
    return list(unique.items())


def mutation_groups(
    pairs: Sequence[tuple[str, bytes]],
) -> Iterator[list[tuple[str, bytes]]]:
    group: list[tuple[str, bytes]] = []
    payload_size = 0
    for pair in pairs:
        if group and (
            len(group) == MUTATION_KEYS or payload_size + len(pair[1]) > MAX_VALUE_SIZE
        ):
            yield group
            group = []
            payload_size = 0
        group.append(pair)
        payload_size += len(pair[1])
    if group:
        yield group


def check_mutation_budget(
    roots: Sequence[Root | None], values: Sequence[bytes] | None = None
) -> None:
    """Budget fixed-position chunk writes and cleanup from transactional roots."""
    if values is None:
        count = sum(1 + root.count for root in roots if root is not None)
    else:
        count = 0
        for root, value in zip(roots, values, strict=True):
            chunks = (
                (len(value) + CHUNK_SIZE - 1) // CHUNK_SIZE
                if len(value) > CHUNK_SIZE
                else 0
            )
            count += 1 + max(root.count if root is not None else 0, chunks)
    if count > TRANSACTION_RECORDS:
        msg = "Byte-store mutation exceeds the transaction record budget."
        raise MutationBudgetExceededError(msg)


def encode_value(key: str, value: bytes) -> tuple[Bins, list[Bins]]:
    version = uuid4().hex
    chunks: list[Bins] = []
    if len(value) > CHUNK_SIZE:
        chunks = [
            {
                "kind": 1,
                "fmt": 1,
                "version": version,
                "data": value[offset : offset + CHUNK_SIZE],
            }
            for offset in range(0, len(value), CHUNK_SIZE)
        ]
    root: Bins = {
        "kind": 0,
        "fmt": 1,
        "key": key,
        "version": version,
        "count": len(chunks),
        "data": b"" if chunks else value,
    }
    return root, chunks


def validate_header(bins: Mapping[str, object], kind: int) -> str:
    if (
        type(bins.get("kind")) is not int
        or bins["kind"] != kind
        or type(bins.get("fmt")) is not int
        or bins["fmt"] != 1
    ):
        msg = "Corrupt or incompatible byte-store record kind/schema."
        raise ValueError(msg)
    version = bins.get("version")
    if not isinstance(version, str) or len(version) != UUID_HEX_LENGTH:
        msg = "Corrupt byte-store replacement version."
        raise ValueError(msg)
    try:
        token = UUID(hex=version)
    except ValueError as exc:
        msg = "Corrupt byte-store replacement version."
        raise ValueError(msg) from exc
    if token.version != UUID_VERSION or token.hex != version:
        msg = "Byte-store replacement version must be UUID4 hex."
        raise ValueError(msg)
    return version


def decode_root(bins: Mapping[str, object], key: str) -> Root:
    version = validate_header(bins, 0)
    if not isinstance(bins.get("key"), str) or bins["key"] != key:
        msg = "Byte-store root key mismatch (corruption or hash collision)."
        raise ValueError(msg)
    count = bins.get("count")
    data = bins.get("data")
    if (
        type(count) is not int
        or not 0 <= count <= MAX_CHUNKS
        or count == 1
        or not isinstance(data, bytes)
        or len(data) > CHUNK_SIZE
        or (count > 0 and data != b"")
    ):
        msg = "Corrupt byte-store root count/payload."
        raise ValueError(msg)
    return Root(key, version, count, data)


def validate_chunk_owner(root: Root, bins: Bins | None) -> Bins:
    if bins is None or validate_header(bins, 1) != root.version:
        msg = "Missing byte-store chunk or replacement version mismatch."
        raise ValueError(msg)
    return bins


def decode_chunks(root: Root, chunks: Sequence[Bins | None]) -> bytes:
    parts: list[bytes] = []
    for index, bins in enumerate(chunks):
        validated = validate_chunk_owner(root, bins)
        data = validated.get("data")
        if (
            not isinstance(data, bytes)
            or not 0 < len(data) <= CHUNK_SIZE
            or (index < root.count - 1 and len(data) != CHUNK_SIZE)
        ):
            msg = "Corrupt byte-store chunk payload/length."
            raise ValueError(msg)
        parts.append(data)
    if len(parts) != root.count:
        msg = "Incomplete byte-store chunk sequence."
        raise ValueError(msg)
    return b"".join(parts)


def prefix_successor(prefix: str) -> str | None:
    """Least upper bound for a prefix in Unicode scalar/UTF-8 byte order."""
    key_hash(prefix)
    for index in range(len(prefix) - 1, -1, -1):
        codepoint = ord(prefix[index])
        if codepoint < MAX_SCALAR:
            successor = codepoint + 1
            if successor == SURROGATE_START:
                successor = SURROGATE_STOP
            return prefix[:index] + chr(successor)
    return None


def enumeration_filter(prefix: str | None) -> Exp:
    conditions = [
        Exp.eq(Exp.int_bin("kind"), Exp.int_val(0)),
        Exp.eq(Exp.int_bin("fmt"), Exp.int_val(1)),
    ]
    if prefix is not None:
        upper = prefix_successor(prefix)
        if prefix:
            conditions.append(Exp.ge(Exp.string_bin("key"), Exp.string_val(prefix)))
        if upper is not None:
            conditions.append(Exp.lt(Exp.string_bin("key"), Exp.string_val(upper)))
    return Exp.and_(conditions)
