"""Aerospike implementation boundary for LangChain's byte-store contract."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from aerospike_sdk import (
    DataSet,
    ErrorStrategy,
    Exp,
    QueryHint,
)
from aerospike_sdk.sync import TransactionalSession
from langchain_core.stores import ByteStore

from langchain_aerospike_kv._config import namespace_record_limit, validate_names
from langchain_aerospike_kv._io import (
    check_write_results,
    conditional_conflicts,
    read_records,
)
from langchain_aerospike_kv._schema import (
    CHUNK_SIZE,
    GROUP_KEYS,
    MUTATION_KEYS,
    READ_ATTEMPTS,
    READ_KEYS,
    Bins,
    MutationBudgetExceededError,
    Root,
    check_mutation_budget,
    chunk_ids,
    decode_chunks,
    decode_root,
    encode_value,
    enumeration_filter,
    key_hash,
    mutation_groups,
    prepare_pairs,
    root_id,
    validate_chunk_owner,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from aerospike_sdk.sync import Session
    from aerospike_sdk.sync.operations.query import WriteSegmentBuilder


_INLINE_LAYOUT = Exp.and_(
    [
        Exp.eq(Exp.int_bin("kind"), Exp.int_val(0)),
        Exp.eq(Exp.int_bin("fmt"), Exp.int_val(1)),
        Exp.eq(Exp.int_bin("count"), Exp.int_val(0)),
    ]
)


class AerospikeByteStore(ByteStore):
    """Byte-store interface backed by a caller-owned synchronous SDK session.

    !!! warning "Experimental integration"
        Requires an operational SC/MRT namespace and a compatible SDK/server.
        Validate the combination in your deployment before production use.

    The caller owns the process-level cluster/client and its shutdown. Construction
    checks the existing cluster's SC/MRT configuration, but neither opens a
    connection nor reads environment variables. Async operations inherit
    `BaseStore`'s executor-backed implementations.

    The set must be owned exclusively by this schema. Values up to 8 MiB are
    replaced/deleted atomically. Inline mutations use conditional record
    operations, including native batches. Chunked values and definite-conflict
    recovery use multi-record transactions. Multi-key calls can partially succeed
    and have no cross-key isolation or rollback guarantee. Enumeration is a
    full-set query, without an ordering or snapshot guarantee.

    Overwrite/delete can repair a recognized inline root's old payload/UUID
    without decoding it. Reads and protected chunked operations reject corruption.
    """

    def __init__(
        self,
        session: Session,
        *,
        namespace: str,
        set_name: str,
    ) -> None:
        """Validate explicit configuration without taking session ownership.

        Args:
            session: Synchronous session from the caller's shared SDK cluster.
            namespace: Caller-provisioned strong-consistency namespace.
            set_name: Caller-selected set for the byte store.

        Raises:
            TypeError: If names are not strings or the session is transactional.
            ValueError: If configuration or cluster capability is unsuitable.
            AerospikeError: If cluster inspection fails.
        """
        validate_names(namespace, set_name)
        if isinstance(session, TransactionalSession):
            msg = "Pass a process-level Session, not a TransactionalSession."
            raise TypeError(msg)
        self._record_limit = namespace_record_limit(session, namespace)
        self._session = session
        self._dataset = DataSet.of(namespace, set_name)

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        """Get values in input order, returning `None` for missing keys.

        Args:
            keys: Keys to retrieve, including any duplicates.

        Returns:
            One optional byte value per input key.

        Raises:
            TypeError: If a key is not a string.
            ValueError: If a key is not UTF-8 encodable or stored data is corrupt.
            RuntimeError: If concurrent replacement exhausts bounded read retries.
            AerospikeError: If a database operation fails.
        """
        unique = list(dict.fromkeys(keys))
        for key in unique:
            key_hash(key)
        values: dict[str, bytes | None] = {}
        for offset in range(0, len(unique), READ_KEYS):
            values.update(self._read_group(unique[offset : offset + READ_KEYS]))
        return [values[key] for key in keys]

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        """Replace the complete byte value associated with each key.

        Keys are independently atomic; errors can leave other keys committed.
        No cross-key isolation or group rollback is promised.

        Args:
            key_value_pairs: Keys and their replacement byte values.

        Raises:
            TypeError: If keys/values are not strings/bytes.
            ValueError: If input exceeds size/encoding limits or stored data is corrupt.
            AerospikeError: If an operation or transaction fails.
        """
        pairs = prepare_pairs(key_value_pairs, self._record_limit)
        for group in mutation_groups(pairs):
            self._mutate_batch(group)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the requested keys, ignoring keys that do not exist.

        Keys are independently atomic; errors can leave other keys deleted.
        No cross-key isolation or group rollback is promised.

        Args:
            keys: Keys to delete.

        Raises:
            TypeError: If a key is not a string.
            ValueError: If a key is not UTF-8 encodable or stored data is corrupt.
            AerospikeError: If an operation or transaction fails.
        """
        unique = list(dict.fromkeys(keys))
        for key in unique:
            key_hash(key)
        for offset in range(0, len(unique), MUTATION_KEYS):
            self._mutate_batch(
                [(key, None) for key in unique[offset : offset + MUTATION_KEYS]]
            )

    def _mutate_batch(self, group: Sequence[tuple[str, bytes | None]]) -> None:
        pending = dict(group)
        native = [
            (key, value)
            for key, value in group
            if value is None or len(value) <= CHUNK_SIZE
        ]
        conflicts = self._write_inline_batch(native)
        for index, (key, _) in enumerate(native):
            if index not in conflicts:
                del pending[key]
        if pending:
            if group[0][1] is None:
                self._remove_group(list(pending))
            else:
                self._set_group(
                    [
                        (key, value)
                        for key, value in pending.items()
                        if value is not None
                    ]
                )

    def _write_inline_batch(
        self, native: Sequence[tuple[str, bytes | None]]
    ) -> set[int]:
        builder: WriteSegmentBuilder | None = None
        for key, value in native:
            owner = self._session if builder is None else builder
            identifier = self._dataset.id(root_id(key))
            if value is None:
                builder = owner.delete(identifier).with_durable_delete()
            else:
                builder = owner.replace(identifier)
                builder = builder.put(encode_value(key, value)[0]).never_expire()
            # Evaluate the current layout atomically with the record mutation.
            # REPLACE creates missing records; existing chunked roots cannot pass.
            builder = builder.where(
                Exp.and_(
                    [
                        _INLINE_LAYOUT,
                        Exp.eq(Exp.string_bin("key"), Exp.string_val(key)),
                    ]
                )
            )
        if builder is None:
            return set()
        return conditional_conflicts(
            builder.with_txn(None)
            .include_missing_keys()
            .fail_on_filtered_out()
            .execute(on_error=ErrorStrategy.IN_STREAM),
            len(native),
            missing_ok=native[0][1] is None,
        )

    def _set_group(self, group: Sequence[tuple[str, bytes]]) -> None:
        try:
            self._session.do_in_transaction(
                partial(self._replace_group, group=group),
                max_attempts=5,
                sleep_between_retries=0.01,
            )
        except MutationBudgetExceededError:
            if len(group) <= 1:  # A valid single value always fits the budget.
                raise
            middle = len(group) // 2
            self._set_group(group[:middle])
            self._set_group(group[middle:])

    def _remove_group(self, keys: Sequence[str]) -> None:
        try:
            self._session.do_in_transaction(
                partial(self._delete_group, keys=keys),
                max_attempts=5,
                sleep_between_retries=0.01,
            )
        except MutationBudgetExceededError:
            if len(keys) <= 1:
                raise
            middle = len(keys) // 2
            self._remove_group(keys[:middle])
            self._remove_group(keys[middle:])

    def yield_keys(self, *, prefix: str | None = None) -> Generator[str, None, None]:
        """Iterate over keys matching an optional literal string prefix.

        Args:
            prefix: Literal prefix to match, or `None` to enumerate every key.

        Yields:
            Matching string keys.

        Raises:
            TypeError: If prefix is not a string or `None`.
            ValueError: If prefix is not UTF-8 encodable or a projected key is corrupt.
            AerospikeError: If the query fails.
        """
        query = self._session.query(self._dataset)
        # Keep the sync builder type: SDK where()'s overload annotations refer
        # to its async sibling, although the method mutates this same builder.
        query.where(enumeration_filter(prefix))
        query.bins(["key"])
        query.with_hint(QueryHint(allow_scans_with_where=True))
        stream = query.execute(on_error=ErrorStrategy.IN_STREAM)
        try:
            for row in stream:
                key = row.record_or_raise().bins.get("key")
                if not isinstance(key, str):
                    msg = "Corrupt byte-store enumeration key."
                    raise ValueError(msg)  # noqa: TRY004 - on-disk schema corruption
                key_hash(key)
                yield key
        finally:
            stream.close()

    def _roots(self, session: Session, keys: Sequence[str]) -> list[Root | None]:
        records = read_records(session, self._dataset, [root_id(key) for key in keys])
        return [
            None if bins is None else decode_root(bins, key)
            for key, bins in zip(keys, records, strict=True)
        ]

    def _read_group(self, keys: Sequence[str]) -> dict[str, bytes | None]:
        roots = self._roots(self._session, keys)
        values: dict[str, bytes | None] = {}
        for offset in range(0, len(keys), GROUP_KEYS):
            values.update(
                self._read_values(
                    keys[offset : offset + GROUP_KEYS],
                    roots[offset : offset + GROUP_KEYS],
                )
            )
        return values

    def _read_values(
        self, keys: Sequence[str], roots: list[Root | None]
    ) -> dict[str, bytes | None]:
        pending = list(keys)
        values: dict[str, bytes | None] = {}
        for _ in range(READ_ATTEMPTS):
            identifiers = [
                identifier
                for root in roots
                if root is not None and root.count
                for identifier in chunk_ids(root.key, 0, root.count)
            ]
            records = (
                read_records(self._session, self._dataset, identifiers)
                if identifiers
                else []
            )
            offset = 0
            invalid: dict[str, tuple[Root, ValueError]] = {}
            for key, root in zip(pending, roots, strict=True):
                if root is None:
                    values[key] = None
                elif root.count == 0:
                    values[key] = root.data
                else:
                    chunks = records[offset : offset + root.count]
                    offset += root.count
                    try:
                        values[key] = decode_chunks(root, chunks)
                    except ValueError as exc:
                        invalid[key] = (root, exc)
            if not invalid:
                return values
            pending = list(invalid)
            roots = self._roots(self._session, pending)
            for key, root in zip(pending, roots, strict=True):
                old, cause = invalid[key]
                if root is not None and root.version == old.version:
                    msg = f"Corrupt chunks for unchanged byte-store root {key!r}."
                    raise ValueError(msg) from cause
        msg = "Concurrent byte-store replacements exhausted read retries."
        raise RuntimeError(msg)

    def _verify_cleanup(
        self, tx: Session, roots_and_ids: Sequence[tuple[Root, list[str]]]
    ) -> list[str]:
        identifiers = [
            identifier for _, identifiers in roots_and_ids for identifier in identifiers
        ]
        records = read_records(
            tx, self._dataset, identifiers, bins=["kind", "fmt", "version"]
        )
        owners = [root for root, identifiers in roots_and_ids for _ in identifiers]
        for root, bins in zip(owners, records, strict=True):
            try:
                validate_chunk_owner(root, bins)
            except ValueError as exc:
                # A concurrent replacement may have committed between the
                # metadata and cleanup reads. Re-reading this previously read
                # root in the SAME MRT raises MRT_VERSION_MISMATCH in that
                # case, allowing the SDK to abort/retry the whole callback.
                # An unchanged root instead proves on-disk corruption.
                self._roots(tx, [root.key])
                msg = "Missing or corrupt chunk during transactional cleanup."
                raise ValueError(msg) from exc
        return identifiers

    def _replace_group(
        self, tx: TransactionalSession, *, group: Sequence[tuple[str, bytes]]
    ) -> None:
        keys = [key for key, _ in group]
        old_roots = self._roots(tx, keys)
        check_mutation_budget(old_roots, [value for _, value in group])
        writes: list[tuple[str, Bins]] = []
        creates: set[str] = set()
        obsolete: list[tuple[Root, list[str]]] = []
        for (key, value), old in zip(group, old_roots, strict=True):
            root, chunks = encode_value(key, value)
            identifier = root_id(key)
            writes.append((identifier, root))
            if old is None:
                creates.add(identifier)
            writes.extend(zip(chunk_ids(key, 0, len(chunks)), chunks, strict=True))
            if old is not None and old.count > len(chunks):
                obsolete.append((old, chunk_ids(key, len(chunks), old.count)))
        deletes = self._verify_cleanup(tx, obsolete)
        identifier, bins = writes[0]
        # A missing point read need not enter the SDK's MRT read set.
        start = tx.insert if identifier in creates else tx.replace
        builder = start(self._dataset.id(identifier)).put(bins).never_expire()
        for identifier, bins in writes[1:]:
            segment = builder.insert if identifier in creates else builder.replace
            builder = segment(self._dataset.id(identifier)).put(bins).never_expire()
        if deletes:
            builder = builder.delete(self._dataset.ids(*deletes)).with_durable_delete()
        check_write_results(
            builder.execute(on_error=ErrorStrategy.IN_STREAM),
            len(writes) + len(deletes),
        )

    def _delete_group(self, tx: TransactionalSession, *, keys: Sequence[str]) -> None:
        roots = self._roots(tx, keys)
        check_mutation_budget(roots)
        existing = [root for root in roots if root is not None]
        chunks = self._verify_cleanup(
            tx, [(root, chunk_ids(root.key, 0, root.count)) for root in existing]
        )
        identifiers = [root_id(root.key) for root in existing] + chunks
        # Deleting a missing/tombstoned key inside MRT can fail commit
        # verification. Transactional root/chunk reads above prove existence.
        if identifiers:
            builder = tx.delete(self._dataset.ids(*identifiers)).with_durable_delete()
            check_write_results(
                builder.execute(on_error=ErrorStrategy.IN_STREAM), len(identifiers)
            )
