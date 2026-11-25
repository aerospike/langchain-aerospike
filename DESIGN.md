# Aerospike ByteStore: design and schema guide

This is the shipped form of the schema approved on 2026-11-17, reconciled
with the implemented `_schema`, `_config`, `_io`, and store behavior.
[SCHEMA.md](SCHEMA.md) is derived from this guide's contract and bounds,
not an independently designed schema. [README.md](README.md) covers setup
and runnable public usage.

## Background and resolved clarification

The extension point is LangChain's existing `ByteStore`, not a new application
or an API for graph, vector search, chat history, or expiring cache entries.
`InMemoryByteStore`, `LocalFileStore`, and the unchanged standard store tests
establish ordering, duplicates, misses, replacement, literal prefixes, and
sync/async behavior. No host interface or other backend needs modification.

The approved clarification selected:

- Opaque string-keyed byte values, independently created/replaced/deleted.
- An inclusive 8 MiB maximum per value, requiring a bounded chunked path.
- Atomic publication/removal of each complete value; no all-or-nothing
  multi-key guarantee or cross-key read snapshot.
- A caller-injected synchronous session from one shared process-level cluster.
- No implicit expiry: the host interface has no TTL.

There is one entity group: a logical key owns one root and, only when needed,
its chunks. Chunks cannot be edited or retained independently. There are no
relationships between logical keys, joins, accumulating collections, or
data-dependent bin names.

### Workload assumptions and reconsideration triggers

| Judgment or missing input | Current decision | Evidence that reopens it |
| --- | --- | --- |
| Point reads/writes are the primary access path | Key lookups and bounded batches; enumeration is a query | Measured enumeration frequency/SLO makes full-set work unacceptable |
| No production size/skew/update-rate distribution or latency SLO was supplied | No throughput or production-capacity claim; monitor real distributions | Representative traffic, p95/p99 value sizes, hot-key rates, and an SLO |
| 10k/100k/1m entries with 1 KiB values were selected as synthetic comparison workloads | Sizing examples only, not a production forecast or measured result | Measured storage and operation distributions |
| Independent values have independent lifecycles | Do not consolidate unrelated small keys | A new, explicitly bounded shared-collection requirement |
| Whole-value replacement is the only mutation | Bounded opaque byte slices, not CDT mutations | A different host contract requires partial edits |
| An operational SC/MRT namespace is mandatory | Single-key inline mutations use conditional record operations; MRT handles chunked values and multi-key groups | A change to the value or atomicity contract |

Large-value update frequency remains unknown, not presumed low. Layout
optimizations beyond the approved bounds are deferred until measurement.

## Access patterns

| Method | Resolution | Observable behavior and cost |
| --- | --- | --- |
| `mget` | Up to 128 unique root IDs per dispatch; chunk reconstruction in 16-key windows, with physical batches of 128 | Restore input order/duplicates; absent roots give `None`; materialize complete values |
| `mset` | Validate all input and coalesce duplicates; conditional single-key inline operation or adaptively budgeted MRT groups | Last duplicate wins; publish complete values and remove obsolete positions atomically |
| `mdelete` | Conditional single-key inline deletion or transactional root/owned-chunk reads and cleanup | Durable removal of complete values; missing roots are skipped |
| `yield_keys` | Set query with typed root/schema/literal-prefix expression and `key` projection | Full-set work, no index, ordering, pagination, or snapshot promise |
| Async variants | Inherited executor-backed sync methods | Same data/error contract and same caller-owned session |

Root queries are key-based batches, not secondary-index queries. Enumeration
still traverses the physical set (roots **and** chunks); filtering reduces
returned rows and projection reduces network bytes, not device record I/O.
Use it for enumeration, not a latency-critical per-request lookup. A large
`mget` still allocates all requested values in application memory.

## Record contract

One caller-selected set in one operational SC namespace holds both roles.
That set is exclusively owned by this schema. Deliberately using the same
namespace/set shares data; separate Python store objects do not isolate it.
Aerospike does not enforce a set-level schema.

Let `h = SHA256(key.encode("utf-8")).hexdigest()` (all 64 lowercase hex digits).

| Role | Physical user key | Live records per logical value |
| --- | --- | --- |
| Root | `r:<h>` | Exactly 1 while present |
| Chunk | `c:<h>:<zero-based decimal position>` | 0 when inline; otherwise 2 through 128 |

Role prefixes separate root and chunk identities. Position ordering is
numeric during reconstruction, never lexicographic (`10` does not precede
`2`). Keys may be empty and contain Unicode/NUL; no normalization or regex
alphabet is imposed. Invalid UTF-8 surrogate strings are rejected.

### Root bins

| Bin | Aerospike type | Constraint and purpose |
| --- | --- | --- |
| `kind` | Integer | Exactly `0`, root marker |
| `fmt` | Integer | Exactly `1`, on-disk format |
| `key` | String | Original logical key; checked against the requested key |
| `version` | String | Canonical 32-character lowercase UUID4 hex, fresh on replacement |
| `count` | Integer | `0` inline, otherwise `2..128`; `1` is invalid |
| `data` | Blob | Complete value of at most 65,536 bytes inline; empty when chunked |

Example inline root for logical key `documents/42`, at `r:<h>`:

```json
{
  "kind": 0,
  "fmt": 1,
  "key": "documents/42",
  "version": "3f45fd632d15469eafc6499061bdc353",
  "count": 0,
  "data": "hex:68656c6c6f"
}
```

`hex:` is explanatory JSON notation for a native blob, not a stored string.
The literal empty blob is `b""`; it is a value, unlike an absent root.

### Chunk bins

| Bin | Aerospike type | Constraint and purpose |
| --- | --- | --- |
| `kind` | Integer | Exactly `1`, chunk marker |
| `fmt` | Integer | Exactly `1`, on-disk format |
| `version` | String | Same replacement token as the owning root |
| `data` | Blob | 1 through 65,536 bytes; every nonfinal chunk is exactly 65,536 bytes |

For a 65,540-byte value, the root instead has `count=2`, empty `data`,
and the same token as both chunks. Chunk 0 holds the first 65,536 bytes.
An example final chunk at `c:<h>:1` is:

```json
{
  "kind": 1,
  "fmt": 1,
  "version": "3f45fd632d15469eafc6499061bdc353",
  "data": "hex:00010203"
}
```

All bin names fit the 15-character limit. Fields share their value's lifecycle
and are needed for identity, reconstruction, or validation; there is no
unbounded list/map bin. No secondary indexes or index-creation privileges
are needed.

## Transactions and reads

Single-key and multi-key inline mutations use native operations without reading
old roots first. An atomic typed server condition requires an existing root to
have `kind=0`, `fmt=1`, `count=0`, and the requested original key. Native
`REPLACE` creates missing roots or fully replaces matching inline roots.
Missing deletes are no-ops. A current chunked root cannot pass, including after
generation wrap or delete/recreate ABA. Concurrent inline overwrites do not
unnecessarily conflict on an earlier observed UUID.

`with_txn(None)` explicitly disables implicit SDK batch transactions. Missing
and filtered-out results are requested, drained, and checked. Successful keys
are not replayed; only definite condition failures enter the bounded MRT
fallback. Unknown/in-doubt outcomes and infrastructure errors propagate and can
prevent subsequent transactional work. Uncertain errors take precedence over
definite errors from another row.

A recognized inline root may have its corrupt old payload/UUID replaced or
deleted without decoding it, as with Redis replacing an opaque value. Reads
still reject such corruption. Unknown layouts or original-key mismatches do
not pass the native guard and fail closed during MRT validation.

Chunked old/new values and definite conditional conflicts use an explicit SDK
`do_in_transaction` callback, including reads of previous root metadata and
cleanup ownership checks. All row results are checked before callback success.
Shrinking or deleting removes obsolete chunks durably in that same transaction;
fixed positions avoid a normal-operation orphan-GC lifecycle. Missing/tombstoned
roots are not blindly deleted inside MRT.

Inside MRT, new roots are inserted create-only. A missing point read
is not necessarily tracked in the SDK's transaction read set. A conflicting
creation must abort rather than overwrite a rival's chunked root without
cleaning its old chunks. A definite create-only conflict can surface as
`RecordExistsError`; before retrying, multi-key callers must account for native
keys that may already have committed.

Candidate `mset` groups contain at most **128 distinct keys and 8 MiB of new
payload**, whichever bound is reached first. Deletes consider up to 128 keys.
After reading old roots inside MRT, replacement budgets
`sum(1 + max(old_chunk_count, new_chunk_count))` distinct application records;
deletion budgets `sum(1 + old_chunk_count)` for existing roots. If this exceeds
the 4,096-record write limit, abort before any writes, halve the candidate, and
retry each half with new transactional reads. A valid single value always fits.
Every transaction retry recalculates its budget; a concurrent grow cannot
silently invalidate the count. Server-managed monitor records are excluded.
The new-payload budget does **not** bound old-chunk cleanup, metadata,
transaction-monitor overhead, replication, or application memory. The
[performance revision](PERFORMANCE.md) records the design and validation scope.

SDK mutation retries use at most five attempts with a 10 ms pause between
retries. A refused native inline attempt may precede those callback attempts;
there is no preliminary metadata lookup on the ordinary inline path.
These fallback cases can do more work than directly entering MRT. Errors still
surface on exhaustion. MRT needs an SC namespace;
a sequence of independent batch writes is not an alternative for atomic
root/chunk lifecycle changes. The inline optimization does not impose an
inline-only data contract: finding a chunked root takes the MRT path.
Multi-key calls have **no cross-key isolation or rollback guarantee**, including
within one candidate group. Native keys may already be committed when a chunked
key fails. A racing reader may see part of a batch. This intentionally removes
the previous group-rollback behavior; group boundaries remain adaptive internal
details.

RedisStore's default transactional `SET` pipeline prevents interleaved execution,
while Redis runtime command errors do not roll back other successful commands.
Its single `MGET` and multi-key `DEL` also have stronger cross-key visibility than
this adapter promises. The unchanged LangChain interface and in-memory backend
do not require those Redis-specific guarantees. The performance comparison must
identify these differences rather than claim identical batch semantics; a
separately labeled nontransactional Redis pipeline isolates write-isolation cost.

Reads do not open a snapshot transaction. Up to 128 unique roots are fetched
together independently of mutation bounds. Chunk reconstruction retains
16-logical-key windows to avoid multiplying the maximum chunk-buffer window;
each physical read batch is still limited to 128 records. An inline value is a
single-record read. A chunked read checks the root count and each chunk's format, UUID token,
and length, then concatenates numeric positions. A missing/invalid/different-token
chunk causes a root reread: an unchanged root proves corruption (`ValueError`);
a changed/deleted root permits retry. Three reconstruction attempts are
allowed before `RuntimeError`. A reader returns complete bytes or an error,
not mixed replacement versions.

Malformed roots, incompatible formats, hash/key mismatches, and stable
chunk corruption are not misses. Infrastructure/ACL/transaction/per-row
failures propagate; only genuine record-not-found rows have missing-key
semantics. Invalid/duplicate/omitted SDK result positions are errors. Streams
close in `finally`, including explicitly closed early enumeration generators.
Close a sync `yield_keys` generator if stopping early.

Enumeration uses typed expressions: `kind == 0`, `fmt == 1`, and optional
`key >= prefix` / `key < successor(prefix)`. The successor increments the
last incrementable Unicode scalar, skipping surrogates; no upper bound is
added if none exists. `None`/empty prefix enumerate all roots. Prefixes are
literal, not interpolated AEL or regexes. Enumeration filters/projected keys
are not a complete integrity audit: incompatible-format rows can be filtered
out, while point reads validate their full required schema.

## Configuration and security

Construction accepts a nontransactional synchronous SDK `Session` and explicit
keyword-only `namespace`/`set_name`. It neither connects nor owns shutdown,
but **does perform network info inspection**. It reads no environment
configuration. A `TransactionalSession` is rejected rather than silently
joining an application transaction.

For each discovered node, construction requires:

- A decimal server build major of at least 8 (a capability gate, not a full
  SDK/server compatibility certification).
- `strong-consistency=true`, `disable-mrt-writes=false`.
- `unavailable_partitions=0`, `dead_partitions=0`, `stop_writes=false`,
  `clock_skew_stop_writes=false`.
- Reported `max-record-size >= 69,632` (`MIN_RECORD_SIZE`) and positive
  `mrt-duration`; matching node coverage for build/namespace info.

The smallest reported record limit is cached. Recreate/revalidate stores
after configuration changes; construction is not continuous health monitoring.
Input names must be nonempty UTF-8 strings without NUL; namespace names also
exclude info-command delimiters (`;:=/`, CR, LF). Server naming limits still
apply.

Before any `mset` writes, every input entry, including shadowed duplicates,
must have a UTF-8 string key and a `bytes` value no larger than 8,388,608
bytes. Additionally:

```text
UTF8_key_bytes + inline_payload_bytes <= cached_record_limit - 1,024
inline_payload_bytes = value length if length <= 65,536, otherwise 0
```

This reserves root metadata headroom; a long original key can fail the
budget even though its physical hash is short. No dynamic chunk-size/record
limit negotiation or automatic schema migration occurs.

Roots and chunks are explicitly **never-expire**, independent of namespace
default TTL. There is no TTL option. Deletion uses durable deletes; tombstone
reclamation and transaction-monitor lifecycle are server/operator concerns.
Direct record edits, expiration, set truncation, and mixing schemas can
violate the contract. Backups/restores must preserve complete logical values.

Hashes bound physical-key length, not data disclosure or permissions: the
original key and bytes remain readable. Use SDK credentials/TLS and server
ACLs; choosing a namespace/set is not authorization. Grant the constructor's
read-only info access plus required data/query permissions, not administrative
roster privileges. Treat host object-deserialization adapters separately from
this opaque-byte boundary.

## Sizing, growth, and hot keys

All estimates are logical/uncompressed and **not measured storage results**.
Record framing, keys, bin names/types, allocation, replicas, tombstones,
transaction monitors, defragmentation, and namespace/set metadata add costs.
Storage compression does not raise the uncompressed record-size limit.

| Shape | Logical payload/layout | Primary-index estimate per replica |
| --- | --- | --- |
| Typical synthetic 1 KiB value | One root, about 1 KiB plus key/token/schema/framing | 64 bytes |
| 1,000,000 such values | 1,000,000 roots, about 1 GB payload plus metadata | 64,000,000 bytes (about 61 MiB) |
| Maximum value | 8 MiB in 128 bounded chunks plus a small root | `129 * 64 = 8,256` bytes |

The baseline stays one record for small independent values, even below the
usual single-digit-KiB target: consolidation would couple unrelated keys,
lifecycles, and contention. The 64 KiB chunks deliberately exceed the
roughly 50 KiB review threshold. They are replacement-only bounded slices,
not accumulating or independently hot mutable collections. Every update
rewrites a record; a whole-value replacement necessarily rewrites all its
payload. Large-value write frequency must be measured before accepting its
I/O cost, not inferred from a small-value workload.

Growth is bounded per value: 65,536 bytes triggers chunking, 8 MiB is the hard
application ceiling, and shrink/delete cleans obsolete positions atomically.
The total number of logical keys is **not** bounded by the store. Monitor
capacity, value/key-size distributions, live physical records, tombstones,
enumeration cost, per-key write skew, conflicts/retries, and transaction
duration versus configured `mrt-duration`. Revisit the model if those miss
an operational SLO; it does not auto-shard a hot logical key or promise a
maximum transaction latency.

### Rejected alternatives

| Alternative | Why not | Reopen when |
| --- | --- | --- |
| One record for every value | Full 8 MiB plus metadata does not fit | Required maximum is lowered |
| One collection containing unrelated keys | Unbounded growth, hot records, coupled lifecycle | A new bounded-collection contract |
| AP writes plus manifest swap | Violates required transactional publication/removal | Not within this contract |
| Versioned chunk IDs with orphan collection | Adds GC ownership, retention, and cleanup races | Measured contention justifies a separately reviewed design |
| Prefix secondary indexes | Arbitrary-prefix/all-key enumeration is still full-set work here | A specific bounded indexed access path is required |
| New client/cluster per operation | Breaks caller ownership and shared lifecycle | Not within this contract |

## Validation evidence and limits

The preceding implementation test pass recorded 123 network-disabled unit
tests and 78 real SC integration tests passing with zero skips. The suite
subclasses unchanged sync/async standard contract tests and exercises real
`create_kv_docstore` composition. Focused tests cover exact 8 MiB/129-record
commit, numeric chunk order, grow/shrink/delete cleanup, never-expire writes,
post-write abort, failed commit, per-row conflicts, deterministic concurrency,
and literal Unicode/NUL prefix semantics.

These are single-node, file-backed Enterprise SC functional checks, not
production SC/MRT, failover, process-crash recovery, performance, or
released-image compatibility claims. A successful constructor is not a
deployment certification. Validate your licensed build, SDK, ACL/TLS,
roster, storage, transaction-duration headroom, and failure behavior in
your deployment. The repository commands in [README.md](README.md#repository-development)
include `--require-live` to prevent an unreachable server becoming a
misleading all-skipped pass. No benchmark or overall integration-completion
claim follows from this documentation.
