# Aerospike ByteStore schema reference

Derived from the approved schema as reconciled in
[DESIGN.md](DESIGN.md#record-contract), including its configuration and bounds.
Regenerate this reference from that guide when the contract changes.

## One caller-owned physical set

| Property | Contract |
| --- | --- |
| Namespace | Explicit constructor argument; operational SC/MRT required |
| Set | Explicit `set_name`; exclusively owned by this schema |
| Hash | `h = SHA256(key.encode("utf-8")).hexdigest()`; 64 lowercase hex digits |
| Root key | `r:<h>` |
| Chunk key | `c:<h>:<zero-based decimal position>` |
| Expiry | Explicit never-expire writes; no TTL option |
| Delete | Durable; conditional native operations/batches for inline values, MRT for root/chunk cleanup and definite-conflict recovery |
| Indexes | Primary index only; no secondary indexes |

| Record role | Bin | Type | Purpose/constraint |
| --- | --- | --- | --- |
| Root | `kind` | Integer | `0`, root marker |
| Root | `fmt` | Integer | `1`, format version |
| Root | `key` | String | Original UTF-8-encodable logical key |
| Root | `version` | String | Fresh canonical lowercase UUID4 hex, 32 characters |
| Root | `count` | Integer | `0` inline or `2..128` chunks; never `1` |
| Root | `data` | Blob | Inline bytes up to 65,536 bytes; empty if chunked |
| Chunk | `kind` | Integer | `1`, chunk marker |
| Chunk | `fmt` | Integer | `1`, format version |
| Chunk | `version` | String | Owning root replacement token |
| Chunk | `data` | Blob | 1..65,536 bytes; nonfinal chunks exactly 65,536 |

## Bounds and required actions

| Bound or trigger | Behavior |
| --- | --- |
| Value length 0..65,536 bytes | Inline root, no chunks |
| Value length 65,537..8,388,608 bytes | Root plus 2..128 fixed-position chunks |
| Value length > 8,388,608 bytes | Reject entire `mset` input before writes |
| Key/value type or UTF-8 error | Reject before writes, including shadowed duplicate entries |
| Root budget | UTF-8 key bytes + inline payload bytes <= cached record limit - 1,024 |
| Configured record limit < 69,632 bytes | Reject construction (`MIN_RECORD_SIZE`) |
| Constructor capability evidence | All discovered nodes: build major >= 8, SC enabled, MRT writes enabled, zero unavailable/dead partitions, no stop-writes/clock-skew stop-writes, positive MRT duration |
| Mutation candidate | <= 128 distinct keys; `mset` also <= 8 MiB new payload |
| Group record bound | Transactional root reads determine <= 4,096 distinct application writes; oversized read-only candidates abort and split before writing |
| Replacement record count | `sum(1 + max(old_chunk_count, new_chunk_count))` |
| Deletion record count | `sum(1 + old_chunk_count)` for existing roots |
| Inline mutation | No root pre-read; atomic typed kind=0/fmt=1/count=0/original-key guard; native REPLACE can create missing roots, durable DELETE ignores missing roots |
| Native mutation batches | Implicit MRT disabled; only definite conflicts fall back; successful or uncertain keys are not deliberately replayed |
| Corrupt inline payload/UUID | Reads reject; overwrite/delete may repair a recognized inline root without decoding the old payload/UUID; unknown layout/key/count fails closed |
| Missing root creation inside MRT | Create-only root insertion; a concurrent create aborts rather than leaving unowned trailing chunks |
| Multi-key mutation | Per-value atomicity only; no cross-key isolation or rollback, including within one group; partial success on errors is possible |
| Root/chunk batch reads | <= 128 roots per dispatch; chunk reconstruction in 16-key windows; <= 128 physical records per batch |
| Mutation retry bound | SDK transaction callback: 5 attempts, 10 ms between retries |
| Read version/chunk mismatch | Reread root; retry changed roots, at most 3 reconstruction attempts |
| Stable-root chunk corruption or invalid root schema | Raise error, not cache miss |
| Exhausted concurrent read retries | Raise `RuntimeError` |
| Shrink/delete | Remove obsolete chunks durably in the same MRT |
| Missing root | `mget`: `None`; `mdelete`: no-op |
| Enumeration | Typed root/format/literal-prefix filter, `key` projection; full-set query, no order or snapshot |
| Total logical-key population | Application/operator capacity management; no automatic eviction |
