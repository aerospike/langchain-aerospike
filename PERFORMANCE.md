# ByteStore performance revision

## Scope and resolved requirements

Keep the existing public constructor and methods, on-disk format, 8 MiB value
limit, caller-owned session, never-expire policy, read corruption errors, and atomic
publication/removal of a complete value. No schema migration is needed.
Each logical value retains atomic publication/removal. Multi-key mutations no
longer promise group rollback or cross-key isolation: successful keys may remain
committed when another key fails. Other providers and the host contract are
unchanged.

The measured workload has inline 1 KiB values. Production value-size, skew, and
concurrency distributions remain unknown: benchmark results are synthetic, not
a production sizing recommendation. Revisit the chosen batch bounds when a real
distribution and latency SLO are available.

## Read batching

Fetch up to 128 unique roots per dispatch, independently of mutation grouping.
Reconstruct chunked values in windows of 16 logical keys, with at most 128
physical records per request. This avoids multiplying the maximum chunk-buffer
window by eight. Preserve input ordering, duplicates, missing values, bounded
version retries, stream closure, and individual result validation.

## Direct inline mutations

Single-key and multi-key inline mutations share a native, one-dispatch path:
there is **no preliminary root read**. Atomically require an existing root to
have typed `kind=0`, `fmt=1`, `count=0`, and the requested original key.
Native `REPLACE` creates a missing root or completely replaces a matching inline
root. The server evaluates filters on existing records; missing records can be
created. A missing durable delete is a no-op.

The condition tests the **current layout at mutation time**, rather than an
observed UUID/generation. Concurrent inline writers retain last-writer-wins
behavior without unnecessary CAS retries. A rival's growth to chunked storage
fails the condition even after delete/recreate or generation wrap; fresh
transactional metadata reads then protect its chunk lifecycle.

Recognized inline roots need not have their old payload or UUID decoded before
replacement/deletion. Those mutations can repair a corrupt inline payload/UUID,
analogous to Redis replacing an opaque old value. Reads still reject corruption.
Unknown kind/format, mismatched original key, and invalid/chunked counts do not
pass this guard; MRT validation fails closed on incompatible metadata.

Dispatch eligible keys together with `with_txn(None)` to disable the SDK's
implicit batch-write MRT. Drain and validate every result, including missing and
filtered-out rows. Only definite conditional conflicts enter the bounded MRT
fallback, together with keys whose old/new values are chunked. Successful native
keys are never replayed. Unknown/in-doubt outcomes and other server errors
propagate after draining the native batch, without replaying uncertain writes.
An uncertain error takes precedence over a definite error from another row.
Dispatch failures before a result stream is returned propagate directly; their
outcomes can be partial or unknown. An error may prevent later MRT work from
being attempted.

New roots use create-only insertion **inside MRT**. The SDK's missing
single-record read does not track absence in its read set; a blind transactional
replacement could overwrite a concurrently created chunked root and orphan
trailing chunks. A conflicting insertion aborts and can propagate
`RecordExistsError`, preserving the rival's complete value.

The fallback re-reads current metadata transactionally and can still find an
inline value after a race; MRT is retained there as a bounded, conservative
recovery path. It is not a namespace-wide requirement for ordinary inline writes.
New chunked values go directly to MRT; an inline mutation of an old chunked value
adds one refused native attempt before MRT. Such fallbacks need not be faster
than entering MRT directly.

### Redis comparison and intentional differences

LangChain's `BaseStore` does not specify multi-key isolation or group rollback;
the in-memory backend also updates keys individually. RedisStore queues one
`SET` per key in a default transactional pipeline. Redis `MULTI/EXEC` prevents
other clients from interleaving execution, but runtime command errors do not
roll back successful commands. Queue-time errors can abort the whole transaction.
Its `mget` is one `MGET`; its `mdelete` is one multi-key `DEL`.
See the [Redis transaction semantics](https://redis.io/docs/latest/develop/using-commands/transactions/).

Native Aerospike batches preserve each complete value but **do not match Redis's
whole-batch visibility or all-or-none dispatch before EXEC**. A concurrent reader
may see a subset of completed mutations. Even an exception may leave some keys
changed; callers must not assume group rollback or blindly retry uncertain
outcomes. Chunked values still need root/chunk atomicity because Redis stores the
same byte string as one logical server-side value.

Compare unchanged RedisStore as the real backend, and label a separate
`transaction=False` Redis pipeline control when isolating write-isolation cost.
That control is not unchanged RedisStore behavior. Neither comparison establishes
equivalent replication, failure recovery, or persistence implementation.

## Adaptive mutation budgets

Candidate groups contain at most 128 logical keys and 8 MiB of new payload.
Inside the transaction, read and validate all old roots before any writes.
The number of distinct application records that can be mutated is:

```text
replace: sum(1 + max(old_chunk_count, new_chunk_count))
delete:  sum(1 + old_chunk_count) for existing roots
```

Fixed-position chunk IDs make the maximum, not the sum, correct for replacement:
overlapping positions are rewritten, trailing old positions are deleted.
Reject a candidate above 4,096 application records before building any writes.
Abort that read-only transaction, halve the group, and retry each half with new
transactional reads. Splitting makes bounded progress down to a single logical
value (at most 129 records). A concurrent grow after the budget read invalidates
the transaction's read set; every retry recomputes the budget.

The record budget does not bound old payload bytes, server transaction-monitor
overhead, replication, or total caller memory. A limit on new payload is not a
claim that old-chunk cleanup is small.

## Validation and measurement

Unit tests cover dispatch boundaries, complete input validation, per-row errors,
stream cleanup, conditional failures, uncertain errors, record budgets, and
recursive splitting. Real SC/MRT tests cover inline/chunk transitions,
deterministic batch races, ABA, per-key partial success, chunked rollback,
corruption, and worst-case old-chunk cleanup.

Compare the unchanged baseline, optimized adapter, and unchanged RedisStore in
the separate benchmark repository. Record exact source hashes, runtime builds,
storage/acknowledgment settings, latency distributions, wall-clock throughput,
and resource telemetry. Preserve all original reports. Resource isolation must
distinguish CPU affinity from independent physical CPUs or machines. The local
two-logical-CPU/one-physical-core host cannot establish production scale-out or
an out-of-RAM Redis crossover.

### Direct-guard comparison

The sibling benchmark checkout retains the selected aggregate report under
`langchain/reports/inline-batches-20261125/REPORT.md`, with HTML/PDF, charts,
source archives, raw observations, and `verification.json`. It compares the
adapter immediately before this revision, the direct-guard adapter, unchanged
RedisStore, and a separately labeled nontransactional Redis pipeline control.
The baseline already includes the earlier read-batching and adaptive-budget
optimizations; it is not the original adapter from the first benchmark.

Four callers, 50/50 reads and overwrites, calls/second:

| Profile / resident keys | Keys per call | Aerospike before | Aerospike after | RedisStore | Redis per-key control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volatile / 1m | 1 | 2,528.8 | 3,169.1 | 4,539.5 | 5,138.3 |
| Volatile / 1m | 32 | 297.9 | 399.9 | 1,323.3 | 1,232.8 |
| Per-write persistence / 10k | 1 | 1,097.1 | 1,156.4 | 681.1 | 732.8 |
| Per-write persistence / 10k | 32 | 17.9 | 34.3 | 545.7 | 396.2 |

Mixed bulk throughput improves approximately 34% volatile and 92% persistent.
Persistent point throughput changes only about 5%; do not overinterpret that
small single-pass difference. A separate warm, serial 32-key delete diagnostic
reduces mean Aerospike latency from 5.517 to 2.159 ms volatile, and from 78.455
to 37.304 ms persistent. It excludes verification/restoration and is not a
sustained throughput result.

The comparison contains **111 completed sustained cells out of 112 planned**,
with 1,069,821 retained observations. The old MRT adapter timed out in the
16-caller persistent 32-key mixed cell; no throughput is assigned to that
failure. The remaining ten Redis persistent cells were completed in a separate
run with identical timing/workload functions and client versions. Another 16
serial-delete cells retain 2,048 observations. The raw audit checks all retained
statistics, source identities, fixture counts, and resource settings.

The new inline workload does not use MRT, yet Redis still leads bulk operations.
This is no longer an unnecessary transaction-guarantee explanation for the
remaining gap. SDK/schema processing and storage I/O grouping still differ;
the experiment does not isolate every cost or equate per-record persistence
with Redis AOF grouping. Neither durability nor complete-value safety was
weakened to improve the numbers.

These are local experimental-stack results, not released-artifact compatibility
or production sizing evidence. The host has one physical core with two SMT
threads, and the million-key dataset fits Redis memory. There are no confidence
intervals or repetition study; unchanged read rates also vary across runs.
Independent-machine, larger durable-capacity, recovery, and scale-out work
remain outside the evidence. Enumeration/cursor behavior is unchanged.
