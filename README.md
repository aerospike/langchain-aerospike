# langchain-aerospike-kv

An independently installable LangChain integration maintained in
[`aerospike/langchain-aerospike`](https://github.com/aerospike/langchain-aerospike).
It depends on the official `langchain-core` package; no LangChain fork,
source checkout, or changes to LangChain itself are required.

`AerospikeByteStore` implements LangChain's unchanged `ByteStore`
(`BaseStore[str, bytes]`) interface. It stores opaque bytes, including values
up to 8 MiB, in a caller-provisioned Aerospike strong-consistency (SC)
namespace using conditional record operations and multi-record transactions (MRT).

- Distribution: `langchain-aerospike-kv`; import: `langchain_aerospike_kv`.
- Client: `aerospike-sdk`, **not** the legacy `aerospike` Python client.
- [Design and schema guide](DESIGN.md): decisions, costs, and validation limits.
- [Schema reference](SCHEMA.md): condensed on-disk contract derived from the guide.

## Prerequisites

Use an SDK/server combination that supports synchronous sessions, typed query
expressions, and MRT. A licensed, SC/MRT-capable server build, an operational
SC namespace, and an active roster are required. An image name or an open TCP
port does not establish these capabilities; Community Edition is not a
substitute for an SC-capable distribution.

The constructor inspects **every discovered node** using the `build` and
`namespace/<namespace>` info commands. The caller needs permission for those
read-only commands as well as data reads, writes, deletes, and set queries.
It fails closed on unsuitable or unverifiable configuration: no AP fallback.
The configured record limit must be at least 69,632 bytes; the local test
configuration uses 1 MiB. Original-key metadata must also fit the root record.
See [configuration checks](DESIGN.md#configuration-and-security).

### Optional local Docker server

An operator must first provision a licensed SC/MRT-capable distribution and
its configuration. Set `AEROSPIKE_CONFIG` to the absolute path of that
configuration and `AEROSPIKE_SERVER_IMAGE` to the appropriate image. The
released reference `aerospike/aerospike-server` below is a naming default,
**not a claim that its default image supports SC**. Supply any license
mounts/options required by your distribution before the image argument.

```bash
docker run --detach --name langchain-aerospike --ulimit nofile=65536:65536 --publish 127.0.0.1:3000:3000 --mount "type=bind,src=${AEROSPIKE_CONFIG:?Set an absolute SC config path},dst=/etc/aerospike/aerospike.conf,readonly" "${AEROSPIKE_SERVER_IMAGE:-aerospike/aerospike-server}"
```

For an isolated one-node test instance,
[`tests/integration_tests/aerospike.conf`](tests/integration_tests/aerospike.conf)
shows file-backed storage, `test_sc`, replication factor 1,
`strong-consistency true`, `max-record-size 1M`, `mrt-duration 10`, and a
loopback advertised client address. It is unauthenticated test configuration,
not a production template. Preserve a writable data directory and use a
client-reachable advertised address/port when changing the topology.

On **only a newly created isolated test instance**, initialize the roster
after startup. With a separate Aerospike tools installation, read the node ID,
substitute it for `<node-id>` below, stage it, and recluster:

```bash
asinfo -h 127.0.0.1 -p 3000 -v "node"
asinfo -h 127.0.0.1 -p 3000 -v "roster-set:namespace=test_sc;nodes=<node-id>"
asinfo -h 127.0.0.1 -p 3000 -v "recluster:"
asinfo -h 127.0.0.1 -p 3000 -v "roster:namespace=test_sc"
asinfo -h 127.0.0.1 -p 3000 -v "namespace/test_sc"
```

Wait for zero unavailable/dead partitions and no stop-writes conditions.
Never apply these roster changes to an existing cluster; its operator owns
roster administration. Server images need not contain `asinfo`, `aql`, or
`asadm`: install tools separately or use an appropriately networked tools
container. The Docker template above was not validated against a released
image; functional testing used a local, file-backed Enterprise SC instance.

## Quickstart

Install from a local checkout with `uv`:

```bash
git clone https://github.com/aerospike/langchain-aerospike.git
uv add ./langchain-aerospike
```

Run `uv add` from your application's project directory. This installs the
distribution `langchain-aerospike-kv` and its runtime dependencies,
`langchain-core` and `aerospike-sdk`. It does not install a fork of `langchain`.
The existing `langchain_aerospike_kv` import path remains unchanged.

The following example uses one caller-owned **synchronous** cluster/session
for both sync and async store calls. Environment reads belong to this
example, not the library. Defaults target `127.0.0.1:3000/test_sc`.

```python
import asyncio
import os
from uuid import uuid4

from aerospike_sdk.sync import ClusterDefinition

from langchain_aerospike_kv import AerospikeByteStore


async def use_async(store: AerospikeByteStore) -> None:
    await store.amset([("async/example", b"\x00\xff")])
    assert await store.amget(["async/example", "missing"]) == [b"\x00\xff", None]
    assert {key async for key in store.ayield_keys(prefix="async/")} == {
        "async/example"
    }
    await store.amdelete(["async/example"])


host = os.environ.get("AEROSPIKE_HOST", "127.0.0.1")
port = int(os.environ.get("AEROSPIKE_PORT", "3000"))
namespace = os.environ.get("AEROSPIKE_NAMESPACE", "test_sc")

with ClusterDefinition(host, port).connect() as cluster:
    session = cluster.create_session()
    store = AerospikeByteStore(
        session, namespace=namespace, set_name=f"lc_example_{uuid4().hex}"
    )
    try:
        store.mset([("sync/hello", b"hello"), ("sync/empty", b"")])
        assert store.mget(["sync/hello", "missing", "sync/hello", "sync/empty"]) == [
            b"hello",
            None,
            b"hello",
            b"",
        ]
        assert set(store.yield_keys(prefix="sync/")) == {"sync/hello", "sync/empty"}
        asyncio.run(use_async(store))
    finally:
        store.mdelete(["sync/hello", "sync/empty", "async/example"])
```

The example chooses a unique set and deletes only its own keys. For persistent
application data, choose a stable set exclusively owned by this schema.
Reuse one cluster per process; the caller closes it only after all sync/async
operations finish. The store has no connection ownership or `close()` method.
Async methods are the base class's executor-backed wrappers, not a separate
native-async SDK client. In an existing event loop, `await use_async(store)`
instead of calling `asyncio.run`. Construction performs synchronous server
checks: initialize the cluster/session/store before entering the event loop,
or use `await asyncio.to_thread(...)` for synchronous initialization inside an
async application. Do not call the synchronous store methods directly from an
async task; the SDK rejects blocking calls on an event-loop thread. Cancelling
a coroutine is not a transaction cancellation API.

### Use with LangChain documents

If your application also depends on `langchain-classic`, its existing adapter
can serialize `Document` objects through this byte store. Inside the same
cluster context, with `store` still live:

```python
from langchain_classic.storage import create_kv_docstore
from langchain_core.documents import Document

documents = create_kv_docstore(store)
try:
    document = Document(page_content="Stored in Aerospike", metadata={"source": "demo"})
    documents.mset([("docs/example", document)])
    assert documents.mget(["docs/example", "missing"]) == [document, None]
finally:
    documents.mdelete(["docs/example"])
```

The serialized bytes, not just `page_content`, must fit the 8 MiB limit.
The byte store itself performs no object deserialization. Treat any adapter
that reconstructs Python objects as a separate trust boundary.

## Behavior and limits

| Operation or limit | Contract |
| --- | --- |
| `mget` / `amget` | Input order and duplicates preserved; missing keys are `None` |
| `mset` / `amset` | Whole-value replacement; last duplicate wins; entire input validated before writes |
| `mdelete` / `amdelete` | Missing/repeated keys are harmless; complete-value durable deletion |
| `yield_keys` / `ayield_keys` | Literal prefix, including Unicode and NUL; no sorting or snapshot guarantee |
| Value size | `bytes` only; 0 through 8,388,608 bytes inclusive |
| Physical layout | Up to 65,536 bytes inline; larger values use 2 through 128 fixed-position chunks |
| Mutation groups | Up to 128 candidate keys and 8 MiB of new payload; chunked MRT work splits to enforce a 4,096-record write budget; **no cross-key isolation or rollback**, even within a group |
| Inline mutations | One native dispatch, no old-root lookup; atomic kind/format/key/count guard prevents overwriting chunked roots; missing roots can be created and missing deletes are no-ops |
| Native mutation batches | Implicit MRT disabled; no successful-key replay when another key fails; chunked values and definite-conflict recovery use MRT |
| Read batches | Up to 128 roots per dispatch; chunk reconstruction retains 16-key memory windows and 128-record physical batches |
| Expiry | Explicit never-expire writes; no TTL option is introduced |
| Enumeration cost | Full-set query, including chunk population; server-side root/prefix filter, key-only network projection; no secondary index |

Keys are UTF-8-encodable strings, not filesystem paths or regexes. Their
original text remains in root records; hashing physical keys is neither
encryption nor authorization. Namespace/set selection is not an access-control
boundary: configure SDK authentication/TLS and server ACLs separately.

Invalid input/configuration and corrupt reads raise errors. An inline overwrite/delete
can repair a recognized inline root's corrupt old payload/UUID without decoding
it; unknown layouts and key mismatches still fail closed.
Infrastructure, authorization, transaction, and per-row failures propagate;
they are not converted to cache misses. Concurrent chunked reads either return
a complete value, retry, or fail explicitly. Any multi-key mutation can partially
succeed, including within one batch. This removes the previous group-rollback
behavior. RedisStore's default transactional pipeline additionally prevents
interleaved batch execution; native Aerospike batches do not promise that
cross-key visibility. See the
[design](DESIGN.md#transactions-and-reads) for retry and corruption behavior.

## Repository development

Run commands from this repository's root. Python 3.11 or newer, `uv`, and
Make are required. Dependencies resolve from the package index, not from
adjacent LangChain or Aerospike checkouts. Install explicitly, then use
the Makefile's `uv run --no-sync` commands:

```bash
uv sync --all-groups --locked
make test
make lint
make type
make build
```

For an existing local SC server (the test layer alone reads these variables):

```bash
AEROSPIKE_HOST=127.0.0.1 AEROSPIKE_PORT=3000 AEROSPIKE_NAMESPACE=test_sc make integration_tests PYTEST_EXTRA=--require-live
```

`--require-live` fails rather than skipping if the server is unreachable.
A reachable but unsuitable server, inactive SC namespace, or permission error
always fails. Do not treat an all-skipped integration run as validation.

The live suite exercises upstream sync/async store contracts, document
serialization, `CacheBackedEmbeddings` cache reuse across store instances, and
`MultiVectorRetriever` loading chunked parent documents. It uses deterministic
test embeddings and an in-memory vector index; Aerospike persists the embedding
cache and parent documents. No model API credentials are needed.

`make local_server` verifies the prescribed local stack, adopts the existing
server without changing its roster or stopping it, proves MRT commit/abort/
cleanup in its own set, and runs strict live tests. It is a maintainer-only
helper requiring separately provisioned local SDK/server builds; ordinary
installation and `make integration_tests` do not require sibling checkouts.
Use `SERVER_ARGS="--sdk-path /absolute/path/to/sdk"` if that SDK checkout is
not adjacent to this repository. For a new isolated
temporary file-backed instance on a free port, use:

```bash
AEROSPIKE_PORT=3100 make local_server SERVER_ARGS=--create
```

The helper initializes only that new server's roster, never pulls an image,
and stops/removes only its token-owned container. It refuses an occupied
port; it does not repair an adopted server. Test fixtures use unique sets and
clean only their records, never truncate shared data. Empty set metadata and
durable-delete tombstones may remain after logical cleanup.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and
[SECURITY.md](SECURITY.md) for vulnerability reporting.
The repository uses the [Apache License 2.0](LICENSE). The extracted integration
retains its original [MIT license notice](LICENSE-MIT); see [NOTICE](NOTICE).
