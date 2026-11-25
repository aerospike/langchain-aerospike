from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests.base_store import (
    BaseStoreAsyncTests,
    BaseStoreSyncTests,
)

if TYPE_CHECKING:
    from langchain_aerospike_kv import AerospikeByteStore
    from tests._server import LiveStore

pytestmark = pytest.mark.integration


class TestAerospikeSync(BaseStoreSyncTests[bytes]):
    @pytest.fixture
    def kv_store(self, live_store: LiveStore) -> AerospikeByteStore:
        return live_store.store

    @pytest.fixture
    def three_values(self) -> tuple[bytes, bytes, bytes]:
        return b"", b"\x00\xff", b"third"


class TestAerospikeAsync(BaseStoreAsyncTests[bytes]):
    @pytest.fixture
    async def kv_store(self, live_store: LiveStore) -> AerospikeByteStore:
        return live_store.store

    @pytest.fixture
    def three_values(self) -> tuple[bytes, bytes, bytes]:
        return b"", b"\x00\xff", b"third"
