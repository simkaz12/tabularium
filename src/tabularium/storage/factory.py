"""Store-open factory: picks and opens the storage adapter."""

from __future__ import annotations

from tabularium.domain import StoreBackend
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.port import StorageAdapter


async def open_adapter(
    store_adapter: StoreBackend,
    *,
    embedding_model: str,
    dim: int,
    store_path: str | None = None,
) -> StorageAdapter:
    """Choose and open a storage adapter at store-open time.

    ``store_adapter`` is ``"memory"`` or ``"lancedb"``; ``embedding_model`` is the
    litellm model string (provenance stamp); ``dim`` is the vector dimensionality;
    ``store_path`` is required for ``"lancedb"`` and ignored for ``"memory"``. The
    adapter is chosen once at open; there is no hot-swap.
    """
    if store_adapter == "memory":
        return MemoryAdapter(embedding_model=embedding_model, dim=dim)

    if store_adapter == "lancedb":
        if store_path is None:
            raise ValueError("store_path is required when store_adapter='lancedb'")

        return await LanceDBAdapter.open(
            store_path,
            embedding_model=embedding_model,
            dim=dim,
        )

    raise ValueError(f"Unknown store_adapter: {store_adapter!r}")
