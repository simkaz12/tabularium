"""Storage package: seam types, the ``StorageAdapter`` Protocol, and adapters.

- Types: ``Filters``, ``Hit``, ``RowMeta``, ``ReinforcementEvent``,
  ``DistinctPair``, ``Tombstone``, ``TombstoneId``, ``CleanerState``
- Protocol: ``StorageAdapter``
- Reference adapter: ``tabularium.storage.memory.MemoryAdapter``
- Production adapter: ``LanceDBAdapter``
- Factory: ``open_adapter`` picks an adapter at store-open time
"""

from tabularium.storage.factory import open_adapter
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import (
    CleanerState,
    DistinctPair,
    Filters,
    Hit,
    ReinforcementEvent,
    RowMeta,
    StorageAdapter,
    Tombstone,
    TombstoneId,
)

__all__ = [
    "CleanerState",
    "DistinctPair",
    "Filters",
    "Hit",
    "LanceDBAdapter",
    "ReinforcementEvent",
    "RowMeta",
    "StorageAdapter",
    "Tombstone",
    "TombstoneId",
    "open_adapter",
]
