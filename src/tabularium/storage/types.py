"""Seam types, re-exported from the modules that define them.

- ``tabularium.storage.records``: ids, filters, hits, row metadata, tombstones and pairs.
- ``tabularium.storage.audit``: event, pass-history, review, judgment and verdict rows.
- ``tabularium.storage.port``: the ``StorageAdapter`` Protocol.
"""

from __future__ import annotations

from tabularium.storage.audit import (
    CleanerPassRecord,
    CleanerState,
    ReadEvent,
    ReadSource,
    RecordJudgment,
    RecordReview,
    ReinforcementEvent,
    RevalidationEvent,
    UnsupersedeEvent,
    VerdictRow,
)
from tabularium.storage.port import StorageAdapter
from tabularium.storage.records import (
    BackfillTarget,
    DecisionInvalidation,
    DistinctPair,
    Filters,
    Hit,
    MaintenanceDebt,
    PredecessorFlip,
    RepairReport,
    RowMeta,
    Tombstone,
    TombstoneId,
)

__all__ = [
    "BackfillTarget",
    "CleanerPassRecord",
    "CleanerState",
    "DecisionInvalidation",
    "DistinctPair",
    "Filters",
    "Hit",
    "MaintenanceDebt",
    "PredecessorFlip",
    "ReadEvent",
    "ReadSource",
    "RecordJudgment",
    "RecordReview",
    "ReinforcementEvent",
    "RepairReport",
    "RevalidationEvent",
    "RowMeta",
    "StorageAdapter",
    "Tombstone",
    "TombstoneId",
    "UnsupersedeEvent",
    "VerdictRow",
]
