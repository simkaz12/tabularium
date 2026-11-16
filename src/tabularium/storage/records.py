"""Record-side seam types: ids, filters, search hits, row metadata, tombstones and pairs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from tabularium.domain import (
    Decision,
    DecisionLifecycle,
    Fact,
    FactLifecycle,
    HypomnemaId,
    InvalidationReason,
    Kind,
    Scope,
    SupersessionReason,
    Validity,
)

# ---------------------------------------------------------------------------
# TombstoneId
# ---------------------------------------------------------------------------

TombstoneId = uuid.UUID  # adapter-portable; not a HypomnemaId


# PredecessorFlip: one predecessor's half of a ``replace`` call


@dataclass(frozen=True, slots=True)
class PredecessorFlip:
    """One predecessor's flip inside a ``StorageAdapter.replace`` call.

    ``id`` names the predecessor; ``reason`` and ``note`` land on its row beside
    ``superseded_by``. The engine mints ``edge_id`` (``uuid4``) before the call, one
    per predecessor, so undo can key on the edge rather than on a (predecessor,
    successor, reason) triple that a reused successor makes ambiguous.
    """

    id: HypomnemaId
    reason: SupersessionReason
    note: str | None
    edge_id: str


@dataclass(frozen=True, slots=True)
class DecisionInvalidation:
    """One Decision target's invalidation inside a ``StorageAdapter.replace`` call.

    A Fact successor invalidating an in-force Decision lands in the same
    ``merge_insert`` as the successor row and every ``PredecessorFlip``.

    ``id`` names the Decision; ``reason`` is the supersession reason vocabulary
    narrowed to the two values an invalidation allows (``elaborated`` bounces
    earlier; see ``engine.admission.legal_exit``). It is not a reused
    ``PredecessorFlip``: a flip draws a ``superseded_by`` / edge-id forward pointer,
    and no Fact-to-Decision edge exists. This type carries only what
    ``invalidated_by`` and ``invalidation_reason`` need.
    """

    id: HypomnemaId
    reason: InvalidationReason


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class Filters(BaseModel):
    """Pre-filter pushed to every retrieval call.

    ``scopes`` is the engine-computed span:
    - recall in scope X → ``[X, global]``
    - dedup probe into X → same
    - read on global → ``[global]``

    Both ``validity`` and ``lifecycle_in`` are sets so that the adapter can
    apply native SQL/filter-column predicates without translating Python
    enumerations.

    The adapter MUST apply all three as pre-filters — not as post-hoc drops.
    """

    model_config = {"frozen": True}

    scopes: list[Scope]
    validity: frozenset[Validity]
    lifecycle_in: frozenset[FactLifecycle | DecisionLifecycle]


# ---------------------------------------------------------------------------
# Hit
# ---------------------------------------------------------------------------


class Hit(BaseModel):
    """A single retrieval result from one search leg.

    ``leg`` identifies which index produced the hit. ``score`` is the raw per-leg
    score: cosine similarity for the vector leg, BM25 for the text leg (higher is
    better for both). Adapters never fuse, rank by importance or alter scores;
    fusion and the importance boost live above the seam. ``rank`` is 1-indexed and
    ascending within the leg's result list.
    """

    model_config = {"frozen": True}

    record: Fact | Decision = Field(discriminator="type")
    score: float
    rank: int
    leg: Literal["vector", "text"]


# ---------------------------------------------------------------------------
# RowMeta — metadata-only scan row
# ---------------------------------------------------------------------------


class RowMeta(BaseModel):
    """Metadata-only row returned by ``scan``.

    Used by engine-internal maintenance (sweep, backfill, doctor), never
    model-facing. Content is deliberately absent. ``embedding_model`` is the
    provenance stamp set at upsert; ``None`` means the row was written before
    provenance was tracked (legacy stores) or has no vector yet.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    type: Kind
    scope: Scope
    lifecycle: FactLifecycle | DecisionLifecycle
    validity: Validity
    recall_count: int
    last_referenced_at: datetime | None
    feedback_score: float
    decay_due_at: datetime | None
    embedding_model: str | None


# ---------------------------------------------------------------------------
# DistinctPair
# ---------------------------------------------------------------------------


class DistinctPair(BaseModel):
    """A judged-distinct pair in the dedup pair registry.

    Stored normalized, ``lower_id`` < ``higher_id`` (lexicographic on the UUID
    string); the adapter normalizes on write so ``(a, b)`` and ``(b, a)`` share one
    row. First-judgment-wins: later ``mark_distinct`` calls for the pair are
    ignored. The registry is append-only and never pruned, even if an id is later
    buried.

    ``similarity`` is the raw cosine the gate's probe computed when
    ``confirm_distinct`` resolved the pair, threaded through rather than
    recomputed. ``None`` for pairs written before the field existed.
    """

    model_config = {"frozen": True}

    lower_id: HypomnemaId
    higher_id: HypomnemaId
    source: Literal["save", "compact"]
    judged_at: datetime
    similarity: float | None = None


# ---------------------------------------------------------------------------
# BackfillTarget
# ---------------------------------------------------------------------------


class BackfillTarget(BaseModel):
    """One row ``backfill_targets()`` reports as awaiting a backfill embed.

    ``content_hash`` is the raw-content hash captured at read time. The caller
    embeds ``content`` and passes the hash back to ``backfill`` as
    ``expected_hash``, so a concurrent edit, supersession or undo makes the write a
    no-op.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    content: str
    content_hash: str


# ---------------------------------------------------------------------------
# MaintenanceDebt
# ---------------------------------------------------------------------------


class MaintenanceDebt(BaseModel):
    """A snapshot of how much maintenance a store currently needs.

    ``num_small_fragments``: on-disk fragments below the vendor's merge target (the
    compaction half). ``num_rows``: the live row count of the primary records
    table. ``num_unindexed_rows``: rows written since the index (e.g. FTS) was last
    built or folded in (the ranking half); they rank stale until ``upkeep()`` runs.

    A pure report: it never mutates the store, and two calls with no writes between
    them return equal values.
    """

    model_config = {"frozen": True}

    num_small_fragments: int
    num_rows: int
    num_unindexed_rows: int


# ---------------------------------------------------------------------------
# RepairReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepairReport:
    """Counts from one ``StorageAdapter.repair_unsupersedes`` pass.

    ``completed`` events finished their undo, ``skipped`` events needed nothing, and
    ``refused`` events were withdrawn because the successor changed while the undo was
    pending.
    """

    completed: int
    skipped: int
    refused: int


# ---------------------------------------------------------------------------
# Tombstone
# ---------------------------------------------------------------------------


class Tombstone(BaseModel):
    """Tombstone written by ``bury``.

    - ``id``: the ``TombstoneId`` the adapter assigned.
    - ``buried_id``: the buried record's ``HypomnemaId``.
    - ``type``: the buried record's arm, ``"fact"`` or ``"decision"``.
    - ``reason``: the reason the caller gave ``bury``.
    - ``timestamp``: the wall-clock instant of the burial.
    - ``model_generated``: ``True`` for burials initiated by an automated system
      (e.g. the offline reconciliation judge), ``False`` for human ones.
    - ``restores``: the undo marker. ``None`` for every burial through ``bury()``;
      set only by ``plant_tombstone``, to the predecessor Decision that
      ``buried_id`` will restore once ``restore()`` lands. Startup repair completes
      a restores-marked tombstone only while the predecessor's ``superseded_by``
      still equals ``buried_id``. A legacy tombstone (``restores=None``) never
      authorises restoring anything.
    - ``restores_content_hash``: ``content_hash`` of ``buried_id``'s row when
      ``plant_tombstone`` wrote the tombstone; never set by ``bury()``. It lives
      here, not on the content-free ``UnsupersedeEvent``, beside the ``restores``
      pointer it guards. ``None`` on a tombstone planted before the field existed;
      repair treats ``None`` as a mismatch, since a NULL cannot prove the row was
      not edited since.
    """

    model_config = {"frozen": True}

    id: TombstoneId
    buried_id: HypomnemaId
    type: Kind
    reason: str
    timestamp: datetime
    model_generated: bool = False
    restores: HypomnemaId | None = None
    restores_content_hash: str | None = None
