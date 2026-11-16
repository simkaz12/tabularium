"""In-memory storage adapter: the reference implementation and hermetic test backend.

Pure Python, no I/O. It is authoritative for semantics; every other adapter
must pass the same conformance suite.

- BM25: k1=1.5, b=0.75; tokens are lowercase ``[a-z0-9]+`` runs; an empty
  query returns an empty result.
- Vector search: exact cosine over stored non-null vectors. Buried rows have no
  vector and are never returned.
- Ranked visibility: nothing is ever stale here, so ``maintenance_debt()`` is
  permanently zero and ``upkeep()`` is a no-op. The conformance suite asserts
  the property so adapters with index lag cannot regress.
- Audit: ``set_validity`` and ``bury`` take a ``reason`` that the domain models
  have no column for; it is kept in ``_audit`` as ``(operation, reason)``
  tuples and is not exposed through the Protocol.
- Tombstones: a buried record leaves ``_rows`` and its tombstone goes to
  ``_tombstones``. The event log and pair registry keep their entries
  (ids and timestamps only).
- Registries: ``create_scope`` / ``create_facet`` / ``create_role`` are
  idempotent and keep registration order.
- Iteration: ``scan`` yields rows in insertion order.
- Sessions live in a separate append-only ``_sessions`` list, so they cannot
  reach ``vector_search`` / ``text_search`` / ``scan`` / ``scan_records`` by
  construction. ``search_sessions`` matches text by case-insensitive
  substring, not BM25; adapters agree on which rows match, not on order.

Concurrency fidelity: the mutating methods (``transition``, ``upsert``,
``set_validity``, ``update_projections``, ``bury`` and the others that carry
one) contain a bare ``await asyncio.sleep(0)`` between read and write. This
adapter has no I/O to yield on, so without it read-then-write would be
uninterruptible and ``asyncio.gather`` interleaving tests would pass for the
wrong reason. The yield changes no state; do not factor it away.
``transition``'s supersede edge and the other guarded writes re-check their
precondition after the yield against a fresh read, matching the compare-and-swap
semantics of ``LanceDBAdapter``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import datetime

from tabularium.clock import Clock, SystemClock
from tabularium.domain import (
    Decision,
    DecisionLifecycle,
    Fact,
    FactLifecycle,
    HypomnemaId,
    Scope,
    Session,
    Validity,
)
from tabularium.storage._revisit_pool import (
    PoolCandidate,
    in_revisit_pool,
    latest_review_per_target,
    order_due_candidates,
)
from tabularium.storage._supersede import (
    resolve_invalidation_target,
    resolve_replace_predecessor,
    resolve_replace_successor,
    resolve_unsupersede_append,
    resolve_unsupersede_write,
    resolve_withdraw_tombstone,
    run_unsupersede_repair,
)
from tabularium.storage._upsert_gate import check_upsert_allowed
from tabularium.storage.content_key import content_hash, content_key
from tabularium.storage.types import (
    BackfillTarget,
    CleanerPassRecord,
    CleanerState,
    DecisionInvalidation,
    DistinctPair,
    Filters,
    Hit,
    MaintenanceDebt,
    PredecessorFlip,
    ReadEvent,
    RecordJudgment,
    RecordReview,
    ReinforcementEvent,
    RepairReport,
    RevalidationEvent,
    RowMeta,
    Tombstone,
    TombstoneId,
    UnsupersedeEvent,
    VerdictRow,
)

_log = logging.getLogger("tabularium")

# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic/digit token split — deterministic."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    doc_freq: dict[str, int],
    num_docs: int,
    avg_dl: float,
) -> float:
    """BM25 score for a single (query, document) pair.

    Uses standard Okapi BM25: k1=1.5, b=0.75.  IDF uses the smooth
    Robertson variant: ``ln((N - df + 0.5) / (df + 0.5) + 1)`` which
    is always non-negative.
    """
    dl = len(doc_tokens)
    tf_map: dict[str, int] = defaultdict(int)
    for tok in doc_tokens:
        tf_map[tok] += 1

    score = 0.0
    for tok in set(query_tokens):
        df = doc_freq.get(tok, 0)
        if df == 0:
            continue
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
        tf = tf_map.get(tok, 0)
        numerator = tf * (_BM25_K1 + 1)
        denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / (avg_dl or 1))
        score += idf * numerator / denominator
    return score


# ---------------------------------------------------------------------------
# Cosine helper
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Exact cosine similarity.  Returns 0.0 for zero-norm vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Internal row
# ---------------------------------------------------------------------------


class _Row:
    """Internal storage unit: record, vector and embedding provenance.

    ``content_key`` / ``content_hash`` are derived from ``record.content`` on every
    write through the ``record`` setter, never supplied by a caller. Hence
    ``populate_content_keys`` finds zero rows on this adapter.

    ``vector`` and ``embedding_model`` are ``None`` together (a row awaiting its
    first embed, or a stale-model row before backfill). The ``record`` getter
    re-derives ``record.embedding_model`` from this row's own attribute on every
    read, because ``replace()``'s new-successor branch can store a domain object
    carrying whatever stamp its constructor used.
    """

    __slots__ = (
        "_record",
        "content_hash",
        "content_key",
        "embedding_model",
        "insert_order",
        "vector",
    )

    def __init__(
        self,
        record: Fact | Decision,
        vector: list[float] | None,
        embedding_model: str | None,
        insert_order: int,
    ) -> None:
        self.vector: list[float] | None = vector
        self.embedding_model = embedding_model
        self.insert_order = insert_order
        self.record = record

    @property
    def record(self) -> Fact | Decision:
        if self._record.embedding_model != self.embedding_model:
            return type(self._record).model_validate(
                {**self._record.model_dump(mode="python"), "embedding_model": self.embedding_model}
            )
        return self._record

    @record.setter
    def record(self, value: Fact | Decision) -> None:
        self._record = value
        self.content_key = content_key(value.content)
        self.content_hash = content_hash(value.content)


# ---------------------------------------------------------------------------
# Filter predicate helper
# ---------------------------------------------------------------------------


def _row_passes(row: _Row, filters: Filters) -> bool:
    """Return True if the row passes all Filters pre-conditions."""
    record = row.record
    if record.scope not in filters.scopes:
        return False
    if record.validity not in filters.validity:
        return False
    lc = record.lifecycle
    if lc not in filters.lifecycle_in:
        return False
    return True


# ---------------------------------------------------------------------------
# MemoryAdapter
# ---------------------------------------------------------------------------


class _MemoryRepairIO:
    """``RepairIO`` over the in-memory maps: a missing or buried id is absent from ``_rows``."""

    def __init__(self, adapter: MemoryAdapter) -> None:
        self._adapter = adapter

    async def list_events(self) -> list[UnsupersedeEvent]:
        return list(self._adapter._unsupersedes)

    async def load_record(self, id: HypomnemaId) -> Fact | Decision | None:
        row = self._adapter._rows.get(id)
        return row.record if row is not None else None

    async def load_tombstone(self, buried_id: HypomnemaId) -> Tombstone | None:
        return self._adapter._tombstone_by_buried.get(buried_id)

    async def mark_refused(self, edge_id: str) -> None:
        await self._adapter.mark_unsupersede_refused(edge_id, at=self._adapter._clock.now())

    async def withdraw_tombstone(self, buried_id: HypomnemaId) -> None:
        await self._adapter.withdraw_tombstone(buried_id)

    async def plant_tombstone(
        self, buried_id: HypomnemaId, reason: str, *, restores: HypomnemaId
    ) -> None:
        await self._adapter.plant_tombstone(buried_id, reason, restores=restores)

    async def restore(self, event: UnsupersedeEvent) -> None:
        await self._adapter.restore(
            event.predecessor_id,
            successor_id=event.successor_id,
            edge_id=event.edge_id,
            at=self._adapter._clock.now(),
        )


class MemoryAdapter:
    """In-memory ``StorageAdapter`` — pure Python, no external services.

    Usage::

        adapter = MemoryAdapter(embedding_model="nomic-embed-text", dim=768)
        await adapter.upsert(fact, vector)
        hits = await adapter.vector_search(qvec, filters=filters, k=5)
    """

    def __init__(
        self,
        *,
        embedding_model: str,
        dim: int,
        clock: Clock | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._embedding_model = embedding_model
        self._dim = dim
        self._clock: Clock = clock if clock is not None else SystemClock()

        # Live rows: id → _Row, in insertion order (Python 3.7+ dict is ordered)
        self._rows: dict[HypomnemaId, _Row] = {}
        self._insert_counter = 0

        # Tombstones: tombstone_id → Tombstone
        self._tombstones: dict[TombstoneId, Tombstone] = {}
        # Reverse index for O(1) lookup: buried record id → Tombstone
        self._tombstone_by_buried: dict[HypomnemaId, Tombstone] = {}
        # Fast membership check
        self._buried_ids: set[HypomnemaId] = set()

        # Audit list: id → [(operation, reason), ...]
        self._audit: dict[HypomnemaId, list[tuple[str, str]]] = defaultdict(list)

        # Event log: insertion-ordered list
        self._events: list[ReinforcementEvent] = []

        # Read-event log: insertion-ordered, separate from ``_events``, never read by
        # ``update_projections``.
        self._read_events: list[ReadEvent] = []

        # Distinct-pair registry: frozenset({lower, higher}) → DistinctPair
        self._distinct: dict[frozenset[HypomnemaId], DistinctPair] = {}

        # Registries (ordered, unique)
        self._scopes: list[Scope] = []
        self._facets: list[str] = []
        self._roles: list[str] = []

        # Sessions: append-only, kept apart from ``_rows`` so they are excluded from
        # every record search and scan by construction.
        self._sessions: list[Session] = []

        # Cleaner state: one row per scope, upserted. An absent key is ``None``, never
        # a manufactured default.
        self._cleaner_state: dict[Scope, CleanerState] = {}

        # Pass history: append-only, never keyed; two appends for one scope leave two
        # entries.
        self._cleaner_pass_history: list[CleanerPassRecord] = []

        # Record reviews: append-only; ``list_due_for_review`` takes the latest per
        # record.
        self._record_reviews: list[RecordReview] = []

        # Record judgments: append-only.
        self._record_judgments: list[RecordJudgment] = []

        # Unsupersede events: append-only; idempotent on ``edge_id`` at the call level,
        # enforced through ``_unsupersede_by_edge``.
        self._unsupersedes: list[UnsupersedeEvent] = []
        self._unsupersede_by_edge: dict[str, UnsupersedeEvent] = {}
        self._revalidations: list[RevalidationEvent] = []

        # Assessor verdicts: id -> VerdictRow in insertion order, updated in place by
        # ``update_verdicts``.
        self._verdicts: dict[str, VerdictRow] = {}

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        vector: list[float],
        *,
        filters: Filters,
        k: int,
    ) -> list[Hit]:
        """Exact cosine over stored non-null vectors, pre-filtered."""
        if len(vector) != self._dim:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self._dim}, got {len(vector)}"
            )
        results: list[tuple[float, _Row]] = []
        for row in self._rows.values():
            if not _row_passes(row, filters):
                continue
            if not row.vector:
                # Rows without a vector are skipped (buried rows have no entry
                # but we guard anyway).
                continue
            sim = _cosine(vector, row.vector)
            results.append((sim, row))

        # Sort descending by similarity, then stable by insert order
        results.sort(key=lambda t: (-t[0], t[1].insert_order))
        results = results[:k]

        return [
            Hit(record=row.record, score=score, rank=i + 1, leg="vector")
            for i, (score, row) in enumerate(results)
        ]

    # ------------------------------------------------------------------
    # Text search
    # ------------------------------------------------------------------

    async def text_search(
        self,
        query: str,
        *,
        filters: Filters,
        k: int,
    ) -> list[Hit]:
        """BM25 over content tokens, pre-filtered.  Empty query → empty list."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Collect candidate rows that pass pre-filters
        candidates: list[_Row] = [
            row for row in self._rows.values() if _row_passes(row, filters)
        ]
        if not candidates:
            return []

        # Build corpus stats
        tokenized: list[list[str]] = [_tokenize(row.record.content) for row in candidates]
        num_docs = len(candidates)
        avg_dl = sum(len(t) for t in tokenized) / num_docs

        doc_freq: dict[str, int] = defaultdict(int)
        for tokens in tokenized:
            for tok in set(tokens):
                doc_freq[tok] += 1

        # Score each candidate
        scored: list[tuple[float, _Row]] = []
        for row, tokens in zip(candidates, tokenized, strict=True):
            score = _bm25_score(query_tokens, tokens, doc_freq, num_docs, avg_dl)
            if score > 0.0:
                scored.append((score, row))

        # Sort descending by score, then stable by insert order
        scored.sort(key=lambda t: (-t[0], t[1].insert_order))
        scored = scored[:k]

        return [
            Hit(record=row.record, score=score, rank=i + 1, leg="text")
            for i, (score, row) in enumerate(scored)
        ]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def upsert(
        self,
        record: Fact | Decision,
        vector: list[float] | None,
    ) -> None:
        """Insert or update a record.

        Raises ``ValueError`` on:
        - a supplied vector whose length differs from the configured dim;
        - a buried id (burial is permanent);
        - an existing Decision whose content differs (content is immutable);
        - a lifecycle or validity change on an existing row (use ``transition`` /
          ``set_validity``);
        - a changed Decision ``superseded_by``, ``supersession_chain`` or
          ``promoted_from``, or a changed Fact ``superseded_by`` (only ``replace``
          draws edges);
        - a changed projection field (``recall_count``, ``last_referenced_at``,
          ``feedback_score``, ``decay_due_at``; use ``update_projections``) or
          recurrence counter (use ``update_recurrence``).

        ``vector=None`` lands the row with a null vector and null ``embedding_model``
        (embedder unavailable); backfill fills it later. Lock-free: the existence
        check and the write are separated by ``asyncio.sleep(0)``.
        """
        if vector and len(vector) != self._dim:
            raise ValueError(
                f"Vector dimension mismatch (id={record.id!s}): "
                f"expected {self._dim}, got {len(vector)}"
            )
        # Buried ids cannot be resurrected via upsert — burial is permanent.
        if record.id in self._buried_ids:
            raise ValueError(
                f"Cannot upsert a buried id (id={record.id!s}): "
                "burial is permanent; the tombstone archive is monotonic"
            )

        existing = self._rows.get(record.id)
        if existing is not None:
            check_upsert_allowed(existing.record, record)
            # Yield point between validation and write; see the module docstring.
            await asyncio.sleep(0)
            # Update in-place: preserve insert_order
            existing.record = record
            existing.vector = list(vector) if vector else None
            existing.embedding_model = self._embedding_model if vector else None
        else:
            # Yield point; see the module docstring.
            await asyncio.sleep(0)
            self._insert_counter += 1
            self._rows[record.id] = _Row(
                record=record,
                vector=list(vector) if vector else None,
                embedding_model=self._embedding_model if vector else None,
                insert_order=self._insert_counter,
            )

    async def transition(
        self,
        id: HypomnemaId,
        lifecycle: FactLifecycle | DecisionLifecycle,
        *,
        at: datetime,
        audience_hint: tuple[str, ...] | None = None,
        provisional: bool | None = None,
    ) -> None:
        """Mutate the lifecycle of a record and persist ``updated_at = at``.

        Never draws a forward supersession edge: ``replace`` is the only path that
        sets ``superseded_by``, and a Decision never takes ``lifecycle="superseded"``
        here. Apart from ``replace``, ``transition`` is the only lifecycle-change
        path.

        ``audience_hint`` is tri-state: ``None`` leaves the stored value, a tuple
        replaces it, an empty tuple clears it (stored as ``None``, as the LanceDB
        codec does). ``provisional`` (Fact-only): ``None`` leaves it, ``True`` /
        ``False`` sets it. Both persist in the same update as ``lifecycle``.

        One ``asyncio.sleep(0)`` between the initial read and the write lets a
        concurrent caller interleave.
        """
        row = self._rows.get(id)
        if row is None:
            raise KeyError(f"Unknown id: {id!s}")

        record = row.record
        extra_updates: dict[str, object] = {}
        if audience_hint is not None:
            extra_updates["audience_hint"] = tuple(audience_hint) if audience_hint else None
        if provisional is not None:
            extra_updates["provisional"] = provisional
        updates: dict[str, object] = {"lifecycle": lifecycle, "updated_at": at, **extra_updates}

        # Yield point between read and write; see the module docstring.
        await asyncio.sleep(0)

        if isinstance(record, Fact):
            # Ordinary Fact transition: re-fetch after the yield so a concurrent supersede
            # or bury is not clobbered by the stale pre-yield record.
            current_row = self._rows.get(id)
            if current_row is None:
                raise KeyError(
                    f"Unknown id: {id!s} — buried while transitioning "
                    "(lost a concurrent race)"
                )
            row = current_row
            record = current_row.record

        # Produce updated record via model_validate (validates invariants)
        updated = type(record).model_validate(
            {**record.model_dump(mode="python"), **updates}
        )
        row.record = updated

    async def replace(
        self,
        successor: Fact | Decision,
        vector: list[float] | None,
        flips: Sequence[PredecessorFlip],
        *,
        invalidations: Sequence[DecisionInvalidation] = (),
        at: datetime,
    ) -> None:
        """See ``StorageAdapter.replace`` for the contract: the one atomic call for a
        supersession and for a Fact successor invalidating Decisions.

        Legality lives in ``storage._supersede``; this method does the fresh reads and
        the single yield point. The yield comes before every read, so a concurrent
        writer is observed rather than overwritten, and from the yield to the last
        write there is no further ``await``, so no in-process interleaving sees a
        partial ``replace``.
        """
        if not flips and not invalidations:
            raise ValueError("replace requires at least one flip or invalidation")

        seen_ids: set[HypomnemaId] = set()
        for flip in flips:
            if flip.id in seen_ids:
                raise ValueError(
                    f"replace: duplicate predecessor id={flip.id!s} in one call"
                )
            seen_ids.add(flip.id)
        flip_ids = frozenset(flip.id for flip in flips)
        for invalidation in invalidations:
            if invalidation.id in seen_ids:
                raise ValueError(
                    f"replace: duplicate id={invalidation.id!s} in one call"
                )
            seen_ids.add(invalidation.id)

        # Single yield point, before every read below; see the module docstring.
        await asyncio.sleep(0)

        existing_row = self._rows.get(successor.id)
        existing_successor = existing_row.record if existing_row is not None else None
        successor_to_write = resolve_replace_successor(
            successor=successor,
            existing=existing_successor,
            vector=vector,
            buried=successor.id in self._buried_ids,
        )
        successor_vector = existing_row.vector if existing_row is not None else vector
        if successor_vector is not None and len(successor_vector) != self._dim:
            raise ValueError(
                f"Vector dimension mismatch (id={successor.id!s}): "
                f"expected {self._dim}, got {len(successor_vector)}"
            )

        predecessor_rows: list[tuple[_Row, Fact | Decision]] = []
        for flip in flips:
            pred_row = self._rows.get(flip.id)
            existing_pred = pred_row.record if pred_row is not None else None
            updated_pred = resolve_replace_predecessor(
                predecessor=existing_pred,
                predecessor_id=flip.id,
                successor_id=successor.id,
                successor_type=type(successor),
                flip=flip,
                at=at,
            )
            assert pred_row is not None  # resolve_replace_predecessor already raised otherwise
            predecessor_rows.append((pred_row, updated_pred))

        invalidation_rows: list[tuple[_Row, Decision]] = []
        for invalidation in invalidations:
            target_row = self._rows.get(invalidation.id)
            existing_target = target_row.record if target_row is not None else None
            updated_target = resolve_invalidation_target(
                target=existing_target,
                target_id=invalidation.id,
                successor_id=successor.id,
                successor_type=type(successor),
                invalidation=invalidation,
                flip_ids=flip_ids,
            )
            assert target_row is not None
            invalidation_rows.append((target_row, updated_target))

        # No ``await`` from here to the last write.
        if existing_row is not None:
            existing_row.record = successor_to_write
            existing_row.vector = (
                list(successor_vector) if successor_vector is not None else None
            )
            # ``embedding_model`` stays as it is: the vector is retained, not re-embedded,
            # and relabeling it would hide it from a model-migration backfill scan.
        else:
            self._insert_counter += 1
            self._rows[successor.id] = _Row(
                record=successor_to_write,
                vector=list(successor_vector) if successor_vector is not None else None,
                embedding_model=(
                    self._embedding_model if successor_vector is not None else None
                ),
                insert_order=self._insert_counter,
            )

        for pred_row, updated_pred in predecessor_rows:
            pred_row.record = updated_pred

        for target_row, updated_target in invalidation_rows:
            target_row.record = updated_target

    async def set_validity(
        self,
        id: HypomnemaId,
        valid: bool,
        reason: str,
    ) -> None:
        """Flip validity.  Same-value re-call is an idempotent no-op ack."""
        row = self._rows.get(id)
        if row is None:
            raise KeyError(f"Unknown id: {id!s}")
        new_value: Validity = "valid" if valid else "invalid"
        if row.record.validity == new_value:
            # Idempotent no-op
            return
        self._audit[id].append(("set_validity", reason))
        # Yield point; see the module docstring.
        await asyncio.sleep(0)
        updated = type(row.record).model_validate(
            {**row.record.model_dump(mode="python"), "validity": new_value}
        )
        row.record = updated

    async def revalidate(
        self, id: HypomnemaId, reason: str, *, at: datetime
    ) -> RevalidationEvent | None:
        """See ``StorageAdapter.revalidate``. Writes the event row, then flips the row,
        in one uninterrupted stretch after the single yield point."""
        row = self._rows.get(id)
        if row is None:
            raise KeyError(f"Unknown id: {id!s}")
        record = row.record
        if not isinstance(record, Decision) or record.invalidated_by is None:
            return None
        assert record.invalidation_reason is not None, (
            "invalidated_by and invalidation_reason are set together in one "
            "transaction (replace's invalidations=) — one set without the other "
            "is a storage-layer bug, never a legitimate state"
        )

        await asyncio.sleep(0)

        event = RevalidationEvent(
            id=uuid.uuid4(),
            decision_id=id,
            withdrawn_cause=HypomnemaId(uuid.UUID(record.invalidated_by)),
            withdrawn_reason=record.invalidation_reason,
            reason=reason,
            at=at,
        )
        self._revalidations.append(event)
        updated = type(record).model_validate(
            {
                **record.model_dump(mode="python"),
                "validity": "valid",
                "invalidated_by": None,
                "invalidation_reason": None,
            }
        )
        row.record = updated
        return event

    async def list_revalidations(self, decision_id: HypomnemaId) -> list[RevalidationEvent]:
        """See ``StorageAdapter.list_revalidations``."""
        return [e for e in self._revalidations if e.decision_id == decision_id]

    async def bury(
        self,
        id: HypomnemaId,
        reason: str,
        *,
        model_generated: bool = False,
    ) -> TombstoneId:
        """Bury a record: remove it from the live store and write its tombstone.

        ``model_generated=True`` marks a tombstone produced by an automated system
        rather than a human decision.

        Every record, Fact or Decision, passes the same single yield point. The
        post-yield segment has no ``await``, so a ``transition`` that resumes after a
        ``bury`` re-reads, finds the row gone and raises ``KeyError``.
        """
        row = self._rows.get(id)
        if row is None:
            if id in self._buried_ids:
                raise KeyError(f"Already buried: {id!s}")
            raise KeyError(f"Unknown id: {id!s}")

        record = row.record
        self._audit[id].append(("bury", reason))

        # Yield point; see the module docstring.
        await asyncio.sleep(0)

        tombstone_id = uuid.uuid4()
        tombstone = Tombstone(
            id=tombstone_id,
            buried_id=id,
            type=record.type,
            reason=reason,
            timestamp=self._clock.now(),  # wall-clock burial instant
            model_generated=model_generated,
        )
        self._tombstones[tombstone_id] = tombstone
        self._tombstone_by_buried[id] = tombstone
        self._buried_ids.add(id)

        # Drop from live store (content + vector + text-index row — all in _rows)
        del self._rows[id]

        return tombstone_id

    # ------------------------------------------------------------------
    # Startup population passes
    # ------------------------------------------------------------------

    async def populate_content_keys(self) -> int:
        """See ``StorageAdapter.populate_content_keys``.

        Returns 0 on a store built through ``save``: the ``_Row.record`` setter derives
        both columns on every write. The per-row check stays real rather than a
        hardcoded 0.
        """
        written = 0
        for row in self._rows.values():
            if row.content_key is None or row.content_hash is None:
                row.content_key = content_key(row.record.content)
                row.content_hash = content_hash(row.record.content)
                written += 1
        return written

    async def populate_edge_ids(self) -> int:
        """See ``StorageAdapter.populate_edge_ids``."""
        written = 0
        for row in self._rows.values():
            record = row.record
            if record.superseded_by is not None and record.supersession_edge_id is None:
                row.record = type(record).model_validate(
                    {**record.model_dump(mode="python"), "supersession_edge_id": str(uuid.uuid4())}
                )
                written += 1
        return written

    # ------------------------------------------------------------------
    # Unsupersede
    # ------------------------------------------------------------------

    async def append_unsupersede(self, event: UnsupersedeEvent) -> UnsupersedeEvent:
        """See ``StorageAdapter.append_unsupersede``; idempotent on ``event.edge_id``.

        An existing row is a retry only when predecessor, successor and arm all match
        ``event``; otherwise the collision is refused. A matching retry whose
        ``refused_at`` is set is reopened: the row returned by
        ``resolve_unsupersede_append`` (``refused_at`` cleared) is written back before
        returning.
        """
        existing = self._unsupersede_by_edge.get(event.edge_id)
        if existing is not None:
            return self._resolve_and_reopen_unsupersede(existing=existing, event=event)
        # Yield point; see the module docstring.
        await asyncio.sleep(0)
        existing = self._unsupersede_by_edge.get(event.edge_id)
        if existing is not None:
            return self._resolve_and_reopen_unsupersede(existing=existing, event=event)
        self._unsupersedes.append(event)
        self._unsupersede_by_edge[event.edge_id] = event
        return event

    def _resolve_and_reopen_unsupersede(
        self, *, existing: UnsupersedeEvent, event: UnsupersedeEvent
    ) -> UnsupersedeEvent:
        resolved = resolve_unsupersede_append(existing=existing, event=event)
        if resolved.refused_at != existing.refused_at:
            self._replace_unsupersede(resolved)
        return resolved

    def _replace_unsupersede(self, updated: UnsupersedeEvent) -> None:
        """Overwrite an existing event in both ``_unsupersede_by_edge`` and
        ``_unsupersedes``. ``UnsupersedeEvent`` is frozen, so both structures get the
        new object."""
        self._unsupersede_by_edge[updated.edge_id] = updated
        for i, existing_event in enumerate(self._unsupersedes):
            if existing_event.edge_id == updated.edge_id:
                self._unsupersedes[i] = updated
                break

    async def get_unsupersede(self, edge_id: str) -> UnsupersedeEvent | None:
        """See ``StorageAdapter.get_unsupersede``."""
        return self._unsupersede_by_edge.get(edge_id)

    async def list_unsupersedes(self) -> list[UnsupersedeEvent]:
        """See ``StorageAdapter.list_unsupersedes``."""
        return list(self._unsupersedes)

    async def mark_unsupersede_refused(self, edge_id: str, *, at: datetime) -> None:
        """See ``StorageAdapter.mark_unsupersede_refused``."""
        existing = self._unsupersede_by_edge.get(edge_id)
        if existing is None:
            raise KeyError(f"No undo event for edge_id={edge_id!r}")
        self._replace_unsupersede(existing.model_copy(update={"refused_at": at}))

    async def plant_tombstone(
        self, buried_id: HypomnemaId, reason: str, *, restores: HypomnemaId
    ) -> TombstoneId:
        """See ``StorageAdapter.plant_tombstone``.

        Writes the tombstone without removing ``buried_id`` from ``_rows`` or adding it
        to ``_buried_ids``; ``restore`` removes the row in the same transaction that
        restores the predecessor. Refuses a second tombstone for the same
        ``buried_id``, checked again after the yield point.
        """
        row = self._rows.get(buried_id)
        if row is None:
            raise KeyError(f"Unknown or buried id: {buried_id!s}")
        if self._tombstone_by_buried.get(buried_id) is not None:
            raise KeyError(f"Already buried: {buried_id!s}")
        # Yield point; see the module docstring.
        await asyncio.sleep(0)
        if self._tombstone_by_buried.get(buried_id) is not None:
            raise KeyError(f"Already buried: {buried_id!s}")
        tombstone_id = uuid.uuid4()
        tombstone = Tombstone(
            id=tombstone_id,
            buried_id=buried_id,
            type=row.record.type,
            reason=reason,
            timestamp=self._clock.now(),
            model_generated=False,
            restores=restores,
            restores_content_hash=content_hash(row.record.content),
        )
        self._tombstones[tombstone_id] = tombstone
        self._tombstone_by_buried[buried_id] = tombstone
        return tombstone_id

    async def withdraw_tombstone(self, buried_id: HypomnemaId) -> None:
        """See ``StorageAdapter.withdraw_tombstone``."""
        tombstone = self._tombstone_by_buried.get(buried_id)
        row = self._rows.get(buried_id)
        buried = row.record if row is not None else None
        resolve_withdraw_tombstone(buried_id=buried_id, tombstone=tombstone, buried=buried)

        assert tombstone is not None
        del self._tombstones[tombstone.id]
        del self._tombstone_by_buried[buried_id]

    async def restore(
        self,
        predecessor_id: HypomnemaId,
        *,
        successor_id: HypomnemaId,
        edge_id: str,
        at: datetime,
    ) -> None:
        """See ``StorageAdapter.restore``."""
        # Yield point; see the module docstring.
        await asyncio.sleep(0)
        pred_row = self._rows.get(predecessor_id)
        existing_pred = pred_row.record if pred_row is not None else None
        updated_pred = resolve_unsupersede_write(
            predecessor=existing_pred,
            predecessor_id=predecessor_id,
            successor_id=successor_id,
            edge_id=edge_id,
            at=at,
        )
        assert pred_row is not None
        pred_row.record = updated_pred
        if isinstance(updated_pred, Decision):
            self._rows.pop(successor_id, None)
            self._buried_ids.add(successor_id)

    async def repair_unsupersedes(self) -> RepairReport:
        """See ``StorageAdapter.repair_unsupersedes``."""
        return await run_unsupersede_repair(_MemoryRepairIO(self))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_by_id(self, id: HypomnemaId) -> Fact | Decision:
        """Retrieve a record by id.  Raises ``KeyError`` if unknown or buried."""
        row = self._rows.get(id)
        if row is None:
            raise KeyError(f"Unknown or buried id: {id!s}")
        return row.record

    async def find_by_content_key(
        self, key: str, scopes: list[Scope]
    ) -> list[Fact | Decision]:
        """See ``StorageAdapter.find_by_content_key``.

        Every row carries a derived ``content_key``, so there is no unbackfilled
        legacy row to guard against.
        """
        scope_set = set(scopes)
        return [
            row.record
            for row in self._rows.values()
            if row.content_key == key and row.record.scope in scope_set
        ]

    async def scan(
        self,
        *,
        lifecycle_in: frozenset[FactLifecycle | DecisionLifecycle] | None = None,
    ) -> AsyncIterator[RowMeta]:
        """Yield metadata-only rows in insertion order; buried rows are excluded.

        ``async def`` returning an async generator object, not an async generator
        function: callers write ``async for meta in await adapter.scan():``. This is
        the shape the Protocol declares (``-> AsyncIterator[RowMeta]``).
        """
        return self._rows_to_meta(self._snapshot(), lifecycle_in=lifecycle_in)

    def _snapshot(self) -> list[_Row]:
        """Insertion-order snapshot of live rows, shared by ``scan`` and ``scan_records``."""
        return sorted(self._rows.values(), key=lambda r: r.insert_order)

    @staticmethod
    def _rows_to_meta(
        rows: list[_Row],
        *,
        lifecycle_in: frozenset[FactLifecycle | DecisionLifecycle] | None,
    ) -> AsyncIterator[RowMeta]:
        """Async generator yielding RowMeta for each row in ``rows``."""

        async def _gen() -> AsyncIterator[RowMeta]:
            for row in rows:
                if lifecycle_in is not None and row.record.lifecycle not in lifecycle_in:
                    continue
                rec = row.record
                feedback_score = getattr(rec, "feedback_score", 0.0)
                decay_due_at = getattr(rec, "decay_due_at", None)
                yield RowMeta(
                    id=rec.id,
                    type=rec.type,
                    scope=rec.scope,
                    lifecycle=rec.lifecycle,
                    validity=rec.validity,
                    recall_count=rec.recall_count,
                    last_referenced_at=rec.last_referenced_at,
                    feedback_score=feedback_score,
                    decay_due_at=decay_due_at,
                    embedding_model=row.embedding_model,
                )

        return _gen()

    async def scan_records(
        self,
        *,
        lifecycle_in: frozenset[FactLifecycle | DecisionLifecycle] | None = None,
    ) -> AsyncIterator[Fact | Decision]:
        """Yield full domain records in insertion order; buried rows are excluded.

        Same coroutine-returning-async-iterator shape as ``scan``, and the same
        snapshot: a ``bury()`` racing this call after the snapshot does not remove a
        record already captured in it.
        """
        snapshot = self._snapshot()

        async def _gen() -> AsyncIterator[Fact | Decision]:
            for row in snapshot:
                if lifecycle_in is not None and row.record.lifecycle not in lifecycle_in:
                    continue
                yield row.record

        return _gen()

    # ------------------------------------------------------------------
    # Event log + projections
    # ------------------------------------------------------------------

    async def append_event(self, event: ReinforcementEvent) -> None:
        """Append a reinforcement event to the log."""
        self._events.append(event)

    async def query_events(
        self,
        *,
        target_id: HypomnemaId | None = None,
        source: str | None = None,
    ) -> list[ReinforcementEvent]:
        """Return events filtered by optional ``target_id`` and ``source``, ascending by ``at``.

        ``list.sort`` is stable, so events with equal ``at`` keep insertion order.
        """
        results = []
        for ev in self._events:
            if target_id is not None and ev.target_id != target_id:
                continue
            if source is not None and ev.source != source:
                continue
            results.append(ev)
        results.sort(key=lambda ev: ev.at)
        return results

    # ------------------------------------------------------------------
    # Read events: a separate log from ``_events``
    # ------------------------------------------------------------------

    async def append_read_event(self, event: ReadEvent) -> None:
        """Append a read event to the log."""
        self._read_events.append(event)

    # ------------------------------------------------------------------
    # Assessor verdicts: id -> VerdictRow; see ``storage.types.VerdictRow``.
    # ------------------------------------------------------------------

    async def append_verdicts(self, rows: Sequence[VerdictRow]) -> None:
        """Append new rows keyed by their own ``id``; never merges against an existing row."""
        for row in rows:
            self._verdicts[row.id] = row

    async def update_verdicts(self, rows: Sequence[VerdictRow]) -> None:
        """Overwrite existing rows in place, matched by ``id``.

        The row count never changes for existing ids. An ``id`` not already present
        is inserted here, whereas LanceDB's matched-only ``merge_insert`` skips it.
        """
        for row in rows:
            self._verdicts[row.id] = row

    async def verdicts_by_bounce(self, bounce_id: str) -> list[VerdictRow]:
        """Every row carrying ``bounce_id``, ordered by ``probe_rank``."""
        rows = [row for row in self._verdicts.values() if row.bounce_id == bounce_id]
        rows.sort(key=lambda row: (row.probe_rank is None, row.probe_rank))
        return rows

    async def pending_verdicts(self) -> list[VerdictRow]:
        """Every row with ``status="pending"``."""
        return [row for row in self._verdicts.values() if row.status == "pending"]

    async def verdicts_by_effect(self, ref: str) -> list[VerdictRow]:
        """Every row whose ``effect_ref`` or ``new_record_id`` equals ``ref``."""
        return [
            row
            for row in self._verdicts.values()
            if row.effect_ref == ref or row.new_record_id == ref
        ]

    async def verdicts_for_record(self, id: HypomnemaId) -> list[VerdictRow]:
        """Every row naming ``id`` as ``candidate_id`` or ``new_record_id``, newest first."""
        id_str = str(id)
        rows = [
            row
            for row in self._verdicts.values()
            if row.candidate_id == id_str or row.new_record_id == id_str
        ]
        rows.sort(key=lambda row: row.created_at, reverse=True)
        return rows

    async def scrub_verdict_text(self, candidate_id: HypomnemaId) -> None:
        """Null ``incoming_text`` on every row naming ``candidate_id``."""
        id_str = str(candidate_id)
        for row_id, row in list(self._verdicts.items()):
            if row.candidate_id == id_str and row.incoming_text is not None:
                self._verdicts[row_id] = row.model_copy(update={"incoming_text": None})

    async def update_projections(
        self,
        id: HypomnemaId,
        *,
        recall_count: int,
        last_referenced_at: datetime | None,
        feedback_score: float,
        decay_due_at: datetime | None,
    ) -> None:
        """Persist recomputed projection fields verbatim.

        Re-fetches the row after the yield point: a ``bury()`` landing in that window
        deletes ``id`` from ``_rows``, and writing through the pre-yield reference
        would mutate an orphan. The re-fetch makes a lost race raise ``KeyError``,
        which callers (including the lock-free ``rebuild_projections``, which catches
        it) rely on.
        """
        row = self._rows.get(id)
        if row is None:
            raise KeyError(f"Unknown id: {id!s}")
        rec = row.record
        updates: dict[str, object] = {
            "recall_count": recall_count,
            "last_referenced_at": last_referenced_at,
        }
        if isinstance(rec, Fact):
            updates["feedback_score"] = feedback_score
            updates["decay_due_at"] = decay_due_at
        # Yield point; see the module docstring.
        await asyncio.sleep(0)
        # Re-fetch rather than trust the pre-yield ``row``: a concurrent ``bury()``
        # may have removed ``id``.
        current_row = self._rows.get(id)
        if current_row is None:
            raise KeyError(f"Unknown id: {id!s}")
        updated = type(current_row.record).model_validate(
            {**current_row.record.model_dump(mode="python"), **updates}
        )
        current_row.record = updated

    async def update_recurrence(
        self,
        id: HypomnemaId,
        *,
        owner_recurrence_count: int,
        agent_recurrence_count: int,
        escalation_baseline: int,
    ) -> None:
        """Persist the ember recurrence-ladder counters verbatim."""
        row = self._rows.get(id)
        if row is None:
            raise KeyError(f"Unknown id: {id!s}")
        rec = row.record
        if not isinstance(rec, Fact):
            raise ValueError(
                f"update_recurrence is Facts-only (id={id!s} is a {rec.type!r})"
            )
        updates: dict[str, object] = {
            "owner_recurrence_count": owner_recurrence_count,
            "agent_recurrence_count": agent_recurrence_count,
            "escalation_baseline": escalation_baseline,
        }
        # Yield point; see the module docstring.
        await asyncio.sleep(0)
        updated = Fact.model_validate({**rec.model_dump(mode="python"), **updates})
        row.record = updated

    # ------------------------------------------------------------------
    # Distinct-pair registry
    # ------------------------------------------------------------------

    async def mark_distinct(self, pair: DistinctPair) -> None:
        """Register a distinct pair; first-judgment-wins, idempotent. Lock-free."""
        key: frozenset[HypomnemaId] = frozenset({pair.lower_id, pair.higher_id})
        if key not in self._distinct:
            # Normalize: ensure lower_id < higher_id lexicographically
            a, b = pair.lower_id, pair.higher_id
            if str(a) > str(b):
                a, b = b, a
            normalized = DistinctPair(
                lower_id=a,
                higher_id=b,
                source=pair.source,
                judged_at=pair.judged_at,
                similarity=pair.similarity,
            )
            self._distinct[key] = normalized

    async def is_distinct(self, a: HypomnemaId, b: HypomnemaId) -> bool:
        """Return True if the pair is in the distinct registry (symmetric)."""
        key: frozenset[HypomnemaId] = frozenset({a, b})
        return key in self._distinct

    # ------------------------------------------------------------------
    # Embedding maintenance
    # ------------------------------------------------------------------

    async def backfill_targets(self) -> list[BackfillTarget]:
        """Return rows with a null vector or a stale ``embedding_model``, excluding buried rows."""
        return [
            BackfillTarget(
                id=row_id,
                content=row.record.content,
                content_hash=row.content_hash,
            )
            for row_id, row in self._rows.items()
            if row_id not in self._buried_ids
            and (row.vector is None or row.embedding_model != self._embedding_model)
        ]

    async def backfill(
        self,
        id: HypomnemaId,
        vector: list[float],
        *,
        embedding_model: str,
        expected_hash: str,
    ) -> bool:
        """Write only ``vector`` and ``embedding_model``, guarded by a ``content_hash`` check."""
        if len(vector) != self._dim:
            raise ValueError(
                f"Backfill vector dimension mismatch (id={id!s}): "
                f"expected {self._dim}, got {len(vector)}"
            )
        row = self._rows.get(id)
        if row is None or id in self._buried_ids:
            return False
        if row.content_hash != expected_hash:
            return False
        row.vector = list(vector)
        row.embedding_model = embedding_model
        return True

    # ------------------------------------------------------------------
    # Registries
    # ------------------------------------------------------------------

    async def create_scope(self, name: Scope) -> None:
        """Register a scope.  Idempotent."""
        if name not in self._scopes:
            self._scopes.append(name)

    async def list_scopes(self) -> list[Scope]:
        """Return registered scopes in registration order."""
        return list(self._scopes)

    async def create_facet(self, name: str) -> None:
        """Register a facet value.  Idempotent."""
        if name not in self._facets:
            self._facets.append(name)

    async def list_facets(self) -> list[str]:
        """Return registered facets in registration order."""
        return list(self._facets)

    async def create_role(self, name: str) -> None:
        """Register a role.  Idempotent."""
        if name not in self._roles:
            self._roles.append(name)

    async def list_roles(self) -> list[str]:
        """Return registered roles in registration order."""
        return list(self._roles)

    async def delete_role(self, name: str) -> None:
        """Deregister a role; a no-op if it is not registered.

        Never touches a record's ``audience_hint``: a deleted role's hint entries go
        inert and the delete never refuses on their account (see the ``StorageAdapter``
        Protocol).
        """
        if name in self._roles:
            self._roles.remove(name)

    # ------------------------------------------------------------------
    # Tombstone lookup
    # ------------------------------------------------------------------

    async def get_tombstone(self, id: HypomnemaId) -> Tombstone | None:
        """Return the tombstone for ``id`` if buried, else ``None``. Never raises."""
        return self._tombstone_by_buried.get(id)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def maintenance_debt(self) -> MaintenanceDebt:
        """Report permanently-zero debt: an in-memory store has no fragments or index lag."""
        return MaintenanceDebt(
            num_small_fragments=0,
            num_rows=len(self._rows),
            num_unindexed_rows=0,
        )

    async def upkeep(self) -> None:
        """No-op: nothing here is ever stale (see ``maintenance_debt``)."""
        return None

    async def reclaim(self) -> None:
        """No-op: there is no version history to prune.

        Irreversible on production adapters and never invoked automatically; a human
        names it via ``tabularium reclaim``.
        """
        return None

    # ------------------------------------------------------------------
    # Sessions: a parallel seam, kept apart from ``_rows``
    # ------------------------------------------------------------------

    async def append_session(self, session: Session) -> None:
        """Append a session to ``_sessions``; it is never visible to record search or scan."""
        self._sessions.append(session)

    async def latest_session_end(self, *, scope: str) -> datetime | None:
        """Maximum ``ended_at`` among sessions in ``scope``, or ``None``."""
        ends = [s.ended_at for s in self._sessions if s.scope == scope]
        if not ends:
            return None
        return max(ends)

    async def search_sessions(
        self,
        *,
        scopes: Sequence[str],
        query: str | None,
        since: datetime | None,
        until: datetime | None,
        limit: int,
    ) -> list[Session]:
        """Filter by scope and inclusive ``ended_at`` range, then match ``query``.

        ``query`` is a case-insensitive substring match, not BM25 (LanceDB's leg is
        BM25); adapters agree on which rows match, not on their order. Without
        ``query``: ``ended_at`` descending. With one: matches in insertion order. Both
        are capped by ``limit``.
        """
        scope_set = set(scopes)
        candidates = [s for s in self._sessions if s.scope in scope_set]
        if since is not None:
            candidates = [s for s in candidates if s.ended_at >= since]
        if until is not None:
            candidates = [s for s in candidates if s.ended_at <= until]

        if query is not None:
            needle = query.lower()
            candidates = [s for s in candidates if needle in s.content.lower()]
            return candidates[:limit]

        candidates = sorted(candidates, key=lambda s: s.ended_at, reverse=True)
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Cleaner state: per scope, apart from ``_rows``
    # ------------------------------------------------------------------

    async def get_cleaner_state(self, *, scope: Scope) -> CleanerState | None:
        """Return this scope's Cleaner state, or ``None`` if never judged.

        Absence is reported as ``None``; the caller decides what "never judged" means.
        """
        return self._cleaner_state.get(scope)

    async def set_cleaner_state(self, *, scope: Scope, state: CleanerState) -> None:
        """Upsert this scope's Cleaner state; writing twice leaves one entry."""
        self._cleaner_state[scope] = state

    # ------------------------------------------------------------------
    # Born-since query: the pool for one Cleaner pass
    # ------------------------------------------------------------------

    async def list_born_since(
        self, *, scope: Scope, since: datetime, limit: int
    ) -> list[Fact | Decision]:
        """Records in ``scope`` created strictly after ``since``, oldest first, up to ``limit``.

        Exactly one scope, with no implicit ``global`` union. Buried rows are already
        absent. No lifecycle or validity filter: Facts and Decisions both count. Ties
        on ``created_at`` break by insertion order.
        """
        matches = [
            row
            for row in self._rows.values()
            if row.record.scope == scope and row.record.created_at > since
        ]
        matches.sort(key=lambda r: (r.record.created_at, r.insert_order))
        return [row.record for row in matches[:limit]]

    # ------------------------------------------------------------------
    # Pass history: append-only, beside ``_cleaner_state``
    # ------------------------------------------------------------------

    async def append_cleaner_pass(self, *, record: CleanerPassRecord) -> None:
        """Append one Cleaner-pass record; two calls for one scope leave two entries."""
        self._cleaner_pass_history.append(record)

    async def list_cleaner_passes(
        self, *, scope: Scope, limit: int
    ) -> list[CleanerPassRecord]:
        """Return ``scope``'s pass-history rows, newest first, up to ``limit``.

        A scope with no rows returns ``[]``. Ties on ``pass_at`` break by reverse
        insertion order (the later append ranks first).
        """
        matches = [
            (i, r) for i, r in enumerate(self._cleaner_pass_history) if r.scope == scope
        ]
        matches.sort(key=lambda pair: (pair[1].pass_at, pair[0]), reverse=True)
        return [r for _, r in matches[:limit]]

    # ------------------------------------------------------------------
    # Record reviews: the revisit axis's rotation pool
    # ------------------------------------------------------------------

    async def append_review(self, review: RecordReview) -> None:
        """Append one review row; two calls for one ``target_id`` leave two entries."""
        self._record_reviews.append(review)

    async def list_due_for_review(
        self, *, scope: Scope, now: datetime, created_before: datetime, limit: int
    ) -> list[Fact]:
        """Facts in ``scope`` due for another revisit-axis look; see the Protocol for the contract.

        Only ``validity == "valid"`` Facts with ``superseded_by is None`` qualify.
        Buried rows are already absent from ``_rows``.
        """
        latest_by_target = latest_review_per_target(
            review for review in self._record_reviews if review.scope == scope
        )
        pool = self._revisit_pool(scope=scope, created_before=created_before)
        return order_due_candidates(pool, latest_by_target, now=now)[:limit]

    def _revisit_pool(
        self, *, scope: Scope, created_before: datetime
    ) -> Iterator[PoolCandidate[Fact]]:
        for row in self._rows.values():
            record = row.record
            if not isinstance(record, Fact):
                continue
            if in_revisit_pool(record, scope=scope, created_before=created_before):
                yield PoolCandidate(record.id, record.created_at, record)

    async def list_reviews_for(self, *, target_id: HypomnemaId) -> list[RecordReview]:
        """Every review row for ``target_id``, ascending by ``reviewed_at``.

        A narrow single-id lookback, not a pool-selection export (see the Protocol).
        """
        matches = [r for r in self._record_reviews if r.target_id == target_id]
        matches.sort(key=lambda r: r.reviewed_at)
        return matches

    # ------------------------------------------------------------------
    # Record judgments; see ``storage.types.RecordJudgment``
    # ------------------------------------------------------------------

    async def append_record_judgment(self, judgment: RecordJudgment) -> None:
        """Append one ``RecordJudgment`` row; two calls for one ``target_id`` leave two entries."""
        self._record_judgments.append(judgment)
