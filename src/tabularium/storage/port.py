"""The ``StorageAdapter`` Protocol.

``StorageAdapter`` is an async Protocol: every method is a coroutine so adapters
can do network or disk I/O (LanceDB) without blocking. ``MemoryAdapter`` meets
the same interface with trivial async bodies.

The adapter carries mechanisms only; it never fuses, computes importance or runs
the dedup gate. Those live above the seam in the engine. The seam guarantees:

- ``Filters`` are pre-filters pushed to the store, never post-filtered.
- ``k`` is a required ``int`` on every retrieval leg; there is no unbounded mode.
- ``vector_search`` skips rows whose stored vector is NULL (buried or never
  embedded).
- The event log and pair registry carry ids and timestamps only, never content,
  so ``bury`` is complete: content lives in one place.
- ``TombstoneId`` is a plain UUID newtype, kept apart from ``HypomnemaId`` so
  callers know what ``bury`` returned.
- Adapters are constructed with ``embedding_model: str`` and ``dim: int``.
  ``upsert`` stamps every row with the configured ``embedding_model`` when a
  vector is supplied; a ``None`` or empty vector writes a null vector and a
  null stamp instead of raising, and backfill fills the row in later.
  ``backfill_targets`` returns ids of rows with no vector or a stamp differing
  from the configured value.
- ``create_scope`` / ``create_facet`` / ``create_role`` are idempotent.
  ``delete_role`` is the only deregistration verb (scopes and facets have none)
  and is an idempotent no-op for an unregistered name.
- ``set_validity`` with the current value is an idempotent no-op.
- ``mark_distinct`` is first-judgment-wins and stores the pair normalized (lower
  UUID string first), so both orderings collide on one key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Protocol

from tabularium.domain import (
    Decision,
    DecisionLifecycle,
    Fact,
    FactLifecycle,
    Hypomnema,
    HypomnemaId,
    Scope,
    Session,
)
from tabularium.storage.audit import (
    CleanerPassRecord,
    CleanerState,
    ReadEvent,
    RecordJudgment,
    RecordReview,
    ReinforcementEvent,
    RevalidationEvent,
    UnsupersedeEvent,
    VerdictRow,
)
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

# ---------------------------------------------------------------------------
# StorageAdapter Protocol
# ---------------------------------------------------------------------------


class StorageAdapter(Protocol):
    """Pure-async storage seam.

    Every retrieval method requires a ``Filters`` argument; the only scopeless calls
    are the engine-internal maintenance family (``scan``, ``backfill_targets``,
    ``backfill``). Adapters are pure stores: they never fuse, never rank by
    importance and never import litellm. They enforce the structural invariants:
    Decision content immutability on re-upsert, supersession atomicity, set-once
    ``superseded_by`` and complete burial.

    Ranked visibility: a committed write must be correctly ranked, not merely
    findable. After maintenance every record ranks correctly; between maintenance
    runs, records written since the last run may rank stale. ``MemoryAdapter`` is
    never stale. Adapters with a build-time index snapshot (LanceDB FTS) close the
    window through ``upkeep()``, and ``maintenance_debt()`` reports it; the
    conformance suite asserts the property on both.

    ``set_validity`` and ``bury`` accept a ``reason`` the domain models have no
    column for; adapters keep it in an internal audit list or sidecar, not on the
    domain object.

    ``create_scope`` / ``create_facet`` / ``create_role`` are idempotent (a
    duplicate name is silently accepted), and the ``list_*`` methods return all
    registered names in registration order.
    """

    # ------------------------------------------------------------------
    # Retrieval legs: raw, never fused
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        vector: list[float],
        *,
        filters: Filters,
        k: int,
    ) -> list[Hit]:
        """Search by vector similarity.

        Returns up to ``k`` hits ranked descending by cosine similarity, after applying
        ``filters`` as a pre-filter. Rows whose stored vector is NULL (buried or never
        embedded) are skipped.
        """
        ...

    async def text_search(
        self,
        query: str,
        *,
        filters: Filters,
        k: int,
    ) -> list[Hit]:
        """Search by BM25 full-text score.

        Returns up to ``k`` hits ranked descending by BM25, after applying ``filters``
        as a pre-filter. An empty ``query`` returns an empty result, not an error.
        """
        ...

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def upsert(
        self,
        record: Fact | Decision,
        vector: list[float] | None,
    ) -> None:
        """Insert or update a record.

        Stamps the row with the configured ``embedding_model`` when ``vector`` is
        supplied. Rejects a changed ``content`` on an existing Decision id (content is
        immutable through this path). A ``None`` or empty ``vector`` lands the row with
        a null vector and null stamp instead of raising (embedder unavailable);
        backfill fills it in later.
        """
        ...

    async def transition(
        self,
        id: HypomnemaId,
        lifecycle: FactLifecycle | DecisionLifecycle,
        *,
        at: datetime,
        audience_hint: tuple[str, ...] | None = None,
        provisional: bool | None = None,
    ) -> None:
        """Change the lifecycle state of a record and persist ``updated_at = at``.

        Apart from ``replace``, this is the only path that mutates lifecycle.
        ``replace`` is the only path that draws a forward supersession edge, so this
        method never sets ``superseded_by`` and never accepts ``lifecycle="superseded"``
        for a Decision. ``updated_at`` means "time of last lifecycle transition":
        ``set_validity`` and ``update_projections`` do not touch it.

        ``audience_hint`` is tri-state:

        - ``None`` (default): leave the stored value untouched.
        - a tuple: replace the stored value.
        - ``()``: clear it.

        ``None`` is silence and ``()`` is an explicit un-hint; conflating them would
        make un-hinting impossible. A cleared hint reaches no role on the push tiers
        (the routing consequence is the engine's, in ``engine/context.py::routes_to``).

        ``provisional`` (Fact-only): ``None`` leaves the stored value, ``True`` /
        ``False`` sets it. Its only production caller sets ``False`` on an
        owner-supplied audience retag; only a new write through ``upsert`` can mint a
        provisional record. Both persist in the same row write as ``lifecycle`` and
        ``updated_at``.
        """
        ...

    async def replace(
        self,
        successor: Fact | Decision,
        vector: list[float] | None,
        flips: Sequence[PredecessorFlip],
        *,
        invalidations: Sequence[DecisionInvalidation] = (),
        at: datetime,
    ) -> None:
        """Land a supersession, or a Fact successor's invalidation of Decisions, atomically.

        The only path that draws a forward ``superseded_by`` edge (both arms) and the
        only path that sets ``invalidated_by`` / ``invalidation_reason``. The successor
        row, every predecessor flip and every invalidation target's row land in one
        ``merge_insert`` under one write-lock hold: all of it commits or none of it
        does. The arm is read from ``type(successor)``: every ``flip`` must name a
        predecessor of the same type; every ``invalidation`` target must be a Decision
        and ``successor`` must be a Fact.

        An existing successor keeps its stored vector and ``embedding_model`` (the
        ``supersede`` path). A new successor takes ``vector``, and ``vector=None`` lands
        it vector-less (embedder unavailable) for backfill to fill later.
        ``flips`` and ``invalidations`` may not both be empty; either alone is legal.

        Under the lock the method re-reads the successor, every ``flip.id`` and every
        ``invalidation.id`` fresh and refuses the whole call, writing nothing, on the
        first illegal referent:

        - the successor, if it exists, must be valid and unretired (a Fact:
          ``superseded_by is None`` and ``lifecycle != "shelved"``; a Decision:
          ``in_force``);
        - every predecessor must exist, have the successor's type, differ from the
          successor, and be supersedable (a Fact: valid and not already superseded; a
          Decision: ``in_force``), so one stale predecessor fails the whole call;
        - every invalidation target must exist, be a Decision that is ``in_force`` and
          valid, and not also be a flip predecessor (invalidation draws no edge).

        The engine mints ``flip.edge_id`` (``uuid4``) before calling; an invalidation
        carries no edge id. Raises ``ValueError`` on any legality failure.
        """
        ...

    async def set_validity(
        self,
        id: HypomnemaId,
        valid: bool,
        reason: str,
    ) -> None:
        """Flip the validity bit.

        Re-calling with the current value is an idempotent no-op. ``reason`` is
        required and kept in the adapter's internal audit list.
        """
        ...

    async def revalidate(
        self, id: HypomnemaId, reason: str, *, at: datetime
    ) -> RevalidationEvent | None:
        """Clear a Fact-drawn invalidation off a Decision.

        Writes the ``RevalidationEvent`` row first, then flips ``validity="valid"`` and
        nulls ``invalidated_by`` and ``invalidation_reason`` in one row write.

        Returns ``None`` and writes nothing when the Decision carries no invalidation
        (``invalidated_by is None``); this call decides whether there is anything to
        revalidate, so callers must not pre-check. The engine's ``set_validity`` is the
        sole caller, routing a Decision ``valid=True`` call here and falling through to
        the adapter's ``set_validity`` only on ``None``.

        A torn state (event written, row flip not, e.g. a crash between them) is
        repaired by re-running ``set_validity(valid=True)``: it finds the invalidation
        still set, writes a second event and retries the idempotent row flip.

        Raises:
            KeyError: unknown or buried id.
        """
        ...

    async def list_revalidations(self, decision_id: HypomnemaId) -> list[RevalidationEvent]:
        """Every revalidation event ever written for ``decision_id``.

        Revalidation is rare, so there is no filtered or paginated query.
        """
        ...

    async def bury(
        self,
        id: HypomnemaId,
        reason: str,
        *,
        model_generated: bool = False,
    ) -> TombstoneId:
        """Bury a record: drop its content, vector and text-index row, and write a tombstone.

        Returns the tombstone's ``TombstoneId``. Afterwards ``get_by_id(id)`` raises
        ``KeyError`` and ``vector_search`` / ``text_search`` never return the id. Event
        log and pair registry entries for the id are kept (ids and timestamps only;
        removing them would break lineage). ``model_generated=True`` marks tombstones
        produced by automated systems (e.g. the offline reconciliation judge), for
        model-slip auditing.
        """
        ...

    # ------------------------------------------------------------------
    # Startup population passes
    #
    # Run once, synchronously, from ``Tabularium.initialize()`` before the server
    # takes requests, never inside ``open()``. Both are resumable: the NULL
    # filter is the resume cursor, so a second call picks up where a crash left
    # off and never reassigns a filled value.
    # ------------------------------------------------------------------

    async def populate_content_keys(self) -> int:
        """Fill ``content_key`` and ``content_hash`` on every row where ``content_key IS NULL``.

        Both derive from the row's own ``content``, by the same computation every new
        write applies. Returns the number of rows written. ``MemoryAdapter`` derives
        both at write time, so it returns 0 there.
        """
        ...

    async def populate_edge_ids(self) -> int:
        """Assign a fresh ``uuid4`` ``supersession_edge_id`` to superseded rows that lack one.

        Returns the number of rows written. Never touches a row that already has an
        edge id.
        """
        ...

    # ------------------------------------------------------------------
    # Unsupersede
    # ------------------------------------------------------------------

    async def append_unsupersede(self, event: UnsupersedeEvent) -> UnsupersedeEvent:
        """Append an undo event, idempotent on ``event.edge_id``.

        A second append for an edge id on record returns the existing stored row; it
        never duplicates the row or overwrites the first attempt's ``reason`` and
        ``at``. The table is content-free: ids, the withdrawn supersession class and
        note, and the caller's retraction ``reason``.

        Exception: a matching existing row with ``refused_at`` set is written back with
        ``refused_at`` cleared. A fresh live ``unsupersede`` reopening a refused edge
        asks again, and the edge must stop reading as refused before the caller's
        ``plant_tombstone`` / ``restore`` sequence runs.
        """
        ...

    async def get_unsupersede(self, edge_id: str) -> UnsupersedeEvent | None:
        """Return the undo event for ``edge_id``, or ``None`` if none exists."""
        ...

    async def mark_unsupersede_refused(self, edge_id: str, *, at: datetime) -> None:
        """Stamp ``refused_at=at`` on the undo event for ``edge_id``.

        The first of ``repair_unsupersedes``'s two refusal writes, always followed by
        ``withdraw_tombstone``. Raises ``KeyError`` if no event exists for ``edge_id``.
        Writing the stamp before the withdrawal makes a crash between them recoverable:
        the next repair pass finds a refused event whose tombstone remains and finishes
        the withdrawal. Once refused, always refused, until a live retry reopens it.
        """
        ...

    async def list_unsupersedes(self) -> list[UnsupersedeEvent]:
        """Return every undo event, the pool ``repair_unsupersedes`` walks.

        Undo is rare, so this is the full list with no filtered or paginated query.
        """
        ...

    async def plant_tombstone(
        self, buried_id: HypomnemaId, reason: str, *, restores: HypomnemaId
    ) -> TombstoneId:
        """Write a tombstone for ``buried_id`` carrying ``restores``, without deleting its row.

        The Decision-arm half-step of undo: the row stays live and readable until
        ``restore`` deletes it in the transaction that restores ``restores``. Raises
        ``KeyError`` if ``buried_id`` is unknown or already buried. Unlike ``bury``, it
        never adds ``buried_id`` to the buried set.

        Captures ``buried_id``'s current ``content_hash`` into the tombstone's
        ``restores_content_hash``; ``repair_unsupersedes`` and the live undo path
        compare against it before deleting the row.
        """
        ...

    async def withdraw_tombstone(self, buried_id: HypomnemaId) -> None:
        """Delete a pending undo tombstone for ``buried_id`` without touching its row.

        The inverse of ``plant_tombstone``, used when the successor's content changed
        while the undo was pending and the undo is abandoned. Raises ``KeyError``,
        writing nothing, unless a fresh read shows a tombstone exists for
        ``buried_id``, it carries a real ``restores`` (never a legacy tombstone), and
        ``buried_id``'s row is present and, for a Decision, ``in_force``. A completed
        burial is permanent. Both adapters share the guard
        ``storage._supersede.resolve_withdraw_tombstone`` under their own lock.
        """
        ...

    async def restore(
        self,
        predecessor_id: HypomnemaId,
        *,
        successor_id: HypomnemaId,
        edge_id: str,
        at: datetime,
    ) -> None:
        """Reverse one supersession edge: the records-transaction half of undo, on both arms.

        Under the adapter's lock, re-reads ``predecessor_id`` fresh and raises
        ``ValueError``, writing nothing, unless its current ``superseded_by`` is
        ``successor_id`` and its ``supersession_edge_id`` is ``edge_id`` (the shared
        resolver is ``storage._supersede.resolve_unsupersede_write``).

        The restored row is built from that fresh read, so ``vector`` and
        ``embedding_model`` are unchanged. It clears ``superseded_by``,
        ``supersession_reason``, ``supersession_note`` and ``supersession_edge_id``; a
        Fact also gets ``validity="valid"``, a Decision ``lifecycle="in_force"`` with
        ``validity`` as it was.

        Fact arm: a one-row update; the successor is untouched. Decision arm: one
        storage transaction updates the predecessor and deletes ``successor_id``'s row
        (on LanceDB, one ``merge_insert`` with ``when_matched_update_all()`` and
        ``when_not_matched_by_source_delete()``). ``plant_tombstone`` must already have
        landed for the successor; ``restore`` never touches the tombstones table.
        """
        ...

    async def repair_unsupersedes(self) -> RepairReport:
        """Finish every undo a kill interrupted.

        Run from ``Tabularium.initialize()`` after ``populate_edge_ids``, never from
        ``open()``. For each event from ``list_unsupersedes()``, load the predecessor
        fresh and skip if it is missing or buried, or if its current
        ``supersession_edge_id`` is not the event's (a later generation). Skip with a
        WARNING if the edge id matches but ``superseded_by`` is not the event's
        ``successor_id`` (a data-integrity anomaly).

        Already-refused events (``refused_at`` set): if a tombstone naming this
        predecessor remains, the original refusal was interrupted between its two
        writes, so finish the withdrawal only, touching neither row, and count
        ``refused``. Otherwise the refusal is resolved history: count ``skipped``. The
        refusal is never re-derived from a hash compare once ``refused_at`` is set;
        only a live ``unsupersede`` retry (via ``append_unsupersede``) reopens it.

        Fresh events (``refused_at is None``): the Fact arm always restores. Decision
        arm: with no tombstone for the successor, ``plant_tombstone`` then ``restore``;
        with one whose ``restores_content_hash`` matches the successor's current
        content, ``restore`` directly; with one whose hash no longer matches (the
        successor was edited while the undo was pending), refuse: ``mark_unsupersede_
        refused`` then ``withdraw_tombstone`` (that order makes the crash recovery
        above possible), leave the predecessor superseded and the successor untouched,
        log a WARNING naming both ids, and count ``refused``. A legacy tombstone
        (``restores=None``) never authorises a restore, and a Decision buried through
        ordinary ``bury`` has no undo event, so this pass never visits it.

        A successor already withdrawn from earlier in the same pass is skipped for
        every later event naming it: ``replace`` lets one successor take several
        predecessors, and ``refused_at`` marks only the event that was refused.

        Returns a ``RepairReport``. Idempotent: a second call after a
        completed or refused pass reports ``completed == 0``.
        """
        ...

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_by_id(self, id: HypomnemaId) -> Fact | Decision:
        """Retrieve a record by id.

        Raises ``KeyError`` if the id is unknown or buried.
        """
        ...

    async def find_by_content_key(
        self, key: str, scopes: list[Scope]
    ) -> list[Hypomnema]:
        """Return every live row whose ``content_key`` is ``key`` and whose scope is in ``scopes``.

        Both kinds, every validity and lifecycle: the engine (``engine/admission.py``)
        applies the validity, lifecycle and kind filters; this is a pure key and scope
        lookup. A legacy row with ``content_key IS NULL`` never matches. Buried rows
        are excluded.
        """
        ...

    async def scan(
        self,
        *,
        lifecycle_in: frozenset[FactLifecycle | DecisionLifecycle] | None = None,
    ) -> AsyncIterator[RowMeta]:
        """Yield metadata-only rows for engine-internal maintenance; never model-facing.

        ``lifecycle_in=None`` yields all rows. Buried rows are excluded. Only
        ``MemoryAdapter`` guarantees insertion order. A coroutine that returns an async
        iterator: ``async for meta in await adapter.scan():``.
        """
        ...

    async def scan_records(
        self,
        *,
        lifecycle_in: frozenset[FactLifecycle | DecisionLifecycle] | None = None,
    ) -> AsyncIterator[Fact | Decision]:
        """Yield full domain records, the content-carrying sibling of ``scan``.

        Same filter contract as ``scan`` (``lifecycle_in=None`` yields all rows; buried
        rows excluded; engine-internal), but yields ``Fact | Decision``. It replaces a
        ``scan`` followed by one ``get_by_id`` per candidate with a single round trip,
        and is a coroutine that returns an async iterator like ``scan``.

        Snapshot semantics: a yielded record reflects the adapter's state at the scan,
        and a lifecycle or burial change racing the call is not synchronized against.
        """
        ...

    # ------------------------------------------------------------------
    # Event log and projections
    # ------------------------------------------------------------------

    async def append_event(self, event: ReinforcementEvent) -> None:
        """Append a reinforcement event to the append-only log."""
        ...

    async def query_events(
        self,
        *,
        target_id: HypomnemaId | None = None,
        source: str | None = None,
    ) -> list[ReinforcementEvent]:
        """Query events, optionally filtered by target id and/or source.

        Results are sorted ascending by ``at`` whatever the physical order, which
        LanceDB does not guarantee once fragments are added or compacted.
        ``compute_ewma`` is order-sensitive and trusts this order. Ties on identical
        ``at`` are adapter-defined (``MemoryAdapter`` keeps insertion order via a stable
        sort; LanceDB promises nothing), so callers may rely only on the ascending
        guarantee.
        """
        ...

    async def update_projections(
        self,
        id: HypomnemaId,
        *,
        recall_count: int,
        last_referenced_at: datetime | None,
        feedback_score: float,
        decay_due_at: datetime | None,
    ) -> None:
        """Persist recomputed projection fields verbatim; the formulas live above the seam."""
        ...

    async def update_recurrence(
        self,
        id: HypomnemaId,
        *,
        owner_recurrence_count: int,
        agent_recurrence_count: int,
        escalation_baseline: int,
    ) -> None:
        """Persist the ember recurrence-ladder counters verbatim.

        Fact-only: raises if ``id`` names a Decision. Same "engine computes, seam
        persists" contract as ``update_projections``, but a separate call because these
        fields are not derived from the reinforcement event log.
        """
        ...

    # ------------------------------------------------------------------
    # Distinct-pair registry
    # ------------------------------------------------------------------

    async def mark_distinct(self, pair: DistinctPair) -> None:
        """Register a judged-distinct pair; idempotent, first-judgment-wins.

        The adapter normalizes the pair (``lower_id`` / ``higher_id``) on write, so
        ``(a, b)`` and ``(b, a)`` map to one row.
        """
        ...

    async def is_distinct(self, a: HypomnemaId, b: HypomnemaId) -> bool:
        """Return ``True`` if the pair ``(a, b)`` is in the distinct registry.

        Symmetric: ``is_distinct(a, b)`` == ``is_distinct(b, a)``.
        """
        ...

    # ------------------------------------------------------------------
    # Embedding maintenance
    # ------------------------------------------------------------------

    async def backfill_targets(self) -> list[BackfillTarget]:
        """Return every row awaiting a backfill embed: a null vector or a stale ``embedding_model``.

        Buried rows are excluded. Each target carries ``content`` and ``content_hash``
        so the caller can embed and build the ``backfill`` guard without another round
        trip.
        """
        ...

    async def backfill(
        self,
        id: HypomnemaId,
        vector: list[float],
        *,
        embedding_model: str,
        expected_hash: str,
    ) -> bool:
        """Write only ``vector`` and ``embedding_model`` for ``id``, guarded by ``content_hash``.

        Re-reads ``content_hash`` immediately before writing. If it no longer matches
        ``expected_hash`` (an edit, supersession or undo landed since the caller's
        embed), writes nothing and returns ``False``; the next pass picks the row up.
        Returns ``True`` on a write. Used only by the backfill pass.
        """
        ...

    # ------------------------------------------------------------------
    # Registries
    # ------------------------------------------------------------------

    async def create_scope(self, name: Scope) -> None:
        """Register a scope name.  Idempotent — duplicate name is a no-op."""
        ...

    async def list_scopes(self) -> list[Scope]:
        """Return all registered scope names in registration order."""
        ...

    async def create_facet(self, name: str) -> None:
        """Register a purpose-facet value.  Idempotent."""
        ...

    async def list_facets(self) -> list[str]:
        """Return all registered facet values in registration order."""
        ...

    async def create_role(self, name: str) -> None:
        """Register a role name.  Idempotent."""
        ...

    async def list_roles(self) -> list[str]:
        """Return all registered role names in registration order."""
        ...

    async def delete_role(self, name: str) -> None:
        """Deregister a role name.

        Deleting a role that is not registered is an idempotent no-op and does not
        raise, like a duplicate ``create_role``. This is load-bearing:
        ``Tabularium.initialize()`` calls ``delete_role("default")`` unconditionally on
        every startup, including against a fresh store that never had it.

        It does not touch any record's ``audience_hint`` and does not refuse because a
        record still names the role. A hint is a list of strings, not a foreign key.
        The entry goes inert rather than dangling: ``context(role=...)`` validates the
        role against this registry, so a deleted role can never be requested again,
        while other roles in the same hint are unaffected.
        """
        ...

    # ------------------------------------------------------------------
    # Tombstone lookup
    # ------------------------------------------------------------------

    async def get_tombstone(self, id: HypomnemaId) -> Tombstone | None:
        """Return the tombstone for a buried record, or ``None`` if it is not buried.

        Callers use it to confirm a record is truly buried. Returns ``None`` for any
        live or unknown id and never raises.
        """
        ...

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def maintenance_debt(self) -> MaintenanceDebt:
        """Report how much maintenance the store currently needs.

        A pure read: it never mutates the store, and two calls with no writes between
        them return equal values. Adapters with no maintenance concept report zero debt
        rather than omitting the method.
        """
        ...

    async def upkeep(self) -> None:
        """Fold pending writes into the index and compact on-disk fragments.

        After ``upkeep()`` every record ranks correctly; before it, records written
        since the last run may rank stale. It never prunes history (any retention
        window it applies is long enough to be a no-op). Calling it twice in a row is
        a stable no-op.
        """
        ...

    async def reclaim(self) -> None:
        """Prune version history. Irreversible; it buys disk and nothing else.

        Deliberately separate from ``upkeep()``: a flag that turns a routine method
        irreversible let an unattended script prune history past its own warning
        banner. It is reachable only by name and never invoked by engine code.

        PRECONDITION (LanceDB's corruption warning, also in ``docs/backup.md``): no
        other process may be working on this dataset. The caller establishes this
        (e.g. stops the MCP server); the CLI ``tabularium reclaim`` prints it and
        requires interactive confirmation. Adapters with no version history implement
        it as a no-op.
        """
        ...

    # ------------------------------------------------------------------
    # Sessions: a parallel seam, deliberately not ``Filters``
    # ------------------------------------------------------------------
    #
    # A ``Session`` is never embedded, deduped or decayed and has no lifecycle
    # or validity, so it cannot take ``Filters`` (which requires
    # ``lifecycle_in``). Rows written here must not be reachable through
    # ``vector_search``, ``text_search``, ``scan`` or ``scan_records``; that keeps
    # sessions out of recall, context, the dedup probe and any decay sweep with no
    # filter needed elsewhere.

    async def append_session(self, session: Session) -> None:
        """Append a session row. Append-only, with no update or delete leg.

        Must never make ``session`` reachable from ``vector_search``, ``text_search``,
        ``scan`` or ``scan_records``.
        """
        ...

    async def latest_session_end(self, *, scope: str) -> datetime | None:
        """Return the maximum ``ended_at`` among sessions in ``scope``, or ``None`` if none exist.

        Per scope: a later row in another scope never raises this answer.
        """
        ...

    async def search_sessions(
        self,
        *,
        scopes: Sequence[str],
        query: str | None,
        since: datetime | None,
        until: datetime | None,
        limit: int,
    ) -> list[Session]:
        """Search session rows. Does not take ``Filters``.

        With ``query=None``, results are ``ended_at`` descending. With a ``query``,
        they come in the adapter's own relevance order (BM25 on LanceDB,
        case-insensitive substring on ``MemoryAdapter``); adapters agree on which rows
        match, not on their order. Both are capped by ``limit``.

        ``scopes`` is an explicit list (e.g. ``[scope, "global"]``) with no implicit
        ``global`` union. ``since`` and ``until`` are inclusive; ``None`` is
        unbounded on that side.
        """
        ...

    # ------------------------------------------------------------------
    # Cleaner state: per scope, not part of ``records``; see ``CleanerState``
    # ------------------------------------------------------------------

    async def get_cleaner_state(self, *, scope: Scope) -> CleanerState | None:
        """Return this scope's Cleaner state, or ``None`` if it was never judged.

        ``None`` is never replaced by a zero-value or epoch-backed ``CleanerState``.
        What "never judged" implies (initialize the watermark to now, judge nothing)
        is the caller's job.
        """
        ...

    async def set_cleaner_state(self, *, scope: Scope, state: CleanerState) -> None:
        """Upsert this scope's Cleaner state; writing twice leaves exactly one row."""
        ...

    # ------------------------------------------------------------------
    # Born-since query: the pool for one Cleaner pass
    # ------------------------------------------------------------------

    async def list_born_since(
        self, *, scope: Scope, since: datetime, limit: int
    ) -> list[Hypomnema]:
        """Records in ``scope`` created strictly after ``since``, oldest first, up to ``limit``.

        Takes exactly one scope with no implicit ``global`` union (scope spans are the
        caller's concern, as with ``search_sessions``). No lifecycle or validity
        pre-filter: the Cleaner pass sees Facts and Decisions alike, though it writes
        only to Facts. Buried records are excluded.
        """
        ...

    # ------------------------------------------------------------------
    # Record reviews: the revisit axis's rotation pool; see ``RecordReview``
    # ------------------------------------------------------------------

    async def append_review(self, review: RecordReview) -> None:
        """Append one review row. Append-only, with no update or delete leg.

        Two calls for one ``target_id`` must leave two rows; ``list_due_for_review``
        reduces to the latest, not this write.
        """
        ...

    async def list_due_for_review(
        self, *, scope: Scope, now: datetime, created_before: datetime, limit: int
    ) -> list[Fact]:
        """Facts in ``scope`` due for another revisit-axis look, oldest-reviewed first.

        Returns Facts with ``lifecycle in ("active", "archived")``, ``validity ==
        "valid"``, ``superseded_by is None`` and ``created_at <= created_before`` that
        either have no review row or whose latest review (by ``reviewed_at``) has
        ``next_review_due_at <= now``. Ordered by that latest ``reviewed_at`` ascending,
        never-reviewed Facts first, so once everything has been reviewed recently it
        returns nothing and the pass costs nothing. Capped at ``limit``.

        Invalid and superseded Facts are not candidates: judging and shelving one would
        pull it out of the dedup probe. That differs from the probe's own
        ``validity={"valid", "invalid"}`` admission, which governs candidate matching
        and deliberately includes invalid candidates so a stale ghost cannot gatekeep
        fresh knowledge.

        The adapter owns the whole rotation query (the latest-per-target reduction and
        the due or never-reviewed predicate), never a raw review-row export the engine
        would re-join. Buried rows are excluded. Exactly one scope, no implicit
        ``global`` union.
        """
        ...

    async def list_reviews_for(self, *, target_id: HypomnemaId) -> list[RecordReview]:
        """Every review row for ``target_id``, ascending by ``reviewed_at``; ``[]`` if none.

        A narrow single-id lookback that feeds ``engine.review_schedule.next_review_due_at``'s
        adaptive-interval compounding for one record the caller already holds. It never
        enumerates a scope or computes due-ness, so it cannot rebuild the rotation pool.
        """
        ...

    # ------------------------------------------------------------------
    # Contested queue over ``record_reviews``; see ``RecordReview``

    # ------------------------------------------------------------------
    # Record judgments: alongside ``record_reviews``; see ``RecordJudgment``
    # ------------------------------------------------------------------

    async def append_record_judgment(self, judgment: RecordJudgment) -> None:
        """Append one ``RecordJudgment`` row. Append-only.

        Two calls for one ``target_id`` must leave two rows, so a record contested,
        adjudicated, then contested again keeps every panel's verdict history.
        """
        ...

    # ------------------------------------------------------------------
    # Pass history: append-only, beside ``cleaner_state``; see ``CleanerPassRecord``
    # ------------------------------------------------------------------

    async def append_cleaner_pass(self, *, record: CleanerPassRecord) -> None:
        """Append one Cleaner-pass record. Append-only, with no update or delete leg.

        Two calls for one scope must leave two rows; the upsert belongs to
        ``set_cleaner_state`` alone.
        """
        ...

    async def list_cleaner_passes(
        self, *, scope: Scope, limit: int
    ) -> list[CleanerPassRecord]:
        """Return ``scope``'s pass-history rows, newest first, up to ``limit``.

        Exactly one scope, no implicit ``global`` union. A scope with no history
        returns ``[]``.
        """
        ...

    # ------------------------------------------------------------------
    # Read events: a separate table from ``events``; see ``ReadEvent``
    # ------------------------------------------------------------------

    async def append_read_event(self, event: ReadEvent) -> None:
        """Append a read event to the append-only log.

        Mirrors ``append_event`` over the separate ``ReadEvent`` table; never touches
        ``events``, the projection fields or ``update_projections``.
        """
        ...

    # ------------------------------------------------------------------
    # Assessor verdicts: one row per Assessor verdict per candidate; see ``VerdictRow``
    # ------------------------------------------------------------------

    async def append_verdicts(self, rows: Sequence[VerdictRow]) -> None:
        """Append new ``VerdictRow`` rows, one row per ``id``, never merged against an existing row.

        ``update_verdicts`` is the in-place update path.
        """
        ...

    async def update_verdicts(self, rows: Sequence[VerdictRow]) -> None:
        """Update existing ``VerdictRow`` rows in place, matched by ``id``.

        The row count never changes. An ``id`` not already present is not inserted on
        LanceDB (matched-only ``merge_insert``); ``MemoryAdapter`` inserts it, so
        callers pass only existing ids.
        """
        ...

    async def verdicts_by_bounce(self, bounce_id: str) -> list[VerdictRow]:
        """Every row carrying ``bounce_id``, in probe-rank order; the resolve path's lookup."""
        ...

    async def pending_verdicts(self) -> list[VerdictRow]:
        """Every row with ``status="pending"``; the startup reconcile's input."""
        ...

    async def verdicts_by_effect(self, ref: str) -> list[VerdictRow]:
        """Every row whose ``effect_ref`` or ``new_record_id`` is ``ref`` (reversal tools)."""
        ...

    async def verdicts_for_record(self, id: HypomnemaId) -> list[VerdictRow]:
        """Every row naming ``id`` as ``candidate_id`` or ``new_record_id``, newest first."""
        ...

    async def scrub_verdict_text(self, candidate_id: HypomnemaId) -> None:
        """Null ``incoming_text`` on every row naming ``candidate_id``; other fields stay.

        The scrub run on a fresh burial.
        """
        ...
