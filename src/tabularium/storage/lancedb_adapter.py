"""LanceDB storage adapter: the production backend.

LanceDB 0.34.0 (pinned, pre-1.0) through its native async API
(``lancedb.connect_async``). Every method is a coroutine over the async table
API; no ``asyncio.to_thread`` wrapping is needed.

Tables:

- ``records``: flat rows with a ``type`` discriminator. ``supersession_chain``,
  ``promoted_from`` and ``audience_hint`` are JSON text columns.
- Auxiliary tables hold ids and timestamps, never content: ``events``,
  ``read_events``, ``pairs``, ``tombstones``, ``scopes``, ``facets``, ``roles``,
  ``unsupersedes`` (idempotent on ``edge_id`` at the call level, with no table
  constraint), ``revalidations``, ``assessor_verdicts``, ``cleaner_state`` (one
  row per scope, upserted on ``scope``), and the append-only ``pass_history``,
  ``record_reviews`` and ``record_judgments``. Append-only tables are written
  with ``table.add``; the "latest governs" reads reduce at read time.
  ``manual_contests`` is opened and maintained, and nothing writes it.
- ``sessions`` is the exception that stores content: it has its own ``content``
  column and FTS index and no vector column. A Session is never embedded,
  deduped or decayed, and is unreachable from ``vector_search`` /
  ``text_search`` / ``scan`` / ``scan_records`` by construction.
- The retired ``parks`` table in old stores is left alone; ``open()`` never
  opens or writes it.

A table added after a store was created needs no migration call:
``_open_or_create`` creates it fresh on reopen. Only a column added to an
existing table needs ``_migrate_table_schema``.

Vector search is exact brute-force cosine (``distance_type('cosine')``). No
approximate index is built, permanently: one measured against real data gave no
speedup at the corpus sizes in use and lost true neighbours. Forcing exact
search through ``bypass_vector_index()`` is not provisional.

FTS and ranked visibility: every LanceDB index is a snapshot taken at build
time. Rows added later are still searchable (unindexed scan) but rank against a
corpus that excludes them, so ranking between maintenance runs may be stale.
``upkeep()`` folds pending rows into the index and ``maintenance_debt()``
reports how many are unfolded. ``create_index`` (FTS, ``lower_case=True`` on
``content``) runs once at table creation to seed the index.

Typing: LanceDB ships partial stubs, so a ``[[tool.mypy.overrides]]`` entry
suppresses import-time errors for this module; the adapter's own types are
annotated.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.query import ColumnOrdering

from tabularium.clock import Clock, SystemClock
from tabularium.domain import (
    Decision,
    DecisionLifecycle,
    Fact,
    FactLifecycle,
    Hypomnema,
    HypomnemaId,
    Scope,
    Session,
    Validity,
)
from tabularium.storage._revisit_pool import (
    REVISIT_POOL_LIFECYCLES,
    PoolCandidate,
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
from tabularium.storage.content_key import content_hash
from tabularium.storage.lance_filters import (
    and_,
    eq,
    in_,
    is_not_null,
    is_null,
    ne,
    or_,
    ts_cmp,
)
from tabularium.storage.lance_protocols import (
    LanceConnection,
    LanceMergeResult,
    LanceTable,
)
from tabularium.storage.lancedb_codec import (
    ASSESSOR_VERDICTS_SCHEMA,
    CLEANER_STATE_SCHEMA,
    EVENTS_SCHEMA,
    MANUAL_CONTESTS_SCHEMA,
    PAIRS_SCHEMA,
    PASS_HISTORY_SCHEMA,
    READ_EVENTS_SCHEMA,
    RECORD_JUDGMENTS_SCHEMA,
    REGISTRY_SCHEMA,
    REVALIDATIONS_SCHEMA,
    REVIEWS_SCHEMA,
    SESSIONS_SCHEMA,
    TOMBSTONES_SCHEMA,
    UNSUPERSEDES_SCHEMA,
    cleaner_pass_to_row,
    cleaner_state_to_row,
    content_key_columns,
    distinct_pair_to_row,
    event_to_row,
    make_records_schema,
    read_event_to_row,
    record_judgment_to_row,
    record_review_to_row,
    record_to_row,
    registry_name_to_row,
    revalidation_event_to_row,
    row_created_at,
    row_embedding_model,
    row_id,
    row_ids,
    row_to_backfill_target,
    row_to_cleaner_pass,
    row_to_cleaner_state,
    row_to_distinct_pair,
    row_to_ended_at,
    row_to_event,
    row_to_record,
    row_to_record_review,
    row_to_registry_name,
    row_to_revalidation_event,
    row_to_row_meta,
    row_to_session,
    row_to_text_hit_row,
    row_to_tombstone,
    row_to_unsupersede_event,
    row_to_vector_hit_row,
    row_to_verdict_row,
    row_vector,
    session_to_row,
    sessions_where_clause,
    to_utc,
    tombstone_to_row,
    unsupersede_event_to_row,
    verdict_row_to_row,
    where_clause,
)
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
# Maintenance
# ---------------------------------------------------------------------------

# Always passed explicitly to ``optimize()``: a bare call defaults
# ``cleanup_older_than`` to 7 days and would prune week-old versions. Only
# ``reclaim`` prunes on purpose.
_UPKEEP_RETENTION = timedelta(days=365)

# ``reclaim()`` prunes every non-latest version immediately
# (``cleanup_older_than=0``) and passes ``delete_unverified=True`` to remove
# leftover in-progress-looking files regardless of age. Irreversible;
# ``upkeep()`` never does this.
_RECLAIM_RETENTION = timedelta(0)

# Rows per chunk for ``populate_content_keys`` / ``populate_edge_ids``. Each
# pass re-queries its NULL filter rather than keeping a cursor, so a crash
# between chunks loses at most one chunk of unwritten work.
_STARTUP_CHUNK = 500

# ---------------------------------------------------------------------------
# Table schema migration (open-time, additive-only), applied to every table the
# adapter opens
# ---------------------------------------------------------------------------

# SQL literal that backfills a missing non-nullable scalar column on existing
# rows, keyed by the pyarrow type predicate. Narrow on purpose: only int64
# counters and float64 scores have been added post hoc. A column of another
# kind must fail loudly at open time, not backfill something wrong.
_SCALAR_DEFAULTS: tuple[tuple[Any, str], ...] = (
    (pa.types.is_integer, "0"),
    (pa.types.is_floating, "0.0"),
    (pa.types.is_boolean, "false"),
)

# SQL type used to CAST a bare ``NULL`` for a missing nullable column. An
# uncast ``NULL`` infers as DataFusion's untyped Null, which LanceDB refuses to
# write alongside a real value on a later row. Narrow like ``_SCALAR_DEFAULTS``:
# only string and UTC-timestamp columns have been added post hoc
# (``events.reason``, ``unsupersedes.refused_at``).
# NOTE: the cast and default choices are specific to LanceDB 0.34 / Lance 8.0.
_NULL_CAST_TYPES: tuple[tuple[Any, str], ...] = (
    (pa.types.is_string, "STRING"),
    (pa.types.is_floating, "DOUBLE"),
    (pa.types.is_timestamp, "TIMESTAMP"),
)


async def _migrate_table_schema(table: LanceTable, target_schema: pa.Schema) -> None:
    """Additively backfill columns ``target_schema`` declares and ``table`` lacks.

    Shared by every table the adapter opens. Only missing columns are handled;
    renamed, retyped or removed fields are out of scope.

    A missing nullable column backfills as a type-cast ``NULL`` (``_NULL_CAST_TYPES``),
    the honest answer for rows that predate it. A missing non-nullable column backfills
    from ``_SCALAR_DEFAULTS``, matching the default a new domain object carries, so a
    non-nullable field never reads back as NULL. Raises ``RuntimeError`` for a column
    type with no registered cast or default.
    """
    current_schema = await table.schema()
    current_names = set(current_schema.names)
    missing = [f for f in target_schema if f.name not in current_names]
    if not missing:
        return

    transforms: dict[str, str] = {}
    for field in missing:
        if field.nullable:
            cast_type: str | None = None
            for predicate, type_name in _NULL_CAST_TYPES:
                if predicate(field.type):
                    cast_type = type_name
                    break
            if cast_type is None:
                raise RuntimeError(
                    f"_migrate_table_schema: no NULL-cast type registered for "
                    f"missing nullable column {field.name!r} of type "
                    f"{field.type!r} — extend _NULL_CAST_TYPES rather than "
                    "guessing one"
                )
            transforms[field.name] = f"CAST(NULL AS {cast_type})"
            continue
        default_sql: str | None = None
        for predicate, literal in _SCALAR_DEFAULTS:
            if predicate(field.type):
                default_sql = literal
                break
        if default_sql is None:
            raise RuntimeError(
                f"_migrate_table_schema: no backfill default registered for "
                f"missing column {field.name!r} of type {field.type!r} — "
                "extend _SCALAR_DEFAULTS rather than guessing a default"
            )
        transforms[field.name] = default_sql

    await table.add_columns(transforms)


async def _migrate_embedding_model_nullable(table: LanceTable) -> None:
    """Make ``embedding_model`` nullable on a store created while it was non-nullable.

    ``_migrate_table_schema`` covers missing columns only, and a ``merge_insert``
    carrying a null into a non-nullable column raises. ``alter_columns`` bumps the
    table version even when redundant, so this runs only while the physical column
    is still non-nullable.
    """
    physical_schema = await table.schema()
    if not physical_schema.field("embedding_model").nullable:
        await table.alter_columns({"path": "embedding_model", "nullable": True})


async def _check_vector_dimension(table: LanceTable, *, path: str, dim: int) -> None:
    """Refuse to open a store whose physical vector width differs from ``dim``.

    Reopening at a new ``dim`` leaves the Lance column at its original width.
    ``table.update()`` then casts a wrong-width vector to NULL without raising and
    ``backfill()`` reports success on a row that stays vector-less. A dimension
    change needs a migration, so this is a hard refusal.
    """
    physical_width = (await table.schema()).field("vector").type.list_size
    if physical_width != dim:
        raise ValueError(
            f"Store at {path!r} has a {physical_width}-dim vector column; "
            f"configured dim={dim} does not match. A dimension change needs "
            "a migration, not a reopen — refusing to open."
        )


@dataclass(frozen=True, slots=True)
class _StoredRow:
    """A ``records`` row as read or as to be written: the record plus the columns it does not carry.

    Read from the table, ``vector`` and ``embedding_model`` are the row's own stored
    values, so a rewrite can keep them. Handed to ``_write_rows``, they are whatever the
    calling method chose to write.
    """

    record: Fact | Decision
    vector: list[float] | None
    embedding_model: str | None


@dataclass(frozen=True, slots=True)
class _IndexSpec:
    column: str
    make_config: Callable[[], object]


@dataclass(frozen=True, slots=True)
class _TableSpec:
    """One table the adapter opens: its name, schema, and what ``open()`` does on reopen.

    ``attr`` is the adapter attribute that holds the handle. ``schema`` is a
    ``pa.Schema``, or a function of ``dim`` for the ``records`` table. ``migrate``
    backfills columns added since the store was created. ``guard_vector_column``
    additionally relaxes a non-nullable ``embedding_model`` and refuses a vector width
    that differs from ``dim``.
    """

    name: str
    attr: str
    schema: pa.Schema | Callable[[int], pa.Schema]
    migrate: bool = False
    guard_vector_column: bool = False
    indexes: tuple[_IndexSpec, ...] = ()

    def schema_for(self, dim: int) -> pa.Schema:
        if isinstance(self.schema, pa.Schema):
            return self.schema

        return self.schema(dim)


_FTS_ON_CONTENT = _IndexSpec("content", partial(lancedb.index.FTS, lower_case=True))


def _btree(column: str) -> _IndexSpec:
    return _IndexSpec(column, lancedb.index.BTree)


# Order is the order ``open()`` opens tables in. ``events.reason``,
# ``tombstones.restores`` / ``restores_content_hash``, ``unsupersedes.refused_at`` were
# added after creation and backfill as NULL; only the tables with ``migrate`` have
# missing columns added when an existing store is reopened. ``records`` is born with FTS on
# ``content``, BTree on ``id`` and Bitmap on ``scope``; ``sessions`` has no vector
# column, so no vector index; ``read_events`` gets BTree on ``target_id`` so a
# ``target_id IN (...)`` push-down seeks instead of scanning.
_TABLE_SPECS: tuple[_TableSpec, ...] = (
    _TableSpec(
        "records",
        "_records",
        make_records_schema,
        migrate=True,
        guard_vector_column=True,
        indexes=(_FTS_ON_CONTENT, _btree("id"), _IndexSpec("scope", lancedb.index.Bitmap)),
    ),
    _TableSpec("events", "_events", EVENTS_SCHEMA, migrate=True),
    _TableSpec("pairs", "_pairs", PAIRS_SCHEMA, migrate=True),
    _TableSpec("tombstones", "_tombstones_tbl", TOMBSTONES_SCHEMA, migrate=True),
    _TableSpec("scopes", "_scopes_tbl", REGISTRY_SCHEMA),
    _TableSpec("facets", "_facets_tbl", REGISTRY_SCHEMA),
    _TableSpec("roles", "_roles_tbl", REGISTRY_SCHEMA),
    _TableSpec(
        "sessions",
        "_sessions",
        SESSIONS_SCHEMA,
        indexes=(_FTS_ON_CONTENT, _btree("ended_at")),
    ),
    _TableSpec("cleaner_state", "_cleaner_state", CLEANER_STATE_SCHEMA),
    _TableSpec("pass_history", "_pass_history", PASS_HISTORY_SCHEMA),
    _TableSpec("record_reviews", "_record_reviews", REVIEWS_SCHEMA),
    _TableSpec("record_judgments", "_record_judgments", RECORD_JUDGMENTS_SCHEMA),
    _TableSpec("manual_contests", "_manual_contests", MANUAL_CONTESTS_SCHEMA),
    _TableSpec(
        "read_events",
        "_read_events",
        READ_EVENTS_SCHEMA,
        indexes=(_btree("target_id"),),
    ),
    _TableSpec("unsupersedes", "_unsupersedes", UNSUPERSEDES_SCHEMA, migrate=True),
    _TableSpec("revalidations", "_revalidations", REVALIDATIONS_SCHEMA),
    _TableSpec("assessor_verdicts", "_assessor_verdicts", ASSESSOR_VERDICTS_SCHEMA),
)

async def _ensure_index(table: LanceTable, index: _IndexSpec) -> None:
    """Create ``index`` on ``table`` unless it exists.

    ``replace=False`` raises RuntimeError("already exists") on reopen; only that is
    suppressed, so permission, schema-drift and bad-config errors fail fast.
    """
    try:
        await table.create_index(index.column, config=index.make_config(), replace=False)
    except RuntimeError as exc:
        if "already exists" not in str(exc):
            raise


# ---------------------------------------------------------------------------
# LanceDB adapter
# NOTE: ``self._supersede_lock`` is per-instance, not keyed by store path. Two
# adapters on one path in one process do not share it, and a second
# ``AsyncTable`` handle can read stale data after the first commits; an
# unconditional ``delete()`` from a stale handle can silently miss a row the
# other handle sees. Cross-handle freshness is not handled anywhere in this
# file. The deployment is one adapter per process. ``tabularium backfill`` opens
# a second adapter against a live store; it is safe because ``backfill()``
# uses a conditional ``table.update(where="id = ... AND content_hash = ...")``
# that LanceDB evaluates against the committed row at commit time.
# ---------------------------------------------------------------------------


class _LanceRepairIO:
    """``RepairIO`` over the tables: ``get_by_id`` for records, the tombstone read off its table."""

    def __init__(self, adapter: LanceDBAdapter) -> None:
        self._adapter = adapter

    async def list_events(self) -> list[UnsupersedeEvent]:
        return await self._adapter.list_unsupersedes()

    async def load_record(self, id: HypomnemaId) -> Fact | Decision | None:
        try:
            return await self._adapter.get_by_id(id)
        except KeyError:
            return None

    async def load_tombstone(self, buried_id: HypomnemaId) -> Tombstone | None:
        return await self._adapter._tombstone_from_table(buried_id)

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


class LanceDBAdapter:
    """LanceDB ``StorageAdapter``, the production default.

    Construction::

        adapter = await LanceDBAdapter.open(
            path="/path/to/store",
            embedding_model="ollama/nomic-embed-text",
            dim=768,
        )

    The store directory is created on first open and tables are created
    idempotently, so opening an existing store is safe. ``embedding_model`` and
    ``dim`` are single-store constants (one store, one embedder, one dim); rows
    whose ``embedding_model`` differs from the configured value are backfill
    candidates.
    """

    def __init__(
        self,
        *,
        embedding_model: str,
        dim: int,
        records_schema: pa.Schema,
        db: LanceConnection,
        tables: Mapping[str, LanceTable],
        clock: Clock | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._embedding_model = embedding_model
        self._dim = dim
        self._records_schema: pa.Schema = records_schema
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._db: LanceConnection = db
        self._records: LanceTable = tables["records"]
        self._events: LanceTable = tables["events"]
        self._pairs: LanceTable = tables["pairs"]
        self._tombstones_tbl: LanceTable = tables["tombstones"]
        self._scopes_tbl: LanceTable = tables["scopes"]
        self._facets_tbl: LanceTable = tables["facets"]
        self._roles_tbl: LanceTable = tables["roles"]
        self._sessions: LanceTable = tables["sessions"]
        self._cleaner_state: LanceTable = tables["cleaner_state"]
        self._pass_history: LanceTable = tables["pass_history"]
        self._record_reviews: LanceTable = tables["record_reviews"]
        self._record_judgments: LanceTable = tables["record_judgments"]
        self._manual_contests: LanceTable = tables["manual_contests"]
        self._read_events: LanceTable = tables["read_events"]
        self._unsupersedes: LanceTable = tables["unsupersedes"]
        self._revalidations: LanceTable = tables["revalidations"]
        self._assessor_verdicts: LanceTable = tables["assessor_verdicts"]

        # In-memory buried-id set for O(1) upsert rejection, loaded from the
        # tombstones table at open and then maintained incrementally.
        self._buried_ids: set[HypomnemaId] = set()

        # Tombstone cache (buried_id -> Tombstone), maintained synchronously in
        # ``bury()``.
        self._tombstone_cache: dict[HypomnemaId, Tombstone] = {}

        # Distinct-pair cache, maintained synchronously in ``mark_distinct()``;
        # mirrors ``MemoryAdapter._distinct``.
        self._distinct: dict[frozenset[HypomnemaId], DistinctPair] = {}

        # Serializes the supersede/revoke edges of ``transition()`` and ``bury()``
        # (fresh re-check through the write). Per-instance (see the NOTE above) and
        # global rather than per-id: supersession is rare, and steady-state traffic is
        # already serialized by the engine's write lock.
        self._supersede_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def open(
        cls,
        path: str,
        *,
        embedding_model: str,
        dim: int,
        clock: Clock | None = None,
    ) -> LanceDBAdapter:
        """Open (or create) a LanceDB store at ``path``.

        Idempotent: creates missing tables and opens existing ones, migrating columns
        added since the store was created.
        """
        db: LanceConnection = await lancedb.connect_async(path)
        list_result = await db.list_tables()
        existing = set(list_result.tables) if hasattr(list_result, "tables") else set(list_result)

        tables: dict[str, LanceTable] = {}
        schemas: dict[str, pa.Schema] = {}
        for spec in _TABLE_SPECS:
            schema = spec.schema_for(dim)
            schemas[spec.name] = schema
            if spec.name in existing:
                table = await db.open_table(spec.name)
            else:
                table = await db.create_table(spec.name, schema=schema)

            if spec.name in existing and spec.migrate:
                # On reopen, ``open_table`` returns the table as it physically is. Backfill
                # columns added since the store was created, so write paths building Arrow rows
                # against the current schema do not fail.
                await _migrate_table_schema(table, schema)
            if spec.name in existing and spec.guard_vector_column:
                await _migrate_embedding_model_nullable(table)
                await _check_vector_dimension(table, path=path, dim=dim)
            tables[spec.name] = table

        for spec in _TABLE_SPECS:
            for index in spec.indexes:
                await _ensure_index(tables[spec.name], index)

        adapter = cls(
            embedding_model=embedding_model,
            dim=dim,
            records_schema=schemas["records"],
            db=db,
            tables=tables,
            clock=clock,
        )
        # Populate buried-id cache from existing tombstones
        await adapter._load_buried_ids()
        # Populate distinct-pair cache from existing pairs
        await adapter._load_distinct_pairs()
        return adapter

    async def _load_buried_ids(self) -> None:
        """Populate ``_buried_ids`` and ``_tombstone_cache`` from the tombstones table.

        NOTE: ``open()`` is not write-free. For a legacy tombstone (no ``restores``) it
        deletes any records row a crash left behind between the tombstone write and the
        delete, so the row never surfaces in searches.

        A restores-marked tombstone is handled by whether its buried row still exists.
        A pending undo (row present; ``plant_tombstone`` does not delete it) is
        excluded from ``_buried_ids`` and from the delete repair: finishing it is
        ``repair_unsupersedes``'s job, run later from ``Tabularium.initialize()``. A
        completed undo (row already gone) is a permanent burial and rejoins
        ``_buried_ids``, or the id-reuse guard in ``upsert`` would lose its protection
        across a restart.
        """
        rows = await self._tombstones_tbl.query().to_list()
        legacy_buried_ids: set[HypomnemaId] = set()
        pending_restores_ids: set[HypomnemaId] = set()
        for row in rows:
            tombstone = row_to_tombstone(row)
            buried_id = tombstone.buried_id
            self._tombstone_cache[buried_id] = tombstone
            if tombstone.restores is None:
                legacy_buried_ids.add(buried_id)
                self._buried_ids.add(buried_id)
            else:
                pending_restores_ids.add(buried_id)

        # Repair: delete records rows for legacy tombstones only (a crash between the
        # tombstone write and the delete). Restores-marked tombstones are excluded.
        if legacy_buried_ids:
            await self._records.delete(in_("id", legacy_buried_ids))

        # A restores-marked tombstone whose row is already gone is a completed undo and
        # rejoins ``_buried_ids``. Rows still present belong to a pending undo that
        # ``repair_unsupersedes`` finishes later and stay live here.
        if pending_restores_ids:
            present_rows = await (
                self._records.query()
                .where(in_("id", pending_restores_ids))
                .select(["id"])
                .to_list()
            )
            present_ids = {row_id(r) for r in present_rows}
            self._buried_ids |= pending_restores_ids - present_ids

    async def _load_distinct_pairs(self) -> None:
        """Populate ``_distinct`` from the pairs table on reopen.

        Without it ``is_distinct()`` would return False for every adjudicated pair after
        a restart, re-bouncing pairs the agent already resolved.
        """
        rows = await self._pairs.query().to_list()
        for row in rows:
            pair = row_to_distinct_pair(row)
            cache_key: frozenset[HypomnemaId] = frozenset({pair.lower_id, pair.higher_id})
            if cache_key not in self._distinct:
                self._distinct[cache_key] = pair

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
        """Exact brute-force cosine search over rows with a non-null vector, pre-filtered.

        ``distance_type('cosine')`` returns ``_distance = 1 - cosine_similarity``;
        ``Hit.score`` is ``1 - _distance`` (higher is more similar), matching
        ``MemoryAdapter``. ``bypass_vector_index()`` forces exact search; no approximate
        index is used.

        ``Hit.rank`` increments after the buried skip, so ranks stay contiguous.
        """
        if len(vector) == 0:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self._dim}, got 0"
            )
        if len(vector) != self._dim:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self._dim}, got {len(vector)}"
            )

        where = where_clause(filters)
        # Also exclude rows with a NULL vector (buried records have no vector column value)
        where += " AND vector IS NOT NULL"

        q = (
            self._records.query()
            .nearest_to([float(x) for x in vector])
            .distance_type("cosine")
            .where(where)
            .bypass_vector_index()
            .limit(k)
        )

        rows = await q.to_list()

        hits: list[Hit] = []
        rank = 1
        for row in rows:
            hit_row = row_to_vector_hit_row(row)
            # Defense in depth: skip ids in the buried cache (same-process races, and
            # crash-interrupted burials not yet visible to the query planner).
            if hit_row.record.id in self._buried_ids:
                continue
            # Cosine distance = 1 - cosine_similarity → similarity = 1 - distance
            score = max(0.0, 1.0 - hit_row.raw)
            hits.append(Hit(record=hit_row.record, score=score, rank=rank, leg="vector"))
            rank += 1
        return hits

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
        """BM25 FTS over ``content``, pre-filtered. An empty query returns an empty result.

        ``Hit.rank`` increments after the buried skip.
        """
        if not query.strip():
            return []

        where = where_clause(filters)
        q = self._records.query().nearest_to_text(query).where(where).limit(k)

        rows = await q.to_list()

        hits: list[Hit] = []
        rank = 1
        for row in rows:
            hit_row = row_to_text_hit_row(row)
            # Defense-in-depth: skip buried ids (same guard as vector_search).
            if hit_row.record.id in self._buried_ids:
                continue
            hits.append(Hit(record=hit_row.record, score=hit_row.raw, rank=rank, leg="text"))
            rank += 1
        return hits

    # ------------------------------------------------------------------
    # Single-row reads behind the read-modify-write methods
    # ------------------------------------------------------------------

    async def _fetch_unguarded(self, id: HypomnemaId) -> _StoredRow | None:
        """The ``records`` row for ``id``, or ``None`` when the query finds none.

        Does not consult ``_buried_ids``. A caller that wants the buried guard uses
        ``_fetch``; the others read here and apply their own check, or none.
        """
        rows = await (
            self._records.query()
            .where(eq("id", id))
            .to_list()
        )
        if not rows:
            return None

        return _StoredRow(
            record=row_to_record(rows[0]),
            vector=row_vector(rows[0]),
            embedding_model=row_embedding_model(rows[0]),
        )

    async def _fetch(self, id: HypomnemaId) -> _StoredRow | None:
        """Like ``_fetch_unguarded``, but ``None`` for an id in ``_buried_ids``.

        The buried check runs after the query returns: a ``bury()`` that completed
        while the read was in flight may not yet be visible to the query snapshot.
        """
        stored = await self._fetch_unguarded(id)
        if stored is None or id in self._buried_ids:
            return None

        return stored

    async def _write_rows(
        self,
        rows: Sequence[_StoredRow],
        *,
        insert_missing: bool,
        delete_unmatched: str | None = None,
    ) -> LanceMergeResult:
        """Write ``rows`` to ``records`` in one ``merge_insert`` keyed on ``id``.

        Each row carries the vector and ``embedding_model`` its caller chose. A matched
        id is always updated. ``insert_missing`` also inserts ids that do not match; left
        off, the merge is matched-only, so a row buried between the caller's read and
        this write has nothing to match and is not resurrected. ``delete_unmatched`` adds
        ``when_not_matched_by_source_delete`` with that filter.

        Never a ``where=`` on the merge (see ``transition``).
        """
        built = [record_to_row(row.record, row.vector, row.embedding_model) for row in rows]
        columns = list(built[0].keys())
        arrow_table = pa.table(
            {name: [row[name] for row in built] for name in columns},
            schema=self._records_schema,
        )

        merge = self._records.merge_insert("id").when_matched_update_all()
        if insert_missing:
            merge = merge.when_not_matched_insert_all()
        if delete_unmatched is not None:
            merge = merge.when_not_matched_by_source_delete(delete_unmatched)

        return await merge.execute(arrow_table)

    async def _rewrite(
        self,
        stored: _StoredRow,
        changes: Mapping[str, object],
    ) -> LanceMergeResult:
        """Apply ``changes`` to ``stored.record`` and write it back matched-only.

        Keeps the row's own vector and ``embedding_model`` stamp: relabeling it would
        hide a stale-model row from the backfill scan.
        """
        updated = type(stored.record).model_validate(
            {**stored.record.model_dump(mode="python"), **changes}
        )
        written = _StoredRow(updated, stored.vector, stored.embedding_model)

        return await self._write_rows([written], insert_missing=False)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def upsert(
        self,
        record: Fact | Decision,
        vector: list[float] | None,
    ) -> None:
        """Insert or update a record.

        Raises ``ValueError`` on the same gates as ``MemoryAdapter.upsert``: a vector of
        the wrong length, a buried id, a changed Decision content, lifecycle or
        validity (use ``transition`` / ``set_validity``), a changed projection field
        (use ``update_projections``) or recurrence counter (use ``update_recurrence``),
        or a changed ``superseded_by``, ``supersession_chain`` or ``promoted_from``
        (only ``replace`` draws edges).

        ``vector=None`` lands the row with a null vector and null ``embedding_model``
        (embedder unavailable). Lock-free: the existence check and the ``merge_insert``
        are not atomic.
        """
        if vector and len(vector) != self._dim:
            raise ValueError(
                f"Vector dimension mismatch (id={record.id!s}): "
                f"expected {self._dim}, got {len(vector)}"
            )
        if record.id in self._buried_ids:
            raise ValueError(
                f"Cannot upsert a buried id (id={record.id!s}): "
                "burial is permanent; the tombstone archive is monotonic"
            )

        existing = await self._fetch_unguarded(record.id)
        if existing is not None:
            check_upsert_allowed(existing.record, record)

        await self._write_rows(
            [_StoredRow(record, vector, self._embedding_model)],
            insert_missing=True,
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
        """Change the lifecycle state of a record and persist ``updated_at = at``.

        Never draws a forward supersession edge: ``replace`` is the only path that sets
        ``superseded_by``, and a Decision never takes ``lifecycle="superseded"`` here.

        ``audience_hint`` is tri-state: ``None`` leaves the stored value, a tuple
        replaces it, an empty tuple clears it (normalized to ``None``, as the codec
        encodes it). ``provisional`` (Fact-only): ``None`` leaves it, ``True`` /
        ``False`` sets it. Both persist in the same write as ``lifecycle``. See
        ``StorageAdapter.transition`` for the full contract.

        Every branch re-reads the row under ``self._supersede_lock`` immediately before
        an unconditional ``merge_insert``, never a storage-level ``where=``: Lance 8.0
        panics evaluating any ``where=`` against a ``records`` table that went through
        ``_migrate_table_schema``'s ``add_columns`` backfill, because its fragments
        then carry different field counts. The lock makes the Python re-read
        load-bearing against another in-process ``transition``, ``bury`` or ``replace``.
        A Decision buried in the interim is a silent no-op; a Fact buried in the
        interim raises ``KeyError``.
        """
        existing = await self._fetch_unguarded(id)
        if existing is None:
            raise KeyError(f"Unknown id: {id!s}")

        ex_rec = existing.record
        # audience_hint and provisional collapse into the fields the caller opted to
        # change; absent keys leave the field untouched.
        extra_updates: dict[str, object] = {}
        if audience_hint is not None:
            extra_updates["audience_hint"] = tuple(audience_hint) if audience_hint else None
        if provisional is not None:
            extra_updates["provisional"] = provisional
        changes: dict[str, object] = {"lifecycle": lifecycle, "updated_at": at, **extra_updates}

        # A Decision's other lifecycle edge (revoke) can race ``replace()`` or ``bury()``
        # of the same record, and so can a Fact's (hot, audience, shelved or any other
        # move), so both re-read fresh under the shared lock. The cost is accepted: these
        # are low-frequency administrative calls.
        async with self._supersede_lock:
            fresh = await self._fetch_unguarded(id)
            if fresh is None:
                if isinstance(ex_rec, Decision):
                    return
                raise KeyError(
                    f"Unknown id: {id!s} — buried while transitioning "
                    "(lost a concurrent race)"
                )
            await self._rewrite(fresh, changes)

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

        The successor row, every predecessor flip and every invalidation target land
        in a single ``merge_insert`` under ``self._supersede_lock``: all of it commits
        or none of it does. Every referent is re-read fresh under the lock, and a
        query result for an id in ``_buried_ids`` is treated as absent, since a
        completed ``bury()``'s ``delete()`` may not yet be visible to this query's
        snapshot. Legality lives in ``storage._supersede``; this method does the fresh
        reads, the buried guard and the multi-row Arrow table.
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

        async with self._supersede_lock:
            existing = await self._fetch(successor.id)
            existing_successor: Fact | Decision | None = None
            existing_vector: list[float] | None = None
            existing_embedding_model: str | None = None
            if existing is not None:
                existing_successor = existing.record
                existing_vector = existing.vector
                existing_embedding_model = existing.embedding_model

            successor_to_write = resolve_replace_successor(
                successor=successor,
                existing=existing_successor,
                vector=vector,
                buried=successor.id in self._buried_ids,
            )
            successor_vector = existing_vector if existing_successor is not None else vector
            if successor_vector is not None and len(successor_vector) != self._dim:
                raise ValueError(
                    f"Vector dimension mismatch (id={successor.id!s}): "
                    f"expected {self._dim}, got {len(successor_vector)}"
                )
            # A retained vector keeps its own ``embedding_model`` stamp: relabeling it
            # would hide it from a model-migration backfill scan.
            successor_embedding_model = (
                existing_embedding_model
                if existing_successor is not None and existing_embedding_model is not None
                else self._embedding_model
            )

            rows_to_write: list[_StoredRow] = [
                _StoredRow(successor_to_write, successor_vector, successor_embedding_model)
            ]
            for flip in flips:
                pred = await self._fetch(flip.id)
                existing_pred: Fact | Decision | None = None
                pred_vector: list[float] | None = None
                pred_embedding_model: str | None = None
                if pred is not None:
                    existing_pred = pred.record
                    pred_vector = pred.vector
                    pred_embedding_model = pred.embedding_model
                updated_pred = resolve_replace_predecessor(
                    predecessor=existing_pred,
                    predecessor_id=flip.id,
                    successor_id=successor.id,
                    successor_type=type(successor),
                    flip=flip,
                    at=at,
                )
                # The predecessor's vector is always retained, so it keeps its own stamp.
                rows_to_write.append(
                    _StoredRow(
                        updated_pred,
                        pred_vector,
                        pred_embedding_model or self._embedding_model,
                    )
                )

            for invalidation in invalidations:
                target = await self._fetch(invalidation.id)
                existing_target: Fact | Decision | None = None
                target_vector: list[float] | None = None
                target_embedding_model: str | None = None
                if target is not None:
                    existing_target = target.record
                    target_vector = target.vector
                    target_embedding_model = target.embedding_model
                updated_target = resolve_invalidation_target(
                    target=existing_target,
                    target_id=invalidation.id,
                    successor_id=successor.id,
                    successor_type=type(successor),
                    invalidation=invalidation,
                    flip_ids=flip_ids,
                )
                rows_to_write.append(
                    _StoredRow(
                        updated_target,
                        target_vector,
                        target_embedding_model or self._embedding_model,
                    )
                )

            await self._write_rows(rows_to_write, insert_missing=True)

    async def set_validity(
        self,
        id: HypomnemaId,
        valid: bool,
        reason: str,
    ) -> None:
        """Flip the validity bit; a same-value re-call is an idempotent no-op.

        Does not touch ``updated_at``: validity is metadata about frozen content, not
        an edit. Lock-free read then matched-only write; ``reason`` is not persisted
        by this adapter.
        """
        existing = await self._fetch_unguarded(id)
        if existing is None:
            raise KeyError(f"Unknown id: {id!s}")

        ex_rec = existing.record
        new_value: Validity = "valid" if valid else "invalid"
        if ex_rec.validity == new_value:
            return  # idempotent no-op

        await self._rewrite(existing, {"validity": new_value})
        # ``reason`` is not persisted by this adapter.

    async def revalidate(
        self, id: HypomnemaId, reason: str, *, at: datetime
    ) -> RevalidationEvent | None:
        """See ``StorageAdapter.revalidate``.

        Writes the event row, then flips the Decision row, under
        ``self._supersede_lock`` so a concurrent supersede, bury or revoke of the same
        Decision cannot interleave with either half.
        """
        async with self._supersede_lock:
            existing = await self._fetch_unguarded(id)
            if existing is None:
                raise KeyError(f"Unknown id: {id!s}")

            ex_rec = existing.record
            if not isinstance(ex_rec, Decision) or ex_rec.invalidated_by is None:
                return None
            assert ex_rec.invalidation_reason is not None, (
                "invalidated_by and invalidation_reason are set together in one "
                "transaction (replace's invalidations=) — one set without the "
                "other is a storage-layer bug, never a legitimate state"
            )

            event = RevalidationEvent(
                id=uuid.uuid4(),
                decision_id=id,
                withdrawn_cause=HypomnemaId(uuid.UUID(ex_rec.invalidated_by)),
                withdrawn_reason=ex_rec.invalidation_reason,
                reason=reason,
                at=at,
            )
            event_row = revalidation_event_to_row(event)
            await self._revalidations.add([event_row])

            await self._rewrite(
                existing,
                {"validity": "valid", "invalidated_by": None, "invalidation_reason": None},
            )
            return event

    async def list_revalidations(self, decision_id: HypomnemaId) -> list[RevalidationEvent]:
        """See ``StorageAdapter.list_revalidations``."""
        rows = await (
            self._revalidations.query()
            .where(eq("decision_id", decision_id))
            .to_list()
        )
        return [row_to_revalidation_event(r) for r in rows]

    async def bury(
        self,
        id: HypomnemaId,
        reason: str,
        *,
        model_generated: bool = False,
    ) -> TombstoneId:
        """Bury a record: write its tombstone, then drop it from the records table.

        Event log and pair registry entries are not cleared (ids and timestamps only;
        removing them would break lineage).

        The pre-read of the row is lock-free; the tombstone write and delete run under
        ``self._supersede_lock``, for Facts and Decisions alike. Without it a
        ``transition`` racing a ``bury`` could re-read the row as present, then issue a
        ``merge_insert`` that matches nothing because the delete landed first, and
        return success having written nothing. With it, whichever acquires the lock
        first completes before the other re-reads, so the second sees committed state:
        a sequential success, or a clean ``KeyError`` from ``transition``.
        """
        if id in self._buried_ids:
            raise KeyError(f"Already buried: {id!s}")

        existing = await self._fetch_unguarded(id)
        if existing is None:
            raise KeyError(f"Unknown id: {id!s}")

        rec_type = existing.record.type

        async def _do_bury() -> TombstoneId:
            # Tombstone first, then delete the record.
            tombstone = Tombstone(
                id=uuid.uuid4(),
                buried_id=id,
                type=rec_type,
                reason=reason,
                timestamp=self._clock.now(),
                model_generated=model_generated,
            )
            await self._tombstones_tbl.add([tombstone_to_row(tombstone)])

            # Drop the record row (content + vector + FTS row)
            await self._records.delete(eq("id", id))

            # Update in-memory caches
            self._buried_ids.add(id)
            self._tombstone_cache[id] = tombstone

            return tombstone.id

        async with self._supersede_lock:
            return await _do_bury()

    # ------------------------------------------------------------------
    # Startup population passes
    #
    # Each chunk's select and write run together under ``self._supersede_lock``.
    # Selecting outside it let another lock-holder (a second run of the pass, or the
    # supersede/unsupersede branches) make the selection stale and double-assign an
    # edge id or write one for an edge just cleared. A filled row leaves the
    # ``IS NULL`` predicate, so the loop needs no cursor and resumes after a crash.
    # The merge source carries only the filled columns, so a pass never touches
    # ``content`` or any other field.
    # ------------------------------------------------------------------

    async def populate_content_keys(self) -> int:
        """See ``StorageAdapter.populate_content_keys``."""
        written = 0
        while True:
            async with self._supersede_lock:
                rows = await (
                    self._records.query()
                    .where(is_null("content_key"))
                    .select(["id", "content"])
                    .limit(_STARTUP_CHUNK)
                    .to_list()
                )
                if not rows:
                    break

                arrow_table = pa.table(
                    content_key_columns(rows),
                    schema=pa.schema(
                        [
                            pa.field("id", pa.utf8(), nullable=False),
                            pa.field("content_key", pa.utf8(), nullable=True),
                            pa.field("content_hash", pa.utf8(), nullable=True),
                        ]
                    ),
                )
                await (
                    self._records.merge_insert("id")
                    .when_matched_update_all()
                    .execute(arrow_table)
                )
            written += len(rows)
        return written

    async def populate_edge_ids(self) -> int:
        """See ``StorageAdapter.populate_edge_ids``."""
        written = 0
        while True:
            async with self._supersede_lock:
                rows = await (
                    self._records.query()
                    .where(and_(is_not_null("superseded_by"), is_null("supersession_edge_id")))
                    .select(["id"])
                    .limit(_STARTUP_CHUNK)
                    .to_list()
                )
                if not rows:
                    break

                arrow_table = pa.table(
                    {
                        "id": row_ids(rows),
                        "supersession_edge_id": [str(uuid.uuid4()) for _ in rows],
                    },
                    schema=pa.schema(
                        [
                            pa.field("id", pa.utf8(), nullable=False),
                            pa.field("supersession_edge_id", pa.utf8(), nullable=True),
                        ]
                    ),
                )
                await (
                    self._records.merge_insert("id")
                    .when_matched_update_all()
                    .execute(arrow_table)
                )
            written += len(rows)
        return written

    # ------------------------------------------------------------------
    # Unsupersede
    # ------------------------------------------------------------------

    async def append_unsupersede(self, event: UnsupersedeEvent) -> UnsupersedeEvent:
        """See ``StorageAdapter.append_unsupersede``; idempotent on ``event.edge_id``.

        An existing row is a retry only when predecessor, successor and arm all match
        ``event``; otherwise the collision is refused. A matching retry whose
        ``refused_at`` is set is reopened: the recomputed row
        (``resolve_unsupersede_append`` clears it) is written back before returning.
        """
        async with self._supersede_lock:
            existing = await self.get_unsupersede(event.edge_id)
            if existing is not None:
                resolved = resolve_unsupersede_append(existing=existing, event=event)
                if resolved.refused_at != existing.refused_at:
                    await self._write_unsupersede_row(resolved)
                return resolved
            row = unsupersede_event_to_row(event)
            await self._unsupersedes.add([row])
            return event

    async def get_unsupersede(self, edge_id: str) -> UnsupersedeEvent | None:
        """See ``StorageAdapter.get_unsupersede``."""
        rows = await (
            self._unsupersedes.query().where(eq("edge_id", edge_id)).to_list()
        )
        if not rows:
            return None
        return row_to_unsupersede_event(rows[0])

    async def list_unsupersedes(self) -> list[UnsupersedeEvent]:
        """See ``StorageAdapter.list_unsupersedes``."""
        rows = await self._unsupersedes.query().to_list()
        return [row_to_unsupersede_event(r) for r in rows]

    async def _write_unsupersede_row(self, event: UnsupersedeEvent) -> None:
        """Overwrite an existing ``unsupersedes`` row in place, matched by ``id``.

        Shared by ``append_unsupersede`` (reopening a refused edge) and
        ``mark_unsupersede_refused``. Never creates a row; ``append_unsupersede``'s
        ``.add()`` is the only insert.
        """
        row = unsupersede_event_to_row(event)
        arrow_table = pa.table(
            {k: [v] for k, v in row.items()}, schema=UNSUPERSEDES_SCHEMA,
        )
        await self._unsupersedes.merge_insert("id").when_matched_update_all().execute(
            arrow_table
        )

    async def mark_unsupersede_refused(self, edge_id: str, *, at: datetime) -> None:
        """See ``StorageAdapter.mark_unsupersede_refused``."""
        async with self._supersede_lock:
            existing = await self.get_unsupersede(edge_id)
            if existing is None:
                raise KeyError(f"No undo event for edge_id={edge_id!r}")
            await self._write_unsupersede_row(existing.model_copy(update={"refused_at": at}))

    async def _tombstone_from_table(self, buried_id: HypomnemaId) -> Tombstone | None:
        """Ground-truth tombstone read straight off the table, never ``_tombstone_cache``.

        ``bury()`` writes its tombstone before deleting the record and updates the
        cache only after both succeed, so a caught failure on the delete leaves a
        durable tombstone the cache does not know about. ``plant_tombstone`` and
        ``repair_unsupersedes`` need the durable answer so they never plant a second,
        conflicting tombstone.
        """
        rows = await (
            self._tombstones_tbl.query().where(eq("buried_id", buried_id)).to_list()
        )
        if not rows:
            return None
        return row_to_tombstone(rows[0])

    async def plant_tombstone(
        self, buried_id: HypomnemaId, reason: str, *, restores: HypomnemaId
    ) -> TombstoneId:
        """See ``StorageAdapter.plant_tombstone``.

        Writes the tombstone without deleting ``buried_id``'s row or adding it to
        ``_buried_ids``; ``restore`` removes the row in the same transaction that
        restores the predecessor.
        """
        async with self._supersede_lock:
            target = await self._fetch(buried_id)
            if target is None:
                raise KeyError(f"Unknown or buried id: {buried_id!s}")
            if await self._tombstone_from_table(buried_id) is not None:
                raise KeyError(f"Already buried: {buried_id!s}")

            tombstone = Tombstone(
                id=uuid.uuid4(),
                buried_id=buried_id,
                type=target.record.type,
                reason=reason,
                timestamp=self._clock.now(),
                model_generated=False,
                restores=restores,
                restores_content_hash=content_hash(target.record.content),
            )
            await self._tombstones_tbl.add([tombstone_to_row(tombstone)])

            self._tombstone_cache[buried_id] = tombstone
            return tombstone.id

    async def withdraw_tombstone(self, buried_id: HypomnemaId) -> None:
        """See ``StorageAdapter.withdraw_tombstone``."""
        async with self._supersede_lock:
            tombstone = await self._tombstone_from_table(buried_id)

            stored = await self._fetch(buried_id)
            buried: Fact | Decision | None = stored.record if stored is not None else None
            resolve_withdraw_tombstone(buried_id=buried_id, tombstone=tombstone, buried=buried)

            await self._tombstones_tbl.delete(eq("buried_id", buried_id))
            self._tombstone_cache.pop(buried_id, None)

    async def restore(
        self,
        predecessor_id: HypomnemaId,
        *,
        successor_id: HypomnemaId,
        edge_id: str,
        at: datetime,
    ) -> None:
        """See ``StorageAdapter.restore``."""
        async with self._supersede_lock:
            pred = await self._fetch(predecessor_id)
            existing_pred: Fact | Decision | None = None
            pred_vector: list[float] | None = None
            pred_embedding_model: str | None = None
            if pred is not None:
                existing_pred = pred.record
                pred_vector = pred.vector
                pred_embedding_model = pred.embedding_model

            updated_pred = resolve_unsupersede_write(
                predecessor=existing_pred,
                predecessor_id=predecessor_id,
                successor_id=successor_id,
                edge_id=edge_id,
                at=at,
            )

            delete_unmatched = (
                eq("id", successor_id) if isinstance(updated_pred, Decision) else None
            )
            await self._write_rows(
                [
                    _StoredRow(
                        updated_pred,
                        pred_vector,
                        pred_embedding_model or self._embedding_model,
                    )
                ],
                insert_missing=False,
                delete_unmatched=delete_unmatched,
            )

            if isinstance(updated_pred, Decision):
                self._buried_ids.add(successor_id)

    async def repair_unsupersedes(self) -> RepairReport:
        """See ``StorageAdapter.repair_unsupersedes``."""
        return await run_unsupersede_repair(_LanceRepairIO(self))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_by_id(self, id: HypomnemaId) -> Fact | Decision:
        """Retrieve a record by id.  Raises ``KeyError`` if unknown or buried."""
        if id in self._buried_ids:
            raise KeyError(f"Unknown or buried id: {id!s}")

        rows = await (
            self._records.query()
            .where(eq("id", id))
            .to_list()
        )
        if not rows:
            raise KeyError(f"Unknown or buried id: {id!s}")
        return row_to_record(rows[0])

    async def find_by_content_key(
        self, key: str, scopes: list[Scope]
    ) -> list[Fact | Decision]:
        """See ``StorageAdapter.find_by_content_key``.

        ``content_key`` derives from free text, so the predicate is built through
        ``lance_filters`` and a single quote in it stays inside its literal.
        """
        if not scopes:
            return []

        scope_strs = [str(s) for s in scopes]
        predicate = and_(eq("content_key", key), in_("scope", scope_strs))
        rows = await self._records.query().where(predicate).to_list()
        return [row_to_record(row) for row in rows if row_id(row) not in self._buried_ids]

    async def _scan_rows(
        self,
        lifecycle_in: frozenset[FactLifecycle | DecisionLifecycle] | None,
    ) -> list[dict[str, Any]]:
        """Shared row fetch behind ``scan`` and ``scan_records``, with buried ids filtered out.

        The filter against ``_buried_ids`` is defense in depth: the query can still
        return a row whose burial is recorded but not yet visible to the query planner.
        Filtering here covers both callers (e.g. a hot-cap count must not include a
        buried-but-visible row).
        """
        if lifecycle_in is not None:
            rows = await self._records.query().where(in_("lifecycle", lifecycle_in)).to_list()
        else:
            rows = await self._records.query().to_list()
        return [r for r in rows if row_id(r) not in self._buried_ids]

    async def scan(
        self,
        *,
        lifecycle_in: frozenset[FactLifecycle | DecisionLifecycle] | None = None,
    ) -> AsyncIterator[RowMeta]:
        """Yield metadata-only rows; buried rows are excluded.

        Excludes ids in ``_buried_ids`` even before the physical delete is visible
        (see ``_scan_rows``). LanceDB gives no insertion-order guarantee; only
        ``MemoryAdapter`` iterates in insertion order.
        """
        rows = await self._scan_rows(lifecycle_in)
        return self._rows_to_meta(rows)

    @staticmethod
    def _rows_to_meta(rows: list[dict[str, Any]]) -> AsyncIterator[RowMeta]:
        """Async generator yielding RowMeta for each row."""

        async def _gen() -> AsyncIterator[RowMeta]:
            for row in rows:
                yield row_to_row_meta(row)

        return _gen()

    async def scan_records(
        self,
        *,
        lifecycle_in: frozenset[FactLifecycle | DecisionLifecycle] | None = None,
    ) -> AsyncIterator[Fact | Decision]:
        """Yield full domain records from one query, with no per-candidate ``get_by_id``.

        Same WHERE clause as ``scan``, but each fetched row becomes a domain record
        via ``row_to_record`` instead of a ``RowMeta``, so a caller needs one round
        trip instead of ``scan`` plus N ``get_by_id``. Buried rows are excluded
        twice: the physical delete removes them, and ``_scan_rows`` filters
        ``_buried_ids``.
        """
        rows = await self._scan_rows(lifecycle_in)
        return self._rows_to_records(rows)

    @staticmethod
    def _rows_to_records(rows: list[dict[str, Any]]) -> AsyncIterator[Fact | Decision]:
        """Async generator yielding a domain record for each row."""

        async def _gen() -> AsyncIterator[Fact | Decision]:
            for row in rows:
                yield row_to_record(row)

        return _gen()

    # ------------------------------------------------------------------
    # Event log + projections
    # ------------------------------------------------------------------

    async def append_event(self, event: ReinforcementEvent) -> None:
        """Append a reinforcement event to the append-only log."""
        await self._events.add([event_to_row(event)])

    async def query_events(
        self,
        *,
        target_id: HypomnemaId | None = None,
        source: str | None = None,
    ) -> list[ReinforcementEvent]:
        """Return events filtered by optional ``target_id`` and ``source``, ascending by ``at``.

        The explicit ``order_by`` is required: LanceDB does not guarantee row order
        (fragment layout after ``upkeep()`` need not match write order), and
        ``compute_ewma`` trusts the order returned here.
        """
        conditions: list[str] = []
        if target_id is not None:
            conditions.append(eq("target_id", target_id))
        if source is not None:
            conditions.append(eq("source", source))

        q = self._events.query()
        if conditions:
            q = q.where(and_(*conditions))
        q = q.order_by([ColumnOrdering(column_name="at", ascending=True)])
        rows = await q.to_list()

        return [row_to_event(row) for row in rows]

    # ------------------------------------------------------------------
    # Read events: a separate table from ``events``
    # ------------------------------------------------------------------

    async def append_read_event(self, event: ReadEvent) -> None:
        """Append a read event to the append-only ``read_events`` table."""
        await self._read_events.add([read_event_to_row(event)])

    # ------------------------------------------------------------------
    # Assessor verdicts: updated in place by ``id``; see ``storage.types.VerdictRow``
    # ------------------------------------------------------------------

    async def append_verdicts(self, rows: Sequence[VerdictRow]) -> None:
        """See ``StorageAdapter.append_verdicts``."""
        if not rows:
            return
        await self._assessor_verdicts.add([verdict_row_to_row(row) for row in rows])

    async def update_verdicts(self, rows: Sequence[VerdictRow]) -> None:
        """See ``StorageAdapter.update_verdicts``; ``merge_insert("id")`` updates matched rows."""
        if not rows:
            return
        arrow_table = pa.table(
            {
                key: [verdict_row_to_row(row)[key] for row in rows]
                for key in ASSESSOR_VERDICTS_SCHEMA.names
            },
            schema=ASSESSOR_VERDICTS_SCHEMA,
        )
        await (
            self._assessor_verdicts.merge_insert("id")
            .when_matched_update_all()
            .execute(arrow_table)
        )

    async def verdicts_by_bounce(self, bounce_id: str) -> list[VerdictRow]:
        """See ``StorageAdapter.verdicts_by_bounce``."""
        rows = await (
            self._assessor_verdicts.query().where(eq("bounce_id", bounce_id)).to_list()
        )
        results = [row_to_verdict_row(row) for row in rows]
        results.sort(key=lambda row: (row.probe_rank is None, row.probe_rank))
        return results

    async def pending_verdicts(self) -> list[VerdictRow]:
        """See ``StorageAdapter.pending_verdicts``."""
        rows = await self._assessor_verdicts.query().where(eq("status", "pending")).to_list()
        return [row_to_verdict_row(row) for row in rows]

    async def verdicts_by_effect(self, ref: str) -> list[VerdictRow]:
        """See ``StorageAdapter.verdicts_by_effect``."""
        rows = await (
            self._assessor_verdicts.query()
            .where(or_(eq("effect_ref", ref), eq("new_record_id", ref)))
            .to_list()
        )
        return [row_to_verdict_row(row) for row in rows]

    async def verdicts_for_record(self, id: HypomnemaId) -> list[VerdictRow]:
        """See ``StorageAdapter.verdicts_for_record``."""
        id_str = str(id)
        rows = await (
            self._assessor_verdicts.query()
            .where(or_(eq("candidate_id", id_str), eq("new_record_id", id_str)))
            .to_list()
        )
        results = [row_to_verdict_row(row) for row in rows]
        results.sort(key=lambda row: row.created_at, reverse=True)
        return results

    async def scrub_verdict_text(self, candidate_id: HypomnemaId) -> None:
        """See ``StorageAdapter.scrub_verdict_text``."""
        id_str = str(candidate_id)
        rows = await (
            self._assessor_verdicts.query()
            .where(eq("candidate_id", id_str))
            .to_list()
        )
        targets = [
            verdict
            for verdict in (row_to_verdict_row(row) for row in rows)
            if verdict.incoming_text is not None
        ]
        if not targets:
            return
        scrubbed = [target.model_copy(update={"incoming_text": None}) for target in targets]
        await self.update_verdicts(scrubbed)

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

        Does not touch ``updated_at``: projections are not semantic edits, and render
        byte-stability depends on it.

        Raises ``KeyError`` for a buried or missing id, so callers (including the
        lock-free ``rebuild_projections``) can detect a lost race. The merge is
        matched-only, and LanceDB does not raise on a zero-row match, so
        ``num_updated_rows`` is checked. ``_buried_ids`` is checked before the read
        and again right after it: a completed ``bury()`` may not yet be visible to
        the query snapshot, and the read is itself an ``await`` during which a
        ``bury()`` can complete.
        """
        if id in self._buried_ids:
            raise KeyError(f"Unknown or buried id: {id!s}")

        existing = await self._fetch_unguarded(id)
        if id in self._buried_ids:
            raise KeyError(
                f"Unknown or buried id: {id!s} — buried while the read was in flight"
            )
        if existing is None:
            raise KeyError(f"Unknown id: {id!s}")

        ex_rec = existing.record
        updates: dict[str, object] = {
            "recall_count": recall_count,
            "last_referenced_at": last_referenced_at,
        }
        if isinstance(ex_rec, Fact):
            updates["feedback_score"] = feedback_score
            updates["decay_due_at"] = decay_due_at

        # Matched-only so a burial between read and write cannot resurrect the row;
        # zero updated rows means a concurrent ``bury()`` won, so raise ``KeyError``.
        result = await self._rewrite(existing, updates)
        if result.num_updated_rows == 0:
            raise KeyError(
                f"Unknown id: {id!s} — buried between the read and the write"
            )

    async def update_recurrence(
        self,
        id: HypomnemaId,
        *,
        owner_recurrence_count: int,
        agent_recurrence_count: int,
        escalation_baseline: int,
    ) -> None:
        """Persist the ember recurrence-ladder counters verbatim.

        Does not touch ``updated_at``, like ``update_projections``. Facts only.
        """
        existing = await self._fetch_unguarded(id)
        if existing is None:
            raise KeyError(f"Unknown id: {id!s}")

        ex_rec = existing.record
        if not isinstance(ex_rec, Fact):
            raise ValueError(
                f"update_recurrence is Facts-only (id={id!s} is a {ex_rec.type!r})"
            )

        await self._rewrite(
            existing,
            {
                "owner_recurrence_count": owner_recurrence_count,
                "agent_recurrence_count": agent_recurrence_count,
                "escalation_baseline": escalation_baseline,
            },
        )

    # ------------------------------------------------------------------
    # Distinct-pair registry
    # ------------------------------------------------------------------

    async def mark_distinct(self, pair: DistinctPair) -> None:
        """Register a judged-distinct pair; first-judgment-wins, idempotent.

        Lock-free check-then-write: the cache is checked, then the table, then written.
        """
        # Normalize: lower_id < higher_id lexicographically
        a_str, b_str = str(pair.lower_id), str(pair.higher_id)
        if a_str > b_str:
            a_str, b_str = b_str, a_str
        a_id = HypomnemaId(uuid.UUID(a_str))
        b_id = HypomnemaId(uuid.UUID(b_str))
        cache_key: frozenset[HypomnemaId] = frozenset({pair.lower_id, pair.higher_id})

        # First-judgment-wins: check in-memory cache first
        if cache_key in self._distinct:
            return

        # Also check DB (for reopen scenarios)
        existing = await (
            self._pairs.query()
            .where(and_(eq("lower_id", a_str), eq("higher_id", b_str)))
            .to_list()
        )
        if existing:
            # Populate cache from DB entry (reopen case)
            self._distinct[cache_key] = row_to_distinct_pair(existing[0])
            return

        normalized_pair = DistinctPair(
            lower_id=a_id,
            higher_id=b_id,
            source=pair.source,
            judged_at=pair.judged_at,
            similarity=pair.similarity,
        )
        await self._pairs.add([distinct_pair_to_row(normalized_pair)])
        self._distinct[cache_key] = normalized_pair

    async def is_distinct(self, a: HypomnemaId, b: HypomnemaId) -> bool:
        """Return True if the pair is in the distinct registry (symmetric)."""
        cache_key: frozenset[HypomnemaId] = frozenset({a, b})
        return cache_key in self._distinct

    # ------------------------------------------------------------------
    # Embedding maintenance
    # ------------------------------------------------------------------

    async def backfill_targets(self) -> list[BackfillTarget]:
        """Return rows with a null vector or a stale ``embedding_model``, excluding buried rows.

        ``vector IS NULL`` catches a row written during an embedder outage;
        ``embedding_model != '<current>'`` catches a row embedded by a previous model.
        """
        rows = await (
            self._records.query()
            .where(or_(is_null("vector"), ne("embedding_model", self._embedding_model)))
            .select(["id", "content", "content_hash"])
            .to_list()
        )
        return [row_to_backfill_target(row) for row in rows if row_id(row) not in self._buried_ids]

    async def backfill(
        self,
        id: HypomnemaId,
        vector: list[float],
        *,
        embedding_model: str,
        expected_hash: str,
    ) -> bool:
        """Write only ``vector`` and ``embedding_model``, guarded by ``content_hash``.

        Uses one atomic ``table.update(where="id = ... AND content_hash = ...")``
        (not ``merge_insert``) that LanceDB evaluates against the committed row at
        commit time. That makes it safe while a second connection, such as
        ``tabularium backfill`` against a running server, holds a stale view: a
        two-step read-then-write could match a stale hash against a row a concurrent
        editor already changed. A hash mismatch updates zero rows and returns
        ``False``; there is no whole-row fallback.
        """
        if len(vector) != self._dim:
            raise ValueError(
                f"Backfill vector dimension mismatch (id={id!s}): "
                f"expected {self._dim}, got {len(vector)}"
            )
        physical_width = (await self._records.schema()).field("vector").type.list_size
        if physical_width != self._dim:
            raise ValueError(
                f"Backfill refused (id={id!s}): configured dim={self._dim} does not "
                f"match the physical vector column width={physical_width} — a "
                "dimension change needs a migration, not a reopen"
            )
        if id in self._buried_ids:
            return False

        result = await self._records.update(
            {
                "vector": [float(x) for x in vector],
                "embedding_model": embedding_model,
            },
            where=and_(eq("id", id), eq("content_hash", expected_hash)),
        )
        return bool(int(result.rows_updated) > 0)

    # ------------------------------------------------------------------
    # Registries
    # ------------------------------------------------------------------

    async def _registry_add(self, tbl: LanceTable, name: str) -> None:
        """Idempotent registry insert (first wins).

        The predicate is built with ``lance_filters.eq``: registry names are free
        text, and a quote in one stays inside its literal.
        """
        existing = await tbl.query().where(eq("name", name)).to_list()
        if existing:
            return
        await tbl.add([registry_name_to_row(name)])

    async def _registry_list(self, tbl: LanceTable) -> list[str]:
        """Return all registered names."""
        rows = await tbl.query().to_list()
        return [row_to_registry_name(row) for row in rows]

    async def _registry_delete(self, tbl: LanceTable, name: str) -> None:
        """Idempotent registry delete; a no-op if ``name`` is absent.

        Uses the same ``lance_filters.eq`` predicate as ``_registry_add``.
        """
        await tbl.delete(eq("name", name))

    async def create_scope(self, name: Scope) -> None:
        """Register a scope.  Idempotent."""
        await self._registry_add(self._scopes_tbl, str(name))

    async def list_scopes(self) -> list[Scope]:
        """Return registered scopes."""
        return [Scope(n) for n in await self._registry_list(self._scopes_tbl)]

    async def create_facet(self, name: str) -> None:
        """Register a facet value.  Idempotent."""
        await self._registry_add(self._facets_tbl, name)

    async def list_facets(self) -> list[str]:
        """Return registered facets."""
        return await self._registry_list(self._facets_tbl)

    async def create_role(self, name: str) -> None:
        """Register a role.  Idempotent."""
        await self._registry_add(self._roles_tbl, name)

    async def list_roles(self) -> list[str]:
        """Return registered roles."""
        return await self._registry_list(self._roles_tbl)

    async def delete_role(self, name: str) -> None:
        """Deregister a role; a no-op if it is not registered.

        Never touches a record's ``audience_hint`` (see the ``StorageAdapter``
        Protocol).
        """
        await self._registry_delete(self._roles_tbl, name)

    # ------------------------------------------------------------------
    # Sessions: a parallel seam, apart from the ``records`` table
    # ------------------------------------------------------------------

    async def append_session(self, session: Session) -> None:
        """Append a session row to the ``sessions`` table.

        Append-only, with no update or delete leg. Unreachable from ``vector_search``,
        ``text_search``, ``scan`` and ``scan_records``, which query ``self._records``.
        """
        await self._sessions.add([session_to_row(session)])

    async def latest_session_end(self, *, scope: str) -> datetime | None:
        """Maximum ``ended_at`` among sessions in ``scope``, or ``None``."""
        rows = await (
            self._sessions.query()
            .where(eq("scope", scope))
            .select(["ended_at"])
            .to_list()
        )
        if not rows:
            return None
        return max(row_to_ended_at(row) for row in rows)

    async def search_sessions(
        self,
        *,
        scopes: Sequence[str],
        query: str | None,
        since: datetime | None,
        until: datetime | None,
        limit: int,
    ) -> list[Session]:
        """Search ``sessions``: BM25 on ``content`` given a ``query``, else a scoped-range fetch.

        Both apply the scope IN-list and inclusive ``ended_at`` predicate.
        ``nearest_to_text`` works on a vectorless FTS table without a prior
        ``nearest_to(vector)``. With ``query=None`` results are ``ended_at``
        descending through an explicit ``order_by`` (LanceDB guarantees no row order);
        with a ``query`` they are in BM25 relevance order.
        """
        where = sessions_where_clause(scopes, since=since, until=until)
        if query is not None:
            q = self._sessions.query().nearest_to_text(query).where(where).limit(limit)
        else:
            q = (
                self._sessions.query()
                .where(where)
                .order_by([ColumnOrdering(column_name="ended_at", ascending=False)])
                .limit(limit)
            )
        rows = await q.to_list()
        return [row_to_session(row) for row in rows]

    # ------------------------------------------------------------------
    # Cleaner state: per scope, apart from ``records``
    # ------------------------------------------------------------------

    async def get_cleaner_state(self, *, scope: Scope) -> CleanerState | None:
        """Return this scope's Cleaner state, or ``None`` if never judged.

        A scope with no row returns ``None``, never a default; the caller decides what
        "never judged" means.
        """
        rows = await self._cleaner_state.query().where(eq("scope", scope)).to_list()
        if not rows:
            return None
        return row_to_cleaner_state(rows[0])

    async def set_cleaner_state(self, *, scope: Scope, state: CleanerState) -> None:
        """Upsert this scope's Cleaner state.

        ``merge_insert`` keyed on ``scope``, with no ``where=``; a second call for the
        same scope replaces the row.
        """
        row = cleaner_state_to_row(scope, state)
        arrow_table = pa.table(
            {k: [v] for k, v in row.items()},
            schema=CLEANER_STATE_SCHEMA,
        )
        await (
            self._cleaner_state.merge_insert("scope")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(arrow_table)
        )

    # ------------------------------------------------------------------
    # Born-since query: the pool for one Cleaner pass
    # ------------------------------------------------------------------

    async def list_born_since(
        self, *, scope: Scope, since: datetime, limit: int
    ) -> list[Hypomnema]:
        """Records in ``scope`` created strictly after ``since``, oldest first, up to ``limit``.

        A purpose-built query on ``records``: the existing whole-store enumerations
        (``scan``, ``scan_records``) are unordered, unlimited and unscoped. No
        lifecycle or validity filter, so Facts and Decisions both come back. Ids in
        ``_buried_ids`` are filtered out in case a completed burial is not yet visible
        to the query snapshot.
        """
        where = and_(eq("scope", scope), ts_cmp("created_at", ">", to_utc(since)))
        rows = await (
            self._records.query()
            .where(where)
            .order_by([ColumnOrdering(column_name="created_at", ascending=True)])
            .limit(limit)
            .to_list()
        )
        return [row_to_record(row) for row in rows if row_id(row) not in self._buried_ids]

    # ------------------------------------------------------------------
    # Pass history: append-only, beside ``cleaner_state``
    # ------------------------------------------------------------------

    async def append_cleaner_pass(self, *, record: CleanerPassRecord) -> None:
        """Append one Cleaner-pass record to ``pass_history``.

        Unconditional ``table.add([row])``, unlike the ``merge_insert`` upsert in
        ``set_cleaner_state``. Two calls for one scope leave two rows.
        """
        row = cleaner_pass_to_row(record)
        await self._pass_history.add([row])

    async def list_cleaner_passes(
        self, *, scope: Scope, limit: int
    ) -> list[CleanerPassRecord]:
        """Return ``scope``'s pass-history rows, newest first, up to ``limit``.

        A scope with no rows returns ``[]``. Ordered by an explicit ``order_by`` on
        ``pass_at`` descending (LanceDB guarantees no row order).
        """
        rows = await (
            self._pass_history.query()
            .where(eq("scope", scope))
            .order_by([ColumnOrdering(column_name="pass_at", ascending=False)])
            .limit(limit)
            .to_list()
        )
        return [row_to_cleaner_pass(row) for row in rows]

    # ------------------------------------------------------------------
    # Record reviews: the revisit axis's rotation pool
    # ------------------------------------------------------------------

    async def append_review(self, review: RecordReview) -> None:
        """Append one review row to ``record_reviews`` with an unconditional ``table.add``.

        Two calls for one ``target_id`` leave two rows.
        """
        row = record_review_to_row(review)
        await self._record_reviews.add([row])

    async def list_due_for_review(
        self, *, scope: Scope, now: datetime, created_before: datetime, limit: int
    ) -> list[Fact]:
        """Facts in ``scope`` due for another revisit-axis look; see the Protocol for the contract.

        Two reads composed in Python (LanceDB has no join): every review row for
        ``scope`` reduced to the latest per ``target_id`` by ``reviewed_at`` (strict
        ``>``; LanceDB gives no ordering guarantee, so the reduction must not become
        ``>=``), then ``records`` filtered to the scope's ``active`` / ``archived``,
        valid, never-superseded Facts created at or before ``created_before``. The
        due or never-reviewed predicate and the nulls-first, oldest-reviewed-first
        ordering run over the reduced map, then ``limit`` caps. Ids in ``_buried_ids``
        are filtered out, as in ``list_born_since``.

        Invalid or superseded Facts are not candidates: an invalid Fact reaching this
        pool could be shelved on a unanimous verdict, removing it from the dedup probe.
        That differs from the probe's own ``validity`` handling, which deliberately
        admits invalid candidates for matching.
        """
        review_rows = await self._record_reviews.query().where(eq("scope", scope)).to_list()
        latest_by_target = latest_review_per_target(
            row_to_record_review(row) for row in review_rows
        )

        where = and_(
            eq("scope", scope),
            in_("lifecycle", list(REVISIT_POOL_LIFECYCLES)),
            eq("validity", "valid"),
            is_null("superseded_by"),
            ts_cmp("created_at", "<=", to_utc(created_before)),
        )
        rows = await self._records.query().where(where).to_list()

        pool = (
            PoolCandidate(row_id(row), row_created_at(row), row)
            for row in rows
            if row_id(row) not in self._buried_ids
        )
        candidates = order_due_candidates(pool, latest_by_target, now=now)

        result: list[Fact] = []
        for row in candidates[:limit]:
            record = row_to_record(row)
            assert isinstance(record, Fact)  # lifecycle IN ('active','archived') is Fact-only
            result.append(record)
        return result

    async def list_reviews_for(self, *, target_id: HypomnemaId) -> list[RecordReview]:
        """Every review row for ``target_id``, ascending by ``reviewed_at``.

        A narrow single-id lookback, not a pool-selection export (see the Protocol).
        The explicit ``order_by`` is required: LanceDB guarantees no row order.
        """
        rows = await (
            self._record_reviews.query()
            .where(eq("target_id", target_id))
            .order_by([ColumnOrdering(column_name="reviewed_at", ascending=True)])
            .to_list()
        )
        return [row_to_record_review(row) for row in rows]

    # ------------------------------------------------------------------
    # Record judgments; see ``storage.types.RecordJudgment``
    # ------------------------------------------------------------------

    async def append_record_judgment(self, judgment: RecordJudgment) -> None:
        """Append one ``RecordJudgment`` row to ``record_judgments`` via ``table.add``.

        Two calls for one ``target_id`` leave two rows.
        """
        row = record_judgment_to_row(judgment)
        await self._record_judgments.add([row])

    # ------------------------------------------------------------------
    # Tombstone lookup
    # ------------------------------------------------------------------

    async def get_tombstone(self, id: HypomnemaId) -> Tombstone | None:
        """Return the tombstone for ``id`` if buried, else ``None``. Never raises.

        Reads the cache loaded at open and maintained by ``bury()``.
        """
        return self._tombstone_cache.get(id)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _named_tables(self) -> dict[str, LanceTable]:
        """Every table handle, by table name, in ``_TABLE_SPECS`` order."""
        return {spec.name: getattr(self, spec.attr) for spec in _TABLE_SPECS}

    async def maintenance_debt(self) -> MaintenanceDebt:
        """Report on-disk debt for ``records`` and ``read_events``.

        Those are the two tables with a per-row index that goes stale against its
        build snapshot (``records``' FTS on ``content``, ``read_events``' BTree on
        ``target_id``); an unfolded index turns a seek into a scan.
        ``num_small_fragments`` and ``num_unindexed_rows`` sum both tables.
        ``num_rows`` counts ``records`` only, per ``MaintenanceDebt``.

        Its consumer, ``maybe_upkeep`` (``engine/maintenance.py``), reads only
        ``num_small_fragments`` and runs from write paths, never reads;
        ``log_read_events`` deliberately does not call it (unsafe at read frequency).
        The other tables carry no per-row index. A pure read.
        """
        tables = self._named_tables()

        records_stats = await tables["records"].stats()
        records_indices = await tables["records"].list_indices()
        records_unindexed = sum(idx.num_unindexed_rows for idx in records_indices)

        read_events_stats = await tables["read_events"].stats()
        read_events_indices = await tables["read_events"].list_indices()
        read_events_unindexed = sum(idx.num_unindexed_rows for idx in read_events_indices)

        return MaintenanceDebt(
            num_small_fragments=(
                records_stats["fragment_stats"]["num_small_fragments"]
                + read_events_stats["fragment_stats"]["num_small_fragments"]
            ),
            num_rows=records_stats["num_rows"],
            num_unindexed_rows=records_unindexed + read_events_unindexed,
        )

    async def upkeep(self) -> None:
        """Compact fragments and fold pending writes into every index of every table.

        Covers all tables in ``_TABLE_SPECS``, not just ``records``: other tables take
        writes without touching ``records`` (e.g. ``confirm_distinct`` writes
        ``pairs``), so maintaining everything replaces a per-table trigger.

        Always passes ``cleanup_older_than=_UPKEEP_RETENTION`` (365 days): a bare
        ``optimize()`` defaults to 7 days and would prune week-old versions. Pruning is
        ``reclaim``'s job.
        """
        for tbl in self._named_tables().values():
            await tbl.optimize(cleanup_older_than=_UPKEEP_RETENTION)

    async def reclaim(self) -> None:
        """Prune all non-latest versions across every table. Irreversible.

        Passes ``cleanup_older_than=0`` (prune immediately) and
        ``delete_unverified=True`` (remove leftover files from in-progress-looking
        transactions regardless of age). ``upkeep()`` never does this. Covers the same
        tables as ``upkeep()``.

        PRECONDITION: no other process may be working on this dataset (LanceDB's
        corruption warning; also in ``docs/backup.md``). Never invoked automatically;
        only by explicit call (``tabularium reclaim``).
        """
        for tbl in self._named_tables().values():
            await tbl.optimize(
                cleanup_older_than=_RECLAIM_RETENTION,
                delete_unverified=True,
            )
