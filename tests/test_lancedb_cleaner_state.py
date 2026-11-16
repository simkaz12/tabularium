"""LanceDB-specific: the ``cleaner_state`` table (2026-08-10 cleaner plan) — a brand-new tenth
table, never a widening of an existing one.

Two things this file proves that the dual-parameterized conformance suite
structurally cannot (it always builds a fresh store at the CURRENT declared
table set):

1. Reopening a store that predates this table (no ``cleaner_state`` table at
   all — same shape as the ``parks`` table's own addition in M2
   slice B) must create it fresh, and both reads and writes against it must
   work afterward, without disturbing pre-existing data in the OTHER nine
   tables.
2. Falsifiability for the upsert property: ``get_cleaner_state`` must never
   manufacture a default for an unknown scope.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime, timedelta

import lancedb

from tabularium.domain import Fact, Scope, new_id
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import (
    EVENTS_SCHEMA,
    PAIRS_SCHEMA,
    REGISTRY_SCHEMA,
    SESSIONS_SCHEMA,
    TOMBSTONES_SCHEMA,
    make_records_schema,
    record_to_row,
)
from tabularium.storage.types import CleanerState

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DIM = 8


async def _seed_pre_cleaner_state_store(
    tmp_path: pathlib.Path, name: str, embedding_model: str, fact: Fact, vector: list[float]
) -> pathlib.Path:
    """Build a store carrying every table ``LanceDBAdapter.open`` creates
    EXCEPT ``cleaner_state`` — the exact physical shape a store predating
    this task would have. Bypasses ``LanceDBAdapter`` entirely (it always
    creates the CURRENT full table set); connects directly via
    ``lancedb.connect_async`` instead, the same technique
    ``test_lancedb_provisional_migration.py`` uses for a column-level
    migration, generalized here to a table-level one.
    """
    path = tmp_path / name
    db = await lancedb.connect_async(str(path))

    records = await db.create_table("records", schema=make_records_schema(_DIM))
    await records.add([record_to_row(fact, vector, embedding_model)])

    await db.create_table("events", schema=EVENTS_SCHEMA)
    await db.create_table("pairs", schema=PAIRS_SCHEMA)
    await db.create_table("tombstones", schema=TOMBSTONES_SCHEMA)
    await db.create_table("scopes", schema=REGISTRY_SCHEMA)
    await db.create_table("facets", schema=REGISTRY_SCHEMA)
    await db.create_table("roles", schema=REGISTRY_SCHEMA)
    await db.create_table("sessions", schema=SESSIONS_SCHEMA)

    existing_tables = await db.list_tables()
    names = (
        set(existing_tables.tables)
        if hasattr(existing_tables, "tables")
        else set(existing_tables)
    )
    assert "cleaner_state" not in names, (
        "fixture drifted: the pre-existing store must NOT carry the "
        "cleaner_state table — that absence is the precondition this file "
        "exists to construct"
    )
    del db, records  # close this raw connection before the adapter reopens it
    return path


class TestCleanerStateTableMigration:
    async def test_reopen_creates_cleaner_state_and_reads_write_work(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: comment out the ``cleaner_state = await
        _open_or_create(...)`` line in ``LanceDBAdapter.open`` — this test
        goes red with an ``AttributeError`` on ``adapter._cleaner_state``
        (never constructed) the moment ``get_cleaner_state``/
        ``set_cleaner_state`` are called below.
        """
        embedding_model = "test-model"
        old_fact = Fact(
            id=new_id(),
            type="fact",
            content="a fact saved before the cleaner_state table existed",
            scope=Scope("proj-a"),
            lifecycle="active",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        vector = [0.1] * _DIM

        path = await _seed_pre_cleaner_state_store(
            tmp_path, "pre_cleaner_state_store", embedding_model, old_fact, vector
        )

        adapter = await LanceDBAdapter.open(str(path), embedding_model=embedding_model, dim=_DIM)

        tables = await adapter._db.list_tables()
        names = set(tables.tables) if hasattr(tables, "tables") else set(tables)
        assert "cleaner_state" in names, (
            "open() must have created the cleaner_state table fresh — this "
            "is the precondition the whole test exists to construct"
        )

        # Read: an unknown scope on a brand-new table is None, not an error.
        assert await adapter.get_cleaner_state(scope=Scope("proj-a")) is None

        # Write: set + get round-trips clean.
        state = CleanerState(watermark_at=EPOCH, last_pass_at=EPOCH, report=None)
        await adapter.set_cleaner_state(scope=Scope("proj-a"), state=state)
        got = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert got == state

        # Upsert: a second write for the same scope replaces, not appends.
        state2 = CleanerState(watermark_at=EPOCH, last_pass_at=EPOCH, report="{}")
        await adapter.set_cleaner_state(scope=Scope("proj-a"), state=state2)
        rows = await adapter._cleaner_state.query().to_list()
        assert len(rows) == 1

        # The pre-existing Fact in the migrated store's OTHER tables must
        # still read back clean, unaffected by the new table's creation.
        migrated_fact = await adapter.get_by_id(old_fact.id)
        assert migrated_fact.id == old_fact.id
        assert migrated_fact.content == old_fact.content

        # A second reopen (table already exists) must be a no-op, not an error.
        adapter2 = await LanceDBAdapter.open(str(path), embedding_model=embedding_model, dim=_DIM)
        got2 = await adapter2.get_cleaner_state(scope=Scope("proj-a"))
        assert got2 == state2

    async def test_new_write_against_migrated_store_succeeds(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A born-since query against a migrated store's records table must
        also work — proving the table-level migration did not somehow
        poison an UNRELATED table's own read path."""
        embedding_model = "test-model"
        legacy_fact = Fact(
            id=new_id(),
            type="fact",
            content="legacy pre-migration fact, unrelated to the new write",
            scope=Scope("proj-a"),
            lifecycle="active",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        vector = [0.2] * _DIM
        path = await _seed_pre_cleaner_state_store(
            tmp_path, "dataset_wide_cleaner_state_store", embedding_model, legacy_fact, vector
        )

        adapter = await LanceDBAdapter.open(str(path), embedding_model=embedding_model, dim=_DIM)

        new_fact = Fact(
            id=new_id(),
            type="fact",
            content="brand new fact, written entirely after the migration",
            scope=Scope("proj-a"),
            lifecycle="active",
            created_at=EPOCH + timedelta(days=1),
            updated_at=EPOCH + timedelta(days=1),
        )
        await adapter.upsert(new_fact, [0.3] * _DIM)

        results = await adapter.list_born_since(scope=Scope("proj-a"), since=EPOCH, limit=10)
        assert [r.id for r in results] == [new_fact.id]
