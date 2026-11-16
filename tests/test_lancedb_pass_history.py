"""LanceDB-specific: the ``pass_history`` table (2026-08-11 cleaner-telemetry plan) — a brand-new
eleventh table, never a widening of an existing one.

Two things this file proves that the dual-parameterized conformance suite structurally cannot
(it always builds a fresh store at the CURRENT declared table set):

1. Reopening a store that predates this table (no ``pass_history`` table at all — same shape as
   the ``parks``/``cleaner_state`` tables' own additions) must create it fresh, and both reads and
   writes against it must work afterward, without disturbing pre-existing data in the OTHER ten
   tables — including a pre-existing ``cleaner_state`` row, which this task's own brief is
   explicit must be left untouched.
2. A write against the migrated store's unrelated ``records`` table still works, proving the
   table-level migration did not poison anything else.

Model: ``_seed_pre_cleaner_state_store`` (``tests/test_lancedb_cleaner_state.py``), generalized
one table further — this fixture creates every table ``LanceDBAdapter.open`` creates EXCEPT
``pass_history``, including ``cleaner_state`` itself (present here, since this task's precondition
is a store that already has ``cleaner_state`` but predates only the new table).
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime, timedelta

import lancedb

from tabularium.domain import Fact, Scope, new_id
from tabularium.engine.axis_common import BandCounts
from tabularium.engine.falsity import FalsityBandCounts
from tabularium.engine.judge import JudgeCallRecord
from tabularium.engine.report import (
    CleanerReport,
    FalsityReportSection,
    RelevanceReportSection,
)
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import (
    CLEANER_STATE_SCHEMA,
    EVENTS_SCHEMA,
    PAIRS_SCHEMA,
    REGISTRY_SCHEMA,
    SESSIONS_SCHEMA,
    TOMBSTONES_SCHEMA,
    cleaner_state_to_row,
    make_records_schema,
    record_to_row,
)
from tabularium.storage.types import CleanerPassRecord, CleanerState

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DIM = 8


async def _seed_pre_pass_history_store(
    tmp_path: pathlib.Path,
    name: str,
    embedding_model: str,
    fact: Fact,
    vector: list[float],
    cleaner_state: CleanerState,
) -> pathlib.Path:
    """Build a store carrying every table ``LanceDBAdapter.open`` creates EXCEPT
    ``pass_history`` — the exact physical shape a store predating this task would have.
    Bypasses ``LanceDBAdapter`` entirely (it always creates the CURRENT full table set); connects
    directly via ``lancedb.connect_async`` instead, same technique as
    ``test_lancedb_cleaner_state.py``.
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

    cleaner_state_tbl = await db.create_table("cleaner_state", schema=CLEANER_STATE_SCHEMA)
    row = cleaner_state_to_row(Scope("proj-a"), cleaner_state)
    await cleaner_state_tbl.add([row])

    existing_tables = await db.list_tables()
    names = (
        set(existing_tables.tables)
        if hasattr(existing_tables, "tables")
        else set(existing_tables)
    )
    assert "pass_history" not in names, (
        "fixture drifted: the pre-existing store must NOT carry the pass_history table — "
        "that absence is the precondition this file exists to construct"
    )
    del db, records, cleaner_state_tbl  # close this raw connection before the adapter reopens it
    return path


class TestPassHistoryTableMigration:
    async def test_reopen_creates_pass_history_and_reads_write_work(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: comment out the ``pass_history = await
        _open_or_create(...)`` line in ``LanceDBAdapter.open`` — this test goes red immediately,
        before any of the assertions below even run: the constructor call at the end of ``open()``
        still passes ``_pass_history=pass_history`` and raises ``NameError: name 'pass_history'
        is not defined``. Verified live 2026-08-11: exit 0 before the break, ``NameError`` at
        ``lancedb_adapter.py``'s ``cls(...)`` call after it, restored to exit 0 again after.
        """
        embedding_model = "test-model"
        old_fact = Fact(
            id=new_id(),
            type="fact",
            content="a fact saved before the pass_history table existed",
            scope=Scope("proj-a"),
            lifecycle="active",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        vector = [0.1] * _DIM
        old_cleaner_state = CleanerState(watermark_at=EPOCH, last_pass_at=EPOCH, report=None)

        path = await _seed_pre_pass_history_store(
            tmp_path,
            "pre_pass_history_store",
            embedding_model,
            old_fact,
            vector,
            old_cleaner_state,
        )

        adapter = await LanceDBAdapter.open(str(path), embedding_model=embedding_model, dim=_DIM)

        tables = await adapter._db.list_tables()
        names = set(tables.tables) if hasattr(tables, "tables") else set(tables)
        assert "pass_history" in names, (
            "open() must have created the pass_history table fresh — this is the "
            "precondition the whole test exists to construct"
        )

        # Read: an unknown scope on a brand-new table is [], not an error.
        assert await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10) == []

        # Write: append + list round-trips clean.
        record = CleanerPassRecord(
            scope=Scope("proj-a"),
            pass_at=EPOCH,
            outcome="completed",
            dry_run=False,
            payload='{"pool_size": 1}',
        )
        await adapter.append_cleaner_pass(record=record)
        got = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert got == [record]

        # Append-only: a second append for the same scope leaves TWO rows, never one.
        record2 = CleanerPassRecord(
            scope=Scope("proj-a"),
            pass_at=EPOCH + timedelta(days=1),
            outcome="completed",
            dry_run=False,
            payload='{"pool_size": 2}',
        )
        await adapter.append_cleaner_pass(record=record2)
        got2 = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(got2) == 2

        # The pre-existing Fact in the migrated store's OTHER tables must still read back
        # clean, unaffected by the new table's creation.
        migrated_fact = await adapter.get_by_id(old_fact.id)
        assert migrated_fact.id == old_fact.id
        assert migrated_fact.content == old_fact.content

        # The pre-existing cleaner_state row -- the one this task's own brief says must be
        # left untouched -- reads back exactly as seeded.
        migrated_state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert migrated_state == old_cleaner_state

        # A second reopen (table already exists) must be a no-op, not an error.
        adapter2 = await LanceDBAdapter.open(str(path), embedding_model=embedding_model, dim=_DIM)
        got3 = await adapter2.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(got3) == 2

    async def test_new_write_against_migrated_store_succeeds(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A born-since query against a migrated store's records table must also work --
        proving the table-level migration did not somehow poison an UNRELATED table's own
        read path."""
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
        legacy_state = CleanerState(watermark_at=EPOCH, last_pass_at=EPOCH, report=None)
        path = await _seed_pre_pass_history_store(
            tmp_path,
            "dataset_wide_pass_history_store",
            embedding_model,
            legacy_fact,
            vector,
            legacy_state,
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

        # And the cleaner_state row seeded before the migration is still exactly as written.
        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state == legacy_state


class TestPassHistoryPayloadWithCallTelemetryRoundTrips:
    """cleaner-telemetry change -- acceptance criterion: "the payload is valid
    JSON and round-trips through the storage seam unchanged". A realistic ``CleanerReport``
    payload (per-call telemetry included) written through ``LanceDBAdapter`` must read back
    byte-for-byte equal through a FRESH adapter instance against the same path -- a genuine
    close-and-reopen, not the same in-process object the write happened through."""

    async def test_payload_with_calls_survives_a_real_close_and_reopen(
        self, tmp_path: pathlib.Path
    ) -> None:
        embedding_model = "test-model"
        path = tmp_path / "pass_history_call_telemetry_round_trip"
        adapter = await LanceDBAdapter.open(str(path), embedding_model=embedding_model, dim=_DIM)

        call_a = JudgeCallRecord(
            batch_index=0,
            judge="claude:sonnet",
            rubric_version="cleaner-rubric-v1-abc123def456",
            duration_seconds=4.5,
            status="success",
            error=None,
            prompt_tokens=1234,
            completion_tokens=None,
            cost=None,
        )
        call_b = JudgeCallRecord(
            batch_index=0,
            judge="codex:gpt-5.1-codex",
            rubric_version="cleaner-rubric-v1-abc123def456",
            duration_seconds=6.1,
            status="invalid_output",
            error="response is not valid JSON: Expecting value: line 1 column 1 (char 0)",
            prompt_tokens=1234,
            completion_tokens=None,
            cost=None,
        )
        report = CleanerReport(
            scope=Scope("proj-a"),
            pass_at=EPOCH,
            outcome="completed",
            relevance=RelevanceReportSection(
                judge_pair=("claude:sonnet", "codex:gpt-5.1-codex"),
                rubric_version="cleaner-rubric-v1-abc123def456",
                counts=BandCounts(shelve=0, keep=1, unclear=0),
                shelved_ids=(),
                unclear_ids=(),
                failures=(),
                calls=(call_a, call_b),
            ),
            falsity=FalsityReportSection(
                judge_pair=("claude:sonnet", "codex:gpt-5.1-codex"),
                rubric_version="cleaner-rubric-v1-abc123def456",
                counts=FalsityBandCounts(replace=0, coexist=0),
                superseded_ids=(),
                global_contradicted=(),
                decision_contradicted=(),
                failures=(),
                calls=(),
            ),
            verification_failures=(),
        )
        payload = report.model_dump_json()

        record = CleanerPassRecord(
            scope=Scope("proj-a"),
            pass_at=EPOCH,
            outcome="completed",
            dry_run=False,
            payload=payload,
        )
        await adapter.append_cleaner_pass(record=record)

        # A FRESH adapter instance against the SAME path -- a genuine reopen.
        reopened = await LanceDBAdapter.open(str(path), embedding_model=embedding_model, dim=_DIM)
        got = await reopened.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(got) == 1
        assert got[0] == record
        assert got[0].payload == payload

        round_tripped_report = CleanerReport.model_validate_json(got[0].payload)
        assert round_tripped_report == report
        assert round_tripped_report.relevance.calls == (call_a, call_b)
