"""P1 schema piece: the nine new columns exist after ``open()`` on a legacy
store, the codec round-trips the six later-piece fields, and the storage
layer derives ``content_key``/``content_hash`` on every row write.

Falsifiability (Scope 2 Verify): comment out the ``record_to_row``
copy-through for ``supersession_note`` and ``test_round_trip_optional_fields``
goes red.

Contract-grounded reason ``test_legacy_store_gains_columns`` and
``test_derive_at_write`` are LanceDB-only, not dual-parameterized: both read
the physical Arrow row directly (schema names, raw NULLs) — ``MemoryAdapter``
has no physical schema or column concept to migrate or inspect.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
import pytest

from tabularium.domain import Decision, Fact, HypomnemaId, Scope, new_id
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import TOMBSTONES_SCHEMA, UNSUPERSEDES_SCHEMA
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import PredecessorFlip, UnsupersedeEvent
from tests.support.doubles import TestClock

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DIM = 8

_FIXTURE = Path(__file__).parent / "fixtures" / "legacy-store-294fa14"

_NEW_RECORD_COLUMNS = (
    "content_key",
    "content_hash",
    "previous_content",
    "supersession_reason",
    "supersession_note",
    "supersession_edge_id",
    "invalidated_by",
    "invalidation_reason",
)


def _fact(*, content: str, clock: TestClock, **kw: Any) -> Fact:
    now = clock.now()
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope("proj-a"),
        lifecycle="active",
        validity="valid",
        created_at=now,
        updated_at=now,
        **kw,
    )


async def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "legacy-store"
    shutil.copytree(_FIXTURE, dest)
    return dest


async def test_legacy_store_gains_columns(tmp_path: Path) -> None:
    dest = await _copy_fixture(tmp_path)

    pre_db = await lancedb.connect_async(str(dest))
    pre_tombstones = await (await pre_db.open_table("tombstones")).query().to_list()
    buried_id = pre_tombstones[0]["buried_id"]

    adapter = await LanceDBAdapter.open(str(dest), embedding_model="test-model", dim=_DIM)

    records_schema = await adapter._records.schema()
    for name in _NEW_RECORD_COLUMNS:
        assert name in records_schema.names, f"missing column {name!r}"

    tombstones_schema = await adapter._tombstones_tbl.schema()
    assert "restores" in tombstones_schema.names

    old_rows = await adapter._records.query().to_list()
    assert old_rows, "fixture should carry pre-existing rows"
    for row in old_rows:
        for name in _NEW_RECORD_COLUMNS:
            assert row.get(name) is None, f"{name!r} should be NULL on a legacy row"

    old_tombstones = await adapter._tombstones_tbl.query().to_list()
    for row in old_tombstones:
        assert row.get("restores") is None

    assert buried_id is not None
    assert HypomnemaId(uuid.UUID(buried_id)) in adapter._buried_ids


@pytest.fixture(params=["memory", "lancedb"])
async def adapter(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    clock = TestClock(EPOCH)
    if request.param == "memory":
        return MemoryAdapter(embedding_model="test-model", dim=_DIM, clock=clock)
    return await LanceDBAdapter.open(
        str(tmp_path / "store"), embedding_model="test-model", dim=_DIM, clock=clock
    )


async def test_round_trip_optional_fields(adapter: Any) -> None:
    clock = TestClock(EPOCH)
    fact = _fact(
        content="A fact with a supersession note already set.",
        clock=clock,
        supersession_note="x",
        supersession_edge_id="e1",
    )
    vector = [1.0] + [0.0] * (_DIM - 1)
    await adapter.upsert(fact, vector)

    await adapter.update_projections(
        fact.id,
        recall_count=1,
        last_referenced_at=clock.now(),
        feedback_score=0.5,
        decay_due_at=None,
    )

    read_back = await adapter.get_by_id(fact.id)
    assert read_back.supersession_note == "x"
    assert read_back.supersession_edge_id == "e1"


async def test_restores_marked_tombstone_excluded_from_buried_ids_at_open(
    tmp_path: Path,
) -> None:
    """P4 two-arm unsupersede piece: a restores-
    marked tombstone (``plant_tombstone``, not ``bury``) must NOT be added
    to ``_buried_ids`` at open time, and its row must NOT be deleted by the
    legacy crash-repair sweep — that id is not actually buried yet, and
    finishing the undo is ``repair_unsupersedes``'s job, run later from
    ``Tabularium.initialize()``, never here at ``open()``.

    Falsify: drop the ``restores is None`` guard from ``_load_buried_ids``
    — this test goes red, the successor's row is deleted at ``open()`` and
    ``successor.id`` lands in ``_buried_ids`` before repair ever runs.
    """
    from tabularium.domain import Decision

    store_path = tmp_path / "restores_marked_store"
    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=_DIM)
    clock = TestClock(EPOCH)

    def _decision(*, content: str) -> Decision:
        now = clock.now()
        return Decision(
            id=new_id(),
            type="decision",
            content=content,
            scope=Scope("proj-a"),
            lifecycle="in_force",
            validity="valid",
            created_at=now,
            updated_at=now,
        )

    predecessor = _decision(content="Decision undone by an interrupted repair")
    successor = _decision(content="successor Decision, tombstoned but not yet deleted")
    vector = [1.0] + [0.0] * (_DIM - 1)
    await adapter.upsert(predecessor, vector)
    await adapter.upsert(successor, [0.9] + [0.0] * (_DIM - 1))

    from tabularium.storage.types import PredecessorFlip

    flip = PredecessorFlip(id=predecessor.id, reason="correction", note=None, edge_id="e-1")
    await adapter.replace(successor, None, [flip], at=EPOCH)
    await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)

    reopened = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=_DIM)
    assert successor.id not in reopened._buried_ids
    still_present = await reopened.get_by_id(successor.id)
    assert isinstance(still_present, Decision)
    tombstone = await reopened.get_tombstone(successor.id)
    assert tombstone is not None
    assert tombstone.restores == predecessor.id


async def test_legacy_tombstone_without_hash_column_migrates_and_refuses(
    tmp_path: Path,
) -> None:
    """Edit-during-pending-undo fix: a tombstone
    planted by pre-fix code has no ``restores_content_hash`` COLUMN at
    all, not merely a NULL value in an existing one — a genuinely legacy
    physical schema, built the same way ``test_legacy_store_gains_columns``
    above proves the migration path rather than a fresh-store shortcut.
    This proves the open-time migration backfills the column honestly (a
    typed NULL, not a crash or a silently-wrong default) and that repair
    treats that backfilled NULL as a mismatch — refusing and withdrawing
    the stale tombstone — rather than silently completing the undo and
    deleting the still-live, never-examined successor row.

    Falsify: drop the ``tombstone.restores_content_hash != content_hash(
    successor.content)`` compare from ``resolve_unsupersede_repair`` —
    this test's ``refused == 1`` assertion flips to ``completed == 1`` and
    the successor is deleted.
    """
    store_path = tmp_path / "legacy_hash_store"
    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=_DIM)
    clock = TestClock(EPOCH)

    def _decision(*, content: str) -> Decision:
        now = clock.now()
        return Decision(
            id=new_id(), type="decision", content=content, scope=Scope("proj-a"),
            lifecycle="in_force", validity="valid", created_at=now, updated_at=now,
        )

    predecessor = _decision(content="Decision A, undo left pending by pre-fix code")
    successor = _decision(content="Decision B, tombstoned before the hash column existed")
    vector = [1.0] + [0.0] * (_DIM - 1)
    await adapter.upsert(predecessor, vector)
    await adapter.upsert(successor, [0.9] + [0.0] * (_DIM - 1))

    flip = PredecessorFlip(
        id=predecessor.id, reason="correction", note=None, edge_id="e-legacy-hash",
    )
    await adapter.replace(successor, None, [flip], at=EPOCH)
    event = UnsupersedeEvent(
        id=uuid.uuid4(), edge_id=flip.edge_id, predecessor_id=predecessor.id,
        successor_id=successor.id, arm="decision", withdrawn_reason="correction",
        withdrawn_note=None, reason="pre-fix pending undo", at=EPOCH,
    )
    await adapter.append_unsupersede(event)

    legacy_schema = pa.schema(
        [f for f in TOMBSTONES_SCHEMA if f.name != "restores_content_hash"]
    )
    db = await lancedb.connect_async(str(store_path))
    await db.drop_table("tombstones")
    legacy_table = await db.create_table("tombstones", schema=legacy_schema)
    await legacy_table.add([{
        "tombstone_id": str(uuid.uuid4()),
        "buried_id": str(successor.id),
        "type": "decision",
        "reason": "unsupersede: pre-fix pending undo",
        "timestamp": EPOCH,
        "model_generated": False,
        "restores": str(predecessor.id),
    }])

    reopened = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=_DIM)
    tombstones_schema = await reopened._tombstones_tbl.schema()
    assert "restores_content_hash" in tombstones_schema.names
    raw_rows = await reopened._tombstones_tbl.query().to_list()
    assert raw_rows[0]["restores_content_hash"] is None

    report = await reopened.repair_unsupersedes()
    completed = report.completed
    skipped = report.skipped
    refused = report.refused
    assert completed == 0
    assert skipped == 0
    assert refused == 1

    still_b = await reopened.get_by_id(successor.id)
    assert isinstance(still_b, Decision)
    assert still_b.lifecycle == "in_force"
    unchanged_a = await reopened.get_by_id(predecessor.id)
    assert isinstance(unchanged_a, Decision)
    assert unchanged_a.lifecycle == "superseded"
    assert unchanged_a.superseded_by == successor.id
    assert await reopened.get_tombstone(successor.id) is None


async def test_legacy_unsupersedes_table_without_refused_at_migrates(
    tmp_path: Path,
) -> None:
    """Durable refusal follow-up: ``refused_at`` was added to the
    ``unsupersedes`` table AFTER that table itself first shipped (unlike
    every other column on it, present since the table's own creation) —
    the one column on this table that genuinely needs an open-time
    migration. Builds a genuinely legacy physical schema (the column
    absent, not merely NULL), same shape as ``test_legacy_tombstone_
    without_hash_column_migrates_and_refuses`` above, then proves the
    migrated column round-trips a real write via ``mark_unsupersede_
    refused`` — not just that the name appears in the schema.
    """
    store_path = tmp_path / "legacy_refused_at_store"
    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=_DIM)
    clock = TestClock(EPOCH)

    def _decision(*, content: str) -> Decision:
        now = clock.now()
        return Decision(
            id=new_id(), type="decision", content=content, scope=Scope("proj-a"),
            lifecycle="in_force", validity="valid", created_at=now, updated_at=now,
        )

    predecessor = _decision(content="Decision A, edge drawn before refused_at existed")
    successor = _decision(content="Decision B, undo event predates the column")
    vector = [1.0] + [0.0] * (_DIM - 1)
    await adapter.upsert(predecessor, vector)
    await adapter.upsert(successor, [0.9] + [0.0] * (_DIM - 1))

    flip = PredecessorFlip(
        id=predecessor.id, reason="correction", note=None, edge_id="e-legacy-refused-at",
    )
    await adapter.replace(successor, None, [flip], at=EPOCH)

    legacy_schema = pa.schema([f for f in UNSUPERSEDES_SCHEMA if f.name != "refused_at"])
    db = await lancedb.connect_async(str(store_path))
    await db.drop_table("unsupersedes")
    legacy_table = await db.create_table("unsupersedes", schema=legacy_schema)
    await legacy_table.add([{
        "id": str(uuid.uuid4()),
        "edge_id": flip.edge_id,
        "predecessor_id": str(predecessor.id),
        "successor_id": str(successor.id),
        "arm": "decision",
        "at": EPOCH,
        "withdrawn_reason": "correction",
        "withdrawn_note": None,
        "reason": "pre-refused_at undo event",
    }])

    reopened = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=_DIM)
    unsupersedes_schema = await reopened._unsupersedes.schema()
    assert "refused_at" in unsupersedes_schema.names
    raw_rows = await reopened._unsupersedes.query().to_list()
    assert raw_rows[0]["refused_at"] is None

    event = await reopened.get_unsupersede(flip.edge_id)
    assert event is not None
    assert event.refused_at is None

    await reopened.mark_unsupersede_refused(flip.edge_id, at=EPOCH)
    reread = await reopened.get_unsupersede(flip.edge_id)
    assert reread is not None
    assert reread.refused_at == EPOCH


async def test_derive_at_write(tmp_path: Path) -> None:
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "store"), embedding_model="test-model", dim=_DIM
    )
    clock = TestClock(EPOCH)
    raw_content = "  a   b c "
    fact = _fact(content=raw_content, clock=clock)
    vector = [1.0] + [0.0] * (_DIM - 1)
    await adapter.upsert(fact, vector)

    rows = await adapter._records.query().where(f"id = '{fact.id!s}'").to_list()
    assert len(rows) == 1
    row = rows[0]
    assert row["content_key"] == "a b c"
    assert row["content_hash"] == hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
