"""LanceDB-specific: the ``provisional`` column (curation block)
must be backfilled onto a store written before this task, defaulting to
non-provisional — and the migrated store must open, read, and write without
error afterward.

The dual-parameterized conformance suite (and ``tests/
test_ember_creation_audience.py``) structurally cannot catch this class:
every test there builds a FRESH ``records`` table at the CURRENT declared
schema, so it never exercises a table carrying a legacy, narrower physical
fragment. This is the single most-repeated hazard in this project — the
same shape that produced the ``owner_recurrence_count``/``agent_recurrence_
count``/``escalation_baseline`` migration (``tests/test_escalation.py::
test_lancedb_reopen_migrates_missing_recurrence_columns``) and, before that,
the ``events.reason`` migration. Adding a domain field to ``Fact`` silently
breaks every pre-existing store: write paths build Arrow rows against the
new schema unconditionally and fail inside ``merge_insert`` with
``Field 'provisional' not found in target schema`` unless the open-time
migration (``LanceDBAdapter.open`` → ``_migrate_table_schema``) backfills
the column first.

Backfill polarity (owner decision, curation block): existing
records are NOT provisional. The ten hot records were explicitly curated by
the owner on 2026-08-03 and must not be re-opened for adjudication;
everything else pre-existing is unhinted and off the push tiers anyway. A
``False`` default is therefore both safe and semantically right — and it is
also the SAME default ``_SCALAR_DEFAULTS`` already applies to any missing
non-nullable boolean column (``pa.types.is_boolean -> "false"``), so this
migration needs no new registered default, only the new field on the target
schema.

Falsifiability: every test in this file was confirmed to fail against the
UNMODIFIED (pre-fix) codebase — either ``lancedb.InvalidInput``/``Field
'provisional' not found in target schema`` (no field on ``Fact``/no schema
column/no migration call) or an ``AttributeError``/``KeyError`` reading
``rec.provisional`` (no domain field) — then pass after the fix.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from typing import Any

import lancedb
import pyarrow as pa

from tabularium.config import EngineConfig
from tabularium.domain import Fact, Scope, new_id
from tabularium.engine import Tabularium
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import make_records_schema, record_to_row
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DIM = 8


def _old_schema_and_row(
    dim: int, embedding_model: str, fact: Fact, vector: list[float]
) -> tuple[pa.Schema, dict[str, Any]]:
    """The records schema/row as they stood immediately before this task's
    ``provisional`` column — identical to the CURRENT declared schema minus
    that one field, built by filtering rather than hand-copying (so this
    fixture never drifts independently of the codec it is meant to predate).
    """
    current_schema = make_records_schema(dim)
    old_fields = [f for f in current_schema if f.name != "provisional"]
    old_schema = pa.schema(old_fields)

    full_row = record_to_row(fact, vector, embedding_model)
    old_field_names = {f.name for f in old_fields}
    old_row = {k: v for k, v in full_row.items() if k in old_field_names}
    assert "provisional" not in old_row
    return old_schema, old_row


async def _seed_pre_migration_store(
    tmp_path: pathlib.Path, name: str, embedding_model: str, fact: Fact, vector: list[float]
) -> pathlib.Path:
    """Write ``fact`` directly against the pre-existing schema, bypassing
    ``LanceDBAdapter`` entirely (it always writes the CURRENT schema) —
    the exact shape data written by the pre-diff codebase would have.
    """
    path = tmp_path / name
    old_schema, old_row = _old_schema_and_row(_DIM, embedding_model, fact, vector)
    db = await lancedb.connect_async(str(path))
    old_table = await db.create_table("records", schema=old_schema)
    await old_table.add([old_row])
    physical_field_count = len(await old_table.schema())
    current_field_count = len(make_records_schema(_DIM))
    assert physical_field_count == current_field_count - 1, (
        "fixture drifted: expected exactly one column (provisional) fewer "
        f"than the current schema, got {physical_field_count} vs "
        f"{current_field_count}"
    )
    del db, old_table  # close this raw connection before the adapter reopens it
    return path


# ---------------------------------------------------------------------------
# 1. Adapter-level: open() must migrate, backfilling False, and every
#    subsequent write/read path must round-trip clean.
# ---------------------------------------------------------------------------


class TestProvisionalMigrationAdapterLevel:
    async def test_reopen_migrates_missing_provisional_column_defaulting_false(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: comment out the ``provisional`` field in
        ``make_records_schema`` (or the ``records`` ``_migrate_table_
        schema`` call in ``LanceDBAdapter.open``) — this test goes red:
        without the schema field, ``record_to_row``'s unconditional
        ``"provisional": ...`` key makes every write past this point raise
        ``Field 'provisional' not found in target schema``; without the
        migration call, the physical table never gains the column at all
        and the same failure reproduces on the very first post-reopen
        write below.
        """
        embedder = FakeEmbedder(dim=_DIM)
        old_fact_id = new_id()
        old_fact = Fact(
            id=old_fact_id,
            type="fact",
            content="a fact saved before the provisional column existed",
            scope=Scope("global"),
            lifecycle="ember",
            valence="guideline",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        vector = embedder.peek(old_fact.content)

        path = await _seed_pre_migration_store(
            tmp_path, "pre_provisional_store", embedder.model, old_fact, vector
        )

        adapter = await LanceDBAdapter.open(str(path), embedding_model=embedder.model, dim=_DIM)
        physical = await adapter._records.schema()
        assert "provisional" in {f.name for f in physical}, (
            "the open() migration must have widened the physical schema — "
            "this is the precondition the whole test exists to construct"
        )

        migrated = await adapter.get_by_id(old_fact_id)
        assert isinstance(migrated, Fact)
        assert migrated.provisional is False, (
            "backfill polarity: a pre-existing record must default to "
            "NON-provisional, never re-opened for adjudication"
        )

        # transition() — a lifecycle change on the migrated row must
        # succeed and must not disturb the backfilled default.
        await adapter.transition(old_fact_id, "active", at=EPOCH)
        after_transition = await adapter.get_by_id(old_fact_id)
        assert isinstance(after_transition, Fact)
        assert after_transition.lifecycle == "active"
        assert after_transition.provisional is False

        # transition() can also set it explicitly (the owner-retag path,
        # engine/facts.py) — exercised directly at the adapter level here.
        await adapter.transition(old_fact_id, "ember", at=EPOCH, provisional=True)
        set_true = await adapter.get_by_id(old_fact_id)
        assert isinstance(set_true, Fact)
        assert set_true.provisional is True

        # upsert — re-saving the current (migrated) record must round-trip clean.
        await adapter.upsert(set_true, vector)
        rec3 = await adapter.get_by_id(old_fact_id)
        assert isinstance(rec3, Fact)
        assert rec3.provisional is True

        # A second reopen (migration already applied) must be a no-op, not an error.
        adapter2 = await LanceDBAdapter.open(str(path), embedding_model=embedder.model, dim=_DIM)
        rec4 = await adapter2.get_by_id(old_fact_id)
        assert isinstance(rec4, Fact)
        assert rec4.provisional is True

    async def test_new_write_against_a_migrated_table_succeeds(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Dataset-wide poisoning check (same shape as the Decision-
        supersession migration defect): a single legacy fragment anywhere
        in the table must not break a brand-new row's write, living in its
        own fresh, fully-wide fragment.
        """
        embedder = FakeEmbedder(dim=_DIM)
        legacy_fact = Fact(
            id=new_id(),
            type="fact",
            content="legacy pre-migration fact, unrelated to the new write",
            scope=Scope("global"),
            lifecycle="active",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        vector = embedder.peek(legacy_fact.content)
        path = await _seed_pre_migration_store(
            tmp_path, "dataset_wide_provisional_store", embedder.model, legacy_fact, vector
        )

        adapter = await LanceDBAdapter.open(str(path), embedding_model=embedder.model, dim=_DIM)

        new_id_ = new_id()
        new_content = "brand new ember, written entirely after the migration"
        new_fact = Fact(
            id=new_id_,
            type="fact",
            content=new_content,
            scope=Scope("global"),
            lifecycle="ember",
            valence="guideline",
            created_at=EPOCH,
            updated_at=EPOCH,
            provisional=True,
        )
        await adapter.upsert(new_fact, embedder.peek(new_content))

        rec = await adapter.get_by_id(new_id_)
        assert isinstance(rec, Fact)
        assert rec.provisional is True


# ---------------------------------------------------------------------------
# 2. Engine-level: the real ember() creation path, driven against a
#    migrated store — proves the fix at the level a caller actually uses,
#    not just the adapter seam.
# ---------------------------------------------------------------------------


class TestProvisionalMigrationViaEngine:
    async def test_agent_authored_ember_against_migrated_store_is_provisional(
        self, tmp_path: pathlib.Path
    ) -> None:
        embedder = FakeEmbedder(dim=_DIM)
        legacy_fact = Fact(
            id=new_id(),
            type="fact",
            content="a pre-existing fact, unrelated to the new ember below",
            scope=Scope("global"),
            lifecycle="active",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        vector = embedder.peek(legacy_fact.content)
        path = await _seed_pre_migration_store(
            tmp_path, "engine_level_provisional_store", embedder.model, legacy_fact, vector
        )

        clock = TestClock(EPOCH)
        adapter = await LanceDBAdapter.open(
            str(path), embedding_model=embedder.model, dim=_DIM, clock=clock
        )
        cfg = EngineConfig(candidate_k=1000, gray_floor=0.80)
        engine = Tabularium(adapter, embedder, clock, cfg, token_counter=FakeTokenCounter())
        await engine.initialize()

        r = await engine.ember(
            "an agent-surfaced correction written against a migrated store",
            scope="global",
            valence="guideline",
        )
        assert r.outcome == "saved", r
        assert r.hypomnema is not None

        rec = await adapter.get_by_id(r.hypomnema.id)
        assert isinstance(rec, Fact)
        assert rec.provisional is True
        assert set(rec.audience_hint or ()) == {"decider", "builder", "reviewer", "navigator"}

        # The pre-existing legacy row must still read back clean, unaffected.
        legacy_after = await adapter.get_by_id(legacy_fact.id)
        assert isinstance(legacy_after, Fact)
        assert legacy_after.provisional is False
