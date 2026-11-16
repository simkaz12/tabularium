"""LanceDB-specific: Decision supersession must survive an ``add_columns``
schema-evolution backfill (live-store defect, broken since 2026-08-01's
``f6274eb``, diagnosed 2026-08-02/03).

The dual-parameterized conformance suite structurally cannot catch this
class: every test there builds a FRESH ``records`` table at the CURRENT
declared schema, so it never exercises a table carrying a legacy, narrower
physical fragment. ``_migrate_table_schema``'s ``add_columns`` backfill
(triggered by ``LanceDBAdapter.open()`` reopening an EXISTING store) leaves
the table holding MIXED physical fragments — pre-migration rows at the old
(narrower) field count, post-migration rows at the new (wider) one. Lance
8.0.0 panics evaluating ANY ``where=`` predicate across such a dataset
(``InvalidArgumentError("Incorrect number of arrays for StructArray
fields...")`` inside ``lance-8.0.0/src/dataset/write/merge_insert.rs``), and
the poisoning is DATASET-WIDE — supersession of ANY Decision breaks once a
single legacy fragment exists anywhere in the table, including a Decision
created entirely after the migration. ``table.optimize()`` does not repair
the fragment layout (verified separately, live-store repro: zero fragments
touched, panic persists).

Fixed by moving the supersede-edge CAS out of the storage ``where=``
predicate and into a Python-side fresh re-read immediately before an
UNCONDITIONAL ``merge_insert`` (see ``LanceDBAdapter.transition``'s
docstring for the full mechanism and its accepted, narrowed — not closed —
concurrency guarantee).

Contract-grounded reason this is a LanceDB-only test file, not a
dual-parameterized conformance property (same exemption shape as
``test_lancedb_update_projections_race.py`` / ``test_lancedb_scan_buried_
guard.py``): ``MemoryAdapter`` has no concept of a physical fragment or a
schema migration at all — there is nothing for it to participate in here.

Falsifiability: every test in this file was confirmed to fail against the
UNMODIFIED (pre-fix) ``LanceDBAdapter.transition`` — the Lance
``InvalidInput`` panic described above, surfacing (pre-fix Part 3 too)
through ``engine.decisions._write_decision``'s misleading wrapper message —
then pass after the fix.
"""

from __future__ import annotations

import asyncio
import pathlib
import uuid
from datetime import UTC, datetime
from typing import Any

import lancedb
import pyarrow as pa
import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, HypomnemaId, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.results import BounceResolution, SaveResult
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import PredecessorFlip
from tests.support.doubles import FakeEmbedder, TestClock
from tests.support.lance_raw import raw_table

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DIM = 64  # matches the established convention (test_engine_concurrency.py,
# test_engine_b2.py): large enough that FakeEmbedder's SHA-256-slot fixture
# strings don't collide by accident.


# ---------------------------------------------------------------------------
# Fixture construction — the pre-f6274eb (23-field) records schema, and a
# helper that seeds a store at that schema WITHOUT going anywhere near
# LanceDBAdapter (which always writes the CURRENT schema) — the same
# construction the /tmp repro scripts used during diagnosis, ported here.
# ---------------------------------------------------------------------------


def _pre_f6274eb_schema(dim: int) -> pa.Schema:
    """The ``records`` schema as it stood immediately before ``f6274eb``
    added ``owner_recurrence_count`` / ``agent_recurrence_count`` /
    ``escalation_baseline`` — 23 fields, hand-copied minus those three (the
    exact shape every row on the deployed store carried before that commit).
    """
    return pa.schema(
        [
            pa.field("id", pa.utf8(), nullable=False),
            pa.field("type", pa.utf8(), nullable=False),
            pa.field("content", pa.utf8(), nullable=False),
            pa.field("scope", pa.utf8(), nullable=False),
            pa.field("purpose", pa.utf8(), nullable=True),
            pa.field("audience_hint", pa.utf8(), nullable=True),
            pa.field("source", pa.utf8(), nullable=True),
            pa.field("validity", pa.utf8(), nullable=False),
            pa.field("reconciles", pa.utf8(), nullable=True),
            pa.field("recall_count", pa.int64(), nullable=False),
            pa.field("last_referenced_at", pa.timestamp("us", tz="UTC"), nullable=True),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("lifecycle", pa.utf8(), nullable=False),
            pa.field("valence", pa.utf8(), nullable=True),
            pa.field("feedback_score", pa.float64(), nullable=False),
            pa.field("hot_rejection_count", pa.int64(), nullable=False),
            pa.field("decay_due_at", pa.timestamp("us", tz="UTC"), nullable=True),
            pa.field("superseded_by", pa.utf8(), nullable=True),
            pa.field("supersession_chain", pa.utf8(), nullable=True),
            pa.field("promoted_from", pa.utf8(), nullable=True),
            pa.field("embedding_model", pa.utf8(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=True),
        ]
    )


def _old_decision_row(id_str: str, content: str, vec: list[float]) -> dict[str, Any]:
    """A raw, old-shape ``records`` row for an ``in_force`` Decision — the
    exact 23 keys ``_pre_f6274eb_schema`` declares.
    """
    return {
        "id": id_str,
        "type": "decision",
        "content": content,
        "scope": "global",
        "purpose": None,
        "audience_hint": None,
        "source": None,
        "validity": "valid",
        "reconciles": None,
        "recall_count": 0,
        "last_referenced_at": None,
        "created_at": EPOCH,
        "updated_at": EPOCH,
        "lifecycle": "in_force",
        "valence": None,
        "feedback_score": 0.0,
        "hot_rejection_count": 0,
        "decay_due_at": None,
        "superseded_by": None,
        "supersession_chain": None,
        "promoted_from": None,
        "embedding_model": "fake/nomic-embed-text",
        "vector": vec,
    }


async def _seed_pre_migration_store(
    tmp_path: pathlib.Path, name: str, rows: list[dict[str, Any]]
) -> pathlib.Path:
    """Create a ``records`` table at the pre-``f6274eb`` (23-field) schema
    directly — bypassing ``LanceDBAdapter`` entirely, since it always writes
    the CURRENT schema — insert ``rows`` the old-shape way, and hand back
    the store path ready for ``LanceDBAdapter.open()`` to migrate.
    """
    path = tmp_path / name
    old_schema = _pre_f6274eb_schema(_DIM)
    db = await lancedb.connect_async(str(path))
    old_table = await db.create_table("records", schema=old_schema)
    arrow_tbl = pa.table({k: [r[k] for r in rows] for k in rows[0]}, schema=old_schema)
    await old_table.add(arrow_tbl)
    physical_field_count = len(await old_table.schema())
    assert physical_field_count == 23, (
        "fixture drifted from the pre-f6274eb schema: expected 23 physical "
        f"fields, got {physical_field_count}"
    )
    del db, old_table  # close this raw connection before the adapter reopens it
    return path


# ---------------------------------------------------------------------------
# 1. Adapter-level: drives LanceDBAdapter.transition directly — the minimal,
#    deterministic reproduction of the underlying defect, independent of
#    the engine's dedup gate.
# ---------------------------------------------------------------------------


class TestSchemaMigratedSupersessionAdapterLevel:
    async def test_transition_supersede_succeeds_against_migrated_table(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiable on its own: against unmodified ``LanceDBAdapter.
        transition`` this raises ``lancedb.InvalidInput`` (Lance's
        ``StructArray`` field-count panic) — not the ``ValueError`` the CAS
        contract documents.
        """
        embedder = FakeEmbedder(dim=_DIM)
        predecessor_id = str(new_id())
        predecessor_content = "Predecessor decision, written pre-migration."

        path = await _seed_pre_migration_store(
            tmp_path,
            "adapter_level_store",
            [
                _old_decision_row(
                    predecessor_id,
                    predecessor_content,
                    embedder.peek(predecessor_content),
                )
            ],
        )

        adapter = await LanceDBAdapter.open(
            str(path), embedding_model=embedder.model, dim=_DIM
        )
        physical = await adapter._records.schema()
        assert len(physical) > 23, (
            "the open() migration must have widened the physical schema — "
            "this is the precondition the whole test exists to construct"
        )

        successor_id = new_id()
        successor_content = "Successor decision, written post-migration."
        successor = Decision(
            id=successor_id,
            type="decision",
            content=successor_content,
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        await adapter.upsert(successor, embedder.peek(successor_content))

        predecessor_hid = HypomnemaId(uuid.UUID(predecessor_id))
        flip = PredecessorFlip(
            id=predecessor_hid, reason="changed", note=None, edge_id=str(uuid.uuid4())
        )
        await adapter.replace(successor, None, [flip], at=EPOCH)

        predecessor_after = await adapter.get_by_id(predecessor_hid)
        assert isinstance(predecessor_after, Decision)
        assert predecessor_after.lifecycle == "superseded"
        assert predecessor_after.superseded_by == successor_id


# ---------------------------------------------------------------------------
# 2. Engine-level: the exact originally-reported shape — an unrelated
#    bounce candidate (X) plus a separate ``supersedes`` target (Y), both
#    written pre-migration, resolved via ``resolve={"action":
#    "confirm_distinct"}``.
# ---------------------------------------------------------------------------


class TestSchemaMigratedSupersessionViaEngine:
    async def test_confirm_distinct_plus_supersede_succeeds_against_migrated_store(
        self, tmp_path: pathlib.Path
    ) -> None:
        embedder = FakeEmbedder(dim=_DIM)
        x_id = str(new_id())
        y_id = str(new_id())
        x_content = "Policy: all new services use Python."
        y_content = "Policy: deploy on Tuesdays only."
        successor_content = "Policy: all new services use Python (revised wording)."

        successor_vec = embedder.peek(successor_content)
        y_vec = embedder.peek(y_content)
        assert successor_vec != y_vec, "fixture must not accidentally collide with Y"

        path = await _seed_pre_migration_store(
            tmp_path,
            "confirm_distinct_supersede_store",
            [
                # X's STORED vector is deliberately set to the successor's
                # own embedding — forces cosine=1.0, reproducing the
                # "unrelated bounce candidate" shape from the original
                # diagnosis exactly (X is inserted raw, so its stored
                # vector is fully independent of x_content's own hash).
                _old_decision_row(x_id, x_content, successor_vec),
                _old_decision_row(y_id, y_content, y_vec),
            ],
        )

        clock = TestClock(EPOCH)
        adapter = await LanceDBAdapter.open(
            str(path), embedding_model=embedder.model, dim=_DIM, clock=clock
        )
        physical = await adapter._records.schema()
        assert len(physical) > 23

        cfg = EngineConfig(candidate_k=1000, gray_floor=0.80, decision_gray_floor=0.70)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        x_hid = HypomnemaId(uuid.UUID(x_id))
        y_hid = HypomnemaId(uuid.UUID(y_id))

        bounced = await engine.save(
            successor_content, kind="decision", scope="global",
            supersedes=y_hid, reason="changed",
        )
        assert bounced.outcome == "bounced", bounced
        assert bounced.bounce is not None
        assert bounced.bounce.candidate_id == x_hid

        resolved = await engine.save(
            successor_content,
            kind="decision",
            scope="global",
            supersedes=y_hid,
            reason="changed",
            resolve=BounceResolution(candidate_id=x_hid, action="confirm_distinct"),
        )
        assert resolved.outcome in {"saved", "reconciled"}, resolved
        assert resolved.hypomnema is not None

        y_after = await adapter.get_by_id(y_hid)
        assert isinstance(y_after, Decision)
        assert y_after.lifecycle == "superseded"
        assert y_after.superseded_by == resolved.hypomnema.id

    async def test_plain_supersede_of_post_migration_decision_succeeds(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Dataset-wide poisoning check: a SINGLE legacy fragment anywhere
        in the table must not break supersession of a Decision created
        entirely AFTER the migration, living in its own fresh, fully-wide
        fragment — no bounce, no resolve, nothing to do with the legacy row.
        """
        embedder = FakeEmbedder(dim=_DIM)
        legacy_id = str(new_id())
        legacy_content = "Ancient pre-migration decision, unrelated to anything else."

        path = await _seed_pre_migration_store(
            tmp_path,
            "dataset_wide_store",
            [_old_decision_row(legacy_id, legacy_content, embedder.peek(legacy_content))],
        )

        clock = TestClock(EPOCH)
        adapter = await LanceDBAdapter.open(
            str(path), embedding_model=embedder.model, dim=_DIM, clock=clock
        )
        cfg = EngineConfig(candidate_k=1000, gray_floor=0.80, decision_gray_floor=0.70)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        predecessor_content = "Brand new decision, created post-migration."
        successor_content = "Successor decision replacing the brand new one."
        created = await engine.save(predecessor_content, kind="decision", scope="global")
        assert created.outcome == "saved", created
        assert created.hypomnema is not None
        predecessor_id = created.hypomnema.id

        result = await engine.save(
            successor_content, kind="decision", scope="global",
            supersedes=predecessor_id, reason="changed",
        )
        assert result.outcome == "saved", result
        assert result.hypomnema is not None

        predecessor_after = await adapter.get_by_id(predecessor_id)
        assert isinstance(predecessor_after, Decision)
        assert predecessor_after.lifecycle == "superseded"
        assert predecessor_after.superseded_by == result.hypomnema.id


# ---------------------------------------------------------------------------
# 3. A genuine CAS violation (real concurrency, not a monkeypatch) against a
#    migrated store must still roll back cleanly, and the loser's ToolError
#    must now carry the real cause rather than the removed generic guess
#    (Part 3 of this brief — engine/decisions.py's misleading wrapper).
# ---------------------------------------------------------------------------


class TestSchemaMigratedGenuineCasViolationRollsBackCleanly:
    async def test_concurrent_supersede_rolls_back_cleanly_against_migrated_store(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Mirrors ``test_engine_concurrency.py::TestConcurrentSupersession
        ::test_no_orphan_on_concurrent_supersede`` — which never exercises a
        migrated table — against a store that HAS been through the
        ``add_columns`` backfill, and additionally asserts the tombstone
        ``reason`` and the distinct-pair registry are untouched by the
        rollback (properties that test does not check).
        """
        embedder = FakeEmbedder(dim=_DIM)
        predecessor_id = str(new_id())
        predecessor_content = "Predecessor decision: use Python for the backend."
        succ_a_content = "Successor A: use Python 3.12 with strict typing."
        succ_b_content = "Successor B: switch the backend to Rust."

        path = await _seed_pre_migration_store(
            tmp_path,
            "genuine_cas_violation_store",
            [
                _old_decision_row(
                    predecessor_id,
                    predecessor_content,
                    embedder.peek(predecessor_content),
                )
            ],
        )

        clock = TestClock(EPOCH)
        adapter = await LanceDBAdapter.open(
            str(path), embedding_model=embedder.model, dim=_DIM, clock=clock
        )
        cfg = EngineConfig(candidate_k=1000, gray_floor=0.80, decision_gray_floor=0.70)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        predecessor_hid = HypomnemaId(uuid.UUID(predecessor_id))
        pairs_before = await raw_table(adapter._pairs).count_rows()
        tombstones_before = await raw_table(adapter._tombstones_tbl).count_rows()

        results = await asyncio.gather(
            engine.save(
                succ_a_content, kind="decision", scope="global",
                supersedes=predecessor_hid, reason="changed",
            ),
            engine.save(
                succ_b_content, kind="decision", scope="global",
                supersedes=predecessor_hid, reason="changed",
            ),
            return_exceptions=True,
        )

        successes = [r for r in results if isinstance(r, SaveResult)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1, results
        assert len(failures) == 1, results
        assert isinstance(failures[0], ToolError), failures[0]
        # Part 3 of this brief: the real cause now reaches the caller — not
        # the removed generic "may already be superseded" guess.
        assert "may already be superseded" not in str(failures[0]), (
            f"the misleading wrapper text must be gone: {failures[0]!s}"
        )

        winner = successes[0]
        assert winner.hypomnema is not None

        pairs_after = await raw_table(adapter._pairs).count_rows()
        tombstones_after = await raw_table(adapter._tombstones_tbl).count_rows()
        assert pairs_after == pairs_before, "a failed supersede must not register a distinct pair"
        # P3 one-transaction-supersession piece: the
        # Decision arm's bury-on-failure compensation is gone. The loser's
        # ENTIRE write (new Decision row + predecessor flip) is one
        # ``replace`` transaction — when it fails, nothing lands, so there
        # is no half-written Decision to bury.
        assert tombstones_after == tombstones_before, (
            "a failed replace leaves nothing to bury — the loser's Decision "
            "was never written at all"
        )

        predecessor_after = await adapter.get_by_id(predecessor_hid)
        assert isinstance(predecessor_after, Decision)
        assert predecessor_after.lifecycle == "superseded"
        assert predecessor_after.superseded_by == winner.hypomnema.id

        rows = [row async for row in await adapter.scan()]
        live_in_force = [r for r in rows if r.type == "decision" and r.lifecycle == "in_force"]
        assert len(live_in_force) == 1
        assert live_in_force[0].id == winner.hypomnema.id
        assert len(rows) == 2, (
            "predecessor (superseded) + winner (in_force) — loser must not appear"
        )


# ---------------------------------------------------------------------------
# 4. Direct falsifiable coverage for this requirement: the Python-side CAS lock
# (self._supersede_lock)
#    must protect against a concurrent direct bury()/revoke racing a direct
#    supersede, not only against two concurrent direct supersedes — the OLD
#    storage predicate protected against ANY writer invalidating the
#    precondition before commit, whichever operation caused it.
# ---------------------------------------------------------------------------


class TestSchemaMigratedSupersedeVersusOtherLifecycleWrites:
    async def test_direct_supersede_versus_direct_revoke_no_torn_state(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Mutually exclusive edges (SPEC §1.2 — both must originate at
        in_force): exactly one of {supersede, revoke} must win, the other
        must fail cleanly, and the final state must match whichever won —
        never a state matching neither (torn state). Direct adapter calls,
        bypassing the engine's lock entirely, against a migrated store.
        """
        embedder = FakeEmbedder(dim=_DIM)
        predecessor_id = str(new_id())
        predecessor_content = "Predecessor: supersede-vs-revoke race."
        successor_content = "Successor: supersede-vs-revoke race."

        path = await _seed_pre_migration_store(
            tmp_path,
            "supersede_vs_revoke_store",
            [
                _old_decision_row(
                    predecessor_id, predecessor_content, embedder.peek(predecessor_content)
                )
            ],
        )
        adapter = await LanceDBAdapter.open(str(path), embedding_model=embedder.model, dim=_DIM)

        predecessor_hid = HypomnemaId(uuid.UUID(predecessor_id))
        successor_id = new_id()
        successor = Decision(
            id=successor_id,
            type="decision",
            content=successor_content,
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        await adapter.upsert(successor, embedder.peek(successor_content))

        flip = PredecessorFlip(
            id=predecessor_hid, reason="changed", note=None, edge_id=str(uuid.uuid4())
        )
        results = await asyncio.gather(
            adapter.replace(successor, None, [flip], at=EPOCH),
            adapter.transition(predecessor_hid, "revoked", at=EPOCH),
            return_exceptions=True,
        )

        successes = [r for r in results if not isinstance(r, BaseException)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1, results
        assert len(failures) == 1, results
        assert isinstance(failures[0], (ValueError, KeyError)), failures[0]

        final = await adapter.get_by_id(predecessor_hid)
        assert isinstance(final, Decision)
        if final.lifecycle == "superseded":
            assert final.superseded_by == successor_id
        else:
            assert final.lifecycle == "revoked", (
                f"torn state: neither cleanly superseded nor cleanly revoked: {final!r}"
            )
            assert final.superseded_by is None

    async def test_supersede_after_bury_raises_valueerror_not_silent_success(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The exact failure mode a Decision
        buried strictly before a ``replace`` call's fresh CAS re-check must
        make that call raise — not silently report success with nothing
        actually written. ``replace`` raises ``ValueError`` uniformly, never
        ``KeyError`` (P3 one-transaction-supersession piece design choice —
        one exception type across every illegal-referent case, successor or
        predecessor alike). This is the deterministic construction (bury
        fully completes first, no interleaving needed);
        ``TestAdapterLevelSupersessionCAS``-style genuine concurrency for
        the mutually-exclusive supersede/revoke edges is covered by the test
        above — supersede-after-bury is deliberately NOT a race here because
        burying an already-superseded Decision is legitimate sequencing, not
        a torn state; what must never happen is superseding an
        already-buried one silently.
        """
        embedder = FakeEmbedder(dim=_DIM)
        predecessor_id = str(new_id())
        predecessor_content = "Predecessor: supersede-after-bury."
        successor_content = "Successor: supersede-after-bury."

        path = await _seed_pre_migration_store(
            tmp_path,
            "supersede_after_bury_store",
            [
                _old_decision_row(
                    predecessor_id, predecessor_content, embedder.peek(predecessor_content)
                )
            ],
        )
        adapter = await LanceDBAdapter.open(str(path), embedding_model=embedder.model, dim=_DIM)

        predecessor_hid = HypomnemaId(uuid.UUID(predecessor_id))
        successor_id = new_id()
        successor = Decision(
            id=successor_id,
            type="decision",
            content=successor_content,
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        await adapter.upsert(successor, embedder.peek(successor_content))

        await adapter.bury(predecessor_hid, reason="race test", model_generated=True)

        flip = PredecessorFlip(
            id=predecessor_hid, reason="changed", note=None, edge_id=str(uuid.uuid4())
        )
        with pytest.raises(ValueError):
            await adapter.replace(successor, None, [flip], at=EPOCH)

        # The successor must not have been silently pointed-to by anything;
        # it stays exactly as upserted (no corruption from the failed call).
        successor_after = await adapter.get_by_id(successor_id)
        assert isinstance(successor_after, Decision)
        assert successor_after.lifecycle == "in_force"
