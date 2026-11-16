"""P1 schema piece: the two startup population passes, run through
``Tabularium.initialize()`` and directly through the adapter. P4 two-arm
unsupersede piece adds the third pass, undo repair, to the SAME
``run_startup_passes`` call — its own pass-order and failure-propagation
tests live at the bottom of this file (the brief's own "pass orchestration"
reference names ``test_pass_orchestration.py``, which is the Cleaner's
unrelated pass machinery in ``engine/clean.py`` — a stale cross-reference;
this is the file that actually exercises ``engine.startup.
run_startup_passes``).

Falsifiability:
- ``test_edge_pass_resumes_without_reassigning``/``test_key_pass_resumes``:
  change the pass to assign a fresh value to every matching row regardless
  of whether it already carries one, and the resume assertion (first-filled
  value unchanged after the second, unwrapped run) goes red.

Contract-grounded reason these tests are LanceDB-only, not dual-
parameterized: chunking, a killed mid-chunk write, and a real reopen are
physical-fragment concepts ``MemoryAdapter`` has none of — its own coverage
is ``test_memory_adapter_passes_return_zero_after_build`` below plus the
dual-parameterized conformance tests in ``TestStartupPopulationPasses``.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, Scope, new_id
from tabularium.engine import Tabularium
from tabularium.storage import lancedb_adapter as lancedb_adapter_module
from tabularium.storage.content_key import content_key
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.records import RepairReport
from tests.support.doubles import FakeEmbedder, TestClock

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DIM = 8

_FIXTURE = Path(__file__).parent / "fixtures" / "legacy-store-294fa14"


async def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "legacy-store"
    shutil.copytree(_FIXTURE, dest)
    return dest


class _RaisingRecordsProxy:
    """Wraps a real LanceDB table handle; raises on the Nth ``merge_insert``
    call, simulating a process kill partway through a chunked write. Every
    other attribute (``query``, ``schema``, ...) passes through untouched."""

    def __init__(self, real: Any, *, raise_on_call: int) -> None:
        self._real = real
        self._raise_on_call = raise_on_call
        self._calls = 0

    def merge_insert(self, *args: Any, **kwargs: Any) -> Any:
        self._calls += 1
        if self._calls == self._raise_on_call:
            raise RuntimeError("simulated kill mid-pass")
        return self._real.merge_insert(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


async def test_initialize_fills_legacy_rows(tmp_path: Path) -> None:
    dest = await _copy_fixture(tmp_path)

    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(EPOCH)
    adapter = await LanceDBAdapter.open(
        str(dest), embedding_model=embedder.model, dim=_DIM, clock=clock
    )
    engine = Tabularium(adapter, embedder, clock, EngineConfig())

    pre_rows = await adapter._records.query().to_list()
    padded_before = {
        r["id"]: r["content"]
        for r in pre_rows
        if r["content"] != r["content"].strip() or "  " in r["content"]
    }
    assert len(padded_before) >= 2, "fixture must carry whitespace-padded rows"

    await engine.initialize()

    rows = await adapter._records.query().to_list()
    assert all(r["content_key"] is not None for r in rows)
    assert all(r["content_hash"] is not None for r in rows)
    for r in rows:
        if r["superseded_by"] is not None:
            assert r["supersession_edge_id"] is not None, r["id"]

    rows_by_id = {r["id"]: r for r in rows}
    for row_id, original_content in padded_before.items():
        row = rows_by_id[row_id]
        assert row["content"] == original_content
        assert row["content_key"] == content_key(original_content)
        assert row["content_key"] == row["content_key"].strip()
        assert "  " not in row["content_key"]

    await engine.initialize()
    assert await adapter.populate_content_keys() == 0
    assert await adapter.populate_edge_ids() == 0


async def test_edge_pass_resumes_without_reassigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = await _copy_fixture(tmp_path)
    monkeypatch.setattr(lancedb_adapter_module, "_STARTUP_CHUNK", 1)

    adapter = await LanceDBAdapter.open(str(dest), embedding_model="test-model", dim=_DIM)

    targets_before = await (
        adapter._records.query()
        .where("superseded_by IS NOT NULL AND supersession_edge_id IS NULL")
        .select(["id"])
        .to_list()
    )
    target_ids = {r["id"] for r in targets_before}
    assert len(target_ids) >= 2, "fixture must carry at least two edges to backfill"

    real_records = adapter._records
    adapter._records = _RaisingRecordsProxy(real_records, raise_on_call=2)

    with pytest.raises(RuntimeError):
        await adapter.populate_edge_ids()

    adapter._records = real_records
    mid_after = await adapter._records.query().to_list()
    mid_rows = {r["id"]: r["supersession_edge_id"] for r in mid_after}
    filled_after_crash = {
        rid: edge_id for rid, edge_id in mid_rows.items() if rid in target_ids and edge_id
    }
    assert len(filled_after_crash) == 1
    first_id, first_edge_id = next(iter(filled_after_crash.items()))

    reopened = await LanceDBAdapter.open(str(dest), embedding_model="test-model", dim=_DIM)
    written = await reopened.populate_edge_ids()
    assert written == len(target_ids) - 1

    final_after = await reopened._records.query().to_list()
    final_rows = {r["id"]: r["supersession_edge_id"] for r in final_after}
    for rid in target_ids:
        assert final_rows[rid] is not None, rid
    assert final_rows[first_id] == first_edge_id


async def test_key_pass_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = await _copy_fixture(tmp_path)
    monkeypatch.setattr(lancedb_adapter_module, "_STARTUP_CHUNK", 1)

    adapter = await LanceDBAdapter.open(str(dest), embedding_model="test-model", dim=_DIM)

    targets_before = await (
        adapter._records.query().where("content_key IS NULL").select(["id"]).to_list()
    )
    target_ids = {r["id"] for r in targets_before}
    assert len(target_ids) >= 2

    real_records = adapter._records
    adapter._records = _RaisingRecordsProxy(real_records, raise_on_call=2)

    with pytest.raises(RuntimeError):
        await adapter.populate_content_keys()

    adapter._records = real_records
    mid_after = await adapter._records.query().to_list()
    mid_rows = {r["id"]: r["content_key"] for r in mid_after}
    filled_after_crash = {
        rid: key for rid, key in mid_rows.items() if rid in target_ids and key
    }
    assert len(filled_after_crash) == 1
    first_id, first_key = next(iter(filled_after_crash.items()))

    reopened = await LanceDBAdapter.open(str(dest), embedding_model="test-model", dim=_DIM)
    written = await reopened.populate_content_keys()
    assert written == len(target_ids) - 1

    final_after = await reopened._records.query().to_list()
    final_rows = {r["id"]: r["content_key"] for r in final_after}
    for rid in target_ids:
        assert final_rows[rid] is not None, rid
    assert final_rows[first_id] == first_key


async def test_memory_adapter_passes_return_zero_after_build() -> None:
    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=_DIM, clock=clock)
    engine = Tabularium(adapter, embedder, clock, EngineConfig())
    await engine.initialize()

    result = await engine.save("A fact built through production save.", scope="global")
    assert result.outcome == "saved"

    assert await adapter.populate_content_keys() == 0
    assert await adapter.populate_edge_ids() == 0

    predecessor = Fact(
        id=new_id(),
        type="fact",
        content="predecessor, superseded by hand, never given an edge id",
        scope=Scope("global"),
        lifecycle="active",
        validity="invalid",
        created_at=EPOCH,
        updated_at=EPOCH,
        superseded_by=new_id(),
    )
    await adapter.upsert(predecessor, [1.0] + [0.0] * (_DIM - 1))

    written = await adapter.populate_edge_ids()
    assert written == 1

    filled = await adapter.get_by_id(predecessor.id)
    assert isinstance(filled, Fact)
    assert filled.supersession_edge_id is not None

    assert await adapter.populate_edge_ids() == 0


async def test_pass_order_is_content_key_then_edge_id_then_undo_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three passes run in a fixed order: content key, edge id, undo
    repair.

    Falsify: swap the ``populate_edge_ids``/``repair_unsupersedes`` calls in
    ``run_startup_passes`` — this test's ``order`` assertion goes red.
    """
    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=_DIM, clock=clock)
    engine = Tabularium(adapter, embedder, clock, EngineConfig())

    order: list[str] = []
    real_content_keys = adapter.populate_content_keys
    real_edge_ids = adapter.populate_edge_ids
    real_repair = adapter.repair_unsupersedes

    async def _tracked_content_keys() -> int:
        order.append("content_key")
        return await real_content_keys()

    async def _tracked_edge_ids() -> int:
        order.append("edge_id")
        return await real_edge_ids()

    async def _tracked_repair() -> RepairReport:
        order.append("undo_repair")
        return await real_repair()

    monkeypatch.setattr(adapter, "populate_content_keys", _tracked_content_keys)
    monkeypatch.setattr(adapter, "populate_edge_ids", _tracked_edge_ids)
    monkeypatch.setattr(adapter, "repair_unsupersedes", _tracked_repair)

    await engine.initialize()

    assert order == ["content_key", "edge_id", "undo_repair"]


async def test_backfill_scheduled_after_undo_repair_never_touches_its_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P6 vector-less-rows piece x P4 two-arm unsupersede piece: the
    backfill sweep is scheduled from ``initialize()`` only after
    ``run_startup_passes`` (content key, edge id, undo repair) has already
    returned and released the write lock.

    ``Backfiller.schedule`` is hooked directly rather than the eventual
    ``run_pass`` call it schedules: ``Backfiller.schedule`` is a plain
    (non-async) method ``initialize()`` calls synchronously in its own
    program order, so recording the moment IT runs is a deterministic
    causality check, immune to asyncio's own task-scheduling — unlike
    recording when the scheduled task's body actually executes, which
    races the caller and is not a reliable falsifier.

    The row-safety half is checked behaviourally: the undo's successor
    (vector-less, P6) is deleted by repair, and after the background pass
    settles (``pending_backfill``) it must never reappear as a backfill
    target — proof the deleted row was never a candidate the pass raced
    against, independent of scheduling.

    Falsify: call ``Backfiller.schedule`` before ``run_startup_passes`` in
    ``initialize()`` — ``order``'s last entry would move ahead of
    ``"undo_repair"``.
    """
    from tests.conformance.test_conformance import _decision, _fact, _flip, _unsupersede_event

    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=_DIM, clock=clock)
    engine = Tabularium(adapter, embedder, clock, EngineConfig())

    predecessor = _decision(content="Decision, undo crashed before the records transaction")
    successor = _decision(content="successor Decision, vector-less (P6 outage landing)")
    await adapter.upsert(predecessor, [1.0] + [0.0] * (_DIM - 1))
    flip = _flip(predecessor.id)
    await adapter.replace(successor, None, [flip], at=EPOCH)

    event = _unsupersede_event(
        edge_id=flip.edge_id,
        predecessor_id=predecessor.id,
        successor_id=successor.id,
        arm="decision",
    )
    await adapter.append_unsupersede(event)

    unrelated = _fact(content="unrelated vector-less row, never part of any undo")
    await adapter.upsert(unrelated, None)

    order: list[str] = []
    real_content_keys = adapter.populate_content_keys
    real_edge_ids = adapter.populate_edge_ids
    real_repair = adapter.repair_unsupersedes
    real_schedule_backfill = engine._backfiller.schedule

    async def _tracked_content_keys() -> int:
        order.append("content_key")
        return await real_content_keys()

    async def _tracked_edge_ids() -> int:
        order.append("edge_id")
        return await real_edge_ids()

    async def _tracked_repair() -> RepairReport:
        order.append("undo_repair")
        return await real_repair()

    def _tracked_schedule_backfill(*, trigger: str) -> None:
        order.append("backfill_scheduled")
        return real_schedule_backfill(trigger=trigger)

    monkeypatch.setattr(adapter, "populate_content_keys", _tracked_content_keys)
    monkeypatch.setattr(adapter, "populate_edge_ids", _tracked_edge_ids)
    monkeypatch.setattr(adapter, "repair_unsupersedes", _tracked_repair)
    monkeypatch.setattr(engine._backfiller, "schedule", _tracked_schedule_backfill)

    await engine.initialize()
    await engine.pending_backfill

    assert order == ["content_key", "edge_id", "undo_repair", "backfill_scheduled"]

    restored = await adapter.get_by_id(predecessor.id)
    assert isinstance(restored, type(predecessor))
    assert restored.lifecycle == "in_force"
    assert restored.superseded_by is None
    with pytest.raises(KeyError):
        await adapter.get_by_id(successor.id)

    targets_after = {t.id for t in await adapter.backfill_targets()}
    successor_targeted = successor.id in targets_after
    assert not successor_targeted, (
        "backfill must never carry a target scanned before repair deleted "
        "the row it belonged to"
    )
    assert unrelated.id not in targets_after, "the unrelated vector-less row must be filled"

    filled = await adapter.get_by_id(unrelated.id)
    assert filled.embedding_model == embedder.model


async def test_mid_repair_failure_surfaces_from_initialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure inside ``repair_unsupersedes`` propagates out of
    ``Tabularium.initialize()`` rather than being swallowed — the SAME
    "no behaviour hides a failure" discipline every other startup pass
    already carries (a ``RuntimeError`` from ``populate_edge_ids`` already
    propagates the identical way, per ``test_edge_pass_resumes_without_
    reassigning`` above).

    Falsify: wrap the ``repair_unsupersedes`` call in ``run_startup_passes``
    in a bare ``try/except Exception: pass`` — this test's
    ``pytest.raises(RuntimeError)`` goes red as ``initialize()`` returns
    cleanly instead.
    """
    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=_DIM, clock=clock)
    engine = Tabularium(adapter, embedder, clock, EngineConfig())

    async def _raise_repair_failure() -> tuple[int, int, int]:
        raise RuntimeError("simulated repair failure")

    monkeypatch.setattr(adapter, "repair_unsupersedes", _raise_repair_failure)

    with pytest.raises(RuntimeError, match="simulated repair failure"):
        await engine.initialize()
