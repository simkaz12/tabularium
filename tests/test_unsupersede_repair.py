"""Startup repair for interrupted undos, on the real LanceDB adapter (P4
two-arm unsupersede piece — pointer-guarded, not
marker-alone; — an adapter-protocol pass run from
``Tabularium.initialize()``, after ``populate_edge_ids``).

"Restart" means: close (drop the in-process handle), reopen the SAME
on-disk store fresh via ``open_adapter``, construct a NEW engine over it,
and call ``await initialize()`` — the real end-to-end wiring, not a direct
adapter call, for the cases that only need to prove STATE after repair
(events 1/2/3, legacy tombstone). The cases that need the exact
``(completed, skipped, refused)`` COUNT call ``adapter.repair_unsupersedes()``
directly against the freshly reopened adapter — ``run_startup_passes``
only logs the counts, it does not hand them back to ``initialize()``'s
caller, and log-scraping would be a more brittle assertion than calling the
same pass this whole file already proves is wired into ``initialize()``
(``test_pass_order_is_content_key_then_edge_id_then_undo_repair`` in
``tests/test_startup_passes.py``).

Faults are injected through ``FaultInjectingAdapter`` — the same tool
``tests/test_fact_supersession.py`` uses — wrapping a real
``LanceDBAdapter`` so the interrupted write genuinely lands (or doesn't) on
disk, not a stub.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, HypomnemaId, Scope
from tabularium.engine import Tabularium, ToolError
from tabularium.storage import open_adapter
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import UnsupersedeEvent
from tests.support.doubles import FakeEmbedder, TestClock
from tests.support.edits import edit_content
from tests.support.faults import FaultInjectingAdapter, as_adapter
from tests.support.lance_raw import proxy_table

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DIM = 64
_FIXTURE = Path(__file__).parent / "fixtures" / "legacy-store-294fa14"
_FIXTURE_DIM = 8
_LEGACY_PREDECESSOR_ID = HypomnemaId(uuid.UUID("7579639e-9a69-47a2-86b5-8f74e3c0b310"))
_LEGACY_SUCCESSOR_ID = HypomnemaId(uuid.UUID("bab7204d-0142-46ee-8e63-bb7a82c46906"))


def _make_clock() -> TestClock:
    return TestClock(EPOCH)


def _make_config() -> EngineConfig:
    return EngineConfig(candidate_k=1000, gray_floor=0.80)


async def _reopen(store_path: Path) -> LanceDBAdapter:
    adapter = await open_adapter(
        "lancedb", embedding_model="test-model", dim=_DIM, store_path=str(store_path),
    )
    assert isinstance(adapter, LanceDBAdapter)
    return adapter


async def _restart(store_path: Path) -> tuple[Tabularium, LanceDBAdapter]:
    adapter = await _reopen(store_path)
    embedder = FakeEmbedder(dim=_DIM)
    clock = _make_clock()
    engine = Tabularium(adapter, embedder, clock, _make_config())
    await engine.initialize()
    return engine, adapter


class TestBurialProtectionSurvivesRestart:
    """``_load_buried_ids``
    used to exclude every ``restores``-marked tombstone from
    ``_buried_ids`` on reopen — including a COMPLETED undo's, whose buried
    row is already gone. A restart then let the successor id be reused —
    ``upsert`` refused it before the restart and silently accepted it
    after.

    Falsify: drop the ``pending_restores_ids - present_ids`` rejoin at the
    end of ``_load_buried_ids`` — this test's ``pytest.raises`` around the
    post-restart ``upsert`` goes unfired.
    """

    async def test_upsert_of_the_completed_undos_buried_id_still_refuses(
        self, tmp_path: Path
    ) -> None:
        store_path = tmp_path / "store"
        adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id

        result = await engine.unsupersede(a_id, reason="wrong successor")
        assert result.shape == "decision_restored"
        with pytest.raises(KeyError):
            await adapter.get_by_id(b_id)

        reused = Decision(
            id=b_id, type="decision", content="B reincarnated",
            scope=Scope("global"), lifecycle="in_force", validity="valid",
            created_at=EPOCH, updated_at=EPOCH,
        )
        with pytest.raises(ValueError, match="Cannot upsert a buried id"):
            await adapter.upsert(reused, [0.1] * _DIM)

        adapter2 = await _reopen(store_path)
        with pytest.raises(ValueError, match="Cannot upsert a buried id"):
            await adapter2.upsert(reused, [0.1] * _DIM)

    async def test_pending_undo_stays_live_and_repair_still_finishes_it(
        self, tmp_path: Path
    ) -> None:
        """The other half of that fix: a tombstone whose buried row
        is STILL PRESENT (an undo interrupted before ``restore`` ran) must
        stay OUT of ``_buried_ids`` after a restart, and repair must still
        be able to finish it — the fix must not turn a pending undo into a
        premature permanent burial either."""
        store_path = tmp_path / "store"
        raw_adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        wrapper = FaultInjectingAdapter(raw_adapter)
        engine = Tabularium(as_adapter(wrapper), embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id

        wrapper.fault(on="restore", call=1, exc=RuntimeError("simulated kill"))
        wrapper.arm()
        with pytest.raises(ToolError):
            await engine.unsupersede(a_id, reason="wrong edge")

        adapter2 = await _reopen(store_path)
        still_present = await adapter2.get_by_id(b_id)
        assert isinstance(still_present, Decision)

        report = await adapter2.repair_unsupersedes()
        completed = report.completed
        assert completed == 1
        with pytest.raises(KeyError):
            await adapter2.get_by_id(b_id)


class TestEventOnlyKillDecision:
    async def test_repair_finishes_the_undo(self, tmp_path: Path) -> None:
        store_path = tmp_path / "store"
        raw_adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        wrapper = FaultInjectingAdapter(raw_adapter)
        engine = Tabularium(as_adapter(wrapper), embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id

        wrapper.fault(on="plant_tombstone", call=1, exc=RuntimeError("simulated kill"))
        wrapper.arm()
        with pytest.raises(ToolError):
            await engine.unsupersede(a_id, reason="wrong edge")

        _engine2, adapter2 = await _restart(store_path)

        restored = await adapter2.get_by_id(a_id)
        assert isinstance(restored, Decision)
        assert restored.lifecycle == "in_force"
        assert restored.superseded_by is None
        with pytest.raises(KeyError):
            await adapter2.get_by_id(b_id)
        tombstone = await adapter2.get_tombstone(b_id)
        assert tombstone is not None
        assert tombstone.restores == a_id
        events = await adapter2.list_unsupersedes()
        assert len(events) == 1


class TestEventPlusTombstoneKillDecision:
    async def test_repair_finishes_the_undo(self, tmp_path: Path) -> None:
        store_path = tmp_path / "store"
        raw_adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        wrapper = FaultInjectingAdapter(raw_adapter)
        engine = Tabularium(as_adapter(wrapper), embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id

        wrapper.fault(on="restore", call=1, exc=RuntimeError("simulated kill"))
        wrapper.arm()
        with pytest.raises(ToolError):
            await engine.unsupersede(a_id, reason="wrong edge")

        _engine2, adapter2 = await _restart(store_path)

        restored = await adapter2.get_by_id(a_id)
        assert isinstance(restored, Decision)
        assert restored.lifecycle == "in_force"
        assert restored.superseded_by is None
        with pytest.raises(KeyError):
            await adapter2.get_by_id(b_id)
        tombstone = await adapter2.get_tombstone(b_id)
        assert tombstone is not None
        assert tombstone.restores == a_id
        events = await adapter2.list_unsupersedes()
        assert len(events) == 1


class TestEventOnlyKillFact:
    async def test_repair_finishes_the_undo(self, tmp_path: Path) -> None:
        store_path = tmp_path / "store"
        raw_adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        wrapper = FaultInjectingAdapter(raw_adapter)
        engine = Tabularium(as_adapter(wrapper), embedder, clock, _make_config())
        await engine.initialize()

        pred = await engine.save("predecessor Fact", scope="global")
        succ = await engine.save("successor Fact", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        pred_id = pred.hypomnema.id
        succ_id = succ.hypomnema.id
        await engine.supersede(succ_id, [pred_id], "changed")

        wrapper.fault(on="restore", call=1, exc=RuntimeError("simulated kill"))
        wrapper.arm()
        with pytest.raises(ToolError):
            await engine.unsupersede(pred_id, reason="wrong edge")

        _engine2, adapter2 = await _restart(store_path)

        restored = await adapter2.get_by_id(pred_id)
        assert isinstance(restored, Fact)
        assert restored.validity == "valid"
        assert restored.superseded_by is None
        succ_after = await adapter2.get_by_id(succ_id)
        assert isinstance(succ_after, Fact)


class TestCompletedUndoThenNewSupersession:
    """A→B, undo, A→C: repair must not re-restore A (a prior decision's
    pointer guard) — B's tombstone still says ``restores=A`` (history), but
    A now legitimately points at C."""

    async def test_decision_arm(self, tmp_path: Path) -> None:
        store_path = tmp_path / "store"
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id
        first_edge_id = (await adapter.get_by_id(a_id)).supersession_edge_id
        assert first_edge_id is not None
        await engine.unsupersede(a_id, reason="wrong successor")
        c = await engine.save(
            "Decision C", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert c.hypomnema is not None

        fresh_adapter = await _reopen(store_path)
        report = await fresh_adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped >= 1
        assert refused == 0

        final_a = await fresh_adapter.get_by_id(a_id)
        assert isinstance(final_a, Decision)
        assert final_a.superseded_by == c.hypomnema.id
        assert final_a.supersession_edge_id is not None
        assert final_a.supersession_edge_id != first_edge_id
        tombstone = await fresh_adapter.get_tombstone(b_id)
        assert tombstone is not None
        assert tombstone.restores == a_id

    async def test_fact_arm(self, tmp_path: Path) -> None:
        store_path = tmp_path / "store"
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        pred = await engine.save("predecessor Fact", scope="global")
        b = await engine.save("successor B Fact", scope="global")
        c = await engine.save("successor C Fact", scope="global")
        assert pred.hypomnema is not None
        assert b.hypomnema is not None
        assert c.hypomnema is not None
        pred_id = pred.hypomnema.id

        await engine.supersede(b.hypomnema.id, [pred_id], "changed")
        await engine.unsupersede(pred_id, reason="wrong successor")
        await engine.supersede(c.hypomnema.id, [pred_id], "changed")

        fresh_adapter = await _reopen(store_path)
        report = await fresh_adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped >= 1
        assert refused == 0

        final_pred = await fresh_adapter.get_by_id(pred_id)
        assert isinstance(final_pred, Fact)
        assert final_pred.superseded_by == c.hypomnema.id


class TestPointerGuardIsolated:
    """Neither
    ``TestCompletedUndoThenNewSupersession`` case above actually reaches
    the pointer guard (``predecessor.superseded_by != event.successor_id``)
    in ``resolve_unsupersede_repair`` — a real re-supersession always mints
    a FRESH edge id, so the earlier edge-id guard already skips first. This
    test builds the mismatch directly through the adapter: the SAME edge id
    is kept, only ``superseded_by`` differs from what the event names, so
    the edge-id guard passes and only the pointer guard can produce the
    skip.

    Falsify: comment out the ``predecessor.superseded_by != event.
    successor_id`` check in ``resolve_unsupersede_repair`` — this test's
    ``skipped == 0`` assertion flips (repair wrongly restores A and, on the
    Decision arm, deletes whatever row the event's crafted successor id
    happens to name).
    """

    async def test_repair_skips_on_pointer_mismatch_with_matching_edge_id(
        self, tmp_path: Path
    ) -> None:
        store_path = tmp_path / "store"
        adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id
        real_edge_id = (await adapter.get_by_id(a_id)).supersession_edge_id
        assert real_edge_id is not None

        # A real, live, in_force Decision, wholly unrelated to A/B — if the
        # pointer guard did not fire, repair would fall all the way through
        # to "plant_then_restore" against THIS row (it is found, Decision-
        # typed, in_force, with no tombstone yet), not merely stop at some
        # earlier, unrelated guard. A bare nonexistent id here would let a
        # LATER guard (``successor is None``) mask the pointer guard the
        # same way the edge-id guard masks it elsewhere.
        c = await engine.save("Decision C, unrelated, still in_force", scope="global",
                               kind="decision")
        assert c.hypomnema is not None
        c_id = c.hypomnema.id

        # CRAFTED: a hand-written event sharing A's REAL edge id but naming
        # C as the successor — the edge-id guard alone cannot tell this
        # apart from a legitimate retry of A's own undo.
        event = UnsupersedeEvent(
            id=uuid.uuid4(), edge_id=real_edge_id, predecessor_id=a_id,
            successor_id=c_id, arm="decision",
            withdrawn_reason="correction", withdrawn_note=None,
            reason="pointer guard isolation", at=EPOCH,
        )
        await adapter.append_unsupersede(event)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

        unchanged_a = await adapter.get_by_id(a_id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"
        assert unchanged_a.superseded_by == b_id
        assert unchanged_a.supersession_edge_id == real_edge_id
        still_present_b = await adapter.get_by_id(b_id)
        assert isinstance(still_present_b, Decision)
        unchanged_c = await adapter.get_by_id(c_id)
        assert isinstance(unchanged_c, Decision)
        assert unchanged_c.lifecycle == "in_force"
        assert await adapter.get_tombstone(c_id) is None
        still_present_b = await adapter.get_by_id(b_id)
        assert isinstance(still_present_b, Decision)


class TestReusedSuccessorFactArm:
    """A→B, undo, A→B drawn again on purpose (a NEW edge id) — repair must
    leave the second edge alone, since its edge id differs from the undone
    one. Fact-arm only: the Decision arm's undo DELETES
    the successor, so a Decision successor can never legitimately be
    reused — ``TestCompletedUndoThenNewSupersession.test_decision_arm``
    above is that arm's own generation-safety proof, against a fresh
    successor rather than a reused one."""

    async def test_repair_leaves_the_second_edge_alone(self, tmp_path: Path) -> None:
        store_path = tmp_path / "store"
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        pred = await engine.save("predecessor, superseded by the SAME successor twice",
                                  scope="global")
        succ = await engine.save("successor, reused after an undo", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        pred_id = pred.hypomnema.id
        succ_id = succ.hypomnema.id

        await engine.supersede(succ_id, [pred_id], "changed")
        await engine.unsupersede(pred_id, reason="wrong the first time")
        await engine.supersede(succ_id, [pred_id], "changed")

        fresh_adapter = await _reopen(store_path)
        report = await fresh_adapter.repair_unsupersedes()
        completed = report.completed
        assert completed == 0

        final = await fresh_adapter.get_by_id(pred_id)
        assert isinstance(final, Fact)
        assert final.superseded_by == succ_id
        assert final.validity == "invalid"


class TestLegacyTombstone:
    async def test_bury_through_the_bury_tool_restores_nothing(
        self, tmp_path: Path
    ) -> None:
        """The original
        version of this test buried a Decision with NO supersession edge
        at all, so ``list_unsupersedes()`` was empty and repair's loop
        body never ran — it proved nothing about the legacy-tombstone
        guard. This version gives the buried Decision a REAL A→B edge and
        a real undo event first, so repair actually visits it; the
        legacy tombstone (``restores=None``, from the ordinary ``bury``
        tool) must still never be treated as authorisation, and neither
        row may move.

        Falsify: comment out the ``tombstone.restores is None`` guard in
        ``resolve_unsupersede_repair`` — this test's ``skipped == 0``
        assertion flips.
        """
        store_path = tmp_path / "store"
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A, undo interrupted before landing", scope="global",
                               kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        d = await engine.save("Decision B, buried outright during the gap", scope="global",
                               kind="decision", supersedes=a_id, reason="changed")
        assert d.hypomnema is not None
        d_id = d.hypomnema.id
        edge_id = (await adapter.get_by_id(a_id)).supersession_edge_id
        assert edge_id is not None

        event = UnsupersedeEvent(
            id=uuid.uuid4(), edge_id=edge_id, predecessor_id=a_id, successor_id=d_id,
            arm="decision", withdrawn_reason="changed", withdrawn_note=None,
            reason="wrong successor", at=EPOCH,
        )
        await adapter.append_unsupersede(event)
        await engine.bury(d_id, reason="ordinary burial, unrelated to the pending undo")

        tombstone = await adapter.get_tombstone(d_id)
        assert tombstone is not None
        assert tombstone.restores is None

        _engine2, adapter2 = await _restart(store_path)
        with pytest.raises(KeyError):
            await adapter2.get_by_id(d_id)
        report = await adapter2.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

        unchanged_a = await adapter2.get_by_id(a_id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"
        assert unchanged_a.superseded_by == d_id


class TestTwiceRestartAfterACompletedRepair:
    async def test_second_repair_reports_completed_zero(self, tmp_path: Path) -> None:
        """The original
        version of this test reopened the store ONCE, then called
        ``repair_unsupersedes()`` twice on the SAME live adapter instance —
        it never exercised a second ``open()`` / ``_load_buried_ids()``
        pass at all. This version reopens TWICE, one repair call per
        reopen, so a regression in cache reconstruction across a second
        restart (not just a second in-process call) would be caught.
        """
        store_path = tmp_path / "store"
        raw_adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        wrapper = FaultInjectingAdapter(raw_adapter)
        engine = Tabularium(as_adapter(wrapper), embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None

        wrapper.fault(on="restore", call=1, exc=RuntimeError("simulated kill"))
        wrapper.arm()
        with pytest.raises(ToolError):
            await engine.unsupersede(a_id, reason="wrong edge")

        first_adapter = await _reopen(store_path)
        first_report = await first_adapter.repair_unsupersedes()
        assert first_report.completed == 1

        second_adapter = await _reopen(store_path)
        second_report = await second_adapter.repair_unsupersedes()
        assert second_report.completed == 0


class TestUndoOfALegacyEdgeWithNoReasonClass:
    """A
    supersession edge drawn before the reason-class column existed
    (pre-P3) carries ``supersession_reason=NULL`` even after the startup
    edge-id pass gives it an edge id. ``unsupersede`` must still be able to
    undo it — the withdrawn class is honestly unknown, never fabricated,
    and never blocks the undo.

    Falsify: reintroduce ``assert withdrawn_reason is not None`` in
    ``engine.dispositions.unsupersede`` — this test's call raises
    ``AssertionError`` instead of completing.
    """

    async def test_unsupersede_completes_on_the_legacy_fixture(
        self, tmp_path: Path
    ) -> None:
        dest = tmp_path / "legacy-store"
        shutil.copytree(_FIXTURE, dest)

        adapter = await LanceDBAdapter.open(
            str(dest), embedding_model="test-model", dim=_FIXTURE_DIM
        )
        embedder = FakeEmbedder(dim=_FIXTURE_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        legacy_predecessor = await adapter.get_by_id(_LEGACY_PREDECESSOR_ID)
        assert isinstance(legacy_predecessor, Decision)
        assert legacy_predecessor.superseded_by == _LEGACY_SUCCESSOR_ID
        assert legacy_predecessor.supersession_reason is None
        assert legacy_predecessor.supersession_edge_id is not None

        result = await engine.unsupersede(
            _LEGACY_PREDECESSOR_ID, reason="a pre-P3 edge, undone anyway"
        )
        assert result.shape == "decision_restored"
        assert result.withdrawn_reason is None

        restored = await adapter.get_by_id(_LEGACY_PREDECESSOR_ID)
        assert isinstance(restored, Decision)
        assert restored.lifecycle == "in_force"
        assert restored.superseded_by is None
        with pytest.raises(KeyError):
            await adapter.get_by_id(_LEGACY_SUCCESSOR_ID)


class _RaisingDeleteProxy:
    """Wraps a real LanceDB table handle; raises on ``delete()`` — every
    other attribute (``query``, ``add``, ``schema``, ...) passes through
    untouched. Simulates ``bury()``'s tombstone-write landing durably while
    its own record-delete half fails (a caught error, not a crash)."""

    def __init__(self, real: object) -> None:
        self._real = real

    async def delete(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated delete failure mid-bury")

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


class TestPlantTombstoneAgainstAPartiallyFailedBury:
    """``bury()``
    writes its tombstone BEFORE deleting the record, and only updates
    ``_tombstone_cache``/``_buried_ids`` after BOTH writes succeed. A
    caught failure on the delete half leaves a durable tombstone the cache
    does not yet know about — ``get_tombstone()`` (cache-only) answers
    ``None`` even though a real tombstone row already exists. Without a
    ground-truth check, ``plant_tombstone`` would happily plant a SECOND,
    conflicting tombstone for the same id.
    """

    async def test_plant_tombstone_refuses_against_a_durable_but_uncached_tombstone(
        self, tmp_path: Path
    ) -> None:
        store_path = tmp_path / "store"
        adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        b = await engine.save(
            "Decision B, bury interrupted mid-write", scope="global", kind="decision",
            supersedes=a.hypomnema.id, reason="changed",  # type: ignore[union-attr]
        )
        assert a.hypomnema is not None
        assert b.hypomnema is not None
        b_id = b.hypomnema.id

        real_records = adapter._records
        adapter._records = proxy_table(_RaisingDeleteProxy(real_records))
        with pytest.raises(RuntimeError):
            await engine.bury(b_id, reason="ordinary burial, interrupted mid-write")
        adapter._records = real_records

        still_present = await adapter.get_by_id(b_id)
        assert isinstance(still_present, Decision)
        assert await adapter.get_tombstone(b_id) is None
        assert await adapter._tombstone_from_table(b_id) is not None

        with pytest.raises(KeyError):
            await adapter.plant_tombstone(b_id, "unsupersede: x", restores=a.hypomnema.id)

    async def test_repair_skips_gracefully_against_the_same_durable_but_uncached_tombstone(
        self, tmp_path: Path
    ) -> None:
        store_path = tmp_path / "store"
        adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        b = await engine.save(
            "Decision B, bury interrupted mid-write", scope="global", kind="decision",
            supersedes=a.hypomnema.id, reason="changed",  # type: ignore[union-attr]
        )
        assert a.hypomnema is not None
        assert b.hypomnema is not None
        a_id = a.hypomnema.id
        b_id = b.hypomnema.id

        real_records = adapter._records
        adapter._records = proxy_table(_RaisingDeleteProxy(real_records))
        with pytest.raises(RuntimeError):
            await engine.bury(b_id, reason="ordinary burial, interrupted mid-write")
        adapter._records = real_records

        edge_id = (await adapter.get_by_id(a_id)).supersession_edge_id
        assert edge_id is not None

        event = UnsupersedeEvent(
            id=uuid.uuid4(),
            edge_id=edge_id,
            predecessor_id=a_id,
            successor_id=b_id,
            arm="decision",
            withdrawn_reason="changed",
            withdrawn_note=None,
            reason="racing the interrupted bury",
            at=EPOCH,
        )
        await adapter.append_unsupersede(event)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

        unchanged_a = await adapter.get_by_id(a_id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.superseded_by == b_id


class TestEditDuringPendingUndoIsRefusedByRepair:
    """A pending Decision undo plants B's tombstone (``restores=
    A``), then ``restore`` fails (caught, not a crash — the server stays
    up). Before this fix, a legitimate content edit of B landing in
    that window was silently discarded on the next restart: repair saw the
    edge id and pointer still matching, restored A, and deleted B —
    erasing the edit with no trace beyond the tombstone's stale ``restores``.

    The remedy (owner call, P4 verdict): the edit wins. ``plant_tombstone``
    now captures the successor's ``content_hash`` at plant time;
    ``resolve_unsupersede_repair`` compares it against the successor's
    CURRENT content before ever restoring, and refuses — withdrawing the
    stale tombstone rather than completing the destructive transaction —
    on any mismatch.

    Falsify: this class's own module-level fixture-free run against a
    build with the ``tombstone.restores_content_hash != content_hash(
    successor.content)`` compare removed from ``resolve_unsupersede_repair``
    goes red — see the verdict for the exact command and the pre-fix
    failing output this test produced against the tracer repro.
    """

    async def test_repair_refuses_and_keeps_the_edit(self, tmp_path: Path) -> None:
        store_path = tmp_path / "store"
        raw_adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        wrapper = FaultInjectingAdapter(raw_adapter)
        engine = Tabularium(as_adapter(wrapper), embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id

        wrapper.fault(on="restore", call=1, exc=RuntimeError("simulated failure"))
        wrapper.arm()
        with pytest.raises(ToolError):
            await engine.unsupersede(a_id, reason="wrong successor")

        b_before = await raw_adapter.get_by_id(b_id)
        assert isinstance(b_before, Decision)
        assert b_before.lifecycle == "in_force"
        tombstone_before = await raw_adapter.get_tombstone(b_id)
        assert tombstone_before is not None
        assert tombstone_before.restores == a_id

        edited_content = "Decision B — corrected wording"
        await edit_content(
            raw_adapter, b_id, edited_content, await embedder.embed(edited_content)
        )
        b_after_edit = await raw_adapter.get_by_id(b_id)
        assert isinstance(b_after_edit, Decision)
        assert b_after_edit.content == "Decision B — corrected wording"
        edited_rows = await raw_adapter._records.query().where(f"id = '{b_id!s}'").to_list()
        edited_vector = [float(x) for x in edited_rows[0]["vector"]]

        adapter2 = await _reopen(store_path)
        report = await adapter2.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert refused == 1
        assert completed == 0
        assert skipped == 0

        b_after_repair = await adapter2.get_by_id(b_id)
        assert isinstance(b_after_repair, Decision)
        assert b_after_repair.content == "Decision B — corrected wording"
        rows_after = await adapter2._records.query().where(f"id = '{b_id!s}'").to_list()
        assert [float(x) for x in rows_after[0]["vector"]] == edited_vector

        a_after_repair = await adapter2.get_by_id(a_id)
        assert isinstance(a_after_repair, Decision)
        assert a_after_repair.lifecycle == "superseded"
        assert a_after_repair.superseded_by == b_id

        assert await adapter2.get_tombstone(b_id) is None

        events = await adapter2.list_unsupersedes()
        assert len(events) == 1

    async def test_refusal_survives_a_second_restart(self, tmp_path: Path) -> None:
        """Durable refusal follow-up, closing the cross-restart gap the P4
        verdict originally flagged as unclosed: a refused edge's
        ``refused_at`` stamp is now durable, so a SECOND, unrelated
        restart — no live retry in between, nothing else touching B —
        must not re-attempt the undo it already refused once.

        Two SEPARATE reopens (never a second ``repair_unsupersedes()``
        call on one live adapter instance — same "genuinely re-derive the
        cache from disk" discipline ``TestTwiceRestartAfterACompletedRepair``
        already uses for the completed-undo case).

        Falsify: drop the ``event.refused_at is not None`` short-circuit
        from ``resolve_unsupersede_repair`` — the second reopen's repair
        call sees "no tombstone" (already withdrawn by the first), treats
        it as a never-attempted undo, and deletes B — this test's
        ``still_b.content`` assertion goes red.
        """
        store_path = tmp_path / "store"
        raw_adapter = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=_DIM
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        wrapper = FaultInjectingAdapter(raw_adapter)
        engine = Tabularium(as_adapter(wrapper), embedder, clock, _make_config())
        await engine.initialize()

        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id

        wrapper.fault(on="restore", call=1, exc=RuntimeError("simulated failure"))
        wrapper.arm()
        with pytest.raises(ToolError):
            await engine.unsupersede(a_id, reason="wrong successor")

        edited_content = "Decision B — corrected wording"
        await edit_content(
            raw_adapter, b_id, edited_content, await embedder.embed(edited_content)
        )

        first_adapter = await _reopen(store_path)
        first_report = await first_adapter.repair_unsupersedes()
        first_completed = first_report.completed
        first_refused = first_report.refused
        assert first_completed == 0
        assert first_refused == 1

        second_adapter = await _reopen(store_path)
        second_report = await second_adapter.repair_unsupersedes()
        assert second_report.completed == 0
        assert second_report.refused == 0
        assert second_report.skipped == 1

        still_b = await second_adapter.get_by_id(b_id)
        assert isinstance(still_b, Decision)
        assert still_b.content == "Decision B — corrected wording"
        still_a = await second_adapter.get_by_id(a_id)
        assert isinstance(still_a, Decision)
        assert still_a.lifecycle == "superseded"
        assert still_a.superseded_by == b_id
        assert await second_adapter.get_tombstone(b_id) is None
        events = await second_adapter.list_unsupersedes()
        assert len(events) == 1
        assert events[0].refused_at is not None
