"""Projection-rebuild maintenance-verb tests (M2 slice B, SPEC §8.2 [A9]).

Dual-parameterized over both storage adapters (memory / lancedb) via the
``adapter`` fixture, reusing the conformance suite's adapter-construction
helpers (precedent — see ``tests/test_engine_feedback.py`` /
``tests/test_engine_dispositions.py``) rather than a third construction
pattern.

Falsifiability (task brief's explicit requirement): ``TestReportMode`` and
``TestRepairMode`` corrupt a stored ``feedback_score`` directly via
``update_projections`` — bypassing the normal write path entirely, which is
exactly the "materialized state disagrees with the log" condition this verb
exists to detect — then assert report mode flags **exactly** that record (a
sibling record with a genuinely-clean history is asserted absent from
``drifted``), that report mode leaves the corrupted value in the store
untouched, that repair mode restores the log-derived value, and that a
following report on the now-clean store finds zero drift.

``TestDecayDueAtSameAnalyticPath`` goes through the real engine (``Tabularium
.save`` + ``.feedback``, the live write path) rather than hand-seeding
"correct" projection values, so a rebuild that used a different formula for
``decay_due_at`` would show up as drift against records the live path itself
just wrote — the strongest available proof that this verb reuses the write
path's own formula rather than a second implementation of it.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, HypomnemaId, Scope, new_id
from tabularium.engine import Tabularium
from tabularium.engine.maintenance import rebuild_projections
from tabularium.engine.projections import (
    compute_decay_due_at,
    compute_ewma,
    compute_last_referenced_at,
    compute_recall_count,
)
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import ReinforcementEvent
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.doubles import FakeEmbedder, TestClock

AnyRebuildAdapter = MemoryAdapter | LanceDBAdapter

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DIM = 4


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _cfg(**kw: Any) -> EngineConfig:
    return EngineConfig(**kw)


def _fact(
    *,
    scope: str = "proj-a",
    lifecycle: str = "active",
    validity: str = "valid",
    content: str = "rebuild test fact",
    created_at: datetime = EPOCH,
    **kw: Any,
) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=created_at,
        updated_at=created_at,
        **kw,
    )


def _decision(
    *,
    scope: str = "proj-a",
    lifecycle: str = "in_force",
    validity: str = "valid",
    content: str = "rebuild test decision",
    created_at: datetime = EPOCH,
    **kw: Any,
) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=created_at,
        updated_at=created_at,
        **kw,
    )


def _vec(seed: float = 1.0, dim: int = _DIM) -> list[float]:
    import math

    v = [seed] * dim
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


@pytest.fixture(params=["memory", "lancedb"])
async def adapter(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> AnyRebuildAdapter:
    """A fresh adapter, parameterized over both {"memory", "lancedb"}."""
    if request.param == "memory":
        return _make_memory_adapter(dim=_DIM)
    return await _make_lancedb_adapter(tmp_path, dim=_DIM)


# ===========================================================================
# Report mode
# ===========================================================================


class TestReportMode:
    async def test_clean_store_reports_zero_drift(
        self, adapter: AnyRebuildAdapter
    ) -> None:
        """A never-reinforced Fact/Decision AND a genuinely-reinforced Fact
        whose stored projections already match its event log both report
        clean — report mode is not merely "everything defaults to zero."
        """
        cfg = _cfg()

        never_reinforced = _fact(content="never reinforced")
        await adapter.upsert(never_reinforced, _vec(1.0))

        never_reinforced_decision = _decision(content="never reinforced decision")
        await adapter.upsert(never_reinforced_decision, _vec(2.0))

        # A genuinely-reinforced Fact: build its event log first, derive the
        # SAME projections the live write path would have computed via the
        # SAME pure helpers, and store exactly that — a non-trivial case
        # where "no drift" is not simply every field at its domain default.
        reinforced = _fact(content="reinforced fact")
        events = [
            ReinforcementEvent(target_id=reinforced.id, at=EPOCH, source="dedup"),
            ReinforcementEvent(
                target_id=reinforced.id,
                at=EPOCH + timedelta(days=1),
                source="feedback",
                value=0.8,
            ),
        ]
        recall_count = compute_recall_count(events)
        last_referenced_at = compute_last_referenced_at(events)
        feedback_score = compute_ewma(events)
        decay_due_at = compute_decay_due_at(
            recall_count=recall_count,
            feedback_score=feedback_score,
            last_referenced_at=last_referenced_at,
            created_at=reinforced.created_at,
            cfg=cfg,
        )
        reinforced = reinforced.model_copy(
            update={
                "recall_count": recall_count,
                "last_referenced_at": last_referenced_at,
                "feedback_score": feedback_score,
                "decay_due_at": decay_due_at,
            }
        )
        await adapter.upsert(reinforced, _vec(3.0))
        for ev in events:
            await adapter.append_event(ev)

        report = await rebuild_projections(adapter, cfg, repair=False)

        assert report.mode == "report"
        assert report.scanned == 3
        assert report.drifted == []
        assert report.repaired == 0

    async def test_corrupted_feedback_score_flagged_exactly(
        self, adapter: AnyRebuildAdapter
    ) -> None:
        cfg = _cfg()

        target = _fact(content="corrupted target")
        sibling = _fact(content="clean sibling")
        await adapter.upsert(target, _vec(1.0))
        await adapter.upsert(sibling, _vec(2.0))

        # Corrupt ONLY target's feedback_score, directly via the seam —
        # bypassing the normal write path entirely (no event backs this
        # value; the log alone says feedback_score should be 0.0).
        await adapter.update_projections(
            target.id,
            recall_count=0,
            last_referenced_at=None,
            feedback_score=0.9,
            decay_due_at=None,
        )

        report = await rebuild_projections(adapter, cfg, repair=False)

        assert len(report.drifted) == 1, (
            f"expected exactly one drifted record, got {len(report.drifted)}: "
            f"{report.drifted!r}"
        )
        entry = report.drifted[0]
        assert entry.id == target.id
        assert entry.kind == "fact"
        assert entry.feedback_score == (0.9, 0.0)
        assert entry.recall_count is None
        assert entry.last_referenced_at is None
        assert entry.decay_due_at is None
        assert not any(d.id == sibling.id for d in report.drifted)

    async def test_report_mode_writes_nothing(
        self, adapter: AnyRebuildAdapter
    ) -> None:
        cfg = _cfg()
        target = _fact(content="report must not write")
        await adapter.upsert(target, _vec())
        await adapter.update_projections(
            target.id,
            recall_count=0,
            last_referenced_at=None,
            feedback_score=0.9,
            decay_due_at=None,
        )

        await rebuild_projections(adapter, cfg, repair=False)

        # Assert against the STORE, not the return value: the corrupted
        # value is still there — report mode touched nothing.
        after = await adapter.get_by_id(target.id)
        assert isinstance(after, Fact)
        assert after.feedback_score == pytest.approx(0.9)

    async def test_decision_drift_never_carries_fact_only_fields(
        self, adapter: AnyRebuildAdapter
    ) -> None:
        """A Decision has no feedback_score/decay_due_at at all — a drifted
        Decision's entry must never populate those two fields, even though
        its recall_count genuinely disagrees.
        """
        cfg = _cfg()
        decision = _decision(content="decision with corrupted recall_count")
        await adapter.upsert(decision, _vec())
        await adapter.update_projections(
            decision.id,
            recall_count=7,
            last_referenced_at=None,
            feedback_score=0.0,
            decay_due_at=None,
        )

        report = await rebuild_projections(adapter, cfg, repair=False)

        assert len(report.drifted) == 1
        entry = report.drifted[0]
        assert entry.id == decision.id
        assert entry.kind == "decision"
        assert entry.recall_count == (7, 0)
        assert entry.feedback_score is None
        assert entry.decay_due_at is None


# ===========================================================================
# Repair mode
# ===========================================================================


class TestRepairMode:
    async def test_repair_restores_log_derived_values_then_reports_clean(
        self, adapter: AnyRebuildAdapter
    ) -> None:
        cfg = _cfg()
        target = _fact(content="corrupted, to be repaired")
        await adapter.upsert(target, _vec())
        await adapter.update_projections(
            target.id,
            recall_count=0,
            last_referenced_at=None,
            feedback_score=0.9,
            decay_due_at=None,
        )

        repair_report = await rebuild_projections(adapter, cfg, repair=True)
        assert repair_report.mode == "repair"
        assert len(repair_report.drifted) == 1
        assert repair_report.repaired == 1

        restored = await adapter.get_by_id(target.id)
        assert isinstance(restored, Fact)
        assert restored.feedback_score == pytest.approx(0.0)

        clean_report = await rebuild_projections(adapter, cfg, repair=False)
        assert clean_report.drifted == []
        assert clean_report.repaired == 0

    async def test_rebuild_twice_is_byte_identical_and_idempotent(
        self, adapter: AnyRebuildAdapter
    ) -> None:
        """Determinism: repairing an already-clean store changes nothing,
        and running it a second time in a row reports the same (empty)
        drift and repairs nothing further.
        """
        cfg = _cfg()
        fact = _fact(content="idempotence target")
        await adapter.upsert(fact, _vec())

        first = await rebuild_projections(adapter, cfg, repair=True)
        after_first = await adapter.get_by_id(fact.id)
        second = await rebuild_projections(adapter, cfg, repair=True)
        after_second = await adapter.get_by_id(fact.id)

        assert first.drifted == []
        assert first.repaired == 0
        assert second.drifted == []
        assert second.repaired == 0
        assert after_first == after_second


# ===========================================================================
# Concurrency — a record buried between the scan and its repair write
# ===========================================================================


class TestConcurrentBuryDuringRepair:
    async def test_keyerror_on_repair_write_is_caught_not_raised(
        self, adapter: AnyRebuildAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same lock-free scan-then-fetch failure shape fixed twice
        already (``engine/recall.py``, ``engine/context.py``): a record
        buried between ``scan_records``'s snapshot and this record's own
        repair write must not blow up the whole rebuild. Simulated
        deterministically (no timing-dependent real race) by making
        ``update_projections`` raise ``KeyError`` for one specific id, the
        same idiom ``tests/test_context.py::TestNoPerCandidateFetch`` uses
        for ``get_by_id``.
        """
        cfg = _cfg()
        doomed = _fact(content="buried mid-rebuild")
        survivor = _fact(content="repaired normally")
        for rec in (doomed, survivor):
            await adapter.upsert(rec, _vec())
        for rec in (doomed, survivor):
            await adapter.update_projections(
                rec.id,
                recall_count=0,
                last_referenced_at=None,
                feedback_score=0.9,
                decay_due_at=None,
            )

        real_update_projections = adapter.update_projections

        async def _racy_update_projections(id: HypomnemaId, **kw: Any) -> None:
            if id == doomed.id:
                raise KeyError(f"Unknown id: {id!s}")
            await real_update_projections(id, **kw)

        monkeypatch.setattr(adapter, "update_projections", _racy_update_projections)

        report = await rebuild_projections(adapter, cfg, repair=True)

        assert {d.id for d in report.drifted} == {doomed.id, survivor.id}, (
            "both were genuinely drifted at scan time — the race must not "
            "hide either from the report"
        )
        assert report.repaired == 1, "only the non-doomed write should count"

        # survivor's write landed normally...
        survivor_after = await adapter.get_by_id(survivor.id)
        assert isinstance(survivor_after, Fact)
        assert survivor_after.feedback_score == pytest.approx(0.0)

        # ...but doomed's repair write never reached the adapter — its
        # corrupted value is exactly what it was before this rebuild call.
        doomed_after = await adapter.get_by_id(doomed.id)
        assert isinstance(doomed_after, Fact)
        assert doomed_after.feedback_score == pytest.approx(0.9)


# ===========================================================================
# scan_records seam — never per-id get_by_id
# ===========================================================================


class TestNeverCallsGetById:
    async def test_report_and_repair_never_call_get_by_id(
        self, adapter: AnyRebuildAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _cfg()
        fact = _fact(content="no per-id fetch")
        await adapter.upsert(fact, _vec())
        await adapter.update_projections(
            fact.id,
            recall_count=0,
            last_referenced_at=None,
            feedback_score=0.9,
            decay_due_at=None,
        )

        async def _forbidden_get_by_id(id: HypomnemaId) -> Any:
            raise AssertionError(
                "rebuild_projections must not call get_by_id -- it reads "
                "full records via scan_records in one round trip"
            )

        monkeypatch.setattr(adapter, "get_by_id", _forbidden_get_by_id)

        report = await rebuild_projections(adapter, cfg, repair=True)
        assert report.repaired == 1


# ===========================================================================
# decay_due_at — the same analytic path as the live write path
# ===========================================================================


class TestDecayDueAtSameAnalyticPath:
    async def test_live_write_path_projections_show_zero_drift(
        self, adapter: AnyRebuildAdapter
    ) -> None:
        """Build projections through the REAL engine (save + feedback, the
        live write path) rather than hand-seeding "correct" values -- a
        rebuild using a different decay_due_at formula would show up as
        drift here.
        """
        embedder = FakeEmbedder(dim=_DIM)
        clock = TestClock(EPOCH)
        cfg = _cfg(candidate_k=1000, gray_floor=0.80)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save("decay path fact", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.feedback(fid, value=0.8, reason="useful")

        report = await rebuild_projections(adapter, cfg, repair=False)

        assert report.drifted == [], (
            f"live write path's own projections disagree with the rebuild's "
            f"recomputation: {report.drifted!r}"
        )

    async def test_never_reinforced_fact_keeps_decay_due_at_none(
        self, adapter: AnyRebuildAdapter
    ) -> None:
        """A Fact saved but never reinforced has decay_due_at=None at
        creation (the live write path never computes it before a first
        reinforcement -- SPEC §8.2: "recomputed on each reinforcement").
        The rebuild must mirror that invocation condition, not compute the
        formula's zero-recall/zero-feedback output, or every never-
        reinforced record in the corpus would show spurious drift.
        """
        embedder = FakeEmbedder(dim=_DIM)
        clock = TestClock(EPOCH)
        cfg = _cfg(candidate_k=1000, gray_floor=0.80)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save("never reinforced, real save path", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        stored = await adapter.get_by_id(fid)
        assert isinstance(stored, Fact)
        assert stored.decay_due_at is None

        report = await rebuild_projections(adapter, cfg, repair=False)
        assert report.drifted == []
