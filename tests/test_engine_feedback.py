"""Engine feedback-arm tests (M2 slice B, SPEC §8.1 [A8] / §8.2 /
§9.1 / §9.2).

Dual-parameterized over both storage adapters (memory / lancedb) via the
``engine_ctx`` fixture, which reuses the conformance suite's adapter
construction helpers (precedent — see ``tests/test_engine_
dispositions.py`` / ``tests/test_engine_concurrency.py``) rather than a
third construction pattern.

Falsifiability: the EWMA-sequence test (``TestFeedbackEWMA``) hand-computes
its expected value via inline arithmetic, never by calling
``compute_ewma``/``compute_ewma``-adjacent code — asserting against the
formula under test would prove nothing. It was walked against a deliberate
mutation (pinning ``compute_ewma`` to always return ``0.0``, per the
falsification step) and confirmed red for the stated reason. The floor property
(``TestFeedbackImportanceFloor``) drives >=20 zero-value events, per the
brief's explicit "not two" requirement — two zero events would not
distinguish "monotonically decaying toward, never below, the floor" from a
coincidental non-monotonic dip.
"""

from __future__ import annotations

import asyncio
import math
import pathlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.results import FeedbackResult
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter

# Adapter-construction pattern reused verbatim from the conformance suite
# (precedent) rather than a second one.
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.doubles import FakeEmbedder, TestClock

AnyFeedbackAdapter = MemoryAdapter | LanceDBAdapter
EngineCtx = tuple[Tabularium, AnyFeedbackAdapter, FakeEmbedder, TestClock, str]

_DIM = 64


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"candidate_k": 1000, "gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


@pytest.fixture(params=["memory", "lancedb"])
async def engine_ctx(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> EngineCtx:
    """A fresh, initialized engine + adapter + embedder + clock.

    Parameterized over ``request.param`` in ``{"memory", "lancedb"}`` — every
    test taking this fixture runs once per adapter.
    """
    embedder = FakeEmbedder(dim=_DIM)
    clock = _make_clock()
    adapter_id = request.param
    adapter: AnyFeedbackAdapter
    if adapter_id == "memory":
        adapter = _make_memory_adapter(
            embedding_model=embedder.model, dim=embedder.dim, clock=clock
        )
    else:
        adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model=embedder.model, dim=embedder.dim, clock=clock
        )
    cfg = _make_config()
    engine = Tabularium(adapter, embedder, clock, cfg)
    await engine.initialize()
    return engine, adapter, embedder, clock, adapter_id


async def _save_fact(engine: Tabularium, content: str) -> HypomnemaId:
    r = await engine.save(content, scope="global")
    assert r.hypomnema is not None
    return r.hypomnema.id


# ---------------------------------------------------------------------------
# Basic write-path shape
# ---------------------------------------------------------------------------


class TestFeedbackBasics:
    async def test_appends_one_event_with_value_and_reason(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 1 (brief): one event per call, source="feedback", value and
        reason persisted verbatim.

        Falsify: comment out the ``await adapter.append_event(event)`` call
        in ``feedback`` — ``query_events`` returns zero events.
        """
        engine, adapter, _embedder, _clock, adapter_id = engine_ctx
        fid = await _save_fact(engine, "fact for feedback event-shape test")

        result = await engine.feedback(fid, value=0.75, reason="clear and useful")
        assert isinstance(result, FeedbackResult)

        events = await adapter.query_events(target_id=fid, source="feedback")
        assert len(events) == 1, f"[{adapter_id}] expected exactly one feedback event"
        assert events[0].source == "feedback"
        assert events[0].value == pytest.approx(0.75)
        assert events[0].reason == "clear and useful"

    async def test_reason_absent_is_none_not_empty_string(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 2 (brief): an omitted ``reason`` persists as ``None``, never
        an empty string.

        Falsify: default ``reason`` to ``""`` instead of ``None`` anywhere
        on the write path (engine method signature or event construction).
        """
        engine, adapter, _embedder, _clock, adapter_id = engine_ctx
        fid = await _save_fact(engine, "fact for feedback reason-absent test")

        await engine.feedback(fid, value=0.5)

        events = await adapter.query_events(target_id=fid, source="feedback")
        assert len(events) == 1
        assert events[0].reason is None, (
            f"[{adapter_id}] omitted reason must round-trip as None, "
            f"got {events[0].reason!r}"
        )

    async def test_value_out_of_range_rejected_not_clamped(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 3 (brief): out-of-[0,1] values raise ``ToolError`` at the
        engine boundary and write NOTHING — never silently clamped into
        range.

        Falsify: replace the explicit range check in ``feedback`` with
        ``value = max(0.0, min(1.0, value))`` (silent clamp) — both calls
        below would succeed and each would append an event with a clamped
        value instead of raising.
        """
        engine, adapter, _embedder, _clock, adapter_id = engine_ctx
        fid = await _save_fact(engine, "fact for feedback range-rejection test")

        # One valid baseline call, so a wrongly-permitted out-of-range call
        # would be visible as a SECOND event, not just a nonzero count.
        await engine.feedback(fid, value=0.5)
        before = await adapter.query_events(target_id=fid, source="feedback")
        assert len(before) == 1

        with pytest.raises(ToolError):
            await engine.feedback(fid, value=1.5)
        with pytest.raises(ToolError):
            await engine.feedback(fid, value=-0.1)

        after = await adapter.query_events(target_id=fid, source="feedback")
        assert len(after) == 1, (
            f"[{adapter_id}] an out-of-range value must write nothing; "
            f"found {len(after)} feedback events after two rejected calls"
        )

        record = await adapter.get_by_id(fid)
        assert isinstance(record, Fact)
        assert record.feedback_score == pytest.approx(0.15), (
            f"[{adapter_id}] feedback_score must be exactly the single valid "
            f"call's EWMA (0.3*0.5=0.15), never a value influenced by a "
            f"clamped out-of-range call"
        )

    async def test_feedback_on_decision_raises_category_error(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 4 (brief): SPEC §9.2 category error — feedback is
        Facts-only.

        Falsify: remove the ``isinstance(record, Fact)`` guard in
        ``feedback`` — this call would instead try (and likely fail
        differently, or silently "succeed") against a Decision.
        """
        engine, _adapter, _embedder, _clock, adapter_id = engine_ctx
        d = await engine.save(
            "a wildly unrelated decision content for feedback category test",
            scope="global",
            kind="decision",
        )
        assert d.outcome == "saved"
        assert d.hypomnema is not None

        with pytest.raises(ToolError) as excinfo:
            await engine.feedback(d.hypomnema.id, value=0.5)
        assert "Decision" in str(excinfo.value), (
            f"[{adapter_id}] expected the category-error message to name "
            f"the Decision/Fact mismatch, got: {excinfo.value!s}"
        )

    async def test_feedback_on_unknown_id_raises_unknown_id_error(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 5 (brief): the existing unknown-id ``ToolError`` contract,
        same message shape as every other id-addressed tool.
        """
        engine, _adapter, _embedder, _clock, _adapter_id = engine_ctx
        with pytest.raises(ToolError, match="Unknown id"):
            await engine.feedback(new_id(), value=0.5)


# ---------------------------------------------------------------------------
# EWMA correctness — hand-computed, not implementation-derived
# ---------------------------------------------------------------------------


class TestFeedbackEWMA:
    async def test_feedback_score_matches_hand_computed_ewma_sequence(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 6 (brief) — the headline correctness property.

        Four ``feedback`` calls with values [1.0, 0.0, 0.5, 1.0]; the
        expected ``feedback_score`` is computed by hand, inline, via plain
        arithmetic — NEVER by calling ``compute_ewma`` or any other
        implementation formula as the oracle.

          s0 = 0.0                                   (init)
          s1 = 0.3*1.0 + 0.7*s0 = 0.3
          s2 = 0.3*0.0 + 0.7*s1 = 0.21
          s3 = 0.3*0.5 + 0.7*s2 = 0.15 + 0.147 = 0.297
          s4 = 0.3*1.0 + 0.7*s3 = 0.3 + 0.2079 = 0.5079

        Falsify: pin ``compute_ewma`` (``engine/projections.py``) to always
        return ``0.0`` — this test fails with the stored ``feedback_score``
        at 0.0 instead of ~0.5079. Confirmed red for exactly this reason.

        The clock is advanced a trivial amount between calls so each
        event's ``at`` is strictly increasing — this test asserts the
        DOCUMENTED ``query_events`` ordering contract (ascending ``at``),
        never the same-timestamp tie-break, which is explicitly adapter-
        defined and un-relied-upon (see ``StorageAdapter.query_events``).
        """
        engine, adapter, _embedder, clock, adapter_id = engine_ctx
        fid = await _save_fact(engine, "fact for feedback EWMA sequence test")

        values = [1.0, 0.0, 0.5, 1.0]
        last_result: FeedbackResult | None = None
        for v in values:
            clock.advance(timedelta(seconds=1))
            last_result = await engine.feedback(fid, value=v)

        expected = 0.0
        for v in values:
            expected = 0.3 * v + 0.7 * expected
        assert expected == pytest.approx(0.5079, abs=1e-9)  # sanity on the hand-derivation itself

        record = await adapter.get_by_id(fid)
        assert isinstance(record, Fact)
        assert math.isclose(record.feedback_score, expected, abs_tol=1e-9), (
            f"[{adapter_id}] stored feedback_score={record.feedback_score!r} "
            f"does not match the hand-computed EWMA sequence {expected!r}"
        )
        assert last_result is not None
        assert math.isclose(last_result.feedback_score, expected, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Feedback ⊥ references (SPEC §8.2, normative)
# ---------------------------------------------------------------------------


class TestFeedbackReferencesOrthogonality:
    async def test_recall_count_and_last_referenced_at_byte_unchanged(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 7 (brief): a feedback call must not move ``recall_count`` or
        ``last_referenced_at`` — asserted directly on the STORED record, not
        on a return value.

        The record is first reinforced via a real dedup event (a
        whitespace-variant save that matches on the exact content key,
        per the admission order's step 3) so both fields are non-default
        before the feedback call — otherwise this assertion would be
        vacuously satisfied by two zeros.

        Falsify: change ``feedback`` to call
        ``adapter.update_projections`` with ``recall_count=record.
        recall_count + 1`` (a plausible-looking but wrong "reference" bump)
        instead of the recomputed-from-log value.
        """
        cfg = _make_config()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        adapter: AnyFeedbackAdapter = _make_memory_adapter(
            embedding_model=embedder.model, dim=embedder.dim, clock=clock
        )
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        content = "fact for feedback reference-orthogonality test"
        r0 = await engine.save(content, scope="global")
        assert r0.hypomnema is not None
        fid = r0.hypomnema.id

        clock.advance(timedelta(hours=1))
        # A whitespace variant (P7 admission-order piece, step 3 exact-key
        # match) reinforces the same row — a byte-identical resubmission of
        # ``content`` would ack ``already_saved`` instead (step 1), which
        # never reinforces.
        r1 = await engine.save(content + " ", scope="global")
        assert r1.outcome == "reinforced"
        assert r1.reinforced_id == fid

        before = await adapter.get_by_id(fid)
        assert isinstance(before, Fact)
        assert before.recall_count == 1
        assert before.last_referenced_at is not None

        await engine.feedback(fid, value=1.0, reason="orthogonality check")

        after = await adapter.get_by_id(fid)
        assert isinstance(after, Fact)
        assert after.recall_count == before.recall_count, (
            "feedback must never touch recall_count"
        )
        assert after.last_referenced_at == before.last_referenced_at, (
            "feedback must never touch last_referenced_at"
        )
        # Sanity: the feedback event itself DID land and DID move feedback_score.
        assert after.feedback_score != before.feedback_score

    async def test_lifecycle_untouched_by_feedback(self, engine_ctx: EngineCtx) -> None:
        """Case 8 (brief): ``lifecycle`` is untouched on every path — feedback
        raises importance, never membership. ``tests/test_hot.py::
        TestNoAutoPromotion`` covers the store-wide property; this asserts
        it directly on a record driven through repeated high-value feedback.

        Falsify: any accidental call to ``adapter.transition`` inside
        ``feedback``.
        """
        engine, adapter, _embedder, _clock, adapter_id = engine_ctx
        fid = await _save_fact(engine, "fact for feedback lifecycle-untouched test")
        before = await adapter.get_by_id(fid)
        assert before.lifecycle == "active"

        for _ in range(5):
            await engine.feedback(fid, value=1.0)

        after = await adapter.get_by_id(fid)
        assert after.lifecycle == "active", (
            f"[{adapter_id}] feedback must never move lifecycle, "
            f"got {after.lifecycle!r} after repeated value=1.0 calls"
        )


# ---------------------------------------------------------------------------
# Importance rises / falls, never below the pre-feedback floor ([A8])
# ---------------------------------------------------------------------------


class TestFeedbackImportanceFloor:
    async def test_importance_rises_then_never_falls_below_pre_feedback_floor(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 9 (brief) — the [A8] no-backdoor-eviction guarantee. Uses
        >=20 zero-value events (not two), per the brief: two events cannot
        distinguish "monotonically decaying toward, never below, the floor"
        from a coincidental non-monotonic dip.

        The clock is advanced by a trivial (1-second) step before each
        call — NOT held perfectly fixed — so every event gets a strictly
        increasing ``at`` and this test asserts the DOCUMENTED
        ``query_events`` ordering contract (ascending ``at``) rather than
        an unspecified same-timestamp tie-break. Twenty-one one-second steps drift the recency term
        by
        at most ~8e-6 (relative) over the whole test — negligible next to
        the ``floor_eps`` slack below, which exists to absorb exactly that
        drift, not to weaken the property under test.

        Falsify: drop the "non-negative additive" contract — e.g. center
        feedback on 0.5 (``score - 0.5``) before multiplying by
        ``w_feedback`` — the score would go negative after enough 0.0
        events and importance would fall below the pre-feedback floor by
        many orders of magnitude more than ``floor_eps``.
        """
        engine, _adapter, _embedder, clock, adapter_id = engine_ctx
        fid = await _save_fact(engine, "fact for feedback importance-floor test")

        cfg = engine._cfg
        # Pre-feedback floor, hand-derived: a freshly-saved Fact has
        # recall_count=0 and last_referenced_at=None (-> created_at), so at
        # delta_days~=0 (small clock advances only), recency~=exp(0)=1.0.
        #   floor = w_recall*ln(1+0) + w_recency*1.0 + w_feedback*0.0
        #         = 0.0 + w_recency + 0.0 = w_recency
        floor = cfg.w_recency
        floor_eps = 1e-4  # absorbs the tiny recency drift from the clock advances above

        clock.advance(timedelta(seconds=1))
        r_high = await engine.feedback(fid, value=1.0)
        assert r_high.importance > floor, (
            f"[{adapter_id}] importance must RISE above the pre-feedback "
            f"floor after a value=1.0 event: got {r_high.importance!r}, "
            f"floor={floor!r}"
        )

        last_importance = r_high.importance
        for i in range(20):
            clock.advance(timedelta(seconds=1))
            r_zero = await engine.feedback(fid, value=0.0)
            assert r_zero.importance >= floor - floor_eps, (
                f"[{adapter_id}] importance fell below the pre-feedback "
                f"floor after {i + 1} zero-value events: "
                f"{r_zero.importance!r} < floor={floor!r}"
            )
            assert r_zero.importance <= last_importance + 1e-9, (
                f"[{adapter_id}] importance must monotonically erode toward "
                f"(never bounce back above) its previous value across "
                f"consecutive zero-value events; step {i + 1} went from "
                f"{last_importance!r} to {r_zero.importance!r}"
            )
            last_importance = r_zero.importance

        # After 20 zero events the feedback term should have decayed close
        # to (but, by construction, never below) the floor.
        assert last_importance >= floor - floor_eps
        assert last_importance == pytest.approx(floor, abs=1e-3)


# ---------------------------------------------------------------------------
# Concurrency — two concurrent feedback calls fuse, never lose an update
# ---------------------------------------------------------------------------


class TestFeedbackConcurrency:
    async def test_two_concurrent_calls_produce_fused_ewma_not_lost_update(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Case 10 (brief). Two concurrent ``feedback`` calls on the SAME
        record must both land (two events, no dropped write) and the
        stored ``feedback_score`` must be the EWMA of both, applied in
        whichever order the shared lock actually serializes them — never a
        "lost update" where one call's ``update_projections`` write is
        computed from a stale event-log snapshot and clobbers the other's.

        Determinism, same gating idiom as ``test_engine_concurrency.py``'s
        ``TestBuryVersusSave``: ``adapter.update_projections`` is
        monkeypatched so the FIRST call's write pauses right before it
        actually persists (after it has already appended its own event and
        recomputed from the log it saw). The second call is proven BLOCKED
        (no progress within a short timeout) while the first sits paused
        there — proving the lock spans the full
        append-recompute-write body, not just the append — then the first
        is released, and the second (now free to run) sees the first's
        already-committed event in its own query, so its own write is the
        correct joint EWMA.

        Falsify: release the engine's write lock right after
        ``adapter.append_event`` instead of holding it across
        ``update_projections`` too — the second call would no longer be
        blocked in the middle section above, and could interleave its own
        read-compute-write inside the first's paused window, producing a
        final ``feedback_score`` that reflects only one call's value
        instead of both fused in log order.

        The clock is advanced (a trivial amount) between task_a starting
        and task_b being created, so the two events get strictly increasing
        ``at`` values — the forced order below is therefore backed by BOTH
        the lock (task_b cannot even attempt its own work until task_a's
        entire critical section releases it) AND the documented ``at``-
        ascending ``query_events`` ordering contract, never a same-
        timestamp tie-break.
        """
        engine, adapter, _embedder, clock, adapter_id = engine_ctx
        fid = await _save_fact(engine, "fact for concurrent feedback test")

        real_update_projections = adapter.update_projections
        gate_started = asyncio.Event()
        gate_release = asyncio.Event()

        async def _gated_update_projections(id: Any, **kw: Any) -> None:
            if not gate_started.is_set():
                gate_started.set()
                await gate_release.wait()
            await real_update_projections(id, **kw)

        monkeypatch.setattr(adapter, "update_projections", _gated_update_projections)

        task_a = asyncio.create_task(engine.feedback(fid, value=1.0))
        await asyncio.wait_for(gate_started.wait(), timeout=5.0)
        # task_a's own `now = clock.now()` already ran (and its event is
        # already appended) before this gate fired -- advancing here only
        # affects task_b's later `now()` read, giving it a strictly LATER
        # `at` than task_a's already-fixed one.
        clock.advance(timedelta(seconds=1))

        task_b = asyncio.create_task(engine.feedback(fid, value=0.5))
        try:
            await asyncio.wait_for(asyncio.shield(task_b), timeout=0.5)
        except TimeoutError:
            pass  # expected: task_b made no progress while task_a holds the lock
        else:
            pytest.fail(
                f"[{adapter_id}] the second feedback call completed while the "
                "first was still inside its gated update_projections -- the "
                "shared write lock is not serializing feedback calls"
            )

        gate_release.set()
        result_a, result_b = await asyncio.gather(task_a, task_b)

        events = await adapter.query_events(target_id=fid, source="feedback")
        assert len(events) == 2, (
            f"[{adapter_id}] expected exactly two feedback events (no lost "
            f"write), found {len(events)}"
        )
        # Forced order: task_b's lock acquisition cannot succeed until
        # task_a's ENTIRE critical section (including the gated write) has
        # released it, so task_a's event is always appended first.
        assert [e.value for e in events] == [1.0, 0.5]

        expected = 0.3 * 1.0
        expected = 0.3 * 0.5 + 0.7 * expected
        record = await adapter.get_by_id(fid)
        assert isinstance(record, Fact)
        assert math.isclose(record.feedback_score, expected, abs_tol=1e-9), (
            f"[{adapter_id}] stored feedback_score={record.feedback_score!r} "
            f"is not the fused EWMA of both concurrent calls "
            f"(expected {expected!r}) -- a lost update"
        )
        assert result_a.feedback_score == pytest.approx(0.3, abs=1e-9)
        assert result_b.feedback_score == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# LanceDB open-time schema migration for the events table: a store created BEFORE ``events.reason``
# existed
# has no such column, and ``append_event`` now unconditionally writes a
# ``"reason"`` key — same shape as the records-table migration regression in
# ``tests/test_escalation.py::test_lancedb_reopen_migrates_missing_
# recurrence_columns``, generalized in ``lancedb_adapter._migrate_table_
# schema`` so both tables share one migration path.
# ---------------------------------------------------------------------------


class TestLanceDBEventsSchemaMigration:
    async def test_reopen_migrates_missing_reason_column(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: comment out the events-table ``_migrate_table_
        schema`` call in ``LanceDBAdapter.open`` — this test goes red
        because ``append_event`` raises an Arrow schema-mismatch error
        against the physical (old-schema) events table instead of
        succeeding with the pre-existing row's ``reason`` transparently
        backfilled to NULL.
        """
        import lancedb
        import pyarrow as pa

        from tabularium.domain import new_id
        from tabularium.storage.lancedb_codec import EVENTS_SCHEMA
        from tabularium.storage.types import ReinforcementEvent

        store_path = str(tmp_path / "old_events_schema_store")
        embedding_model = "test-model"
        dim = 8
        now = datetime(2025, 1, 1, tzinfo=UTC)

        # Build the OLD events schema: identical to the current one minus
        # the ``reason`` column this task adds.
        old_fields = [f for f in EVENTS_SCHEMA if f.name != "reason"]
        old_schema = pa.schema(old_fields)

        # Write one row directly against the OLD schema — bypassing this
        # adapter entirely, simulating an event log written by the
        # pre-diff codebase.
        old_target_id = new_id()
        old_row = {
            "target_id": str(old_target_id),
            "at": now,
            "source": "dedup",
            "matched_from_id": None,
            "value": None,
        }

        db = await lancedb.connect_async(store_path)
        await db.create_table("events", schema=old_schema)
        old_table = await db.open_table("events")
        await old_table.add([old_row])

        # Reopen with the CURRENT adapter — must migrate the events table
        # in place, backfilling the pre-existing row's ``reason`` to NULL.
        adapter = await LanceDBAdapter.open(
            store_path, embedding_model=embedding_model, dim=dim
        )

        migrated = await adapter.query_events(target_id=old_target_id)
        assert len(migrated) == 1
        assert migrated[0].reason is None

        # A fresh feedback-shaped event with a real reason must round-trip
        # cleanly against the now-migrated table — this is the write
        # ``append_event`` would have failed on before the fix (Arrow
        # schema mismatch: the row dict carries a "reason" key the old
        # physical table has no column for).
        new_target_id = new_id()
        await adapter.append_event(
            ReinforcementEvent(
                target_id=new_target_id,
                at=now,
                source="feedback",
                value=0.7,
                reason="post-migration reason round-trip",
            )
        )
        after = await adapter.query_events(target_id=new_target_id)
        assert len(after) == 1
        assert after[0].reason == "post-migration reason round-trip"
        assert after[0].value == pytest.approx(0.7)

        # A second reopen (migration already applied) must be a no-op, not
        # an error.
        adapter2 = await LanceDBAdapter.open(
            store_path, embedding_model=embedding_model, dim=dim
        )
        again = await adapter2.query_events(target_id=old_target_id)
        assert len(again) == 1
        assert again[0].reason is None
