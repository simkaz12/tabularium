"""Engine-level integration tests for the read-event log: the two call sites that
log a ``ReadEvent`` (``recall()`` and ``decision()``).

``context()`` (session-start grounding) is deliberately NOT one of them: a later ruling
superseding a prior decision's original "log it but tag it" design removed the write entirely
(see ``engine.read_events``'s own module docstring). ``TestContextNeverLogsReadEvents`` below
pins that state.

Storage-seam round-trip (``append_read_event`` on both
adapters) is covered directly in ``tests/conformance/test_conformance.py``'s
``TestReadEvents`` — this file is the engine-level behaviour suite,
single-adapter (``MemoryAdapter``), the same "engine tests don't re-prove
the seam" convention ``tests/test_context.py``/``tests/test_shelved.py``
already follow.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, HypomnemaId, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import MaintenanceDebt, ReadEvent
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock

EPOCH = datetime(2025, 1, 1, tzinfo=UTC)
_DIM = 16


def _cfg(**kw: object) -> EngineConfig:
    """Deterministic-importance config: importance == feedback_score (same
    idiom as ``tests/test_context.py``'s own ``_cfg``)."""
    defaults: dict[str, object] = {"w_recall": 0.0, "w_recency": 0.0, "w_feedback": 1.0}
    defaults.update(kw)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


async def _engine(
    *,
    cfg: EngineConfig | None = None,
    clock: TestClock | None = None,
    adapter: MemoryAdapter | None = None,
) -> tuple[Tabularium, MemoryAdapter, FakeEmbedder, TestClock]:
    embedder = FakeEmbedder(dim=_DIM)
    clk = clock if clock is not None else TestClock(EPOCH)
    adp = adapter if adapter is not None else MemoryAdapter(
        embedding_model=embedder.model, dim=embedder.dim, clock=clk
    )
    engine = Tabularium(
        adp,
        embedder,
        clk,
        cfg if cfg is not None else _cfg(),
        token_counter=FakeTokenCounter(),
    )
    await engine.initialize()
    await adp.create_scope(Scope("proj-a"))
    await adp.create_scope(Scope("proj-b"))
    return engine, adp, embedder, clk


def _direct_fact(
    *,
    content: str,
    lifecycle: str = "active",
    scope: str = "proj-a",
    audience_hint: tuple[str, ...] | None = None,
    feedback_score: float = 0.0,
    created_at: datetime = EPOCH,
) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        audience_hint=audience_hint,
        feedback_score=feedback_score,
        created_at=created_at,
        updated_at=created_at,
    )


def _direct_decision(
    *, content: str, scope: str = "proj-a", created_at: datetime = EPOCH
) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope(scope),
        lifecycle="in_force",
        validity="valid",
        created_at=created_at,
        updated_at=created_at,
    )


def _read_events_for(adapter: MemoryAdapter, target_id: HypomnemaId) -> list[ReadEvent]:
    return [event for event in adapter._read_events if event.target_id == target_id]


class _RaisingReadEventAdapter(MemoryAdapter):
    """A ``MemoryAdapter`` whose ``append_read_event`` always raises — proves
    a read-event logging failure never fails the read it is attached to."""

    async def append_read_event(self, event: ReadEvent) -> None:
        raise RuntimeError("simulated read_events write failure")


# ---------------------------------------------------------------------------
# grounding — context() logs NOTHING (later ruling superseding decision
# an earlier decision's original "log it but tag it" design)
# ---------------------------------------------------------------------------


class TestContextNeverLogsReadEvents:
    async def test_context_does_not_log_any_read_event(self) -> None:
        """The falsifiable core of this suite: a delivered grounding entry must leave the
        read-events table untouched. Seen to fail (reverted, byte-exact) when a grounding write
        was temporarily restored — see the build verdict for that evidence."""
        engine, adapter, embedder, _clock = await _engine()
        fact = _direct_fact(content="hot fact", lifecycle="hot", audience_hint=("decider",))
        await adapter.upsert(fact, embedder.peek(fact.content))

        await engine.context("proj-a", 10_000, role="decider")

        assert adapter._read_events == []

    async def test_context_never_touches_reinforcement_state(self) -> None:
        engine, adapter, embedder, clock = await _engine()
        fact = _direct_fact(content="hot fact", lifecycle="hot", audience_hint=("decider",))
        await adapter.upsert(fact, embedder.peek(fact.content))
        before = await adapter.get_by_id(fact.id)

        clock.advance(timedelta(days=1))
        await engine.context("proj-a", 10_000, role="decider")

        after = await adapter.get_by_id(fact.id)
        assert after.recall_count == before.recall_count
        assert after.last_referenced_at == before.last_referenced_at
        assert after.updated_at == before.updated_at
        assert await adapter.query_events(target_id=fact.id) == []


# ---------------------------------------------------------------------------
# recall — recall()
# ---------------------------------------------------------------------------


class TestRecallLogsReadEvents:
    async def test_hit_logs_recall_event(self) -> None:
        engine, adapter, embedder, _clock = await _engine()
        fact = _direct_fact(content="a very specific recall target")
        await adapter.upsert(fact, embedder.peek(fact.content))

        result = await engine.recall(fact.content, scope="proj-a")
        assert any(h.id == fact.id for h in result.hits)

        events = _read_events_for(adapter, fact.id)
        assert len(events) == 1
        assert events[0].source == "recall"

    async def test_no_hits_logs_nothing(self) -> None:
        engine, adapter, _embedder, _clock = await _engine()
        result = await engine.recall("nothing in this empty store", scope="proj-a")
        assert result.hits == []
        assert adapter._read_events == []

    async def test_logging_failure_never_fails_recall(self) -> None:
        raising_adapter = _RaisingReadEventAdapter(embedding_model="fake/nomic", dim=_DIM)
        engine, adapter, embedder, _clock = await _engine(adapter=raising_adapter)
        fact = _direct_fact(content="a very specific recall target")
        await adapter.upsert(fact, embedder.peek(fact.content))

        result = await engine.recall(fact.content, scope="proj-a")
        assert any(h.id == fact.id for h in result.hits)


# ---------------------------------------------------------------------------
# decision-view — decision()
# ---------------------------------------------------------------------------


class TestDecisionViewLogsReadEvents:
    async def test_decision_logs_decision_view_event(self) -> None:
        engine, adapter, embedder, _clock = await _engine()
        decision = _direct_decision(content="a decision")
        await adapter.upsert(decision, embedder.peek(decision.content))

        view = await engine.decision(decision.id)
        assert view.id == decision.id

        events = _read_events_for(adapter, decision.id)
        assert len(events) == 1
        assert events[0].source == "decision-view"

    async def test_unknown_id_logs_nothing(self) -> None:
        engine, adapter, _embedder, _clock = await _engine()
        with pytest.raises(ToolError):
            await engine.decision(new_id())
        assert adapter._read_events == []

    async def test_fact_id_logs_nothing(self) -> None:
        engine, adapter, embedder, _clock = await _engine()
        fact = _direct_fact(content="not a decision")
        await adapter.upsert(fact, embedder.peek(fact.content))

        with pytest.raises(ToolError):
            await engine.decision(fact.id)
        assert _read_events_for(adapter, fact.id) == []


# ---------------------------------------------------------------------------
# Failure observability — a broken ``append_read_event``
# must be detectable by SOMETHING other than grepping logs by hand: this
# test IS that something.
# ---------------------------------------------------------------------------


async def test_logging_failure_is_observable_via_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A regression that silently stops emitting the WARNING (or narrows the
    ``except`` clause so a real failure escapes uncaught) goes red here —
    closing the "detectable by nothing but grepping logs" gap named in round
    2, without inventing a new counter/doctor surface for a single
    audit-trail append."""
    raising_adapter = _RaisingReadEventAdapter(embedding_model="fake/nomic", dim=_DIM)
    engine, adapter, embedder, _clock = await _engine(adapter=raising_adapter)
    fact = _direct_fact(content="recall warning target")
    await adapter.upsert(fact, embedder.peek(fact.content))

    with caplog.at_level("WARNING", logger="tabularium"):
        result = await engine.recall(fact.content, scope="proj-a")
    assert any(h.id == fact.id for h in result.hits)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("read event" in r.getMessage() for r in warnings), caplog.text


class TestLogReadEventsDoesNotCallUpkeep:
    async def test_logging_a_hit_never_calls_upkeep_or_checks_debt(self) -> None:
        """An earlier draft wired ``maybe_upkeep`` into ``log_read_events`` and a
        benchmark measuring it through the real write path caught a severe regression (see
        ``engine.read_events``'s own "Maintenance" docstring section) before it shipped — this
        pins the reverted, safe state so a future re-introduction is deliberate, not silent.
        """

        class _CountingMaintenanceCalls(MemoryAdapter):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]
                self.maintenance_debt_calls = 0
                self.upkeep_calls = 0

            async def maintenance_debt(self) -> MaintenanceDebt:
                self.maintenance_debt_calls += 1
                return await super().maintenance_debt()

            async def upkeep(self) -> None:
                self.upkeep_calls += 1
                return await super().upkeep()

        counting_adapter = _CountingMaintenanceCalls(embedding_model="fake/nomic", dim=_DIM)
        engine, adapter, embedder, _clock = await _engine(adapter=counting_adapter)
        fact = _direct_fact(content="no maintenance check target")
        await adapter.upsert(fact, embedder.peek(fact.content))

        result = await engine.recall(fact.content, scope="proj-a")
        assert any(h.id == fact.id for h in result.hits)
        assert counting_adapter.maintenance_debt_calls == 0
        assert counting_adapter.upkeep_calls == 0
