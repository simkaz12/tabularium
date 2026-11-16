"""The verdict log (A2 verdict-log piece) — one test per Observable behavior example plus the
crash-order/reconcile/override/tuning properties the piece's own Constraints name.

Hermetic lane only — ``MemoryAdapter`` + a table-driven embedder (lifted from
``test_auto_exits.py``) so a near-duplicate pair reliably survives the serial probe at a
known cosine, regardless of hash luck.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import HypomnemaId, Scope
from tabularium.engine import Tabularium
from tabularium.engine._errors import ToolError
from tabularium.engine.assessor import (
    AssessorCandidate,
    ContradictionVerdict,
    PanelDisagreement,
    RelationVerdict,
    Unavailable,
    Verdict,
)
from tabularium.engine.cleaner_writes import resolve_split_keep_write, resolve_split_shelve_write
from tabularium.engine.rubric import load_rubric
from tabularium.engine.verdicts import (
    build_cleaner_pending_row,
    build_cleaner_resolved_row,
    caller_exit_matches_pick,
    reconcile_pending_verdicts,
    restamp_cleaner_row,
    verdict_from_cleaner_row,
)
from tabularium.results import BounceResolution
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import RecordReview
from tests.support.doubles import TestClock
from tests.support.fake_assessor import EchoAssessor
from tests.support.faults import FaultInjectingAdapter, as_adapter

_DIM = 16
_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)

_EXISTING = "the quick brown fox jumps over the lazy dog"
_INCOMING = "the quick brown fox jumps over the lazy hound"
_SURVIVOR_B = "a second near-duplicate about the same lazy dog"
_SURVIVOR_C = "a third near-duplicate about the same lazy dog"


def _angle_vec(angle_deg: float, dim: int = _DIM) -> list[float]:
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


class _TableEmbedder:
    dim: int = _DIM
    model: str = "test/verdict-log"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


_DEFAULT_TABLE: dict[str, list[float]] = {
    _EXISTING: _angle_vec(0),
    _INCOMING: _angle_vec(5),
    _SURVIVOR_B: _angle_vec(10),
    _SURVIVOR_C: _angle_vec(15),
}


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {
        "candidate_k": 1000, "gray_floor": 0.80, "decision_gray_floor": 0.70,
    }
    defaults.update(kwargs)
    return EngineConfig(**defaults)


async def _force_distinct_save(engine: Tabularium, content: str, *, scope: str) -> Any:
    """Save ``content``, resolving ``confirm_distinct`` if it bounces against an
    already-seeded near-duplicate — for building a multi-survivor fixture, where
    the seed records themselves must land regardless of their OWN mutual
    similarity."""
    result = await engine.save(content, scope=scope)
    if result.outcome == "bounced":
        assert result.bounce is not None
        result = await engine.save(
            content, scope=scope,
            resolve=BounceResolution(
                candidate_id=result.bounce.candidate_id, action="confirm_distinct",
                bounce_id=result.bounce.bounce_id,
            ),
        )
    assert result.hypomnema is not None
    return result


async def _build_engine(
    table: dict[str, list[float]] | None = None, *, assessor: Any, **cfg_kwargs: Any
) -> tuple[Tabularium, MemoryAdapter]:
    embedder = _TableEmbedder(table if table is not None else dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    cfg = _make_config(**cfg_kwargs)
    engine = Tabularium(adapter, embedder, clock, cfg, assessor=assessor)
    await engine.initialize()
    await adapter.create_scope(Scope("p"))
    return engine, adapter


class _PartialTimeoutAssessor:
    """Wraps ``EchoAssessor`` but raises a bare ``TimeoutError`` for one named
    candidate id — ``engine.admission.ask`` maps that to ``Unavailable(reason=
    "timeout")`` exactly as a real per-caller deadline would."""

    def __init__(self, timeout_id: HypomnemaId, label: str) -> None:
        self._timeout_id = timeout_id
        self._inner = EchoAssessor(label)  # type: ignore[arg-type]

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        if HypomnemaId(uuid.UUID(candidate.id)) == self._timeout_id:
            raise TimeoutError("injected")
        return await self._inner.relation(incoming, candidate, similarity=similarity)

    async def kind(self, text: str, declared_kind: str) -> Any:
        return await self._inner.kind(text, declared_kind)

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Any:
        return await self._inner.worth(text)

    async def perishable(self, text: str) -> Any:
        return await self._inner.perishable(text)

    async def breadth(self, text: str, source_scope: str) -> Any:
        return await self._inner.breadth(text, source_scope)

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Any:
        return await self._inner.scope(text, labels)

    async def relevance(self, text: str) -> Any:
        return await self._inner.relevance(text)

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        return await self._inner.contradiction(existing, incoming, panel=panel)


class _WorldMovingAssessor:
    """Buries ``target_id`` as a side effect of ``relation`` — simulating a
    concurrent write landing during the Assessor call, which runs OUTSIDE the
    lock (D12). Mirrors ``test_auto_exits.py``'s own fixture of the same name."""

    def __init__(self, adapter: MemoryAdapter, target_id: HypomnemaId, label: str) -> None:
        self._adapter = adapter
        self._target_id = target_id
        self._inner = EchoAssessor(label)  # type: ignore[arg-type]

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        await self._adapter.bury(self._target_id, reason="world moved")
        return await self._inner.relation(incoming, candidate, similarity=similarity)

    async def kind(self, text: str, declared_kind: str) -> Any:
        return await self._inner.kind(text, declared_kind)

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Any:
        return await self._inner.worth(text)

    async def perishable(self, text: str) -> Any:
        return await self._inner.perishable(text)

    async def breadth(self, text: str, source_scope: str) -> Any:
        return await self._inner.breadth(text, source_scope)

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Any:
        return await self._inner.scope(text, labels)

    async def relevance(self, text: str) -> Any:
        return await self._inner.relevance(text)

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        return await self._inner.contradiction(existing, incoming, panel=panel)


class _ConcurrentDuplicateAssessor:
    """Saves ``incoming`` a second time as a side effect of the first ``relation``
    call — simulating a concurrent, identical save winning admission while this
    one is judged. Mirrors ``test_auto_exits.py``'s own fixture of the same
    name."""

    def __init__(self, label: str) -> None:
        self.engine: Tabularium | None = None
        self._inner = EchoAssessor(label)  # type: ignore[arg-type]
        self._triggered = False

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        if not self._triggered:
            self._triggered = True
            assert self.engine is not None
            duplicate = await self.engine.save(incoming, scope="p")
            assert duplicate.hypomnema is not None
        return await self._inner.relation(incoming, candidate, similarity=similarity)

    async def kind(self, text: str, declared_kind: str) -> Any:
        return await self._inner.kind(text, declared_kind)

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Any:
        return await self._inner.worth(text)

    async def perishable(self, text: str) -> Any:
        return await self._inner.perishable(text)

    async def breadth(self, text: str, source_scope: str) -> Any:
        return await self._inner.breadth(text, source_scope)

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Any:
        return await self._inner.scope(text, labels)

    async def relevance(self, text: str) -> Any:
        return await self._inner.relevance(text)

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        return await self._inner.contradiction(existing, incoming, panel=panel)


# ---------------------------------------------------------------------------
# Scope 2 — bounce path, bounce_id, caller-resolve stamping
# ---------------------------------------------------------------------------


async def test_no_adapter_bounce_carries_bounce_id_writes_zero_rows() -> None:
    engine, adapter = await _build_engine(assessor=None)
    await engine.save(_EXISTING, scope="p")

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.bounce_id
    rows = list(adapter._verdicts.values())
    assert rows == []


async def test_switch_off_one_survivor_restatement_writes_unanswered_row() -> None:
    engine, adapter = await _build_engine(assessor=EchoAssessor("repetition"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    bounce_id = result.bounce.bounce_id
    rows = await adapter.verdicts_by_bounce(bounce_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.caller == "save"
    assert row.method == "relation"
    assert row.candidate_id == str(existing.hypomnema.id)
    assert row.probe_rank == 0
    assert row.similarity is not None and row.similarity > 0.9
    assert row.pick == "restatement"
    assert row.odds_json is not None
    assert row.confidence is not None
    assert row.status == "unanswered"
    assert row.action is None
    for exit_ in result.bounce.exits:
        assert bounce_id in exit_.how


async def test_caller_replays_reinforce_how_stamps_row_acted() -> None:
    engine, adapter = await _build_engine(assessor=EchoAssessor("repetition"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    bounced = await engine.save(_INCOMING, scope="p")
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id

    result = await engine.save(
        _INCOMING, scope="p",
        resolve=BounceResolution(
            candidate_id=existing_id, action="reinforce", bounce_id=bounce_id
        ),
    )
    assert result.outcome == "reinforced"

    rows = await adapter.verdicts_by_bounce(bounce_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "acted"
    assert row.action == "reinforce"
    assert row.actor == "caller"
    assert row.matched is True
    assert row.effect_ref is not None and row.effect_ref.startswith(str(existing_id))
    assert row.incoming_text == _INCOMING


async def test_caller_confirm_distinct_against_restatement_pick_overrides() -> None:
    engine, adapter = await _build_engine(assessor=EchoAssessor("repetition"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    bounced = await engine.save(_INCOMING, scope="p")
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id

    result = await engine.save(
        _INCOMING, scope="p",
        resolve=BounceResolution(
            candidate_id=existing_id, action="confirm_distinct", bounce_id=bounce_id
        ),
    )
    assert result.outcome == "saved"
    assert result.hypomnema is not None

    rows = await adapter.verdicts_by_bounce(bounce_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "acted"
    assert row.action == "confirm_distinct"
    assert row.matched is False
    assert row.override == "caller_against_pick"
    assert row.new_record_id == str(result.hypomnema.id)
    assert row.incoming_text is None


async def test_follow_up_save_with_no_bounce_id_leaves_rows_unanswered() -> None:
    engine, adapter = await _build_engine(assessor=EchoAssessor("repetition"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    bounced = await engine.save(_INCOMING, scope="p")
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id

    result = await engine.save(
        _INCOMING, scope="p",
        resolve=BounceResolution(candidate_id=existing_id, action="reinforce"),
    )
    assert result.outcome == "reinforced"

    rows = await adapter.verdicts_by_bounce(bounce_id)
    assert len(rows) == 1
    assert rows[0].status == "unanswered"


async def test_unknown_bounce_id_behaves_as_today_stamps_nothing() -> None:
    engine, adapter = await _build_engine(assessor=EchoAssessor("repetition"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    await engine.save(_INCOMING, scope="p")

    result = await engine.save(
        _INCOMING, scope="p",
        resolve=BounceResolution(
            candidate_id=existing_id, action="reinforce", bounce_id="not-a-real-bounce",
        ),
    )
    assert result.outcome == "reinforced"

    rows = list(adapter._verdicts.values())
    assert all(row.status == "unanswered" for row in rows)


async def test_timeout_on_survivor_0_verdict_on_survivor_1_writes_two_rows() -> None:
    table = {
        _EXISTING: _angle_vec(0),
        _SURVIVOR_B: _angle_vec(3),
        _INCOMING: _angle_vec(5),
    }
    engine, adapter = await _build_engine(table, assessor=None)
    a = await engine.save(_EXISTING, scope="p")
    assert a.hypomnema is not None
    await _force_distinct_save(engine, _SURVIVOR_B, scope="p")

    # A fresh engine over the SAME adapter/store, this time with an assessor
    # that times out on ``a`` alone — the two prior saves stay visible.
    assessor = _PartialTimeoutAssessor(a.hypomnema.id, "repetition")
    embedder = _TableEmbedder(table)
    engine2 = Tabularium(
        adapter, embedder, TestClock(_EPOCH), _make_config(), assessor=assessor
    )
    await engine2.initialize()

    result = await engine2.save(_INCOMING, scope="p")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    bounce_id = result.bounce.bounce_id
    rows = await adapter.verdicts_by_bounce(bounce_id)
    assert len(rows) == 2
    by_candidate = {row.candidate_id: row for row in rows}
    timed_out = by_candidate[str(a.hypomnema.id)]
    assert timed_out.status == "unanswered"
    assert timed_out.unavailable_reason == "timeout"
    assert timed_out.odds_json is None
    assert timed_out.pick is None
    judged_row = next(row for row in rows if row.candidate_id != str(a.hypomnema.id))
    assert judged_row.pick == "restatement"


async def test_three_survivors_caller_supersede_top_stamps_siblings() -> None:
    engine, adapter = await _build_engine(assessor=EchoAssessor("repetition"))
    a = await engine.save(_EXISTING, scope="p")
    assert a.hypomnema is not None
    b = await _force_distinct_save(engine, _SURVIVOR_B, scope="p")
    c = await _force_distinct_save(engine, _SURVIVOR_C, scope="p")

    bounced = await engine.save(_INCOMING, scope="p")
    assert bounced.outcome == "bounced"
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id
    candidate_id = bounced.bounce.candidate_id
    rows_before = await adapter.verdicts_by_bounce(bounce_id)
    assert len(rows_before) == 3

    # Bury the two siblings so the replay's own probe lands the supersede
    # cleanly on the named candidate alone (they stay in the verdict log).
    other_ids = {a.hypomnema.id, b.hypomnema.id, c.hypomnema.id} - {candidate_id}
    for other_id in other_ids:
        await engine.bury(other_id, reason="test cleanup")

    result = await engine.save(
        _INCOMING, scope="p", supersedes=[candidate_id], reason="changed",
        bounce_id=bounce_id,
    )
    assert result.outcome in ("saved", "reconciled")

    rows_after = await adapter.verdicts_by_bounce(bounce_id)
    by_candidate = {row.candidate_id: row for row in rows_after}
    assert by_candidate[str(candidate_id)].status == "acted"
    assert by_candidate[str(candidate_id)].action == "supersede"
    for other_id in other_ids:
        assert by_candidate[str(other_id)].status == "sibling_acted"


# ---------------------------------------------------------------------------
# Scope 3 — automatic exits, crash order, startup reconcile
# ---------------------------------------------------------------------------


async def test_automatic_supersede_leaves_acted_row() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "saved"
    assert result.hypomnema is not None

    rows = await adapter.verdicts_for_record(existing_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "acted"
    assert row.actor == "engine"
    assert row.action == "supersede"
    assert row.matched is True
    assert row.bounce_id is None
    assert row.new_record_id == str(result.hypomnema.id)
    predecessor = await adapter.get_by_id(existing_id)
    assert row.effect_ref == predecessor.supersession_edge_id


async def test_automatic_reinforce_leaves_acted_row_with_effect_ref() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("repetition"), auto_exit_restatement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "reinforced"

    rows = await adapter.verdicts_for_record(existing_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "acted"
    assert row.action == "reinforce"
    assert row.actor == "engine"
    assert row.effect_ref is not None and row.effect_ref.startswith(str(existing_id))
    assert row.bounce_id is None


async def test_automatic_follow_up_write_new_leaves_acted_row() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("addition"), auto_exit_follow_up=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "saved"
    assert result.hypomnema is not None

    rows = await adapter.verdicts_for_record(existing_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "acted"
    assert row.action == "write_new"
    assert row.actor == "engine"
    assert row.new_record_id == str(result.hypomnema.id)


async def test_automatic_invalidation_leaves_row_with_reason_class() -> None:
    table = {
        _EXISTING: _angle_vec(0),
        _INCOMING: _angle_vec(5),
    }
    engine, adapter = await _build_engine(
        table, assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    decision = await engine.save(_EXISTING, scope="p", kind="decision")
    assert decision.hypomnema is not None
    decision_id = decision.hypomnema.id

    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "saved"
    assert result.hypomnema is not None
    fact_id = result.hypomnema.id

    rows = await adapter.verdicts_for_record(decision_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "acted"
    assert row.action == "invalidate"
    assert row.effect_ref == str(decision_id)
    assert row.new_record_id == str(fact_id)
    assert row.reason_class == "correction"


async def test_automatic_reinforce_retains_incoming_text() -> None:
    """D4: the reinforced side loses its own words to the merge — the row is the
    only place they survive. Guards against automatic reinforcement never retaining discarded
    text."""
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("repetition"), auto_exit_restatement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "reinforced"

    rows = await adapter.verdicts_for_record(existing_id)
    assert len(rows) == 1
    assert rows[0].incoming_text == _INCOMING


class _TopOnlyRestatementAssessor:
    """``restatement`` for the FIRST candidate ``relation`` is asked about, ``distinct``
    for every other (A4 bounce-judged piece): a single shared material for every
    survivor (the pre-A4 ``EchoAssessor("repetition")`` fixture this replaces) now
    reads as a competing picture (decision D16) and bounces instead of reinforcing,
    since every sibling ALSO carries ``restatement``. ``distinct`` never disturbs a
    clean picture (Observable), so this keeps the ORIGINAL "top reinforces, siblings
    get their own unanswered row" shape this test exists to prove. Scripted by CALL
    ORDER, not candidate id: ``relation_verdicts`` builds one call per probe survivor
    in probe order (D3/D4docstring — "verdicts in probe order") and this hermetic fake
    never truly suspends, so the first call this object receives is always the top
    survivor's own, whichever id that turns out to be."""

    def __init__(self) -> None:
        self._top = EchoAssessor("repetition")
        self._other = EchoAssessor("unrelated")
        self._calls = 0

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        self._calls += 1
        inner = self._top if self._calls == 1 else self._other
        return await inner.relation(incoming, candidate, similarity=similarity)

    async def kind(self, text: str, declared_kind: str) -> Any:
        return await self._top.kind(text, declared_kind)

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Any:
        return await self._top.worth(text)

    async def perishable(self, text: str) -> Any:
        return await self._top.perishable(text)

    async def breadth(self, text: str, source_scope: str) -> Any:
        return await self._top.breadth(text, source_scope)

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Any:
        return await self._top.scope(text, labels)

    async def relevance(self, text: str) -> Any:
        return await self._top.relevance(text)

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        return await self._top.contradiction(existing, incoming, panel=panel)


async def test_automatic_reinforce_with_three_survivors_writes_sibling_rows() -> None:
    """D1: every verdict the Assessor reached gets a row, not only the one the
    automatic exit acted on. Guards against automatic exits discarding secondary survivor verdicts.
    Seeded with no
    assessor (auto-exit off) so the three survivors land distinctly first —
    a real assessor with ``auto_exit_restatement`` on would auto-reinforce
    ``b``/``c`` into ``a`` during seeding instead of writing three records."""
    engine, adapter = await _build_engine(assessor=None)
    a = await engine.save(_EXISTING, scope="p")
    assert a.hypomnema is not None
    b = await _force_distinct_save(engine, _SURVIVOR_B, scope="p")
    c = await _force_distinct_save(engine, _SURVIVOR_C, scope="p")

    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    engine2 = Tabularium(
        adapter, embedder, TestClock(_EPOCH),
        _make_config(auto_exit_restatement=True),
        assessor=_TopOnlyRestatementAssessor(),
    )
    await engine2.initialize()

    result = await engine2.save(_INCOMING, scope="p")
    assert result.outcome == "reinforced"
    assert result.reinforced_id is not None
    assert b.hypomnema is not None
    assert c.hypomnema is not None
    all_ids = {a.hypomnema.id, b.hypomnema.id, c.hypomnema.id}
    assert result.reinforced_id in all_ids
    sibling_ids = all_ids - {result.reinforced_id}

    top_rows = await adapter.verdicts_for_record(result.reinforced_id)
    assert len(top_rows) == 1
    assert top_rows[0].status == "acted"
    assert top_rows[0].bounce_id is None

    sibling_ranks: set[int | None] = set()
    for sibling_id in sibling_ids:
        sibling_rows = await adapter.verdicts_for_record(sibling_id)
        assert len(sibling_rows) == 1
        assert sibling_rows[0].status == "unanswered"
        assert sibling_rows[0].bounce_id is None
        assert sibling_rows[0].pick == "distinct"
        sibling_ranks.add(sibling_rows[0].probe_rank)

    # Automatic secondary rows must not restart probe ranks at zero: the two siblings must keep
    # their ORIGINAL
    # probe positions (1, 2), not both collide on 0.
    assert sibling_ranks == {1, 2}
    assert top_rows[0].probe_rank == 0


async def test_reinforce_replay_against_invalidated_candidate_leaves_row_unanswered() -> None:
    """Reinforce fallbacks leave
    successful writes recorded as failed: the candidate went invalid between
    the bounce and the replay, so the save reconciles instead of reinforcing —
    the caller's row must stay honestly unanswered, never falsely acted OR,
    after a future reconcile pass, falsely failed."""
    engine, adapter = await _build_engine(assessor=EchoAssessor("repetition"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    bounced = await engine.save(_INCOMING, scope="p")
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id

    await engine.set_validity(existing_id, valid=False, reason="turned out wrong")

    result = await engine.save(
        _INCOMING, scope="p",
        resolve=BounceResolution(
            candidate_id=existing_id, action="reinforce", bounce_id=bounce_id,
        ),
    )
    assert result.outcome != "reinforced"
    assert result.hypomnema is not None

    [row] = await adapter.verdicts_by_bounce(bounce_id)
    assert row.status == "unanswered"
    assert row.incoming_text is None
    assert row.new_record_id is None


async def test_world_moved_redispatch_writes_orphaned_verdict_rows() -> None:
    """Concurrent redispatch drops
    completed Assessor batches: the candidate is gone by the time the arm
    re-takes the lock, so no exit will ever act on the verdict just obtained —
    it must still land as a row (D1), not vanish with the abandoned dispatch."""
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    cfg = _make_config(auto_exit_restatement=True)
    existing_engine = Tabularium(adapter, embedder, clock, cfg)
    await existing_engine.initialize()
    await adapter.create_scope(Scope("p"))
    existing = await existing_engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    assessor = _WorldMovingAssessor(adapter, existing_id, "repetition")
    engine = Tabularium(adapter, embedder, clock, cfg, assessor=assessor)

    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "saved"
    assert result.hypomnema is not None
    assert result.hypomnema.id != existing_id

    rows = await adapter.verdicts_for_record(existing_id)
    assert len(rows) == 1
    assert rows[0].status == "unanswered"
    assert rows[0].pick == "restatement"


async def test_write_new_admission_race_writes_orphaned_verdict_rows() -> None:
    """Concurrent redispatch must not drop completed Assessor batches (the WriteNew recheck named
    alongside the
    freshness one): a concurrent identical save wins admission while this one
    is judged — WriteNew's own verdict must still land as a row, not vanish
    because ``admission_hit`` intercepted the write."""
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    cfg = _make_config(auto_exit_follow_up=True)
    assessor = _ConcurrentDuplicateAssessor("addition")
    engine = Tabularium(adapter, embedder, clock, cfg, assessor=assessor)
    assessor.engine = engine
    await engine.initialize()
    await adapter.create_scope(Scope("p"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "already_saved"

    # Two rows land against ``existing_id``: the inner concurrent duplicate's
    # OWN successful WriteNew dispatch (``acted``), and the outer dispatch's
    # verdict, orphaned by the admission race this test is about (the fix
    # under test — must be ``unanswered``, never silently dropped).
    rows = await adapter.verdicts_for_record(existing_id)
    unanswered = [row for row in rows if row.status == "unanswered"]
    assert len(unanswered) == 1
    assert unanswered[0].pick == "follow_up"


async def test_fault_between_pending_and_stamp_reconcile_stamps_acted() -> None:
    inner = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_replacement=True)
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=EchoAssessor("correction")
    )
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    wrapper.fault(on="update_verdicts", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with pytest.raises(RuntimeError):
        await engine.save(_INCOMING, scope="p")

    pending = await inner.pending_verdicts()
    assert len(pending) == 1
    assert pending[0].candidate_id == str(existing_id)

    wrapper.fault(on="update_verdicts", call=999, exc=RuntimeError("never"))
    await engine.initialize()

    acted = await inner.verdicts_for_record(existing_id)
    assert len(acted) == 1
    assert acted[0].status == "acted"


async def test_caller_exit_fault_between_pending_and_stamp_reconciles() -> None:
    """The same crash-order property, this time on a CALLER exit (``begin_caller_exit``
    stamps ``action`` at the pending write too, not only ``build_pending_row``'s
    automatic-exit path) — a torn caller reinforce reconciles from ``pending`` to
    ``acted`` exactly like a torn automatic one."""
    inner = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    cfg = _make_config()
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=EchoAssessor("repetition")
    )
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id
    bounced = await engine.save(_INCOMING, scope="p")
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id

    wrapper.fault(on="update_verdicts", call=2, exc=RuntimeError("injected"))
    wrapper.arm()
    with pytest.raises(RuntimeError):
        await engine.save(
            _INCOMING, scope="p",
            resolve=BounceResolution(
                candidate_id=existing_id, action="reinforce", bounce_id=bounce_id,
            ),
        )

    pending = await inner.pending_verdicts()
    assert len(pending) == 1
    assert pending[0].action == "reinforce"
    assert pending[0].candidate_id == str(existing_id)

    wrapper.fault(on="update_verdicts", call=999, exc=RuntimeError("never"))
    await engine.initialize()

    stamped = await inner.verdicts_by_bounce(bounce_id)
    resolved = next(row for row in stamped if row.status != "unanswered")
    assert resolved.status == "acted"


async def test_replace_raising_stamps_failed_at_once() -> None:
    inner = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_replacement=True)
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=EchoAssessor("correction")
    )
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    wrapper.fault(on="replace", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with pytest.raises(ToolError):
        await engine.save(_INCOMING, scope="p")

    rows = await inner.verdicts_for_record(existing_id)
    assert len(rows) == 1
    assert rows[0].status == "failed"


async def test_pending_row_whose_effect_never_landed_reconciles_to_failed() -> None:
    from tabularium.engine.verdicts import build_pending_row

    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    fact = await _seed_fact(adapter, content="orphaned pending target")
    verdict = RelationVerdict(
        method="relation", pick="replacement", odds={"replacement": 1.0}, confidence=1.0,
        trace=_placeholder_trace(), reason_class="correction",
    )
    row = build_pending_row(
        candidate_id=fact.id, similarity=0.9, verdict=verdict, action="supersede",
        effect_ref="an-edge-id-that-was-never-drawn", new_record_id=str(uuid.uuid4()),
        now=_EPOCH,
    )
    await adapter.append_verdicts([row])

    acted, failed = await reconcile_pending_verdicts(adapter, now=_EPOCH)

    assert acted == 0
    assert failed == 1
    got = await adapter.pending_verdicts()
    assert got == []
    [stamped] = await adapter.verdicts_for_record(fact.id)
    assert stamped.status == "failed"


async def test_split_keep_torn_write_reconciles_to_acted(monkeypatch: pytest.MonkeyPatch) -> None:
    """D14's crash order for an Assessor-caused write, at the relevance split-keep site: the
    pending row lands, the keep review lands, the stamp raises -- a fresh reconcile pass must
    still find the review and flip the row to `acted`, not leave it `pending` forever."""
    from tabularium.engine import cleaner_writes as cleaner_writes_mod
    from tabularium.engine.relevance import run_relevance_axis
    from tabularium.engine.verdicts import stamp_acted_many as verdicts_stamp_acted_many
    from tests.support.doubles import FakeJudgeAdapter

    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    clock = TestClock(_EPOCH)
    fact = await _seed_fact(adapter, content="split-keep record whose stamp fails")

    assessor = EchoAssessor(
        "unrelated",
        relevance_script={"split-keep record whose stamp fails": _keep_lean_verdict()},
    )
    judge_a = FakeJudgeAdapter(identity="a", response=_relevance_payload("shelve-clear"))
    judge_b = FakeJudgeAdapter(identity="b", response=_relevance_payload("keep-clear"))

    real_stamp_acted_many = verdicts_stamp_acted_many
    call_count = {"n": 0}

    async def _torn_stamp_acted_many(*args: Any, **kwargs: Any) -> None:
        call_count["n"] += 1
        raise RuntimeError("injected stamp failure")

    monkeypatch.setattr(cleaner_writes_mod, "stamp_acted_many", _torn_stamp_acted_many)

    with pytest.raises(RuntimeError, match="injected stamp failure"):
        await run_relevance_axis(
            scope=Scope("p"), since=_EPOCH - timedelta(days=1), adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )
    assert call_count["n"] == 1

    # The review really landed before the stamp raised.
    history = await adapter.list_reviews_for(target_id=fact.id)
    assert len(history) == 1
    assert history[0].band == "keep"

    pending = await adapter.pending_verdicts()
    assert len(pending) == 1
    assert pending[0].action == "split_keep"

    monkeypatch.setattr(cleaner_writes_mod, "stamp_acted_many", real_stamp_acted_many)
    acted, failed = await reconcile_pending_verdicts(adapter, now=_EPOCH)
    assert acted == 1
    assert failed == 0
    [stamped] = await adapter.verdicts_for_record(fact.id)
    assert stamped.status == "acted"


async def test_revisit_restamp_does_not_reconcile_against_a_stale_earlier_keep_review() -> None:
    """A revisit split-keep reuse
    (``restamp_cleaner_row``, D10) preserves the row's ORIGINAL ``created_at`` from the pre-pass,
    possibly from a much earlier pass. If this record already went through a DIFFERENT,
    already-completed ``split_keep`` resolution before (its own real ``band="keep"`` review sits
    at some earlier instant, still ``>= created_at``), a fresh restamp attempt that crashes
    before ITS OWN review lands must not be reconciled ``acted`` against that stale review."""
    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    fact = await _seed_fact(adapter, content="revisit reuse over a record with a real prior keep")

    t0 = _EPOCH
    t1 = _EPOCH + timedelta(days=10)
    t2 = _EPOCH + timedelta(days=40)

    original_row = build_cleaner_resolved_row(
        caller="cleaner_prepass", candidate_id=fact.id, verdict=_keep_lean_verdict(),
        action="panel_keep", actor="panel", matched=True, now=t0,
    )
    await adapter.append_verdicts([original_row])

    # The record's OWN earlier, fully-completed split_keep resolution -- a genuine `band="keep"`
    # review already sits in its history, well after `created_at` (t0).
    await adapter.append_review(
        RecordReview(
            target_id=fact.id, scope=Scope("p"), reviewed_at=t1, next_review_due_at=t2,
            outcome="assessor_resolved", band="keep", grades=(), rubric_version="test",
        )
    )

    # A LATER revisit split reuses the SAME row (D10) for a brand-new resolution attempt at t2,
    # then crashes before its OWN resolving review ever lands -- nothing else appended after this.
    restamped = restamp_cleaner_row(
        original_row, action="split_keep", actor="engine", matched=True, now=t2
    )
    await adapter.update_verdicts([restamped])

    acted, failed = await reconcile_pending_verdicts(adapter, now=t2 + timedelta(days=1))

    # Pre-fix: `_effect_landed` checked only `reviewed_at >= created_at` (t0), so the STALE t1
    # review would satisfy it and this row would be wrongly reconciled `acted`.
    assert failed == 1
    assert acted == 0
    [row] = await adapter.verdicts_for_record(fact.id)
    assert row.status == "failed"


async def test_revisit_split_shelve_refuses_a_record_superseded_mid_pass() -> None:
    """The revisit axis's own split-shelve
    write must re-check validity/supersession under the shelve's own lock, exactly like its
    sibling unanimous-shelve write already does -- `set_shelved`'s DEFAULT checks (lifecycle
    only) would otherwise silently shelve a Fact another writer just superseded."""
    from tabularium.domain import Fact, new_id

    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    await adapter.create_scope(Scope("p"))
    successor = Fact(
        id=new_id(), type="fact", content="the successor", scope=Scope("p"),
        lifecycle="active", validity="valid", created_at=_EPOCH, updated_at=_EPOCH,
    )
    await adapter.upsert(successor, [1.0] * _DIM)
    fact = await _seed_fact(adapter, content="record superseded mid-pass")
    # Simulates another writer superseding this record WHILE the revisit judges/Assessor were
    # still running -- `lifecycle` stays `"active"`, only `validity`/`superseded_by` change,
    # exactly what `set_shelved`'s DEFAULT (lifecycle-only) check would miss.
    from tabularium.engine.dispositions import supersede as _real_supersede

    await _real_supersede(
        successor_id=successor.id, predecessor_ids=[fact.id], reason="changed", note=None,
        adapter=adapter, clock=TestClock(_EPOCH), lock=asyncio.Lock(),
    )

    pending = build_cleaner_pending_row(
        caller="cleaner_prepass", candidate_id=fact.id, new_record_id=None,
        verdict=_keep_lean_verdict(), action="split_shelve", effect_ref=None, now=_EPOCH,
    )

    landed = await resolve_split_shelve_write(
        pending_row=pending,
        write_pending=adapter.append_verdicts,
        record=fact,
        reason="Cleaner revisit split: test",
        adapter=adapter,
        clock=TestClock(_EPOCH),
        lock=asyncio.Lock(),
        require_valid_and_unsuperseded=True,
    )

    assert landed is False
    fresh = await adapter.get_by_id(fact.id)
    assert fresh.lifecycle == "active"  # never shelved
    [row] = await adapter.verdicts_for_record(fact.id)
    assert row.status == "failed"


async def test_split_shelve_landed_row_is_findable_by_a_later_owner_unshelve_override() -> None:
    """``resolve_split_shelve_write``'s success path must ALSO carry the shelved
    record's own id as ``new_record_id`` -- same gap as ``panel_shelve``, different call site."""
    from tabularium.engine.dispositions import set_shelved

    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    fact = await _seed_fact(adapter, content="split-shelve record the owner will later un-shelve")

    pending = build_cleaner_pending_row(
        caller="cleaner_prepass", candidate_id=fact.id, new_record_id=None,
        verdict=_keep_lean_verdict(), action="split_shelve", effect_ref=None, now=_EPOCH,
    )

    landed = await resolve_split_shelve_write(
        pending_row=pending,
        write_pending=adapter.append_verdicts,
        record=fact,
        reason="Cleaner relevance split: test",
        adapter=adapter,
        clock=TestClock(_EPOCH),
        lock=asyncio.Lock(),
    )
    assert landed is True

    [row_before] = await adapter.verdicts_for_record(fact.id)
    assert row_before.action == "split_shelve"
    assert row_before.override is None  # the automatic write is never its own override

    await set_shelved(
        id=fact.id, shelved=False, reason="owner reversal",
        adapter=adapter, clock=TestClock(_EPOCH + timedelta(days=1)), lock=asyncio.Lock(),
    )

    [row_after] = await adapter.verdicts_for_record(fact.id)
    assert row_after.override == "set_shelved"


async def test_split_shelve_crash_before_final_stamp_still_reconciles_findable() -> None:
    """The shelved record's own id must be
    on the pending row from the moment it lands ``pending`` (pre-published, mirroring
    ``supersede``'s pre-minted ``edge_id``), not only added by the final ``stamp_acted`` --
    otherwise a crash between the real shelve landing and that final stamp leaves a fresh
    reconcile nothing to link, and a concurrent owner reversal racing in that exact window would
    find no row to stamp at all."""
    from tabularium.engine.dispositions import set_shelved

    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    fact = await _seed_fact(adapter, content="split-shelve record, crash before final stamp")

    pending = build_cleaner_pending_row(
        caller="cleaner_prepass", candidate_id=fact.id, new_record_id=fact.id,
        verdict=_keep_lean_verdict(), action="split_shelve", effect_ref=None, now=_EPOCH,
    )
    await adapter.append_verdicts([pending])

    # The real shelve lands (mirrors `resolve_split_shelve_write`'s own call), but the
    # process "crashes" before its own final `stamp_acted` ever runs -- the row stays `pending`.
    await set_shelved(
        id=fact.id, shelved=True, reason="Cleaner split-shelve", adapter=adapter,
        clock=TestClock(_EPOCH), lock=asyncio.Lock(), skip_override_stamp=True,
    )

    [row_pending] = await adapter.verdicts_for_record(fact.id)
    assert row_pending.status == "pending"
    assert row_pending.new_record_id == str(fact.id)  # already linked, before any crash

    acted, failed = await reconcile_pending_verdicts(adapter, now=_EPOCH + timedelta(days=1))
    assert acted == 1
    assert failed == 0

    # A later owner reversal must still find it.
    await set_shelved(
        id=fact.id, shelved=False, reason="owner reversal", adapter=adapter,
        clock=TestClock(_EPOCH + timedelta(days=2)), lock=asyncio.Lock(),
    )
    [row_final] = await adapter.verdicts_for_record(fact.id)
    assert row_final.status == "acted"
    assert row_final.override == "set_shelved"


async def test_split_shelve_concurrent_owner_reversal_survives_the_final_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolve_split_shelve_write``'s
    final ``stamp_acted`` call must not silently erase a concurrent owner reversal's own
    ``override`` stamp -- it re-reads the row fresh, under the SAME lock, immediately before
    stamping, rather than trusting the stale in-memory ``pending_row`` it captured before
    ``set_shelved`` ever ran."""
    from tabularium.engine import cleaner_writes as cleaner_writes_mod
    from tabularium.engine.dispositions import set_shelved as real_set_shelved

    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    fact = await _seed_fact(adapter, content="split-shelve record racing an owner reversal")

    async def _set_shelved_that_races_an_owner_reversal(**kwargs: Any) -> Any:
        result = await real_set_shelved(**kwargs)
        if kwargs.get("shelved") is True:
            # Simulates an owner's OWN unshelve landing between the automatic shelve committing
            # and `resolve_split_shelve_write` reacquiring the lock for its final stamp.
            await real_set_shelved(
                id=fact.id, shelved=False, reason="owner reversal (race)",
                adapter=kwargs["adapter"], clock=TestClock(_EPOCH + timedelta(days=1)),
                lock=asyncio.Lock(),
            )
        return result

    monkeypatch.setattr(
        cleaner_writes_mod, "set_shelved", _set_shelved_that_races_an_owner_reversal
    )

    pending = build_cleaner_pending_row(
        caller="cleaner_prepass", candidate_id=fact.id, new_record_id=fact.id,
        verdict=_keep_lean_verdict(), action="split_shelve", effect_ref=None, now=_EPOCH,
    )

    landed = await resolve_split_shelve_write(
        pending_row=pending,
        write_pending=adapter.append_verdicts,
        record=fact,
        reason="Cleaner relevance split: test",
        adapter=adapter,
        clock=TestClock(_EPOCH),
        lock=asyncio.Lock(),
    )
    assert landed is True

    fresh = await adapter.get_by_id(fact.id)
    assert fresh.lifecycle == "active"  # the racing reversal actually won

    [row] = await adapter.verdicts_for_record(fact.id)
    assert row.status == "acted"
    assert row.override == "set_shelved"  # NOT erased by the final stamp
    assert row.override_at is not None


async def test_reconcile_split_coexist_requires_both_participants_own_keep_review() -> None:
    """`split_coexist` resolves a PAIR, one
    resolving keep review per participant -- checking only `candidate_id` accepts a half-written
    resolution where the newcomer's own review never landed before a crash between the two
    participant appends."""
    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    candidate = await _seed_fact(adapter, content="split-coexist candidate, half-written")
    newcomer = await _seed_fact(adapter, content="split-coexist newcomer, own review missing")

    from tabularium.engine.assessor import Trace

    contradiction_verdict = ContradictionVerdict(
        pick="coexist", reason_class=None, odds={"coexist": 1.0}, confidence=1.0,
        trace=Trace(
            model="echo", wording_hash="echo", latency_ms=0,
            input_tokens=None, output_tokens=None, request_id=None,
        ),
    )
    pending = build_cleaner_pending_row(
        caller="cleaner_split", candidate_id=candidate.id, new_record_id=newcomer.id,
        verdict=contradiction_verdict, action="split_coexist", effect_ref=None, now=_EPOCH,
    )
    await adapter.append_verdicts([pending])

    # Only the CANDIDATE's own resolving keep review landed -- the newcomer's own append never
    # happened (simulates a crash between the two participant appends).
    await adapter.append_review(
        RecordReview(
            target_id=candidate.id, scope=Scope("p"), reviewed_at=_EPOCH,
            next_review_due_at=_EPOCH + timedelta(days=30), outcome="assessor_resolved",
            band="keep", grades=(), rubric_version="test",
        )
    )

    acted, failed = await reconcile_pending_verdicts(adapter, now=_EPOCH + timedelta(days=1))

    assert failed == 1
    assert acted == 0
    [row] = await adapter.verdicts_for_record(candidate.id)
    assert row.status == "failed"


async def test_panel_shelve_row_is_findable_by_a_later_owner_unshelve_override() -> None:
    """A Cleaner-caused ``panel_shelve``
    row carries neither ``effect_ref`` nor ``new_record_id`` by default, and ``stamp_override``
    looks a row up by those two fields only, never ``candidate_id`` -- an owner un-shelving the
    record later never reached the tuning query's own overridden count. Also proves the fix's
    own safety property: the automatic shelve itself is never mistaken for its own override."""
    from tabularium.engine.dispositions import set_shelved
    from tabularium.engine.relevance import run_relevance_axis
    from tests.support.doubles import FakeJudgeAdapter

    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    clock = TestClock(_EPOCH)
    fact = await _seed_fact(adapter, content="record the owner will later un-shelve")

    assessor = EchoAssessor(
        "unrelated",
        relevance_script={"record the owner will later un-shelve": _keep_lean_verdict()},
    )
    judge_a = FakeJudgeAdapter(identity="a", response=_relevance_payload("shelve-clear"))
    judge_b = FakeJudgeAdapter(identity="b", response=_relevance_payload("shelve-clear"))

    await run_relevance_axis(
        scope=Scope("p"), since=_EPOCH - timedelta(days=1), adapter=adapter, clock=clock,
        lock=asyncio.Lock(), cfg=EngineConfig(), rubric=load_rubric(),
        judge_a=judge_a, judge_b=judge_b, assessor=assessor,
    )

    [row_before] = await adapter.verdicts_for_record(fact.id)
    assert row_before.action == "panel_shelve"
    # The automatic shelve must never be mistaken for its own owner override.
    assert row_before.override is None
    fresh = await adapter.get_by_id(fact.id)
    assert fresh.lifecycle == "shelved"

    await set_shelved(
        id=fact.id, shelved=False, reason="owner reversal",
        adapter=adapter, clock=TestClock(_EPOCH + timedelta(days=1)), lock=asyncio.Lock(),
    )

    [row_after] = await adapter.verdicts_for_record(fact.id)
    assert row_after.override == "set_shelved"
    assert row_after.override_at is not None


async def test_panel_shelve_concurrent_owner_reversal_survives_the_final_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Panel_shelve override race: a unanimous-panel shelve used to call
    ``set_shelved`` and append this row (``build_cleaner_resolved_row``) in TWO separate lock
    holds -- a concurrent owner ``set_shelved(shelved=False)`` landing in that gap found no row to
    stamp its override onto. Mirrors ``test_split_shelve_concurrent_owner_reversal_survives_the_
    final_stamp``: the racing reversal is injected as a side effect of the SAME ``set_shelved``
    call ``resolve_panel_shelve_write`` itself makes, landing between the automatic shelve
    committing and the row's own final stamp."""
    from tabularium.engine import cleaner_writes as cleaner_writes_mod
    from tabularium.engine.dispositions import set_shelved as real_set_shelved
    from tabularium.engine.relevance import run_relevance_axis
    from tests.support.doubles import FakeJudgeAdapter

    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    clock = TestClock(_EPOCH)
    fact = await _seed_fact(adapter, content="panel shelve racing an owner reversal")

    async def _set_shelved_that_races_an_owner_reversal(**kwargs: Any) -> Any:
        result = await real_set_shelved(**kwargs)
        if kwargs.get("shelved") is True:
            # Simulates an owner's OWN unshelve landing between the automatic shelve committing
            # and `resolve_panel_shelve_write` reacquiring the lock for its final stamp.
            await real_set_shelved(
                id=fact.id, shelved=False, reason="owner reversal (race)",
                adapter=kwargs["adapter"], clock=TestClock(_EPOCH + timedelta(days=1)),
                lock=asyncio.Lock(),
            )
        return result

    monkeypatch.setattr(
        cleaner_writes_mod, "set_shelved", _set_shelved_that_races_an_owner_reversal
    )

    assessor = EchoAssessor(
        "unrelated",
        relevance_script={"panel shelve racing an owner reversal": _keep_lean_verdict()},
    )
    judge_a = FakeJudgeAdapter(identity="a", response=_relevance_payload("shelve-clear"))
    judge_b = FakeJudgeAdapter(identity="b", response=_relevance_payload("shelve-clear"))

    await run_relevance_axis(
        scope=Scope("p"), since=_EPOCH - timedelta(days=1), adapter=adapter, clock=clock,
        lock=asyncio.Lock(), cfg=EngineConfig(), rubric=load_rubric(),
        judge_a=judge_a, judge_b=judge_b, assessor=assessor,
    )

    fresh = await adapter.get_by_id(fact.id)
    assert fresh.lifecycle == "active"  # the racing reversal actually won

    [row] = await adapter.verdicts_for_record(fact.id)
    assert row.action == "panel_shelve"
    assert row.status == "acted"
    assert row.override == "set_shelved"  # NOT erased by the final stamp
    assert row.override_at is not None


async def test_panel_shelve_crash_unavailable_prepass_reconciles_unanswered_not_acted() -> None:
    """A crashed pass whose pre-pass verdict
    was ``Unavailable`` still shelves the record (D4 degrades to send-to-panel, not to skipping
    the write) -- normal completion (``resolve_panel_shelve_write``) stamps that row
    ``unanswered``, never ``acted``, so the reconcile must match it rather than stamping every
    landed ``panel_shelve`` row ``acted`` regardless of what its own verdict envelope carries --
    that would inflate the tuning query's ``acted`` count and hide the assessor outage
    ``unanswered`` exists to surface."""
    from tabularium.engine.dispositions import set_shelved

    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    fact = await _seed_fact(
        adapter, content="panel shelve, prepass unavailable, crash before final stamp"
    )

    unavailable = Unavailable(method="relevance", reason="timeout", detail=None, latency_ms=None)
    pending = build_cleaner_resolved_row(
        caller="cleaner_prepass", candidate_id=fact.id, new_record_id=fact.id,
        verdict=unavailable, action="panel_shelve", actor="panel", matched=None, now=_EPOCH,
    ).model_copy(update={"status": "pending", "stamped_at": None})
    await adapter.append_verdicts([pending])

    # The real shelve lands (mirrors `resolve_panel_shelve_write`'s own call), but the process
    # "crashes" before its own final stamp ever runs -- the row stays `pending`.
    await set_shelved(
        id=fact.id, shelved=True, reason="Cleaner unanimous panel shelve", adapter=adapter,
        clock=TestClock(_EPOCH), lock=asyncio.Lock(), skip_override_stamp=True,
    )

    acted, failed = await reconcile_pending_verdicts(adapter, now=_EPOCH + timedelta(days=1))
    assert acted == 1
    assert failed == 0

    [row] = await adapter.verdicts_for_record(fact.id)
    assert row.status == "unanswered"
    assert row.stamped_at is None
    assert row.unavailable_reason == "timeout"


def _keep_lean_verdict() -> Verdict:
    from tabularium.engine.assessor import Trace, compute_confidence

    odds = {"shelve-clear": 0.1, "shelve-lean": 0.1, "keep-lean": 0.7, "keep-clear": 0.1}
    return Verdict(
        method="relevance", pick="keep-lean", odds=odds, confidence=compute_confidence(odds),
        trace=Trace(
            model="echo", wording_hash="echo", latency_ms=0,
            input_tokens=None, output_tokens=None, request_id=None,
        ),
    )


def _relevance_payload(grade: str) -> str:
    return json.dumps({"verdicts": [{"index": 1, "grade": grade, "unclear": False, "why": "x"}]})


async def test_bury_scrub_crash_recovers_on_retry() -> None:
    """Failed burial scrubbing
    leaves text permanently retained: a crash between ``adapter.bury`` landing
    and ``scrub_verdict_text`` running must not orphan the text forever — a
    retry lands on the idempotent ``already_buried`` branch, which must still
    close the gap."""
    inner = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_restatement=True)
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=EchoAssessor("repetition")
    )
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id
    reinforced = await engine.save(_INCOMING, scope="p")
    assert reinforced.outcome == "reinforced"

    row_before = (await inner.verdicts_for_record(existing_id))[0]
    assert row_before.incoming_text is not None

    wrapper.fault(on="scrub_verdict_text", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with pytest.raises(RuntimeError):
        await engine.bury(existing_id, reason="test cleanup")

    row_mid = (await inner.verdicts_for_record(existing_id))[0]
    assert row_mid.incoming_text is not None

    wrapper.fault(on="scrub_verdict_text", call=999, exc=RuntimeError("never"))
    retry = await engine.bury(existing_id, reason="retry")
    assert retry.outcome == "already_buried"

    row_after = (await inner.verdicts_for_record(existing_id))[0]
    assert row_after.incoming_text is None


# ---------------------------------------------------------------------------
# Scope 4 — overrides, bury scrub, tuning query
# ---------------------------------------------------------------------------


async def test_unsupersede_stamps_override() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id
    await engine.save(_INCOMING, scope="p")

    await engine.unsupersede(existing_id, reason="test undo")

    [row] = await adapter.verdicts_for_record(existing_id)
    assert row.override == "unsupersede"
    assert row.override_at is not None


async def test_revoke_stamps_override() -> None:
    table = {_EXISTING: _angle_vec(0), _INCOMING: _angle_vec(5)}
    engine, adapter = await _build_engine(
        table, assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    d1 = await engine.save(_EXISTING, scope="p", kind="decision")
    assert d1.hypomnema is not None
    d2 = await engine.save(_INCOMING, scope="p", kind="decision")
    assert d2.hypomnema is not None
    d2_id = d2.hypomnema.id

    await engine.revoke(d2_id, reason="no longer applicable")

    rows = await adapter.verdicts_for_record(d2_id)
    stamped = [row for row in rows if row.new_record_id == str(d2_id)]
    assert stamped
    assert stamped[0].override == "revoke"


async def test_set_validity_stamps_override_on_revalidate() -> None:
    table = {_EXISTING: _angle_vec(0), _INCOMING: _angle_vec(5)}
    engine, adapter = await _build_engine(
        table, assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    decision = await engine.save(_EXISTING, scope="p", kind="decision")
    assert decision.hypomnema is not None
    decision_id = decision.hypomnema.id
    await engine.save(_INCOMING, scope="p")

    await engine.set_validity(decision_id, valid=True, reason="reviewed")

    [row] = await adapter.verdicts_for_record(decision_id)
    assert row.override == "set_validity"


async def test_set_shelved_stamps_override() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "saved"
    assert result.hypomnema is not None
    successor_id = result.hypomnema.id

    await engine.set_shelved(successor_id, shelved=True, reason="test shelve")

    rows = await adapter.verdicts_for_record(successor_id)
    stamped = [row for row in rows if row.new_record_id == str(successor_id)]
    assert stamped
    assert stamped[0].override == "set_shelved"


async def test_set_validity_same_state_confirmation_does_not_stamp_override() -> None:
    """Same-state confirmations
    incorrectly count as overrides: ``lifecycle``/``validity`` values a fresh
    record or an UNRELATED automatic exit can also produce are not proof that
    THIS tool's own prior write ever ran — a caller confirming a state that
    arose some other way must not inflate the tuning query's override count.
    Stamping unconditionally here was tried and reverted (evidence for both: this test)."""
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    result = await engine.save(_EXISTING, scope="p")
    assert result.outcome == "saved"
    assert result.hypomnema is not None
    fact_id = result.hypomnema.id

    # This Fact was never invalidated by anything — confirming "still valid"
    # is a same-state no-op, not a reversal of a logged effect.
    confirmed = await engine.set_validity(fact_id, valid=True, reason="just checking")
    assert confirmed.outcome == "unchanged"

    rows = await adapter.verdicts_for_record(fact_id)
    assert all(row.override is None for row in rows)


async def test_set_shelved_unshelve_confirmation_on_a_fresh_fact_does_not_stamp_override() -> None:
    """Same-state confirmations
    incorrectly count as overrides — the ``set_shelved`` arm of the same
    fix: ``lifecycle="active"`` is the default state every fresh Fact is
    born into, not proof this tool's own un-shelve action ever ran."""
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    result = await engine.save(_EXISTING, scope="p")
    assert result.outcome == "saved"
    assert result.hypomnema is not None
    fact_id = result.hypomnema.id

    confirmed = await engine.set_shelved(fact_id, shelved=False, reason="just checking")
    assert confirmed.outcome == "unchanged"

    rows = await adapter.verdicts_for_record(fact_id)
    assert all(row.override is None for row in rows)


async def test_second_override_does_not_overwrite_first() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id
    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "saved"
    assert result.hypomnema is not None
    successor_id = result.hypomnema.id

    # First reversal: unsupersede, matched via the row's own effect_ref (the
    # supersession edge id) — restores both rows to live.
    await engine.unsupersede(existing_id, reason="first undo")
    # Second reversal attempt on the SAME row, via its OTHER matching column
    # (new_record_id) — first override wins, this must not overwrite it.
    await engine.set_shelved(successor_id, shelved=True, reason="second attempt")

    rows = await adapter.verdicts_for_record(existing_id)
    original = next(row for row in rows if row.action == "supersede" and row.actor == "engine")
    assert original.override == "unsupersede"


async def test_revoke_repairs_override_stamp_on_idempotent_retry() -> None:
    """Committed reversals can
    permanently lose their override stamp: a crash between the ``revoked``
    transition landing and ``stamp_override`` running must not orphan the
    override forever — revoke's own retry is an idempotent ``unchanged`` ack
    (unlike unsupersede's refusal), and must now repair the stamp too."""
    table = {_EXISTING: _angle_vec(0), _INCOMING: _angle_vec(5)}
    inner = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(table)
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_replacement=True)
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=EchoAssessor("correction")
    )
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    d1 = await engine.save(_EXISTING, scope="p", kind="decision")
    assert d1.hypomnema is not None
    d2 = await engine.save(_INCOMING, scope="p", kind="decision")
    assert d2.hypomnema is not None
    d2_id = d2.hypomnema.id

    wrapper.fault(on="update_verdicts", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with pytest.raises(RuntimeError):
        await engine.revoke(d2_id, reason="first attempt")

    mid = [
        row for row in await inner.verdicts_for_record(d2_id) if row.new_record_id == str(d2_id)
    ]
    assert mid and mid[0].override is None

    wrapper.fault(on="update_verdicts", call=999, exc=RuntimeError("never"))
    retry = await engine.revoke(d2_id, reason="retry")
    assert retry.outcome == "unchanged"

    after = [
        row for row in await inner.verdicts_for_record(d2_id) if row.new_record_id == str(d2_id)
    ]
    assert after and after[0].override == "revoke"


async def test_set_shelved_repairs_override_stamp_on_idempotent_retry() -> None:
    """Committed reversals can
    permanently lose their override stamp — the ``set_shelved`` arm of the
    same fix."""
    inner = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_replacement=True)
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=EchoAssessor("correction")
    )
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    result = await engine.save(_INCOMING, scope="p")
    assert result.outcome == "saved"
    assert result.hypomnema is not None
    successor_id = result.hypomnema.id

    wrapper.fault(on="update_verdicts", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with pytest.raises(RuntimeError):
        await engine.set_shelved(successor_id, shelved=True, reason="first attempt")

    mid = [
        row for row in await inner.verdicts_for_record(successor_id)
        if row.new_record_id == str(successor_id)
    ]
    assert mid and mid[0].override is None

    wrapper.fault(on="update_verdicts", call=999, exc=RuntimeError("never"))
    retry = await engine.set_shelved(successor_id, shelved=True, reason="retry")
    assert retry.outcome == "unchanged"

    after = [
        row for row in await inner.verdicts_for_record(successor_id)
        if row.new_record_id == str(successor_id)
    ]
    assert after and after[0].override == "set_shelved"


async def test_bury_scrubs_incoming_text_for_that_candidate_only() -> None:
    """Two orthogonal near-duplicate pairs (cosine 0 between clusters), so each
    candidate's own bounce never sees the other — burying the first candidate must
    scrub only ITS row, never the second's."""
    pair_1_existing = "the quick brown fox jumps over the lazy dog"
    pair_1_incoming = "the quick brown fox jumps over the lazy hound"
    pair_2_existing = "a committee proposal about quarterly budget allocation"
    pair_2_incoming = "a committee proposal about quarterly budget spending"
    table = {
        pair_1_existing: _angle_vec(0),
        pair_1_incoming: _angle_vec(5),
        pair_2_existing: _angle_vec(90),
        pair_2_incoming: _angle_vec(95),
    }
    engine, adapter = await _build_engine(table, assessor=EchoAssessor("repetition"))
    a = await engine.save(pair_1_existing, scope="p")
    d = await engine.save(pair_2_existing, scope="p")
    assert a.hypomnema is not None
    assert d.hypomnema is not None

    bounced_a = await engine.save(pair_1_incoming, scope="p")
    assert bounced_a.bounce is not None
    await engine.save(
        pair_1_incoming, scope="p",
        resolve=BounceResolution(
            candidate_id=a.hypomnema.id, action="reinforce",
            bounce_id=bounced_a.bounce.bounce_id,
        ),
    )
    bounced_d = await engine.save(pair_2_incoming, scope="p")
    assert bounced_d.bounce is not None
    await engine.save(
        pair_2_incoming, scope="p",
        resolve=BounceResolution(
            candidate_id=d.hypomnema.id, action="reinforce",
            bounce_id=bounced_d.bounce.bounce_id,
        ),
    )

    await engine.bury(a.hypomnema.id, reason="test bury")

    [row_a] = await adapter.verdicts_by_bounce(bounced_a.bounce.bounce_id)
    assert row_a.action == "reinforce"
    assert row_a.incoming_text is None
    [row_d] = await adapter.verdicts_by_bounce(bounced_d.bounce.bounce_id)
    assert row_d.action == "reinforce"
    assert row_d.incoming_text == pair_2_incoming


async def test_keyless_outage_writes_no_verdict_rows() -> None:
    """D16: a keyless install's outage path never writes an ``absent`` row —
    verified end-to-end through a real BM25-only bounce, not a hand-built row.
    Guards against the keyless outage path writing forbidden absent rows."""
    from tests.support.doubles import FakeEmbedder
    from tests.support.faults import FaultInjectingEmbedder, as_embedder

    fake = FakeEmbedder(dim=_DIM)
    fault = FaultInjectingEmbedder(fake)
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
    engine = Tabularium(adapter, as_embedder(fault), clock, _make_config(), assessor=None)
    await engine.initialize()
    await adapter.create_scope(Scope("p"))

    existing = await engine.save(
        "distinctive marker alpha bravo charlie", scope="p"
    )
    assert existing.hypomnema is not None

    fault.arm()
    result = await engine.save("distinctive marker alpha bravo delta", scope="p")
    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.reason == "judge_unavailable"

    all_rows = list(adapter._verdicts.values())
    assert all_rows == []


async def test_outage_reinforce_replay_stamps_row_with_incoming_text() -> None:
    """Outage replay bypasses caller
    verdict stamping: a ``reinforce`` resolve replayed while the embedder is
    STILL down must stamp the ``judge_unavailable`` bounce's own row — the
    discarded text lands too (D4), same as the live-vector caller-exit path."""
    from tests.support.doubles import FakeEmbedder
    from tests.support.faults import FaultInjectingEmbedder, as_embedder

    fake = FakeEmbedder(dim=_DIM)
    fault = FaultInjectingEmbedder(fake)
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
    # A real Assessor (not the keyless ``None`` of the sibling test above) — an
    # ``absent`` row never persists (D16), so a row to replay against requires
    # a genuine judged verdict, not merely a BM25-only bounce.
    engine = Tabularium(
        adapter, as_embedder(fault), clock, _make_config(), assessor=EchoAssessor("repetition")
    )
    await engine.initialize()
    await adapter.create_scope(Scope("p"))

    existing = await engine.save("distinctive marker alpha bravo charlie", scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    fault.arm()
    bounced = await engine.save("distinctive marker alpha bravo delta", scope="p")
    assert bounced.outcome == "bounced"
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id
    candidate_id = bounced.bounce.candidate_id
    assert candidate_id == existing_id

    result = await engine.save(
        "distinctive marker alpha bravo delta", scope="p",
        resolve=BounceResolution(
            candidate_id=candidate_id, action="reinforce", bounce_id=bounce_id,
        ),
    )
    assert result.outcome == "reinforced"

    [row] = await adapter.verdicts_for_record(existing_id)
    assert row.status == "acted"
    assert row.actor == "caller"
    assert row.action == "reinforce"
    assert row.incoming_text == "distinctive marker alpha bravo delta"


async def test_split_keep_write_updates_a_restamped_row_and_appends_the_resolving_review() -> None:
    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    fact = await _seed_fact(adapter, content="split-keep record restamped from a pre-pass row")
    original = build_cleaner_resolved_row(
        caller="cleaner_prepass", candidate_id=fact.id, verdict=_keep_lean_verdict(),
        action="panel_keep", actor="panel", matched=True, now=_EPOCH - timedelta(days=1),
    )
    await adapter.append_verdicts([original])
    pending = restamp_cleaner_row(
        original, action="split_keep", actor="engine", matched=True, now=_EPOCH
    )

    landed = await resolve_split_keep_write(
        pending_row=pending,
        write_pending=adapter.update_verdicts,
        record=fact,
        rubric_version="test",
        next_review_due_at=_EPOCH + timedelta(days=30),
        now=_EPOCH,
        adapter=adapter,
        lock=asyncio.Lock(),
    )

    assert landed is True
    [row] = await adapter.verdicts_for_record(fact.id)
    assert row.id == original.id
    assert row.status == "acted"
    assert row.action == "split_keep"
    [review] = await adapter.list_reviews_for(target_id=fact.id)
    assert review.outcome == "assessor_resolved"
    assert review.band == "keep"


async def test_split_keep_write_review_failure_stamps_failed_and_reports_through_the_callback() -> (
    None
):
    inner = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    fact = await _seed_fact(inner, content="split-keep record whose review append raises")
    wrapper = FaultInjectingAdapter(inner)
    wrapper.fault(on="append_review", call=1, exc=RuntimeError("review boom"))
    wrapper.arm()
    adapter = as_adapter(wrapper)
    pending = build_cleaner_pending_row(
        caller="cleaner_prepass", candidate_id=fact.id, new_record_id=None,
        verdict=_keep_lean_verdict(), action="split_keep", effect_ref=None, now=_EPOCH,
    )
    reported: list[str] = []

    async def _report(exc: Exception) -> None:
        reported.append(str(exc))

    landed = await resolve_split_keep_write(
        pending_row=pending,
        write_pending=adapter.append_verdicts,
        record=fact,
        rubric_version="test",
        next_review_due_at=_EPOCH + timedelta(days=30),
        now=_EPOCH,
        adapter=adapter,
        lock=asyncio.Lock(),
        on_review_append_failure=_report,
    )

    assert landed is False
    assert reported == ["review boom"]
    [row] = await inner.verdicts_for_record(fact.id)
    assert row.status == "failed"
    assert await inner.list_reviews_for(target_id=fact.id) == []


async def test_reconcile_raises_for_a_pending_row_whose_action_has_no_read_path() -> None:
    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    pending = build_cleaner_pending_row(
        caller="cleaner_prepass", candidate_id=HypomnemaId(uuid.UUID(int=7)), new_record_id=None,
        verdict=_keep_lean_verdict(), action="split_keep", effect_ref=None, now=_EPOCH,
    ).model_copy(update={"action": "bogus"})
    await adapter.append_verdicts([pending])

    with pytest.raises(RuntimeError) as raised:
        await reconcile_pending_verdicts(adapter, now=_EPOCH)

    assert str(raised.value) == (
        "reconcile_pending_verdicts: no read path for pending row action='bogus' "
        f"id={pending.id!r} \u2014 stop and report rather than defaulting"
    )


def test_verdict_from_cleaner_row_rejects_a_method_outside_the_assessor_set() -> None:
    from pydantic import ValidationError

    row = build_cleaner_resolved_row(
        caller="cleaner_prepass", candidate_id=HypomnemaId(uuid.UUID(int=7)),
        verdict=_keep_lean_verdict(), action="panel_keep", actor="panel", matched=True, now=_EPOCH,
    ).model_copy(update={"method": "bogus"})

    with pytest.raises(ValidationError, match="1 validation error for Verdict"):
        verdict_from_cleaner_row(row)


def test_verdict_from_cleaner_row_rejects_an_unavailable_reason_outside_the_set() -> None:
    from pydantic import ValidationError

    row = build_cleaner_resolved_row(
        caller="cleaner_prepass", candidate_id=HypomnemaId(uuid.UUID(int=7)),
        verdict=Unavailable(method="relevance", reason="timeout", detail=None, latency_ms=None),
        action=None, actor=None, matched=None, now=_EPOCH,
    ).model_copy(update={"unavailable_reason": "bogus"})

    with pytest.raises(ValidationError, match="1 validation error for Unavailable"):
        verdict_from_cleaner_row(row)


@pytest.mark.parametrize("pick", ["bogus", "keep-clear", ""])
async def test_caller_exit_reads_a_pick_outside_the_relation_set_as_distinct(pick: str) -> None:
    adapter = MemoryAdapter(embedding_model="test/verdict-log", dim=_DIM, clock=TestClock(_EPOCH))
    candidate = await _seed_fact(adapter, content="candidate for an off-set pick")
    row = build_cleaner_resolved_row(
        caller="save", candidate_id=candidate.id, verdict=_keep_lean_verdict(),
        action=None, actor=None, matched=None, now=_EPOCH,
    ).model_copy(update={"pick": pick})
    distinct = row.model_copy(update={"pick": "distinct"})

    for action in ("reinforce", "confirm_distinct", "supersede", "promoted_from"):
        assert caller_exit_matches_pick(
            row, action=action, incoming_kind="fact", candidate=candidate, direction="same"
        ) is caller_exit_matches_pick(
            distinct, action=action, incoming_kind="fact", candidate=candidate, direction="same"
        )


# ---------------------------------------------------------------------------
# Shared seed/placeholder helpers
# ---------------------------------------------------------------------------


def _placeholder_trace() -> Any:
    from tabularium.engine.assessor import Trace

    return Trace(
        model="test", wording_hash="test", latency_ms=0,
        input_tokens=None, output_tokens=None, request_id=None,
    )


async def _seed_fact(adapter: MemoryAdapter, *, content: str) -> Any:
    from tabularium.domain import Fact, new_id

    fact = Fact(
        id=new_id(), type="fact", content=content, scope=Scope("p"),
        lifecycle="active", validity="valid", created_at=_EPOCH, updated_at=_EPOCH,
    )
    await adapter.create_scope(Scope("p"))
    await adapter.upsert(fact, [1.0] * _DIM)
    return fact
