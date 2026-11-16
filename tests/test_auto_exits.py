"""Automatic Assessor exits (P8 assessor-automatic-exits piece) — one test per Observable
behavior example, on the arm it names, plus the switch-off example for all four relations and
the ``unsupersede_fact`` undo of an automatic edge.

Hermetic lane only — ``MemoryAdapter`` + a table-driven embedder that places two texts at a
KNOWN cosine (angle-vector trick, lifted from ``test_gate_probe.py``), so a near-duplicate pair
reliably survives the serial probe regardless of hash luck.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import uuid4

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, HypomnemaId, Scope, new_id
from tabularium.engine import Tabularium
from tabularium.engine._errors import ToolError
from tabularium.engine.assessor import (
    AssessorCandidate,
    ContradictionVerdict,
    PanelDisagreement,
    ReasonClass,
    Relation,
    RelationVerdict,
    Trace,
    Verdict,
)
from tabularium.results import DecisionBounce, JudgedVerdict
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import PredecessorFlip, StorageAdapter
from tests.support.doubles import TestClock
from tests.support.edits import edit_content
from tests.support.fake_assessor import EchoAssessor
from tests.support.faults import FaultInjectingAdapter, as_adapter

_DIM = 16
_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)

# Near-duplicate pair: shares every token but the last word, so BM25 overlaps and the exact-key
# step misses (different canonical form) — "Texts differ enough that the exact-key step misses
# and the serial probe finds the existing row as the sole survivor" (brief, Observable behavior).
_EXISTING = "the quick brown fox jumps over the lazy dog"
_INCOMING = "the quick brown fox jumps over the lazy hound"
_SURVIVOR = "the quick brown fox jumps over the lazy wolf"
_UNRELATED = "a committee proposal about quarterly budget allocation"


def _angle_vec(angle_deg: float, dim: int = _DIM) -> list[float]:
    """Unit vector at ``angle_deg`` from axis 0 — two angle-parameterised vectors ``theta``
    apart have cosine ``cos(theta)`` exactly (lifted from ``test_gate_probe.py``)."""
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


class _TableEmbedder:
    """Deterministic embedder: exact-string -> vector lookup, so a test controls cosine
    similarity directly rather than hoping a hash collides."""

    dim: int = _DIM
    model: str = "test/auto-exits"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


#: The default table every test starts from — near-dup pair at 5 degrees (cosine ~0.9962,
#: above both ``gray_floor`` (0.80) and ``decision_gray_floor`` (0.70)); the unrelated text at
#: 90 degrees (cosine 0) never survives any probe.
_DEFAULT_TABLE: dict[str, list[float]] = {
    _EXISTING: _angle_vec(0),
    _INCOMING: _angle_vec(5),
    _SURVIVOR: _angle_vec(5),
    _UNRELATED: _angle_vec(90),
}


def _placeholder_trace() -> Trace:
    return Trace(
        model="test", wording_hash="test", latency_ms=0,
        input_tokens=None, output_tokens=None, request_id=None,
    )


def _placeholder_verdict(method: str, pick: str) -> Verdict:
    return Verdict(
        method=method,  # type: ignore[arg-type]
        pick=pick, odds={pick: 1.0}, confidence=1.0, trace=_placeholder_trace(),
    )


class _PlaceholderMethods:
    """Fixed placeholder returns for the five Assessor methods this file never exercises —
    only ``relation`` is under test here; wording for the rest lands with A3/A5/A6."""

    async def kind(self, text: str, declared_kind: str) -> Verdict:
        return _placeholder_verdict("kind", declared_kind)

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Verdict:
        return _placeholder_verdict("worth", "keep")

    async def perishable(self, text: str) -> Verdict:
        return _placeholder_verdict("perishable", "standing")

    async def breadth(self, text: str, source_scope: str) -> Verdict:
        return _placeholder_verdict("breadth", "project_specific")

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Verdict:
        return _placeholder_verdict("scope", "global")

    async def relevance(self, text: str) -> Verdict:
        return _placeholder_verdict("relevance", "keep")

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        return ContradictionVerdict(
            pick="coexist", reason_class=None,
            odds={"coexist": 1.0, "correction": 0.0, "changed": 0.0},
            confidence=1.0, trace=_placeholder_trace(),
        )


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {
        "candidate_k": 1000, "gray_floor": 0.80, "decision_gray_floor": 0.70,
    }
    defaults.update(kwargs)
    return EngineConfig(**defaults)


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
    await adapter.create_scope(Scope("q"))
    return engine, adapter


# ---------------------------------------------------------------------------
# Switch off (all four relations) — the production default.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("material", "relation"),
    [("repetition", "restatement"), ("correction", "replacement"),
     ("addition", "follow_up"), ("unrelated", "distinct")],
)
async def test_switch_off_bounces_with_reason(material: str, relation: str) -> None:
    engine, _adapter = await _build_engine(assessor=EchoAssessor(material))  # type: ignore[arg-type]
    await engine.save(_EXISTING, scope="p")

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.judged is not None
    relation_judged = result.bounce.judged["relation"]
    assert isinstance(relation_judged, JudgedVerdict)
    assert relation_judged.pick == relation
    assert result.bounce.withheld == "switch_off"


# ---------------------------------------------------------------------------
# Automatic restatement
# ---------------------------------------------------------------------------


async def test_auto_restatement_same_scope_reinforces() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("repetition"), auto_exit_restatement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "reinforced"
    assert result.reinforced_id == existing_id
    assert result.hypomnema is None
    row_count = 0
    async for _ in await adapter.scan_records(lifecycle_in=None):
        row_count += 1
    assert row_count == 1
    events = await adapter.query_events(target_id=existing_id)
    dedup_events = [e for e in events if e.source == "dedup"]
    assert len(dedup_events) == 1
    assert dedup_events[0].matched_from_id is None


async def test_auto_restatement_downward_reinforces() -> None:
    engine, _adapter = await _build_engine(
        assessor=EchoAssessor("repetition"), auto_exit_restatement=True
    )
    existing = await engine.save(_EXISTING, scope="global")
    assert existing.hypomnema is not None

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "reinforced"
    assert result.reinforced_id == existing.hypomnema.id


async def test_auto_restatement_upward_promotes_by_moved_supersede() -> None:
    """A4 bounce-judged piece, decision D3: upward restatement is a promotion, not a
    perpetual bounce — the switch on writes the incoming global content straight
    through, superseding the named-scope row with ``reason="moved"``."""
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("repetition"), auto_exit_restatement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None

    result = await engine.save(_INCOMING, scope="global")

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    predecessor = await adapter.get_by_id(existing.hypomnema.id)
    assert predecessor.superseded_by == result.hypomnema.id
    assert predecessor.supersession_reason == "moved"


# ---------------------------------------------------------------------------
# Automatic replacement
# ---------------------------------------------------------------------------


async def test_auto_replacement_fact_fact_same_scope_supersedes() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None
    existing_id = existing.hypomnema.id

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    new_id = result.hypomnema.id

    predecessor = await adapter.get_by_id(existing_id)
    assert predecessor.superseded_by == new_id
    assert predecessor.supersession_reason == "correction"
    assert predecessor.supersession_note is None
    assert predecessor.supersession_edge_id is not None
    assert predecessor.content == _EXISTING

    # unsupersede_fact undoes it exactly as a caller-made edge.
    undo = await engine.unsupersede(existing_id, reason="test undo")
    assert undo.shape == "fact_both_live"
    restored = await adapter.get_by_id(existing_id)
    assert restored.superseded_by is None
    assert restored.validity == "valid"


async def test_auto_replacement_cross_scope_bounces_withheld() -> None:
    engine, _adapter = await _build_engine(
        assessor=EchoAssessor("changed state"), auto_exit_replacement=True
    )
    await engine.save(_EXISTING, scope="global")

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.withheld == "cross_scope_replacement"


async def test_auto_replacement_elaborated_fact_fact_supersedes() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("elaboration"), auto_exit_replacement=True
    )
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "saved"
    predecessor = await adapter.get_by_id(existing.hypomnema.id)
    assert predecessor.supersession_reason == "elaborated"


# ---------------------------------------------------------------------------
# Composition: caller supersedes + Assessor-found replacement in one transaction.
# ---------------------------------------------------------------------------


async def test_composition_one_replace_call_both_flip() -> None:
    table = dict(_DEFAULT_TABLE)
    engine, adapter = await _build_engine(
        table, assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    a = await engine.save(_UNRELATED, scope="p")
    c = await engine.save(_SURVIVOR, scope="p")
    assert a.hypomnema is not None
    assert c.hypomnema is not None

    result = await engine.save(
        _INCOMING, scope="p", supersedes=[a.hypomnema.id], reason="changed"
    )

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    successor_id = result.hypomnema.id

    a_row = await adapter.get_by_id(a.hypomnema.id)
    c_row = await adapter.get_by_id(c.hypomnema.id)
    assert a_row.superseded_by == successor_id
    assert a_row.supersession_reason == "changed"
    assert c_row.superseded_by == successor_id
    assert c_row.supersession_reason == "correction"
    assert a_row.supersession_edge_id != c_row.supersession_edge_id


async def test_composition_mixed_scope_allowed() -> None:
    table = dict(_DEFAULT_TABLE)
    engine, adapter = await _build_engine(
        table, assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    a = await engine.save(_UNRELATED, scope="global")
    c = await engine.save(_SURVIVOR, scope="p")
    assert a.hypomnema is not None
    assert c.hypomnema is not None

    result = await engine.save(
        _INCOMING, scope="p", supersedes=[a.hypomnema.id], reason="moved"
    )

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    a_row = await adapter.get_by_id(a.hypomnema.id)
    c_row = await adapter.get_by_id(c.hypomnema.id)
    assert a_row.supersession_reason == "moved"
    assert c_row.supersession_reason == "correction"

    # The caller-list mixed-scope refusal is untouched.
    with pytest.raises(ToolError):
        await engine.save(
            "another new atom entirely",
            scope="p",
            supersedes=[a.hypomnema.id, c.hypomnema.id],
            reason="correction",
        )


async def test_composition_refused_bounces_supersedes_unapplied() -> None:
    table = dict(_DEFAULT_TABLE)
    # Switches default off: the found candidate's exit is Withheld(switch_off).
    engine, adapter = await _build_engine(table, assessor=EchoAssessor("correction"))
    a = await engine.save(_UNRELATED, scope="p")
    c = await engine.save(_SURVIVOR, scope="p")
    assert a.hypomnema is not None
    assert c.hypomnema is not None

    result = await engine.save(
        _INCOMING, scope="p", supersedes=[a.hypomnema.id], reason="changed"
    )

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.withheld == "switch_off"
    a_row = await adapter.get_by_id(a.hypomnema.id)
    c_row = await adapter.get_by_id(c.hypomnema.id)
    assert a_row.superseded_by is None
    assert a_row.validity == "valid"
    assert c_row.superseded_by is None


# ---------------------------------------------------------------------------
# Fact vs Decision candidate: invalidation (scope 2 placeholder), elaboration bounce.
# ---------------------------------------------------------------------------


async def test_fact_invalidates_decision() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    existing = await engine.save(_EXISTING, scope="p", kind="decision")
    assert existing.hypomnema is not None
    decision_id = existing.hypomnema.id

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    fact_id = result.hypomnema.id

    decision = await adapter.get_by_id(decision_id)
    assert isinstance(decision, Decision)
    assert decision.validity == "invalid"
    assert decision.invalidated_by == str(fact_id)
    assert decision.invalidation_reason == "correction"
    assert decision.lifecycle == "in_force"
    assert decision.superseded_by is None


async def test_composed_invalidation_fault_leaves_decision_valid_and_no_fact() -> None:
    inner = MemoryAdapter(embedding_model="test/auto-exits", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_replacement=True)
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=EchoAssessor("correction")
    )
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    existing = await engine.save(_EXISTING, scope="p", kind="decision")
    assert existing.hypomnema is not None

    wrapper.fault(on="replace", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with pytest.raises(ToolError):
        await engine.save(_INCOMING, scope="p")

    decision = await inner.get_by_id(existing.hypomnema.id)
    assert isinstance(decision, Decision)
    assert decision.validity == "valid"
    assert decision.invalidated_by is None
    row_count = 0
    async for _ in await inner.scan_records(lifecycle_in=None):
        row_count += 1
    assert row_count == 1


async def test_revalidation_clears_both_columns_and_writes_event() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("correction"), auto_exit_replacement=True
    )
    existing = await engine.save(_EXISTING, scope="p", kind="decision")
    assert existing.hypomnema is not None
    decision_id = existing.hypomnema.id
    incoming = await engine.save(_INCOMING, scope="p")
    assert incoming.hypomnema is not None
    fact_id = incoming.hypomnema.id

    result = await engine.set_validity(decision_id, valid=True, reason="reviewed")

    assert result.outcome == "revised"
    assert result.validity == "valid"
    decision = await adapter.get_by_id(decision_id)
    assert isinstance(decision, Decision)
    assert decision.validity == "valid"
    assert decision.invalidated_by is None
    assert decision.invalidation_reason is None

    events = await adapter.list_revalidations(decision_id)
    assert len(events) == 1
    assert events[0].decision_id == decision_id
    assert events[0].withdrawn_cause == fact_id
    assert events[0].withdrawn_reason == "correction"
    assert events[0].reason == "reviewed"

    # set_validity(valid=False, ...) on a Decision never touches the two columns.
    await engine.set_validity(decision_id, valid=False, reason="reconsidered")
    still = await adapter.get_by_id(decision_id)
    assert isinstance(still, Decision)
    assert still.validity == "invalid"
    assert still.invalidated_by is None
    assert still.invalidation_reason is None


async def test_fact_elaborates_decision_bounces_withheld() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("elaboration"), auto_exit_replacement=True
    )
    existing = await engine.save(_EXISTING, scope="p", kind="decision")
    assert existing.hypomnema is not None

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.withheld == "elaborated_against_decision"
    candidate = await adapter.get_by_id(existing.hypomnema.id)
    assert candidate.validity == "valid"


# ---------------------------------------------------------------------------
# Decision save against a Fact candidate — bounces regardless of relation.
# ---------------------------------------------------------------------------


async def test_decision_against_fact_bounces_withheld() -> None:
    engine, _adapter = await _build_engine(
        assessor=EchoAssessor("correction"),
        auto_exit_restatement=True,
        auto_exit_replacement=True,
        auto_exit_follow_up=True,
        auto_exit_distinct=True,
    )
    await engine.save(_EXISTING, scope="p")

    result = await engine.save(_INCOMING, scope="p", kind="decision")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.judged is not None
    assert result.bounce.withheld == "decision_against_fact"


# ---------------------------------------------------------------------------
# Retired candidate — a Fact save bounces against an already-revoked Decision
# (the serial probe surfaces retired lifecycles the ordinary dispatch never filters
# for a Decision candidate on the FACT arm).
# ---------------------------------------------------------------------------


async def test_retired_candidate_bounces_withheld() -> None:
    engine, _adapter = await _build_engine(
        assessor=EchoAssessor("correction"),
        auto_exit_restatement=True,
        auto_exit_replacement=True,
        auto_exit_follow_up=True,
        auto_exit_distinct=True,
    )
    existing = await engine.save(_EXISTING, scope="p", kind="decision")
    assert existing.hypomnema is not None
    await engine.revoke(existing.hypomnema.id, reason="no longer applicable")

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.withheld == "candidate_retired"


# ---------------------------------------------------------------------------
# Automatic follow_up / distinct — write the row, register no pair.
# ---------------------------------------------------------------------------


async def test_auto_follow_up_writes_no_pair() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("addition"), auto_exit_follow_up=True
    )
    a = await engine.save(_EXISTING, scope="p")
    assert a.hypomnema is not None

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    registered = await adapter.is_distinct(a.hypomnema.id, result.hypomnema.id)
    assert registered is False


async def test_auto_distinct_writes_no_pair() -> None:
    engine, adapter = await _build_engine(
        assessor=EchoAssessor("unrelated"), auto_exit_distinct=True
    )
    a = await engine.save(_EXISTING, scope="p")
    assert a.hypomnema is not None

    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    registered = await adapter.is_distinct(a.hypomnema.id, result.hypomnema.id)
    assert registered is False


async def test_auto_write_never_calls_mark_distinct() -> None:
    inner = MemoryAdapter(embedding_model="test/auto-exits", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    wrapper.fault(
        on="mark_distinct", call=1, exc=AssertionError("mark_distinct must not be called")
    )
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_follow_up=True)
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=EchoAssessor("addition")
    )
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    await engine.save(_EXISTING, scope="p")

    wrapper.arm()
    result = await engine.save(_INCOMING, scope="p")

    assert result.outcome == "saved"


# ---------------------------------------------------------------------------
# candidate_changed — the world moves between the Assessor call and the re-take.
# ---------------------------------------------------------------------------


class _WorldMovingAssessor(_PlaceholderMethods):
    """Buries ``target_id`` as a side effect of ``relation`` — simulating a concurrent
    write landing during the Assessor call, which runs OUTSIDE the lock (D12)."""

    def __init__(self, adapter: MemoryAdapter, target_id: HypomnemaId) -> None:
        self._adapter = adapter
        self._target_id = target_id

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        await self._adapter.bury(self._target_id, reason="world moved")
        return RelationVerdict(
            method="relation", pick="restatement", odds={"restatement": 1.0}, confidence=1.0,
            trace=_placeholder_trace(),
        )


async def test_candidate_changed_redispatches_fresh() -> None:
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    cfg = _make_config(auto_exit_restatement=True)
    existing_engine = Tabularium(adapter, embedder, clock, cfg)
    await existing_engine.initialize()
    await adapter.create_scope(Scope("p"))
    existing = await existing_engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None

    assessor = _WorldMovingAssessor(adapter, existing.hypomnema.id)
    engine = Tabularium(adapter, embedder, clock, cfg, assessor=assessor)

    result = await engine.save(_INCOMING, scope="p")

    # The judged candidate is gone by the time the arm re-takes the lock; the
    # world-moved redispatch finds no live survivor and writes straight through.
    assert result.outcome == "saved"
    assert result.hypomnema is not None
    assert result.hypomnema.id != existing.hypomnema.id


# ---------------------------------------------------------------------------
# WriteNew re-admission race — a second, identical save lands during the
# Assessor call.
# ---------------------------------------------------------------------------


class _ConcurrentDuplicateAssessor(_PlaceholderMethods):
    """Saves ``incoming`` a second time as a side effect of the first ``relation``
    call — simulating a concurrent, identical save landing while this one is
    judged, which runs OUTSIDE the lock (D12). Proves ``WriteNew`` rechecks
    admission steps 1/3 before writing, rather than writing an undetected
    byte-identical duplicate."""

    def __init__(self) -> None:
        self.engine: Tabularium | None = None
        self.duplicate_id: HypomnemaId | None = None
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
            self.duplicate_id = duplicate.hypomnema.id
        return RelationVerdict(
            method="relation", pick="follow_up", odds={"follow_up": 1.0}, confidence=1.0,
            trace=_placeholder_trace(),
        )


async def test_auto_write_rechecks_admission_before_writing_a_duplicate() -> None:
    embedder = _TableEmbedder(dict(_DEFAULT_TABLE))
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    cfg = _make_config(auto_exit_follow_up=True)
    assessor = _ConcurrentDuplicateAssessor()
    engine = Tabularium(adapter, embedder, clock, cfg, assessor=assessor)
    assessor.engine = engine
    await engine.initialize()
    await adapter.create_scope(Scope("p"))
    await engine.save(_EXISTING, scope="p")

    result = await engine.save(_INCOMING, scope="p")

    assert assessor.duplicate_id is not None
    assert result.outcome == "already_saved"
    assert result.hypomnema is not None
    assert result.hypomnema.id == assessor.duplicate_id
    row_count = 0
    async for _ in await adapter.scan_records(lifecycle_in=None):
        row_count += 1
    assert row_count == 2


class _ConcurrentCanonicalKeyAssessor(_PlaceholderMethods):
    """Saves a canonical-key twin of ``incoming`` (same key, different raw bytes)
    as a side effect of the first ``relation`` call, then always answers
    ``follow_up``. Proves that a ``WriteNew`` re-admission hit's own failure
    (here, the exact-key branch's ``reinforce`` call) is logged exactly once,
    by the caller's outer handler, never twice."""

    def __init__(self, *, kind: Literal["fact", "decision"] = "fact") -> None:
        self.engine: Tabularium | None = None
        self._kind = kind
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
            await self.engine.save(incoming + " ", scope="p", kind=self._kind)
        return RelationVerdict(
            method="relation", pick="follow_up", odds={"follow_up": 1.0}, confidence=1.0,
            trace=_placeholder_trace(),
        )


async def test_writenew_recheck_failure_logs_admission_resolved_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = dict(_DEFAULT_TABLE)
    table[_INCOMING + " "] = _angle_vec(5)
    inner = MemoryAdapter(embedding_model="test/auto-exits", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(table)
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_follow_up=True)
    assessor = _ConcurrentCanonicalKeyAssessor()
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=assessor
    )
    assessor.engine = engine
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    existing = await engine.save(_EXISTING, scope="p")
    assert existing.hypomnema is not None

    wrapper.fault(on="append_event", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with caplog.at_level(logging.DEBUG, logger="tabularium"), pytest.raises(RuntimeError):
        await engine.save(_INCOMING, scope="p")

    lines = [
        r
        for r in caplog.records
        if r.getMessage().startswith("admission_resolved") and "outcome=raised" in r.getMessage()
    ]
    assert len(lines) == 1
    assert f"id={existing.hypomnema.id}" in lines[0].getMessage()


async def test_writenew_recheck_failure_logs_admission_resolved_exactly_once_decision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = dict(_DEFAULT_TABLE)
    table[_INCOMING + " "] = _angle_vec(5)
    inner = MemoryAdapter(embedding_model="test/auto-exits", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(table)
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_follow_up=True)
    assessor = _ConcurrentCanonicalKeyAssessor(kind="decision")
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=assessor
    )
    assessor.engine = engine
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    existing = await engine.save(_EXISTING, scope="p", kind="decision")
    assert existing.hypomnema is not None

    wrapper.fault(on="append_event", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with caplog.at_level(logging.DEBUG, logger="tabularium"), pytest.raises(RuntimeError):
        await engine.save(_INCOMING, scope="p", kind="decision")

    lines = [
        r
        for r in caplog.records
        if r.getMessage().startswith("admission_resolved") and "outcome=raised" in r.getMessage()
    ]
    assert len(lines) == 1
    assert f"id={existing.hypomnema.id}" in lines[0].getMessage()


# ---------------------------------------------------------------------------
# Composed fault: one ``replace`` call, both-or-neither.
# ---------------------------------------------------------------------------


async def test_composed_replace_fault_leaves_nothing_written() -> None:
    table = dict(_DEFAULT_TABLE)
    inner = MemoryAdapter(embedding_model="test/auto-exits", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(table)
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_replacement=True)
    engine = Tabularium(
        as_adapter(wrapper), embedder, clock, cfg, assessor=EchoAssessor("correction")
    )
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    a = await engine.save(_UNRELATED, scope="p")
    c = await engine.save(_SURVIVOR, scope="p")
    assert a.hypomnema is not None
    assert c.hypomnema is not None

    wrapper.fault(on="replace", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with pytest.raises(ToolError):
        await engine.save(_INCOMING, scope="p", supersedes=[a.hypomnema.id], reason="changed")

    a_row = await inner.get_by_id(a.hypomnema.id)
    c_row = await inner.get_by_id(c.hypomnema.id)
    assert a_row.superseded_by is None
    assert c_row.superseded_by is None
    row_count = 0
    async for _ in await inner.scan_records(lifecycle_in=None):
        row_count += 1
    assert row_count == 2


# ---------------------------------------------------------------------------
# Three-survivor fan-out (D3/D4): one ``relation`` call per probe survivor,
# issued concurrently, in probe order.
# ---------------------------------------------------------------------------

_MULTI_A = "alpha bravo charlie delta echo foxtrot golf hotel india"
_MULTI_B = "alpha bravo charlie delta echo foxtrot golf hotel juliet"
_MULTI_C = "alpha bravo charlie delta echo foxtrot golf hotel kilo"
_MULTI_INCOMING = "alpha bravo charlie delta echo foxtrot golf hotel lima"


class _SleepyRelationAssessor(_PlaceholderMethods):
    """Records ``relation`` call order and sleeps ``delay_s`` per call — a save that issues
    three such calls sequentially would take 3x ``delay_s``; concurrently, about 1x."""

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s
        self.calls: list[str] = []

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        self.calls.append(candidate.id)
        await asyncio.sleep(self._delay_s)
        return RelationVerdict(
            method="relation", pick="distinct", odds={"distinct": 1.0}, confidence=1.0,
            trace=Trace(
                model="test", wording_hash="test", latency_ms=0,
                input_tokens=None, output_tokens=None, request_id=None,
            ),
        )


def _separated_vec(w: float, axis: int, dim: int = _DIM) -> list[float]:
    """Three vectors sharing weight ``w`` on axis 0 and 1.0 each on a DIFFERENT other axis:
    every pair has cosine ``w**2/(w**2+1)`` (kept below ``gray_floor`` so seeding never bounces
    them against each other), while each has cosine ``w/sqrt(w**2+1)`` against the pure-axis-0
    ``incoming`` vector below (kept above ``gray_floor`` so all three survive its probe)."""
    v = [0.0] * dim
    v[0] = w
    v[axis] = 1.0
    return v


async def test_three_survivors_fan_out_concurrently_within_one_save() -> None:
    table = dict(_DEFAULT_TABLE)
    table[_MULTI_A] = _separated_vec(1.5, 1)
    table[_MULTI_B] = _separated_vec(1.5, 2)
    table[_MULTI_C] = _separated_vec(1.5, 3)
    table[_MULTI_INCOMING] = [1.0] + [0.0] * (_DIM - 1)
    assessor = _SleepyRelationAssessor(delay_s=0.3)
    engine, _adapter = await _build_engine(table, assessor=assessor)
    a = await engine.save(_MULTI_A, scope="p")
    b = await engine.save(_MULTI_B, scope="p")
    c = await engine.save(_MULTI_C, scope="p")
    assert a.hypomnema is not None
    assert b.hypomnema is not None
    assert c.hypomnema is not None

    start = time.monotonic()
    result = await engine.save(_MULTI_INCOMING, scope="p")
    elapsed = time.monotonic() - start

    assert result.outcome == "bounced"
    assert elapsed < 0.6, f"three concurrent 0.3s calls took {elapsed}s — looks sequential"
    assert len(assessor.calls) == 3
    called_ids = set(assessor.calls)
    assert called_ids == {str(a.hypomnema.id), str(b.hypomnema.id), str(c.hypomnema.id)}


# ---------------------------------------------------------------------------
# A4 bounce-judged piece — clean picture, upward moved composition, caller
# replay generalised to N ids. ``_ScriptedRelationAssessor`` scripts a
# DIFFERENT verdict per candidate text (``EchoAssessor`` cannot — every
# candidate gets the same material).
# ---------------------------------------------------------------------------


class _ScriptedRelationAssessor(_PlaceholderMethods):
    """``relation`` verdict scripted per candidate TEXT — a candidate not named in
    ``script`` gets ``default``. Lets a test give the top and second survivor
    different picks in one save, which ``EchoAssessor`` structurally cannot."""

    def __init__(
        self,
        script: dict[str, tuple[Relation, ReasonClass | None]],
        *,
        default: tuple[Relation, ReasonClass | None] = ("distinct", None),
    ) -> None:
        self._script = script
        self._default = default
        self.calls: list[str] = []

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        self.calls.append(candidate.content)
        pick, reason_class = self._script.get(candidate.content, self._default)
        reason_odds: dict[str, float] | None = (
            {reason_class: 1.0} if reason_class is not None else None
        )
        return RelationVerdict(
            method="relation", pick=pick, odds={pick: 1.0}, confidence=1.0,
            trace=_placeholder_trace(), reason_class=reason_class, reason_odds=reason_odds,
        )


#: Two-survivor fixture: TOP always ranks above SECOND (cosine 0.894 vs 0.832 against
#: ``_TWO_SURVIVOR_INCOMING``, both above ``gray_floor`` 0.80), same word-overlap template as
#: ``_separated_vec``'s own docstring — BM25 sees every text as a near-dup of the others.
_TOP_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel top"
_SECOND_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel second"
_TWO_SURVIVOR_INCOMING = "alpha bravo charlie delta echo foxtrot golf hotel incoming"


def _two_survivor_table() -> dict[str, list[float]]:
    table = dict(_DEFAULT_TABLE)
    table[_TOP_TEXT] = _separated_vec(2.0, 1)
    table[_SECOND_TEXT] = _separated_vec(1.5, 2)
    table[_TWO_SURVIVOR_INCOMING] = [1.0] + [0.0] * (_DIM - 1)
    return table


class _ReplaceCountingAdapter:
    """Delegates every call to ``inner`` via ``__getattr__``, counting ``replace`` calls —
    proves a composed moved set lands in exactly one ``replace`` (review stage)."""

    def __init__(self, inner: StorageAdapter) -> None:
        self._inner = inner
        self.replace_calls = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != "replace" or not callable(attr):
            return attr

        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            self.replace_calls += 1
            return await attr(*args, **kwargs)

        return _wrapped


async def test_clean_picture_top_restatement_second_replacement_competes() -> None:
    """Observable: two survivors, top restatement switched on, second replacement:
    bounce, competing_candidate, both odds visible on other_candidates."""
    table = _two_survivor_table()
    assessor = _ScriptedRelationAssessor(
        {_TOP_TEXT: ("restatement", None), _SECOND_TEXT: ("replacement", "correction")}
    )
    engine, adapter = await _build_engine(
        table, assessor=assessor, auto_exit_restatement=True, auto_exit_replacement=True,
    )
    top = await engine.save(_TOP_TEXT, scope="p")
    second = await engine.save(_SECOND_TEXT, scope="p")
    assert top.hypomnema is not None
    assert second.hypomnema is not None

    result = await engine.save(_TWO_SURVIVOR_INCOMING, scope="p")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.withheld == "competing_candidate"
    assert result.bounce.judged is not None
    top_judged = result.bounce.judged["relation"]
    assert isinstance(top_judged, JudgedVerdict)
    assert top_judged.pick == "restatement"
    assert len(result.bounce.other_candidates) == 1
    second_judged = result.bounce.other_candidates[0].judged
    assert second_judged is not None
    second_relation = second_judged["relation"]
    assert isinstance(second_relation, JudgedVerdict)
    assert second_relation.pick == "replacement"

    # Review stage: assert on the STORE, not only the result — an automatic exit
    # never fires when a non-top survivor competes, so neither candidate is touched.
    top_row = await adapter.get_by_id(top.hypomnema.id)
    second_row = await adapter.get_by_id(second.hypomnema.id)
    assert top_row.superseded_by is None
    assert second_row.superseded_by is None
    row_count = 0
    async for _ in await adapter.scan_records(lifecycle_in=None):
        row_count += 1
    assert row_count == 2


async def test_clean_picture_top_restatement_second_distinct_still_reinforces() -> None:
    """Observable: the same, but the second is distinct: auto-reinforce on the top
    candidate — an unrelated second survivor never blocks a clean picture."""
    table = _two_survivor_table()
    assessor = _ScriptedRelationAssessor(
        {_TOP_TEXT: ("restatement", None), _SECOND_TEXT: ("distinct", None)}
    )
    engine, _adapter = await _build_engine(
        table, assessor=assessor, auto_exit_restatement=True
    )
    top = await engine.save(_TOP_TEXT, scope="p")
    await engine.save(_SECOND_TEXT, scope="p")
    assert top.hypomnema is not None

    result = await engine.save(_TWO_SURVIVOR_INCOMING, scope="p")

    assert result.outcome == "reinforced"
    assert result.reinforced_id == top.hypomnema.id


async def test_upward_promotion_two_candidates_composes_one_replace() -> None:
    """Piece Verify: upward, A in p, B in q, both restatement of an incoming global
    text, switch on: supersedes both with moved in one replace call."""
    table = _two_survivor_table()
    assessor = _ScriptedRelationAssessor(
        {_TOP_TEXT: ("restatement", None), _SECOND_TEXT: ("restatement", None)}
    )
    inner = MemoryAdapter(embedding_model="test/auto-exits", dim=_DIM, clock=TestClock(_EPOCH))
    counting = _ReplaceCountingAdapter(inner)
    embedder = _TableEmbedder(table)
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_restatement=True)
    engine = Tabularium(
        cast(StorageAdapter, counting), embedder, clock, cfg, assessor=assessor
    )
    await engine.initialize()
    await inner.create_scope(Scope("p"))
    await inner.create_scope(Scope("q"))
    a = await engine.save(_TOP_TEXT, scope="p")
    b = await engine.save(_SECOND_TEXT, scope="q")
    assert a.hypomnema is not None
    assert b.hypomnema is not None

    result = await engine.save(_TWO_SURVIVOR_INCOMING, scope="global")

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    a_row = await inner.get_by_id(a.hypomnema.id)
    b_row = await inner.get_by_id(b.hypomnema.id)
    assert a_row.superseded_by == result.hypomnema.id
    assert a_row.supersession_reason == "moved"
    assert b_row.superseded_by == result.hypomnema.id
    assert b_row.supersession_reason == "moved"
    assert counting.replace_calls == 1


async def test_upward_promotion_switch_off_bounces_moved_recommended() -> None:
    """Piece Verify: the same, switch off: bounces with that supersession recommended
    (D9: listed first) and confirm_distinct the other."""
    table = _two_survivor_table()
    assessor = _ScriptedRelationAssessor(
        {_TOP_TEXT: ("restatement", None), _SECOND_TEXT: ("restatement", None)}
    )
    engine, _adapter = await _build_engine(table, assessor=assessor)
    a = await engine.save(_TOP_TEXT, scope="p")
    b = await engine.save(_SECOND_TEXT, scope="q")
    assert a.hypomnema is not None
    assert b.hypomnema is not None

    result = await engine.save(_TWO_SURVIVOR_INCOMING, scope="global")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.withheld == "switch_off"
    exits = result.bounce.exits
    assert len(exits) == 2
    assert exits[0].action == "supersede"
    assert exits[1].action == "confirm_distinct"
    assert str(a.hypomnema.id) in exits[0].how
    assert str(b.hypomnema.id) in exits[0].how
    assert "reason=\"moved\"" in exits[0].how


_A2_A = "alpha bravo charlie delta echo foxtrot golf hotel a2-a"
_A2_C = "alpha bravo charlie delta echo foxtrot golf hotel a2-c"
_A2_INCOMING = "alpha bravo charlie delta echo foxtrot golf hotel a2-incoming"


async def test_caller_replay_plus_independent_auto_exit_stamps_both_rows() -> None:
    """A2 gap test (review stage): a caller replay of A's bounce row PLUS an
    independent auto-exit on a different survivor C compose in one replace, and
    neither verdict row is left ``unanswered``.

    Two engines share one adapter (the SAME pattern ``test_candidate_changed_
    redispatches_fresh`` already uses): the first, switch OFF, produces A's original
    bounce; the second, switch ON, replays it. ``a``/``c`` sit on SEPARATE
    ``_separated_vec`` axes (pairwise cosine 0.692, below ``gray_floor``) so seeding
    ``c`` never itself dispatches against ``a`` — only the incoming pure-axis-0 query
    sees both survive its own probe (cosine 0.832 against each)."""
    table = dict(_DEFAULT_TABLE)
    table[_A2_A] = _separated_vec(1.5, 1)
    table[_A2_C] = _separated_vec(1.5, 2)
    table[_A2_INCOMING] = [1.0] + [0.0] * (_DIM - 1)
    embedder = _TableEmbedder(table)
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    cfg_off = _make_config()
    cfg_on = _make_config(auto_exit_replacement=True)

    bootstrap = Tabularium(
        adapter, embedder, clock, cfg_off, assessor=EchoAssessor("correction")
    )
    await bootstrap.initialize()
    await adapter.create_scope(Scope("p"))
    a = await bootstrap.save(_A2_A, scope="p")
    assert a.hypomnema is not None
    bounced = await bootstrap.save(_A2_INCOMING, scope="p")
    assert bounced.outcome == "bounced"
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id

    engine = Tabularium(adapter, embedder, clock, cfg_on, assessor=EchoAssessor("correction"))
    c = await engine.save(_A2_C, scope="p")
    assert c.hypomnema is not None

    result = await engine.save(
        _A2_INCOMING, scope="p",
        supersedes=[a.hypomnema.id], reason="changed", bounce_id=bounce_id,
    )

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    a_row = await adapter.get_by_id(a.hypomnema.id)
    c_row = await adapter.get_by_id(c.hypomnema.id)
    assert a_row.superseded_by == result.hypomnema.id
    assert a_row.supersession_reason == "changed"
    assert c_row.superseded_by == result.hypomnema.id
    assert c_row.supersession_reason == "correction"

    rows = await adapter.verdicts_by_bounce(bounce_id)
    assert rows
    assert all(row.status != "unanswered" for row in rows)


async def test_multi_id_moved_replay_stamps_both_rows() -> None:
    """Steps: the multi-id ``moved`` replay stamps both rows — a caller replaying the
    upward bounce's own ``how`` verbatim, naming both predecessors, lands one
    ``replace`` and moves both bounce rows off ``unanswered``."""
    table = _two_survivor_table()
    assessor = _ScriptedRelationAssessor(
        {_TOP_TEXT: ("restatement", None), _SECOND_TEXT: ("restatement", None)}
    )
    engine, adapter = await _build_engine(table, assessor=assessor)
    a = await engine.save(_TOP_TEXT, scope="p")
    b = await engine.save(_SECOND_TEXT, scope="q")
    assert a.hypomnema is not None
    assert b.hypomnema is not None

    bounced = await engine.save(_TWO_SURVIVOR_INCOMING, scope="global")
    assert bounced.outcome == "bounced"
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id

    result = await engine.save(
        _TWO_SURVIVOR_INCOMING, scope="global",
        supersedes=[a.hypomnema.id, b.hypomnema.id], reason="moved", bounce_id=bounce_id,
    )

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    a_row = await adapter.get_by_id(a.hypomnema.id)
    b_row = await adapter.get_by_id(b.hypomnema.id)
    assert a_row.superseded_by == result.hypomnema.id
    assert b_row.superseded_by == result.hypomnema.id

    rows = await adapter.verdicts_by_bounce(bounce_id)
    assert rows
    assert all(row.status != "unanswered" for row in rows)


async def test_upward_promotion_torn_write_reconciles_both_rows_acted() -> None:
    """Steps: a torn-write reconcile test for a composed two-target moved replace —
    the stamp raises after ``replace`` lands; a fresh ``initialize()`` stamps both
    ``acted``."""
    table = _two_survivor_table()
    assessor = _ScriptedRelationAssessor(
        {_TOP_TEXT: ("restatement", None), _SECOND_TEXT: ("restatement", None)}
    )
    inner = MemoryAdapter(embedding_model="test/auto-exits", dim=_DIM, clock=TestClock(_EPOCH))
    wrapper = FaultInjectingAdapter(inner)
    embedder = _TableEmbedder(table)
    clock = TestClock(_EPOCH)
    cfg = _make_config(auto_exit_restatement=True)
    engine = Tabularium(as_adapter(wrapper), embedder, clock, cfg, assessor=assessor)
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    await wrapper.create_scope(Scope("q"))
    a = await engine.save(_TOP_TEXT, scope="p")
    b = await engine.save(_SECOND_TEXT, scope="q")
    assert a.hypomnema is not None
    assert b.hypomnema is not None

    wrapper.fault(on="update_verdicts", call=1, exc=RuntimeError("injected"))
    wrapper.arm()
    with pytest.raises(RuntimeError):
        await engine.save(_TWO_SURVIVOR_INCOMING, scope="global")

    pending = await inner.pending_verdicts()
    assert len(pending) == 2
    assert {row.candidate_id for row in pending} == {str(a.hypomnema.id), str(b.hypomnema.id)}

    wrapper.fault(on="update_verdicts", call=999, exc=RuntimeError("never"))
    await engine.initialize()

    a_rows = await inner.verdicts_for_record(a.hypomnema.id)
    b_rows = await inner.verdicts_for_record(b.hypomnema.id)
    assert all(row.status == "acted" for row in a_rows)
    assert all(row.status == "acted" for row in b_rows)


# ---------------------------------------------------------------------------
# Decision arm mirror — upward promotion.
# ---------------------------------------------------------------------------


_TOP_DECISION_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel decision-top"
_SECOND_DECISION_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel decision-second"
_TWO_SURVIVOR_DECISION_INCOMING = (
    "alpha bravo charlie delta echo foxtrot golf hotel decision-incoming"
)


def _two_survivor_decision_table() -> dict[str, list[float]]:
    table = dict(_DEFAULT_TABLE)
    table[_TOP_DECISION_TEXT] = _separated_vec(2.0, 1)
    table[_SECOND_DECISION_TEXT] = _separated_vec(1.5, 2)
    table[_TWO_SURVIVOR_DECISION_INCOMING] = [1.0] + [0.0] * (_DIM - 1)
    return table


async def test_decision_arm_upward_promotion_two_candidates_bounces_competing() -> None:
    """Owner ruling (2026-09-24): the Decision arm caps the upward ``moved`` set at
    one row — its ``save(supersedes=...)`` takes a single id, and ``unsupersede``
    deletes the successor row, so a successor shared by several predecessors could
    never be undone. Two same-kind Decision upward restatements, switch ON, never
    auto-promote: the save bounces ``competing_candidate`` instead, nothing is
    written, the ``supersede`` exit names only the top survivor, and both survivors'
    own judged odds stay visible."""
    table = _two_survivor_decision_table()
    assessor = _ScriptedRelationAssessor(
        {
            _TOP_DECISION_TEXT: ("restatement", None),
            _SECOND_DECISION_TEXT: ("restatement", None),
        }
    )
    engine, adapter = await _build_engine(
        table, assessor=assessor, auto_exit_restatement=True, decision_gray_floor=0.70,
    )
    a = await engine.save(_TOP_DECISION_TEXT, scope="p", kind="decision")
    b = await engine.save(_SECOND_DECISION_TEXT, scope="q", kind="decision")
    assert a.hypomnema is not None
    assert b.hypomnema is not None

    result = await engine.save(
        _TWO_SURVIVOR_DECISION_INCOMING, scope="global", kind="decision"
    )

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert isinstance(result.bounce, DecisionBounce)
    assert result.bounce.withheld == "competing_candidate"
    exits = result.bounce.exits
    assert len(exits) == 2
    assert exits[0].action == "supersede"
    assert exits[1].action == "confirm_distinct"
    assert str(a.hypomnema.id) in exits[0].how
    assert str(b.hypomnema.id) not in exits[0].how
    assert 'reason="moved"' in exits[0].how
    top_judged = result.bounce.judged
    assert top_judged is not None
    top_relation = top_judged["relation"]
    assert isinstance(top_relation, JudgedVerdict)
    assert top_relation.pick == "restatement"
    assert len(result.bounce.other_candidates) == 1
    second_judged = result.bounce.other_candidates[0].judged
    assert second_judged is not None
    second_relation = second_judged["relation"]
    assert isinstance(second_relation, JudgedVerdict)
    assert second_relation.pick == "restatement"

    a_row = await adapter.get_by_id(a.hypomnema.id)
    b_row = await adapter.get_by_id(b.hypomnema.id)
    assert a_row.superseded_by is None
    assert b_row.superseded_by is None
    row_count = 0
    async for _ in await adapter.scan_records(lifecycle_in=None):
        row_count += 1
    assert row_count == 2


async def test_decision_arm_upward_promotion_two_candidates_moved_replay_stamps_top() -> None:
    """Replaying the ``competing_candidate`` bounce's own ``moved`` ``how`` — the
    Decision arm's single-id shape, naming only the top survivor — supersedes that
    one row and stamps that one bounce row acted. The second survivor, excluded from
    THIS call's own probe by the caller's own named ``supersedes``, is independently
    ``Promote``-eligible again on its own: composing that
    auto flip with the caller's own would land two live predecessors on one
    successor, the exact shape the cap forbids. It must stay untouched, and undoing
    the caller's own supersession of the top must still succeed."""
    table = _two_survivor_decision_table()
    assessor = _ScriptedRelationAssessor(
        {
            _TOP_DECISION_TEXT: ("restatement", None),
            _SECOND_DECISION_TEXT: ("restatement", None),
        }
    )
    engine, adapter = await _build_engine(
        table, assessor=assessor, auto_exit_restatement=True, decision_gray_floor=0.70,
    )
    a = await engine.save(_TOP_DECISION_TEXT, scope="p", kind="decision")
    b = await engine.save(_SECOND_DECISION_TEXT, scope="q", kind="decision")
    assert a.hypomnema is not None
    assert b.hypomnema is not None

    bounced = await engine.save(
        _TWO_SURVIVOR_DECISION_INCOMING, scope="global", kind="decision"
    )
    assert bounced.outcome == "bounced"
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id

    result = await engine.save(
        _TWO_SURVIVOR_DECISION_INCOMING, scope="global", kind="decision",
        supersedes=a.hypomnema.id, reason="moved", bounce_id=bounce_id,
    )

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    a_row = await adapter.get_by_id(a.hypomnema.id)
    b_row = await adapter.get_by_id(b.hypomnema.id)
    assert a_row.superseded_by == result.hypomnema.id
    assert a_row.supersession_reason == "moved"
    assert b_row.superseded_by is None

    rows = await adapter.verdicts_by_bounce(bounce_id)
    a_rows = [row for row in rows if row.candidate_id == str(a.hypomnema.id)]
    assert a_rows
    assert all(row.status == "acted" for row in a_rows)

    undo = await engine.unsupersede(a.hypomnema.id, reason="test undo of a single promotion")
    assert undo.shape == "decision_restored"
    restored = await adapter.get_by_id(a.hypomnema.id)
    assert restored.superseded_by is None


_MOVED_EXACT_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel moved-exact"
_MOVED_SIBLING_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel moved-sibling"


def _moved_exact_sibling_table() -> dict[str, list[float]]:
    table = dict(_DEFAULT_TABLE)
    table[_MOVED_EXACT_TEXT] = [1.0] + [0.0] * (_DIM - 1)
    table[_MOVED_SIBLING_TEXT] = _separated_vec(2.0, 1)
    return table


async def test_decision_moved_replay_never_composes_with_independent_supersede() -> None:
    """Include
    automatic Supersede in the caller-promotion cap: replaying a byte-identical
    upward ``moved`` exit must not compose with an UNRELATED same-scope
    ``Supersede`` auto-exit found once the named predecessor is excluded from the
    probe — that would land the moved-promoted row and an unrelated one on the
    same successor, exactly as un-undoable as the ``Moved``-composition
    gap, just reached through a different auto-exit type."""
    table = _moved_exact_sibling_table()
    assessor = _ScriptedRelationAssessor({_MOVED_SIBLING_TEXT: ("replacement", "changed")})
    engine, adapter = await _build_engine(
        table, assessor=assessor,
        auto_exit_replacement=True, auto_exit_distinct=True, decision_gray_floor=0.70,
    )
    a = await engine.save(_MOVED_EXACT_TEXT, scope="p", kind="decision")
    b = await engine.save(_MOVED_SIBLING_TEXT, scope="global", kind="decision")
    assert a.hypomnema is not None
    assert b.hypomnema is not None

    bounced = await engine.save(_MOVED_EXACT_TEXT, scope="global", kind="decision")
    assert bounced.outcome == "bounced"
    assert bounced.bounce is not None
    bounce_id = bounced.bounce.bounce_id
    exits = bounced.bounce.exits
    assert len(exits) == 1
    assert exits[0].action == "supersede"

    result = await engine.save(
        _MOVED_EXACT_TEXT, scope="global", kind="decision",
        supersedes=a.hypomnema.id, reason="moved", bounce_id=bounce_id,
    )

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    a_row = await adapter.get_by_id(a.hypomnema.id)
    b_row = await adapter.get_by_id(b.hypomnema.id)
    assert a_row.superseded_by == result.hypomnema.id
    assert a_row.supersession_reason == "moved"
    assert b_row.superseded_by is None

    undo = await engine.unsupersede(a.hypomnema.id, reason="test undo of a single promotion")
    assert undo.shape == "decision_restored"
    restored = await adapter.get_by_id(a.hypomnema.id)
    assert restored.superseded_by is None


_ASYM_C_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel asym-c"
_ASYM_A_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel asym-a"
_ASYM_INCOMING_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel asym-incoming"


def _asym_composition_table() -> dict[str, list[float]]:
    table = dict(_DEFAULT_TABLE)
    table[_ASYM_C_TEXT] = _separated_vec(1.6, 1)
    table[_ASYM_A_TEXT] = _separated_vec(1.4, 2)
    table[_ASYM_INCOMING_TEXT] = [1.0] + [0.0] * (_DIM - 1)
    return table


async def test_decision_ordinary_replay_never_composes_with_independent_moved() -> None:
    """Also block
    composition when the automatic exit is Moved: the mirror of the previous test. Replaying
    an ORDINARY (non-``moved``) supersede exit against candidate C must not compose
    with a DIFFERENT survivor's own independent ``Moved`` promotion, found once C is
    excluded from the probe — that would land C's unrelated flip beside a promoted
    row on one successor, un-undoable for the promoted predecessor exactly as
    rounds 1 and 2 forbid, just with the ``moved`` tag on the auto side this time."""
    table = _asym_composition_table()
    assessor = _ScriptedRelationAssessor(
        {_ASYM_C_TEXT: ("replacement", "changed"), _ASYM_A_TEXT: ("restatement", None)}
    )
    engine, adapter = await _build_engine(
        table, assessor=assessor, auto_exit_restatement=True, decision_gray_floor=0.70,
    )
    a = await engine.save(_ASYM_A_TEXT, scope="p", kind="decision")
    c = await engine.save(_ASYM_C_TEXT, scope="global", kind="decision")
    assert a.hypomnema is not None
    assert c.hypomnema is not None

    bounced = await engine.save(_ASYM_INCOMING_TEXT, scope="global", kind="decision")
    assert bounced.outcome == "bounced"
    assert bounced.bounce is not None
    assert bounced.bounce.withheld == "switch_off"
    bounce_id = bounced.bounce.bounce_id
    exits = bounced.bounce.exits
    supersede_exits = [exit_ for exit_ in exits if exit_.action == "supersede"]
    assert len(supersede_exits) == 1
    assert str(c.hypomnema.id) in supersede_exits[0].how
    assert str(a.hypomnema.id) not in supersede_exits[0].how

    result = await engine.save(
        _ASYM_INCOMING_TEXT, scope="global", kind="decision",
        supersedes=c.hypomnema.id, reason="changed", bounce_id=bounce_id,
    )

    assert result.outcome == "saved"
    assert result.hypomnema is not None
    c_row = await adapter.get_by_id(c.hypomnema.id)
    a_row = await adapter.get_by_id(a.hypomnema.id)
    assert c_row.superseded_by == result.hypomnema.id
    assert c_row.supersession_reason == "changed"
    assert a_row.superseded_by is None

    undo = await engine.unsupersede(c.hypomnema.id, reason="test undo of an ordinary supersede")
    assert undo.shape == "decision_restored"
    restored = await adapter.get_by_id(c.hypomnema.id)
    assert restored.superseded_by is None


async def test_decision_arm_upward_promotion_single_candidate_switch_off_bounce() -> None:
    """Decision-arm mirror of the switch-off upward bounce, single candidate — the
    ONE shape the Decision arm's own ``supersedes: HypomnemaId | None`` singular
    parameter can actually replay (see this brief's own deviations for the
    multi-candidate Decision moved bounce's ``how`` representability gap)."""
    engine, _adapter = await _build_engine(
        assessor=EchoAssessor("repetition"),
    )
    existing = await engine.save(_EXISTING, scope="p", kind="decision")
    assert existing.hypomnema is not None

    result = await engine.save(_INCOMING, scope="global", kind="decision")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert isinstance(result.bounce, DecisionBounce)
    assert result.bounce.withheld == "switch_off"
    exits = result.bounce.exits
    assert len(exits) == 2
    assert exits[0].action == "supersede"
    assert exits[1].action == "confirm_distinct"
    assert str(existing.hypomnema.id) in exits[0].how


# ---------------------------------------------------------------------------
# A non-top moved
# candidate edited during assessment, and a promotion-then-undo covering
# shared successors.
# ---------------------------------------------------------------------------

_EDITED_SECOND_TEXT = "zulu yankee xray whiskey victor uniform tango sierra edited"


class _EditSiblingDuringMovedAssessor(_PlaceholderMethods):
    """Edits the SECOND candidate's content as a side effect of ITS OWN ``relation``
    call — simulating a concurrent edit landing on a non-top moved-set member while
    the Assessor runs outside the lock (D12). Proves the ``Moved`` branch
    re-validates every candidate's freshness before composing them into one write."""

    def __init__(
        self,
        adapter_box: list[MemoryAdapter],
        second_id_box: list[HypomnemaId],
        edited_vector: list[float],
    ) -> None:
        self._adapter_box = adapter_box
        self._second_id_box = second_id_box
        self._edited_vector = edited_vector
        self._triggered = False

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        if candidate.content == _SECOND_TEXT and not self._triggered:
            self._triggered = True
            await edit_content(
                self._adapter_box[0],
                self._second_id_box[0],
                _EDITED_SECOND_TEXT,
                self._edited_vector,
            )
        return RelationVerdict(
            method="relation", pick="restatement", odds={"restatement": 1.0}, confidence=1.0,
            trace=_placeholder_trace(),
        )


async def test_moved_candidate_edited_during_assessment_never_silently_retired() -> None:
    """A non-top moved-set candidate edited
    while the Assessor runs must not be superseded under a verdict about its OLD
    text — the save redispatches fresh instead (``candidate_changed``), and neither
    candidate is touched."""
    table = _two_survivor_table()
    table[_EDITED_SECOND_TEXT] = _separated_vec(1.5, 5)
    adapter_box: list[MemoryAdapter] = []
    second_id_box: list[HypomnemaId] = []
    assessor = _EditSiblingDuringMovedAssessor(
        adapter_box, second_id_box, table[_EDITED_SECOND_TEXT]
    )
    engine, adapter = await _build_engine(table, assessor=assessor, auto_exit_restatement=True)
    adapter_box.append(adapter)
    top = await engine.save(_TOP_TEXT, scope="p")
    second = await engine.save(_SECOND_TEXT, scope="q")
    assert top.hypomnema is not None
    assert second.hypomnema is not None
    second_id_box.append(second.hypomnema.id)

    result = await engine.save(_TWO_SURVIVOR_INCOMING, scope="global")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    assert result.bounce.withheld == "candidate_changed"
    top_row = await adapter.get_by_id(top.hypomnema.id)
    second_row = await adapter.get_by_id(second.hypomnema.id)
    assert top_row.superseded_by is None
    assert second_row.superseded_by is None
    assert second_row.content == _EDITED_SECOND_TEXT


async def test_decision_undo_shared_successor_refuses_not_corrupts() -> None:
    """Undoing one predecessor of a
    Decision successor shared with a sibling predecessor must refuse cleanly,
    never delete the successor out from under the sibling. Owner ruling
    (2026-09-24) caps the Decision arm's upward ``moved`` promotion at one row, so
    this shared-successor state can no longer arise through promotion — built
    directly at the storage seam instead, the same shape ``replace`` still accepts
    from any other two-predecessor caller."""
    table = _two_survivor_decision_table()
    engine, adapter = await _build_engine(table, assessor=EchoAssessor("repetition"))
    a = await engine.save(_TOP_DECISION_TEXT, scope="p", kind="decision")
    b = await engine.save(_SECOND_DECISION_TEXT, scope="q", kind="decision")
    assert a.hypomnema is not None
    assert b.hypomnema is not None

    successor = Decision(
        id=new_id(), type="decision", content=_TWO_SURVIVOR_DECISION_INCOMING,
        scope=Scope("global"), lifecycle="in_force",
        created_at=_EPOCH, updated_at=_EPOCH,
    )
    flips = [
        PredecessorFlip(id=a.hypomnema.id, reason="moved", note=None, edge_id=str(uuid4())),
        PredecessorFlip(id=b.hypomnema.id, reason="moved", note=None, edge_id=str(uuid4())),
    ]
    await adapter.replace(successor, None, flips, at=_EPOCH)

    with pytest.raises(ToolError):
        await engine.unsupersede(a.hypomnema.id, reason="test undo of a shared successor")

    a_row = await adapter.get_by_id(a.hypomnema.id)
    b_row = await adapter.get_by_id(b.hypomnema.id)
    successor_row = await adapter.get_by_id(successor.id)
    assert a_row.superseded_by == successor.id
    assert b_row.superseded_by == successor.id
    assert successor_row.lifecycle == "in_force"


# ---------------------------------------------------------------------------
# An unavailable primary
# verdict must not hide a sibling's own real verdict.
# ---------------------------------------------------------------------------


class _TopFailsSecondAnswersAssessor(_PlaceholderMethods):
    """The TOP candidate's own ``relation`` call raises (mapped to ``Unavailable`` by
    ``engine.admission.ask``); every OTHER candidate answers for real. Proves a
    sibling's verdict still reaches its own ``other_candidates[i].judged`` even when
    the primary assessment never resolves."""

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        if candidate.content == _TOP_TEXT:
            raise RuntimeError("injected: top assessment unavailable")
        return RelationVerdict(
            method="relation", pick="distinct", odds={"distinct": 1.0}, confidence=1.0,
            trace=_placeholder_trace(),
        )


async def test_unavailable_primary_still_carries_sibling_judged() -> None:
    table = _two_survivor_table()
    engine, _adapter = await _build_engine(table, assessor=_TopFailsSecondAnswersAssessor())
    await engine.save(_TOP_TEXT, scope="p")
    await engine.save(_SECOND_TEXT, scope="p")

    result = await engine.save(_TWO_SURVIVOR_INCOMING, scope="p")

    assert result.outcome == "bounced"
    assert result.bounce is not None
    top_judged = result.bounce.judged
    assert top_judged is not None
    assert top_judged["relation"].status == "unavailable"
    assert len(result.bounce.other_candidates) == 1
    second_judged = result.bounce.other_candidates[0].judged
    assert second_judged is not None
    second_relation = second_judged["relation"]
    assert isinstance(second_relation, JudgedVerdict)
    assert second_relation.pick == "distinct"
