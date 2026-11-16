"""``DecisionView``/render invalidation-cause read contract (P9 invalid-Decision
read contract, decisions D7/D8). Every case drives P8's real Fact-invalidates-
Decision write path (``EchoAssessor("correction")``, ``auto_exit_replacement=True``,
a Decision save then a Fact save) and P8's ``revalidate`` op through
``set_validity`` — never by setting the invalidation columns by hand.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, cast

from mcp.server.lowlevel.helper_types import ReadResourceContents

from tabularium.config import EngineConfig
from tabularium.domain import Decision, HypomnemaId, Scope
from tabularium.engine import Tabularium
from tabularium.render import render_decision
from tabularium.results import DecisionView, RevalidationRecord
from tabularium.server import make_app
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import TestClock
from tests.support.fake_assessor import EchoAssessor

_DIM = 16
_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)

_D = "the quick brown fox jumps over the lazy dog"
_F1 = "the quick brown fox jumps over the lazy hound"
_F2 = "the quick brown fox jumps over the lazy jackal"
_SUCCESSOR = "a wholly rewritten policy about deployment cadence, revised"
_UNTOUCHED = "a plain untouched decision"


def _s(result: Any) -> dict[str, Any]:
    """Extract the structured output dict from a ``call_tool`` result (same
    helper as ``tests/test_server.py``'s own ``_s``)."""
    return cast(dict[str, Any], result.structured_content)


def _angle_vec(angle_deg: float, dim: int = _DIM) -> list[float]:
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


class _TableEmbedder:
    dim: int = _DIM
    model: str = "test/decision-view-invalidation"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


_TABLE: dict[str, list[float]] = {
    _D: _angle_vec(0),
    _F1: _angle_vec(5),
    # A4 bounce-judged piece: -35 degrees keeps _F2 above D's own decision_gray_floor
    # (cos 35 deg ~= 0.819 > 0.70) while dropping it below F1's regular gray_floor
    # (cos 40 deg ~= 0.766 < 0.80) — F1 must NOT survive _F2's own probe, or its own
    # "replacement" verdict (EchoAssessor gives every candidate the same pick) reads
    # as a competing candidate (decision D16) and bounces instead of invalidating D.
    _F2: _angle_vec(-35),
    _SUCCESSOR: _angle_vec(200),
    _UNTOUCHED: _angle_vec(90),
}


async def _build_engine() -> tuple[Tabularium, MemoryAdapter]:
    embedder = _TableEmbedder(dict(_TABLE))
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    cfg = EngineConfig(
        candidate_k=1000,
        gray_floor=0.80,
        decision_gray_floor=0.70,
        auto_exit_replacement=True,
    )
    engine = Tabularium(adapter, embedder, clock, cfg, assessor=EchoAssessor("correction"))
    await engine.initialize()
    await adapter.create_scope(Scope("p"))
    return engine, adapter


async def _invalidate_d(engine: Tabularium) -> tuple[HypomnemaId, HypomnemaId]:
    """Decision D saved, then Fact F1 saved with the Assessor judging
    ``replacement``/``correction`` against it (P8's real write path)."""
    d = await engine.save(_D, scope="p", kind="decision")
    assert d.hypomnema is not None
    fact = await engine.save(_F1, scope="p")
    assert fact.hypomnema is not None
    assert fact.outcome == "saved"
    return d.hypomnema.id, fact.hypomnema.id


async def test_decision_view_shows_invalidation_cause_and_class() -> None:
    engine, _adapter = await _build_engine()
    d_id, fact_id = await _invalidate_d(engine)

    view = await engine.decision(d_id)

    assert view.invalidated_by == str(fact_id)
    assert view.invalidation_reason == "correction"
    assert view.revalidations == []
    assert f"Invalidated by: `{fact_id!s}` (correction)" in view.rendered
    superseded_idx = view.rendered.find("**Superseded by:**")
    invalidated_idx = view.rendered.find("Invalidated by:")
    assert superseded_idx == -1  # no supersession on this row
    assert invalidated_idx != -1


async def test_never_invalidated_decision_has_no_lineage_lines() -> None:
    engine, _adapter = await _build_engine()
    r = await engine.save(_UNTOUCHED, scope="p", kind="decision")
    assert r.hypomnema is not None

    view = await engine.decision(r.hypomnema.id)

    assert view.invalidated_by is None
    assert view.invalidation_reason is None
    assert view.revalidations == []
    assert "Invalidated by:" not in view.rendered
    assert "Revalidated" not in view.rendered


async def test_revalidation_clears_fields_and_appends_rendered_line() -> None:
    engine, _adapter = await _build_engine()
    d_id, fact_id = await _invalidate_d(engine)

    result = await engine.set_validity(d_id, valid=True, reason="reviewed")
    assert result.outcome == "revised"

    view = await engine.decision(d_id)
    assert view.validity == "valid"
    assert view.invalidated_by is None
    assert view.invalidation_reason is None
    assert len(view.revalidations) == 1
    record = view.revalidations[0]
    assert record.withdrawn_cause == fact_id
    assert record.withdrawn_reason == "correction"
    assert record.reason == "reviewed"

    assert "Invalidated by:" not in view.rendered
    dated = record.at.strftime("%Y-%m-%d")
    expected_line = f"Revalidated {dated}: withdrew correction by `{fact_id!s}` (reviewed)"
    assert expected_line in view.rendered

    view_again = await engine.decision(d_id)
    assert view_again.rendered == view.rendered


async def test_second_invalidation_and_revalidation_appends_second_entry() -> None:
    engine, adapter = await _build_engine()
    d_id, fact1_id = await _invalidate_d(engine)
    await engine.set_validity(d_id, valid=True, reason="reviewed once")

    fact2 = await engine.save(_F2, scope="p")
    assert fact2.hypomnema is not None
    assert fact2.outcome == "saved"
    fact2_id = fact2.hypomnema.id

    decision = await adapter.get_by_id(d_id)
    assert isinstance(decision, Decision)
    assert decision.invalidated_by == str(fact2_id)

    await engine.set_validity(d_id, valid=True, reason="reviewed twice")

    view = await engine.decision(d_id)
    assert view.invalidated_by is None
    assert len(view.revalidations) == 2
    first, second = view.revalidations
    assert first.withdrawn_cause == fact1_id
    assert first.reason == "reviewed once"
    assert second.withdrawn_cause == fact2_id
    assert second.reason == "reviewed twice"
    assert view.rendered.index(f"by `{fact1_id!s}`") < view.rendered.index(f"by `{fact2_id!s}`")


async def test_independence_supersede_while_invalid_then_revalidate() -> None:
    engine, adapter = await _build_engine()
    d_id, fact_id = await _invalidate_d(engine)

    successor = await engine.save(
        _SUCCESSOR, scope="p", kind="decision", supersedes=d_id, reason="changed"
    )
    assert successor.hypomnema is not None

    view = await engine.decision(d_id)
    assert view.superseded_by == successor.hypomnema.id
    assert view.invalidated_by == str(fact_id)
    assert view.invalidation_reason == "correction"
    superseded_idx = view.rendered.find("**Superseded by:**")
    invalidated_idx = view.rendered.find("Invalidated by:")
    assert superseded_idx != -1
    assert invalidated_idx != -1
    assert superseded_idx < invalidated_idx

    result = await engine.set_validity(d_id, valid=True, reason="reviewed after supersede")
    assert result.outcome == "revised"

    after = await adapter.get_by_id(d_id)
    assert isinstance(after, Decision)
    assert after.superseded_by == successor.hypomnema.id
    assert after.supersession_reason == "changed"
    assert after.validity == "valid"
    assert after.invalidated_by is None
    assert after.invalidation_reason is None


async def test_resource_identity_for_invalidated_decision() -> None:
    engine, _adapter = await _build_engine()
    d_id, _fact_id = await _invalidate_d(engine)

    app = make_app(engine)
    dv_raw = await app.call_tool("decision", {"id": str(d_id)})
    view = DecisionView(**_s(dv_raw))

    resource_contents = cast(
        list[ReadResourceContents], list(await app.read_resource(f"decision://{d_id!s}"))
    )
    assert len(resource_contents) == 1
    assert view.rendered == resource_contents[0].content
    assert "Invalidated by:" in view.rendered


async def test_render_direct_call_matches_view_for_invalidated_decision() -> None:
    engine, adapter = await _build_engine()
    d_id, _fact_id = await _invalidate_d(engine)

    view = await engine.decision(d_id)
    rec = await adapter.get_by_id(d_id)
    assert isinstance(rec, Decision)

    events = await adapter.list_revalidations(d_id)
    revalidations = [
        RevalidationRecord(
            withdrawn_cause=ev.withdrawn_cause,
            withdrawn_reason=ev.withdrawn_reason,
            reason=ev.reason,
            at=ev.at,
        )
        for ev in sorted(events, key=lambda e: e.at)
    ]
    assert view.rendered == render_decision(rec, revalidations=revalidations)
