"""Fact-invalidates-Decision preservation rule (P8 assessor-automatic-exits piece,
decisions D13/D14): supersession, revoke and unsupersede of an invalidated Decision leave
``invalidated_by``/``invalidation_reason`` untouched; only revalidation (``set_validity``
routed through ``revalidate``) clears them.

The invalidation itself (via the Assessor's automatic ``replacement`` exit) is proven in
``tests/test_auto_exits.py``; this file seeds it directly at the storage layer so each test
is about what the disposition family does to that state AFTER, not about how it arrived.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from tabularium.config import EngineConfig
from tabularium.domain import Decision, HypomnemaId, Scope
from tabularium.engine import Tabularium
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import DecisionInvalidation
from tests.support.doubles import FakeEmbedder, TestClock

_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)


async def _build_engine() -> tuple[Tabularium, MemoryAdapter]:
    embedder = FakeEmbedder(dim=16)
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    engine = Tabularium(adapter, embedder, clock, EngineConfig())
    await engine.initialize()
    await adapter.create_scope(Scope("p"))
    return engine, adapter


async def _invalidate(
    adapter: MemoryAdapter,
    decision_id: HypomnemaId,
    fact_id: HypomnemaId,
    *,
    reason: Literal["correction", "changed"] = "correction",
) -> None:
    fresh_fact = await adapter.get_by_id(fact_id)
    await adapter.replace(
        fresh_fact,
        None,
        [],
        invalidations=[DecisionInvalidation(id=decision_id, reason=reason)],
        at=_EPOCH,
    )


async def test_supersede_preserves_invalidity() -> None:
    engine, adapter = await _build_engine()
    d1 = await engine.save("first decision", kind="decision", scope="p")
    fact = await engine.save("the invalidating fact", scope="p")
    assert d1.hypomnema is not None
    assert fact.hypomnema is not None
    await _invalidate(adapter, d1.hypomnema.id, fact.hypomnema.id)

    d2 = await engine.save(
        "second decision replacing the first",
        kind="decision",
        scope="p",
        supersedes=d1.hypomnema.id,
        reason="elaborated",
    )
    assert d2.hypomnema is not None

    predecessor = await adapter.get_by_id(d1.hypomnema.id)
    assert isinstance(predecessor, Decision)
    assert predecessor.lifecycle == "superseded"
    assert predecessor.superseded_by == d2.hypomnema.id
    assert predecessor.validity == "invalid"
    assert predecessor.invalidated_by == str(fact.hypomnema.id)
    assert predecessor.invalidation_reason == "correction"


async def test_revoke_preserves_invalidity() -> None:
    engine, adapter = await _build_engine()
    d = await engine.save("decision to be revoked", kind="decision", scope="p")
    fact = await engine.save("the invalidating fact", scope="p")
    assert d.hypomnema is not None
    assert fact.hypomnema is not None
    await _invalidate(adapter, d.hypomnema.id, fact.hypomnema.id, reason="changed")

    await engine.revoke(d.hypomnema.id, reason="no longer applicable")

    revoked = await adapter.get_by_id(d.hypomnema.id)
    assert isinstance(revoked, Decision)
    assert revoked.lifecycle == "revoked"
    assert revoked.validity == "invalid"
    assert revoked.invalidated_by == str(fact.hypomnema.id)
    assert revoked.invalidation_reason == "changed"


async def test_unsupersede_preserves_invalidity() -> None:
    engine, adapter = await _build_engine()
    d = await engine.save("decision to be superseded then restored", kind="decision", scope="p")
    fact = await engine.save("the invalidating fact", scope="p")
    assert d.hypomnema is not None
    assert fact.hypomnema is not None
    await _invalidate(adapter, d.hypomnema.id, fact.hypomnema.id)

    successor = await engine.save(
        "successor decision",
        kind="decision",
        scope="p",
        supersedes=d.hypomnema.id,
        reason="correction",
    )
    assert successor.hypomnema is not None

    await engine.unsupersede(d.hypomnema.id, reason="undo the supersession")

    restored = await adapter.get_by_id(d.hypomnema.id)
    assert isinstance(restored, Decision)
    assert restored.lifecycle == "in_force"
    assert restored.superseded_by is None
    assert restored.validity == "invalid"
    assert restored.invalidated_by == str(fact.hypomnema.id)
    assert restored.invalidation_reason == "correction"


async def test_set_validity_false_after_invalidation_keeps_columns() -> None:
    engine, adapter = await _build_engine()
    d = await engine.save("invalidated decision", kind="decision", scope="p")
    fact = await engine.save("the invalidating fact", scope="p")
    assert d.hypomnema is not None
    assert fact.hypomnema is not None
    await _invalidate(adapter, d.hypomnema.id, fact.hypomnema.id)

    result = await engine.set_validity(d.hypomnema.id, valid=False, reason="still invalid")

    assert result.outcome == "unchanged"
    unchanged = await adapter.get_by_id(d.hypomnema.id)
    assert isinstance(unchanged, Decision)
    assert unchanged.validity == "invalid"
    assert unchanged.invalidated_by == str(fact.hypomnema.id)
    assert unchanged.invalidation_reason == "correction"


async def test_set_validity_true_never_invalidated_writes_no_event() -> None:
    engine, adapter = await _build_engine()
    d = await engine.save("never invalidated", kind="decision", scope="p")
    assert d.hypomnema is not None
    await engine.set_validity(d.hypomnema.id, valid=False, reason="mark invalid directly")

    result = await engine.set_validity(d.hypomnema.id, valid=True, reason="reviewed")

    assert result.outcome == "revised"
    events = await adapter.list_revalidations(d.hypomnema.id)
    assert events == []
