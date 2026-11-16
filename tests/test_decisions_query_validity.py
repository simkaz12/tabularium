"""``decisions_query(include_invalid)`` widens to invalid Decisions on request —
the same shape ``recall(include_invalid)`` already uses (P9 invalid-Decision
read contract, decisions D5/D6). The invalid row under test is built through
P8's real Fact-invalidates-Decision write path (``EchoAssessor("correction")``,
``auto_exit_replacement=True``, a Decision save then a Fact save) — never by
setting ``validity``/``invalidated_by`` by hand.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from tabularium.config import EngineConfig
from tabularium.domain import HypomnemaId, Scope
from tabularium.engine import Tabularium
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import TestClock
from tests.support.fake_assessor import EchoAssessor

_DIM = 16
_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)

_EXISTING = "the quick brown fox jumps over the lazy dog"
_INCOMING = "the quick brown fox jumps over the lazy hound"
_SUCCESSOR = "the quick brown fox leaps past the lazy wolf, revised"
_OTHER = "a committee approved the quarterly budget allocation"


def _angle_vec(angle_deg: float, dim: int = _DIM) -> list[float]:
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


class _TableEmbedder:
    dim: int = _DIM
    model: str = "test/decisions-query-validity"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


_TABLE: dict[str, list[float]] = {
    _EXISTING: _angle_vec(0),
    _INCOMING: _angle_vec(5),
    _SUCCESSOR: _angle_vec(200),
    _OTHER: _angle_vec(90),
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


async def _seed_invalid_and_valid() -> (
    tuple[Tabularium, MemoryAdapter, HypomnemaId, HypomnemaId, HypomnemaId]
):
    """Decision D invalidated by Fact F (P8's real path), plus a valid Decision E."""
    engine, adapter = await _build_engine()
    d = await engine.save(_EXISTING, scope="p", kind="decision")
    assert d.hypomnema is not None
    fact = await engine.save(_INCOMING, scope="p")
    assert fact.hypomnema is not None
    e = await engine.save(_OTHER, scope="p", kind="decision")
    assert e.hypomnema is not None
    return engine, adapter, d.hypomnema.id, fact.hypomnema.id, e.hypomnema.id


async def test_default_excludes_invalid_decision() -> None:
    engine, _adapter, d_id, _fact_id, e_id = await _seed_invalid_and_valid()

    links = await engine.decisions_query(_EXISTING, scope="p", limit=10)

    ids = {link.id for link in links}
    assert d_id not in ids
    assert e_id in ids


async def test_include_invalid_widens_and_marks_validity() -> None:
    engine, _adapter, d_id, _fact_id, e_id = await _seed_invalid_and_valid()

    links = await engine.decisions_query(
        _EXISTING, scope="p", limit=10, include_invalid=True
    )

    by_id = {link.id: link for link in links}
    assert d_id in by_id
    assert by_id[d_id].validity == "invalid"
    assert by_id[d_id].lifecycle == "in_force"
    assert e_id in by_id
    assert by_id[e_id].validity == "valid"


async def test_include_invalid_and_include_superseded_compose() -> None:
    engine, _adapter, d_id, _fact_id, _e_id = await _seed_invalid_and_valid()

    successor = await engine.save(
        _SUCCESSOR, scope="p", kind="decision", supersedes=d_id, reason="changed"
    )
    assert successor.hypomnema is not None

    narrow = await engine.decisions_query(_EXISTING, scope="p", limit=10, include_invalid=True)
    assert d_id not in {link.id for link in narrow}

    widened = await engine.decisions_query(
        _EXISTING, scope="p", limit=10, include_invalid=True, include_superseded=True
    )
    by_id = {link.id: link for link in widened}
    assert d_id in by_id
    assert by_id[d_id].validity == "invalid"
    assert by_id[d_id].lifecycle == "superseded"
