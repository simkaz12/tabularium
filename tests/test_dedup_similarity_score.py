"""Dedup-collision similarity persistence (2026-08-27 dedup-collision-score plan).

Before this, a bounce's ``similarity`` was ephemeral: reported once in the ``SaveResult`` the
caller adjudicates, then discarded. Neither ``DistinctPair`` (written by
``resolve(action="confirm_distinct")``) nor ``ReinforcementEvent`` (written by a dedup-sourced
reinforce — auto-reinforce or ``resolve(action="reinforce")``) carried it, so nothing durable ever
recorded whether a resolved collision clustered near the gray floor or the ceiling.

Both new fields are nullable with a ``None`` default — an existing row from before this field
existed reads back with ``similarity=None``, never a fabricated value.
"""

from __future__ import annotations

import math
import pathlib
from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import HypomnemaId, new_id
from tabularium.engine import Tabularium
from tabularium.results import BounceResolution
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import DistinctPair, ReinforcementEvent
from tests.conformance.test_conformance import _make_lancedb_adapter, _make_memory_adapter
from tests.support.doubles import TestClock

pytestmark = pytest.mark.asyncio

_DIM = 4
_NOW = datetime(2025, 1, 1, tzinfo=UTC)

_AnyAdapter = MemoryAdapter | LanceDBAdapter


def _angle_vec(angle_deg: float, dim: int = _DIM) -> list[float]:
    """Unit vector at ``angle_deg`` from axis 0 — two such vectors ``theta`` apart have cosine
    ``cos(theta)`` exactly, letting a test place a probe at a KNOWN, non-trivial similarity rather
    than the degenerate 0.0/1.0 a single-hot hash embedder produces."""
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


class _TableEmbedder:
    dim: int = _DIM
    model: str = "test/table"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"candidate_k": 1000, "gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


async def _adapter(adapter_id: str, tmp_path: pathlib.Path, clock: TestClock) -> _AnyAdapter:
    if adapter_id == "memory":
        return _make_memory_adapter(embedding_model="test/table", dim=_DIM, clock=clock)
    return await _make_lancedb_adapter(
        tmp_path, embedding_model="test/table", dim=_DIM, clock=clock
    )


async def _read_back_pair(
    adapter: _AnyAdapter, a: HypomnemaId, b: HypomnemaId
) -> DistinctPair:
    """Read the persisted ``DistinctPair`` for ``(a, b)`` directly off adapter internals — there
    is no public getter beyond the boolean ``is_distinct`` (deliberately out of this plan's
    scope; see the module docstring)."""
    if isinstance(adapter, MemoryAdapter):
        return adapter._distinct[frozenset({a, b})]
    lo, hi = (str(a), str(b)) if str(a) < str(b) else (str(b), str(a))
    rows = await (
        adapter._pairs.query().where(f"lower_id = '{lo}' AND higher_id = '{hi}'").to_list()
    )
    assert len(rows) == 1
    row = rows[0]
    return DistinctPair(
        lower_id=HypomnemaId(row["lower_id"]),
        higher_id=HypomnemaId(row["higher_id"]),
        source=row["source"],
        judged_at=row["judged_at"],
        similarity=row.get("similarity"),
    )


@pytest.mark.parametrize("adapter_id", ["memory", "lancedb"])
class TestStorageSeamSimilarityRoundTrip:
    """Both fields, both adapters, storage seam only."""

    async def test_reinforcement_event_similarity_round_trips(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        clock = TestClock(_NOW)
        adapter = await _adapter(adapter_id, tmp_path, clock)
        target = new_id()
        await adapter.append_event(
            ReinforcementEvent(
                target_id=target, at=_NOW, source="dedup", similarity=0.9063
            )
        )
        events = await adapter.query_events(target_id=target)
        assert len(events) == 1
        assert events[0].similarity == pytest.approx(0.9063)

    async def test_reinforcement_event_similarity_defaults_to_none(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        clock = TestClock(_NOW)
        adapter = await _adapter(adapter_id, tmp_path, clock)
        target = new_id()
        await adapter.append_event(
            ReinforcementEvent(target_id=target, at=_NOW, source="revival")
        )
        events = await adapter.query_events(target_id=target)
        assert len(events) == 1
        assert events[0].similarity is None

    async def test_distinct_pair_similarity_round_trips(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        clock = TestClock(_NOW)
        adapter = await _adapter(adapter_id, tmp_path, clock)
        a, b = new_id(), new_id()
        lower, higher = (a, b) if str(a) < str(b) else (b, a)
        await adapter.mark_distinct(
            DistinctPair(
                lower_id=lower, higher_id=higher, source="save", judged_at=_NOW,
                similarity=0.8512,
            )
        )
        pair = await _read_back_pair(adapter, a, b)
        assert pair.similarity == pytest.approx(0.8512)

    async def test_distinct_pair_similarity_defaults_to_none(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        clock = TestClock(_NOW)
        adapter = await _adapter(adapter_id, tmp_path, clock)
        a, b = new_id(), new_id()
        lower, higher = (a, b) if str(a) < str(b) else (b, a)
        await adapter.mark_distinct(
            DistinctPair(lower_id=lower, higher_id=higher, source="save", judged_at=_NOW)
        )
        pair = await _read_back_pair(adapter, a, b)
        assert pair.similarity is None


_EXPECTED_COSINE = math.cos(math.radians(25))


class TestEngineResolveSimilarityRoundTrip:
    """The gate's own probe cosine, not a hand-supplied one."""

    async def test_confirm_distinct_persists_the_probe_cosine(self) -> None:
        clock = TestClock(_NOW)
        adapter = MemoryAdapter(embedding_model="test/table", dim=_DIM, clock=clock)
        c0, c1 = "original policy note on release cadence", "a related but distinct note"
        embedder = _TableEmbedder({c0: _angle_vec(0), c1: _angle_vec(25)})
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        r1 = await engine.save(c0, scope="global")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        orig_id = r1.hypomnema.id

        r2 = await engine.save(c1, scope="global")
        assert r2.outcome == "bounced"
        assert r2.bounce is not None
        assert r2.bounce.similarity == pytest.approx(_EXPECTED_COSINE, abs=1e-4)

        resolution = BounceResolution(candidate_id=orig_id, action="confirm_distinct")
        r3 = await engine.save(c1, scope="global", resolve=resolution)
        assert r3.outcome == "saved"
        assert r3.hypomnema is not None

        pair = await _read_back_pair(adapter, orig_id, r3.hypomnema.id)
        assert pair.similarity == pytest.approx(_EXPECTED_COSINE, abs=1e-4)

    async def test_reinforce_persists_the_probe_cosine(self) -> None:
        clock = TestClock(_NOW)
        adapter = MemoryAdapter(embedding_model="test/table", dim=_DIM, clock=clock)
        c0, c1 = "original policy note on release cadence", "a related but distinct note"
        embedder = _TableEmbedder({c0: _angle_vec(0), c1: _angle_vec(25)})
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        r1 = await engine.save(c0, scope="global")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        orig_id = r1.hypomnema.id

        r2 = await engine.save(c1, scope="global")
        assert r2.outcome == "bounced"

        resolution = BounceResolution(candidate_id=orig_id, action="reinforce")
        r3 = await engine.save(c1, scope="global", resolve=resolution)
        assert r3.outcome == "reinforced"

        events = await adapter.query_events(target_id=orig_id, source="dedup")
        assert len(events) == 1
        assert events[0].similarity == pytest.approx(_EXPECTED_COSINE, abs=1e-4)


class TestLanceDBSimilarityColumnMigration:
    """A store created before ``similarity`` existed on either the events or the pairs table
    must reopen, read pre-existing rows as ``similarity=None``, and accept new writes with a
    real value -- same shape as ``tests/test_engine_feedback.py::
    TestLanceDBEventsSchemaMigration``, generalized to a float column."""

    async def test_reopen_migrates_missing_events_similarity_column(
        self, tmp_path: pathlib.Path
    ) -> None:
        import lancedb
        import pyarrow as pa

        from tabularium.storage.lancedb_codec import EVENTS_SCHEMA

        store_path = str(tmp_path / "old_events_similarity_schema_store")
        old_fields = [f for f in EVENTS_SCHEMA if f.name != "similarity"]
        old_schema = pa.schema(old_fields)

        old_target_id = new_id()
        old_row = {
            "target_id": str(old_target_id),
            "at": _NOW,
            "source": "dedup",
            "matched_from_id": None,
            "value": None,
            "reason": None,
        }

        db = await lancedb.connect_async(store_path)
        await db.create_table("events", schema=old_schema)
        old_table = await db.open_table("events")
        await old_table.add([old_row])

        adapter = await LanceDBAdapter.open(store_path, embedding_model="test-model", dim=_DIM)

        migrated = await adapter.query_events(target_id=old_target_id)
        assert len(migrated) == 1
        assert migrated[0].similarity is None

        new_target_id = new_id()
        await adapter.append_event(
            ReinforcementEvent(
                target_id=new_target_id, at=_NOW, source="dedup", similarity=0.87
            )
        )
        after = await adapter.query_events(target_id=new_target_id)
        assert len(after) == 1
        assert after[0].similarity == pytest.approx(0.87)

        adapter2 = await LanceDBAdapter.open(store_path, embedding_model="test-model", dim=_DIM)
        again = await adapter2.query_events(target_id=old_target_id)
        assert len(again) == 1
        assert again[0].similarity is None

    async def test_reopen_migrates_missing_pairs_similarity_column(
        self, tmp_path: pathlib.Path
    ) -> None:
        import lancedb
        import pyarrow as pa

        from tabularium.storage.lancedb_codec import PAIRS_SCHEMA

        store_path = str(tmp_path / "old_pairs_similarity_schema_store")
        old_fields = [f for f in PAIRS_SCHEMA if f.name != "similarity"]
        old_schema = pa.schema(old_fields)

        old_a, old_b = new_id(), new_id()
        old_lower, old_higher = (str(old_a), str(old_b)) if str(old_a) < str(old_b) else (
            str(old_b), str(old_a)
        )
        old_row = {
            "lower_id": old_lower,
            "higher_id": old_higher,
            "source": "save",
            "judged_at": _NOW,
        }

        db = await lancedb.connect_async(store_path)
        await db.create_table("pairs", schema=old_schema)
        old_table = await db.open_table("pairs")
        await old_table.add([old_row])

        adapter = await LanceDBAdapter.open(store_path, embedding_model="test-model", dim=_DIM)

        assert await adapter.is_distinct(old_a, old_b)
        old_pair = await _read_back_pair(adapter, old_a, old_b)
        assert old_pair.similarity is None

        new_a, new_b = new_id(), new_id()
        new_lower, new_higher = (new_a, new_b) if str(new_a) < str(new_b) else (new_b, new_a)
        await adapter.mark_distinct(
            DistinctPair(
                lower_id=new_lower, higher_id=new_higher, source="save", judged_at=_NOW,
                similarity=0.93,
            )
        )
        new_pair = await _read_back_pair(adapter, new_a, new_b)
        assert new_pair.similarity == pytest.approx(0.93)

        adapter2 = await LanceDBAdapter.open(store_path, embedding_model="test-model", dim=_DIM)
        assert await adapter2.is_distinct(old_a, old_b)
        again = await _read_back_pair(adapter2, old_a, old_b)
        assert again.similarity is None
