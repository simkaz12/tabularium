"""LanceDB-specific: ``replace``'s fresh re-checks — of BOTH the successor
and every predecessor — must not trust a query result
that ``_buried_ids`` already contradicts.

``LanceDBAdapter.replace`` re-reads BOTH the predecessor and the successor
fresh, immediately before the write, under ``self._supersede_lock`` (see
``storage/_supersede.py`` and ``replace``'s own docstring). Those fresh
reads are plain ``self._records.query()`` calls — the same shape
``get_by_id``/``vector_search``/``text_search``/``_scan_rows`` already
defend against the documented query-planner-staleness / same-process race:
a completed ``bury()`` (tombstone written, physical ``delete()`` already
awaited, ``_buried_ids`` already updated) whose delete has not yet
propagated to a LATER, independently-issued query's own snapshot. Without
consulting ``_buried_ids``, that window would let a record this adapter
already considers buried still read back as "live" here, and
``resolve_replace_successor``/``resolve_replace_predecessor`` — which trust
their arguments to be genuinely fresh reads (see that module's docstring)
— would accept it, landing a pointer at an id ``get_by_id`` already refuses
to return.

Contract-grounded reason this is a LanceDB-only test file, not a
dual-parameterized conformance property (same shape as
``test_lancedb_scan_buried_guard.py``'s / ``test_lancedb_update_projections_
race.py``'s own exemption notes): the ``_buried_ids`` staleness cache is a
LanceDB-specific storage detail. ``MemoryAdapter`` has no separate
cache-vs-table split at all — its ``bury()`` removes a row from ``_rows``
and adds to ``_buried_ids`` in the same synchronous step, so this exact
staleness window is structurally impossible to construct there (a
dual-parameterized version of this test could only ever exercise the
LanceDB leg; asserting on the memory leg would be vacuous).

Marks ``_buried_ids`` directly rather than calling ``adapter.bury()`` — the
same idiom the two precedent files above use, and for the same reason:
``bury()``'s own ``delete()`` would also remove the physical row, collapsing
the very staleness window under test down to the already-covered
"successor genuinely gone" case (``TestReplaceAtomicity`` in the
conformance suite).
"""

from __future__ import annotations

import pathlib
import uuid
from datetime import UTC, datetime

import pytest

from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import row_to_record
from tabularium.storage.types import PredecessorFlip

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _vec(dim: int = 4, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _fact(*, id: HypomnemaId | None = None, content: str) -> Fact:
    return Fact(
        id=id if id is not None else new_id(),
        type="fact",
        content=content,
        scope=Scope("proj-a"),
        lifecycle="active",
        validity="valid",
        created_at=EPOCH,
        updated_at=EPOCH,
    )


def _flip(predecessor_id: HypomnemaId) -> PredecessorFlip:
    return PredecessorFlip(
        id=predecessor_id, reason="correction", note=None, edge_id=str(uuid.uuid4()),
    )


async def _open(tmp_path: pathlib.Path, name: str) -> LanceDBAdapter:
    return await LanceDBAdapter.open(str(tmp_path / name), embedding_model="test-model", dim=4)


class TestFactSupersedeBuriedSuccessorCacheGuard:
    async def test_raises_and_leaves_predecessor_untouched_when_successor_buried_id_cache_is_set(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Simulates the staleness window directly: the successor's cache
        entry is set (as ``bury()`` leaves it after its own delete already
        landed) WITHOUT physically removing the successor's row from THIS
        query's view.

        Falsify: remove the ``successor.id not in self._buried_ids``
        condition from ``LanceDBAdapter.replace``'s fresh successor
        re-check — this test goes red: the supersede succeeds, and the
        predecessor is found pointing at the cache-buried successor below.
        """
        adapter = await _open(tmp_path, "fact_supersede_buried_successor_store")
        predecessor = _fact(content="predecessor targeting a cache-buried successor")
        successor = _fact(content="successor, cache-buried but physically still present")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        adapter._buried_ids.add(successor.id)

        with pytest.raises(ValueError, match="does not exist or is buried"):
            await adapter.replace(successor, None, [_flip(predecessor.id)], at=EPOCH)

        final_predecessor = await adapter.get_by_id(predecessor.id)
        assert isinstance(final_predecessor, Fact)
        assert final_predecessor.superseded_by is None, (
            "the supersede must not write through the _buried_ids guard"
        )
        assert final_predecessor.validity == "valid"


class TestFactSupersedeBuriedPredecessorCacheGuard:
    """Round 4: the symmetric gap on the PREDECESSOR side of the same fresh
    re-check block — a fresh predecessor read must not be trusted over
    ``_buried_ids`` any more than the successor's is.
    """

    async def test_raises_and_leaves_row_untouched_when_predecessor_buried_id_cache_is_set(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Simulates the staleness window directly on the predecessor:
        the predecessor's cache entry is set (as ``bury()`` leaves it after
        its own delete already landed) WITHOUT physically removing the
        predecessor's row from THIS query's view.

        Falsify: remove the ``flip.id not in self._buried_ids`` guard on
        ``LanceDBAdapter.replace``'s fresh predecessor re-check — this test
        goes red: the supersede succeeds against a predecessor this adapter
        already considers buried, and the physical row is found mutated
        below.
        """
        adapter = await _open(tmp_path, "fact_supersede_buried_predecessor_store")
        predecessor = _fact(content="predecessor, cache-buried but physically still present")
        successor = _fact(content="successor, unaffected — the race is on the predecessor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        adapter._buried_ids.add(predecessor.id)

        with pytest.raises(ValueError, match="does not exist or is buried"):
            await adapter.replace(successor, None, [_flip(predecessor.id)], at=EPOCH)

        # Read through the same raw query path replace() itself uses —
        # get_by_id would ALSO raise here via its own _buried_ids check, so
        # it could not distinguish "never wrote" from "wrote, then this
        # became unreadable".
        rows = await adapter._records.query().where(f"id = '{predecessor.id!s}'").to_list()
        assert rows, "the physical row must still be present (never deleted in this test)"
        raw_predecessor = row_to_record(rows[0])
        assert isinstance(raw_predecessor, Fact)
        assert raw_predecessor.superseded_by is None, (
            "the supersede must not write through the _buried_ids guard"
        )
        assert raw_predecessor.validity == "valid"


class TestUnsupersedeRestoreBuriedPredecessorCacheGuard:
    """The same ``_buried_ids`` staleness discipline ``replace``'s own
    predecessor re-check carries, proven on ``restore`` (P4 two-arm
    unsupersede piece): a fresh query result for a
    predecessor already recorded in ``_buried_ids`` must not be trusted,
    even though the physical row has not yet propagated as gone.
    """

    async def test_raises_and_leaves_row_untouched_when_predecessor_buried_id_cache_is_set(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Simulates the staleness window directly on the predecessor:
        the predecessor's cache entry is set (as ``bury()`` leaves it after
        its own delete already landed) WITHOUT physically removing the
        predecessor's row from THIS query's view.

        Falsify: remove the ``predecessor_id not in self._buried_ids``
        guard on ``LanceDBAdapter.restore``'s fresh predecessor re-check —
        this test goes red: ``restore`` succeeds against a predecessor this
        adapter already considers buried.
        """
        adapter = await _open(tmp_path, "unsupersede_restore_buried_predecessor_store")
        predecessor = _fact(content="predecessor, cache-buried but physically still present")
        successor = _fact(content="successor, unaffected — the race is on the predecessor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        adapter._buried_ids.add(predecessor.id)

        with pytest.raises(ValueError, match="does not exist or is buried"):
            await adapter.restore(
                predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
            )

        rows = await adapter._records.query().where(f"id = '{predecessor.id!s}'").to_list()
        assert rows, "the physical row must still be present (never deleted in this test)"
        raw_predecessor = row_to_record(rows[0])
        assert isinstance(raw_predecessor, Fact)
        assert raw_predecessor.superseded_by == successor.id, (
            "restore must not write through the _buried_ids guard"
        )
        assert raw_predecessor.validity == "invalid"
