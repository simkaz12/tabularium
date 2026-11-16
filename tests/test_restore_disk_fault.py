"""The disk-fault case for ``restore`` (P4 two-arm unsupersede piece): an
in-batch storage failure inside ``restore`` lands nothing — the predecessor
still superseded, the successor still present, its tombstone still carrying
``restores`` (a prior decision's "nothing landed and the caller sees a
raise", the same shape ``test_replace_disk_fault.py`` proves for the forward
edge).

Standalone rather than folded into the conformance suite (same precedent as
``test_replace_disk_fault.py``) — the disk-permission fault is LanceDB-
specific and needs a real on-disk store, never ``MemoryAdapter``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from tabularium.domain import Decision, Scope, new_id
from tabularium.storage._supersede import resolve_unsupersede_write
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import record_to_row, row_to_record
from tabularium.storage.types import PredecessorFlip
from tests.support.faults import read_only_table_dir
from tests.support.lance_raw import raw_table

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _vec(dim: int = 4, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _decision(*, content: str) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope("proj-a"),
        lifecycle="in_force",
        validity="valid",
        created_at=EPOCH,
        updated_at=EPOCH,
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="running as root, read-only bit has no effect")
async def test_restore_disk_fault_lands_nothing(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    predecessor = _decision(content="predecessor, undo interrupted by a disk fault")
    successor = _decision(content="successor, tombstoned but never deleted")
    await adapter.upsert(predecessor, _vec())
    await adapter.upsert(successor, _vec(seed=0.9))

    flip = PredecessorFlip(
        id=predecessor.id, reason="correction", note=None, edge_id="edge-disk-fault",
    )
    await adapter.replace(successor, None, [flip], at=EPOCH)
    await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)

    with read_only_table_dir(store_path, "records"):
        with pytest.raises(Exception):  # noqa: B017 — any OSError/PermissionError-derived surface
            await adapter.restore(
                predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
            )

    reopened = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    predecessor_after = await reopened.get_by_id(predecessor.id)
    assert isinstance(predecessor_after, Decision)
    assert predecessor_after.lifecycle == "superseded"
    assert predecessor_after.superseded_by == successor.id
    assert predecessor_after.supersession_edge_id == flip.edge_id

    successor_after = await reopened.get_by_id(successor.id)
    assert isinstance(successor_after, Decision)

    tombstone = await reopened.get_tombstone(successor.id)
    assert tombstone is not None
    assert tombstone.restores == predecessor.id


@pytest.mark.skipif(os.geteuid() == 0, reason="running as root, read-only bit has no effect")
async def test_restore_split_into_two_writes_would_produce_a_half_state(
    tmp_path: Path,
) -> None:
    """The test above makes the WHOLE directory
    read-only before calling ``restore`` at all, so it can only ever observe "nothing was
    attempted" — it cannot distinguish that from "write 1 landed, write 2 failed," the genuine
    half state a SPLIT restore would allow.

    This test proves that danger directly: it performs the predecessor-update write under full
    permissions (landing it), then denies ONLY the successor-delete write. Neither call goes
    through ``LanceDBAdapter.restore`` (D1 forbids exactly this two-write shape; the shipped
    method never does it) — this exists solely to show WHY D1 mandates one ``merge_insert``
    rather than two separate writes, by reaching the half state a split would allow.
    """
    store_path = tmp_path / "store_split"
    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    predecessor = _decision(content="predecessor, split-restore half-state proof")
    successor = _decision(content="successor, split-restore half-state proof")
    await adapter.upsert(predecessor, _vec())
    await adapter.upsert(successor, _vec(seed=0.9))

    flip = PredecessorFlip(
        id=predecessor.id, reason="correction", note=None, edge_id="edge-split",
    )
    await adapter.replace(successor, None, [flip], at=EPOCH)
    await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)

    async with adapter._supersede_lock:
        pred_rows = await (
            raw_table(adapter._records)
            .query()
            .where(f"id = '{predecessor.id!s}'")
            .to_list()
        )
        existing_pred = row_to_record(pred_rows[0])
        pred_vector_raw = pred_rows[0].get("vector")
        pred_embedding_model = pred_rows[0].get("embedding_model")
        updated_pred = resolve_unsupersede_write(
            predecessor=existing_pred, predecessor_id=predecessor.id,
            successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
        )
        pred_vector = [float(x) for x in pred_vector_raw]
        row = record_to_row(updated_pred, pred_vector, pred_embedding_model)
        arrow_table = pa.table({k: [v] for k, v in row.items()}, schema=adapter._records_schema)
        # WRITE 1 (the predecessor update) — lands under full permissions.
        await (
            raw_table(adapter._records)
            .merge_insert("id")
            .when_matched_update_all()
            .execute(arrow_table)
        )

    # WRITE 2 (the successor delete) — denied.
    with read_only_table_dir(store_path, "records"):
        with pytest.raises(Exception):  # noqa: B017
            async with adapter._supersede_lock:
                await raw_table(adapter._records).delete(f"id = '{successor.id!s}'")

    reopened = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    predecessor_after = await reopened.get_by_id(predecessor.id)
    assert isinstance(predecessor_after, Decision)
    assert predecessor_after.lifecycle == "in_force", (
        "write 1 landed — this half-restored predecessor IS the danger D1 rules out"
    )
    successor_after = await reopened.get_by_id(successor.id)
    assert isinstance(successor_after, Decision), (
        "write 2 was denied — the successor row surviving alongside a restored predecessor "
        "IS the half state a split restore would allow"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="running as root, read-only bit has no effect")
async def test_restore_disk_fault_denied_only_at_the_final_write_lands_nothing(
    tmp_path: Path,
) -> None:
    """The other half of T5: against the SHIPPED single-execute ``restore``, every read/setup
    step succeeds under full permissions and the fault is denied as late as reachable —
    immediately before the one ``merge_insert(...).execute()`` call ``restore`` actually makes.
    With one execute() call there is no earlier write that could have already landed, so this
    cannot produce the half state the test above demonstrates.
    """
    store_path = tmp_path / "store_late_deny"
    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    predecessor = _decision(content="predecessor, fault denied only at the write")
    successor = _decision(content="successor, fault denied only at the write")
    await adapter.upsert(predecessor, _vec())
    await adapter.upsert(successor, _vec(seed=0.9))

    flip = PredecessorFlip(
        id=predecessor.id, reason="correction", note=None, edge_id="edge-disk-fault-late",
    )
    await adapter.replace(successor, None, [flip], at=EPOCH)
    await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)

    real_merge_insert = raw_table(adapter._records).merge_insert

    class _LateDenyBuilder:
        def __init__(self, real_builder: Any) -> None:
            self._real = real_builder

        def when_matched_update_all(self, *a: Any, **kw: Any) -> _LateDenyBuilder:
            self._real = self._real.when_matched_update_all(*a, **kw)
            return self

        def when_not_matched_by_source_delete(self, *a: Any, **kw: Any) -> _LateDenyBuilder:
            self._real = self._real.when_not_matched_by_source_delete(*a, **kw)
            return self

        async def execute(self, *a: Any, **kw: Any) -> Any:
            with read_only_table_dir(store_path, "records"):
                return await self._real.execute(*a, **kw)

    def _merge_insert_then_deny(*a: Any, **kw: Any) -> _LateDenyBuilder:
        return _LateDenyBuilder(real_merge_insert(*a, **kw))

    raw_table(adapter._records).merge_insert = _merge_insert_then_deny
    try:
        with pytest.raises(Exception):  # noqa: B017
            await adapter.restore(
                predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
            )
    finally:
        raw_table(adapter._records).merge_insert = real_merge_insert

    reopened = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    predecessor_after = await reopened.get_by_id(predecessor.id)
    assert isinstance(predecessor_after, Decision)
    assert predecessor_after.lifecycle == "superseded", (
        "the fault landed at the single execute() call, as late as reachable — the "
        "predecessor half must not have moved either, or this WOULD BE the half state "
        "T5 rules out"
    )
    assert predecessor_after.superseded_by == successor.id

    successor_after = await reopened.get_by_id(successor.id)
    assert isinstance(successor_after, Decision)

    tombstone = await reopened.get_tombstone(successor.id)
    assert tombstone is not None
    assert tombstone.restores == predecessor.id


@pytest.mark.skipif(os.geteuid() == 0, reason="running as root, read-only bit has no effect")
async def test_restore_never_calls_a_separate_delete(tmp_path: Path) -> None:
    """The two tests above fault
    ``merge_insert``/the whole directory, so they cannot tell "one atomic call" apart from "the
    first of two calls" — a regression that split ``restore`` into a ``merge_insert`` update
    followed by a separate ``self._records.delete(...)`` would still pass both, since the fault
    always lands on the (now-first) merge_insert call before the second is ever reached.

    This test targets ``self._records.delete`` itself: it raises unconditionally if ever called
    while the REAL, unmodified ``restore`` runs. The shipped ``restore`` deletes the successor
    INSIDE its one ``merge_insert(...).when_not_matched_by_source_delete(...).execute()`` call —
    it never calls ``self._records.delete`` at all — so this call must complete normally. The
    companion test right below proves the sentinel actually fires if ``restore`` regresses to a
    separate delete call, so a silent "the fault was never reachable" is not what makes this one
    pass.
    """
    store_path = tmp_path / "store_delete_sentinel"
    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    predecessor = _decision(content="predecessor, delete-sentinel regression guard")
    successor = _decision(content="successor, delete-sentinel regression guard")
    await adapter.upsert(predecessor, _vec())
    await adapter.upsert(successor, _vec(seed=0.9))

    flip = PredecessorFlip(
        id=predecessor.id, reason="correction", note=None, edge_id="edge-delete-sentinel",
    )
    await adapter.replace(successor, None, [flip], at=EPOCH)
    await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)

    async def _raising_delete(*a: Any, **kw: Any) -> None:
        raise RuntimeError(
            "restore() called records.delete() separately — it must delete the successor "
            "only inside its one merge_insert().execute() call (D1)"
        )

    raw_table(adapter._records).delete = _raising_delete

    await adapter.restore(
        predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
    )

    restored = await adapter.get_by_id(predecessor.id)
    assert isinstance(restored, Decision)
    assert restored.lifecycle == "in_force"
    with pytest.raises(KeyError):
        await adapter.get_by_id(successor.id)


@pytest.mark.skipif(os.geteuid() == 0, reason="running as root, read-only bit has no effect")
async def test_a_regression_to_a_separate_delete_call_would_trip_the_sentinel(
    tmp_path: Path,
) -> None:
    """Falsifiability companion to the test above: proves the ``records.delete`` sentinel is a
    real detector, not a fault that happens to never fire. ``LanceDBAdapter.restore`` is
    monkeypatched — ONLY for this one test — into exactly the split shape D1 forbids: the
    predecessor update lands via its own ``merge_insert().execute()``, then the successor is
    removed via a SEPARATE ``self._records.delete(...)`` call. With the same sentinel armed, that
    second call raises, and the state left behind is the half state D1 rules out: the predecessor
    already restored to ``in_force``, the successor row still present.
    """
    store_path = tmp_path / "store_delete_sentinel_split"
    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    predecessor = _decision(content="predecessor, split-restore trips the sentinel")
    successor = _decision(content="successor, split-restore trips the sentinel")
    await adapter.upsert(predecessor, _vec())
    await adapter.upsert(successor, _vec(seed=0.9))

    flip = PredecessorFlip(
        id=predecessor.id, reason="correction", note=None, edge_id="edge-delete-sentinel-split",
    )
    await adapter.replace(successor, None, [flip], at=EPOCH)
    await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)

    async def _raising_delete(*a: Any, **kw: Any) -> None:
        raise RuntimeError("sentinel: a separate records.delete() call was made")

    raw_table(adapter._records).delete = _raising_delete

    async def _split_restore(
        self: LanceDBAdapter, predecessor_id: Any, *, successor_id: Any, edge_id: str, at: Any,
    ) -> None:
        async with self._supersede_lock:
            pred_rows = await (
                raw_table(self._records).query().where(f"id = '{predecessor_id!s}'").to_list()
            )
            existing_pred = row_to_record(pred_rows[0])
            pred_vector_raw = pred_rows[0].get("vector")
            pred_embedding_model = pred_rows[0].get("embedding_model")
            updated_pred = resolve_unsupersede_write(
                predecessor=existing_pred, predecessor_id=predecessor_id,
                successor_id=successor_id, edge_id=edge_id, at=at,
            )
            pred_vector = [float(x) for x in pred_vector_raw]
            row = record_to_row(updated_pred, pred_vector, pred_embedding_model)
            arrow_table = pa.table(
                {k: [v] for k, v in row.items()}, schema=self._records_schema,
            )
            await self._records.merge_insert("id").when_matched_update_all().execute(arrow_table)
            await self._records.delete(f"id = '{successor_id!s}'")

    original_restore = LanceDBAdapter.restore
    LanceDBAdapter.restore = _split_restore  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="sentinel"):
            await adapter.restore(
                predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
            )
    finally:
        LanceDBAdapter.restore = original_restore  # type: ignore[method-assign]

    half_state_predecessor = await adapter.get_by_id(predecessor.id)
    assert isinstance(half_state_predecessor, Decision)
    assert half_state_predecessor.lifecycle == "in_force", (
        "the split restore's write 1 landed before the sentinel fired on write 2 — this IS "
        "the half state the sentinel exists to catch on a real regression"
    )
    half_state_successor = await adapter.get_by_id(successor.id)
    assert isinstance(half_state_successor, Decision)
