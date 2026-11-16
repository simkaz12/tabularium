"""The disk-fault case (P2's fault hook, P3's storage layer): an in-batch
storage failure inside ``replace`` lands nothing — the successor absent,
every predecessor unchanged (a prior decision's "failure means nothing
landed").

Standalone rather than folded into the conformance suite (same precedent as
``test_admission_support.py``'s own ``read_only_table_dir`` test) — the
disk-permission fault is LanceDB-specific and needs a real on-disk store,
never ``MemoryAdapter``.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import PredecessorFlip
from tests.support.faults import read_only_table_dir

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _vec(dim: int = 4, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _fact(*, content: str) -> Fact:
    return Fact(
        id=new_id(),
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


@pytest.mark.skipif(os.geteuid() == 0, reason="running as root, read-only bit has no effect")
async def test_replace_disk_fault_lands_nothing(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    predecessor = _fact(content="predecessor, alive before the disk fault")
    await adapter.upsert(predecessor, _vec())

    successor = _fact(content="successor that never lands")
    flip = _flip(predecessor.id)

    with read_only_table_dir(store_path, "records"):
        with pytest.raises(Exception):  # noqa: B017 — any OSError/PermissionError-derived surface
            await adapter.replace(successor, _vec(seed=0.9), [flip], at=EPOCH)

    reopened = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    with pytest.raises(KeyError):
        await reopened.get_by_id(successor.id)

    predecessor_after = await reopened.get_by_id(predecessor.id)
    assert isinstance(predecessor_after, Fact)
    assert predecessor_after.validity == "valid"
    assert predecessor_after.superseded_by is None
    assert predecessor_after.supersession_reason is None
    assert predecessor_after.supersession_edge_id is None
