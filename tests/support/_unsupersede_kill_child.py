"""Child process body for the Decision-arm undo process-kill atomicity check
(``tests/test_unsupersede_process_kill.py``, P4 two-arm unsupersede piece).

Run standalone (``python3 -u _unsupersede_kill_child.py <store_path> <n>``) —
a genuine OS process the parent test SIGKILLs, never a coroutine or thread,
since the property under test is durability of an in-flight ``restore``
commit against a hard process kill, which only a real process boundary
exercises.

Predecessor/successor ids and edge ids are DERIVED, not communicated over
stdout: both this script and the parent test compute the same ``uuid5`` ids
from the pair index, so the parent can check every pair after a kill
without relying on output the SIGKILL might have cut off mid-line.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime

from tabularium.domain import Decision, HypomnemaId, Scope
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import PredecessorFlip, UnsupersedeEvent

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_NAMESPACE = uuid.UUID("b7c1a2e4-9f3d-4a6e-8b2c-1d5e6f7a8b9c")


def _pred_id(i: int) -> HypomnemaId:
    return HypomnemaId(uuid.uuid5(_NAMESPACE, f"pred-{i}"))


def _succ_id(i: int) -> HypomnemaId:
    return HypomnemaId(uuid.uuid5(_NAMESPACE, f"succ-{i}"))


def _edge_id(i: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"edge-{i}"))


def _vec(seed: float) -> list[float]:
    return [seed, seed + 0.1, seed + 0.2, seed + 0.3]


async def _main(store_path: str, n: int) -> None:
    adapter = await LanceDBAdapter.open(store_path, embedding_model="test-model", dim=4)

    for i in range(n):
        predecessor = Decision(
            id=_pred_id(i),
            type="decision",
            content=f"predecessor {i}, superseded before the kill window",
            scope=Scope("proj-a"),
            lifecycle="in_force",
            validity="valid",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        successor = Decision(
            id=_succ_id(i),
            type="decision",
            content=f"successor {i}, undone under the kill window",
            scope=Scope("proj-a"),
            lifecycle="in_force",
            validity="valid",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        await adapter.upsert(predecessor, _vec(1.0 + i * 0.001))
        await adapter.upsert(successor, _vec(2.0 + i * 0.001))
        flip = PredecessorFlip(
            id=predecessor.id, reason="correction", note=None, edge_id=_edge_id(i),
        )
        await adapter.replace(successor, None, [flip], at=EPOCH)

    print("READY", flush=True)

    for i in range(n):
        pred_id = _pred_id(i)
        succ_id = _succ_id(i)
        edge_id = _edge_id(i)
        event = UnsupersedeEvent(
            id=uuid.uuid4(),
            edge_id=edge_id,
            predecessor_id=pred_id,
            successor_id=succ_id,
            arm="decision",
            withdrawn_reason="correction",
            withdrawn_note=None,
            reason="undone under the kill window",
            at=EPOCH,
        )
        await adapter.append_unsupersede(event)
        await adapter.plant_tombstone(
            succ_id, f"unsupersede: pair {i}", restores=pred_id,
        )
        await adapter.restore(pred_id, successor_id=succ_id, edge_id=edge_id, at=EPOCH)
        print(f"OK {i}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1], int(sys.argv[2])))
