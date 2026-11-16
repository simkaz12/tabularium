"""Child process body for the ``replace`` process-kill atomicity check.

Run standalone (``python3 -u _replace_kill_child.py <store_path> <n>``) — a
genuine OS process the parent test SIGKILLs, never a coroutine or thread,
since the property under test is durability of an in-flight commit against
a hard process kill, which only a real process boundary exercises.

Predecessor/successor ids are DERIVED, not communicated over stdout: both
this script and the parent test compute the same ``uuid5`` ids from the
pair index, so the parent can check every pair after a kill without relying
on output the SIGKILL might have cut off mid-line.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime

from tabularium.domain import Fact, HypomnemaId, Scope
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import PredecessorFlip

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_NAMESPACE = uuid.UUID("a3f7b6f0-3b1a-4e2c-9c8d-6f1a2b3c4d5e")


def _pred_id(i: int) -> HypomnemaId:
    return HypomnemaId(uuid.uuid5(_NAMESPACE, f"pred-{i}"))


def _succ_id(i: int) -> HypomnemaId:
    return HypomnemaId(uuid.uuid5(_NAMESPACE, f"succ-{i}"))


def _vec(seed: float) -> list[float]:
    return [seed, seed + 0.1, seed + 0.2, seed + 0.3]


async def _main(store_path: str, n: int) -> None:
    adapter = await LanceDBAdapter.open(store_path, embedding_model="test-model", dim=4)

    predecessors: list[Fact] = []
    for i in range(n):
        fact = Fact(
            id=_pred_id(i),
            type="fact",
            content=f"predecessor {i}, seeded before the kill window",
            scope=Scope("proj-a"),
            lifecycle="active",
            validity="valid",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        await adapter.upsert(fact, _vec(1.0 + i * 0.001))
        predecessors.append(fact)

    print("READY", flush=True)

    for i in range(n):
        successor = Fact(
            id=_succ_id(i),
            type="fact",
            content=f"successor {i}, replacing predecessor {i}",
            scope=Scope("proj-a"),
            lifecycle="active",
            validity="valid",
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        flip = PredecessorFlip(
            id=predecessors[i].id, reason="correction", note=None, edge_id=str(uuid.uuid4()),
        )
        await adapter.replace(successor, _vec(2.0 + i * 0.001), [flip], at=EPOCH)
        print(f"OK {i}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1], int(sys.argv[2])))
