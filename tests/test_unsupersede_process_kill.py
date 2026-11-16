"""The Decision-arm undo process-kill atomicity check (P4 two-arm
unsupersede piece).

Shape: a child process opens a LanceDB store in a tmp dir, seeds N
Decision A->B edges through ``replace``, prints a marker, then loops
``append_unsupersede`` + ``plant_tombstone`` + ``restore`` calls over those
pairs; the parent SIGKILLs it at a random short delay after the marker;
after each kill the parent reopens the store, runs ``repair_unsupersedes``,
and asserts every pair ends in exactly one of two states: not started (A
still superseded by B, B still present, no tombstone) or completed (A
restored to in_force with every reason column cleared, B absent, B's
tombstone carries ``restores=A``) — never a half state.

A standalone integration test, not a case (same precedent as
``test_admission_process_kill.py``): it proves a storage property, not an
admission outcome.
"""

from __future__ import annotations

import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest

from tabularium.domain import Decision
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import UnsupersedeEvent
from tests.support import _unsupersede_kill_child as child

_CHILD_SCRIPT = Path(child.__file__)
_N_PAIRS = 30
_N_KILLS = 12


async def _run_one_kill(store_path: Path) -> list[str]:
    """Launch the child, kill it at a random short delay after it signals
    readiness, then check the not-started-or-completed invariant on every
    pair after ``repair_unsupersedes`` runs.

    Returns, per pair index, which state it landed in — ``"not_started"``
    or ``"completed"`` — the caller uses this to report how many kills
    landed mid-loop.
    """
    proc = subprocess.Popen(
        [sys.executable, "-u", str(_CHILD_SCRIPT), str(store_path), str(_N_PAIRS)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line.strip() == "READY", (
            f"child did not signal readiness: {line!r} "
            f"(stderr so far: {proc.stderr.read() if proc.stderr else ''!r})"
        )

        delay = random.uniform(0.0002, 0.30)
        time.sleep(delay)
        proc.kill()
        proc.wait(timeout=10)
        assert proc.returncode == -9, (
            f"child did not die by SIGKILL (returncode={proc.returncode!r}) — a child that "
            "exited or crashed on its own before the kill proves nothing about restore's "
            f"atomicity (stderr: {proc.stderr.read() if proc.stderr else ''!r})"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)
    report = await adapter.repair_unsupersedes()
    refused = report.refused
    assert refused == 0, (
        "no pair in this child's loop ever edits content, so a hash mismatch can never "
        "legitimately fire here — a nonzero refused count means the compare is spuriously "
        "triggering, not proving anything about the kill window this test exists to check"
    )

    all_events = await adapter.list_unsupersedes()
    events_by_edge: dict[str, list[UnsupersedeEvent]] = defaultdict(list)
    for e in all_events:
        events_by_edge[e.edge_id].append(e)

    states: list[str] = []
    for i in range(_N_PAIRS):
        pred_id = child._pred_id(i)
        succ_id = child._succ_id(i)
        edge_id = child._edge_id(i)
        edge_events = events_by_edge.get(edge_id, [])

        predecessor = await adapter.get_by_id(pred_id)
        assert isinstance(predecessor, Decision)
        try:
            successor = await adapter.get_by_id(succ_id)
            successor_present = True
        except KeyError:
            successor_present = False
        tombstone = await adapter.get_tombstone(succ_id)

        if successor_present:
            assert isinstance(successor, Decision)
            assert predecessor.lifecycle == "superseded", (
                f"pair {i}: half state — B present but A not superseded "
                f"(lifecycle={predecessor.lifecycle!r})"
            )
            assert predecessor.superseded_by == succ_id, (
                f"pair {i}: half state — B present but A not pointed at it "
                f"(superseded_by={predecessor.superseded_by!s})"
            )
            assert predecessor.supersession_edge_id == edge_id, (
                f"pair {i}: half state — B present but A missing its edge id"
            )
            assert tombstone is None, (
                f"pair {i}: half state — B present but its tombstone already "
                "exists and repair did not finish restoring A"
            )
            assert not edge_events, (
                f"pair {i}: not_started requires NO undo event for this edge — "
                f"found {len(edge_events)}, so an undo was attempted and repair "
                "never finished it"
            )
            states.append("not_started")
        else:
            assert predecessor.lifecycle == "in_force", (
                f"pair {i}: half state — B absent but A not restored "
                f"(lifecycle={predecessor.lifecycle!r})"
            )
            assert predecessor.superseded_by is None, (
                f"pair {i}: half state — B absent but A still points at it "
                f"(superseded_by={predecessor.superseded_by!s})"
            )
            assert predecessor.supersession_reason is None, (
                f"pair {i}: half state — B absent but A still carries a "
                f"supersession_reason ({predecessor.supersession_reason!r})"
            )
            assert predecessor.supersession_note is None
            assert predecessor.supersession_edge_id is None
            assert tombstone is not None, (
                f"pair {i}: half state — B absent but no tombstone was ever planted"
            )
            assert tombstone.restores == pred_id, (
                f"pair {i}: B's tombstone does not carry restores=A "
                f"(restores={tombstone.restores!s})"
            )
            assert len(edge_events) == 1, (
                f"pair {i}: completed requires EXACTLY ONE undo event for this "
                f"edge, found {len(edge_events)}"
            )
            event = edge_events[0]
            assert event.predecessor_id == pred_id, (
                f"pair {i}: the undo event names the wrong predecessor"
            )
            assert event.successor_id == succ_id, (
                f"pair {i}: the undo event names the wrong successor"
            )
            assert event.arm == "decision", f"pair {i}: the undo event names the wrong arm"
            states.append("completed")

    return states


@pytest.mark.integration
async def test_restore_is_atomic_under_a_process_kill(tmp_path: Path) -> None:
    mid_loop_kills = 0
    for round_idx in range(_N_KILLS):
        store_path = tmp_path / f"kill_round_{round_idx}"
        states = await _run_one_kill(store_path)
        n_completed = sum(1 for s in states if s == "completed")
        if 0 < n_completed < _N_PAIRS:
            mid_loop_kills += 1
        first_not_started = next(
            (i for i, s in enumerate(states) if s == "not_started"), _N_PAIRS
        )
        assert all(s == "completed" for s in states[:first_not_started]), (
            f"round {round_idx}: a later pair completed while an earlier one did not "
            f"(states={states!r}) — writes landed out of order"
        )
        assert all(s == "not_started" for s in states[first_not_started:]), (
            f"round {round_idx}: a pair completed after the first not-started pair "
            f"(states={states!r}) — writes landed out of order"
        )

    print(f"kill runs: {_N_KILLS}, reopened+repaired every time, mid-loop kills: {mid_loop_kills}")
    assert mid_loop_kills > 0, (
        f"none of {_N_KILLS} kills landed mid-loop (all pairs either completed or none did) — "
        "this run exercised no interrupted transaction at all, so it cannot have proven "
        "atomicity under an in-flight kill; widen the delay range or re-run"
    )
