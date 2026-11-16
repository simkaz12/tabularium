"""The process-kill atomicity check — the third fault hook.

P3 lands the ``replace`` operation this test proves atomic. Shape: a child process opens a
LanceDB store in a tmp dir, seeds predecessors through the real adapter, prints a marker, then
loops ``replace`` calls; the parent SIGKILLs it at a random short delay after the marker; after
each kill the parent reopens the store fresh and asserts, for every (predecessor, successor)
pair the child could have reached: the successor is present and the predecessor's flip landed,
or the successor is absent and the predecessor is untouched — never a half state. A standalone
integration test, not a case: it proves a storage property, not an
admission outcome.
"""

from __future__ import annotations

import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tests.support import _replace_kill_child as child

_CHILD_SCRIPT = Path(child.__file__)
_N_PAIRS = 30
_N_KILLS = 12


async def _run_one_kill(store_path: Path) -> list[bool]:
    """Launch the child, kill it at a random short delay after it signals
    readiness, then check the all-or-nothing invariant on every pair.

    Returns, per pair index, whether that pair committed (successor
    present) — the caller uses this to report how many kills landed
    mid-loop (some pairs committed, some not) versus before any write.
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

        delay = random.uniform(0.0002, 0.28)
        time.sleep(delay)
        proc.kill()
        proc.wait(timeout=10)
        assert proc.returncode == -9, (
            f"child did not die by SIGKILL (returncode={proc.returncode!r}) — a child that "
            "exited or crashed on its own before the kill proves nothing about replace's "
            f"atomicity (stderr: {proc.stderr.read() if proc.stderr else ''!r})"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    adapter = await LanceDBAdapter.open(str(store_path), embedding_model="test-model", dim=4)

    committed: list[bool] = []
    for i in range(_N_PAIRS):
        succ_id = child._succ_id(i)
        pred_id = child._pred_id(i)

        try:
            successor = await adapter.get_by_id(succ_id)
            successor_present = True
        except KeyError:
            successor_present = False

        predecessor = await adapter.get_by_id(pred_id)

        # Every field the flip touches is checked explicitly, on both
        # branches — a compound boolean (``superseded_by == succ_id and
        # validity == "invalid"``) would pass on ANY other combination
        # that also evaluates false, silently accepting a predecessor
        # whose validity flipped without its pointer landing, or vice
        # versa, as if it were cleanly untouched.
        if successor_present:
            assert successor.validity == "valid", f"pair {i}: successor present but not valid"
            assert predecessor.superseded_by == succ_id, (
                f"pair {i}: half state — successor present but predecessor not "
                f"pointed at it (superseded_by={predecessor.superseded_by!s})"
            )
            assert predecessor.validity == "invalid", (
                f"pair {i}: half state — successor present but predecessor still valid"
            )
            assert predecessor.supersession_reason == "correction", (
                f"pair {i}: half state — successor present but predecessor missing "
                f"supersession_reason (got {predecessor.supersession_reason!r})"
            )
            assert predecessor.supersession_edge_id is not None, (
                f"pair {i}: half state — successor present but predecessor missing "
                "supersession_edge_id"
            )
        else:
            assert predecessor.superseded_by is None, (
                f"pair {i}: half state — successor absent but predecessor already "
                f"points at it (superseded_by={predecessor.superseded_by!s})"
            )
            assert predecessor.validity == "valid", (
                f"pair {i}: half state — successor absent but predecessor already invalid"
            )
            assert predecessor.supersession_reason is None, (
                f"pair {i}: half state — successor absent but predecessor already "
                f"carries a supersession_reason ({predecessor.supersession_reason!r})"
            )
            assert predecessor.supersession_edge_id is None, (
                f"pair {i}: half state — successor absent but predecessor already "
                "carries a supersession_edge_id"
            )
        committed.append(successor_present)

    return committed


@pytest.mark.integration
async def test_replace_is_atomic_under_a_process_kill(tmp_path: Path) -> None:
    mid_loop_kills = 0
    for round_idx in range(_N_KILLS):
        store_path = tmp_path / f"kill_round_{round_idx}"
        committed = await _run_one_kill(store_path)
        n_committed = sum(committed)
        # A prefix of pairs committed and the rest didn't — a genuine
        # in-flight kill, not a kill before the loop even started or after
        # it fully finished.
        if 0 < n_committed < _N_PAIRS:
            mid_loop_kills += 1
        # Sequential awaits within one process commit in order: once a
        # later pair is absent, no earlier pair should be absent either.
        first_absent = next((i for i, c in enumerate(committed) if not c), _N_PAIRS)
        assert all(committed[:first_absent]), (
            f"round {round_idx}: a later pair committed while an earlier one did not "
            f"(committed={committed!r}) — writes landed out of order"
        )

    print(f"kill runs: {_N_KILLS}, reopened every time, mid-loop kills: {mid_loop_kills}")
    assert mid_loop_kills > 0, (
        f"none of {_N_KILLS} kills landed mid-loop (all committed either everything or "
        "nothing) — this run exercised no interrupted transaction at all, so it cannot "
        "have proven atomicity under an in-flight kill; widen the delay range or re-run"
    )
