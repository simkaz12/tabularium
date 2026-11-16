"""Rerun the 2026-09-17 Jev pair fixture against the live typesafe.ai endpoint (Assessor map
ticket 01/02). Read-only against the store: this script only ever calls
``JevAssessor.relation()`` over fixture text, it never touches the adapter and writes nothing.

Prints, per store label, two counts against the store's own disposition — strict (the exact
relation label matches) and coarse (``replacement`` vs not: ``restatement``/``distinct``/
``follow_up`` all mean "keep both, no supersession", so a ``follow_up`` pick against a
``distinct`` store label is a coarse agreement even though it strictly disagrees) — plus the
strict count agreeing with the 2026-09-17 prototype's own pick, and median latency across
every pair judged. The vendor gives no seed and no determinism guarantee, so this reports
agreement — it never asserts a number and carries no exit-code gate on the rate.

Usage::

    TYPESAFE_API_KEY=... uv run python scripts/jev_pair_rerun.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tabularium.engine.assessor import AssessorCandidate, JevAssessor

_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "jev_pairs_sample.json"
_CONCURRENCY = 8
_STORE_LABELS = ("replacement", "distinct")


def _coarse(pick: str | None) -> str:
    """``replacement`` vs everything else — ``restatement``/``distinct``/``follow_up`` all
    mean "keep both, no supersession" at the disposition level the store label speaks to."""
    return "replacement" if pick == "replacement" else "not-replacement"


async def _judge_one(
    assessor: JevAssessor, sem: asyncio.Semaphore, pair: dict[str, Any]
) -> tuple[dict[str, Any], str | None, float]:
    async with sem:
        candidate = AssessorCandidate(
            id=str(pair["pair_id"]),
            content=str(pair["existing_content"]),
            kind="fact" if pair["existing_type"] == "fact" else "decision",
            scope=str(pair["existing_scope"]),
            created_at=datetime.fromisoformat(str(pair["existing_created_at"])),
        )
        similarity = pair.get("similarity")
        start = time.monotonic()
        try:
            verdict = await assessor.relation(
                str(pair["incoming_content"]),
                candidate,
                similarity=float(similarity) if isinstance(similarity, int | float) else None,
            )
        except Exception:
            return pair, None, time.monotonic() - start
        return pair, verdict.pick, time.monotonic() - start


async def _main() -> int:
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not api_key:
        print("TYPESAFE_API_KEY is unset; nothing to run.", file=sys.stderr)
        return 1

    pairs: list[dict[str, Any]] = json.loads(_FIXTURE.read_text())
    assessor = JevAssessor(api_key, "jev-1.13.0")
    sem = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(*[_judge_one(assessor, sem, pair) for pair in pairs])

    latencies = [elapsed for _pair, _pick, elapsed in results]
    failed = sum(1 for _pair, pick, _elapsed in results if pick is None)
    print(f"{len(pairs)} pairs, {failed} failed to judge")
    if latencies:
        print(f"median latency {statistics.median(latencies):.2f}s")

    labeled_rows: list[tuple[dict[str, Any], str | None]] = []
    for label in _STORE_LABELS:
        rows = [
            (pair, pick) for pair, pick, _elapsed in results if pair.get("store_label") == label
        ]
        if not rows:
            continue
        labeled_rows.extend(rows)
        agree_strict = sum(1 for _pair, pick in rows if pick == label)
        agree_coarse = sum(1 for _pair, pick in rows if _coarse(pick) == _coarse(label))
        agree_prototype = sum(
            1 for pair, pick in rows if pick == pair.get("jev_relation_choice")
        )
        print(
            f"{label}: {len(rows)} pairs, {agree_strict} agree with the store label (strict), "
            f"{agree_coarse} agree with the store label (coarse: replacement vs not), "
            f"{agree_prototype} agree with the prototype's own pick"
        )

    if labeled_rows:
        combined_coarse = sum(
            1 for pair, pick in labeled_rows if _coarse(pick) == _coarse(pair.get("store_label"))
        )
        pct = 100 * combined_coarse / len(labeled_rows)
        print(
            f"combined coarse agreement: {combined_coarse}/{len(labeled_rows)} "
            f"({pct:.1f}%) across replacement + distinct"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
