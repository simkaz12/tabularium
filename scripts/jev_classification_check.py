"""Live check of the Jev classifications against labelled cases.
Needs ``TYPESAFE_API_KEY``; read-only, touches no store.

Two things are measured:

* ``relation`` over the admission case set (``tests/cases/admission``), once without neighbours and
  once with up to five neighbours drawn from the other cases' texts (most token overlap first).
  A case's own ``expect.material`` label is the truth; agreement is strict (the exact relation) and
  coarse (``replacement`` versus everything else).
* ``worth``, ``kind``, ``scope``, ``perishable`` and ``breadth`` over labelled texts. ``kind`` is
  scored against every admission case's own ``incoming.kind`` plus authored rows, always with the
  same fixed ``declared_kind`` so the writer's label never reaches Jev; the others have no label
  in the case set, so they run on the small authored rows below. Each agreement line states its
  row count and whether the rows are case-set or authored.

Tripwire: ``relation`` with neighbours scoring below ``relation`` without them, or any new
classification under 70% agreement, makes the script exit 2. Stop and report the numbers when that
happens; the wording is not tuned until it passes.

Usage::

    TYPESAFE_API_KEY=... uv run python scripts/jev_classification_check.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from tests.support.cases import (  # noqa: E402
    MATERIAL_TO_RELATION,
    Case,
    load_cases,
)

from tabularium.engine.assessor import AssessorCandidate, JevAssessor  # noqa: E402

_MODEL = "jev-1.13.0"
_CONCURRENCY = 8
_MIN_AGREEMENT = 0.70
_NEIGHBOUR_K = 5
_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)
_NEUTRAL_DECLARED_KIND: Literal["fact"] = "fact"

_WORTH_ROWS: list[tuple[str, str, list[str]]] = [
    (
        "The deploy script drains judge subprocesses for up to 1800 seconds before restarting.",
        "keep", [],
    ),
    (
        "LanceDB open() issues a real DELETE when tombstones exist, so read-only harnesses "
        "must not call it.",
        "keep", [],
    ),
    (
        "This team does not use rebases; merge long-running branches with plain merge commits.",
        "keep", [],
    ),
    (
        "The function parse_config in config.py reads the TOML file and returns a dict.",
        "derivable", [],
    ),
    ("The pull request was merged and the suite is green.", "derivable", []),
    ("The tests directory contains one file per engine module.", "derivable", []),
    (
        "The Cleaner never writes to the global scope.",
        "covered",
        ["The Cleaner reads global records but never writes to the global scope."],
    ),
    (
        "Embeddings come from nomic-embed-text served by Ollama.",
        "covered",
        ["Tabularium embeds with nomic-embed-text through a local Ollama server."],
    ),
    (
        "Restatements must not reinforce the existing record.",
        "covered",
        ["A restatement exit writes nothing and never bumps the record's importance."],
    ),
]

_KIND_ROWS: list[tuple[str, str]] = [
    (
        "We chose LanceDB over Qdrant because it embeds in-process and needs no second service.",
        "decision",
    ),
    (
        "Rejected a systemd timer for the nightly Cleaner; a server task reuses single-flight.",
        "decision",
    ),
    ("Decided that Jev only judges and never writes text.", "decision"),
    (
        "merge_insert fails with 'Field not found in target schema' on stores without the "
        "new column.",
        "fact",
    ),
    ("The nightly export job failed for a month without any alert.", "fact"),
    ("scan() snapshots rows up front, so a concurrently buried id can still be yielded.", "fact"),
    ("Next I will look at the schema migration and then run the suite.", "neither"),
    ("TODO: ask the team about the threshold.", "neither"),
    ("Working on the session store now, about halfway through.", "neither"),
]

_PERISHABLE_ROWS: list[tuple[str, str]] = [
    (
        "Never call LanceDBAdapter.open() from a read-only harness; it runs a crash-repair DELETE.",
        "standing",
    ),
    (
        "Scopes are not repo names; a project's API repo and web repo share one scope.",
        "standing",
    ),
    ("Always use Clock.now(); ruff bans datetime.now.", "standing"),
    ("Phase 2 is done and its tests pass; phase 3 is next.", "status"),
    ("The deploy is in progress and the service is restarting.", "status"),
    ("The second review round raised three findings, one still open.", "status"),
    ("Run the schema rehearsal first, then deploy, then check the logs.", "sequencing"),
    ("Phase 4 is blocked until the schema change lands.", "sequencing"),
    (
        "Do the session store before the export tool because the tool needs the tables.",
        "sequencing",
    ),
]

_BREADTH_ROWS: list[tuple[str, str, str]] = [
    ("Never rebase shared branches, in any repository.", "global", "owner_wide"),
    (
        "Never add inline comments to code; this applies to every repository.",
        "global", "owner_wide",
    ),
    (
        "Close headless browser tabs when done; a left-open one can pin the CPU.",
        "global", "owner_wide",
    ),
    ("Shell commands run from a fresh working directory on every call.", "global", "owner_wide"),
    (
        "The todo app's queue worker needs its supervisor restarted after a deploy.",
        "todo-app", "project_specific",
    ),
    (
        "Tabularium's events table needs an open-time migration for new columns.",
        "tabularium", "project_specific",
    ),
    (
        "The bookstore app's chart renderer must be torn down in onBeforeUnmount.",
        "bookstore",
        "project_specific",
    ),
    ("The todo app's export uses the tasks endpoint.", "todo-app", "project_specific"),
]

_SCOPE_LABELS: list[tuple[str, str | None]] = [
    ("global", "applies to work in every project"),
    ("tabularium", "the Tabularium memory server: engine, storage, Cleaner, MCP tools"),
    ("todo-app", "the todo app: its API, its web client and its tickets"),
    ("bookstore", "the bookstore app: catalogue, orders and checkout"),
]

_SCOPE_ROWS: list[tuple[str, str]] = [
    ("Keep replies terse by default in every session.", "global"),
    ("The Cleaner's falsity axis treats global candidates as detect-only.", "tabularium"),
    ("LanceDB merge_insert needs the new column in the target schema.", "tabularium"),
    ("The todo app's export endpoint paginates at 200 rows.", "todo-app"),
    ("The todo app reads task totals from a database view.", "todo-app"),
    ("The bookstore checkout socket reconnects with exponential backoff.", "bookstore"),
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / (len(ta | tb) or 1)


def _candidate(cid: str, text: str, kind: str = "fact", scope: str = "x") -> AssessorCandidate:
    return AssessorCandidate(
        id=cid,
        content=text,
        kind="decision" if kind == "decision" else "fact",
        scope=scope,
        created_at=_EPOCH,
    )


def _neighbours_for(case: Case, cases: Sequence[Case]) -> list[AssessorCandidate]:
    pool: list[AssessorCandidate] = []
    for other in cases:
        if other.id == case.id:
            continue
        for row in other.existing:
            pool.append(_candidate(f"{other.id}:{row.alias}", row.text, row.kind, row.scope))
    pool.sort(key=lambda c: _overlap(case.incoming.content, c.content), reverse=True)
    return pool[:_NEIGHBOUR_K]


async def _gather(coros: Sequence[object]) -> list[object]:
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _run(coro: object) -> object:
        async with sem:
            try:
                return await coro  # type: ignore[misc]
            except Exception as exc:
                return exc

    return list(await asyncio.gather(*[_run(c) for c in coros]))


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


async def _relation_check(assessor: JevAssessor, cases: Sequence[Case]) -> dict[str, float]:
    judged = [c for c in cases if c.expect is not None and c.existing]
    truth = [MATERIAL_TO_RELATION[c.expect.material] for c in judged if c.expect is not None]
    existing = [
        _candidate(c.existing[0].alias, c.existing[0].text, c.existing[0].kind, c.existing[0].scope)
        for c in judged
    ]
    without = await _gather(
        [
            assessor.relation(c.incoming.content, e, similarity=None)
            for c, e in zip(judged, existing, strict=True)
        ]
    )
    with_neighbours = await _gather(
        [
            assessor.relation(
                c.incoming.content,
                e,
                similarity=None,
                neighbours=_neighbours_for(c, cases),
            )
            for c, e in zip(judged, existing, strict=True)
        ]
    )
    out: dict[str, float] = {}
    for name, results in (("without", without), ("with", with_neighbours)):
        strict = sum(
            1 for r, t in zip(results, truth, strict=True) if getattr(r, "pick", None) == t
        )
        coarse = sum(
            1
            for r, t in zip(results, truth, strict=True)
            if hasattr(r, "pick") and (r.pick == "replacement") == (t == "replacement")
        )
        failed = sum(1 for r in results if isinstance(r, Exception))
        print(
            f"relation {name:>7} neighbours: strict {strict}/{len(judged)} "
            f"coarse {coarse}/{len(judged)} failed {failed}"
        )
        out[f"strict_{name}"] = _rate(strict, len(judged))
        out[f"coarse_{name}"] = _rate(coarse, len(judged))
    return out


async def _agreement(
    label: str, results: Sequence[object], truth: Sequence[str], source: str
) -> float:
    hits = sum(1 for r, t in zip(results, truth, strict=True) if getattr(r, "pick", None) == t)
    failed = sum(1 for r in results if isinstance(r, Exception))
    rate = _rate(hits, len(truth))
    print(
        f"{label:>10}: {hits}/{len(truth)} = {rate:.0%} failed {failed} "
        f"rows {len(truth)} [{source}]"
    )
    return rate


async def _main() -> int:
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not api_key:
        print("TYPESAFE_API_KEY is unset; nothing to run.", file=sys.stderr)
        return 1

    cases = load_cases()
    assessor = JevAssessor(api_key, _MODEL)
    relation = await _relation_check(assessor, cases)

    classification: dict[str, float] = {}

    worth = await _gather(
        [
            assessor.worth(
                text,
                neighbours=[_candidate(f"n{i}", n) for i, n in enumerate(neighbours)] or None,
            )
            for text, _label, neighbours in _WORTH_ROWS
        ]
    )
    classification["worth"] = await _agreement(
        "worth", worth, [r[1] for r in _WORTH_ROWS], "authored"
    )

    kind_rows: list[tuple[str, str]] = [(c.incoming.content, c.incoming.kind) for c in cases]
    kind_rows += _KIND_ROWS
    kind = await _gather([assessor.kind(text, _NEUTRAL_DECLARED_KIND) for text, _l in kind_rows])
    kind_source = f"{len(cases)} case-set + {len(_KIND_ROWS)} authored"
    classification["kind"] = await _agreement(
        "kind", kind, [label for _t, label in kind_rows], kind_source
    )

    scope = await _gather([assessor.scope(text, _SCOPE_LABELS) for text, _l in _SCOPE_ROWS])
    classification["scope"] = await _agreement(
        "scope", scope, [label for _t, label in _SCOPE_ROWS], "authored"
    )

    perishable = await _gather([assessor.perishable(text) for text, _l in _PERISHABLE_ROWS])
    classification["perishable"] = await _agreement(
        "perishable", perishable, [label for _t, label in _PERISHABLE_ROWS], "authored"
    )

    breadth = await _gather(
        [assessor.breadth(text, source) for text, source, _l in _BREADTH_ROWS]
    )
    classification["breadth"] = await _agreement(
        "breadth", breadth, [label for _t, _s, label in _BREADTH_ROWS], "authored"
    )

    tripped: list[str] = []
    if relation["strict_with"] < relation["strict_without"]:
        tripped.append(
            f"relation with neighbours strict {relation['strict_with']:.0%} "
            f"< without {relation['strict_without']:.0%}"
        )
    if relation["coarse_with"] < relation["coarse_without"]:
        tripped.append(
            f"relation with neighbours coarse {relation['coarse_with']:.0%} "
            f"< without {relation['coarse_without']:.0%}"
        )
    for name, rate in classification.items():
        if rate < _MIN_AGREEMENT:
            tripped.append(f"{name} agreement {rate:.0%} < {_MIN_AGREEMENT:.0%}")

    if tripped:
        print("U1 TRIPWIRE:", "; ".join(tripped), file=sys.stderr)
        return 2
    print("U1 tripwire clear")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
