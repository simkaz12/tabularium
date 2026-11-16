"""Build the P1 legacy fixture store — production ``save`` path only, revision
``294fa14`` (the pre-schema-migration commit ``tests/fixtures/legacy-store-
294fa14/`` is pinned to).

Never edit the fixture store directly. Rebuild it from a detached worktree at
that revision:

    git worktree add /tmp/tab-p1-base 294fa14
    cp tests/fixtures/build_legacy_store.py /tmp/tab-p1-base/tests/fixtures/build_legacy_store.py
    cd /tmp/tab-p1-base && uv run python tests/fixtures/build_legacy_store.py \\
        --out <repo>/tests/fixtures/legacy-store-294fa14
    cd <repo> && git worktree remove /tmp/tab-p1-base

Content: eight Facts across scopes ``tabularium`` and ``global`` (two carry
padded whitespace, for the content-key pass to canonicalise later), drawn
into two Fact supersession edges; one Decision superseding another; one
revoked Decision; one buried Fact. Every row goes through ``Tabularium.save``
/ ``bury`` / ``revoke`` — none of this script writes Arrow rows directly.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from tests.support.doubles import FakeEmbedder, TestClock
from tests.support.lance_raw import raw_table

from tabularium.config import EngineConfig
from tabularium.domain import HypomnemaId, Scope
from tabularium.engine import Tabularium
from tabularium.results import BounceResolution
from tabularium.storage.lancedb_adapter import LanceDBAdapter

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


async def _write(
    engine: Tabularium,
    content: str,
    *,
    scope: str,
    kind: str = "fact",
    supersedes: object = None,
) -> HypomnemaId:
    """Save through the production tool; confirm-distinct once if the dedup
    gate bounces so the intended row still lands. Any other non-``saved``
    outcome (an accidental auto-reinforce/revival/reconciliation) is a
    fixture-construction bug, not something to paper over — raise."""
    kwargs: dict[str, object] = {"kind": kind, "scope": scope}
    if supersedes is not None:
        kwargs["supersedes"] = supersedes
    result = await engine.save(content, **kwargs)  # type: ignore[arg-type]
    if result.outcome == "bounced":
        assert result.bounce is not None
        result = await engine.save(
            content,
            resolve=BounceResolution(
                candidate_id=result.bounce.candidate_id,
                action="confirm_distinct",
            ),
            **kwargs,  # type: ignore[arg-type]
        )
    if result.outcome not in ("saved", "reconciled"):
        raise RuntimeError(
            f"fixture build: unexpected outcome {result.outcome!r} for "
            f"content {content!r} (scope={scope!r}, kind={kind!r})"
        )
    assert result.hypomnema is not None
    return result.hypomnema.id


async def build(out: str) -> None:
    embedder = FakeEmbedder(dim=8)
    clock = TestClock(EPOCH)
    adapter = await LanceDBAdapter.open(
        out, embedding_model=embedder.model, dim=8, clock=clock
    )
    await adapter.create_scope(Scope("tabularium"))

    engine = Tabularium(adapter, embedder, clock, EngineConfig())
    await engine.initialize()

    f1 = await _write(
        engine, "The cleaner pass sweeps the hot tier every hour.", scope="tabularium"
    )
    f2 = await _write(engine, "Weekly review lands every Friday afternoon.", scope="global")
    f3 = await _write(
        engine, "  Startup now populates  content keys before serving. ", scope="tabularium"
    )
    f4 = await _write(
        engine, "  Read events live in their own   append-only table.  ", scope="global"
    )
    f5 = await _write(
        engine, "The old dedup gate matched content by exact string only.", scope="tabularium"
    )
    f7 = await _write(engine, "Sessions never carry a vector column at all.", scope="global")
    f9 = await _write(
        engine, "Manual contests append one row per owner-initiated dispute.", scope="tabularium"
    )
    f10 = await _write(
        engine, "The pass-history table records one row per Cleaner run.", scope="global"
    )

    f6 = await _write(
        engine,
        "The dedup gate now normalises content before comparing candidates.",
        scope="tabularium",
        supersedes=[f5],
    )
    f8 = await _write(
        engine,
        "A Session is written once, at close, never mid-conversation.",
        scope="global",
        supersedes=[f7],
    )

    d1 = await _write(
        engine,
        "Decision: schema migrations stay additive-only, never a rename or a drop.",
        scope="tabularium",
        kind="decision",
    )
    d2 = await _write(
        engine,
        "Decision: a schema migration may add a nullable column or a whole new table only.",
        scope="tabularium",
        kind="decision",
        supersedes=d1,
    )
    d3 = await _write(
        engine,
        "Decision: the Cleaner pass runs unconditionally on every digest call.",
        scope="global",
        kind="decision",
    )
    await engine.revoke(
        d3,
        reason="superseded by the gated cleaner trigger, never landed as a save",
    )

    await engine.bury(
        f10, reason="fixture: exercise the tombstones-table migration"
    )

    records = await raw_table(adapter._records).count_rows()
    tombstones = await raw_table(adapter._tombstones_tbl).count_rows()
    print(f"records={records} tombstones={tombstones}")
    for label, res in (
        ("f1", f1), ("f2", f2), ("f3", f3), ("f4", f4), ("f5", f5),
        ("f6", f6), ("f7", f7), ("f8", f8), ("f9", f9), ("f10", f10),
        ("d1", d1), ("d2", d2), ("d3", d3),
    ):
        print(f"{label}={res}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    asyncio.run(build(args.out))


if __name__ == "__main__":
    main()
