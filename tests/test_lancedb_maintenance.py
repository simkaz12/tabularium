"""LanceDB-specific maintenance test: upkeep covers more than ``records``.

Proves the aux tables are not orphaned by ``upkeep()``. If a future change
optimizes only the high-traffic ``records`` table, this test fails — the
``pairs`` table (populated by ``mark_distinct``) would keep accumulating
small fragments forever (SPEC §3.3, ADR 0017).
"""

from __future__ import annotations

import pathlib
import uuid
from datetime import UTC, datetime
from typing import Any

from tabularium.domain import Scope, new_id
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import CleanerPassRecord, DistinctPair, UnsupersedeEvent
from tests.support.lance_raw import raw_table

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


async def test_upkeep_compacts_pairs_table_not_just_records(
    tmp_path: pathlib.Path,
) -> None:
    """~30 separate mark_distinct() calls fragment ``pairs``; upkeep() compacts it."""
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "maintenance_store"),
        embedding_model="test-model",
        dim=4,
    )

    for _ in range(30):
        a, b = new_id(), new_id()
        lower, higher = (a, b) if str(a) < str(b) else (b, a)
        pair = DistinctPair(
            lower_id=lower, higher_id=higher, source="save", judged_at=EPOCH
        )
        await adapter.mark_distinct(pair)

    stats_before = await raw_table(adapter._pairs).stats()
    assert stats_before["fragment_stats"]["num_small_fragments"] > 1

    await adapter.upkeep()

    stats_after = await raw_table(adapter._pairs).stats()
    assert stats_after["fragment_stats"]["num_small_fragments"] <= 1


async def test_upkeep_passes_explicit_retention(tmp_path: pathlib.Path) -> None:
    """``upkeep()`` must always pass an explicit ``cleanup_older_than``.

    A bare ``optimize()`` call defaults that window to 7 days, silently
    enabling part of ``reclaim()``'s job (version pruning) inside the
    routine maintenance path.  This spies on the vendor call directly so
    that *deleting* the keyword argument — not merely changing its value —
    fails the test (SPEC §3.3, ADR 0017).
    """
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "retention_store"),
        embedding_model="test-model",
        dim=4,
    )

    calls: list[dict[str, Any]] = []
    original_optimize = raw_table(adapter._records).optimize

    async def _spy_optimize(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return await original_optimize(*args, **kwargs)

    raw_table(adapter._records).optimize = _spy_optimize

    await adapter.upkeep()

    assert calls, "optimize() was never called on the records table"
    assert calls[0].get("cleanup_older_than") is not None, (
        "upkeep() must pass an explicit cleanup_older_than — a bare optimize() "
        "defaults to 7 days and would silently do part of reclaim()'s job"
    )


async def test_sessions_table_is_covered_by_upkeep_and_reclaim(
    tmp_path: pathlib.Path,
) -> None:
    """``sessions`` (2026-08-02 sessions plan) must appear in BOTH
    the ``upkeep()`` tuple and the separate ``reclaim()`` tuple.

    Two independent assertions on purpose — the two tuples are separate
    literals in the adapter, so a single check (e.g. only exercising
    ``upkeep()``) would pass even if ``reclaim()`` silently left the table
    out of maintenance.
    """
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "sessions_maintenance_store"),
        embedding_model="test-model",
        dim=4,
    )
    original_optimize = raw_table(adapter._sessions).optimize

    upkeep_calls: list[dict[str, Any]] = []

    async def _spy_upkeep(*args: Any, **kwargs: Any) -> Any:
        upkeep_calls.append(kwargs)
        return await original_optimize(*args, **kwargs)

    raw_table(adapter._sessions).optimize = _spy_upkeep
    await adapter.upkeep()
    assert upkeep_calls, "upkeep() did not call optimize() on the sessions table"

    reclaim_calls: list[dict[str, Any]] = []

    async def _spy_reclaim(*args: Any, **kwargs: Any) -> Any:
        reclaim_calls.append(kwargs)
        return await original_optimize(*args, **kwargs)

    raw_table(adapter._sessions).optimize = _spy_reclaim
    await adapter.reclaim()
    assert reclaim_calls, "reclaim() did not call optimize() on the sessions table"


async def test_cleaner_state_table_is_covered_by_upkeep_and_reclaim(
    tmp_path: pathlib.Path,
) -> None:
    """``cleaner_state`` (2026-08-10 cleaner plan) must appear in
    BOTH the ``upkeep()`` tuple and the separate ``reclaim()`` tuple.

    Two independent assertions on purpose — same rationale as the sessions
    coverage test above: the two tuples are separate literals in the
    adapter, so a single check would pass even if one silently left the
    table out of maintenance.
    """
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "cleaner_state_maintenance_store"),
        embedding_model="test-model",
        dim=4,
    )
    original_optimize = raw_table(adapter._cleaner_state).optimize

    upkeep_calls: list[dict[str, Any]] = []

    async def _spy_upkeep(*args: Any, **kwargs: Any) -> Any:
        upkeep_calls.append(kwargs)
        return await original_optimize(*args, **kwargs)

    raw_table(adapter._cleaner_state).optimize = _spy_upkeep
    await adapter.upkeep()
    assert upkeep_calls, "upkeep() did not call optimize() on the cleaner_state table"

    reclaim_calls: list[dict[str, Any]] = []

    async def _spy_reclaim(*args: Any, **kwargs: Any) -> Any:
        reclaim_calls.append(kwargs)
        return await original_optimize(*args, **kwargs)

    raw_table(adapter._cleaner_state).optimize = _spy_reclaim
    await adapter.reclaim()
    assert reclaim_calls, "reclaim() did not call optimize() on the cleaner_state table"


async def test_pass_history_table_is_covered_by_upkeep_and_reclaim(
    tmp_path: pathlib.Path,
) -> None:
    """``pass_history`` (cleaner-telemetry change) must appear in BOTH
    the ``upkeep()`` tuple and the separate ``reclaim()`` tuple.

    Two independent assertions on purpose — same rationale as the sessions/cleaner_state coverage
    tests above: the two tuples are separate literals in the adapter, so a single check would pass
    even if one silently left the table out of maintenance. Falsifiability (task verify step):
    delete ``self._pass_history,`` from the ``reclaim()`` tuple only — ``test_upkeep_...`` above
    stays green (nothing changed there) while THIS test's ``reclaim_calls`` assertion goes red,
    naming exactly the gap the two-tuple trap creates.
    """
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "pass_history_maintenance_store"),
        embedding_model="test-model",
        dim=4,
    )
    await adapter.append_cleaner_pass(
        record=CleanerPassRecord(
            scope=Scope("proj-a"),
            pass_at=EPOCH,
            outcome="completed",
            dry_run=False,
            payload="{}",
        )
    )
    original_optimize = raw_table(adapter._pass_history).optimize

    upkeep_calls: list[dict[str, Any]] = []

    async def _spy_upkeep(*args: Any, **kwargs: Any) -> Any:
        upkeep_calls.append(kwargs)
        return await original_optimize(*args, **kwargs)

    raw_table(adapter._pass_history).optimize = _spy_upkeep
    await adapter.upkeep()
    assert upkeep_calls, "upkeep() did not call optimize() on the pass_history table"

    reclaim_calls: list[dict[str, Any]] = []

    async def _spy_reclaim(*args: Any, **kwargs: Any) -> Any:
        reclaim_calls.append(kwargs)
        return await original_optimize(*args, **kwargs)

    raw_table(adapter._pass_history).optimize = _spy_reclaim
    await adapter.reclaim()
    assert reclaim_calls, "reclaim() did not call optimize() on the pass_history table"


async def test_read_events_table_is_covered_by_upkeep_and_reclaim(
    tmp_path: pathlib.Path,
) -> None:
    """``read_events`` must appear in BOTH the ``upkeep()`` tuple and the separate ``reclaim()``
    tuple.

    Two independent assertions on purpose — same rationale as every other coverage test above:
    the two tuples are separate literals in the adapter, so a single check would pass even if one
    silently left the table out of maintenance. The table's own BTree index on ``target_id`` was
    never folded when
    nothing called ``optimize()`` on this table — ``upkeep()``/``reclaim()`` omitted it.
    """
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "read_events_maintenance_store"),
        embedding_model="test-model",
        dim=4,
    )
    original_optimize = raw_table(adapter._read_events).optimize

    upkeep_calls: list[dict[str, Any]] = []

    async def _spy_upkeep(*args: Any, **kwargs: Any) -> Any:
        upkeep_calls.append(kwargs)
        return await original_optimize(*args, **kwargs)

    raw_table(adapter._read_events).optimize = _spy_upkeep
    await adapter.upkeep()
    assert upkeep_calls, "upkeep() did not call optimize() on the read_events table"

    reclaim_calls: list[dict[str, Any]] = []

    async def _spy_reclaim(*args: Any, **kwargs: Any) -> Any:
        reclaim_calls.append(kwargs)
        return await original_optimize(*args, **kwargs)

    raw_table(adapter._read_events).optimize = _spy_reclaim
    await adapter.reclaim()
    assert reclaim_calls, "reclaim() did not call optimize() on the read_events table"


async def test_unsupersedes_table_is_covered_by_upkeep_and_reclaim(
    tmp_path: pathlib.Path,
) -> None:
    """``unsupersedes`` (P4 two-arm unsupersede piece, 2026-09-20) must appear in BOTH the
    ``upkeep()`` tuple and the separate ``reclaim()`` tuple — both maintenance tuples once omitted
    the table entirely, the same
    gap ``read_events`` had above.

    Seeds several separate ``append_unsupersede`` calls (each its own ``.add()`` write, so its
    own fragment) and asserts the fragment count actually DROPS after ``upkeep()`` and the
    version count actually DROPS after ``reclaim()`` — not just that ``optimize()`` was called,
    mirroring ``test_upkeep_compacts_pairs_table_not_just_records`` above with a real measurement
    rather than a spy.
    """
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "unsupersedes_maintenance_store"),
        embedding_model="test-model",
        dim=4,
    )

    for _ in range(8):
        event = UnsupersedeEvent(
            id=uuid.uuid4(),
            edge_id=str(uuid.uuid4()),
            predecessor_id=new_id(),
            successor_id=new_id(),
            arm="decision",
            withdrawn_reason="correction",
            withdrawn_note=None,
            reason="maintenance coverage seed",
            at=EPOCH,
        )
        await adapter.append_unsupersede(event)

    stats_before = await raw_table(adapter._unsupersedes).stats()
    fragments_before = stats_before["fragment_stats"]["num_fragments"]
    assert fragments_before > 1

    await adapter.upkeep()
    stats_after_upkeep = await raw_table(adapter._unsupersedes).stats()
    fragments_after_upkeep = stats_after_upkeep["fragment_stats"]["num_fragments"]
    assert fragments_after_upkeep < fragments_before, (
        "upkeep() did not compact the unsupersedes table's fragments"
    )

    versions_after_upkeep = await raw_table(adapter._unsupersedes).list_versions()
    await adapter.reclaim()
    versions_after_reclaim = await raw_table(adapter._unsupersedes).list_versions()
    assert len(versions_after_reclaim) < len(versions_after_upkeep), (
        "reclaim() did not prune the unsupersedes table's old versions"
    )


async def test_maintenance_debt_reflects_read_events_fragmentation(
    tmp_path: pathlib.Path,
) -> None:
    """``maintenance_debt()`` must react to ``read_events`` fragmentation, not only
    ``records`` — this is the SIGNAL ``maybe_upkeep`` (``engine/maintenance.py``, called from the
    save/edit/transition write path only — never from a read path; ``log_read_events`` does not
    call it, see ``engine.read_events``'s own docstring for why) reads to decide whether to fold.
    Before this fix, ``maintenance_debt()`` did not look at ``read_events`` at all, so a store
    with heavy read traffic and no save traffic would report near-zero debt forever, even while
    ``read_events`` accumulated thousands of never-folded one-row fragments — the NEXT save's
    ``maybe_upkeep`` check now sees and folds that debt, where before it could not.
    """
    from tabularium.storage.types import ReadEvent

    adapter = await LanceDBAdapter.open(
        str(tmp_path / "read_events_debt_store"),
        embedding_model="test-model",
        dim=4,
    )
    debt_before = await adapter.maintenance_debt()

    for _ in range(30):
        await adapter.append_read_event(
            ReadEvent(target_id=new_id(), at=EPOCH, source="recall")
        )

    debt_after = await adapter.maintenance_debt()
    assert debt_after.num_small_fragments > debt_before.num_small_fragments

    await adapter.upkeep()
    debt_folded = await adapter.maintenance_debt()
    assert debt_folded.num_small_fragments <= 2  # records + read_events, each folded to ~1
