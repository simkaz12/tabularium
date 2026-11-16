"""LanceDB-specific: ``scan`` and ``scan_records`` must both apply the same
``_buried_ids`` defense-in-depth guard ``get_by_id``/``vector_search``/
``text_search`` already carry.

The gap showed first in ``scan_records`` alone (a content leak: a row whose
burial is recorded but not yet physically deleted would surface full
content). The identical gap existed in ``scan`` — both methods share
one row fetch (``LanceDBAdapter._scan_rows``), and the fix now lives there
once, closing it for both callers (e.g. ``dispositions.set_hot``'s
per-(scope, role) hot_cap count — curation block — reads
``scan_records`` and could otherwise over-count a buried-but-still-visible
hot row).

Contract-grounded reason this is a LanceDB-only test file, not a
dual-parameterized conformance property: ``_buried_ids`` is an adapter-private
cache that exists ONLY because LanceDB's physical query can lag behind a
completed burial (query-planner staleness / same-process race — see
``_scan_rows``'s docstring in ``storage/lancedb_adapter.py``). ``MemoryAdapter``
has no async storage engine and no such cache at all — ``bury()``
synchronously deletes the row from the live dict, so the race manufactured
here is structurally impossible to construct there, not merely inconvenient
to.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.storage.lancedb_adapter import LanceDBAdapter

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _vec(dim: int = 4, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _fact(*, id: HypomnemaId | None = None, content: str = "buried-id guard test fact") -> Fact:
    return Fact(
        id=id if id is not None else new_id(),
        type="fact",
        content=content,
        scope=Scope("proj-a"),
        lifecycle="hot",
        validity="valid",
        created_at=EPOCH,
        updated_at=EPOCH,
    )


async def _open(tmp_path: pathlib.Path, name: str) -> LanceDBAdapter:
    return await LanceDBAdapter.open(str(tmp_path / name), embedding_model="test-model", dim=4)


class TestScanRecordsBuriedIdGuard:
    async def test_excludes_a_row_the_buried_id_cache_marks_buried(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Manufacture exactly the gap: the physical
        ``records`` row is still present and query-visible (never deleted),
        but ``_buried_ids`` already carries the id — the query-planner-
        staleness / same-process-race window ``get_by_id`` defends against
        via the identical cache check. Without the fix, ``scan_records``
        would yield this row even though ``get_by_id`` for the same id
        correctly raises ``KeyError``.
        """
        adapter = await _open(tmp_path, "scan_records_buried_guard_store")
        fact = _fact()
        await adapter.upsert(fact, _vec())

        # Sanity: before any burial bookkeeping, scan_records sees the row —
        # proves the test isn't vacuously passing on an empty result.
        before = [rec async for rec in await adapter.scan_records()]
        assert [r.id for r in before] == [fact.id]

        # Simulate the staleness window directly: the cache is updated (as
        # bury() does) WITHOUT the physical delete having taken effect on
        # this query's view. Not calling adapter.bury() on purpose: that
        # call's delete would also remove the physical row, collapsing the
        # very race being tested down to the (already-covered) normal case.
        adapter._buried_ids.add(fact.id)

        after = [rec async for rec in await adapter.scan_records()]
        assert fact.id not in {r.id for r in after}, (
            "scan_records must reject a row whose id is in the buried-id cache, "
            "matching get_by_id's guard for the same id"
        )

    async def test_lifecycle_filter_also_respects_the_buried_id_cache(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Same guard, exercised with an explicit ``lifecycle_in`` filter —
        the two code paths inside ``_scan_rows`` (filtered vs. unfiltered
        query) must both route through the same post-fetch buried-id check.
        """
        adapter = await _open(tmp_path, "scan_records_buried_guard_lifecycle_store")
        fact = _fact()
        await adapter.upsert(fact, _vec())
        adapter._buried_ids.add(fact.id)

        collected = [
            rec async for rec in await adapter.scan_records(lifecycle_in=frozenset({"hot"}))
        ]
        assert fact.id not in {r.id for r in collected}

    async def test_filter_is_not_over_broad_a_live_sibling_row_still_appears(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The guard must reject ONLY the id in ``_buried_ids`` — a second,
        genuinely live row in the same store must still come through
        ``scan_records`` unaffected (proves the
        filter is not over-broad, not merely that it excludes something)."""
        adapter = await _open(tmp_path, "scan_records_buried_guard_not_over_broad_store")
        buried = _fact(content="this one gets marked buried in the cache")
        live = _fact(content="this one stays fully live")
        await adapter.upsert(buried, _vec())
        await adapter.upsert(live, _vec(seed=0.5))
        adapter._buried_ids.add(buried.id)

        collected = [rec async for rec in await adapter.scan_records()]
        ids = {r.id for r in collected}
        assert buried.id not in ids
        assert live.id in ids


class TestScanBuriedIdGuard:
    """Same guard, same shared ``_scan_rows`` fetch, exercised through
    ``scan`` (metadata-only ``RowMeta``) instead of ``scan_records``."""

    async def test_scan_excludes_a_row_the_buried_id_cache_marks_buried(
        self, tmp_path: pathlib.Path
    ) -> None:
        adapter = await _open(tmp_path, "scan_buried_guard_store")
        fact = _fact()
        await adapter.upsert(fact, _vec())

        before = [meta async for meta in await adapter.scan()]
        assert [m.id for m in before] == [fact.id]

        adapter._buried_ids.add(fact.id)

        after = [meta async for meta in await adapter.scan()]
        assert fact.id not in {m.id for m in after}, (
            "scan must reject a row whose id is in the buried-id cache, "
            "matching get_by_id's guard for the same id"
        )

    async def test_scan_lifecycle_filter_also_respects_the_buried_id_cache(
        self, tmp_path: pathlib.Path
    ) -> None:
        adapter = await _open(tmp_path, "scan_buried_guard_lifecycle_store")
        fact = _fact()
        await adapter.upsert(fact, _vec())
        adapter._buried_ids.add(fact.id)

        collected = [
            meta async for meta in await adapter.scan(lifecycle_in=frozenset({"hot"}))
        ]
        assert fact.id not in {m.id for m in collected}

    async def test_scan_filter_is_not_over_broad_a_live_sibling_row_still_appears(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The same not-over-broad property, for scan's RowMeta output —
        this is the exact path dispositions.set_hot's hot_cap count reads,
        so a live sibling must still be counted."""
        adapter = await _open(tmp_path, "scan_buried_guard_not_over_broad_store")
        buried = _fact(content="this one gets marked buried in the cache")
        live = _fact(content="this one stays fully live")
        await adapter.upsert(buried, _vec())
        await adapter.upsert(live, _vec(seed=0.5))
        adapter._buried_ids.add(buried.id)

        collected = [meta async for meta in await adapter.scan()]
        ids = {m.id for m in collected}
        assert buried.id not in ids
        assert live.id in ids

