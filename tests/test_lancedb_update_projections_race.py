"""LanceDB-specific: ``update_projections`` must raise ``KeyError`` — not
silently no-op — when the target row is physically deleted strictly between
its own initial existence read and its terminal ``merge_insert`` write
.

Round 1: the matched-only ``merge_insert`` LanceDBAdapter.update_projections
issues touches zero rows (and does not raise by itself) when the row is gone
by write time — without checking the returned ``MergeResult.num_updated_rows``,
the call returned normally as if the write had landed, defeating every
caller (``facts.reinforce``, ``feedback.feedback``, and the lock-free
``rebuild_projections`` maintenance verb) that depends on this method raising
to detect a lost race. Fixed by checking ``num_updated_rows == 0`` (the same
idiom ``transition``'s supersede-edge CAS check already uses).

Round 2: ``tests/conformance/test_conformance.py``'s dual-parameterized race
test (``TestUpdateProjectionsConcurrentBury``) drives this via two genuinely
concurrent ``asyncio.create_task`` calls — deterministic for ``MemoryAdapter``
(a single synchronous cooperative yield point), but LanceDB's read and write
are two independently-awaited real I/O calls with no externally-hookable
checkpoint between them, so task-creation order alone does not *guarantee*
which of the two coroutines' internal awaits interleaves first — only
observed empirically there (20/20 probe runs landed in the intended branch,
judged by ``KeyError``'s message, before this file existed). That test stays
(it is real coverage, calling the actual public method end-to-end, and has
not been observed to flip in practice) — this file adds a deterministic
companion that does not depend on scheduler behavior at all: it drives the
exact two primitives ``LanceDBAdapter.update_projections`` itself issues,
with the row's physical deletion (``self._records.delete(...)`` — the same
primitive ``bury()`` uses) injected explicitly between them, and asserts the
resulting ``MergeResult`` is exactly the zero-matched-rows outcome that
method's own ``num_updated_rows`` check converts into ``KeyError``.

Round 3: a SECOND, distinct gap in the same method — ``LanceDBAdapter.
update_projections`` never checked ``_buried_ids`` at all (unlike
``get_by_id``/``scan``/``scan_records``, which already defend against the
documented query-planner-staleness / same-process race: a completed
``bury()`` — tombstone written, physical delete already awaited,
``_buried_ids`` already updated — whose delete has not yet propagated to a
*later*, independently-issued query's own snapshot). The round-1 fix (the
``num_updated_rows`` check) only catches a row genuinely deleted strictly
between THIS call's own read and write; it does not catch a row that is
STALE-VISIBLE to this call's read despite already being fully, durably
buried moments earlier — that read would still see it, the merge would
still match it (``num_updated_rows == 1``), and the method would return as
if the write had landed on a live record. Fixed by adding the same
``_buried_ids`` pre-read guard ``get_by_id`` already carries.
``TestUpdateProjectionsBuriedIdCacheGuard`` below is the regression test,
same construction idiom as ``test_lancedb_scan_buried_guard.py``'s own
guard tests (mark ``_buried_ids`` directly, leave the physical row present —
not calling ``bury()``, whose own delete would collapse the very staleness
window being tested down to the already-covered ``num_updated_rows`` case).

Contract-grounded reason this is a LanceDB-only test file, not a
dual-parameterized conformance property (same shape as
``test_lancedb_scan_buried_guard.py``'s own exemption note): ``merge_insert``'s
matched-only, zero-rows-touched-is-not-an-error semantics AND the
``_buried_ids`` staleness cache are both LanceDB-specific storage details.
``MemoryAdapter`` has neither — its own version of the round-1 race (a stale
pre-yield object reference written through instead of raising) is a
completely different failure shape, already covered deterministically by
the dual-parameterized conformance test (a single synchronous cooperative
yield point makes task-creation order alone sufficient there); it has no
async storage engine and no query-planner-staleness concept at all, so
round 3's gap is structurally impossible to construct there.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import record_to_row, row_to_record
from tests.support.lance_raw import raw_table

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _vec(dim: int = 4, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _fact(
    *, id: HypomnemaId | None = None, content: str = "update_projections race target"
) -> Fact:
    return Fact(
        id=id if id is not None else new_id(),
        type="fact",
        content=content,
        scope=Scope("proj-a"),
        lifecycle="active",
        validity="valid",
        created_at=EPOCH,
        updated_at=EPOCH,
    )


async def _open(tmp_path: pathlib.Path, name: str) -> LanceDBAdapter:
    return await LanceDBAdapter.open(str(tmp_path / name), embedding_model="test-model", dim=4)


class TestUpdateProjectionsRaceMechanism:
    async def test_deleted_mid_call_produces_the_zero_matched_merge_the_fix_checks_for(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Deterministic construction of the round-1 race window: read the
        row (proving it exists at read time — what a real race leaves
        behind, never a vacuous already-buried starting state), physically
        delete it via the exact primitive ``bury()`` itself uses, strictly
        between that read and a write built with the identical shape
        ``LanceDBAdapter.update_projections`` issues — then assert the
        resulting ``MergeResult.num_updated_rows == 0``, the precise
        condition that method's ``if result.num_updated_rows == 0: raise
        KeyError`` line converts into the caller-visible signal every caller
        depends on.

        This proves the MECHANISM the fix relies on deterministically,
        without needing to win a scheduler race to observe it — see the
        module docstring for how this complements (not replaces) the
        dual-parameterized conformance test that exercises the real public
        method end-to-end.

        Falsify: without the round-1 fix in place (delete the ``if
        result.num_updated_rows == 0`` block in ``LanceDBAdapter.
        update_projections``), the real method silently no-ops on this exact
        scenario — that regression is what
        ``tests/conformance/test_conformance.py::
        TestUpdateProjectionsConcurrentBury`` and ``tests/
        test_projection_rebuild.py::TestConcurrentBuryDuringRepair`` both go
        red on; this test pins the underlying primitive behaviour those two
        rely on being true.
        """
        adapter = await _open(tmp_path, "update_projections_race_store")
        fact = _fact()
        await adapter.upsert(fact, _vec())

        # Phase 1 — the exact read LanceDBAdapter.update_projections performs.
        rows = await raw_table(adapter._records).query().where(f"id = '{fact.id!s}'").to_list()
        assert rows, "the row must exist at read time -- what a real race leaves behind"
        ex_rec = row_to_record(rows[0])
        existing_vector_raw = rows[0].get("vector")
        existing_vector = (
            [float(x) for x in existing_vector_raw] if existing_vector_raw else None
        )
        updated_rec = ex_rec.model_copy(
            update={"recall_count": 1, "feedback_score": 0.5}
        )
        row = record_to_row(updated_rec, existing_vector, adapter._embedding_model)
        arrow_table = pa.table(
            {k: [v] for k, v in row.items()}, schema=adapter._records_schema
        )

        # Phase boundary — the exact window round 1 found: a concurrent
        # bury() lands here, strictly after the read above and before the
        # write below.
        await raw_table(adapter._records).delete(f"id = '{fact.id!s}'")

        # Phase 2 — the exact write LanceDBAdapter.update_projections issues.
        result = await (
            raw_table(adapter._records).merge_insert("id")
            .when_matched_update_all()
            .execute(arrow_table)
        )
        assert result.num_updated_rows == 0, (
            "a row deleted strictly between update_projections' own read "
            "and write must produce a zero-matched merge -- the exact "
            "outcome its num_updated_rows check converts into KeyError"
        )


class TestUpdateProjectionsBuriedIdCacheGuard:
    """Round 3: the ``_buried_ids`` staleness guard ``get_by_id``/``scan``/
    ``scan_records`` already carry (``test_lancedb_scan_buried_guard.py``),
    now on ``update_projections`` too — a completed burial whose physical
    delete has not yet propagated to a later query's own snapshot must not
    let ``update_projections`` silently succeed against the tombstoned row.
    """

    async def test_raises_and_writes_nothing_when_buried_id_cache_is_set(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Simulates the staleness window directly: the cache is updated
        (as ``bury()`` does, after its own delete already landed) WITHOUT
        physically removing the row from THIS query's view. Not calling
        ``adapter.bury()`` on purpose — its delete would also remove the
        physical row, collapsing the very race being tested down to the
        already-covered ``num_updated_rows`` case (the test above).

        Falsify: remove the ``if id in self._buried_ids: raise KeyError``
        guard added at the top of ``LanceDBAdapter.update_projections`` —
        this test goes red (no exception, and the row's ``recall_count`` is
        found mutated below) instead of green.
        """
        adapter = await _open(tmp_path, "update_projections_buried_cache_store")
        fact = _fact()
        await adapter.upsert(fact, _vec())

        adapter._buried_ids.add(fact.id)

        with pytest.raises(KeyError):
            await adapter.update_projections(
                fact.id,
                recall_count=1,
                last_referenced_at=None,
                feedback_score=0.9,
                decay_due_at=None,
            )

        # The row must be untouched — read through the same raw query path
        # update_projections itself uses (adapter.get_by_id would ALSO
        # raise here via its own _buried_ids check, so it could not
        # distinguish "never wrote" from "wrote, then this became
        # unreadable").
        rows = await raw_table(adapter._records).query().where(f"id = '{fact.id!s}'").to_list()
        assert rows, "the physical row must still be present (never deleted in this test)"
        assert rows[0]["recall_count"] == 0, (
            "update_projections must not have written through the "
            "_buried_ids guard"
        )

    async def test_buried_while_the_read_itself_is_in_flight_is_still_caught(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Round 4: the pre-read ``_buried_ids``
        check (round 3, test above) cannot see a burial that completes
        WHILE ``update_projections``'s own read is itself being awaited —
        only a re-check immediately after that read returns can. Driven
        deterministically (no real concurrency needed): a monkeypatch on
        ``raw_table(adapter._records).query`` — the same "reassign an attribute on
        ``adapter._records``" idiom ``tests/test_lancedb_maintenance.py``
        already uses — marks ``_buried_ids`` synchronously at the exact
        moment ``update_projections``'s real read begins, simulating a
        ``bury()`` that finishes during that read's real await window,
        entirely inside this one coroutine's single execution.

        The two ``_buried_ids`` checks raise with distinct message text
        (see ``LanceDBAdapter.update_projections``), so asserting on the
        message — not just ``KeyError``'s type — proves the SECOND
        (post-read) check is what fired, not merely the first one having
        somehow fired again.

        Falsify: remove the second ``if id in self._buried_ids`` check
        (immediately after the awaited read) added in round 4 — this test
        goes red (no exception at all, real write lands) instead of green.
        """
        adapter = await _open(tmp_path, "update_projections_buried_mid_read_store")
        fact = _fact()
        await adapter.upsert(fact, _vec())

        real_query = raw_table(adapter._records).query

        def _query_that_buries_mid_flight(*args: object, **kwargs: object) -> object:
            # Runs at the exact point update_projections' own read begins —
            # simulating a bury() that completes during that read's real
            # await window, without needing a second concurrent coroutine.
            adapter._buried_ids.add(fact.id)
            return real_query(*args, **kwargs)

        raw_table(adapter._records).query = _query_that_buries_mid_flight

        with pytest.raises(KeyError) as exc_info:
            await adapter.update_projections(
                fact.id,
                recall_count=1,
                last_referenced_at=None,
                feedback_score=0.9,
                decay_due_at=None,
            )
        assert "while the read was in flight" in str(exc_info.value), (
            f"expected the post-read _buried_ids check to fire, got: "
            f"{exc_info.value!r}"
        )
