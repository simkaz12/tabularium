"""Adapter conformance suite (SPEC §3 + BUILD-PLAN M0/M1).

Every ``StorageAdapter`` implementation must pass this suite unchanged.
LanceDB (M1) plugs in via the ``adapter`` fixture parametrization, alongside
the in-memory reference adapter.  Any future adapter would follow the same
pattern.  No test-body changes needed.

Coverage:
- Write-visibility rule [A4]: committed upsert visible to both legs immediately
- Pre-filter correctness: scope, validity, lifecycle_in
- k semantics: k is honoured exactly at any (n, k); vector leg skips vector-less (buried)
- Decision content-immutability at upsert
- Vector required on normal path [A2]
- Supersession atomicity + set-exactly-once
- transition is the ONLY lifecycle mutator; updated_at=at persisted
- set_validity and update_projections do NOT touch updated_at
- Bury completeness: content gone, both legs miss it, get_by_id raises,
  tombstone written with all §1.3 fields
- Event log: append + query by target/source; update_projections round-trip
- DistinctPair: normalization, first-judgment-wins, is_distinct symmetric
- Registries: create + list roundtrip, duplicate idempotent
- backfill_targets/backfill round-trip after a real embedding-model reopen
- Hypothesis property tests for filter correctness (both adapters)
- Guard: every property above runs on both adapters, or is named in
  ``_MEMORY_ONLY_EXEMPTIONS`` with a contract-grounded reason (currently
  empty — see ``test_every_property_runs_on_both_adapters``)
"""

from __future__ import annotations

import asyncio
import inspect
import math
import sys
import uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tabularium.domain import (
    Decision,
    DecisionLifecycle,
    Fact,
    FactLifecycle,
    HypomnemaId,
    PromotedFromSnapshot,
    Scope,
    Session,
    SupersedesSnapshot,
    SupersessionReason,
    new_id,
)
from tabularium.storage.content_key import content_hash, content_key
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import record_judgment_to_row
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import (
    CleanerPassRecord,
    CleanerState,
    DecisionInvalidation,
    DistinctPair,
    Filters,
    PredecessorFlip,
    ReadEvent,
    RecordJudgment,
    RecordReview,
    ReinforcementEvent,
    RowMeta,
    Tombstone,
    UnsupersedeEvent,
    VerdictRow,
)
from tests.support.doubles import TestClock, reopen_memory_at
from tests.support.edits import edit_content
from tests.support.lance_raw import raw_table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UTC = UTC
EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

# Union type accepted by all conformance tests
AnyAdapter = MemoryAdapter | LanceDBAdapter

# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------


_LANCEDB_STORE_SUBDIR = "conformance_store"


def _make_memory_adapter(
    embedding_model: str = "test-model",
    dim: int = 4,
    clock: TestClock | None = None,
) -> MemoryAdapter:
    return MemoryAdapter(embedding_model=embedding_model, dim=dim, clock=clock)


async def _make_lancedb_adapter(
    tmp_path: Any,
    embedding_model: str = "test-model",
    dim: int = 4,
    clock: TestClock | None = None,
) -> LanceDBAdapter:
    return await LanceDBAdapter.open(
        str(tmp_path / _LANCEDB_STORE_SUBDIR),
        embedding_model=embedding_model,
        dim=dim,
        clock=clock,
    )


async def _reopen_at_model(
    adapter: AnyAdapter,
    *,
    embedding_model: str,
    tmp_path: Any,
    clock: TestClock,
    dim: int = 4,
) -> AnyAdapter:
    """Return a SECOND adapter view over the same underlying store as
    ``adapter``, configured at a different ``embedding_model`` — the honest
    way to express "the store persists, the configured embedder changed"
    (SPEC §3.4), without reaching into adapter internals from test code.

    Memory expresses this via ``MemoryAdapter.reopen_at`` (shares the backing
    rows dict); LanceDB via a second ``LanceDBAdapter.open()`` call against
    the same on-disk path (a real reopen, including its own tombstone/pair
    cache reload — see ``LanceDBAdapter._load_buried_ids``).
    """
    if isinstance(adapter, MemoryAdapter):
        return reopen_memory_at(adapter, embedding_model=embedding_model)
    return await LanceDBAdapter.open(
        str(tmp_path / _LANCEDB_STORE_SUBDIR),
        embedding_model=embedding_model,
        dim=dim,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# Fixtures — parameterized over memory + lancedb
# ---------------------------------------------------------------------------


@pytest.fixture()
def clock() -> TestClock:
    """A pinned TestClock for deterministic timestamps."""
    return TestClock(EPOCH)


@pytest.fixture(params=["memory", "lancedb"])
async def adapter(request: pytest.FixtureRequest, clock: TestClock, tmp_path: Any) -> AnyAdapter:
    """A fresh adapter with an injected TestClock.  Parameterized over both adapters."""
    if request.param == "memory":
        return _make_memory_adapter(clock=clock)
    return await _make_lancedb_adapter(tmp_path, clock=clock)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _vec(dim: int = 4, *, seed: float = 1.0) -> list[float]:
    """Unit vector of length ``dim`` with all components equal to ``seed``."""
    v = [seed] * dim
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _orthogonal_vec(dim: int = 4) -> list[float]:
    """Vector orthogonal to ``_vec()`` in 4D — cosine sim = 0."""
    # [1, -1, 0, 0, ...] normalized
    v = [0.0] * dim
    v[0] = 1.0
    if dim > 1:
        v[1] = -1.0
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _ts(offset_days: int = 0) -> datetime:
    return EPOCH + timedelta(days=offset_days)


def _scope(name: str = "proj-a") -> Scope:
    return Scope(name)


def _fact(
    *,
    scope: str = "proj-a",
    lifecycle: str = "active",
    validity: str = "valid",
    content: str = "Test fact content",
    clock: TestClock | None = None,
    **kw: Any,
) -> Fact:
    now = clock.now() if clock else EPOCH
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
        **kw,
    )


def _decision(
    *,
    scope: str = "proj-a",
    lifecycle: str = "in_force",
    validity: str = "valid",
    content: str = "Test decision content",
    clock: TestClock | None = None,
    **kw: Any,
) -> Decision:
    now = clock.now() if clock else EPOCH
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
        **kw,
    )


def _flip(
    predecessor_id: HypomnemaId,
    *,
    reason: SupersessionReason = "correction",
    note: str | None = None,
) -> PredecessorFlip:
    """One ``replace()`` flip against ``predecessor_id`` — the conformance
    suite's stand-in for the engine's own ``uuid4()``-minted edge id."""
    return PredecessorFlip(id=predecessor_id, reason=reason, note=note, edge_id=str(uuid.uuid4()))


async def _raw_vector(adapter: AnyAdapter, id: HypomnemaId) -> list[float] | None:
    """Read a row's raw stored vector directly — ``Fact``/``Decision`` carry
    no vector field, so ``get_by_id`` cannot answer this; there is no such
    method on the public ``StorageAdapter`` protocol either, so this reaches
    into each adapter's own storage directly (test-only, conformance suite
    convention already established for adapter-specific probes)."""
    if isinstance(adapter, MemoryAdapter):
        row = adapter._rows.get(id)
        return list(row.vector) if row is not None and row.vector is not None else None
    rows = await raw_table(adapter._records).query().where(f"id = '{id!s}'").to_list()
    if not rows:
        return None
    raw = rows[0].get("vector")
    return [float(x) for x in raw] if raw is not None else None


def _session(
    *,
    scope: str = "proj-a",
    started_at: datetime = EPOCH,
    ended_at: datetime = EPOCH,
    content: str = "Digested the sessions plan.",
    **kw: Any,
) -> Session:
    return Session(
        id=uuid.uuid4(),
        scope=scope,
        started_at=started_at,
        ended_at=ended_at,
        content=content,
        **kw,
    )


def _filters(
    scopes: list[str] | None = None,
    validity: set[str] | None = None,
    lifecycle_in: set[FactLifecycle | DecisionLifecycle] | None = None,
) -> Filters:
    return Filters(
        scopes=[Scope(s) for s in (scopes or ["proj-a", "global"])],
        validity=frozenset(validity or {"valid"}),  # type: ignore[arg-type]
        lifecycle_in=frozenset(lifecycle_in or {"active", "hot", "ember"}),
    )


def _all_filters(scopes: list[str] | None = None) -> Filters:
    """Filters that include every lifecycle state and both validity values."""
    return Filters(
        scopes=[Scope(s) for s in (scopes or ["proj-a", "global"])],
        validity=frozenset({"valid", "invalid"}),
        lifecycle_in=frozenset({"active", "hot", "ember", "archived", "in_force",
                                "superseded", "revoked"}),
    )


# ===========================================================================
# 1 — Write-visibility rule [A4]
# ===========================================================================


class TestWriteVisibility:
    """A committed upsert is visible to BOTH legs on the immediately-next search."""

    async def test_vector_leg_sees_upsert(self, adapter: AnyAdapter) -> None:
        fact = _fact(content="LanceDB is the default vector store")
        v = _vec()
        await adapter.upsert(fact, v)

        hits = await adapter.vector_search(v, filters=_filters(), k=5)
        assert len(hits) == 1
        assert hits[0].record.id == fact.id
        assert hits[0].leg == "vector"

    async def test_text_leg_sees_upsert(self, adapter: AnyAdapter) -> None:
        fact = _fact(content="LanceDB is the default vector store")
        v = _vec()
        await adapter.upsert(fact, v)

        hits = await adapter.text_search("LanceDB vector store", filters=_filters(), k=5)
        assert len(hits) >= 1
        ids = {h.record.id for h in hits}
        assert fact.id in ids
        assert all(h.leg == "text" for h in hits)


# ===========================================================================
# 1b — Ranked visibility (maintenance_debt / upkeep)
# ===========================================================================


def _seed_ranked_visibility_fixture(clock: TestClock) -> tuple[list[Fact], Fact]:
    """29 genuine lexical competitors + one lexically obvious answer.

    Every distractor shares ALL THREE query tokens ("quokka", "happiest",
    "animal") with the query below — they are not zero-overlap filler, so
    BM25 must actually do its job (and ``upkeep()`` must actually have run)
    for the obvious match to rank first. A distractor set with no token
    overlap would let ``text_search`` return exactly one hit regardless of
    whether maintenance ran at all, which is precisely the test-power defect
    this fixture corrects (see Attempt 1 review).
    """
    distractors = [
        _fact(
            content=(
                "Various surveys about the happiest animal on the planet often "
                f"mention the quokka alongside the {name}, though opinions vary widely"
            ),
            clock=clock,
        )
        for name in (
            "elephant", "dolphin", "penguin", "koala", "panda", "otter", "fox",
            "rabbit", "meerkat", "lemur", "sloth", "hedgehog", "raccoon", "badger",
            "beaver", "squirrel", "chipmunk", "opossum", "wombat", "capybara",
            "platypus", "armadillo", "tapir", "aardvark", "pangolin", "okapi",
            "quoll", "bandicoot", "numbat",
        )
    ]
    obvious = _fact(content="the quokka is the happiest animal alive", clock=clock)
    return distractors, obvious


async def _upkeep_until_stable(adapter: AnyAdapter, *, max_calls: int = 6) -> Any:
    """Call ``upkeep()`` repeatedly until ``maintenance_debt()`` stops changing.

    LanceDB's on-disk compaction can take more than one pass to fully merge a
    previously-optimized fragment with newly-written ones (confirmed against
    the pinned version); this loop absorbs that implementation detail so
    idempotence assertions test the real invariant — repeat calls at a fixed
    point are stable — rather than an accidental mid-compaction snapshot.
    Always calls ``upkeep()`` at least once.
    """
    debt = await adapter.maintenance_debt()
    for _ in range(max_calls):
        await adapter.upkeep()
        next_debt = await adapter.maintenance_debt()
        if next_debt == debt:
            return next_debt
        debt = next_debt
    raise AssertionError(f"maintenance_debt() did not stabilize within {max_calls} upkeep() calls")


@pytest.mark.ranked_visibility
class TestRankedVisibility:
    """After maintenance, every record ranks correctly (SPEC §3.3, ADR 0017).

    Corrects ``TestWriteVisibility``'s structural weakness: that suite asserts
    mere membership at a sample size where BM25 cannot possibly rank wrong. This
    class writes a record among enough *genuinely competing* distractors that a
    ranking defect — not a presence/absence defect — would actually surface,
    and pins the contract to the two operations that make ranked visibility
    true: ``upkeep()`` folds pending writes into the index; ``maintenance_debt()``
    reports what upkeep would fix.
    """

    async def test_text_leg_ranks_obvious_match(
        self, adapter: AnyAdapter, clock: TestClock
    ) -> None:
        distractors, obvious = _seed_ranked_visibility_fixture(clock)
        for i, d in enumerate(distractors):
            await adapter.upsert(d, _vec(seed=float(i + 1) * 0.01))
        await adapter.upsert(obvious, _vec(seed=0.5))

        await adapter.upkeep()

        hits = await adapter.text_search("quokka happiest animal", filters=_filters(), k=5)
        assert hits[0].record.id == obvious.id

    async def test_upkeep_is_idempotent(self, adapter: AnyAdapter, clock: TestClock) -> None:
        """A second ``upkeep()`` call, once caught up, is a stable no-op.

        Unlike a bare "call it twice with nothing in between" check — which
        would agree trivially even for a fully broken (no-op) ``upkeep()`` —
        this test writes a fresh record *after* the first maintenance run and
        confirms the second run actually resolves the debt that write left
        behind (real work), before asserting the literal idempotence
        invariant on the now-settled state.
        """
        distractors, obvious = _seed_ranked_visibility_fixture(clock)
        for i, d in enumerate(distractors):
            await adapter.upsert(d, _vec(seed=float(i + 1) * 0.01))
        await adapter.upsert(obvious, _vec(seed=0.5))

        stable_debt = await _upkeep_until_stable(adapter)
        hits = await adapter.text_search("quokka happiest animal", filters=_filters(), k=5)
        assert hits[0].record.id == obvious.id

        # Intervening write: real, measurable debt the next maintenance run
        # must resolve. A no-op upkeep() would leave the ranking check below
        # failing exactly like `obvious` failed to rank before the first run.
        latecomer = _fact(
            content="a naked mole rat rarely discusses happiness underground",
            clock=clock,
        )
        await adapter.upsert(latecomer, _vec(seed=0.4))
        dirtied_debt = await adapter.maintenance_debt()
        assert dirtied_debt != stable_debt

        resettled_debt = await _upkeep_until_stable(adapter)
        hits_after_write = await adapter.text_search(
            "quokka happiest animal", filters=_filters(), k=5
        )
        assert hits_after_write[0].record.id == obvious.id

        # The actual idempotence property: at a fixed point, one more call
        # with no intervening writes changes nothing.
        await adapter.upkeep()
        final_debt = await adapter.maintenance_debt()
        assert final_debt == resettled_debt

        hits_final = await adapter.text_search(
            "quokka happiest animal", filters=_filters(), k=5
        )
        assert hits_final[0].record.id == obvious.id


# ===========================================================================
# 1c — Reclaim: the third maintenance verb, reachable only by name
# ===========================================================================


class TestReclaim:
    """``reclaim()`` is a separate, irreversible, named door (SPEC §3.3, ADR 0017).

    It is never a parameter on ``upkeep()`` — a boolean that flips a routine
    method into irreversible behaviour is the exact shape that let an
    unattended script prune version history past its own warning banner.
    On LanceDB, "prune versions" is directly observable: a version present
    before maintenance stays ``checkout()``-able after ``upkeep()`` and stops
    being ``checkout()``-able after an explicit ``reclaim()``.
    """

    async def test_upkeep_never_reclaims(
        self, adapter: AnyAdapter, clock: TestClock
    ) -> None:
        """``upkeep()`` must never prune version history — only ``reclaim()`` does."""
        f = _fact(content="version pruning target", clock=clock)
        await adapter.upsert(f, _vec())

        if isinstance(adapter, LanceDBAdapter):
            versions_before = await raw_table(adapter._records).list_versions()
            earliest_version = versions_before[0]["version"]

            await adapter.upkeep()

            # The version present before upkeep() must still be checkout-able —
            # upkeep() compacts and folds indexes but never prunes history.
            await raw_table(adapter._records).checkout(earliest_version)
            await raw_table(adapter._records).checkout_latest()
        else:
            # An in-memory store keeps no version history to prune — the
            # property holds trivially; upkeep() must still be callable.
            await adapter.upkeep()
            retrieved = await adapter.get_by_id(f.id)
            assert retrieved.id == f.id

    async def test_reclaim_prunes_versions(
        self, adapter: AnyAdapter, clock: TestClock
    ) -> None:
        """An explicit ``reclaim()`` call does what ``upkeep()`` never does."""
        f = _fact(content="reclaim pruning target", clock=clock)
        await adapter.upsert(f, _vec())

        if isinstance(adapter, LanceDBAdapter):
            versions_before = await raw_table(adapter._records).list_versions()
            earliest_version = versions_before[0]["version"]

            # A second write ensures the earliest version is not the latest —
            # reclaim() never removes the current version, only history.
            f2 = _fact(content="second write ages out the earlier version", clock=clock)
            await adapter.upsert(f2, _vec(seed=0.9))

            await adapter.reclaim()

            with pytest.raises(ValueError, match="no longer exists"):
                await raw_table(adapter._records).checkout(earliest_version)
            await raw_table(adapter._records).checkout_latest()

            # Live data is untouched — reclaim prunes history, not content.
            retrieved1 = await adapter.get_by_id(f.id)
            retrieved2 = await adapter.get_by_id(f2.id)
            assert retrieved1.id == f.id
            assert retrieved2.id == f2.id
        else:
            # Honest no-op: no version history exists to prune.
            await adapter.reclaim()
            retrieved = await adapter.get_by_id(f.id)
            assert retrieved.id == f.id


# ===========================================================================
# 2 — Pre-filter correctness
# ===========================================================================


class TestPreFilterCorrectness:
    """Scope / validity / lifecycle_in are enforced as pre-filters."""

    async def test_scope_isolation_vector(self, adapter: AnyAdapter) -> None:
        """A record in scope 'other' must never surface in a proj-a search."""
        in_scope = _fact(scope="proj-a", content="In scope record")
        out_scope = _fact(scope="other", content="Out of scope record")
        v = _vec()
        await adapter.upsert(in_scope, v)
        await adapter.upsert(out_scope, _vec(seed=0.9))

        hits = await adapter.vector_search(v, filters=_filters(scopes=["proj-a"]), k=10)
        ids = {h.record.id for h in hits}
        assert in_scope.id in ids
        assert out_scope.id not in ids

    async def test_scope_isolation_text(self, adapter: AnyAdapter) -> None:
        in_scope = _fact(scope="proj-a", content="unique keyword alpha")
        out_scope = _fact(scope="other", content="unique keyword alpha")
        v = _vec()
        await adapter.upsert(in_scope, v)
        await adapter.upsert(out_scope, _vec(seed=0.9))

        hits = await adapter.text_search(
            "unique keyword alpha", filters=_filters(scopes=["proj-a"]), k=10
        )
        ids = {h.record.id for h in hits}
        assert in_scope.id in ids
        assert out_scope.id not in ids

    async def test_global_scope_also_searched_vector(self, adapter: AnyAdapter) -> None:
        """Read in scope X sees X + global (SPEC §2)."""
        proj = _fact(scope="proj-a", content="project record")
        glob = _fact(scope="global", content="global record")
        v = _vec()
        await adapter.upsert(proj, v)
        await adapter.upsert(glob, _vec(seed=0.9))

        hits = await adapter.vector_search(
            v, filters=_filters(scopes=["proj-a", "global"]), k=10
        )
        ids = {h.record.id for h in hits}
        assert proj.id in ids
        assert glob.id in ids

    async def test_validity_filter_excludes_invalid_vector(self, adapter: AnyAdapter) -> None:
        valid = _fact(content="valid fact", validity="valid")
        invalid = _fact(content="invalid fact", validity="invalid")
        v = _vec()
        await adapter.upsert(valid, v)
        await adapter.upsert(invalid, _vec(seed=0.9))

        hits = await adapter.vector_search(v, filters=_filters(validity={"valid"}), k=10)
        ids = {h.record.id for h in hits}
        assert valid.id in ids
        assert invalid.id not in ids

    async def test_validity_filter_excludes_invalid_text(self, adapter: AnyAdapter) -> None:
        valid = _fact(content="chocolate keyword valid", validity="valid")
        invalid = _fact(content="chocolate keyword invalid", validity="invalid")
        v = _vec()
        await adapter.upsert(valid, v)
        await adapter.upsert(invalid, _vec(seed=0.9))

        hits = await adapter.text_search(
            "chocolate keyword", filters=_filters(validity={"valid"}), k=10
        )
        ids = {h.record.id for h in hits}
        assert valid.id in ids
        assert invalid.id not in ids

    async def test_validity_both_values_included_when_requested(
        self, adapter: AnyAdapter
    ) -> None:
        valid = _fact(content="valid record", validity="valid")
        invalid = _fact(content="invalid record", validity="invalid")
        v = _vec()
        await adapter.upsert(valid, v)
        await adapter.upsert(invalid, _vec(seed=0.9))

        hits = await adapter.vector_search(
            v, filters=_filters(validity={"valid", "invalid"}), k=10
        )
        ids = {h.record.id for h in hits}
        assert valid.id in ids
        assert invalid.id in ids

    async def test_lifecycle_filter_vector(self, adapter: AnyAdapter) -> None:
        active = _fact(content="active fact", lifecycle="active")
        archived = _fact(content="archived fact", lifecycle="archived")
        v = _vec()
        await adapter.upsert(active, v)
        await adapter.upsert(archived, _vec(seed=0.9))

        hits = await adapter.vector_search(
            v,
            filters=_filters(lifecycle_in={"active"}),
            k=10,
        )
        ids = {h.record.id for h in hits}
        assert active.id in ids
        assert archived.id not in ids

    async def test_lifecycle_filter_text(self, adapter: AnyAdapter) -> None:
        active = _fact(content="frozen banana active", lifecycle="active")
        archived = _fact(content="frozen banana archived", lifecycle="archived")
        v = _vec()
        await adapter.upsert(active, v)
        await adapter.upsert(archived, _vec(seed=0.9))

        hits = await adapter.text_search(
            "frozen banana",
            filters=_filters(lifecycle_in={"active"}),
            k=10,
        )
        ids = {h.record.id for h in hits}
        assert active.id in ids
        assert archived.id not in ids

    async def test_all_lifecycle_states_included_when_broadened(
        self, adapter: AnyAdapter
    ) -> None:
        active = _fact(content="active one", lifecycle="active")
        archived = _fact(content="archived one", lifecycle="archived")
        v = _vec()
        await adapter.upsert(active, v)
        await adapter.upsert(archived, _vec(seed=0.9))

        hits = await adapter.vector_search(
            v,
            filters=_filters(lifecycle_in={"active", "archived"}),
            k=10,
        )
        ids = {h.record.id for h in hits}
        assert active.id in ids
        assert archived.id in ids

    async def test_shelved_lifecycle_included_when_requested(
        self, adapter: AnyAdapter
    ) -> None:
        """New seam capability (ADR 0021): ``lifecycle_in={"shelved"}`` is a
        real pre-filter value on both adapters, same as any other lifecycle
        string — the ``lifecycle``/``lifecycle_in`` columns are untyped
        ``utf8`` at the storage seam (no schema migration needed for a new
        value)."""
        active = _fact(content="active fact for shelved filter test", lifecycle="active")
        shelved = _fact(content="shelved fact for shelved filter test", lifecycle="shelved")
        v = _vec()
        await adapter.upsert(active, v)
        await adapter.upsert(shelved, _vec(seed=0.9))

        hits = await adapter.vector_search(
            v, filters=_filters(lifecycle_in={"shelved"}), k=10
        )
        ids = {h.record.id for h in hits}
        assert shelved.id in ids
        assert active.id not in ids

    async def test_shelved_lifecycle_excluded_when_not_requested(
        self, adapter: AnyAdapter
    ) -> None:
        """The dedup-probe shape: a filter naming every OTHER lifecycle but
        NOT ``shelved`` never returns a shelved record — this is exactly
        what ``engine.filters.probe_filters_for`` relies on (ADR 0021)."""
        shelved = _fact(
            content="shelved fact excluded from a probe-shaped filter",
            lifecycle="shelved",
        )
        v = _vec()
        await adapter.upsert(shelved, v)

        probe_shaped = _filters(
            lifecycle_in={
                "active", "hot", "ember", "archived",
                "in_force", "superseded", "revoked",
            }
        )
        hits = await adapter.vector_search(v, filters=probe_shaped, k=10)
        ids = {h.record.id for h in hits}
        assert shelved.id not in ids

        text_hits = await adapter.text_search(
            "shelved fact excluded", filters=probe_shaped, k=10
        )
        assert shelved.id not in {h.record.id for h in text_hits}


# ===========================================================================
# 3 — k semantics
# ===========================================================================


class TestKSemantics:
    """k caps results and is honoured exactly at any (n, k); buried skipped."""

    @pytest.mark.parametrize(
        "n,k",
        [(50, 25), (120, 100), (120, 11)],
        ids=["n50_k25", "n120_k100", "n120_k11"],
    )
    async def test_k_is_honoured_exactly(self, adapter: AnyAdapter, n: int, k: int) -> None:
        """``k`` is honoured exactly: ``len(hits) == min(n, k)`` on both legs.

        Covers ``n > 10`` and ``k > 10`` -- the region a vendor-hardcoded
        default (LanceDB's own ``limit(10)``, applied only when the caller
        never calls ``.limit()`` at all) could silently substitute for the
        requested ``k`` without either leg's count ever proving it (ledger
        7999 / ADR 0018).
        """
        for i in range(n):
            f = _fact(content=f"honoured record {i} keyword")
            await adapter.upsert(f, _vec(seed=float(i + 1) * 0.001))

        vector_hits = await adapter.vector_search(_vec(), filters=_filters(), k=k)
        assert len(vector_hits) == min(n, k)

        text_hits = await adapter.text_search("honoured keyword", filters=_filters(), k=k)
        assert len(text_hits) == min(n, k)

    async def test_k_caps_vector(self, adapter: AnyAdapter) -> None:
        for i in range(5):
            f = _fact(content=f"record {i}")
            await adapter.upsert(f, _vec(seed=float(i + 1) * 0.1))

        hits = await adapter.vector_search(_vec(), filters=_filters(), k=3)
        assert len(hits) == 3

    async def test_k_caps_text(self, adapter: AnyAdapter) -> None:
        for i in range(5):
            f = _fact(content=f"keyword apple {i} extra words")
            await adapter.upsert(f, _vec(seed=float(i + 1) * 0.1))

        hits = await adapter.text_search("keyword apple", filters=_filters(), k=2)
        assert len(hits) == 2

    async def test_vector_search_skips_buried(self, adapter: AnyAdapter) -> None:
        f = _fact(content="I will be buried")
        v = _vec()
        await adapter.upsert(f, v)
        await adapter.bury(f.id, "test burial")

        hits = await adapter.vector_search(v, filters=_all_filters(), k=10)
        ids = {h.record.id for h in hits}
        assert f.id not in ids

    async def test_rank_field_is_1_indexed(self, adapter: AnyAdapter) -> None:
        for i in range(3):
            f = _fact(content=f"ranked record {i}")
            await adapter.upsert(f, _vec(seed=float(i + 1) * 0.1))

        hits = await adapter.vector_search(_vec(), filters=_filters(), k=3)
        ranks = [h.rank for h in hits]
        assert ranks == [1, 2, 3]

    async def test_text_rank_field_is_1_indexed(self, adapter: AnyAdapter) -> None:
        for i in range(3):
            f = _fact(content=f"mango keyword {i}")
            await adapter.upsert(f, _vec(seed=float(i + 1) * 0.1))

        hits = await adapter.text_search("mango keyword", filters=_filters(), k=10)
        assert len(hits) >= 1
        for i, h in enumerate(hits):
            assert h.rank == i + 1

    async def test_empty_query_text_search_returns_empty(self, adapter: AnyAdapter) -> None:
        f = _fact(content="some content")
        await adapter.upsert(f, _vec())

        hits = await adapter.text_search("", filters=_filters(), k=10)
        assert hits == []


# ===========================================================================
# 4 — Decision content-immutability
# ===========================================================================


class TestDecisionImmutability:
    """upsert rejects changed content on an existing Decision (SPEC §3.1 + §6)."""

    async def test_same_content_upsert_is_accepted(self, adapter: AnyAdapter) -> None:
        d = _decision(content="Tabularium uses the Clock seam.")
        v = _vec()
        await adapter.upsert(d, v)
        # Same content, different vector — must succeed
        await adapter.upsert(d, _vec(seed=0.9))

    async def test_changed_content_raises(self, adapter: AnyAdapter) -> None:
        d = _decision(content="Original decision content.")
        v = _vec()
        await adapter.upsert(d, v)

        mutated = Decision(
            id=d.id,
            type="decision",
            content="Changed content — must be rejected.",
            scope=d.scope,
            lifecycle=d.lifecycle,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        with pytest.raises(ValueError, match="immutable"):
            await adapter.upsert(mutated, _vec(seed=0.8))

    async def test_fact_content_change_is_accepted(self, adapter: AnyAdapter) -> None:
        """Facts have no content-immutability requirement (only Decisions do)."""
        f = _fact(content="original fact content")
        await adapter.upsert(f, _vec())

        updated = Fact(
            id=f.id,
            type="fact",
            content="changed fact content",
            scope=f.scope,
            lifecycle=f.lifecycle,
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        await adapter.upsert(updated, _vec(seed=0.8))  # must not raise
        result = await adapter.get_by_id(f.id)
        assert result.content == "changed fact content"


async def _raw_row(adapter: AnyAdapter, id: HypomnemaId) -> dict[str, Any]:
    """Storage-layer-only view of a row (P5 edit-on-the-row piece): ``content_key``/
    ``content_hash`` are never read back onto the domain object, and ``embedding_model``
    lives only on the storage row, so a test pinning either has nowhere else to look."""
    if isinstance(adapter, LanceDBAdapter):
        rows = await raw_table(adapter._records).query().where(f"id = '{id!s}'").to_list()
        row = rows[0]
        return {
            "content": row["content"],
            "previous_content": row.get("previous_content"),
            "content_key": row["content_key"],
            "content_hash": row["content_hash"],
            "vector": row["vector"],
            "embedding_model": row["embedding_model"],
        }

    mem_row = adapter._rows[id]
    return {
        "content": mem_row.record.content,
        "previous_content": mem_row.record.previous_content,
        "content_key": mem_row.content_key,
        "content_hash": mem_row.content_hash,
        "vector": list(mem_row.vector) if mem_row.vector is not None else None,
        "embedding_model": mem_row.embedding_model,
    }


# ===========================================================================
# 5 — Vector required [A2]
# ===========================================================================


class TestVectorOptional:
    """upsert accepts a null vector (P6 vector-less-rows piece, D3, superseding [A2]'s old
    non-empty requirement): the row lands with ``vector=None`` and ``embedding_model=None``
    instead of raising — the embedder was unavailable at write time."""

    async def test_none_vector_lands_vectorless(self, adapter: AnyAdapter) -> None:
        f = _fact(content="will land vectorless")
        await adapter.upsert(f, None)
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.content == "will land vectorless"
        assert retrieved.embedding_model is None

    async def test_empty_list_vector_lands_vectorless(self, adapter: AnyAdapter) -> None:
        f = _fact(content="will also land vectorless")
        await adapter.upsert(f, [])
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.content == "will also land vectorless"
        assert retrieved.embedding_model is None

    async def test_wrong_dim_upsert_raises(self, adapter: AnyAdapter) -> None:
        """Dim mismatch on upsert raises (one-store one-dim contract)."""
        f = _fact(content="dim mismatch test")
        with pytest.raises(ValueError, match="dimension"):
            await adapter.upsert(f, [0.5, 0.5])  # dim=2, adapter expects 4

    async def test_wrong_dim_vector_search_raises(self, adapter: AnyAdapter) -> None:
        """Dim mismatch on vector_search raises (including empty vector)."""
        with pytest.raises(ValueError, match="dimension"):
            await adapter.vector_search([0.5, 0.5], filters=_filters(), k=5)

    async def test_empty_vector_search_raises(self, adapter: AnyAdapter) -> None:
        """An empty query vector raises rather than returning arbitrary results (F10)."""
        with pytest.raises(ValueError, match="dimension"):
            await adapter.vector_search([], filters=_filters(), k=5)

    async def test_wrong_dim_backfill_raises(self, adapter: AnyAdapter) -> None:
        """Dim mismatch on backfill raises.

        ``adapter`` is already opened at dim=4 (the fixture's default) — the
        exact value this test needs — so no separate construction is needed
        to exercise the non-default-dim contract on both backends.
        """
        f = _fact(content="backfill dim test")
        await adapter.upsert(f, _vec())
        with pytest.raises(ValueError, match="dimension"):
            await adapter.backfill(
                f.id, [0.5, 0.5], embedding_model="nomic-v2", expected_hash="irrelevant"
            )


# ===========================================================================
# 5b — Lifecycle-change-via-upsert rejected (F1 invariant)
# ===========================================================================


class TestLifecycleUpsertRejection:
    """Lifecycle changes on existing rows must go through transition, not upsert."""

    async def test_lifecycle_change_on_fact_via_upsert_raises(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact(content="lifecycle change test", lifecycle="active")
        await adapter.upsert(f, _vec())

        # Try to change lifecycle via re-upsert — must raise
        updated = Fact(
            id=f.id,
            type="fact",
            content=f.content,
            scope=f.scope,
            lifecycle="archived",  # changed
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        with pytest.raises(ValueError, match="lifecycle"):
            await adapter.upsert(updated, _vec(seed=0.8))

    async def test_same_lifecycle_upsert_is_accepted(self, adapter: AnyAdapter) -> None:
        f = _fact(content="same lifecycle ok", lifecycle="active")
        await adapter.upsert(f, _vec())

        # Re-upsert same lifecycle — must succeed
        same = Fact(
            id=f.id,
            type="fact",
            content="updated content ok",  # Facts allow content change
            scope=f.scope,
            lifecycle="active",
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        await adapter.upsert(same, _vec(seed=0.8))  # must not raise

    async def test_superseded_by_change_via_upsert_raises(
        self, adapter: AnyAdapter
    ) -> None:
        """Changing superseded_by via re-upsert must be rejected; use transition."""
        d = _decision(content="decision content")
        await adapter.upsert(d, _vec())

        d2 = _decision(content="decision two")
        await adapter.upsert(d2, _vec(seed=0.9))

        # Set superseded_by properly via replace first
        await adapter.replace(d2, None, [_flip(d.id)], at=EPOCH)

        d3 = _decision(content="decision three")
        await adapter.upsert(d3, _vec(seed=0.8))

        # Now try to change superseded_by to d3 via re-upsert — must raise
        # Build the Decision object with the change (lifecycle already "superseded")
        mutated = Decision(
            id=d.id,
            type="decision",
            content=d.content,
            scope=d.scope,
            lifecycle="superseded",
            superseded_by=d3.id,  # changed from d2 — must be rejected
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        with pytest.raises(ValueError):
            await adapter.upsert(mutated, _vec(seed=0.6))

    async def test_fact_superseded_by_change_via_upsert_raises(
        self, adapter: AnyAdapter
    ) -> None:
        """Fact.superseded_by must not change via upsert either — the SAME guard as the Decision
        case
        above (``test_superseded_by_change_via_upsert_raises``), mirrored
        for the Fact arm now that ``superseded_by`` is legal there too.
        Set-exactly-once, ONLY through the guarded ``transition()`` path
        (lock, span check, first-edge-wins, legality) — never a bare
        re-upsert, which would bypass all of it.

        Falsify: remove the ``isinstance(ex_rec, Fact) and
        ex_rec.superseded_by != record.superseded_by`` guard from either
        adapter's ``upsert`` — this test goes red on that adapter with the
        pointer silently overwritten.
        """
        a_id = new_id()
        b_id = new_id()

        f = Fact.model_validate(
            {
                **_fact(content="fact superseded_by upsert test").model_dump(mode="python"),
                "validity": "invalid",
                "superseded_by": a_id,
            }
        )
        await adapter.upsert(f, _vec())
        retrieved = await adapter.get_by_id(f.id)
        assert isinstance(retrieved, Fact)
        assert retrieved.superseded_by == a_id

        mutated = Fact.model_validate(
            {**f.model_dump(mode="python"), "superseded_by": b_id}
        )
        with pytest.raises(ValueError, match="superseded_by"):
            await adapter.upsert(mutated, _vec(seed=0.6))

        # Untouched by the rejected attempt.
        after = await adapter.get_by_id(f.id)
        assert isinstance(after, Fact)
        assert after.superseded_by == a_id

    async def test_two_violation_upsert_reports_validity_before_lineage(
        self, adapter: AnyAdapter
    ) -> None:
        """A Fact upsert that changes both validity and superseded_by reports the validity
        violation on both adapters: lineage checks run after validity in the shared gate."""
        a_id = new_id()
        b_id = new_id()
        stored = Fact.model_validate(
            {
                **_fact(content="two violations").model_dump(mode="python"),
                "validity": "invalid",
                "superseded_by": a_id,
            }
        )
        await adapter.upsert(stored, _vec())

        mutated = stored.model_copy(update={"validity": "valid", "superseded_by": b_id})
        with pytest.raises(ValueError, match="Cannot change validity via upsert"):
            await adapter.upsert(mutated, _vec(seed=0.6))

        after = await adapter.get_by_id(stored.id)
        assert isinstance(after, Fact)
        assert after.validity == "invalid"
        assert after.superseded_by == a_id

    async def test_two_violation_upsert_reports_projection_before_decision_lineage(
        self, adapter: AnyAdapter
    ) -> None:
        """A Decision upsert that changes recall_count and promoted_from reports recall_count."""
        stored = _decision(content="two violations")
        await adapter.upsert(stored, _vec())

        snapshot = PromotedFromSnapshot(
            id=new_id(), content="x", valence=None, recurrence_count=1
        )
        mutated = stored.model_copy(update={"recall_count": 3, "promoted_from": (snapshot,)})
        with pytest.raises(ValueError, match="Cannot change recall_count via upsert"):
            await adapter.upsert(mutated, _vec(seed=0.6))

    async def test_supersession_chain_change_via_upsert_raises(
        self, adapter: AnyAdapter
    ) -> None:
        """supersession_chain is birth-frozen; re-upsert with a changed chain raises (F6).

        The engine upserts the new record with the full chain already attached in the
        initial write.  A subsequent re-upsert with a different chain must be rejected.
        """
        d = _decision(content="decision for chain test")
        await adapter.upsert(d, _vec())

        # Engine builds the new record WITH the chain already set, then upserts
        snap = SupersedesSnapshot(
            id=d.id,
            title="decision for chain test",
            created_at=d.created_at,
        )
        d2 = Decision(
            id=new_id(),
            type="decision",
            content="superseder",
            scope=Scope("proj-a"),
            lifecycle="in_force",
            supersession_chain=(snap,),  # chain set at birth
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        await adapter.upsert(d2, _vec(seed=0.9))  # must succeed (new record)

        # Now try to re-upsert d2 with a different chain — must raise
        other_snap = SupersedesSnapshot(
            id=d.id,
            title="different title",
            created_at=d.created_at,
        )
        d2_mutated = Decision(
            id=d2.id,
            type="decision",
            content=d2.content,
            scope=d2.scope,
            lifecycle=d2.lifecycle,
            supersession_chain=(other_snap,),  # changed — must be rejected
            created_at=d2.created_at,
            updated_at=d2.updated_at,
        )
        with pytest.raises(ValueError):
            await adapter.upsert(d2_mutated, _vec(seed=0.7))

    async def test_promoted_from_change_via_upsert_raises(
        self, adapter: AnyAdapter
    ) -> None:
        """promoted_from is birth-frozen; re-upsert with a changed list raises (F6)."""
        ember_id = new_id()
        snap = PromotedFromSnapshot(
            id=ember_id,
            content="ember content",
            valence="guideline",
            recurrence_count=3,
        )
        d = Decision(
            id=new_id(),
            type="decision",
            content="promoted decision",
            scope=Scope("proj-a"),
            lifecycle="in_force",
            promoted_from=(snap,),
            created_at=EPOCH,
            updated_at=EPOCH,
        )
        await adapter.upsert(d, _vec())

        # Re-upsert with empty promoted_from — must raise
        d_mutated = Decision(
            id=d.id,
            type="decision",
            content=d.content,
            scope=d.scope,
            lifecycle="in_force",
            promoted_from=(),  # changed
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        with pytest.raises(ValueError):
            await adapter.upsert(d_mutated, _vec(seed=0.8))


# ===========================================================================
# 6 — Supersession atomicity + set-exactly-once
# ===========================================================================


class TestSupersessionAtomicity:
    """``replace`` is the ONLY path that draws a forward supersession edge;
    ``transition`` remains the only OTHER lifecycle mutator; superseded_by
    is set exactly once either way."""

    async def test_replace_sets_superseded_by(self, adapter: AnyAdapter) -> None:
        old_d = _decision(content="Old decision")
        new_d = _decision(content="New decision")
        v = _vec()
        await adapter.upsert(old_d, v)
        await adapter.upsert(new_d, _vec(seed=0.9))

        # Supersede old_d → new_d, in one commit.
        await adapter.replace(new_d, None, [_flip(old_d.id)], at=EPOCH)

        old_retrieved = await adapter.get_by_id(old_d.id)
        assert isinstance(old_retrieved, Decision)
        assert old_retrieved.lifecycle == "superseded"
        assert old_retrieved.superseded_by == new_d.id

    async def test_superseded_by_set_exactly_once(self, adapter: AnyAdapter) -> None:
        d1 = _decision(content="Decision one")
        d2 = _decision(content="Decision two")
        d3 = _decision(content="Decision three")
        v = _vec()
        await adapter.upsert(d1, v)
        await adapter.upsert(d2, _vec(seed=0.9))
        await adapter.upsert(d3, _vec(seed=0.8))

        await adapter.replace(d2, None, [_flip(d1.id)], at=EPOCH)

        # Second supersession attempt on the same old id must raise
        with pytest.raises((ValueError, RuntimeError)):
            await adapter.replace(d3, None, [_flip(d1.id)], at=EPOCH)

    async def test_transition_is_only_other_lifecycle_mutator(self, adapter: AnyAdapter) -> None:
        """transition changes lifecycle; upsert must not (see TestLifecycleUpsertRejection)."""
        f = _fact(content="active fact", lifecycle="active")
        await adapter.upsert(f, _vec())

        await adapter.transition(f.id, "archived", at=EPOCH)
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.lifecycle == "archived"

    async def test_transition_unknown_id_raises(self, adapter: AnyAdapter) -> None:
        unknown = new_id()
        with pytest.raises(KeyError):
            await adapter.transition(unknown, "archived", at=EPOCH)

    async def test_transition_unsupported_keyword_raises(self, adapter: AnyAdapter) -> None:
        """An unsupported keyword to transition is a static error (mypy), and
        with the ``**kw`` bag gone it is now also a TypeError at call time —
        no more silent no-op. mypy catches this earlier still; this is the
        runtime backstop the suite can assert.
        """
        f = _fact(content="active fact", lifecycle="active")
        await adapter.upsert(f, _vec())

        with pytest.raises(TypeError):
            await adapter.transition(  # type: ignore[call-arg]
                f.id, "hot", at=EPOCH, bogus_field=1
            )

    async def test_replace_self_reference_raises(self, adapter: AnyAdapter) -> None:
        """A Decision cannot supersede itself (F8 — referential integrity)."""
        d = _decision(content="self-supersession attempt")
        await adapter.upsert(d, _vec())
        with pytest.raises(ValueError, match=r"itself|self"):
            await adapter.replace(d, None, [_flip(d.id)], at=EPOCH)

    async def test_replace_unknown_predecessor_raises(self, adapter: AnyAdapter) -> None:
        """A flip naming an unknown predecessor id refuses the whole call."""
        d = _decision(content="supersede from nowhere")
        await adapter.upsert(d, _vec())
        ghost_id = new_id()
        with pytest.raises(ValueError, match=r"does not exist|not found"):
            await adapter.replace(d, None, [_flip(ghost_id)], at=EPOCH)

    async def test_replace_cross_arm_predecessor_raises(
        self, adapter: AnyAdapter
    ) -> None:
        """A Decision successor with a Fact predecessor raises (F8) — both
        sides of one edge must be the same arm."""
        d = _decision(content="wants to supersede a Fact")
        f = _fact(content="I am a Fact, not a Decision")
        await adapter.upsert(d, _vec())
        await adapter.upsert(f, _vec(seed=0.9))
        with pytest.raises(ValueError, match=r"not a Decision|Decision"):
            await adapter.replace(d, None, [_flip(f.id)], at=EPOCH)


class TestReplaceAtomicity:
    """The full P3 ``replace`` contract: one
    ``merge_insert`` commits the successor row plus every predecessor's
    flip — validity/lifecycle, ``superseded_by``, ``supersession_reason``,
    ``supersession_note``, ``supersession_edge_id`` — or none of it."""

    async def test_fact_successor_two_predecessors_land_together(
        self, adapter: AnyAdapter
    ) -> None:
        p1 = _fact(content="predecessor one")
        p2 = _fact(content="predecessor two")
        await adapter.upsert(p1, _vec())
        await adapter.upsert(p2, _vec(seed=0.9))

        successor = _fact(content="successor replacing both")
        flips = [
            _flip(p1.id, reason="correction", note="p1 was wrong"),
            _flip(p2.id, reason="changed", note=None),
        ]
        await adapter.replace(successor, _vec(seed=0.8), flips, at=EPOCH)

        landed_successor = await adapter.get_by_id(successor.id)
        assert isinstance(landed_successor, Fact)

        landed_p1 = await adapter.get_by_id(p1.id)
        landed_p2 = await adapter.get_by_id(p2.id)
        assert isinstance(landed_p1, Fact)
        assert isinstance(landed_p2, Fact)
        for landed, flip in ((landed_p1, flips[0]), (landed_p2, flips[1])):
            assert landed.validity == "invalid"
            assert landed.superseded_by == successor.id
            assert landed.supersession_reason == flip.reason
            assert landed.supersession_note == flip.note
            assert landed.supersession_edge_id == flip.edge_id

    async def test_stale_predecessor_refuses_whole_call_and_leaves_successor_absent(
        self, adapter: AnyAdapter
    ) -> None:
        healthy = _fact(content="healthy predecessor")
        stale = _fact(content="predecessor already superseded by someone else")
        earlier_successor = _fact(content="earlier successor already retired stale")
        await adapter.upsert(healthy, _vec())
        await adapter.upsert(stale, _vec(seed=0.9))
        await adapter.upsert(earlier_successor, _vec(seed=0.8))
        await adapter.replace(earlier_successor, None, [_flip(stale.id)], at=EPOCH)

        successor = _fact(content="new successor naming one dead edge")
        with pytest.raises(ValueError, match=str(stale.id)):
            await adapter.replace(
                successor, _vec(seed=0.7),
                [_flip(healthy.id), _flip(stale.id)],
                at=EPOCH,
            )

        with pytest.raises(KeyError):
            await adapter.get_by_id(successor.id)
        healthy_after = await adapter.get_by_id(healthy.id)
        assert isinstance(healthy_after, Fact)
        assert healthy_after.validity == "valid"
        assert healthy_after.superseded_by is None

    async def test_retired_existing_successor_refuses(self, adapter: AnyAdapter) -> None:
        pred = _fact(content="predecessor targeting a retired successor")
        old_successor = _fact(content="the successor that will itself be retired")
        newer = _fact(content="whatever retired old_successor")
        await adapter.upsert(pred, _vec())
        await adapter.upsert(old_successor, _vec(seed=0.9))
        await adapter.upsert(newer, _vec(seed=0.8))
        await adapter.replace(newer, None, [_flip(old_successor.id)], at=EPOCH)

        with pytest.raises(ValueError, match=str(old_successor.id)):
            await adapter.replace(old_successor, None, [_flip(pred.id)], at=EPOCH)

        pred_after = await adapter.get_by_id(pred.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "valid"
        assert pred_after.superseded_by is None

    async def test_absent_successor_without_vector_lands_vectorless(
        self, adapter: AnyAdapter
    ) -> None:
        """P6 vector-less-rows piece: a superseding write
        during an embedder outage must not fail — ``replace`` lands the
        brand-new successor with ``vector=None``/``embedding_model=None``,
        the same vector-less landing an ordinary outage save gets, and
        still flips the predecessor atomically."""
        pred = _fact(content="predecessor for the vectorless-successor check")
        await adapter.upsert(pred, _vec())
        successor = _fact(content="brand-new successor with no vector supplied")
        await adapter.replace(successor, None, [_flip(pred.id)], at=EPOCH)

        landed = await adapter.get_by_id(successor.id)
        assert isinstance(landed, Fact)
        assert landed.embedding_model is None
        assert await _raw_vector(adapter, successor.id) is None

        pred_after = await adapter.get_by_id(pred.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.superseded_by == successor.id
        assert pred_after.validity == "invalid"

    async def test_existing_vectorless_successor_is_reused_without_a_vector(
        self, adapter: AnyAdapter
    ) -> None:
        """Second half of the above: ``replace`` must also accept an
        EXISTING vector-less successor row (one that already landed
        vector-less under an earlier outage and has not been backfilled
        yet) as its target, reusing the row's own persisted (null) vector
        state rather than refusing it."""
        pred = _fact(content="predecessor for the existing-vectorless-successor check")
        await adapter.upsert(pred, _vec())
        successor = _fact(content="successor, already vector-less")
        await adapter.upsert(successor, None)

        await adapter.replace(successor, None, [_flip(pred.id)], at=EPOCH)

        landed = await adapter.get_by_id(successor.id)
        assert isinstance(landed, Fact)
        assert landed.embedding_model is None
        assert await _raw_vector(adapter, successor.id) is None

        pred_after = await adapter.get_by_id(pred.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.superseded_by == successor.id

    async def test_reused_successor_keeps_its_own_embedding_model_stamp(
        self, adapter: AnyAdapter, clock: TestClock, tmp_path: Any
    ) -> None:
        """``replace`` never re-embeds ANY row it retains a vector
        for — the ``vector=None`` successor (the explicit ``supersede``
        call against an already-existing successor) and every predecessor
        alike. Each retained row must keep ITS OWN stored
        ``embedding_model``, never relabeled as the CURRENTLY configured
        model. Getting this wrong hides a genuinely stale vector from
        ``backfill_targets`` after a model migration.

        Falsify: stamp a retained vector with the adapter's own configured
        ``embedding_model`` instead of the row's original one (either side)
        — this test goes red because that row drops out of
        ``backfill_targets`` after the ``replace`` call despite its vector
        never having been re-embedded.
        """
        pred = _fact(content="predecessor for the embedding-model-stamp test")
        successor = _fact(content="successor for the embedding-model-stamp test")
        await adapter.upsert(pred, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        adapter_v2 = await _reopen_at_model(
            adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock
        )
        targets_before = {t.id for t in await adapter_v2.backfill_targets()}
        assert successor.id in targets_before
        assert pred.id in targets_before

        await adapter_v2.replace(successor, None, [_flip(pred.id)], at=EPOCH)

        targets_after = {t.id for t in await adapter_v2.backfill_targets()}
        assert successor.id in targets_after, (
            "the successor's retained (never re-embedded) vector must stay a "
            "backfill target after replace — its embedding_model stamp must "
            "not have been silently relabeled"
        )
        assert pred.id in targets_after, (
            "the predecessor's retained (never re-embedded) vector must stay a "
            "backfill target after replace — its embedding_model stamp must "
            "not have been silently relabeled"
        )

    async def test_decision_arm_one_flip(self, adapter: AnyAdapter) -> None:
        pred = _decision(content="predecessor decision")
        successor = _decision(content="successor decision")
        await adapter.upsert(pred, _vec())
        await adapter.replace(
            successor, _vec(seed=0.9), [_flip(pred.id, reason="moved")], at=EPOCH,
        )

        landed_successor = await adapter.get_by_id(successor.id)
        assert isinstance(landed_successor, Decision)
        landed_pred = await adapter.get_by_id(pred.id)
        assert isinstance(landed_pred, Decision)
        assert landed_pred.lifecycle == "superseded"
        assert landed_pred.superseded_by == successor.id
        assert landed_pred.supersession_reason == "moved"


# ===========================================================================
# 5c — Fact-invalidates-Decision transaction and revalidation (P8
# assessor-automatic-exits piece, decisions D13/D14)
# ===========================================================================


class TestInvalidation:
    """A Fact successor invalidating one or more Decisions lands in the SAME
    ``replace`` transaction as the successor row (D13); ``revalidate`` clears it
    (D14)."""

    async def test_replace_with_invalidation_sets_both_columns(
        self, adapter: AnyAdapter
    ) -> None:
        target = _decision(content="a decision a Fact will invalidate")
        await adapter.upsert(target, _vec())

        successor = _fact(content="the Fact that invalidates it")
        await adapter.replace(
            successor, _vec(seed=0.9), [],
            invalidations=[DecisionInvalidation(id=target.id, reason="correction")],
            at=EPOCH,
        )

        landed_target = await adapter.get_by_id(target.id)
        assert isinstance(landed_target, Decision)
        assert landed_target.validity == "invalid"
        assert landed_target.invalidated_by == str(successor.id)
        assert landed_target.invalidation_reason == "correction"
        assert landed_target.lifecycle == "in_force"
        assert landed_target.superseded_by is None

        landed_successor = await adapter.get_by_id(successor.id)
        assert isinstance(landed_successor, Fact)

    async def test_replace_zero_flips_one_invalidation_is_legal(
        self, adapter: AnyAdapter
    ) -> None:
        """P8 Uncertainty 1, settled: ``flips=()`` with exactly one invalidation is
        accepted by both adapters — the guard that used to require at least one
        flip is relaxed to "at least one flip OR invalidation"."""
        target = _decision(content="invalidated with no flips alongside it")
        await adapter.upsert(target, _vec())
        successor = _fact(content="lone invalidator")

        await adapter.replace(
            successor, _vec(seed=0.9), [],
            invalidations=[DecisionInvalidation(id=target.id, reason="changed")],
            at=EPOCH,
        )

        landed = await adapter.get_by_id(target.id)
        assert isinstance(landed, Decision)
        assert landed.validity == "invalid"

    async def test_replace_neither_flips_nor_invalidations_raises(
        self, adapter: AnyAdapter
    ) -> None:
        successor = _fact(content="nothing to flip or invalidate")
        with pytest.raises(ValueError):
            await adapter.replace(successor, _vec(), [], at=EPOCH)

    async def test_invalidation_preserves_every_other_column(
        self, adapter: AnyAdapter
    ) -> None:
        """P8 Uncertainty 2, settled: the ``_records_schema``/row round-trip
        preserves a Decision's full column set through the batch — read back
        every field the fixture set and confirm none of them moved."""
        target = _decision(
            content="content that must survive byte-for-byte",
            scope="proj-b",
            source="agent-x",
            purpose="constraint",
            audience_hint=["builder"],
        )
        await adapter.upsert(target, _vec())
        successor = _fact(content="invalidator")

        await adapter.replace(
            successor, _vec(seed=0.9), [],
            invalidations=[DecisionInvalidation(id=target.id, reason="correction")],
            at=EPOCH,
        )

        landed = await adapter.get_by_id(target.id)
        assert isinstance(landed, Decision)
        assert landed.content == target.content
        assert landed.scope == target.scope
        assert landed.source == target.source
        assert landed.purpose == target.purpose
        assert landed.audience_hint == target.audience_hint
        assert landed.created_at == target.created_at
        assert landed.supersession_chain == target.supersession_chain
        assert landed.promoted_from == target.promoted_from

    async def test_invalidation_target_must_be_decision(self, adapter: AnyAdapter) -> None:
        target = _fact(content="a Fact, not a Decision")
        await adapter.upsert(target, _vec())
        successor = _fact(content="wants to invalidate a Fact")
        with pytest.raises(ValueError, match=r"Decision"):
            await adapter.replace(
                successor, _vec(seed=0.9), [],
                invalidations=[DecisionInvalidation(id=target.id, reason="correction")],
                at=EPOCH,
            )

    async def test_invalidation_successor_must_be_fact(self, adapter: AnyAdapter) -> None:
        target = _decision(content="target decision")
        await adapter.upsert(target, _vec())
        successor = _decision(content="a Decision successor cannot invalidate")
        with pytest.raises(ValueError, match=r"Fact"):
            await adapter.replace(
                successor, _vec(seed=0.9), [],
                invalidations=[DecisionInvalidation(id=target.id, reason="correction")],
                at=EPOCH,
            )

    async def test_invalidation_target_not_in_force_raises(self, adapter: AnyAdapter) -> None:
        pred = _decision(content="will be superseded first")
        other_successor = _decision(content="supersedes pred")
        await adapter.upsert(pred, _vec())
        await adapter.replace(other_successor, _vec(seed=0.9), [_flip(pred.id)], at=EPOCH)

        successor = _fact(content="too late to invalidate a superseded Decision")
        with pytest.raises(ValueError, match=r"in_force"):
            await adapter.replace(
                successor, _vec(seed=0.8), [],
                invalidations=[DecisionInvalidation(id=pred.id, reason="correction")],
                at=EPOCH,
            )

    async def test_invalidation_target_overlapping_a_flip_raises(
        self, adapter: AnyAdapter
    ) -> None:
        shared = _decision(content="named as both a flip predecessor and a target")
        await adapter.upsert(shared, _vec())
        successor = _decision(content="successor Decision")
        with pytest.raises(ValueError):
            await adapter.replace(
                successor, _vec(seed=0.9),
                [_flip(shared.id)],
                invalidations=[DecisionInvalidation(id=shared.id, reason="correction")],
                at=EPOCH,
            )

    async def test_fault_on_replace_leaves_decision_valid_and_no_fact(
        self, adapter: AnyAdapter
    ) -> None:
        """A single ``merge_insert`` has no application-level midpoint — the
        legality check runs before any write, so a caller-side failure after a
        legal invalidation is proven by the earlier unknown/wrong-kind cases
        above; this confirms the illegal-target case ALSO leaves the target
        untouched and mints no Fact."""
        target = _fact(content="not a Decision")
        await adapter.upsert(target, _vec())
        successor = _fact(content="never lands")
        with pytest.raises(ValueError):
            await adapter.replace(
                successor, _vec(seed=0.9), [],
                invalidations=[DecisionInvalidation(id=target.id, reason="correction")],
                at=EPOCH,
            )
        with pytest.raises(KeyError):
            await adapter.get_by_id(successor.id)

    async def test_revalidate_clears_columns_and_writes_event(
        self, adapter: AnyAdapter
    ) -> None:
        target = _decision(content="will be invalidated then revalidated")
        await adapter.upsert(target, _vec())
        successor = _fact(content="invalidator")
        await adapter.replace(
            successor, _vec(seed=0.9), [],
            invalidations=[DecisionInvalidation(id=target.id, reason="changed")],
            at=EPOCH,
        )

        event = await adapter.revalidate(target.id, "reviewed", at=EPOCH)
        assert event is not None
        assert event.decision_id == target.id
        assert event.withdrawn_cause == successor.id
        assert event.withdrawn_reason == "changed"
        assert event.reason == "reviewed"

        landed = await adapter.get_by_id(target.id)
        assert isinstance(landed, Decision)
        assert landed.validity == "valid"
        assert landed.invalidated_by is None
        assert landed.invalidation_reason is None

        events = await adapter.list_revalidations(target.id)
        assert len(events) == 1
        assert events[0].id == event.id

    async def test_revalidate_returns_none_when_not_invalidated(
        self, adapter: AnyAdapter
    ) -> None:
        target = _decision(content="never invalidated")
        await adapter.upsert(target, _vec())

        event = await adapter.revalidate(target.id, "reviewed", at=EPOCH)

        assert event is None
        landed = await adapter.get_by_id(target.id)
        assert isinstance(landed, Decision)
        assert landed.validity == "valid"

    async def test_revalidate_unknown_id_raises(self, adapter: AnyAdapter) -> None:
        with pytest.raises(KeyError):
            await adapter.revalidate(new_id(), "reviewed", at=EPOCH)


# ===========================================================================
# 5b — Fact.superseded_by round-trips through both adapters
# (2026-08-10 fact-supersession brief, Scope 1). Falsifiability: revert the
# ``row_to_record`` Fact-branch read alone and watch the LanceDB leg fail
# while the memory leg still passes (MemoryAdapter holds the frozen pydantic
# object directly and needs no storage change).
# ===========================================================================


class TestFactSupersededByRoundTrip:
    async def test_fact_superseded_by_round_trips(self, adapter: AnyAdapter) -> None:
        predecessor = _fact(content="predecessor fact", validity="invalid")
        successor_id = new_id()
        predecessor = Fact.model_validate(
            {**predecessor.model_dump(mode="python"), "superseded_by": successor_id}
        )
        await adapter.upsert(predecessor, _vec())

        retrieved = await adapter.get_by_id(predecessor.id)
        assert isinstance(retrieved, Fact)
        assert retrieved.superseded_by == successor_id

    async def test_fact_superseded_by_defaults_null(self, adapter: AnyAdapter) -> None:
        f = _fact(content="ordinary fact, never superseded")
        await adapter.upsert(f, _vec())

        retrieved = await adapter.get_by_id(f.id)
        assert isinstance(retrieved, Fact)
        assert retrieved.superseded_by is None


# ===========================================================================
# 5c — Fact-arm supersede edge concurrency (2026-08-10 fact-supersession
# brief, Scope 2). Falsifiability: remove the fresh re-read under
# ``self._supersede_lock``/the CAS-recheck-after-yield and watch this test
# fail with two writes landing (both calls succeed instead of one raising).
# ===========================================================================


class TestFactSupersedeConcurrency:
    async def test_two_concurrent_supersessions_of_same_predecessor(
        self, adapter: AnyAdapter
    ) -> None:
        predecessor = _fact(content="predecessor fact, targeted twice")
        successor_a = _fact(content="successor A")
        successor_b = _fact(content="successor B")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor_a, _vec(seed=0.9))
        await adapter.upsert(successor_b, _vec(seed=0.8))

        results = await asyncio.gather(
            adapter.replace(successor_a, None, [_flip(predecessor.id)], at=EPOCH),
            adapter.replace(successor_b, None, [_flip(predecessor.id)], at=EPOCH),
            return_exceptions=True,
        )

        successes = [r for r in results if not isinstance(r, BaseException)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1, results
        assert len(failures) == 1, results
        assert isinstance(failures[0], ValueError), failures[0]

        final = await adapter.get_by_id(predecessor.id)
        assert isinstance(final, Fact)
        assert final.validity == "invalid"
        assert final.superseded_by in (successor_a.id, successor_b.id)
        # The winner's pointer is intact — exactly one of the two ids, never
        # both, never neither (torn state).

    async def test_supersede_after_bury_raises(self, adapter: AnyAdapter) -> None:
        predecessor = _fact(content="buried before the race resolves")
        successor = _fact(content="would-be successor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        await adapter.bury(predecessor.id, reason="race test")

        with pytest.raises(ValueError, match="does not exist or is buried"):
            await adapter.replace(successor, None, [_flip(predecessor.id)], at=EPOCH)


# ===========================================================================
# 5d — Ordinary Fact transition vs a concurrent Fact-supersede
# (fact-supersession-fixes review, defect 1). This is an INTERLEAVED
# variant of 5c — the sequential case (supersede-then-shelve, or
# shelve-then-supersede, run as two separate awaits) never exercised the
# bug: it only appears when the ordinary transition's read happens BEFORE
# the supersede's write lands, and its write happens AFTER.
# ===========================================================================


class TestOrdinaryFactTransitionSupersedeRace:
    """``transition()``'s "ordinary" branch (``set_hot``/``set_audience``/
    ``set_shelved`` — exercised here directly via ``transition(id,
    "shelved", at=...)``, the shape that matters, not which disposition
    calls it) racing a Fact-arm supersede of the SAME predecessor must not
    silently lose either write (fact-supersession-fixes review, defect 1):
    before the fix, this branch rebuilt the row from a snapshot read before
    ``transition()``'s own first ``await`` — no lock (``LanceDBAdapter``),
    no post-yield re-fetch (``MemoryAdapter``) — unlike the Fact-supersede
    branch beside it, which already re-reads fresh. A concurrent supersede
    landing in that window got silently erased by the ordinary write's
    stale rebuild.

    Real interleaving via two ``asyncio.create_task`` calls, the supersede
    created FIRST. On ``MemoryAdapter`` this is deterministic: both
    branches share ONE cooperative yield point per call
    (``await asyncio.sleep(0)``), so creating the supersede task first
    gives its post-yield write priority in the FIFO ready queue — it lands
    BEFORE the ordinary (shelve) task even resumes from its own yield,
    so the ordinary write's pre-fix stale rebuild (captured before EITHER
    write landed) clobbers the just-committed pointer. On
    ``LanceDBAdapter`` the same creation order exercises the equivalent
    real-I/O race under (or, pre-fix, WITHOUT) ``self._supersede_lock``.

    Falsify: revert either adapter's ordinary-Fact-transition branch to the
    pre-fix shape (rebuild from the top-of-method snapshot, no lock /
    re-read) — the assertion on ``final.superseded_by`` goes red.
    """

    async def test_ordinary_transition_does_not_erase_concurrent_supersede(
        self, adapter: AnyAdapter
    ) -> None:
        predecessor = _fact(content="raced: shelved vs superseded, defect 1")
        successor = _fact(content="successor of the raced-shelve predecessor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        # Created FIRST: gives the supersede's write priority over the
        # ordinary (shelve) transition, created second (see class
        # docstring).
        supersede_task = asyncio.create_task(
            adapter.replace(successor, None, [_flip(predecessor.id)], at=EPOCH)
        )
        shelve_task = asyncio.create_task(
            adapter.transition(predecessor.id, "shelved", at=EPOCH)
        )

        results = await asyncio.wait_for(
            asyncio.gather(supersede_task, shelve_task, return_exceptions=True),
            timeout=5.0,
        )
        supersede_result, shelve_result = results
        assert not isinstance(supersede_result, BaseException), (
            f"the supersede must not itself fail in this race: {supersede_result!r}"
        )
        assert not isinstance(shelve_result, BaseException), (
            f"the ordinary transition must not itself fail in this race: {shelve_result!r}"
        )

        final = await adapter.get_by_id(predecessor.id)
        assert isinstance(final, Fact)
        assert final.superseded_by == successor.id, (
            "the concurrent supersede's pointer must not be silently erased "
            "by the ordinary transition's write rebuilding from a stale "
            "pre-race snapshot"
        )
        assert final.validity == "invalid"
        assert final.lifecycle == "shelved", (
            "the ordinary transition's own write must still land — this "
            "fix composes both writes, it does not make the ordinary one "
            "lose instead"
        )


# ===========================================================================
# 5e — Fact bury vs a concurrent Fact-supersede of the SAME predecessor
# (fact-supersession-fixes review, defect 2). An INTERLEAVED variant of
# ``test_supersede_after_bury_raises_keyerror`` above, which only covers the
# sequential case (bury fully completes, THEN the supersede is attempted) —
# the phantom-success shape needs the bury's delete to land INSIDE the
# supersede's own read-then-write window.
# ===========================================================================


class TestFactBuryConcurrency:
    """A Fact bury racing a ``replace`` of the SAME predecessor must never
    produce a phantom success — ``replace`` returning cleanly while nothing
    it wrote actually survives.

    Pre-P3 (fact-supersession-fixes review, defect 2), the forward edge's
    own fresh CAS re-read and its write were two separate steps inside the
    lock, with a cheap PRE-lock check before either — the phantom-success
    shape came from that split: a fresh re-read seeing the row still
    present, then a write issued after bury's ``delete()`` landed in the
    gap, silently matching nothing. P3's ``replace`` closes this by
    construction, not by timing: every read AND the write happen inside
    ONE ``self._supersede_lock`` hold (LanceDB) / one yield-then-write
    stretch (Memory), so ``bury()`` (which takes the SAME lock) and
    ``replace()`` can never interleave — whichever acquires the lock first
    runs to completion before the other's reads even happen. There is
    consequently no reliable "created first wins" ordering to pin (unlike
    the pre-P3 shape, where the supersede's longer pre-lock pipeline gave
    bury's shorter one a predictable head start) — this test asserts the
    INVARIANT that holds regardless of which task wins the race to acquire
    the lock: the predecessor ends up buried either way (bury has no CAS
    of its own), and ``replace`` either lands cleanly (it won the lock
    first) or fails with a clean ``ValueError`` naming the predecessor
    (bury won first) — never anything else, never a torn state.

    Falsify: split ``replace``'s fresh reads from its write with an
    ``await`` in between (reintroducing the pre-P3 shape) — this test
    would then need re-tuning to pin a winner, and a naive assertion that
    ``replace`` always fails here would go red once bury reliably won
    again.
    """

    async def test_bury_interleaved_with_supersede_never_torn(
        self, adapter: AnyAdapter
    ) -> None:
        predecessor = _fact(content="raced: buried vs superseded, defect 2")
        successor = _fact(content="successor of the raced-bury predecessor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        bury_task = asyncio.create_task(
            adapter.bury(predecessor.id, reason="races a concurrent supersede")
        )
        supersede_task = asyncio.create_task(
            adapter.replace(successor, None, [_flip(predecessor.id)], at=EPOCH)
        )

        results = await asyncio.wait_for(
            asyncio.gather(bury_task, supersede_task, return_exceptions=True),
            timeout=5.0,
        )
        bury_result, supersede_result = results

        assert not isinstance(bury_result, BaseException), (
            f"bury() must not itself fail in this race: {bury_result!r}"
        )
        if isinstance(supersede_result, BaseException):
            assert isinstance(supersede_result, ValueError), (
                "a losing replace must fail cleanly with ValueError, never "
                f"{type(supersede_result).__name__}: {supersede_result!r}"
            )

        # Bury has no CAS of its own — the predecessor is gone either way,
        # whether replace won the lock first (and bury's unconditional
        # delete then removed the now-superseded row) or lost it.
        with pytest.raises(KeyError):
            await adapter.get_by_id(predecessor.id)


# ===========================================================================
# 5f — Fact-arm supersede vs a concurrent bury of the SUCCESSOR
# (dangling successor pointer).
# Every prior round on this seam re-validated one MORE referent inside the
# guarding lock/yield window — first the predecessor, then a concurrent
# bury of the predecessor — but never the successor: ``transition(...,
# superseded_by=succ)`` validated that ``succ`` existed and was a Fact
# BEFORE acquiring the guard, then never looked at it again. A concurrent
# ``bury(succ)`` landing in that window let the predecessor's write land
# anyway, with a pointer at a record that no longer exists.
# ===========================================================================


class TestFactSupersedeDanglingSuccessor:
    """A ``replace`` racing a concurrent ``bury()`` of its OWN successor
    (not the predecessor — see ``TestFactBuryConcurrency`` above for that
    case) must never land a pointer at a buried record.

    Same mutual-exclusion argument as ``TestFactBuryConcurrency``:
    ``replace``'s successor read and its write happen inside ONE lock hold
    shared with ``bury()``, so there is no reliable "created first wins"
    ordering to pin here either — this test asserts the invariant that
    holds regardless of which task wins the race to acquire the lock:
    either ``replace`` wins first (the successor was genuinely still live
    at that instant, so the edge is legitimately drawn, and the
    now-superseded successor can still be buried afterward — ``bury`` has
    no CAS of its own) or ``bury`` wins first (``replace``'s fresh read
    correctly observes the successor gone and refuses the whole call,
    leaving the predecessor untouched) — never a dangling pointer at a
    record ``get_by_id`` can no longer find.

    Falsify: read the successor ONCE, before acquiring the lock, and trust
    that read inside it (reintroducing the pre-P3 shape) — a bury landing
    in the gap would let the predecessor's write go through pointing at a
    record that no longer exists, regardless of which task the scheduler
    happens to run first.
    """

    async def test_bury_of_successor_never_lands_a_dangling_pointer(
        self, adapter: AnyAdapter
    ) -> None:
        predecessor = _fact(content="predecessor, successor buried mid-race")
        successor = _fact(content="successor, buried out from under the edge")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        bury_task = asyncio.create_task(
            adapter.bury(successor.id, reason="races a concurrent supersede targeting it")
        )
        supersede_task = asyncio.create_task(
            adapter.replace(successor, None, [_flip(predecessor.id)], at=EPOCH)
        )

        results = await asyncio.wait_for(
            asyncio.gather(bury_task, supersede_task, return_exceptions=True),
            timeout=5.0,
        )
        bury_result, supersede_result = results

        assert not isinstance(bury_result, BaseException), (
            f"burying the successor must not itself fail in this race: {bury_result!r}"
        )
        with pytest.raises(KeyError):
            await adapter.get_by_id(successor.id)

        final_predecessor = await adapter.get_by_id(predecessor.id)
        assert isinstance(final_predecessor, Fact)
        if isinstance(supersede_result, BaseException):
            assert isinstance(supersede_result, ValueError), (
                "a losing replace must fail cleanly with ValueError — never "
                f"succeed against a successor that no longer exists; got {supersede_result!r}"
            )
            assert "lost a concurrent race" in str(supersede_result), (
                f"expected the fresh in-lock wording, got: {supersede_result!r}"
            )
            assert final_predecessor.superseded_by is None, (
                "a lost race must never leave the predecessor pointing at a "
                "record that no longer exists"
            )
            assert final_predecessor.validity == "valid"
        else:
            # replace won the lock first: the edge was legitimate at that
            # instant, and bury's own unconditional delete still landed
            # afterward on the now-superseded successor.
            assert final_predecessor.superseded_by == successor.id
            assert final_predecessor.validity == "invalid"


class TestFactSupersedeAlreadySupersededSuccessor:
    """A direct adapter-level supersede targeting an already-superseded
    successor must raise, not silently draw a second edge onto a record
    that has itself already been retired.

    Sequential, not raced — the engine layer
    (``engine.dispositions.supersede``) already refuses this before
    ever calling the adapter, but that check runs BEFORE a caller reaches
    ``self._supersede_lock``/the yield point, so it protects only callers
    that go through the engine. A caller that talks to the adapter
    directly (bypassing the engine, same category ``TestFactBuryConcurrency``
    and ``TestFactSupersedeDanglingSuccessor`` above already exercise) has
    nothing else standing between it and drawing an illegal edge, so the
    adapter's own guarded path — ``resolve_replace_successor`` — is where
    this precondition has to live too: the successor's own "not already
    superseded" state is exactly as much a referent this mutation asserts
    as its existence and its type, which is why this lands beside those
    checks rather than as a special case.

    Falsify: remove the ``existing.superseded_by is not None`` check from
    ``resolve_replace_successor`` — this test goes red, with ``b`` ending up
    superseded twice (once by ``c``, once by ``a``).
    """

    async def test_supersede_targeting_an_already_superseded_successor_raises(
        self, adapter: AnyAdapter
    ) -> None:
        a = _fact(content="a: wants to supersede an already-retired b")
        b = _fact(content="b: superseded by c already, before a's attempt")
        c = _fact(content="c: b's real successor")
        await adapter.upsert(a, _vec())
        await adapter.upsert(b, _vec(seed=0.9))
        await adapter.upsert(c, _vec(seed=0.8))

        # b -> c lands first, uncontested.
        await adapter.replace(c, None, [_flip(b.id)], at=EPOCH)

        # a -> b must now be refused: b is no longer a legal successor.
        with pytest.raises(ValueError, match="already superseded"):
            await adapter.replace(b, None, [_flip(a.id)], at=EPOCH)

        final_a = await adapter.get_by_id(a.id)
        assert isinstance(final_a, Fact)
        assert final_a.superseded_by is None
        assert final_a.validity == "valid"


# ===========================================================================
# 5g — Fact-arm supersede vs a concurrent ORDINARY lifecycle change on the
# SAME predecessor, shelve landing FIRST (adapter divergence on ``lifecycle``). This is the MIRROR
# ordering of ``TestOrdinaryFactTransitionSupersedeRace`` above (which races
# the supersede to land FIRST and only proves the two writes compose, not
# that neither adapter reverts the other's field): here the ordinary
# transition wins the race and lands first, and the supersede's own write
# must not silently revert it back to a stale snapshot.
#
# Written as ONE test that runs on both adapters via the SAME shared
# assertions (not two adapter-specific tests) — this is the shape the round
# 3b defect needed and did not have: a per-adapter test can pass on
# ``LanceDBAdapter`` (which never had the bug) while missing the identical
# divergence on ``MemoryAdapter`` entirely.
# ===========================================================================


class TestFactSupersedeLifecycleParity:
    """A Fact-arm ``replace`` racing a concurrent ORDINARY transition
    (``set_shelved``'s shape, exercised directly via ``transition(id,
    "shelved", at=...)``) on the SAME predecessor, with the ordinary
    transition's write landing FIRST, must leave BOTH the lifecycle change
    AND the supersede pointer intact — identically on both adapters.

    History (pre-P3, when the forward edge was drawn by ``transition``):
    ``MemoryAdapter`` used to seed one ``updates`` dict at the top of
    ``transition()`` — including ``"lifecycle"`` — before branching, and
    applied it verbatim even on the Fact-supersede branch, built from the
    caller's OWN pre-race snapshot. If the shelve landed first, that stale
    lifecycle would silently overwrite the freshly "shelved" one on
    ``MemoryAdapter``. ``resolve_replace_predecessor`` closes this by
    construction now: it builds the predecessor's flip fields
    (``superseded_by``, the reason columns, ``validity``) from a FRESH
    read taken under the lock/after the yield, and never mentions
    ``lifecycle`` for a Fact at all — there is no shared ``updates`` dict
    for a stale value to leak in from.

    Real interleaving via two ``asyncio.create_task`` calls, the ordinary
    (shelve) transition created FIRST — its pipeline has fewer awaits
    before reaching the shared guard (one top-of-method read vs.
    ``replace``'s successor-then-predecessor reads), so it reliably wins
    the race to land first on both adapters.

    Falsify: build the predecessor's ``replace`` write from a snapshot
    taken before the lock/yield instead of the fresh in-lock read — this
    test's ``final.lifecycle`` assertion goes red on the ``memory`` adapter
    parametrization specifically, while the ``lancedb`` parametrization
    keeps passing (proving this is a genuine cross-adapter divergence, not
    a shared bug).
    """

    async def test_shelve_then_supersede_preserves_shelved_on_both_adapters(
        self, adapter: AnyAdapter
    ) -> None:
        predecessor = _fact(content="raced: shelved THEN superseded")
        successor = _fact(content="successor of the shelve-first predecessor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        # Created FIRST: gives the ordinary (shelve) transition's shorter
        # pipeline priority over the supersede's longer one (see class
        # docstring) — this reliably lands "shelved" before the supersede's
        # write is built.
        shelve_task = asyncio.create_task(
            adapter.transition(predecessor.id, "shelved", at=EPOCH)
        )
        supersede_task = asyncio.create_task(
            adapter.replace(successor, None, [_flip(predecessor.id)], at=EPOCH)
        )

        results = await asyncio.wait_for(
            asyncio.gather(shelve_task, supersede_task, return_exceptions=True),
            timeout=5.0,
        )
        shelve_result, supersede_result = results
        assert not isinstance(shelve_result, BaseException), (
            f"the ordinary (shelve) transition must not itself fail in this race: "
            f"{shelve_result!r}"
        )
        assert not isinstance(supersede_result, BaseException), (
            f"the supersede must not itself fail in this race: {supersede_result!r}"
        )

        final = await adapter.get_by_id(predecessor.id)
        assert isinstance(final, Fact)
        assert final.lifecycle == "shelved", (
            "the ordinary transition's already-landed lifecycle change must "
            "survive the supersede's write, identically on both adapters — "
            "a stale pre-race snapshot must never revert it"
        )
        assert final.superseded_by == successor.id
        assert final.validity == "invalid"


# ===========================================================================
# 5h — the two-arm undo storage seam (P4 two-arm unsupersede piece,
# 2026-09-20). ``restore`` replaces the Fact-only
# ``transition(..., clear_superseded_by=True)`` (2026-08-11
# fact-supersession-undo) with an arm-general reversal of ``replace``'s
# forward edge; ``append_unsupersede``/``plant_tombstone``/``repair_
# unsupersedes`` are new. The ENGINE-level idempotent outcome lives one
# layer up (``engine.dispositions.unsupersede``, tested in
# ``tests/test_engine_dispositions.py``); the storage layer itself has no
# no-write ack — ``restore`` raises on a stale predecessor, the same
# set-exactly-once discipline ``test_superseded_by_set_exactly_once``
# already proves for the forward edge.
# ===========================================================================


def _unsupersede_event(
    *,
    edge_id: str,
    predecessor_id: HypomnemaId,
    successor_id: HypomnemaId,
    arm: str = "fact",
    reason: str = "the edge was wrong",
    withdrawn_reason: SupersessionReason = "correction",
    at: datetime = EPOCH,
) -> UnsupersedeEvent:
    return UnsupersedeEvent(
        id=uuid.uuid4(),
        edge_id=edge_id,
        predecessor_id=predecessor_id,
        successor_id=successor_id,
        arm=arm,  # type: ignore[arg-type]
        withdrawn_reason=withdrawn_reason,
        withdrawn_note=None,
        reason=reason,
        at=at,
    )


async def _plant_null_hash_tombstone(
    adapter: AnyAdapter, buried_id: HypomnemaId, *, restores: HypomnemaId
) -> None:
    """Plant a pending tombstone carrying ``restores`` but NO captured
    ``restores_content_hash`` — the shape a tombstone planted before the
    edit-during-pending-undo fix landed would carry, on disk, forever.
    Bypasses ``plant_tombstone`` (which always computes a real hash now)
    to write straight to each adapter's own storage, the same "raw"
    escape hatch ``_raw_vector`` above uses."""
    if isinstance(adapter, LanceDBAdapter):
        rows = await raw_table(adapter._records).query().where(f"id = '{buried_id!s}'").to_list()
        rec_type: str = rows[0]["type"]
        await adapter._tombstones_tbl.add([{
            "tombstone_id": str(uuid.uuid4()),
            "buried_id": str(buried_id),
            "type": rec_type,
            "reason": "unsupersede: legacy-shaped, no captured hash",
            "timestamp": EPOCH,
            "model_generated": False,
            "restores": str(restores),
            "restores_content_hash": None,
        }])
        return
    assert isinstance(adapter, MemoryAdapter)
    row = adapter._rows[buried_id]
    tombstone = Tombstone(
        id=uuid.uuid4(),
        buried_id=buried_id,
        type=row.record.type,
        reason="unsupersede: legacy-shaped, no captured hash",
        timestamp=EPOCH,
        model_generated=False,
        restores=restores,
        restores_content_hash=None,
    )
    adapter._tombstones[tombstone.id] = tombstone
    adapter._tombstone_by_buried[buried_id] = tombstone


class TestUnsupersedeRestore:
    """Happy path + the storage-level guard's own idempotency shape, both
    arms."""

    async def test_restore_clears_pointer_and_restores_validity_fact(
        self, adapter: AnyAdapter, clock: TestClock, tmp_path: Any
    ) -> None:
        predecessor = _fact(content="wrongly superseded predecessor, happy path")
        successor = _fact(content="the (possibly-wrong) successor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)
        superseded = await adapter.get_by_id(predecessor.id)
        assert isinstance(superseded, Fact)
        assert superseded.superseded_by == successor.id
        assert superseded.validity == "invalid"

        adapter_v2 = await _reopen_at_model(
            adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock
        )
        targets_before = {t.id for t in await adapter_v2.backfill_targets()}
        assert predecessor.id in targets_before
        vector_before = await _raw_vector(adapter_v2, predecessor.id)

        later = EPOCH + timedelta(days=1)
        await adapter_v2.restore(
            predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=later,
        )
        restored = await adapter_v2.get_by_id(predecessor.id)
        assert isinstance(restored, Fact)
        assert restored.superseded_by is None
        assert restored.supersession_reason is None
        assert restored.supersession_note is None
        assert restored.supersession_edge_id is None
        assert restored.validity == "valid"
        assert restored.updated_at == later

        targets_after = {t.id for t in await adapter_v2.backfill_targets()}
        assert predecessor.id in targets_after, (
            "restore must never re-embed the predecessor's vector — its "
            "embedding_model stamp must still read as stale after the call"
        )
        vector_after = await _raw_vector(adapter_v2, predecessor.id)
        assert vector_after == vector_before, (
            "restore must carry the vector through byte-identical — stale "
            "backfill-target MEMBERSHIP alone does not prove the actual "
            "vector VALUES were never touched (T4)"
        )

        successor_after = await adapter_v2.get_by_id(successor.id)
        assert isinstance(successor_after, Fact)

    async def test_restore_restores_lifecycle_and_deletes_successor_decision(
        self, adapter: AnyAdapter
    ) -> None:
        predecessor = _decision(content="wrongly superseded Decision", validity="invalid")
        predecessor = Decision.model_validate(
            {**predecessor.model_dump(mode="python"), "lifecycle": "in_force", "validity": "valid"}
        )
        successor = _decision(content="the (possibly-wrong) Decision successor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))

        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)
        superseded = await adapter.get_by_id(predecessor.id)
        assert isinstance(superseded, Decision)
        assert superseded.lifecycle == "superseded"
        assert superseded.superseded_by == successor.id

        await adapter.plant_tombstone(
            successor.id, "unsupersede: wrong edge", restores=predecessor.id,
        )
        still_present = await adapter.get_by_id(successor.id)
        assert isinstance(still_present, Decision)

        await adapter.restore(
            predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
        )
        restored = await adapter.get_by_id(predecessor.id)
        assert isinstance(restored, Decision)
        assert restored.lifecycle == "in_force"
        assert restored.superseded_by is None
        assert restored.supersession_reason is None
        assert restored.supersession_note is None
        assert restored.supersession_edge_id is None
        assert restored.validity == "valid"

        with pytest.raises(KeyError):
            await adapter.get_by_id(successor.id)
        tombstone = await adapter.get_tombstone(successor.id)
        assert tombstone is not None
        assert tombstone.restores == predecessor.id

    async def test_restore_preserves_validity_when_it_was_invalid(
        self, adapter: AnyAdapter
    ) -> None:
        """D8: the restored row's ``validity`` is whatever it was set to —
        the restore never forces it back to ``valid`` on the Decision arm."""
        predecessor = _decision(content="Decision superseded, then separately invalidated")
        successor = _decision(content="successor Decision")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)
        await adapter.set_validity(predecessor.id, valid=False, reason="separately invalid")

        await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)
        await adapter.restore(
            predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
        )
        restored = await adapter.get_by_id(predecessor.id)
        assert isinstance(restored, Decision)
        assert restored.lifecycle == "in_force"
        assert restored.validity == "invalid"

    async def test_restore_refuses_on_wrong_successor_id(self, adapter: AnyAdapter) -> None:
        predecessor = _fact(content="predecessor for the wrong-successor-id refusal")
        successor = _fact(content="the real successor")
        other = _fact(content="a red herring, not the real successor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        await adapter.upsert(other, _vec(seed=0.8))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        with pytest.raises(ValueError, match="does not match"):
            await adapter.restore(
                predecessor.id, successor_id=other.id, edge_id=flip.edge_id, at=EPOCH,
            )

        unchanged = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged, Fact)
        assert unchanged.superseded_by == successor.id
        assert unchanged.validity == "invalid"

    async def test_restore_refuses_on_wrong_edge_id(self, adapter: AnyAdapter) -> None:
        predecessor = _fact(content="predecessor for the wrong-edge-id refusal")
        successor = _fact(content="the successor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        with pytest.raises(ValueError, match="does not match"):
            await adapter.restore(
                predecessor.id,
                successor_id=successor.id,
                edge_id=str(uuid.uuid4()),
                at=EPOCH,
            )

        unchanged = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged, Fact)
        assert unchanged.superseded_by == successor.id
        assert unchanged.supersession_edge_id == flip.edge_id

    async def test_restore_on_never_superseded_fact_raises(
        self, adapter: AnyAdapter
    ) -> None:
        """Storage-level guard: restore on a Fact that was never superseded
        raises — a silent no-op return here would paper over a caller bug.
        The engine's idempotent ack lives ABOVE this seam."""
        f = _fact(content="never superseded, restore attempted anyway")
        await adapter.upsert(f, _vec())
        with pytest.raises(ValueError, match=r"does not exist or is buried|does not match"):
            await adapter.restore(
                f.id, successor_id=new_id(), edge_id=str(uuid.uuid4()), at=EPOCH,
            )

    async def test_restore_called_twice_raises_on_second_call(
        self, adapter: AnyAdapter
    ) -> None:
        """Idempotent re-call at the STORAGE seam: unlike the engine's
        no-write ack, a second raw ``restore`` call against an
        already-cleared predecessor raises — the same set-exactly-once-style
        discipline the forward edge's own ``test_superseded_by_set_exactly_
        once`` proves, mirrored for the reverse direction."""
        predecessor = _fact(content="restored once, then again")
        successor = _fact(content="successor for the double-restore test")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)
        await adapter.restore(
            predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
        )
        with pytest.raises(ValueError, match="does not match"):
            await adapter.restore(
                predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
            )

    async def test_two_concurrent_restores_of_same_predecessor(
        self, adapter: AnyAdapter
    ) -> None:
        """CAS proof, not just a pre-call cheap check: two GENUINELY
        concurrent ``restore`` calls on the SAME predecessor — mirrors
        ``test_two_concurrent_supersessions_of_same_predecessor`` above.

        Falsify: remove the fresh-CAS-recheck in ``resolve_unsupersede_
        write`` (or move the fresh read/write outside the lock) — both
        calls would then succeed, and the ``len(failures) == 1`` assertion
        goes red.
        """
        predecessor = _fact(content="predecessor, restored by two racing callers")
        successor = _fact(content="successor of the racing-restore predecessor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        results = await asyncio.gather(
            adapter.restore(
                predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
            ),
            adapter.restore(
                predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
            ),
            return_exceptions=True,
        )

        successes = [r for r in results if not isinstance(r, BaseException)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1, results
        assert len(failures) == 1, results
        assert isinstance(failures[0], ValueError), failures[0]
        assert "does not match" in str(failures[0])

        final = await adapter.get_by_id(predecessor.id)
        assert isinstance(final, Fact)
        assert final.superseded_by is None
        assert final.validity == "valid"

    async def test_restore_a_vectorless_successor_on_the_decision_arm(
        self, adapter: AnyAdapter
    ) -> None:
        """P6 vector-less-rows piece x P4 two-arm unsupersede piece: a
        successor that landed vector-less (an embedder
        outage at ``replace`` time) can still be undone. ``plant_tombstone``
        must derive ``restores_content_hash`` from the successor's stored
        ``content``, never from its (absent) vector, and ``restore`` must
        delete the vector-less row the same way it deletes a vectored one.
        """
        predecessor = _decision(content="Decision, wrongly superseded by an outage-landed edge")
        successor = _decision(content="successor Decision, landed vector-less mid-outage")
        await adapter.upsert(predecessor, _vec())

        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        landed = await adapter.get_by_id(successor.id)
        assert isinstance(landed, Decision)
        assert landed.embedding_model is None
        assert await _raw_vector(adapter, successor.id) is None

        await adapter.plant_tombstone(
            successor.id, "unsupersede: wrong edge", restores=predecessor.id,
        )
        tombstone = await adapter.get_tombstone(successor.id)
        assert tombstone is not None
        assert tombstone.restores_content_hash == content_hash(successor.content)

        await adapter.restore(
            predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
        )
        restored = await adapter.get_by_id(predecessor.id)
        assert isinstance(restored, Decision)
        assert restored.lifecycle == "in_force"
        assert restored.superseded_by is None
        assert restored.validity == "valid"

        with pytest.raises(KeyError):
            await adapter.get_by_id(successor.id)


class TestUnsupersedeBuriedPredecessorRace:
    """A Fact bury racing a concurrent restore of the SAME predecessor must
    never produce a phantom success — the restore mirror of
    ``TestFactBuryConcurrency`` above."""

    async def test_bury_interleaved_with_restore_never_phantom_succeeds(
        self, adapter: AnyAdapter
    ) -> None:
        predecessor = _fact(content="superseded, then raced: buried vs restored")
        successor = _fact(content="successor of the raced-bury-vs-restore predecessor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        bury_task = asyncio.create_task(
            adapter.bury(predecessor.id, reason="races a concurrent restore")
        )
        restore_task = asyncio.create_task(
            adapter.restore(
                predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
            )
        )

        results = await asyncio.wait_for(
            asyncio.gather(bury_task, restore_task, return_exceptions=True),
            timeout=5.0,
        )
        bury_result, restore_result = results

        assert not isinstance(bury_result, BaseException), (
            f"bury() must not itself fail in this race: {bury_result!r}"
        )
        if isinstance(restore_result, BaseException):
            assert isinstance(restore_result, ValueError), (
                "the restore losing this race must raise a clean "
                f"ValueError (predecessor gone), never "
                f"{type(restore_result).__name__}: {restore_result!r}"
            )

        # Whichever order won, bury has no CAS precondition of its own —
        # the predecessor ends up buried either way.
        with pytest.raises(KeyError):
            await adapter.get_by_id(predecessor.id)


class TestUnsupersedeVsOrdinaryTransition:
    """A concurrent restore racing an ORDINARY lifecycle change
    (``set_shelved``'s shape) on the SAME predecessor must not silently
    erase either write — the restore mirror of
    ``TestOrdinaryFactTransitionSupersedeRace`` above."""

    async def test_ordinary_transition_does_not_erase_concurrent_restore(
        self, adapter: AnyAdapter
    ) -> None:
        predecessor = _fact(content="superseded; raced: restored vs shelved")
        successor = _fact(content="successor of the raced-shelve-vs-restore predecessor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        # Created FIRST: gives the restore's write priority over the
        # ordinary (shelve) transition, created second (same ordering as
        # ``TestOrdinaryFactTransitionSupersedeRace``).
        restore_task = asyncio.create_task(
            adapter.restore(
                predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
            )
        )
        shelve_task = asyncio.create_task(
            adapter.transition(predecessor.id, "shelved", at=EPOCH)
        )

        results = await asyncio.wait_for(
            asyncio.gather(restore_task, shelve_task, return_exceptions=True),
            timeout=5.0,
        )
        restore_result, shelve_result = results
        assert not isinstance(restore_result, BaseException), (
            f"the restore must not itself fail in this race: {restore_result!r}"
        )
        assert not isinstance(shelve_result, BaseException), (
            f"the ordinary transition must not itself fail in this race: {shelve_result!r}"
        )

        final = await adapter.get_by_id(predecessor.id)
        assert isinstance(final, Fact)
        assert final.superseded_by is None, (
            "the concurrent restore's cleared pointer must not be silently "
            "re-set by the ordinary transition's write rebuilding from a "
            "stale pre-race snapshot"
        )
        assert final.validity == "valid"
        assert final.lifecycle == "shelved", (
            "the ordinary transition's own write must still land — this "
            "fix composes both writes, it does not make the ordinary one lose"
        )


class TestUnsupersedeVsForwardSupersedeOnSuccessorSide:
    """A Fact ``p`` currently superseded by ``a`` (so ``p`` is itself
    invalid + retired) racing a concurrent restore of ITS OWN edge against a
    DIFFERENT forward supersede that names ``p`` as its OWN successor
    (``q -> p``) exercises the boundary between the two guards:
    ``resolve_replace_successor``'s "successor must not itself already be
    superseded" check must observe ``p``'s TRUE, freshest state."""

    async def test_forward_supersede_targeting_a_concurrently_restored_successor(
        self, adapter: AnyAdapter
    ) -> None:
        a = _fact(content="a: p's original (possibly wrong) successor")
        p = _fact(content="p: superseded by a, also wants to become q's successor")
        q = _fact(content="q: wants p as its successor, racing p's own restore")
        await adapter.upsert(a, _vec())
        await adapter.upsert(p, _vec(seed=0.9))
        await adapter.upsert(q, _vec(seed=0.8))

        flip_pa = _flip(p.id)
        await adapter.replace(a, None, [flip_pa], at=EPOCH)

        restore_task = asyncio.create_task(
            adapter.restore(p.id, successor_id=a.id, edge_id=flip_pa.edge_id, at=EPOCH)
        )
        supersede_task = asyncio.create_task(
            adapter.replace(p, None, [_flip(q.id)], at=EPOCH)
        )

        results = await asyncio.wait_for(
            asyncio.gather(restore_task, supersede_task, return_exceptions=True),
            timeout=5.0,
        )
        restore_result, supersede_result = results
        assert not isinstance(restore_result, BaseException), (
            f"restoring p must not itself fail in this race: {restore_result!r}"
        )

        # p's own state must be fully restored regardless of what happened
        # to q — the reverse direction never touches a second referent, so
        # nothing q's supersede does (win or lose) can affect p's own row.
        final_p = await adapter.get_by_id(p.id)
        assert isinstance(final_p, Fact)
        assert final_p.superseded_by is None
        assert final_p.validity == "valid"

        final_q = await adapter.get_by_id(q.id)
        assert isinstance(final_q, Fact)
        if isinstance(supersede_result, BaseException):
            assert isinstance(supersede_result, ValueError), (
                f"q's supersede must lose cleanly with ValueError, not "
                f"{type(supersede_result).__name__}: {supersede_result!r}"
            )
            assert "already superseded" in str(supersede_result)
            assert final_q.superseded_by is None
            assert final_q.validity == "valid"
        else:
            assert final_q.superseded_by == p.id
            assert final_q.validity == "invalid"


class TestAppendUnsupersedeIdempotent:
    async def test_append_twice_with_one_edge_id_yields_one_row(
        self, adapter: AnyAdapter
    ) -> None:
        predecessor = _fact(content="predecessor for the append-idempotency test")
        successor = _fact(content="successor for the append-idempotency test")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id, predecessor_id=predecessor.id, successor_id=successor.id,
        )
        first = await adapter.append_unsupersede(event)
        assert first == event

        retry_event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            reason="a completely different retry reason",
        )
        second = await adapter.append_unsupersede(retry_event)
        assert second == event, "a second append for the SAME edge_id must return the FIRST row"

        all_events = await adapter.list_unsupersedes()
        assert len(all_events) == 1
        assert all_events[0] == event

        by_edge = await adapter.get_unsupersede(flip.edge_id)
        assert by_edge == event

    async def test_append_refuses_a_collision_naming_a_different_undo(
        self, adapter: AnyAdapter
    ) -> None:
        """An existing row
        for the SAME ``edge_id`` is only a legitimate retry when its
        predecessor, successor and arm all match — a collision naming a
        different undo must be refused, never silently kept as though it
        were the caller's own."""
        predecessor = _fact(content="predecessor for the collision-refusal test")
        successor = _fact(content="successor for the collision-refusal test")
        other_predecessor = _fact(content="a different predecessor entirely")
        other_successor = _fact(content="a different successor entirely")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        await adapter.upsert(other_predecessor, _vec(seed=0.8))
        await adapter.upsert(other_successor, _vec(seed=0.7))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id, predecessor_id=predecessor.id, successor_id=successor.id,
        )
        await adapter.append_unsupersede(event)

        colliding_event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=other_predecessor.id,
            successor_id=other_successor.id,
        )
        with pytest.raises(ValueError, match="already recorded"):
            await adapter.append_unsupersede(colliding_event)

        all_events = await adapter.list_unsupersedes()
        assert len(all_events) == 1
        assert all_events[0] == event

    async def test_get_unsupersede_unknown_edge_id_returns_none(
        self, adapter: AnyAdapter
    ) -> None:
        assert await adapter.get_unsupersede(str(uuid.uuid4())) is None


class TestPlantTombstone:
    async def test_plant_tombstone_leaves_buried_row_present(
        self, adapter: AnyAdapter
    ) -> None:
        d = _decision(content="Decision about to be undo-tombstoned")
        await adapter.upsert(d, _vec())
        predecessor_id = new_id()

        tombstone_id = await adapter.plant_tombstone(
            d.id, "unsupersede: testing", restores=predecessor_id,
        )
        assert tombstone_id is not None

        still_present = await adapter.get_by_id(d.id)
        assert isinstance(still_present, Decision)

        tombstone = await adapter.get_tombstone(d.id)
        assert tombstone is not None
        assert tombstone.restores == predecessor_id

    async def test_plant_tombstone_unknown_id_raises(self, adapter: AnyAdapter) -> None:
        with pytest.raises(KeyError):
            await adapter.plant_tombstone(new_id(), "unsupersede: x", restores=new_id())

    async def test_plant_tombstone_captures_the_current_content_hash(
        self, adapter: AnyAdapter
    ) -> None:
        """Edit-during-pending-undo fix: the hash
        ``resolve_unsupersede_repair``/the live undo path later compare
        against is captured at PLANT time, from the buried row's own
        content — never left ``None`` by a fresh ``plant_tombstone`` call."""
        d = _decision(content="Decision, hash captured at plant time")
        await adapter.upsert(d, _vec())

        await adapter.plant_tombstone(d.id, "unsupersede: testing", restores=new_id())

        tombstone = await adapter.get_tombstone(d.id)
        assert tombstone is not None
        assert tombstone.restores_content_hash == content_hash(d.content)

    async def test_plant_tombstone_refuses_a_second_conflicting_marker(
        self, adapter: AnyAdapter
    ) -> None:
        """A second
        ``plant_tombstone`` call for the same buried id must refuse on
        BOTH adapters — the memory adapter used to silently overwrite the
        reverse index with the conflicting ``restores``, while LanceDB
        already refused."""
        d = _decision(content="Decision about to get two conflicting tombstones")
        await adapter.upsert(d, _vec())
        predecessor_id = new_id()
        other_id = new_id()

        await adapter.plant_tombstone(d.id, "unsupersede: first", restores=predecessor_id)

        with pytest.raises(KeyError):
            await adapter.plant_tombstone(d.id, "unsupersede: conflicting", restores=other_id)

        tombstone = await adapter.get_tombstone(d.id)
        assert tombstone is not None
        assert tombstone.restores == predecessor_id


class TestWithdrawTombstone:
    """``withdraw_tombstone`` — the inverse of ``plant_tombstone``, used by
    both the live undo path and ``repair_unsupersedes``' ``"withdraw"``
    action on a hash mismatch (edit-during-pending-undo fix). Guarded by
    ``storage.supersede.resolve_withdraw_tombstone``
    on both adapters — nothing written unless a fresh read confirms the
    tombstone is genuinely still pending."""

    async def test_withdraw_removes_a_pending_tombstone_without_touching_the_row(
        self, adapter: AnyAdapter
    ) -> None:
        d = _decision(content="Decision with a pending undo tombstone, about to be withdrawn")
        await adapter.upsert(d, _vec())
        predecessor_id = new_id()
        await adapter.plant_tombstone(d.id, "unsupersede: pending", restores=predecessor_id)

        await adapter.withdraw_tombstone(d.id)

        assert await adapter.get_tombstone(d.id) is None
        still_present = await adapter.get_by_id(d.id)
        assert isinstance(still_present, Decision)
        assert still_present.lifecycle == "in_force"

    async def test_withdraw_refuses_on_a_tombstone_without_restores(
        self, adapter: AnyAdapter
    ) -> None:
        """A legacy tombstone from ordinary ``bury()`` (D8: ``restores=
        None``) never authorised a restore in the first place — there is
        no pending undo to withdraw."""
        d = _decision(content="Decision, ordinarily buried, no pending undo")
        await adapter.upsert(d, _vec())
        await adapter.bury(d.id, reason="ordinary burial, unrelated to any undo")

        with pytest.raises(KeyError):
            await adapter.withdraw_tombstone(d.id)

        tombstone = await adapter.get_tombstone(d.id)
        assert tombstone is not None
        assert tombstone.restores is None

    async def test_withdraw_refuses_on_a_completed_burial(self, adapter: AnyAdapter) -> None:
        """A tombstone whose buried row is ALREADY GONE — a completed
        undo — is exactly as permanent as an ordinary burial; it must
        never be withdrawn."""
        predecessor = _decision(content="Decision A, undo already completed")
        successor = _decision(content="Decision B, already deleted by restore")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)
        await adapter.plant_tombstone(
            successor.id, "unsupersede: x", restores=predecessor.id,
        )
        await adapter.restore(
            predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
        )
        with pytest.raises(KeyError):
            await adapter.get_by_id(successor.id)

        with pytest.raises(KeyError):
            await adapter.withdraw_tombstone(successor.id)

        tombstone = await adapter.get_tombstone(successor.id)
        assert tombstone is not None
        assert tombstone.restores == predecessor.id

    async def test_withdraw_unknown_id_raises(self, adapter: AnyAdapter) -> None:
        with pytest.raises(KeyError):
            await adapter.withdraw_tombstone(new_id())


class TestRepairUnsupersedes:
    """``repair_unsupersedes`` — the pure decision function
    (``resolve_unsupersede_repair``) exercised through the adapter's own
    implementation, both arms (P4 two-arm unsupersede piece)."""

    async def test_repair_completes_event_only_state_decision(
        self, adapter: AnyAdapter
    ) -> None:
        """Event landed, tombstone never planted, records transaction never
        ran — repair must plant the tombstone FIRST, then restore."""
        predecessor = _decision(content="Decision, undo crashed before the tombstone")
        successor = _decision(content="successor Decision")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 1
        assert skipped == 0
        assert refused == 0

        restored = await adapter.get_by_id(predecessor.id)
        assert isinstance(restored, Decision)
        assert restored.lifecycle == "in_force"
        assert restored.superseded_by is None
        with pytest.raises(KeyError):
            await adapter.get_by_id(successor.id)
        tombstone = await adapter.get_tombstone(successor.id)
        assert tombstone is not None
        assert tombstone.restores == predecessor.id

    async def test_repair_completes_event_plus_tombstone_state_decision(
        self, adapter: AnyAdapter
    ) -> None:
        """Event AND tombstone landed, the records transaction never ran —
        repair restores directly, without re-planting."""
        predecessor = _decision(content="Decision, undo crashed before the records transaction")
        successor = _decision(content="successor Decision")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)
        await adapter.plant_tombstone(
            successor.id, "unsupersede: x", restores=predecessor.id,
        )

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 1
        assert skipped == 0
        assert refused == 0

        restored = await adapter.get_by_id(predecessor.id)
        assert isinstance(restored, Decision)
        assert restored.lifecycle == "in_force"
        with pytest.raises(KeyError):
            await adapter.get_by_id(successor.id)

    async def test_repair_completes_event_only_state_fact(self, adapter: AnyAdapter) -> None:
        predecessor = _fact(content="Fact, undo crashed before the records transaction")
        successor = _fact(content="successor Fact")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id, predecessor_id=predecessor.id, successor_id=successor.id,
        )
        await adapter.append_unsupersede(event)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 1
        assert skipped == 0
        assert refused == 0

        restored = await adapter.get_by_id(predecessor.id)
        assert isinstance(restored, Fact)
        assert restored.superseded_by is None
        assert restored.validity == "valid"
        still_present = await adapter.get_by_id(successor.id)
        assert isinstance(still_present, Fact)

    async def test_repair_skips_a_completed_undo(self, adapter: AnyAdapter) -> None:
        predecessor = _fact(content="Fact, undo already fully completed")
        successor = _fact(content="successor Fact")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)
        event = _unsupersede_event(
            edge_id=flip.edge_id, predecessor_id=predecessor.id, successor_id=successor.id,
        )
        await adapter.append_unsupersede(event)
        await adapter.restore(
            predecessor.id, successor_id=successor.id, edge_id=flip.edge_id, at=EPOCH,
        )

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

    async def test_repair_skips_a_new_generation(self, adapter: AnyAdapter) -> None:
        """A→B undone, then A→B drawn again on purpose (a fresh edge id) —
        repair must leave the second edge alone."""
        predecessor = _fact(content="Fact, superseded, undone, superseded again")
        successor = _fact(content="successor Fact, reused after undo")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        first_flip = _flip(predecessor.id, reason="changed")
        await adapter.replace(successor, None, [first_flip], at=EPOCH)
        first_event = _unsupersede_event(
            edge_id=first_flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
        )
        await adapter.append_unsupersede(first_event)
        await adapter.restore(
            predecessor.id, successor_id=successor.id, edge_id=first_flip.edge_id, at=EPOCH,
        )

        second_flip = _flip(predecessor.id, reason="changed")
        await adapter.replace(successor, None, [second_flip], at=EPOCH)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

        final = await adapter.get_by_id(predecessor.id)
        assert isinstance(final, Fact)
        assert final.superseded_by == successor.id
        assert final.supersession_edge_id == second_flip.edge_id

    async def test_repair_never_visits_a_burial_with_no_undo_event(
        self, adapter: AnyAdapter
    ) -> None:
        """A Decision buried through the ordinary ``bury`` tool, with NO
        unsupersede event naming it, is never visited by repair at all —
        repair only ever walks ``list_unsupersedes()``."""
        d = _decision(content="a plain Decision, buried outright")
        await adapter.upsert(d, _vec())
        await adapter.bury(d.id, reason="ordinary burial, unrelated to any undo")

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 0
        assert refused == 0

        with pytest.raises(KeyError):
            await adapter.get_by_id(d.id)

    async def test_repair_skips_a_legacy_tombstone(self, adapter: AnyAdapter) -> None:
        """The original
        version of this test buried a Decision with NO supersession edge
        at all, so ``list_unsupersedes()`` was empty and repair's loop
        body never ran — it proved nothing about the legacy-tombstone
        guard (see ``test_repair_never_visits_a_burial_with_no_undo_event``
        above, which is what that scenario actually tests). This version
        gives the buried Decision a REAL A→B edge and a real undo event
        first, so repair actually visits it; the legacy tombstone
        (``restores=None``, from the ordinary ``bury`` tool) must still
        never be treated as authorisation, and neither row may move.

        Falsify: comment out the ``tombstone.restores is None`` guard in
        ``resolve_unsupersede_repair`` — this test's ``skipped == 0``
        assertion flips.
        """
        predecessor = _decision(content="Decision A, undo interrupted before landing")
        successor = _decision(content="Decision B, buried outright during the gap")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)
        await adapter.bury(successor.id, reason="ordinary burial, unrelated to the pending undo")

        tombstone = await adapter.get_tombstone(successor.id)
        assert tombstone is not None
        assert tombstone.restores is None

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

        unchanged_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"
        assert unchanged_a.superseded_by == successor.id
        with pytest.raises(KeyError):
            await adapter.get_by_id(successor.id)

    async def test_repair_skips_when_successor_resuperseded_during_pending_undo(
        self, adapter: AnyAdapter
    ) -> None:
        """A
        pending undo's own event/tombstone writes carry no marker on the
        successor's own row — a caught storage failure between them (never
        a process crash) leaves the server alive, and another caller can
        legally supersede the successor before a restart's repair runs.
        Restoring the predecessor anyway would delete a successor that has
        since grown its own, unrelated history.

        Falsify: drop the ``successor.lifecycle != "in_force"`` guard from
        ``resolve_unsupersede_repair`` — this test's ``skipped == 0``
        assertion would flip and C would be found deleted alongside B.
        """
        predecessor = _decision(content="Decision A, undo interrupted before landing")
        successor = _decision(content="Decision B, re-superseded during the gap")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)

        third = _decision(content="Decision C, B's own legitimate successor")
        await adapter.upsert(third, _vec(seed=0.8))
        await adapter.replace(third, None, [_flip(successor.id)], at=EPOCH)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

        unchanged_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.superseded_by == successor.id
        unchanged_b = await adapter.get_by_id(successor.id)
        assert isinstance(unchanged_b, Decision)
        assert unchanged_b.lifecycle == "superseded"
        assert unchanged_b.superseded_by == third.id
        unchanged_c = await adapter.get_by_id(third.id)
        assert isinstance(unchanged_c, Decision)
        assert unchanged_c.lifecycle == "in_force"

    async def test_repair_skips_when_successor_buried_during_pending_undo(
        self, adapter: AnyAdapter
    ) -> None:
        """The same gap as above, via an ordinary ``bury`` instead of a
        fresh supersession — the successor is gone with a legacy tombstone
        (``restores=None``) that never authorises a restore (D8), so repair
        must leave the predecessor alone rather than crash trying to plant
        a second tombstone for an already-buried id.

        Falsify: drop the ``successor is None`` guard from
        ``resolve_unsupersede_repair`` — this test's ``skipped == 0``
        assertion would flip and the predecessor would be wrongly restored.
        """
        predecessor = _decision(content="Decision A, undo interrupted before landing")
        successor = _decision(content="Decision B, buried during the gap")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)
        await adapter.bury(successor.id, reason="unrelated cleanup during the gap")

        tombstone = await adapter.get_tombstone(successor.id)
        assert tombstone is not None
        assert tombstone.restores is None

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

        unchanged_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.superseded_by == successor.id

    async def test_repair_skips_when_tombstone_names_a_different_predecessor(
        self, adapter: AnyAdapter
    ) -> None:
        """The same
        authority rule the live undo path enforces applies to repair too —
        a tombstone that exists for the successor but names a DIFFERENT
        predecessor must never be treated as this event's authorisation."""
        predecessor = _decision(content="Decision A, real edge to B")
        successor = _decision(content="Decision B, tombstoned by an unrelated undo")
        other = _decision(content="Decision X, unrelated predecessor")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        await adapter.upsert(other, _vec(seed=0.8))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)
        await adapter.plant_tombstone(successor.id, "crafted", restores=other.id)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

        unchanged_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"
        still_present_b = await adapter.get_by_id(successor.id)
        assert isinstance(still_present_b, Decision)

    async def test_repair_refuses_when_event_declares_the_wrong_arm(
        self, adapter: AnyAdapter
    ) -> None:
        """Repair used to
        dispatch the destructive Decision shape purely from
        ``isinstance(predecessor, Fact)``, never reading ``event.arm`` — a
        bare event row declaring the wrong arm for a real, ordinary,
        never-undone edge must not authorise repair to act on it."""
        predecessor = _decision(content="Decision A, real ordinary edge, never undone")
        successor = _decision(content="Decision B")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        fake_event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="fact",
        )
        await adapter.append_unsupersede(fake_event)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 1
        assert refused == 0

        unchanged_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"
        still_present_b = await adapter.get_by_id(successor.id)
        assert isinstance(still_present_b, Decision)

    async def test_repair_refuses_and_withdraws_on_a_content_hash_mismatch(
        self, adapter: AnyAdapter
    ) -> None:
        """Edit-during-pending-undo fix: the
        successor's content changed after the tombstone was planted but
        before repair ran — the edit wins. Repair must withdraw the stale
        tombstone rather than restore the predecessor and delete the
        successor, counting the edge as ``refused``, never ``completed``
        or ``skipped``.

        Falsify: drop the ``tombstone.restores_content_hash != content_
        hash(successor.content)`` compare from ``resolve_unsupersede_
        repair`` — this test's ``refused == 1`` assertion flips to
        ``completed == 1`` and the successor's edited content is gone.
        """
        predecessor = _decision(content="Decision A, undo interrupted before landing")
        successor = _decision(content="Decision B, about to be edited mid-undo")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)
        await adapter.plant_tombstone(
            successor.id, "unsupersede: x", restores=predecessor.id,
        )

        edited_vector = _vec(seed=0.5)
        await edit_content(adapter, successor.id, "Decision B — edited mid-undo", edited_vector)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 0
        assert refused == 1

        unchanged_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"
        assert unchanged_a.superseded_by == successor.id

        edited_b = await adapter.get_by_id(successor.id)
        assert isinstance(edited_b, Decision)
        assert edited_b.content == "Decision B — edited mid-undo"
        assert await _raw_vector(adapter, successor.id) == edited_vector

        assert await adapter.get_tombstone(successor.id) is None

    async def test_repair_treats_a_null_stored_hash_as_a_mismatch(
        self, adapter: AnyAdapter
    ) -> None:
        """A tombstone planted before this fix landed carries no captured
        hash at all (D3: NULL is never treated as a match — it cannot
        prove the successor was never edited since)."""
        predecessor = _decision(content="Decision A, undo interrupted before landing")
        successor = _decision(content="Decision B, tombstoned before the hash fix existed")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)
        await _plant_null_hash_tombstone(adapter, successor.id, restores=predecessor.id)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 0
        assert refused == 1

        unchanged_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"
        still_present_b = await adapter.get_by_id(successor.id)
        assert isinstance(still_present_b, Decision)
        assert await adapter.get_tombstone(successor.id) is None

    async def test_repair_refusal_survives_a_sibling_predecessor_in_the_same_pass(
        self, adapter: AnyAdapter
    ) -> None:
        """A same-pass
        bypass (distinct from and more severe than the known cross-restart
        gap: no restart is needed at all). ``StorageAdapter.replace`` lets
        one successor take several predecessors in a single call (D2), so
        two DIFFERENT predecessors (A and C) can each hold their own
        interrupted undo against the SAME successor B: A's tombstone is
        planted and pending; C's own live attempt appends its own event
        before the authority guard refuses it, since B's
        tombstone already names A. Once B is edited and repair runs, A's
        event correctly refuses (hash mismatch) and withdraws the
        tombstone — but C's event, walked LATER in the SAME pass, must not
        then see "no tombstone" and complete a fresh, destructive undo
        against the content A's refusal just protected.

        Falsify: drop the ``refused_successors_this_pass`` guard from
        ``resolve_unsupersede_repair`` — this test's edited-content
        assertion goes red for whichever processing order this adapter's
        ``list_unsupersedes()`` happens to return (deterministic per
        adapter: neither reorders a freshly-appended, single-fragment
        event log).
        """
        a = _decision(content="Decision A, one of two predecessors sharing B")
        c = _decision(content="Decision C, the other predecessor sharing B")
        b = _decision(content="Decision B, shared successor, about to be edited")
        await adapter.upsert(a, _vec())
        await adapter.upsert(c, _vec(seed=0.8))
        await adapter.upsert(b, _vec(seed=0.9))

        flip_a = _flip(a.id, reason="changed")
        flip_c = _flip(c.id, reason="changed")
        await adapter.replace(b, None, [flip_a, flip_c], at=EPOCH)

        event_a = _unsupersede_event(
            edge_id=flip_a.edge_id, predecessor_id=a.id, successor_id=b.id, arm="decision",
        )
        await adapter.append_unsupersede(event_a)
        await adapter.plant_tombstone(b.id, "unsupersede: a's attempt", restores=a.id)

        event_c = _unsupersede_event(
            edge_id=flip_c.edge_id, predecessor_id=c.id, successor_id=b.id, arm="decision",
        )
        await adapter.append_unsupersede(event_c)

        edited_vector = _vec(seed=0.5)
        await edit_content(adapter, b.id, "Decision B — edited mid-undo", edited_vector)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert refused == 1
        assert skipped == 1

        still_b = await adapter.get_by_id(b.id)
        assert isinstance(still_b, Decision)
        assert still_b.content == "Decision B — edited mid-undo"
        assert await _raw_vector(adapter, b.id) == edited_vector

        unchanged_a = await adapter.get_by_id(a.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"
        unchanged_c = await adapter.get_by_id(c.id)
        assert isinstance(unchanged_c, Decision)
        assert unchanged_c.lifecycle == "superseded"

    async def test_repair_finishes_a_withdrawal_interrupted_between_the_two_writes(
        self, adapter: AnyAdapter
    ) -> None:
        """Durable refusal follow-up: the refusal itself is two writes —
        ``mark_unsupersede_refused`` first, then ``withdraw_tombstone`` —
        so a crash between them is possible. Built directly here rather
        than via fault injection: an event whose ``refused_at`` is already
        set, with its tombstone STILL present. The next repair pass must
        finish the withdrawal ALONE — no hash re-compare, no touch to
        either the predecessor or the successor row — and still count it
        under ``refused``.

        Falsify: drop the ``event.refused_at is not None`` short-circuit
        from ``resolve_unsupersede_repair`` — this test's tombstone-gone
        assertion still passes (the hash happens to match, since nothing
        edited the successor here), but ``test_repair_refuses_and_
        withdraws_on_a_content_hash_mismatch``'s sibling case combined
        with a stale captured hash would instead re-derive the wrong
        decision from a hash that no longer means what it once did.
        """
        predecessor = _decision(content="Decision A, refusal interrupted between its two writes")
        successor = _decision(content="Decision B, tombstone still pending the withdrawal")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)
        await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)
        await adapter.mark_unsupersede_refused(flip.edge_id, at=EPOCH)

        pred_before = await adapter.get_by_id(predecessor.id)
        succ_before = await adapter.get_by_id(successor.id)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 0
        assert refused == 1

        assert await adapter.get_tombstone(successor.id) is None
        pred_after = await adapter.get_by_id(predecessor.id)
        succ_after = await adapter.get_by_id(successor.id)
        assert pred_after == pred_before
        assert succ_after == succ_before

        stored_event = await adapter.get_unsupersede(flip.edge_id)
        assert stored_event is not None
        assert stored_event.refused_at is not None

    async def test_repair_never_replays_a_fully_resolved_refusal(
        self, adapter: AnyAdapter
    ) -> None:
        """Once a refusal is fully resolved (``refused_at`` set, tombstone
        already gone), a LATER repair pass must never re-attempt it — this
        is exactly the cross-restart gap the durable marker closes.
        """
        predecessor = _decision(content="Decision A, refusal already fully resolved")
        successor = _decision(content="Decision B, edited then protected by refusal")
        await adapter.upsert(predecessor, _vec())
        await adapter.upsert(successor, _vec(seed=0.9))
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)
        await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)

        edited_vector = _vec(seed=0.5)
        await edit_content(
            adapter, successor.id, "Decision B — edited, then refused", edited_vector
        )

        first_report = await adapter.repair_unsupersedes()
        first_completed = first_report.completed
        first_refused = first_report.refused
        assert first_completed == 0
        assert first_refused == 1

        second_report = await adapter.repair_unsupersedes()
        second_completed = second_report.completed
        second_skipped = second_report.skipped
        second_refused = second_report.refused
        assert second_completed == 0
        assert second_refused == 0
        assert second_skipped == 1

        still_b = await adapter.get_by_id(successor.id)
        assert isinstance(still_b, Decision)
        assert still_b.content == "Decision B — edited, then refused"
        assert await _raw_vector(adapter, successor.id) == edited_vector
        unchanged_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"

    async def test_a_refused_undo_survives_backfill_of_its_vectorless_successor(
        self, adapter: AnyAdapter
    ) -> None:
        """P6 vector-less-rows piece x P4 two-arm unsupersede piece: the
        successor never had a vector at all (an outage at ``replace`` time,
        not just an outage at edit time) when its mid-undo edit triggers a
        content-hash refusal. Refusal is durable (P4's ``refused_at``); a
        LATER backfill sweep filling the row's vector must not resurrect
        the undo, and the refusal must still read back the same way once
        the row is no longer vector-less.
        """
        predecessor = _decision(content="Decision A, undo interrupted before landing")
        successor = _decision(content="Decision B, never given a vector")
        await adapter.upsert(predecessor, _vec())
        flip = _flip(predecessor.id)
        await adapter.replace(successor, None, [flip], at=EPOCH)

        event = _unsupersede_event(
            edge_id=flip.edge_id,
            predecessor_id=predecessor.id,
            successor_id=successor.id,
            arm="decision",
        )
        await adapter.append_unsupersede(event)
        await adapter.plant_tombstone(successor.id, "unsupersede: x", restores=predecessor.id)

        await edit_content(adapter, successor.id, "Decision B — edited, still vector-less", None)

        report = await adapter.repair_unsupersedes()
        completed = report.completed
        skipped = report.skipped
        refused = report.refused
        assert completed == 0
        assert skipped == 0
        assert refused == 1
        assert await adapter.get_tombstone(successor.id) is None

        stamped_before = await adapter.get_by_id(successor.id)
        assert stamped_before.embedding_model is None
        assert await _raw_vector(adapter, successor.id) is None
        targets = {t.id: t for t in await adapter.backfill_targets()}
        assert successor.id in targets

        new_vector = _vec(seed=0.4)
        ok = await adapter.backfill(
            successor.id,
            new_vector,
            embedding_model="nomic-v2",
            expected_hash=targets[successor.id].content_hash,
        )
        assert ok is True

        filled = await adapter.get_by_id(successor.id)
        assert filled.content == "Decision B — edited, still vector-less"
        assert filled.embedding_model == "nomic-v2"
        assert await _raw_vector(adapter, successor.id) == new_vector

        still_superseded_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(still_superseded_a, Decision)
        assert still_superseded_a.lifecycle == "superseded"
        assert still_superseded_a.superseded_by == successor.id

        stored_event = await adapter.get_unsupersede(flip.edge_id)
        assert stored_event is not None
        assert stored_event.refused_at is not None

        again_report = await adapter.repair_unsupersedes()
        again_completed = again_report.completed
        again_skipped = again_report.skipped
        again_refused = again_report.refused
        assert again_completed == 0
        assert again_refused == 0
        assert again_skipped == 1

        unchanged_a = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged_a, Decision)
        assert unchanged_a.lifecycle == "superseded"


# ===========================================================================
# 6b — transition persists audience_hint (per-role grounding — this brief)
#
# The tri-state is the whole contract: ``None`` leaves the stored value
# untouched, a tuple replaces it, ``()`` clears it (restoring universal
# reach). This is the new seam capability the module docstring's "Corrected
# 2026-08-02" note describes — proven identically on BOTH adapters here; an
# arm-module test alone (e.g. only against MemoryAdapter) would leave the
# LanceDB mirror unverified.
# ===========================================================================


class TestTransitionAudienceHint:
    async def test_transition_persists_audience_hint_tuple(self, adapter: AnyAdapter) -> None:
        """A tuple passed to ``audience_hint`` replaces the stored value.

        Falsifiability (step 6): comment out the ``audience_hint`` persist
        line in ``storage/memory.py``/``storage/lancedb_adapter.py`` (the
        exact silent-discard bug the storage-seam prerequisite brief fixed)
        — the stored value stays ``None`` and this test goes red. See the
        run verdict for the captured failure output.
        """
        f = _fact(content="hint target", lifecycle="active", audience_hint=None)
        await adapter.upsert(f, _vec())

        await adapter.transition(f.id, "hot", at=EPOCH, audience_hint=("reviewer",))

        updated = await adapter.get_by_id(f.id)
        assert updated.audience_hint == ("reviewer",)

    async def test_transition_leaves_audience_hint_untouched_when_none(
        self, adapter: AnyAdapter
    ) -> None:
        """``audience_hint=None`` (the default) is silence, not an un-hint —
        a transition call that omits it must not disturb an existing hint.

        Falsifiability (step 6): drop the ``is not None`` guard so the
        write always applies (conflating "no opinion" with "clear it") —
        an unrelated transition (e.g. plain ``archived``) would then wipe
        the pre-existing hint to ``None`` and this test goes red.
        """
        f = _fact(
            content="hint survives an unrelated transition",
            lifecycle="active",
            audience_hint=("builder",),
        )
        await adapter.upsert(f, _vec())

        await adapter.transition(f.id, "archived", at=EPOCH)  # no audience_hint kwarg

        updated = await adapter.get_by_id(f.id)
        assert updated.audience_hint == ("builder",)

    async def test_transition_clears_audience_hint_with_empty_tuple(
        self, adapter: AnyAdapter
    ) -> None:
        """``audience_hint=()`` is an explicit un-hint — it must clear a
        pre-existing hint, restoring universal reach. ``None`` and ``()``
        are NOT the same request.

        Falsifiability (step 6): conflate ``None`` with ``()`` by changing
        the write guard from ``is not None`` to plain truthiness (``if
        audience_hint:``) — an empty tuple would then be silently treated
        as "no opinion" instead of "clear it", the stored hint would stay
        ``("builder",)``, and this test goes red.
        """
        f = _fact(
            content="hint explicitly cleared",
            lifecycle="active",
            audience_hint=("builder",),
        )
        await adapter.upsert(f, _vec())

        await adapter.transition(f.id, "hot", at=EPOCH, audience_hint=())

        updated = await adapter.get_by_id(f.id)
        assert not updated.audience_hint, (
            "an empty tuple must clear the hint (falsy — restores universal "
            "reach), not conflate with 'leave untouched'"
        )


class TestTransitionProvisional:
    """``transition``'s ``provisional`` tri-state (curation block,
    ``Fact``-only field) — folded in from review as an
    organizational hygiene fix (curation block): this mirrors
    ``TestTransitionAudienceHint`` above, the one in-repo precedent for a
    tri-state ``transition`` kwarg, so the dual-parameterized conformance
    suite (not just ``tests/test_ember_creation_audience.py`` plus its
    LanceDB migration test) exercises the contract directly.

    The tri-state itself is simpler than ``audience_hint``'s three-way
    split: ``provisional`` is a plain bool with no domain-level "absent"
    state, so there is no third "clear" value distinct from ``False`` —
    ``None`` (leave untouched) / ``True`` (set) / ``False`` (set) is the
    whole contract (``StorageAdapter.transition`` docstring).
    """

    async def test_transition_persists_provisional_true(
        self, adapter: AnyAdapter
    ) -> None:
        """``provisional=True`` is persisted by ``transition``.

        Falsifiability: comment out the ``provisional`` persist line in
        ``storage/memory.py``/``storage/lancedb_adapter.py`` — the stored
        value stays ``False`` (the domain default) and this test goes red.
        """
        f = _fact(content="provisional target", lifecycle="active", provisional=False)
        await adapter.upsert(f, _vec())

        await adapter.transition(f.id, "ember", at=EPOCH, provisional=True)

        updated = await adapter.get_by_id(f.id)
        assert isinstance(updated, Fact)
        assert updated.provisional is True

    async def test_transition_leaves_provisional_untouched_when_none(
        self, adapter: AnyAdapter
    ) -> None:
        """``provisional=None`` (the default) is silence — a transition
        call that omits it must not disturb an existing ``True`` value.

        Falsifiability: drop the ``is not None`` guard so the write always
        applies the parameter's Python-level default — an unrelated
        transition (e.g. plain ``archived``) would then silently reset
        ``provisional`` and this test goes red.
        """
        f = _fact(content="provisional survives an unrelated transition", provisional=True)
        await adapter.upsert(f, _vec())

        await adapter.transition(f.id, "archived", at=EPOCH)  # no provisional kwarg

        updated = await adapter.get_by_id(f.id)
        assert isinstance(updated, Fact)
        assert updated.provisional is True

    async def test_transition_sets_provisional_false_explicitly(
        self, adapter: AnyAdapter
    ) -> None:
        """``provisional=False`` explicitly clears a previously-``True``
        mark — the mechanism an owner-asserted retag / ``set_hot``
        promotion actually uses (ADR 0016's amendment). Unlike
        ``audience_hint``, there is no separate "empty" sentinel: ``False``
        IS the clear.

        Falsifiability: conflate ``False`` with ``None`` (e.g. treat both
        as "no opinion" via bare truthiness) — the stored value would stay
        ``True`` and this test goes red.
        """
        f = _fact(content="provisional mark explicitly cleared", provisional=True)
        await adapter.upsert(f, _vec())

        await adapter.transition(f.id, "hot", at=EPOCH, provisional=False)

        updated = await adapter.get_by_id(f.id)
        assert isinstance(updated, Fact)
        assert updated.provisional is False


# ===========================================================================
# 7 — Bury completeness
# ===========================================================================


class TestBuryCompleteness:
    """bury drops content + both legs + tombstone written with all §1.3 fields."""

    async def test_bury_removes_from_get_by_id(self, adapter: AnyAdapter) -> None:
        f = _fact(content="will be buried")
        await adapter.upsert(f, _vec())
        await adapter.bury(f.id, "test reason")

        with pytest.raises(KeyError):
            await adapter.get_by_id(f.id)

    async def test_bury_removes_from_vector_search(self, adapter: AnyAdapter) -> None:
        f = _fact(content="will be buried vector")
        v = _vec()
        await adapter.upsert(f, v)
        await adapter.bury(f.id, "test reason")

        hits = await adapter.vector_search(v, filters=_all_filters(), k=10)
        assert all(h.record.id != f.id for h in hits)

    async def test_bury_removes_from_text_search(self, adapter: AnyAdapter) -> None:
        f = _fact(content="unique buried keyword zephyr")
        await adapter.upsert(f, _vec())
        await adapter.bury(f.id, "test reason")

        hits = await adapter.text_search(
            "unique buried keyword zephyr", filters=_all_filters(), k=10
        )
        assert all(h.record.id != f.id for h in hits)

    async def test_bury_writes_tombstone_with_required_fields(
        self, adapter: AnyAdapter, clock: TestClock
    ) -> None:
        f = _fact(content="tombstone test fact")
        await adapter.upsert(f, _vec())
        # Advance clock so burial time differs from record timestamps
        clock.advance(timedelta(days=1))
        burial_instant = clock.now()
        tid = await adapter.bury(f.id, "test reason for tombstone")

        tombstone = await adapter.get_tombstone(f.id)
        assert tombstone is not None
        assert isinstance(tombstone, Tombstone)

        # All §1.3 fields
        assert tombstone.id == tid
        assert tombstone.buried_id == f.id
        assert tombstone.type == "fact"
        assert tombstone.reason == "test reason for tombstone"
        # Tombstone timestamp = burial instant from the Clock seam, not record.updated_at
        assert tombstone.timestamp == burial_instant
        assert tombstone.timestamp.tzinfo is not None  # tz-aware
        assert isinstance(tombstone.model_generated, bool)

    async def test_bury_decision_tombstone_type(self, adapter: AnyAdapter) -> None:
        d = _decision(content="buried decision")
        await adapter.upsert(d, _vec())
        await adapter.bury(d.id, "reason")
        t = await adapter.get_tombstone(d.id)
        assert t is not None
        assert t.type == "decision"

    async def test_bury_returns_tombstone_id(self, adapter: AnyAdapter) -> None:
        f = _fact(content="id check")
        await adapter.upsert(f, _vec())
        tid = await adapter.bury(f.id, "reason")
        assert isinstance(tid, uuid.UUID)

    async def test_event_log_not_cleared_by_bury(self, adapter: AnyAdapter) -> None:
        """Event log entries survive burial — they carry ids only."""
        f = _fact(content="event log survives burial")
        await adapter.upsert(f, _vec())
        ev = ReinforcementEvent(
            target_id=f.id,
            at=EPOCH,
            source="dedup",
        )
        await adapter.append_event(ev)
        await adapter.bury(f.id, "reason")

        events = await adapter.query_events(target_id=f.id)
        assert len(events) == 1

    async def test_distinct_pair_not_cleared_by_bury(self, adapter: AnyAdapter) -> None:
        """Pair registry entries survive burial — monotonic."""
        a = _fact(content="fact a")
        b = _fact(content="fact b")
        await adapter.upsert(a, _vec())
        await adapter.upsert(b, _vec(seed=0.9))

        a_id, b_id = (a.id, b.id) if str(a.id) < str(b.id) else (b.id, a.id)
        pair = DistinctPair(lower_id=a_id, higher_id=b_id, source="save", judged_at=EPOCH)
        await adapter.mark_distinct(pair)

        await adapter.bury(a.id, "reason")
        assert await adapter.is_distinct(a.id, b.id)

    async def test_bury_buried_id_raises(self, adapter: AnyAdapter) -> None:
        f = _fact(content="double bury attempt")
        await adapter.upsert(f, _vec())
        await adapter.bury(f.id, "first")

        with pytest.raises(KeyError):
            await adapter.bury(f.id, "second")

    async def test_buried_id_cannot_be_resurrected_via_upsert(
        self, adapter: AnyAdapter
    ) -> None:
        """Burial is permanent: re-upsert of a buried id must raise (F5)."""
        f = _fact(content="will be buried and must stay buried")
        v = _vec()
        await adapter.upsert(f, v)
        await adapter.bury(f.id, "permanent burial test")

        # Attempt to reinsert using the same id — must raise
        resurrected = Fact(
            id=f.id,
            type="fact",
            content=f.content,
            scope=f.scope,
            lifecycle="active",
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        with pytest.raises((ValueError, KeyError)):
            await adapter.upsert(resurrected, v)

        # Confirm the id is still absent from get_by_id and both search legs
        with pytest.raises(KeyError):
            await adapter.get_by_id(f.id)
        vector_hits = await adapter.vector_search(v, filters=_all_filters(), k=10)
        assert all(h.record.id != f.id for h in vector_hits)

    async def test_model_generated_tombstone_flag(self, adapter: AnyAdapter) -> None:
        """Tombstone model_generated flag is persisted per caller (F9)."""
        f_human = _fact(content="human burial")
        f_model = _fact(content="model burial")
        await adapter.upsert(f_human, _vec())
        await adapter.upsert(f_model, _vec(seed=0.9))

        await adapter.bury(f_human.id, "human decision", model_generated=False)
        await adapter.bury(f_model.id, "offline judge", model_generated=True)

        t_human = await adapter.get_tombstone(f_human.id)
        t_model = await adapter.get_tombstone(f_model.id)
        assert t_human is not None and t_human.model_generated is False
        assert t_model is not None and t_model.model_generated is True

    async def test_get_tombstone_returns_tombstone_after_burial(
        self, adapter: AnyAdapter
    ) -> None:
        """get_tombstone(id) returns the tombstone after burial (B5 Protocol method)."""
        f = _fact(content="get_tombstone after burial")
        await adapter.upsert(f, _vec())
        tid = await adapter.bury(f.id, "protocol method test")

        tombstone = await adapter.get_tombstone(f.id)
        assert tombstone is not None
        assert tombstone.buried_id == f.id
        assert tombstone.id == tid
        assert tombstone.reason == "protocol method test"

    async def test_get_tombstone_returns_none_for_live_id(
        self, adapter: AnyAdapter
    ) -> None:
        """get_tombstone(id) returns None for a live (not buried) record."""
        f = _fact(content="live get_tombstone test")
        await adapter.upsert(f, _vec())

        tombstone = await adapter.get_tombstone(f.id)
        assert tombstone is None

    async def test_get_tombstone_returns_none_for_unknown_id(
        self, adapter: AnyAdapter
    ) -> None:
        """get_tombstone(id) returns None for an unknown id (never raises)."""
        from tabularium.domain import new_id
        unknown = new_id()
        tombstone = await adapter.get_tombstone(unknown)
        assert tombstone is None


# ===========================================================================
# 8 — Event log
# ===========================================================================


class TestEventLog:
    """append_event + query_events + update_projections."""

    async def test_append_and_query_by_target(self, adapter: AnyAdapter) -> None:
        f = _fact(content="event target")
        await adapter.upsert(f, _vec())

        ev = ReinforcementEvent(target_id=f.id, at=EPOCH, source="dedup")
        await adapter.append_event(ev)

        results = await adapter.query_events(target_id=f.id)
        assert len(results) == 1
        assert results[0].target_id == f.id
        assert results[0].source == "dedup"

    async def test_query_by_source_filter(self, adapter: AnyAdapter) -> None:
        f = _fact(content="event source filter")
        await adapter.upsert(f, _vec())

        ev1 = ReinforcementEvent(target_id=f.id, at=EPOCH, source="dedup")
        ev2 = ReinforcementEvent(target_id=f.id, at=EPOCH, source="feedback", value=0.8)
        await adapter.append_event(ev1)
        await adapter.append_event(ev2)

        dedup_only = await adapter.query_events(source="dedup")
        assert all(e.source == "dedup" for e in dedup_only)

        feedback_only = await adapter.query_events(source="feedback")
        assert all(e.source == "feedback" for e in feedback_only)

    async def test_query_no_filter_returns_all(self, adapter: AnyAdapter) -> None:
        f = _fact(content="all events")
        await adapter.upsert(f, _vec())

        for src in ("dedup", "revival", "compact"):
            ev = ReinforcementEvent(target_id=f.id, at=EPOCH, source=src)
            await adapter.append_event(ev)

        all_events = await adapter.query_events()
        assert len(all_events) == 3

    async def test_update_projections_persists_verbatim(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact(content="projection target")
        await adapter.upsert(f, _vec())

        new_rc = 5
        new_lra = _ts(10)
        new_fs = 0.7
        new_dda = _ts(45)

        await adapter.update_projections(
            f.id,
            recall_count=new_rc,
            last_referenced_at=new_lra,
            feedback_score=new_fs,
            decay_due_at=new_dda,
        )

        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.recall_count == new_rc
        assert retrieved.last_referenced_at == new_lra
        assert isinstance(retrieved, Fact)
        assert math.isclose(retrieved.feedback_score, new_fs, abs_tol=1e-9)
        assert retrieved.decay_due_at == new_dda

    async def test_update_projections_decision_subset(
        self, adapter: AnyAdapter
    ) -> None:
        """Decisions carry recall_count + last_referenced_at but no feedback/decay."""
        d = _decision(content="decision projection")
        await adapter.upsert(d, _vec())

        new_rc = 3
        new_lra = _ts(5)

        # feedback_score and decay_due_at are Fact-only; adapter silently skips them
        await adapter.update_projections(
            d.id,
            recall_count=new_rc,
            last_referenced_at=new_lra,
            feedback_score=0.5,
            decay_due_at=_ts(100),
        )

        retrieved = await adapter.get_by_id(d.id)
        assert retrieved.recall_count == new_rc
        assert retrieved.last_referenced_at == new_lra

    async def test_feedback_event_carries_value(self, adapter: AnyAdapter) -> None:
        f = _fact(content="feedback target")
        await adapter.upsert(f, _vec())

        ev = ReinforcementEvent(target_id=f.id, at=EPOCH, source="feedback", value=0.6)
        await adapter.append_event(ev)

        results = await adapter.query_events(target_id=f.id)
        assert results[0].value == pytest.approx(0.6)

    async def test_matched_from_id_preserved(self, adapter: AnyAdapter) -> None:
        f1 = _fact(content="original")
        f2 = _fact(content="duplicate")
        await adapter.upsert(f1, _vec())
        await adapter.upsert(f2, _vec(seed=0.9))

        ev = ReinforcementEvent(
            target_id=f1.id, at=EPOCH, source="dedup", matched_from_id=f2.id
        )
        await adapter.append_event(ev)

        results = await adapter.query_events(target_id=f1.id)
        assert results[0].matched_from_id == f2.id

    async def test_feedback_event_carries_reason(self, adapter: AnyAdapter) -> None:
        """New seam capability (M2 slice B): ``reason`` round-trips
        through ``append_event``/``query_events`` verbatim, carried as-is —
        the same round-trip contract as ``value`` (test above) and
        ``matched_from_id``, on both adapters.
        """
        f = _fact(content="feedback reason target")
        await adapter.upsert(f, _vec())

        ev = ReinforcementEvent(
            target_id=f.id, at=EPOCH, source="feedback", value=0.6,
            reason="agent judged this unhelpful",
        )
        await adapter.append_event(ev)

        results = await adapter.query_events(target_id=f.id)
        assert results[0].reason == "agent judged this unhelpful"

    async def test_feedback_event_reason_absent_is_none(self, adapter: AnyAdapter) -> None:
        """An omitted ``reason`` round-trips as ``None`` — never an empty
        string — on both adapters (M2 slice B).
        """
        f = _fact(content="feedback reason absent target")
        await adapter.upsert(f, _vec())

        ev = ReinforcementEvent(target_id=f.id, at=EPOCH, source="feedback", value=0.4)
        await adapter.append_event(ev)

        results = await adapter.query_events(target_id=f.id)
        assert results[0].reason is None

    async def test_query_events_returns_chronological_order(
        self, adapter: AnyAdapter
    ) -> None:
        """New seam capability / ordering contract (M2 slice B,
        ``query_events`` returns rows sorted
        ascending by ``at`` — REGARDLESS of the order they were appended
        in. ``compute_ewma`` (SPEC §8.2 [A8]) is order-sensitive and
        trusts this. Appended out of chronological order (t3, t1, t2) on
        purpose so a naive "insertion/physical order" implementation on
        either adapter would fail this — only an adapter that actually
        sorts by ``at`` passes.
        """
        f = _fact(content="event chronological order target")
        await adapter.upsert(f, _vec())

        t1 = EPOCH
        t2 = EPOCH + timedelta(seconds=10)
        t3 = EPOCH + timedelta(seconds=20)

        await adapter.append_event(
            ReinforcementEvent(target_id=f.id, at=t3, source="feedback", value=0.3)
        )
        await adapter.append_event(
            ReinforcementEvent(target_id=f.id, at=t1, source="feedback", value=0.9)
        )
        await adapter.append_event(
            ReinforcementEvent(target_id=f.id, at=t2, source="feedback", value=0.1)
        )

        results = await adapter.query_events(target_id=f.id)
        assert [ev.at for ev in results] == [t1, t2, t3]
        assert [ev.value for ev in results] == [0.9, 0.1, 0.3]


class TestReadEvents:
    """``append_read_event`` — a
    SEPARATE table from ``events``/``ReinforcementEvent``.
    """

    async def test_read_events_separate_from_reinforcement_events(
        self, adapter: AnyAdapter
    ) -> None:
        """The whole point of the decision: a read event never lands in the
        reinforcement log, and vice versa — the two tables are genuinely
        separate, not a filtered view of one shared table.
        """
        f = _fact(content="separation target")
        await adapter.upsert(f, _vec())

        await adapter.append_read_event(ReadEvent(target_id=f.id, at=EPOCH, source="recall"))
        await adapter.append_event(ReinforcementEvent(target_id=f.id, at=EPOCH, source="dedup"))

        if isinstance(adapter, LanceDBAdapter):
            read_rows = await raw_table(adapter._read_events).query().to_list()
            read_sources = [row["source"] for row in read_rows]
        else:
            read_sources = [ev.source for ev in adapter._read_events]

        reinforcement_only = await adapter.query_events(target_id=f.id)
        assert read_sources == ["recall"]
        assert len(reinforcement_only) == 1
        assert reinforcement_only[0].source == "dedup"

    async def test_read_event_never_reachable_via_records(self, adapter: AnyAdapter) -> None:
        """A read event writes nothing to the records table — it never
        touches ``recall_count``/``last_referenced_at``/``updated_at``.
        """
        f = _fact(
            content="untouched by read event",
            recall_count=3,
            last_referenced_at=EPOCH,
        )
        await adapter.upsert(f, _vec())
        before = await adapter.get_by_id(f.id)

        await adapter.append_read_event(ReadEvent(target_id=f.id, at=EPOCH, source="recall"))

        after = await adapter.get_by_id(f.id)
        assert after.recall_count == before.recall_count
        assert after.last_referenced_at == before.last_referenced_at
        assert after.updated_at == before.updated_at


# ===========================================================================
# 8b — update_projections vs. a concurrent bury
# ===========================================================================


class TestUpdateProjectionsConcurrentBury:
    """A record buried strictly between ``update_projections``' own initial
    existence read and its terminal write must make THAT call raise
    ``KeyError``, not silently no-op — the postcondition the lock-free
    ``rebuild_projections`` maintenance verb depends on to detect a
    concurrently-buried id during its repair-mode write, on both adapters.

    Real interleaving via two ``asyncio.create_task`` calls (never a
    monkeypatched pre-call ``KeyError``), both started before either is
    given a single scheduling tick to run: each of ``bury`` and
    ``update_projections`` has exactly one internal cooperative yield point
    between its own read and its own write on ``MemoryAdapter`` (a bare
    ``await asyncio.sleep(0)``, the concurrency-control fix's test-fidelity
    device), so creating ``bury``'s task first gives it priority in the
    FIFO ready queue: both tasks' reads run (in creation order) while the
    row still exists, ``bury``'s write (the deletion) then runs before
    ``update_projections``'s write is given its turn — deterministic under
    CPython's single-threaded asyncio scheduler, not a timing gamble.
    ``asyncio.gather`` bounds the whole exchange so a hang reports as a
    failure rather than a stuck suite.

    Falsify: revert either adapter's ``update_projections`` to the
    pre-fix shape (``MemoryAdapter``: write through the stale pre-yield
    row reference without re-fetching; ``LanceDBAdapter``: don't check
    ``MergeResult.num_updated_rows``) — this test goes red with a bare
    ``AssertionError`` (no ``KeyError`` raised) instead of green, on the
    adapter that was reverted.
    """

    async def test_bury_landing_mid_update_projections_raises_keyerror(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact(content="race target for update_projections vs bury")
        await adapter.upsert(f, _vec())

        # Created FIRST: gives bury's own internal yield-then-delete
        # priority in the FIFO ready queue over update_projections's
        # yield-then-write, scheduled second (see class docstring).
        bury_task = asyncio.create_task(
            adapter.bury(f.id, reason="races update_projections")
        )
        update_task = asyncio.create_task(
            adapter.update_projections(
                f.id,
                recall_count=1,
                last_referenced_at=_ts(1),
                feedback_score=0.5,
                decay_due_at=_ts(30),
            )
        )

        results = await asyncio.wait_for(
            asyncio.gather(bury_task, update_task, return_exceptions=True),
            timeout=5.0,
        )
        bury_result, update_result = results

        assert not isinstance(bury_result, BaseException), (
            f"bury() must not itself fail in this race: {bury_result!r}"
        )
        assert isinstance(update_result, KeyError), (
            f"update_projections must raise KeyError when the record was "
            f"buried mid-call, not silently no-op; got {update_result!r}"
        )


# ===========================================================================
# 9 — DistinctPair registry
# ===========================================================================


class TestDistinctPairRegistry:
    """mark_distinct normalization, first-judgment-wins, is_distinct symmetric."""

    def _ordered_pair(self, a: HypomnemaId, b: HypomnemaId) -> tuple[HypomnemaId, HypomnemaId]:
        return (a, b) if str(a) < str(b) else (b, a)

    async def test_basic_roundtrip(self, adapter: AnyAdapter) -> None:
        a = new_id()
        b = new_id()
        lo, hi = self._ordered_pair(a, b)
        pair = DistinctPair(lower_id=lo, higher_id=hi, source="save", judged_at=EPOCH)
        await adapter.mark_distinct(pair)
        assert await adapter.is_distinct(a, b) is True

    async def test_normalization_both_orderings_collide(self, adapter: AnyAdapter) -> None:
        """Calling with (b,a) then (a,b) produces one row."""
        a = new_id()
        b = new_id()
        lo, hi = self._ordered_pair(a, b)

        pair1 = DistinctPair(lower_id=lo, higher_id=hi, source="save", judged_at=EPOCH)
        # Deliberately pass reversed order in the second call
        pair2 = DistinctPair(lower_id=hi, higher_id=lo, source="compact", judged_at=EPOCH)

        await adapter.mark_distinct(pair1)
        await adapter.mark_distinct(pair2)  # first-judgment-wins: should be a no-op

        # Only one row exists (implementation detail: frozenset key)
        assert await adapter.is_distinct(a, b) is True
        assert await adapter.is_distinct(b, a) is True

        # First judgment wins: source should be "save", not "compact"
        key: frozenset[HypomnemaId] = frozenset({a, b})
        stored = adapter._distinct[key]
        assert stored.source == "save"

    async def test_first_judgment_wins(self, adapter: AnyAdapter) -> None:
        a = new_id()
        b = new_id()
        lo, hi = self._ordered_pair(a, b)

        p1 = DistinctPair(lower_id=lo, higher_id=hi, source="save", judged_at=EPOCH)
        p2 = DistinctPair(lower_id=lo, higher_id=hi, source="compact", judged_at=EPOCH)

        await adapter.mark_distinct(p1)
        await adapter.mark_distinct(p2)  # ignored — first-judgment-wins

        key: frozenset[HypomnemaId] = frozenset({a, b})
        assert adapter._distinct[key].source == "save"

    async def test_is_distinct_symmetric(self, adapter: AnyAdapter) -> None:
        a = new_id()
        b = new_id()
        lo, hi = self._ordered_pair(a, b)
        pair = DistinctPair(lower_id=lo, higher_id=hi, source="save", judged_at=EPOCH)
        await adapter.mark_distinct(pair)

        assert await adapter.is_distinct(a, b) == await adapter.is_distinct(b, a)

    async def test_is_distinct_false_for_unregistered(self, adapter: AnyAdapter) -> None:
        assert await adapter.is_distinct(new_id(), new_id()) is False


# ===========================================================================
# 10 — Registries
# ===========================================================================


class TestRegistries:
    """create + list roundtrip; duplicate create is idempotent (no error)."""

    async def test_scope_create_and_list(self, adapter: AnyAdapter) -> None:
        await adapter.create_scope(Scope("alpha"))
        await adapter.create_scope(Scope("beta"))
        scopes = await adapter.list_scopes()
        assert "alpha" in scopes
        assert "beta" in scopes

    async def test_scope_duplicate_is_idempotent(self, adapter: AnyAdapter) -> None:
        await adapter.create_scope(Scope("alpha"))
        await adapter.create_scope(Scope("alpha"))  # must not raise
        scopes = await adapter.list_scopes()
        assert scopes.count(Scope("alpha")) == 1

    async def test_scope_registration_order_preserved(self, adapter: AnyAdapter) -> None:
        for name in ("first", "second", "third"):
            await adapter.create_scope(Scope(name))
        scopes = await adapter.list_scopes()
        assert scopes == ["first", "second", "third"]

    async def test_facet_create_and_list(self, adapter: AnyAdapter) -> None:
        await adapter.create_facet("constraint")
        await adapter.create_facet("convention")
        facets = await adapter.list_facets()
        assert "constraint" in facets
        assert "convention" in facets

    async def test_facet_duplicate_is_idempotent(self, adapter: AnyAdapter) -> None:
        await adapter.create_facet("constraint")
        await adapter.create_facet("constraint")
        assert await adapter.list_facets() == ["constraint"]

    async def test_role_create_and_list(self, adapter: AnyAdapter) -> None:
        await adapter.create_role("default")
        await adapter.create_role("implementer")
        roles = await adapter.list_roles()
        assert "default" in roles
        assert "implementer" in roles

    async def test_role_duplicate_is_idempotent(self, adapter: AnyAdapter) -> None:
        await adapter.create_role("default")
        await adapter.create_role("default")
        assert await adapter.list_roles() == ["default"]

    # -----------------------------------------------------------------
    # delete_role (curation block) — dual-parameterized coverage
    # for the new storage-seam capability. See its own docstring
    # (``StorageAdapter.delete_role``) for the two settled interface
    # questions this class exercises: missing-name behaviour, and
    # audience_hint interaction.
    # -----------------------------------------------------------------

    async def test_role_delete_removes_a_registered_role(
        self, adapter: AnyAdapter
    ) -> None:
        """A registered role is actually gone from ``list_roles()`` after
        ``delete_role`` — the guarantee this task exists to build, not
        merely "no longer seeded on a fresh store".

        Falsifiability: make ``delete_role`` a silent no-op (return
        immediately without touching the table/list) — this assertion
        goes red on both adapters. See the task verdict for the captured
        failure output.
        """
        await adapter.create_role("temp-role")
        assert "temp-role" in await adapter.list_roles()

        await adapter.delete_role("temp-role")

        roles = await adapter.list_roles()
        assert "temp-role" not in roles

    async def test_role_delete_leaves_other_roles_untouched(
        self, adapter: AnyAdapter
    ) -> None:
        await adapter.create_role("keep-me")
        await adapter.create_role("delete-me")

        await adapter.delete_role("delete-me")

        roles = await adapter.list_roles()
        assert "keep-me" in roles
        assert "delete-me" not in roles

    async def test_role_delete_of_unregistered_role_is_noop(
        self, adapter: AnyAdapter
    ) -> None:
        """Interface decision (task brief): deleting a role that was never
        registered is a no-op, not an error — symmetric with
        ``create_role``'s duplicate-registration no-op, and what lets
        ``Tabularium.initialize()`` call ``delete_role("default")``
        unconditionally on every engine startup, including a fresh store
        that never had ``"default"`` at all, with no existence check
        first."""
        await adapter.create_role("survivor")

        await adapter.delete_role("never-registered")  # must not raise

        assert await adapter.list_roles() == ["survivor"]

    async def test_role_delete_then_recreate_is_registered_again(
        self, adapter: AnyAdapter
    ) -> None:
        """delete then create is a real round-trip, not a tombstoned name —
        confirms delete_role does a plain row removal, not a soft-delete
        marker that would block re-registration."""
        await adapter.create_role("cycle-role")
        await adapter.delete_role("cycle-role")
        assert "cycle-role" not in await adapter.list_roles()

        await adapter.create_role("cycle-role")
        assert "cycle-role" in await adapter.list_roles()

    async def test_role_delete_does_not_touch_audience_hint_on_existing_records(
        self, adapter: AnyAdapter
    ) -> None:
        """Interface decision (task brief): deleting a role never touches
        any record's ``audience_hint`` and never refuses because a record
        still names it — the roles table and the ``records`` table are
        unrelated; a hint is a list of strings, not a foreign key. The
        stale hint entry goes INERT (nobody can ever again request the
        deleted role via ``context()``, which validates against this same
        registry — see ``engine/core.py``'s ``_require_roles``), not
        dangling, and the delete itself must not raise or refuse."""
        f = _fact(content="hinted at a role about to be deleted", audience_hint=("temp-role",))
        await adapter.upsert(f, _vec())
        await adapter.create_role("temp-role")

        await adapter.delete_role("temp-role")  # must not raise

        stored = await adapter.get_by_id(f.id)
        assert stored.audience_hint == ("temp-role",), (
            "the hint entry is left exactly as it was -- storage never "
            "cascades into records on a role delete"
        )
        assert "temp-role" not in await adapter.list_roles()

    async def test_role_create_and_delete_roundtrip_with_quote_in_name(
        self, adapter: AnyAdapter
    ) -> None:
        """A role name containing a single quote is created, listed, and
        deleted correctly on both adapters: the registry seam has no character restriction beyond
        non-empty (``cli.py``'s ``role add`` only checks ``name.strip()``),
        so a naive f-string SQL predicate (``f"name = '{name}'"``) would
        either raise a parse error or, worse, silently widen the match on
        LanceDB — while ``MemoryAdapter``'s plain Python string equality
        would be unaffected, an observable adapter divergence. Both
        adapters must treat the quote as ordinary data.

        Falsifiability: revert ``_registry_add``/``_registry_delete`` to
        raw f-string interpolation — this test raises (LanceDB) or
        collects the wrong role set on the ``[lancedb]`` parametrization
        while ``[memory]`` stays green, the exact adapter-divergence this
        test exists to catch.
        """
        quoted_name = "o'brien-role"
        await adapter.create_role(quoted_name)
        assert quoted_name in await adapter.list_roles()

        await adapter.delete_role(quoted_name)

        assert quoted_name not in await adapter.list_roles()


# ===========================================================================
# 11 — Backfill
# ===========================================================================


class TestBackfill:
    """backfill_targets / backfill (P6 vector-less-rows piece, D5/D6)."""

    async def test_no_targets_when_same_model(self, adapter: AnyAdapter) -> None:
        f = _fact(content="up to date")
        await adapter.upsert(f, _vec())
        targets = await adapter.backfill_targets()
        assert f.id not in {t.id for t in targets}

    async def test_backfill_targets_after_model_change(
        self, adapter: AnyAdapter, clock: TestClock, tmp_path: Any
    ) -> None:
        """A second adapter view opened at a different model sees stale rows.

        The model change is expressed via ``_reopen_at_model`` — construction
        for memory (shares the backing store), a real reopen of the same path
        for LanceDB — never by mutating ``adapter._embedding_model``.
        """
        f = _fact(content="needs re-embedding")
        await adapter.upsert(f, _vec())

        adapter_v2 = await _reopen_at_model(
            adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock
        )
        targets = await adapter_v2.backfill_targets()
        by_id = {t.id: t for t in targets}
        assert f.id in by_id
        assert by_id[f.id].content == "needs re-embedding"

    async def test_null_vector_row_is_a_target(self, adapter: AnyAdapter) -> None:
        """A row that landed vector-less (P6 vector-less-rows piece, D1/D3)
        is a backfill target even though its stamp was never set at all."""
        f = _fact(content="landed without a vector")
        await adapter.upsert(f, None)
        targets = await adapter.backfill_targets()
        assert f.id in {t.id for t in targets}

    async def test_buried_row_is_never_a_target(self, adapter: AnyAdapter) -> None:
        f = _fact(content="will be buried")
        await adapter.upsert(f, None)
        await adapter.bury(f.id, "test burial")
        targets = await adapter.backfill_targets()
        assert f.id not in {t.id for t in targets}

    async def test_backfill_updates_vector_and_stamp(
        self, adapter: AnyAdapter, clock: TestClock, tmp_path: Any
    ) -> None:
        f = _fact(content="backfill me")
        await adapter.upsert(f, _vec())

        adapter_v2 = await _reopen_at_model(
            adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock
        )
        targets = await adapter_v2.backfill_targets()
        by_id = {t.id: t for t in targets}
        assert f.id in by_id

        # Orthogonal to the original _vec() — any positive seed of _vec() is
        # collinear with every other, so only a genuinely orthogonal query
        # can prove the stored vector actually changed (rather than a
        # same-direction vector scoring 1.0 regardless).
        new_vec = _orthogonal_vec()
        ok = await adapter_v2.backfill(
            f.id,
            new_vec,
            embedding_model="nomic-v2",
            expected_hash=by_id[f.id].content_hash,
        )
        assert ok is True

        # After backfill: no longer a target under the v2 config
        targets_after = await adapter_v2.backfill_targets()
        assert f.id not in {t.id for t in targets_after}

        # Vector actually replaced — verified via the public search surface.
        hits = await adapter_v2.vector_search(new_vec, filters=_all_filters(), k=5)
        match = next(h for h in hits if h.record.id == f.id)
        assert math.isclose(match.score, 1.0, abs_tol=1e-6)

        # Stamp updated — verified via scan(), never via a private attribute.
        stamped = [m async for m in await adapter_v2.scan() if m.id == f.id]
        assert len(stamped) == 1
        assert stamped[0].embedding_model == "nomic-v2"

    async def test_open_refuses_a_dim_that_disagrees_with_the_physical_column(
        self, adapter: AnyAdapter, tmp_path: Any
    ) -> None:
        """P6 vector-less-rows piece: reopening a store at a
        NEW ``dim`` leaves the Lance column at its original width —
        ``_migrate_table_schema`` only adds missing columns, it never
        widens an existing one. Before this fix, ``open()`` accepted the
        mismatched config silently and ``backfill()`` then reported success
        against a row LanceDB had cast to NULL. ``open()`` must refuse
        outright, before a single row can be touched."""
        if not isinstance(adapter, LanceDBAdapter):
            pytest.skip("LanceDB-only: MemoryAdapter has no physical vector column")

        f = _fact(content="four-dim row, store born at dim=4")
        await adapter.upsert(f, _vec())

        with pytest.raises(ValueError, match="does not match"):
            await LanceDBAdapter.open(
                str(tmp_path / _LANCEDB_STORE_SUBDIR),
                embedding_model="nomic-v2",
                dim=adapter._dim + 4,
            )

    async def test_backfill_refuses_a_dim_that_disagrees_with_the_physical_column(
        self, adapter: AnyAdapter
    ) -> None:
        """Defense-in-depth alongside the ``open()`` refusal above, for an
        adapter constructed directly rather than through ``open()`` (P6
        vector-less-rows piece): ``backfill()`` must treat a
        physical-column mismatch as a hard error BEFORE any write, never
        silently accept a wrong-width vector that LanceDB would cast to
        NULL."""
        if not isinstance(adapter, LanceDBAdapter):
            pytest.skip("LanceDB-only: MemoryAdapter has no physical vector column")

        f = _fact(content="landed vector-less, then the configured dim drifts")
        await adapter.upsert(f, None)

        adapter._dim = adapter._dim + 4  # simulates a construction bypassing open()'s own guard
        with pytest.raises(ValueError, match="physical vector column width"):
            await adapter.backfill(
                f.id, _vec(dim=adapter._dim), embedding_model="nomic-v2",
                expected_hash="irrelevant — the dim check runs before any hash check",
            )

        after = await adapter.get_by_id(f.id)
        assert after.embedding_model is None

    async def test_backfill_returns_false_and_writes_nothing_on_hash_mismatch(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact(content="will be edited before backfill lands")
        await adapter.upsert(f, None)
        before = await adapter.get_by_id(f.id)

        ok = await adapter.backfill(
            f.id, _vec(), embedding_model="nomic-v2", expected_hash="stale-hash-value"
        )
        assert ok is False

        after = await adapter.get_by_id(f.id)
        assert after.content == before.content
        assert after.embedding_model is None

        # T5 (P6 vector-less-rows piece, backfill test-gap fix): Fact/
        # Decision domain objects carry no vector field at all, so the
        # ``after`` checks above cannot see a vector-only corruption (a
        # backfill() that writes the vector but correctly reports False
        # and leaves the stamp None). Read the raw stored vector directly.
        assert await _raw_vector(adapter, f.id) is None


# ===========================================================================
# 11b — Every rewrite path preserves the row's OWN embedding_model stamp
# (P6 vector-less-rows piece). None of these methods ever
# re-embeds; relabeling a retained vector under the currently configured
# model would hide a genuinely stale vector from ``backfill_targets`` after
# a model migration. ``replace()`` already had this property (see
# ``test_reused_successor_keeps_its_own_embedding_model_stamp`` above);
# these four paths did not, on ``LanceDBAdapter`` only — ``MemoryAdapter``
# never touches the stamp on any of them, so it always had this property.
# ===========================================================================


class TestRewritePathsPreserveEmbeddingModelStamp:
    async def test_update_projections_keeps_the_stale_model_a_backfill_target(
        self, adapter: AnyAdapter, clock: TestClock, tmp_path: Any
    ) -> None:
        f = _fact(content="row for update_projections restamp check")
        await adapter.upsert(f, _vec())

        adapter_v2 = await _reopen_at_model(
            adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock
        )
        await adapter_v2.update_projections(
            f.id, recall_count=1, last_referenced_at=EPOCH,
            feedback_score=0.0, decay_due_at=None,
        )
        targets = {t.id for t in await adapter_v2.backfill_targets()}
        assert f.id in targets

    async def test_transition_keeps_the_stale_model_a_backfill_target(
        self, adapter: AnyAdapter, clock: TestClock, tmp_path: Any
    ) -> None:
        f = _fact(content="row for transition restamp check")
        await adapter.upsert(f, _vec())

        adapter_v2 = await _reopen_at_model(
            adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock
        )
        await adapter_v2.transition(f.id, "hot", at=EPOCH)
        targets = {t.id for t in await adapter_v2.backfill_targets()}
        assert f.id in targets

    async def test_set_validity_keeps_the_stale_model_a_backfill_target(
        self, adapter: AnyAdapter, clock: TestClock, tmp_path: Any
    ) -> None:
        f = _fact(content="row for set_validity restamp check")
        await adapter.upsert(f, _vec())

        adapter_v2 = await _reopen_at_model(
            adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock
        )
        await adapter_v2.set_validity(f.id, False, "test")
        targets = {t.id for t in await adapter_v2.backfill_targets()}
        assert f.id in targets

    async def test_update_recurrence_keeps_the_stale_model_a_backfill_target(
        self, adapter: AnyAdapter, clock: TestClock, tmp_path: Any
    ) -> None:
        f = _fact(content="row for update_recurrence restamp check")
        await adapter.upsert(f, _vec())

        adapter_v2 = await _reopen_at_model(
            adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock
        )
        await adapter_v2.update_recurrence(
            f.id, owner_recurrence_count=1, agent_recurrence_count=0, escalation_baseline=0,
        )
        targets = {t.id for t in await adapter_v2.backfill_targets()}
        assert f.id in targets


# ===========================================================================
# 12 — set_validity idempotency
# ===========================================================================


class TestSetValidity:
    """set_validity is idempotent on same-value re-call."""

    async def test_set_validity_changes_value(self, adapter: AnyAdapter) -> None:
        f = _fact(content="validity flip", validity="valid")
        await adapter.upsert(f, _vec())
        await adapter.set_validity(f.id, valid=False, reason="obsolete")
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.validity == "invalid"

    async def test_set_validity_idempotent_same_value(self, adapter: AnyAdapter) -> None:
        f = _fact(content="already invalid", validity="invalid")
        await adapter.upsert(f, _vec())
        # Re-setting to the same value must not raise
        await adapter.set_validity(f.id, valid=False, reason="same")
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.validity == "invalid"

    async def test_set_validity_bidirectional(self, adapter: AnyAdapter) -> None:
        f = _fact(content="flip back and forth", validity="valid")
        await adapter.upsert(f, _vec())
        await adapter.set_validity(f.id, valid=False, reason="step 1")
        await adapter.set_validity(f.id, valid=True, reason="step 2")
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.validity == "valid"

    async def test_set_validity_unknown_id_raises(self, adapter: AnyAdapter) -> None:
        with pytest.raises(KeyError):
            await adapter.set_validity(new_id(), valid=False, reason="unknown")


# ===========================================================================
# 12a — shelved lifecycle: new storage-seam capability (ADR 0021)
# ===========================================================================
#
# The engine's ``set_shelved`` verb (``engine/dispositions.py``) composes
# these EXISTING adapter primitives — ``transition`` + ``append_event`` —
# rather than adding a new Protocol method (unlike ``set_validity``, which
# has its own dedicated adapter method for its in-memory audit map). The
# "new seam capability" this section proves is therefore a new DATA VALUE
# flowing through those existing primitives on both adapters: the
# ``lifecycle="shelved"`` column value (``transition``/``get_by_id``) and
# the ``source="shelved"`` event value carrying a ``reason`` that survives a
# reopen (``append_event``/``query_events``) — not a new method signature.


class TestShelvedLifecycle:
    """``transition`` + ``append_event``/``query_events`` handle the new
    ``"shelved"`` lifecycle/event-source value on both adapters (ADR 0021).
    """

    async def test_transition_to_shelved_persists(self, adapter: AnyAdapter) -> None:
        f = _fact(content="shelve me", lifecycle="active")
        await adapter.upsert(f, _vec())
        await adapter.transition(f.id, "shelved", at=_ts(1))
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.lifecycle == "shelved"

    async def test_transition_archived_to_shelved_persists(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact(content="archived, then shelved", lifecycle="archived")
        await adapter.upsert(f, _vec())
        await adapter.transition(f.id, "shelved", at=_ts(1))
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.lifecycle == "shelved"

    async def test_transition_shelved_to_active_persists(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact(content="un-shelve me", lifecycle="shelved")
        await adapter.upsert(f, _vec())
        await adapter.transition(f.id, "active", at=_ts(1))
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.lifecycle == "active"

    async def test_shelved_event_carries_reason(self, adapter: AnyAdapter) -> None:
        """Same round-trip contract as ``TestEventLog::
        test_feedback_event_carries_reason`` — ``source="shelved"`` is the
        SECOND source that populates ``reason`` (ADR 0021)."""
        f = _fact(content="shelved reason target")
        await adapter.upsert(f, _vec())

        ev = ReinforcementEvent(
            target_id=f.id, at=EPOCH, source="shelved",
            reason="curator judged this not worth carrying",
        )
        await adapter.append_event(ev)

        results = await adapter.query_events(target_id=f.id)
        assert results[0].source == "shelved"
        assert results[0].reason == "curator judged this not worth carrying"

    async def test_shelved_reason_survives_reopen(
        self, adapter: AnyAdapter, clock: TestClock, tmp_path: Any
    ) -> None:
        """Acceptance criterion: readable back from the events table after a
        REOPEN of the store, not merely from the live adapter instance."""
        f = _fact(content="shelved reason reopen target")
        await adapter.upsert(f, _vec())
        await adapter.append_event(
            ReinforcementEvent(
                target_id=f.id, at=EPOCH, source="shelved",
                reason="reason must survive a reopen",
            )
        )
        await adapter.transition(f.id, "shelved", at=EPOCH)

        reopened = await _reopen_at_model(
            adapter, embedding_model="test-model", tmp_path=tmp_path, clock=clock
        )
        rec = await reopened.get_by_id(f.id)
        assert rec.lifecycle == "shelved"

        results = await reopened.query_events(target_id=f.id)
        shelved_events = [ev for ev in results if ev.source == "shelved"]
        assert len(shelved_events) == 1
        assert shelved_events[0].reason == "reason must survive a reopen"


# ===========================================================================
# 12b — Projection and validity write gate (F7 invariant)
# ===========================================================================


class TestProjectionWriteGate:
    """Re-upsert cannot bypass set_validity or update_projections (F7).

    validity → must go through set_validity (audited)
    recall_count / last_referenced_at / feedback_score / decay_due_at →
        must go through update_projections [A3]
    """

    async def test_validity_change_via_upsert_raises(self, adapter: AnyAdapter) -> None:
        f = _fact(content="validity gate test", validity="valid")
        await adapter.upsert(f, _vec())

        invalid_f = Fact(
            id=f.id,
            type="fact",
            content=f.content,
            scope=f.scope,
            lifecycle=f.lifecycle,
            validity="invalid",  # changed — must be rejected
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        with pytest.raises(ValueError, match="validity"):
            await adapter.upsert(invalid_f, _vec(seed=0.8))

        # Confirm the record is still valid
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.validity == "valid"

    async def test_recall_count_change_via_upsert_raises(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact(content="recall_count gate test", recall_count=0)
        await adapter.upsert(f, _vec())

        bumped = Fact(
            id=f.id,
            type="fact",
            content=f.content,
            scope=f.scope,
            lifecycle=f.lifecycle,
            recall_count=5,  # changed — must be rejected
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        with pytest.raises(ValueError, match="recall_count"):
            await adapter.upsert(bumped, _vec(seed=0.8))

        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.recall_count == 0

    async def test_feedback_score_change_via_upsert_raises(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact(content="feedback_score gate test", feedback_score=0.0)
        await adapter.upsert(f, _vec())

        patched = Fact(
            id=f.id,
            type="fact",
            content=f.content,
            scope=f.scope,
            lifecycle=f.lifecycle,
            feedback_score=0.8,  # changed — must be rejected
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        with pytest.raises(ValueError, match="feedback_score"):
            await adapter.upsert(patched, _vec(seed=0.8))

        retrieved = await adapter.get_by_id(f.id)
        assert isinstance(retrieved, Fact)
        assert retrieved.feedback_score == pytest.approx(0.0)

    async def test_decay_due_at_change_via_upsert_raises(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact(content="decay_due_at gate test", decay_due_at=None)
        await adapter.upsert(f, _vec())

        patched = Fact(
            id=f.id,
            type="fact",
            content=f.content,
            scope=f.scope,
            lifecycle=f.lifecycle,
            decay_due_at=_ts(30),  # changed — must be rejected
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        with pytest.raises(ValueError, match="decay_due_at"):
            await adapter.upsert(patched, _vec(seed=0.8))

        retrieved = await adapter.get_by_id(f.id)
        assert isinstance(retrieved, Fact)
        assert retrieved.decay_due_at is None

    async def test_update_projections_is_the_correct_path(
        self, adapter: AnyAdapter
    ) -> None:
        """update_projections succeeds where re-upsert would fail."""
        f = _fact(content="projections via correct path")
        await adapter.upsert(f, _vec())

        await adapter.update_projections(
            f.id,
            recall_count=3,
            last_referenced_at=_ts(5),
            feedback_score=0.6,
            decay_due_at=_ts(45),
        )
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.recall_count == 3
        assert isinstance(retrieved, Fact)
        assert retrieved.feedback_score == pytest.approx(0.6)


# ===========================================================================
# 12c — transition sets updated_at; set_validity and update_projections do not
# ===========================================================================


class TestTransitionUpdatesUpdatedAt:
    """transition persists updated_at=at; set_validity and update_projections leave it unchanged."""

    async def test_transition_sets_updated_at(self, adapter: AnyAdapter, clock: TestClock) -> None:
        """transition(at=T) must persist updated_at=T on the transitioned record."""
        f = _fact(content="lifecycle transition timestamp test")
        await adapter.upsert(f, _vec())
        original_updated_at = f.updated_at

        # Advance clock so the transition instant differs from creation
        transition_time = original_updated_at + timedelta(days=5)

        await adapter.transition(f.id, "archived", at=transition_time)
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.updated_at == transition_time
        assert retrieved.updated_at != original_updated_at

    async def test_supersession_sets_updated_at_on_old_record(
        self, adapter: AnyAdapter
    ) -> None:
        """Supersession: the OLD record gets updated_at=at; the new record keeps its own."""
        old_d = _decision(content="old decision")
        new_d = _decision(content="new decision superseder")
        await adapter.upsert(old_d, _vec())
        await adapter.upsert(new_d, _vec(seed=0.9))

        transition_time = EPOCH + timedelta(days=3)
        await adapter.replace(new_d, None, [_flip(old_d.id)], at=transition_time)

        old_retrieved = await adapter.get_by_id(old_d.id)
        new_retrieved = await adapter.get_by_id(new_d.id)

        assert old_retrieved.updated_at == transition_time
        # The new record was upserted with its own updated_at; transition does not touch it
        assert new_retrieved.updated_at == new_d.updated_at

    async def test_set_validity_does_not_touch_updated_at(self, adapter: AnyAdapter) -> None:
        """set_validity must NOT modify updated_at (validity is metadata, not a semantic edit)."""
        f = _fact(content="validity does not change updated_at")
        await adapter.upsert(f, _vec())
        original_updated_at = f.updated_at

        await adapter.set_validity(f.id, valid=False, reason="test")
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.updated_at == original_updated_at

    async def test_update_projections_does_not_touch_updated_at(
        self, adapter: AnyAdapter
    ) -> None:
        """update_projections must NOT modify updated_at (projections are not semantic)."""
        f = _fact(content="projections do not change updated_at")
        await adapter.upsert(f, _vec())
        original_updated_at = f.updated_at

        await adapter.update_projections(
            f.id,
            recall_count=5,
            last_referenced_at=_ts(10),
            feedback_score=0.7,
            decay_due_at=_ts(45),
        )
        retrieved = await adapter.get_by_id(f.id)
        assert retrieved.updated_at == original_updated_at


# ===========================================================================
# 13 — scan
# ===========================================================================


class TestScan:
    """scan yields RowMeta in insertion order; lifecycle_in filters work."""

    async def test_scan_yields_all_rows(self, adapter: AnyAdapter) -> None:
        facts = [_fact(content=f"scan fact {i}") for i in range(3)]
        for f in facts:
            await adapter.upsert(f, _vec())

        collected: list[RowMeta] = []
        async for meta in await adapter.scan():
            collected.append(meta)

        assert len(collected) == 3
        collected_ids = {m.id for m in collected}
        assert {f.id for f in facts} == collected_ids

    async def test_scan_insertion_order_stable(self, adapter: AnyAdapter) -> None:
        facts = [_fact(content=f"order fact {i}") for i in range(4)]
        for f in facts:
            await adapter.upsert(f, _vec())

        collected: list[RowMeta] = []
        async for meta in await adapter.scan():
            collected.append(meta)

        assert [m.id for m in collected] == [f.id for f in facts]

    async def test_scan_lifecycle_filter(self, adapter: AnyAdapter) -> None:
        active = _fact(content="active scan", lifecycle="active")
        archived = _fact(content="archived scan", lifecycle="archived")
        await adapter.upsert(active, _vec())
        await adapter.upsert(archived, _vec(seed=0.9))

        active_metas: list[RowMeta] = []
        async for meta in await adapter.scan(lifecycle_in=frozenset({"active"})):
            active_metas.append(meta)

        ids = {m.id for m in active_metas}
        assert active.id in ids
        assert archived.id not in ids

    async def test_scan_excludes_buried(self, adapter: AnyAdapter) -> None:
        f = _fact(content="buried scan target")
        await adapter.upsert(f, _vec())
        await adapter.bury(f.id, "reason")

        collected: list[RowMeta] = []
        async for meta in await adapter.scan():
            collected.append(meta)

        assert all(m.id != f.id for m in collected)

    async def test_scan_meta_has_no_content(self, adapter: AnyAdapter) -> None:
        f = _fact(content="should not appear in RowMeta")
        await adapter.upsert(f, _vec())

        async for meta in await adapter.scan():
            assert not hasattr(meta, "content")


# ===========================================================================
# 13b — scan_records: the content-carrying sibling of scan (
# ``findings-context-latency.md``). Same lifecycle/exclusion contract as
# ``scan``, but yields the full domain record so a caller never needs a
# separate per-candidate ``get_by_id`` round trip.
# ===========================================================================


class TestScanRecords:
    """scan_records yields full Fact | Decision records; buried rows
    excluded; lifecycle filtering works — the same coverage TestScan gives
    ``scan``, plus the content-equivalence property that is scan_records'
    whole reason to exist."""

    async def test_scan_records_yields_all_rows(self, adapter: AnyAdapter) -> None:
        facts = [_fact(content=f"scan_records fact {i}") for i in range(3)]
        for f in facts:
            await adapter.upsert(f, _vec())

        collected: list[Fact | Decision] = []
        async for rec in await adapter.scan_records():
            collected.append(rec)

        assert len(collected) == 3
        assert {r.id for r in collected} == {f.id for f in facts}

    async def test_scan_records_lifecycle_filter(self, adapter: AnyAdapter) -> None:
        active = _fact(content="active scan_records", lifecycle="active")
        archived = _fact(content="archived scan_records", lifecycle="archived")
        await adapter.upsert(active, _vec())
        await adapter.upsert(archived, _vec(seed=0.9))

        collected: list[Fact | Decision] = []
        async for rec in await adapter.scan_records(lifecycle_in=frozenset({"active"})):
            collected.append(rec)

        ids = {r.id for r in collected}
        assert active.id in ids
        assert archived.id not in ids

    async def test_scan_records_excludes_buried(self, adapter: AnyAdapter) -> None:
        f = _fact(content="buried scan_records target")
        await adapter.upsert(f, _vec())
        await adapter.bury(f.id, "reason")

        collected: list[Fact | Decision] = []
        async for rec in await adapter.scan_records():
            collected.append(rec)

        assert all(r.id != f.id for r in collected)

    async def test_scan_records_content_matches_get_by_id(self, adapter: AnyAdapter) -> None:
        """The property scan_records exists for: what it yields must be the
        SAME content get_by_id would return for the same id -- proving the
        collapse (building the record from the already-fetched scan row)
        is not a second, divergent code path."""
        f = _fact(content="scan_records must match get_by_id exactly")
        await adapter.upsert(f, _vec())

        scanned = [rec async for rec in await adapter.scan_records()]
        assert len(scanned) == 1
        via_get_by_id = await adapter.get_by_id(f.id)

        assert scanned[0] == via_get_by_id

    async def test_scan_records_yields_decisions_too(self, adapter: AnyAdapter) -> None:
        d = _decision(content="scan_records decision content")
        await adapter.upsert(d, _vec())

        collected = [rec async for rec in await adapter.scan_records()]
        assert len(collected) == 1
        assert isinstance(collected[0], Decision)
        assert collected[0].content == "scan_records decision content"


# ===========================================================================
# 13c — Sessions: the parallel seam, on BOTH adapters (2026-08-02 sessions
# plan). This class is the dual-adapter conformance pass, so the
# LanceDB mirror is never left with zero automated verification.
#
# ``search_sessions``'s text leg is case-insensitive substring on
# ``MemoryAdapter`` and BM25 on ``LanceDBAdapter`` — assertions below check
# presence/absence of specific rows, never relative score or exact
# relevance ordering across the two strategies (SPEC).
# ===========================================================================


class TestSessions:
    """append_session / latest_session_end / search_sessions on both
    ``memory`` and ``lancedb`` parameterizations of the shared ``adapter``
    fixture."""

    async def test_append_then_read_back(self, adapter: AnyAdapter) -> None:
        session = _session(content="landed the sessions seam")
        await adapter.append_session(session)

        results = await adapter.search_sessions(
            scopes=["proj-a"], query=None, since=None, until=None, limit=10
        )
        assert [s.id for s in results] == [session.id]

    async def test_scope_isolation(self, adapter: AnyAdapter) -> None:
        session = _session(scope="a")
        await adapter.append_session(session)

        other_scope = await adapter.search_sessions(
            scopes=["b"], query=None, since=None, until=None, limit=10
        )
        assert other_scope == []

        own_scope = await adapter.search_sessions(
            scopes=["a"], query=None, since=None, until=None, limit=10
        )
        assert [s.id for s in own_scope] == [session.id]

    async def test_scope_plus_global_union(self, adapter: AnyAdapter) -> None:
        session = _session(scope="a")
        await adapter.append_session(session)

        results = await adapter.search_sessions(
            scopes=["a", "global"], query=None, since=None, until=None, limit=10
        )
        assert [s.id for s in results] == [session.id]

    async def test_range_filter_excludes_outside_range(self, adapter: AnyAdapter) -> None:
        session = _session(started_at=EPOCH, ended_at=EPOCH)
        await adapter.append_session(session)

        results = await adapter.search_sessions(
            scopes=["proj-a"],
            query=None,
            since=EPOCH + timedelta(days=1),
            until=EPOCH + timedelta(days=2),
            limit=10,
        )
        assert results == []

    async def test_range_filter_since_boundary_inclusive(self, adapter: AnyAdapter) -> None:
        session = _session(started_at=EPOCH, ended_at=EPOCH)
        await adapter.append_session(session)

        results = await adapter.search_sessions(
            scopes=["proj-a"], query=None, since=EPOCH, until=None, limit=10
        )
        assert [s.id for s in results] == [session.id]

    async def test_range_filter_until_boundary_inclusive(self, adapter: AnyAdapter) -> None:
        session = _session(started_at=EPOCH, ended_at=EPOCH)
        await adapter.append_session(session)

        results = await adapter.search_sessions(
            scopes=["proj-a"], query=None, since=None, until=EPOCH, limit=10
        )
        assert [s.id for s in results] == [session.id]

    async def test_text_filter_matching_and_nonmatching(self, adapter: AnyAdapter) -> None:
        """Presence + absence only — never relative score or ordering, which
        would either pin LanceDB's BM25 internals or force the substring
        matcher to fake a ranking it does not compute."""
        matching = _session(content="landed the sessions seam")
        nonmatching = _session(content="totally unrelated content entirely")
        await adapter.append_session(matching)
        await adapter.append_session(nonmatching)

        results = await adapter.search_sessions(
            scopes=["proj-a"], query="sessions", since=None, until=None, limit=10
        )
        ids = {s.id for s in results}
        assert matching.id in ids
        assert nonmatching.id not in ids

    async def test_combined_text_and_range(self, adapter: AnyAdapter) -> None:
        matches_both = _session(
            started_at=EPOCH, ended_at=EPOCH, content="landed the sessions seam"
        )
        matches_text_only = _session(
            started_at=EPOCH + timedelta(days=5),
            ended_at=EPOCH + timedelta(days=5),
            content="landed the sessions seam",
        )
        matches_range_only = _session(
            started_at=EPOCH, ended_at=EPOCH, content="totally unrelated content"
        )
        await adapter.append_session(matches_both)
        await adapter.append_session(matches_text_only)
        await adapter.append_session(matches_range_only)

        results = await adapter.search_sessions(
            scopes=["proj-a"],
            query="sessions",
            since=EPOCH,
            until=EPOCH,
            limit=10,
        )
        assert {s.id for s in results} == {matches_both.id}

    async def test_latest_session_end_none_for_empty_scope(
        self, adapter: AnyAdapter
    ) -> None:
        assert await adapter.latest_session_end(scope="empty") is None

    async def test_latest_session_end_per_scope_maximum(self, adapter: AnyAdapter) -> None:
        earlier = _session(scope="a", started_at=EPOCH, ended_at=EPOCH)
        later = _session(
            scope="a",
            started_at=EPOCH + timedelta(days=1),
            ended_at=EPOCH + timedelta(days=1),
        )
        other_scope_later = _session(
            scope="b",
            started_at=EPOCH + timedelta(days=5),
            ended_at=EPOCH + timedelta(days=5),
        )
        await adapter.append_session(earlier)
        await adapter.append_session(later)
        await adapter.append_session(other_scope_later)

        result = await adapter.latest_session_end(scope="a")
        assert result == later.ended_at

    async def test_limit_is_honoured(self, adapter: AnyAdapter) -> None:
        for i in range(5):
            await adapter.append_session(
                _session(
                    started_at=EPOCH + timedelta(days=i),
                    ended_at=EPOCH + timedelta(days=i),
                )
            )

        results = await adapter.search_sessions(
            scopes=["proj-a"], query=None, since=None, until=None, limit=2
        )
        assert len(results) == 2

    async def test_no_query_order_is_ended_at_descending(self, adapter: AnyAdapter) -> None:
        sessions = [
            _session(
                started_at=EPOCH + timedelta(days=i),
                ended_at=EPOCH + timedelta(days=i),
            )
            for i in range(4)
        ]
        # Append out of chronological order to prove the result is sorted,
        # not merely reflecting insertion order.
        for session in [sessions[2], sessions[0], sessions[3], sessions[1]]:
            await adapter.append_session(session)

        results = await adapter.search_sessions(
            scopes=["proj-a"], query=None, since=None, until=None, limit=10
        )
        assert [s.id for s in results] == [s.id for s in reversed(sessions)]

    async def test_structural_exclusion_from_vector_search(
        self, adapter: AnyAdapter
    ) -> None:
        fact = _fact(content="the only vector-searchable row")
        v = _vec()
        await adapter.upsert(fact, v)
        await adapter.append_session(_session(content="a private session digest"))

        hits = await adapter.vector_search(v, filters=_all_filters(), k=50)
        assert len(hits) == 1
        assert hits[0].record.id == fact.id

    async def test_structural_exclusion_from_text_search(self, adapter: AnyAdapter) -> None:
        fact = _fact(content="digest sessions plan")
        await adapter.upsert(fact, _vec())
        await adapter.append_session(_session(content="digest sessions plan"))

        hits = await adapter.text_search(
            "digest sessions plan", filters=_all_filters(), k=50
        )
        assert len(hits) == 1
        assert hits[0].record.id == fact.id

    async def test_structural_exclusion_from_scan(self, adapter: AnyAdapter) -> None:
        fact = _fact()
        await adapter.upsert(fact, _vec())
        session = _session()
        await adapter.append_session(session)

        rows: list[RowMeta] = [row async for row in await adapter.scan()]
        assert len(rows) == 1
        assert rows[0].id == fact.id
        assert all(row.id != session.id for row in rows)

    async def test_structural_exclusion_from_scan_records(
        self, adapter: AnyAdapter
    ) -> None:
        fact = _fact()
        await adapter.upsert(fact, _vec())
        session = _session()
        await adapter.append_session(session)

        records: list[Fact | Decision] = [
            rec async for rec in await adapter.scan_records()
        ]
        assert len(records) == 1
        assert records[0].id == fact.id
        assert all(rec.id != session.id for rec in records)

    async def test_round_trip_fidelity_all_fields(self, adapter: AnyAdapter) -> None:
        """Every field of an appended Session comes back equal, including
        ``source=None`` and non-UTC input on both timestamps (both backends
        must land on the same instant, whatever tzinfo the caller handed
        in)."""
        non_utc = timezone(timedelta(hours=-5))
        started = datetime(2025, 6, 1, 9, 0, 0, tzinfo=non_utc)
        ended = datetime(2025, 6, 1, 15, 30, 0, tzinfo=UTC)
        session = _session(
            scope="proj-a",
            started_at=started,
            ended_at=ended,
            content="round trip fidelity content",
        )
        await adapter.append_session(session)

        results = await adapter.search_sessions(
            scopes=["proj-a"], query=None, since=None, until=None, limit=10
        )
        assert len(results) == 1
        retrieved = results[0]
        assert retrieved.id == session.id
        assert retrieved.scope == session.scope
        assert retrieved.started_at == session.started_at
        assert retrieved.ended_at == session.ended_at
        assert retrieved.content == session.content
        assert retrieved.source is None


# ===========================================================================
# 13d — Cleaner state: get_cleaner_state / set_cleaner_state, on BOTH
# adapters (2026-08-10 cleaner plan). Missing scope means "never
# judged" — never a default; upsert leaves one row.
# ===========================================================================


class TestCleanerState:
    async def test_unknown_scope_returns_none(self, adapter: AnyAdapter) -> None:
        assert await adapter.get_cleaner_state(scope=Scope("proj-a")) is None

    async def test_round_trip(self, adapter: AnyAdapter) -> None:
        state = CleanerState(watermark_at=EPOCH, last_pass_at=EPOCH)
        await adapter.set_cleaner_state(scope=Scope("proj-a"), state=state)

        got = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert got == state

    async def test_round_trip_with_report(self, adapter: AnyAdapter) -> None:
        state = CleanerState(
            watermark_at=EPOCH,
            last_pass_at=EPOCH + timedelta(hours=1),
            report='{"kept": 5, "shelved": 1}',
        )
        await adapter.set_cleaner_state(scope=Scope("proj-a"), state=state)

        got = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert got == state

    async def test_set_is_upsert_not_append(self, adapter: AnyAdapter) -> None:
        """Break it deliberately (task verify step): comment out
        ``.when_not_matched_insert_all()`` on the LanceDB side, or make
        ``MemoryAdapter.set_cleaner_state`` append to a list instead of
        assigning a dict key — this test goes red because the SECOND
        ``get_cleaner_state`` no longer equals ``second`` (a naive
        first-row-wins read would still return ``first``), and the
        LanceDB-side physical-row-count assertion below fails outright.
        """
        first = CleanerState(watermark_at=EPOCH, last_pass_at=EPOCH)
        second = CleanerState(
            watermark_at=EPOCH + timedelta(days=1),
            last_pass_at=EPOCH + timedelta(days=1),
            report="{}",
        )
        await adapter.set_cleaner_state(scope=Scope("proj-a"), state=first)
        await adapter.set_cleaner_state(scope=Scope("proj-a"), state=second)

        got = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert got == second

        if isinstance(adapter, LanceDBAdapter):
            rows = await raw_table(adapter._cleaner_state).query().to_list()
            assert len(rows) == 1
        else:
            assert len(adapter._cleaner_state) == 1

    async def test_per_scope_isolation(self, adapter: AnyAdapter) -> None:
        state_a = CleanerState(watermark_at=EPOCH, last_pass_at=EPOCH)
        await adapter.set_cleaner_state(scope=Scope("proj-a"), state=state_a)

        assert await adapter.get_cleaner_state(scope=Scope("proj-b")) is None
        got_a = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert got_a == state_a


# ===========================================================================
# 13e — list_born_since: the pool for one Cleaner pass (2026-08-10 cleaner
# plan). Strictly-after, one scope only, oldest-first, limited.
# ===========================================================================


def _fact_born(
    *,
    scope: str = "proj-a",
    lifecycle: str = "active",
    validity: str = "valid",
    created_at: datetime,
    content: str = "born fact",
    recall_count: int = 0,
    last_referenced_at: datetime | None = None,
) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=created_at,
        updated_at=created_at,
        recall_count=recall_count,
        last_referenced_at=last_referenced_at,
    )


def _decision_born(
    *,
    scope: str = "proj-a",
    lifecycle: str = "in_force",
    validity: str = "valid",
    created_at: datetime,
    content: str = "born decision",
) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=created_at,
        updated_at=created_at,
    )


class TestListBornSince:
    async def test_excludes_at_and_before_since(self, adapter: AnyAdapter) -> None:
        at_since = _fact_born(created_at=EPOCH, content="at since")
        before = _fact_born(created_at=EPOCH - timedelta(days=1), content="before")
        after = _fact_born(created_at=EPOCH + timedelta(days=1), content="after")
        for f in (at_since, before, after):
            await adapter.upsert(f, _vec())

        results = await adapter.list_born_since(
            scope=Scope("proj-a"), since=EPOCH, limit=10
        )
        assert [r.id for r in results] == [after.id]

    async def test_scope_isolation_no_global_leak(self, adapter: AnyAdapter) -> None:
        """A call for scope X must never return a ``global`` record — scope
        spans are the caller's job, not this method's (acceptance
        criterion)."""
        in_scope = _fact_born(scope="proj-a", created_at=EPOCH + timedelta(days=1))
        other_scope = _fact_born(scope="proj-b", created_at=EPOCH + timedelta(days=1))
        global_scope = _fact_born(scope="global", created_at=EPOCH + timedelta(days=1))
        for f in (in_scope, other_scope, global_scope):
            await adapter.upsert(f, _vec())

        results = await adapter.list_born_since(
            scope=Scope("proj-a"), since=EPOCH, limit=10
        )
        assert [r.id for r in results] == [in_scope.id]

    async def test_ordered_oldest_first(self, adapter: AnyAdapter) -> None:
        oldest = _fact_born(created_at=EPOCH + timedelta(days=1), content="oldest")
        newest = _fact_born(created_at=EPOCH + timedelta(days=3), content="newest")
        middle = _fact_born(created_at=EPOCH + timedelta(days=2), content="middle")
        # Insert out of chronological order to prove the result is sorted,
        # not merely reflecting insertion order.
        for f in (newest, oldest, middle):
            await adapter.upsert(f, _vec())

        results = await adapter.list_born_since(
            scope=Scope("proj-a"), since=EPOCH, limit=10
        )
        assert [r.id for r in results] == [oldest.id, middle.id, newest.id]

    async def test_limit_is_honoured(self, adapter: AnyAdapter) -> None:
        facts = [
            _fact_born(created_at=EPOCH + timedelta(days=i + 1), content=f"f{i}")
            for i in range(5)
        ]
        for f in facts:
            await adapter.upsert(f, _vec())

        results = await adapter.list_born_since(
            scope=Scope("proj-a"), since=EPOCH, limit=2
        )
        assert len(results) == 2
        # limit + oldest-first compose: the two returned are the two oldest.
        assert [r.id for r in results] == [facts[0].id, facts[1].id]

    async def test_returns_both_facts_and_decisions(self, adapter: AnyAdapter) -> None:
        fact = _fact_born(created_at=EPOCH + timedelta(days=1))
        decision = _decision_born(created_at=EPOCH + timedelta(days=1))
        await adapter.upsert(fact, _vec())
        await adapter.upsert(decision, _vec(seed=0.9))

        results = await adapter.list_born_since(
            scope=Scope("proj-a"), since=EPOCH, limit=10
        )
        ids = {r.id for r in results}
        assert fact.id in ids
        assert decision.id in ids

    async def test_excludes_buried_records(self, adapter: AnyAdapter) -> None:
        f = _fact_born(created_at=EPOCH + timedelta(days=1))
        await adapter.upsert(f, _vec())
        await adapter.bury(f.id, "no longer needed")

        results = await adapter.list_born_since(
            scope=Scope("proj-a"), since=EPOCH, limit=10
        )
        assert results == []


# ===========================================================================
# 13f — Pass history: append_cleaner_pass / list_cleaner_passes, on BOTH
# adapters (cleaner-telemetry change). Append-only, beside
# ``cleaner_state`` — appending twice for one scope MUST leave two rows,
# never one (the property ``cleaner_state``'s upsert violates and the whole
# reason this table exists).
# ===========================================================================


def _pass_record(
    *,
    scope: str = "proj-a",
    pass_at: datetime,
    outcome: str = "completed",
    dry_run: bool = False,
    payload: str = "{}",
) -> CleanerPassRecord:
    return CleanerPassRecord(
        scope=Scope(scope),
        pass_at=pass_at,
        outcome=outcome,  # type: ignore[arg-type]
        dry_run=dry_run,
        payload=payload,
    )


class TestCleanerPassHistory:
    async def test_unknown_scope_returns_empty_list_never_an_error(
        self, adapter: AnyAdapter
    ) -> None:
        assert await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10) == []

    async def test_append_then_list_round_trips(self, adapter: AnyAdapter) -> None:
        record = _pass_record(pass_at=EPOCH, payload='{"pool_size": 3}')
        await adapter.append_cleaner_pass(record=record)

        got = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert got == [record]

    async def test_append_twice_leaves_two_rows_not_one(self, adapter: AnyAdapter) -> None:
        """Falsifiability (task verify step): change ``append_cleaner_pass`` to a
        ``merge_insert`` keyed on ``scope`` — the ``cleaner_state`` shape — and this test goes
        red: the second append would silently replace the first instead of adding a second row,
        so ``len(got)`` comes back 1, not 2. This is the property the whole table exists to
        guarantee, in contrast to ``cleaner_state.set_cleaner_state``'s deliberate upsert.
        """
        first = _pass_record(pass_at=EPOCH, outcome="completed")
        second = _pass_record(pass_at=EPOCH + timedelta(days=1), outcome="completed")
        await adapter.append_cleaner_pass(record=first)
        await adapter.append_cleaner_pass(record=second)

        got = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(got) == 2
        assert {r.pass_at for r in got} == {first.pass_at, second.pass_at}

    async def test_newest_first_ordering(self, adapter: AnyAdapter) -> None:
        oldest = _pass_record(pass_at=EPOCH, payload="oldest")
        newest = _pass_record(pass_at=EPOCH + timedelta(days=2), payload="newest")
        middle = _pass_record(pass_at=EPOCH + timedelta(days=1), payload="middle")
        # Appended out of chronological order to prove the result is SORTED, not merely
        # reflecting insertion order.
        for r in (middle, oldest, newest):
            await adapter.append_cleaner_pass(record=r)

        got = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert [r.payload for r in got] == ["newest", "middle", "oldest"]

    async def test_limit_is_honoured(self, adapter: AnyAdapter) -> None:
        for i in range(5):
            await adapter.append_cleaner_pass(
                record=_pass_record(pass_at=EPOCH + timedelta(days=i), payload=str(i))
            )

        got = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=2)
        assert len(got) == 2
        # limit + newest-first compose: the two returned are the two newest.
        assert [r.payload for r in got] == ["4", "3"]

    async def test_scope_isolation(self, adapter: AnyAdapter) -> None:
        await adapter.append_cleaner_pass(
            record=_pass_record(scope="proj-a", pass_at=EPOCH, payload="a")
        )
        await adapter.append_cleaner_pass(
            record=_pass_record(scope="proj-b", pass_at=EPOCH, payload="b")
        )

        got_a = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert [r.payload for r in got_a] == ["a"]
        got_b = await adapter.list_cleaner_passes(scope=Scope("proj-b"), limit=10)
        assert [r.payload for r in got_b] == ["b"]

    async def test_dry_run_flag_round_trips(self, adapter: AnyAdapter) -> None:
        await adapter.append_cleaner_pass(
            record=_pass_record(pass_at=EPOCH, dry_run=True, payload="dry")
        )
        got = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert got[0].dry_run is True

    async def test_all_five_outcomes_round_trip(self, adapter: AnyAdapter) -> None:
        """The closed set is defined whole now even though only ``completed`` writes so far
        (task brief) — every value must persist and read back cleanly, on both adapters."""
        outcomes = ["initialized", "empty_pool", "completed", "blocked", "errored"]
        for i, outcome in enumerate(outcomes):
            await adapter.append_cleaner_pass(
                record=_pass_record(
                    pass_at=EPOCH + timedelta(days=i), outcome=outcome, payload=outcome
                )
            )
        got = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert {r.outcome for r in got} == set(outcomes)


# ===========================================================================
# 13g — Record reviews: append_review / list_due_for_review, on BOTH adapters
# (2026-08-12 cleaner-revisit-pass plan, scope 1). Append-only, beside
# ``pass_history`` — appending twice for one target MUST leave two rows,
# never one; ``list_due_for_review`` reduces to "the latest one governs" at
# READ time.
# ===========================================================================


def _review(
    *,
    target_id: HypomnemaId,
    scope: str = "proj-a",
    reviewed_at: datetime,
    next_review_due_at: datetime,
    outcome: str = "judged",
    band: str | None = "keep",
    grades: tuple[str, ...] = ("keep-clear", "keep-lean"),
    rubric_version: str = "cleaner-rubric-v1-test",
) -> RecordReview:
    return RecordReview(
        target_id=target_id,
        scope=Scope(scope),
        reviewed_at=reviewed_at,
        next_review_due_at=next_review_due_at,
        outcome=outcome,  # type: ignore[arg-type]
        band=band,  # type: ignore[arg-type]
        grades=grades,
        rubric_version=rubric_version,
    )


def _judgment(
    *,
    target_id: HypomnemaId,
    scope: str = "proj-a",
    judged_at: datetime,
    rubric_version: str = "cleaner-rubric-v1-test",
    judge_a_identity: str = "claude:opus",
    judge_a_grade: str = "shelve-clear",
    judge_a_why: str = "derivable from the shipped code",
    judge_b_identity: str = "codex:gpt",
    judge_b_grade: str = "keep-clear",
    judge_b_why: str = "still the only record of a defect nobody has fixed",
    paired_with: HypomnemaId | None = None,
    similarity: float | None = None,
) -> RecordJudgment:
    return RecordJudgment(
        target_id=target_id,
        scope=Scope(scope),
        judged_at=judged_at,
        rubric_version=rubric_version,
        judge_a_identity=judge_a_identity,
        judge_a_grade=judge_a_grade,
        judge_a_why=judge_a_why,
        judge_b_identity=judge_b_identity,
        judge_b_grade=judge_b_grade,
        judge_b_why=judge_b_why,
        paired_with=paired_with,
        similarity=similarity,
    )


class TestRecordReviews:
    async def test_never_reviewed_fact_older_than_created_before_is_returned(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact_born(created_at=EPOCH)
        await adapter.upsert(f, _vec())

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=1),
            limit=10,
        )
        assert [r.id for r in got] == [f.id]

    async def test_fact_newer_than_created_before_is_excluded(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact_born(created_at=EPOCH + timedelta(days=10))
        await adapter.upsert(f, _vec())

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=1),
            limit=10,
        )
        assert got == []

    async def test_future_next_review_due_at_is_not_returned(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact_born(created_at=EPOCH)
        await adapter.upsert(f, _vec())
        await adapter.append_review(
            _review(
                target_id=f.id,
                reviewed_at=EPOCH + timedelta(days=5),
                next_review_due_at=EPOCH + timedelta(days=95),
            )
        )

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=10),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert got == []

    async def test_past_next_review_due_at_is_returned(self, adapter: AnyAdapter) -> None:
        f = _fact_born(created_at=EPOCH)
        await adapter.upsert(f, _vec())
        await adapter.append_review(
            _review(
                target_id=f.id,
                reviewed_at=EPOCH + timedelta(days=5),
                next_review_due_at=EPOCH + timedelta(days=15),
            )
        )

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=20),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert [r.id for r in got] == [f.id]

    async def test_due_at_equal_to_now_is_returned(self, adapter: AnyAdapter) -> None:
        """The predicate is ``<=``, not ``<`` — a review due exactly now is due."""
        f = _fact_born(created_at=EPOCH)
        await adapter.upsert(f, _vec())
        now = EPOCH + timedelta(days=20)
        await adapter.append_review(
            _review(target_id=f.id, reviewed_at=EPOCH + timedelta(days=5), next_review_due_at=now)
        )

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"), now=now, created_before=EPOCH + timedelta(days=365), limit=10
        )
        assert [r.id for r in got] == [f.id]

    async def test_ordering_oldest_reviewed_first_never_reviewed_first(
        self, adapter: AnyAdapter
    ) -> None:
        never_reviewed = _fact_born(created_at=EPOCH, content="never")
        reviewed_long_ago = _fact_born(created_at=EPOCH, content="long ago")
        reviewed_recently = _fact_born(created_at=EPOCH, content="recent")
        for f in (reviewed_recently, never_reviewed, reviewed_long_ago):
            await adapter.upsert(f, _vec())

        await adapter.append_review(
            _review(
                target_id=reviewed_long_ago.id,
                reviewed_at=EPOCH + timedelta(days=1),
                next_review_due_at=EPOCH + timedelta(days=2),
            )
        )
        await adapter.append_review(
            _review(
                target_id=reviewed_recently.id,
                reviewed_at=EPOCH + timedelta(days=8),
                next_review_due_at=EPOCH + timedelta(days=9),
            )
        )

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert [r.id for r in got] == [
            never_reviewed.id,
            reviewed_long_ago.id,
            reviewed_recently.id,
        ]

    async def test_limit_caps_the_result(self, adapter: AnyAdapter) -> None:
        facts = [_fact_born(created_at=EPOCH, content=f"f{i}") for i in range(5)]
        for f in facts:
            await adapter.upsert(f, _vec())

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=365),
            limit=2,
        )
        assert len(got) == 2

    async def test_hot_ember_shelved_facts_are_excluded(self, adapter: AnyAdapter) -> None:
        for i, lifecycle in enumerate(("hot", "ember", "shelved")):
            f = Fact(
                id=new_id(),
                type="fact",
                content=f"lc-{lifecycle}",
                scope=Scope("proj-a"),
                lifecycle=lifecycle,  # type: ignore[arg-type]
                validity="valid",
                created_at=EPOCH,
                updated_at=EPOCH,
            )
            await adapter.upsert(f, _vec(seed=1.0 + i))

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert got == []

    async def test_decisions_are_excluded(self, adapter: AnyAdapter) -> None:
        d = _decision_born(created_at=EPOCH)
        await adapter.upsert(d, _vec())

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert got == []

    async def test_invalid_fact_is_excluded_even_with_active_lifecycle(
        self, adapter: AnyAdapter
    ) -> None:
        """Decision 17 leaves invalid Facts alone -- an invalid, still-``active``-lifecycle Fact
        must never reach the revisit pool, or it could be judged and shelved, pulling it out of
        the dedup probe and breaking the non-blocking reconciliation link."""
        f = _fact_born(created_at=EPOCH)
        invalid_f = Fact.model_validate({**f.model_dump(mode="python"), "validity": "invalid"})
        await adapter.upsert(invalid_f, _vec())

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert got == []

    async def test_superseded_fact_is_excluded_even_with_active_lifecycle(
        self, adapter: AnyAdapter
    ) -> None:
        """Same finding as above -- ``replace`` preserves the predecessor's own
        ``lifecycle`` (never forces it to ``archived``), so a superseded Fact can be
        ``lifecycle="active"`` with ``superseded_by`` set; it must be excluded on that field
        alone, independent of validity."""
        f = _fact_born(created_at=EPOCH)
        successor_id = new_id()
        superseded_f = Fact.model_validate(
            {**f.model_dump(mode="python"), "validity": "invalid", "superseded_by": successor_id}
        )
        await adapter.upsert(superseded_f, _vec())

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert got == []

    async def test_buried_record_is_excluded(self, adapter: AnyAdapter) -> None:
        f = _fact_born(created_at=EPOCH)
        await adapter.upsert(f, _vec())
        await adapter.bury(f.id, "no longer needed")

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert got == []

    async def test_scope_isolation(self, adapter: AnyAdapter) -> None:
        in_scope = _fact_born(scope="proj-a", created_at=EPOCH)
        other_scope = _fact_born(scope="proj-b", created_at=EPOCH)
        await adapter.upsert(in_scope, _vec())
        await adapter.upsert(other_scope, _vec(seed=0.5))

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=100),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert [r.id for r in got] == [in_scope.id]

    async def test_appending_two_reviews_keeps_both_rows_and_latest_governs(
        self, adapter: AnyAdapter
    ) -> None:
        """Falsifiability (task verify step): change ``append_review`` to a ``merge_insert`` keyed
        on ``target_id`` and this test goes red — the second review would silently replace the
        first physical row instead of adding a second one, and the LanceDB-side row-count
        assertion below fails outright.
        """
        f = _fact_born(created_at=EPOCH)
        await adapter.upsert(f, _vec())

        # First review: due soon (would be selected if it were the only/latest one).
        await adapter.append_review(
            _review(
                target_id=f.id,
                reviewed_at=EPOCH + timedelta(days=1),
                next_review_due_at=EPOCH + timedelta(days=2),
            )
        )
        # Second, LATER review: pushes the due date far out. This must govern.
        await adapter.append_review(
            _review(
                target_id=f.id,
                reviewed_at=EPOCH + timedelta(days=10),
                next_review_due_at=EPOCH + timedelta(days=200),
            )
        )

        got = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=50),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        # Not due -- the SECOND (latest) review's far-out due date governs, not the first's.
        assert got == []

        if isinstance(adapter, LanceDBAdapter):
            rows = await raw_table(adapter._record_reviews).query().to_list()
            assert len(rows) == 2
        else:
            assert len(adapter._record_reviews) == 2

    async def test_list_reviews_for_unknown_target_returns_empty_list(
        self, adapter: AnyAdapter
    ) -> None:
        """2026-08-12 cleaner-revisit-pass plan: the narrow single-id
        lookback method feeding ``next_review_due_at``'s adaptive-interval compounding."""
        assert await adapter.list_reviews_for(target_id=new_id()) == []

    async def test_list_reviews_for_returns_ascending_by_reviewed_at(
        self, adapter: AnyAdapter
    ) -> None:
        f = _fact_born(created_at=EPOCH)
        await adapter.upsert(f, _vec())

        newest = _review(
            target_id=f.id,
            reviewed_at=EPOCH + timedelta(days=10),
            next_review_due_at=EPOCH + timedelta(days=11),
            band="contested",
        )
        oldest = _review(
            target_id=f.id,
            reviewed_at=EPOCH + timedelta(days=1),
            next_review_due_at=EPOCH + timedelta(days=2),
            band="keep",
        )
        middle = _review(
            target_id=f.id,
            reviewed_at=EPOCH + timedelta(days=5),
            next_review_due_at=EPOCH + timedelta(days=6),
            band="keep",
        )
        # Appended out of chronological order to prove the result is SORTED.
        for r in (newest, oldest, middle):
            await adapter.append_review(r)

        got = await adapter.list_reviews_for(target_id=f.id)
        assert [r.reviewed_at for r in got] == [
            oldest.reviewed_at,
            middle.reviewed_at,
            newest.reviewed_at,
        ]
        assert [r.band for r in got] == ["keep", "keep", "contested"]

    async def test_list_reviews_for_is_isolated_per_target(self, adapter: AnyAdapter) -> None:
        f1 = _fact_born(created_at=EPOCH, content="f1")
        f2 = _fact_born(created_at=EPOCH, content="f2")
        await adapter.upsert(f1, _vec())
        await adapter.upsert(f2, _vec(seed=0.5))

        await adapter.append_review(
            _review(
                target_id=f1.id,
                reviewed_at=EPOCH + timedelta(days=1),
                next_review_due_at=EPOCH + timedelta(days=2),
            )
        )
        await adapter.append_review(
            _review(
                target_id=f2.id,
                reviewed_at=EPOCH + timedelta(days=1),
                next_review_due_at=EPOCH + timedelta(days=2),
            )
        )

        got_f1 = await adapter.list_reviews_for(target_id=f1.id)
        assert [r.target_id for r in got_f1] == [f1.id]

    async def test_created_on_open_and_usable_against_a_pre_existing_store(
        self, adapter: AnyAdapter, tmp_path: Any, clock: TestClock
    ) -> None:
        """A store built by a version of this codebase that predates ``record_reviews`` must get
        the table created fresh on open and be immediately usable — the fresh-store fixture
        pattern (a brand-new ``_make_lancedb_adapter`` call) cannot catch this by itself, since
        every fresh open already includes every CURRENT table; a store missing ONLY this one
        table is what a real pre-existing deployment looks like.
        """
        if not isinstance(adapter, LanceDBAdapter):
            # MemoryAdapter carries no on-disk schema to predate this table -- `_record_reviews`
            # always exists from construction. The property under test (an EXISTING store built
            # before this table existed) is LanceDB-only by construction; still exercise the
            # basic read/write round trip here so this test runs meaningfully on both adapters.
            f = _fact_born(created_at=EPOCH)
            await adapter.upsert(f, _vec())
            got = await adapter.list_due_for_review(
                scope=Scope("proj-a"),
                now=EPOCH + timedelta(days=100),
                created_before=EPOCH + timedelta(days=365),
                limit=10,
            )
            assert [r.id for r in got] == [f.id]
            return

        import lancedb as _lancedb

        from tabularium.storage.lancedb_codec import (
            CLEANER_STATE_SCHEMA as _cs_schema,
        )
        from tabularium.storage.lancedb_codec import (
            EVENTS_SCHEMA as _ev_schema,
        )
        from tabularium.storage.lancedb_codec import (
            PAIRS_SCHEMA as _pr_schema,
        )
        from tabularium.storage.lancedb_codec import (
            PASS_HISTORY_SCHEMA as _ph_schema,
        )
        from tabularium.storage.lancedb_codec import (
            REGISTRY_SCHEMA as _reg_schema,
        )
        from tabularium.storage.lancedb_codec import (
            SESSIONS_SCHEMA as _se_schema,
        )
        from tabularium.storage.lancedb_codec import (
            TOMBSTONES_SCHEMA as _to_schema,
        )
        from tabularium.storage.lancedb_codec import make_records_schema

        store_path = str(tmp_path / "pre_existing_store")
        db = await _lancedb.connect_async(store_path)
        # Every table EXCEPT `record_reviews` -- the exact shape of a store written by a version
        # of this codebase before this feature shipped.
        await db.create_table("records", schema=make_records_schema(4))
        await db.create_table("events", schema=_ev_schema)
        await db.create_table("pairs", schema=_pr_schema)
        await db.create_table("tombstones", schema=_to_schema)
        await db.create_table("scopes", schema=_reg_schema)
        await db.create_table("facets", schema=_reg_schema)
        await db.create_table("roles", schema=_reg_schema)
        await db.create_table("sessions", schema=_se_schema)
        await db.create_table("cleaner_state", schema=_cs_schema)
        await db.create_table("pass_history", schema=_ph_schema)

        reopened = await LanceDBAdapter.open(
            store_path, embedding_model="test-model", dim=4, clock=clock
        )

        list_result = await reopened._db.list_tables()
        names = set(list_result.tables) if hasattr(list_result, "tables") else set(list_result)
        assert "record_reviews" in names

        f = _fact_born(created_at=EPOCH)
        await reopened.upsert(f, _vec())
        await reopened.append_review(
            _review(
                target_id=f.id,
                reviewed_at=EPOCH,
                next_review_due_at=EPOCH + timedelta(days=90),
            )
        )
        due = await reopened.list_due_for_review(
            scope=Scope("proj-a"),
            now=EPOCH + timedelta(days=200),
            created_before=EPOCH + timedelta(days=365),
            limit=10,
        )
        assert [r.id for r in due] == [f.id]


# ===========================================================================
# 13j — Record judgments: append_record_judgment
# (M2+ "who judged what and why" slice). A NEW, separate, APPEND-ONLY table
# (owner correction, 2026-08-16, reversing an earlier draft of this slice
# that upserted it), never a widening of ``record_reviews`` — see
# ``storage.types.RecordJudgment`` for the full contract, including the
# join-key reasoning.
# ===========================================================================


async def _judgment_rows(adapter: AnyAdapter, target_id: HypomnemaId) -> list[dict[str, Any]]:
    if isinstance(adapter, LanceDBAdapter):
        rows = await raw_table(adapter._record_judgments).query().to_list()
        matched = [row for row in rows if row["target_id"] == str(target_id)]
    else:
        matched = [
            record_judgment_to_row(j) for j in adapter._record_judgments if j.target_id == target_id
        ]

    return sorted(matched, key=lambda row: row["judged_at"])


class TestRecordJudgments:
    async def test_round_trips_exactly_as_written(self, adapter: AnyAdapter) -> None:
        """Falsifiability (task brief): break persistence -- e.g. drop a key from
        ``lancedb_codec.record_judgment_to_row`` -- and this goes red on the LanceDB adapter
        (MemoryAdapter stores the ``RecordJudgment`` object directly, so it can never lose a
        field this way — this is the LanceDB-specific half of the drill; the round-trip
        assertion itself runs on both adapters via this same parametrized test)."""
        f = _fact_born(created_at=EPOCH)
        paired = _fact_born(created_at=EPOCH, content="the paired record")
        await adapter.upsert(f, _vec())
        await adapter.upsert(paired, _vec(seed=0.5))

        await adapter.append_record_judgment(
            _judgment(
                target_id=f.id,
                judged_at=EPOCH,
                judge_a_identity="claude:opus",
                judge_a_grade="shelve-clear",
                judge_a_why="this finding is now derivable from the shipped code",
                judge_b_identity="codex:gpt",
                judge_b_grade="keep-clear",
                judge_b_why="still the only record of a defect nobody has fixed",
                paired_with=paired.id,
                similarity=0.91,
            )
        )

        [got] = await _judgment_rows(adapter, f.id)
        assert got["judge_a_identity"] == "claude:opus"
        assert got["judge_a_grade"] == "shelve-clear"
        assert got["judge_a_why"] == "this finding is now derivable from the shipped code"
        assert got["judge_b_identity"] == "codex:gpt"
        assert got["judge_b_grade"] == "keep-clear"
        assert got["judge_b_why"] == "still the only record of a defect nobody has fixed"
        assert got["paired_with"] == str(paired.id)
        assert got["similarity"] == 0.91

    async def test_appending_two_judgments_keeps_both_rows_and_latest_governs(
        self, adapter: AnyAdapter
    ) -> None:
        """The falsifiability bar this rework exists to satisfy: two successive contested
        judgments on the SAME record must BOTH survive as their own rows, in ``judged_at``
        order."""
        f = _fact_born(created_at=EPOCH)
        await adapter.upsert(f, _vec())

        await adapter.append_record_judgment(
            _judgment(target_id=f.id, judged_at=EPOCH, judge_a_why="first contest")
        )
        await adapter.append_record_judgment(
            _judgment(target_id=f.id, judged_at=EPOCH + timedelta(days=5), judge_a_why="second")
        )

        rows = await _judgment_rows(adapter, f.id)
        assert [row["judge_a_why"] for row in rows] == ["first contest", "second"]

    async def test_second_append_does_not_overwrite_the_first_row(
        self, adapter: AnyAdapter
    ) -> None:
        """The old upsert behaviour is genuinely gone: a second write must NOT replace the first
        physical row. Falsifiability: change ``append_record_judgment`` back to a ``merge_insert``
        keyed on ``target_id`` and this test goes red — the second call would silently replace the
        first physical row instead of adding a second one, and the row-count assertion below fails
        outright (one row instead of two)."""
        f = _fact_born(created_at=EPOCH)
        await adapter.upsert(f, _vec())

        await adapter.append_record_judgment(
            _judgment(target_id=f.id, judged_at=EPOCH, judge_a_why="first contest")
        )
        await adapter.append_record_judgment(
            _judgment(target_id=f.id, judged_at=EPOCH + timedelta(days=5), judge_a_why="second")
        )

        assert len(await _judgment_rows(adapter, f.id)) == 2

    async def test_falsity_pair_shape_survives_two_passes(self, adapter: AnyAdapter) -> None:
        """Tripwire 2 (do not lose the falsity pair shape): a record contested by TWO separate
        falsity-axis passes (e.g. against a different candidate each time) must keep each pass's
        OWN ``paired_with``/``similarity`` intact on its own physical row — appending the second
        pass's judgment must not corrupt or blend the first pass's pairing.
        """
        f = _fact_born(created_at=EPOCH)
        first_pair = _fact_born(created_at=EPOCH, content="first pass's paired record")
        second_pair = _fact_born(created_at=EPOCH, content="second pass's paired record")
        await adapter.upsert(f, _vec())
        await adapter.upsert(first_pair, _vec(seed=0.4))
        await adapter.upsert(second_pair, _vec(seed=0.6))

        await adapter.append_record_judgment(
            _judgment(
                target_id=f.id,
                judged_at=EPOCH,
                paired_with=first_pair.id,
                similarity=0.81,
            )
        )
        await adapter.append_record_judgment(
            _judgment(
                target_id=f.id,
                judged_at=EPOCH + timedelta(days=30),
                paired_with=second_pair.id,
                similarity=0.95,
            )
        )

        # Each pass's pairing lives on its own physical row (falsifiability: an upsert-shaped
        # write would leave only ONE physical row, so `pairings` below would have length 1,
        # not 2).
        rows = await _judgment_rows(adapter, f.id)
        pairings = {row["paired_with"]: row["similarity"] for row in rows}
        assert len(pairings) == 2
        assert pairings[str(first_pair.id)] == 0.81
        assert pairings[str(second_pair.id)] == 0.95

    async def test_created_on_open_and_usable_against_a_pre_existing_store_with_reviews(
        self, adapter: AnyAdapter, tmp_path: Any, clock: TestClock
    ) -> None:
        """Owner-mandated replacement for the migration-widening drill this slice's earlier draft
        attempted against ``record_reviews`` itself (which the owner ruled out): a store that
        ALREADY has a populated ``record_reviews`` table and NO ``record_judgments`` table at all
        (the exact shape a real pre-slice deployment has) must get the new table created cleanly
        on open, with the OLD ``record_reviews`` row still readable and correctly reported as
        carrying no judge data (there is nowhere for it to have come from — this table did not
        exist yet when that row was written).

        Falsifiability: delete the ``record_judgments = await _open_or_create("record_judgments",
        RECORD_JUDGMENTS_SCHEMA)`` line this test exists to justify (``lancedb_adapter.py``'s
        ``open()``) and this goes red — ``AttributeError`` on ``adapter._record_judgments`` (the
        table was never created, so ``__init__`` never receives it either).
        """
        if not isinstance(adapter, LanceDBAdapter):
            pytest.skip(
                "LanceDB-only: MemoryAdapter carries no on-disk schema for a table to predate"
            )

        import json as _json

        import lancedb as _lancedb

        from tabularium.storage.lancedb_codec import (
            CLEANER_STATE_SCHEMA as _cs_schema,
        )
        from tabularium.storage.lancedb_codec import (
            EVENTS_SCHEMA as _ev_schema,
        )
        from tabularium.storage.lancedb_codec import (
            PAIRS_SCHEMA as _pr_schema,
        )
        from tabularium.storage.lancedb_codec import (
            PASS_HISTORY_SCHEMA as _ph_schema,
        )
        from tabularium.storage.lancedb_codec import (
            REGISTRY_SCHEMA as _reg_schema,
        )
        from tabularium.storage.lancedb_codec import (
            REVIEWS_SCHEMA as _rv_schema,
        )
        from tabularium.storage.lancedb_codec import (
            SESSIONS_SCHEMA as _se_schema,
        )
        from tabularium.storage.lancedb_codec import (
            TOMBSTONES_SCHEMA as _to_schema,
        )
        from tabularium.storage.lancedb_codec import make_records_schema

        store_path = str(tmp_path / "pre_judgments_table_store")
        db = await _lancedb.connect_async(store_path)
        # Every table a real pre-slice store already has -- `record_reviews` POPULATED, at its
        # OWN unchanged, current schema (this slice never touched it) -- and `record_judgments`
        # deliberately ABSENT, the exact shape a prior decision's own "a new table needs
        # no migration branch" precedent this test exists to confirm still applies here.
        await db.create_table("records", schema=make_records_schema(4))
        await db.create_table("events", schema=_ev_schema)
        await db.create_table("pairs", schema=_pr_schema)
        await db.create_table("tombstones", schema=_to_schema)
        await db.create_table("scopes", schema=_reg_schema)
        await db.create_table("facets", schema=_reg_schema)
        await db.create_table("roles", schema=_reg_schema)
        await db.create_table("sessions", schema=_se_schema)
        await db.create_table("cleaner_state", schema=_cs_schema)
        await db.create_table("pass_history", schema=_ph_schema)
        reviews_table = await db.create_table("record_reviews", schema=_rv_schema)

        old_target_id = new_id()
        await reviews_table.add(
            [
                {
                    "target_id": str(old_target_id),
                    "scope": "proj-a",
                    "reviewed_at": EPOCH,
                    "next_review_due_at": EPOCH + timedelta(days=90),
                    "outcome": "judged",
                    "band": "contested",
                    "grades": _json.dumps(["shelve-clear", "keep-clear"]),
                    "rubric_version": "cleaner-rubric-v1-test",
                }
            ]
        )

        reopened = await LanceDBAdapter.open(
            store_path, embedding_model="test-model", dim=4, clock=clock
        )

        # The new table exists and is immediately usable.
        list_result = await reopened._db.list_tables()
        names = set(list_result.tables) if hasattr(list_result, "tables") else set(list_result)
        assert "record_judgments" in names

        # The OLD record_reviews row is untouched and still readable -- `record_reviews` itself
        # was never migrated by this slice, so there is nothing here to have gone wrong.
        old_history = await reopened.list_reviews_for(target_id=old_target_id)
        assert len(old_history) == 1
        assert old_history[0].band == "contested"
        assert old_history[0].grades == ("shelve-clear", "keep-clear")

        # Correctly reported as carrying no judge data -- the new table has nothing for an id
        # that predates its own existence.
        assert await _judgment_rows(reopened, old_target_id) == []

        # The new table is immediately usable for a FRESH write.
        f = _fact_born(created_at=EPOCH)
        await reopened.upsert(f, _vec())
        await reopened.append_record_judgment(
            _judgment(target_id=f.id, judged_at=EPOCH, judge_a_why="fresh after reopen")
        )
        [fresh] = await _judgment_rows(reopened, f.id)
        assert fresh["judge_a_why"] == "fresh after reopen"

        # A second open() is idempotent -- no crash, table still there, no duplicate state.
        reopened_again = await LanceDBAdapter.open(
            store_path, embedding_model="test-model", dim=4, clock=clock
        )
        [again] = await _judgment_rows(reopened_again, f.id)
        assert again["judge_a_why"] == "fresh after reopen"


# ===========================================================================
# 13b — Startup population passes (P1 schema piece, 2026-09-20)
#
# ``populate_content_keys``/``populate_edge_ids`` — protocol coverage only;
# the resumable-under-a-crash property and the timing number live in
# ``tests/test_startup_passes.py`` (LanceDB-only — chunking and a killed
# write are storage-fragment concepts ``MemoryAdapter`` has none of).
# ===========================================================================


class TestStartupPopulationPasses:
    async def test_populate_content_keys_zero_after_fresh_write(
        self, adapter: AnyAdapter
    ) -> None:
        """Both adapters derive ``content_key``/``content_hash`` on every
        row write — a freshly upserted row is
        never a candidate for this pass."""
        f = _fact(content="freshly written, never a backfill candidate")
        d = _decision(content="freshly written decision")
        await adapter.upsert(f, _vec())
        await adapter.upsert(d, _vec(seed=0.9))

        written = await adapter.populate_content_keys()
        assert written == 0

    async def test_populate_edge_ids_fills_missing_edge_ids(
        self, adapter: AnyAdapter
    ) -> None:
        """A row superseded before this piece's ``replace`` existed (or
        migrated from a pre-P3 store) carries a ``superseded_by`` pointer
        with no edge id — constructed directly here via ``upsert`` (never
        through ``replace``, which always mints one in the same
        transaction) to stand in for that legacy shape. The pass fills it,
        exactly once, on first call, and never reassigns on a second."""
        successor = _decision(content="successor decision")
        await adapter.upsert(successor, _vec(seed=0.9))
        predecessor = Decision.model_validate(
            {
                **_decision(content="predecessor decision").model_dump(mode="python"),
                "lifecycle": "superseded",
                "superseded_by": successor.id,
            }
        )
        await adapter.upsert(predecessor, _vec())

        retrieved = await adapter.get_by_id(predecessor.id)
        assert isinstance(retrieved, Decision)
        assert retrieved.supersession_edge_id is None

        written = await adapter.populate_edge_ids()
        assert written == 1

        filled = await adapter.get_by_id(predecessor.id)
        assert isinstance(filled, Decision)
        first_id = filled.supersession_edge_id
        assert first_id is not None

        written_again = await adapter.populate_edge_ids()
        assert written_again == 0

        unchanged = await adapter.get_by_id(predecessor.id)
        assert isinstance(unchanged, Decision)
        assert unchanged.supersession_edge_id == first_id

    async def test_concurrent_populate_edge_ids_calls_do_not_double_assign(
        self, adapter: AnyAdapter
    ) -> None:
        """Selecting a chunk's ids
        outside the adapter lock let two lock-holding writers race a stale
        selection. Fixed by selecting fresh inside ``self._supersede_lock``
        (LanceDB) / relying on the no-yield-point single-pass body
        (memory) — two full concurrent passes must land exactly one id per
        edge, never a double assignment, never a dropped one."""
        decisions = [_decision(content=f"decision {i}") for i in range(6)]
        for i, d in enumerate(decisions):
            if i % 2 == 0:
                d = Decision.model_validate(
                    {
                        **d.model_dump(mode="python"),
                        "lifecycle": "superseded",
                        "superseded_by": decisions[i + 1].id,
                    }
                )
                decisions[i] = d
            await adapter.upsert(d, _vec())

        results = await asyncio.gather(
            adapter.populate_edge_ids(),
            adapter.populate_edge_ids(),
        )
        assert sum(results) == 3

        edge_ids = []
        for i in range(0, 6, 2):
            rec = await adapter.get_by_id(decisions[i].id)
            assert isinstance(rec, Decision)
            assert rec.supersession_edge_id is not None
            edge_ids.append(rec.supersession_edge_id)
        assert len(set(edge_ids)) == 3


# ===========================================================================
# 13c — find_by_content_key (P7 admission-order piece): a pure key+scope
# lookup, both kinds, every validity and every lifecycle -- the engine
# (``engine/admission.py``) applies the population filter, this method never
# does.
# ===========================================================================


class TestFindByContentKey:
    async def test_finds_same_key_in_two_scopes(self, adapter: AnyAdapter) -> None:
        a = _fact(scope="proj-a", content="shared duplicate content")
        b = _fact(scope="proj-b", content="shared duplicate content")
        await adapter.upsert(a, _vec())
        await adapter.upsert(b, _vec(seed=0.9))

        found = await adapter.find_by_content_key(
            content_key("shared duplicate content"),
            [Scope("proj-a"), Scope("proj-b")],
        )

        assert {r.id for r in found} == {a.id, b.id}

    async def test_scope_outside_the_list_is_excluded(self, adapter: AnyAdapter) -> None:
        a = _fact(scope="proj-a", content="scoped duplicate content")
        outside = _fact(scope="proj-b", content="scoped duplicate content")
        await adapter.upsert(a, _vec())
        await adapter.upsert(outside, _vec(seed=0.9))

        found = await adapter.find_by_content_key(
            content_key("scoped duplicate content"), [Scope("proj-a")]
        )

        assert {r.id for r in found} == {a.id}

    async def test_invalid_row_is_returned_unfiltered(self, adapter: AnyAdapter) -> None:
        """The engine filters validity, not this method — an invalid row
        sharing the key is still returned."""
        invalid = _fact(
            scope="proj-a", content="invalid duplicate content", validity="invalid"
        )
        await adapter.upsert(invalid, _vec())

        found = await adapter.find_by_content_key(
            content_key("invalid duplicate content"), [Scope("proj-a")]
        )

        assert {r.id for r in found} == {invalid.id}
        assert found[0].validity == "invalid"

    async def test_decision_and_fact_sharing_a_key_are_both_returned(
        self, adapter: AnyAdapter
    ) -> None:
        fact = _fact(scope="proj-a", content="both kinds duplicate content")
        decision = _decision(scope="proj-a", content="both kinds duplicate content")
        await adapter.upsert(fact, _vec())
        await adapter.upsert(decision, _vec(seed=0.9))

        found = await adapter.find_by_content_key(
            content_key("both kinds duplicate content"), [Scope("proj-a")]
        )

        assert {r.id for r in found} == {fact.id, decision.id}
        assert {type(r) for r in found} == {Fact, Decision}

    async def test_no_match_returns_empty(self, adapter: AnyAdapter) -> None:
        found = await adapter.find_by_content_key(
            content_key("never written anywhere"), [Scope("proj-a")]
        )
        assert found == []

    async def test_key_with_a_single_quote_is_matched_safely(
        self, adapter: AnyAdapter
    ) -> None:
        """A content-derived key can carry a single quote (free text, no
        character restriction) — it must match exactly, never widen the
        predicate and never raise a parse error."""
        needle = _fact(scope="proj-a", content="owner's rule about quoting")
        decoy = _fact(scope="proj-a", content="owner rule about quoting")
        await adapter.upsert(needle, _vec())
        await adapter.upsert(decoy, _vec(seed=0.9))

        found = await adapter.find_by_content_key(
            content_key("owner's rule about quoting"), [Scope("proj-a")]
        )

        assert {r.id for r in found} == {needle.id}


# ===========================================================================
# 13n — Assessor verdicts (A2 verdict-log piece) — a
# NEW table, updated in place by ``id``, never a widening of
# ``record_judgments``. See ``storage.types.VerdictRow`` for the full
# contract.
# ===========================================================================


def _verdict(
    *,
    candidate_id: HypomnemaId | None,
    created_at: datetime,
    bounce_id: str | None = None,
    caller: str = "save",
    method: str = "relation",
    probe_rank: int | None = 0,
    similarity: float | None = 0.87,
    pick: str | None = "restatement",
    status: str = "unanswered",
    action: str | None = None,
    actor: str | None = None,
    matched: bool | None = None,
    effect_ref: str | None = None,
    new_record_id: str | None = None,
    incoming_text: str | None = None,
    override: str | None = None,
    override_at: datetime | None = None,
    stamped_at: datetime | None = None,
) -> VerdictRow:
    return VerdictRow(
        id=str(uuid.uuid4()),
        created_at=created_at,
        caller=caller,  # type: ignore[arg-type]
        method=method,
        bounce_id=bounce_id,
        candidate_id=str(candidate_id) if candidate_id is not None else None,
        probe_rank=probe_rank,
        similarity=similarity,
        pick=pick,
        odds_json='{"restatement": 0.7, "distinct": 0.3}' if pick is not None else None,
        confidence=0.4 if pick is not None else None,
        reason_class=None,
        unavailable_reason=None,
        unavailable_detail=None,
        model="jev-1.13.0" if pick is not None else None,
        wording_hash="abc123" if pick is not None else None,
        latency_ms=120,
        input_tokens=50,
        output_tokens=10,
        request_id=None,
        status=status,  # type: ignore[arg-type]
        action=action,
        actor=actor,  # type: ignore[arg-type]
        matched=matched,
        effect_ref=effect_ref,
        new_record_id=new_record_id,
        incoming_text=incoming_text,
        override=override,
        override_at=override_at,
        stamped_at=stamped_at,
    )


class TestAssessorVerdicts:
    async def test_append_then_read_by_bounce(self, adapter: AnyAdapter) -> None:
        candidate = _fact_born(created_at=EPOCH)
        await adapter.upsert(candidate, _vec())

        row = _verdict(candidate_id=candidate.id, created_at=EPOCH, bounce_id="bounce-1")
        await adapter.append_verdicts([row])

        got = await adapter.verdicts_by_bounce("bounce-1")
        assert [r.id for r in got] == [row.id]
        assert got[0].candidate_id == str(candidate.id)
        assert got[0].pick == "restatement"

    async def test_read_by_effect_matches_either_column(self, adapter: AnyAdapter) -> None:
        candidate = _fact_born(created_at=EPOCH)
        await adapter.upsert(candidate, _vec())

        via_effect_ref = _verdict(
            candidate_id=candidate.id, created_at=EPOCH, effect_ref="edge-xyz"
        )
        via_new_record_id = _verdict(
            candidate_id=candidate.id, created_at=EPOCH, new_record_id="edge-xyz"
        )
        unrelated = _verdict(candidate_id=candidate.id, created_at=EPOCH, effect_ref="other-edge")
        await adapter.append_verdicts([via_effect_ref, via_new_record_id, unrelated])

        got = await adapter.verdicts_by_effect("edge-xyz")
        assert {r.id for r in got} == {via_effect_ref.id, via_new_record_id.id}

    async def test_read_for_record_matches_either_column_newest_first(
        self, adapter: AnyAdapter
    ) -> None:
        candidate = _fact_born(created_at=EPOCH)
        successor = _fact_born(created_at=EPOCH, content="successor content")
        await adapter.upsert(candidate, _vec())
        await adapter.upsert(successor, _vec(seed=0.5))

        as_candidate = _verdict(candidate_id=candidate.id, created_at=EPOCH)
        as_new_record = _verdict(
            candidate_id=None,
            created_at=EPOCH + timedelta(minutes=5),
            new_record_id=str(candidate.id),
        )
        unrelated = _verdict(candidate_id=successor.id, created_at=EPOCH)
        await adapter.append_verdicts([as_candidate, as_new_record, unrelated])

        got = await adapter.verdicts_for_record(candidate.id)
        assert [r.id for r in got] == [as_new_record.id, as_candidate.id]

    async def test_update_flips_status_in_place_leaving_one_physical_row(
        self, adapter: AnyAdapter
    ) -> None:
        candidate = _fact_born(created_at=EPOCH)
        await adapter.upsert(candidate, _vec())

        row = _verdict(candidate_id=candidate.id, created_at=EPOCH, status="pending")
        await adapter.append_verdicts([row])

        stamped = row.model_copy(
            update={"status": "acted", "actor": "engine", "action": "reinforce", "matched": True}
        )
        await adapter.update_verdicts([stamped])

        got = await adapter.verdicts_for_record(candidate.id)
        assert len(got) == 1
        assert got[0].status == "acted"
        assert got[0].actor == "engine"

    async def test_pending_verdicts_sees_only_pending(self, adapter: AnyAdapter) -> None:
        candidate = _fact_born(created_at=EPOCH)
        await adapter.upsert(candidate, _vec())

        pending = _verdict(candidate_id=candidate.id, created_at=EPOCH, status="pending")
        acted = _verdict(candidate_id=candidate.id, created_at=EPOCH, status="acted")
        unanswered = _verdict(candidate_id=candidate.id, created_at=EPOCH, status="unanswered")
        await adapter.append_verdicts([pending, acted, unanswered])

        got = await adapter.pending_verdicts()
        assert [r.id for r in got] == [pending.id]

    async def test_scrub_nulls_incoming_text_for_that_candidate_only(
        self, adapter: AnyAdapter
    ) -> None:
        target = _fact_born(created_at=EPOCH)
        other = _fact_born(created_at=EPOCH, content="a different candidate")
        await adapter.upsert(target, _vec())
        await adapter.upsert(other, _vec(seed=0.5))

        target_row = _verdict(
            candidate_id=target.id, created_at=EPOCH, incoming_text="the discarded text"
        )
        other_row = _verdict(
            candidate_id=other.id, created_at=EPOCH, incoming_text="a different discarded text"
        )
        await adapter.append_verdicts([target_row, other_row])

        await adapter.scrub_verdict_text(target.id)

        got_target = await adapter.verdicts_for_record(target.id)
        got_other = await adapter.verdicts_for_record(other.id)
        assert got_target[0].incoming_text is None
        assert got_other[0].incoming_text == "a different discarded text"

    async def test_opening_an_existing_store_without_the_table_creates_it(
        self, adapter: AnyAdapter, tmp_path: Any, clock: TestClock
    ) -> None:
        """Falsifiability (this piece's own Steps): a real pre-A2 store — every OTHER
        table present, ``assessor_verdicts`` deliberately ABSENT — must gain the table on
        reopen with no migration branch, the same brand-new-table precedent every other aux
        table on this seam already follows."""
        if not isinstance(adapter, LanceDBAdapter):
            pytest.skip(
                "LanceDB-only: MemoryAdapter carries no on-disk schema for a table to predate"
            )

        import lancedb as _lancedb

        from tabularium.storage.lancedb_codec import EVENTS_SCHEMA as _ev_schema
        from tabularium.storage.lancedb_codec import make_records_schema

        store_path = str(tmp_path / "pre_verdicts_table_store")
        db = await _lancedb.connect_async(store_path)
        await db.create_table("records", schema=make_records_schema(4))
        await db.create_table("events", schema=_ev_schema)

        reopened = await LanceDBAdapter.open(
            store_path, embedding_model="test-model", dim=4, clock=clock
        )

        list_result = await reopened._db.list_tables()
        names = set(list_result.tables) if hasattr(list_result, "tables") else set(list_result)
        assert "assessor_verdicts" in names

        candidate = _fact_born(created_at=EPOCH)
        await reopened.upsert(candidate, _vec())
        row = _verdict(candidate_id=candidate.id, created_at=EPOCH, bounce_id="fresh-bounce")
        await reopened.append_verdicts([row])

        got = await reopened.verdicts_by_bounce("fresh-bounce")
        assert [r.id for r in got] == [row.id]

        reopened_again = await LanceDBAdapter.open(
            store_path, embedding_model="test-model", dim=4, clock=clock
        )
        still_there = await reopened_again.verdicts_by_bounce("fresh-bounce")
        assert [r.id for r in still_there] == [row.id]

    async def test_verdicts_by_bounce_handles_a_quoted_bounce_id(
        self, adapter: AnyAdapter
    ) -> None:
        """``bounce_id`` is the one lookup key on this table that is not a
        ``HypomnemaId`` (UUID-typed, quote-free by construction) — it rides the
        wire as a plain caller-supplied string (D14), so a quote in it must not
        reach a raw filter unescaped (on
        LanceDB this either raised a parse error, breaking D15's "never fails
        the save", or widened the match)."""
        if not isinstance(adapter, LanceDBAdapter):
            pytest.skip("MemoryAdapter does exact dict comparison; no filter string to break")

        candidate = _fact_born(created_at=EPOCH)
        await adapter.upsert(candidate, _vec())

        quoted_id = "b1' OR '1'='1"
        row = _verdict(candidate_id=candidate.id, created_at=EPOCH, bounce_id=quoted_id)
        other_row = _verdict(candidate_id=candidate.id, created_at=EPOCH, bounce_id="b2")
        await adapter.append_verdicts([row, other_row])

        got = await adapter.verdicts_by_bounce(quoted_id)
        assert [r.id for r in got] == [row.id]

        got_unknown = await adapter.verdicts_by_bounce("no-such-bounce'; --")
        assert got_unknown == []


# ===========================================================================
# 14 — Hypothesis property tests
# ===========================================================================

# Strategies for Hypothesis

_SCOPE_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"


def _scope_strategy() -> st.SearchStrategy[str]:
    """Generate valid scope names matching ^[a-z][a-z0-9-]{0,31}$."""
    return st.builds(
        lambda first, rest: first + rest,
        st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
        st.text(alphabet=_SCOPE_ALPHABET, min_size=0, max_size=10),
    )


def _lifecycle_strategy(arm: str = "fact") -> st.SearchStrategy[str]:
    if arm == "fact":
        return st.sampled_from(["active", "hot", "ember", "archived"])
    return st.sampled_from(["in_force", "superseded", "revoked"])


def _validity_strategy() -> st.SearchStrategy[str]:
    return st.sampled_from(["valid", "invalid"])


def _fact_strategy(
    scope_st: st.SearchStrategy[str] | None = None,
) -> st.SearchStrategy[Fact]:
    scope_s = scope_st if scope_st is not None else _scope_strategy()
    return st.builds(
        lambda scope, lifecycle, validity, content: Fact(
            id=new_id(),
            type="fact",
            content=content,
            scope=Scope(scope),
            lifecycle=lifecycle,
            validity=validity,
            created_at=EPOCH,
            updated_at=EPOCH,
        ),
        scope=scope_s,
        lifecycle=_lifecycle_strategy("fact"),
        validity=_validity_strategy(),
        content=st.text(min_size=1, max_size=80),
    )


@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    # Explicit (not None) deadline: every example does a real async write
    # against the shared ``adapter`` fixture, including an on-disk LanceDB
    # write on the [lancedb] id — the default 200ms budget is a per-example
    # wall-clock check that max_examples cannot buy back (it bounds one
    # example's runtime, not the total run). Measured: 0-231ms per example
    # with the host otherwise idle; up to ~417ms, with 30+ DeadlineExceeded
    # failures at the 200ms default, when a full ``pytest -q`` ran
    # concurrently in the background. 2000ms leaves ~5x margin over the
    # worst observed spike while still failing fast on a genuine
    # algorithmic regression, which would run into whole seconds against
    # this fixture size, not milliseconds.
    deadline=2000,
)
@given(
    records=st.lists(
        _fact_strategy(scope_st=st.just("proj-a")),
        min_size=1,
        max_size=10,
    )
)
async def test_hypothesis_scope_filter_correctness(
    records: list[Fact], adapter: AnyAdapter
) -> None:
    """A vector search in scope proj-a never returns records from other scopes."""
    for f in records:
        await adapter.upsert(f, _vec())

    # Add a record in a different scope
    other = _fact(scope="other", content="should not appear")
    await adapter.upsert(other, _vec(seed=0.5))

    filters = _filters(scopes=["proj-a"])
    hits = await adapter.vector_search(_vec(), filters=filters, k=20)
    ids = {h.record.id for h in hits}
    assert other.id not in ids


@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    # Explicit (not None) deadline: every example does a real async write
    # against the shared ``adapter`` fixture, including an on-disk LanceDB
    # write on the [lancedb] id — the default 200ms budget is a per-example
    # wall-clock check that max_examples cannot buy back (it bounds one
    # example's runtime, not the total run). Measured: 0-231ms per example
    # with the host otherwise idle; up to ~417ms, with 30+ DeadlineExceeded
    # failures at the 200ms default, when a full ``pytest -q`` ran
    # concurrently in the background. 2000ms leaves ~5x margin over the
    # worst observed spike while still failing fast on a genuine
    # algorithmic regression, which would run into whole seconds against
    # this fixture size, not milliseconds.
    deadline=2000,
)
@given(
    lifecycle=_lifecycle_strategy("fact"),
    validity=_validity_strategy(),
)
async def test_hypothesis_filter_combination_correctness(
    lifecycle: FactLifecycle, validity: str, adapter: AnyAdapter
) -> None:
    """A record with specific lifecycle+validity is returned iff both match the filter."""
    f = Fact(
        id=new_id(),
        type="fact",
        content="hypothesis filter target",
        scope=Scope("proj-a"),
        lifecycle=lifecycle,
        validity=validity,  # type: ignore[arg-type]
        created_at=EPOCH,
        updated_at=EPOCH,
    )
    await adapter.upsert(f, _vec())

    # Query with filters that MATCH
    matching_filters = Filters(
        scopes=[Scope("proj-a")],
        validity=frozenset({validity}),  # type: ignore[arg-type]
        lifecycle_in=frozenset({lifecycle}),
    )
    hits = await adapter.vector_search(_vec(), filters=matching_filters, k=5)
    assert any(h.record.id == f.id for h in hits), (
        f"Expected record with lifecycle={lifecycle!r}, validity={validity!r} to appear"
    )

    # Query with filters that EXCLUDE by lifecycle
    other_lifecycle: FactLifecycle = (
        "archived"
        if lifecycle in ("active", "hot", "ember")
        else "active"
    )
    excluding_filters = Filters(
        scopes=[Scope("proj-a")],
        validity=frozenset({validity}),  # type: ignore[arg-type]
        lifecycle_in=frozenset({other_lifecycle}),
    )
    hits2 = await adapter.vector_search(_vec(), filters=excluding_filters, k=5)
    assert all(h.record.id != f.id for h in hits2), (
        f"Record with lifecycle={lifecycle!r} must not appear in lifecycle_in={other_lifecycle!r}"
    )


# ===========================================================================
# 15 — Guard: every property actually runs on both adapters
# ===========================================================================

# Tests named here are known NOT to request the shared ``adapter`` fixture and
# therefore never exercise LanceDB. Every entry MUST carry a reason that is
# about the *contract* — something structurally impossible to express via the
# shared fixture — never mere convenience. Empty is the goal: as of this
# audit every conformance property in this module has been proven expressible
# against both adapters.
_MEMORY_ONLY_EXEMPTIONS: frozenset[str] = frozenset()


def _conformance_test_callables() -> list[tuple[str, Any]]:
    """Every ``test_*`` function/method defined directly in this module.

    Returns ``(qualified_name, callable)`` pairs: bare module-level functions
    under their own name, class methods as ``ClassName.method_name``. Walks
    ``vars(module)`` directly (not pytest's collection tree) so the guard has
    no dependency on collection internals — it gives the same answer whether
    run via ``-k``, ``--collect-only``, or a full suite run.
    """
    module = sys.modules[__name__]
    found: list[tuple[str, Any]] = []
    for name, obj in vars(module).items():
        if inspect.isfunction(obj) and name.startswith("test_"):
            found.append((name, obj))
        elif inspect.isclass(obj) and name.startswith("Test"):
            for meth_name, meth in vars(obj).items():
                if inspect.isfunction(meth) and meth_name.startswith("test_"):
                    found.append((f"{name}.{meth_name}", meth))
    return found


def test_every_property_runs_on_both_adapters() -> None:
    """Guard: every property either requests the shared ``adapter`` fixture
    (and therefore collects as both ``[memory]`` and ``[lancedb]``) or is
    named in ``_MEMORY_ONLY_EXEMPTIONS`` with a contract-grounded reason.

    A test that can pass by only ever running the implementation that
    behaves is not a gate — that is exactly the shape that let the
    silent-ten defect (ADR 0018) through the exhaustiveness test this module
    used to carry. This is the property that stops the hole reopening: it
    fails, listing offenders by name, the moment a new test lands in this
    module without either requesting ``adapter`` or being deliberately and
    visibly exempted.
    """
    offenders = sorted(
        qualname
        for qualname, func in _conformance_test_callables()
        if qualname != "test_every_property_runs_on_both_adapters"
        and "adapter" not in inspect.signature(func).parameters
        and qualname not in _MEMORY_ONLY_EXEMPTIONS
    )
    assert not offenders, (
        "Tests neither requesting the shared 'adapter' fixture nor listed in "
        f"_MEMORY_ONLY_EXEMPTIONS: {offenders}"
    )
