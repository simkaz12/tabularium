"""``engine/context.py`` — fixed default push-side selection tests

(M2 slice C, SPEC §7 / §8.4).

Deterministic-importance strategy (same idiom as ``test_lifecycle.py``):
every test config zeroes ``w_recall`` and ``w_recency`` and sets
``w_feedback=1.0``, so ``importance == feedback_score`` exactly — no
clock-dependent recency term to reason about when a test wants to control
importance directly. Tests that DO want to exercise the recency secondary
key set ``last_referenced_at`` explicitly (with ``w_recency`` still zeroed,
so importance itself stays tied and only the secondary/tertiary sort key
decides order).

Falsifiability (non-negotiable per the M2A/M2C precedent — see
``test_lifecycle.py``'s module docstring): the rank-merge property is proven
sensitive to a real defect by deliberately mutating ``_select_hot`` into a
scope-priority concatenation, watching ``TestRankMerge`` go red, then
restoring — see the task verdict for the captured failure output. The break
itself is not committed to this file; it was applied and reverted by hand
against ``src/tabularium/engine/context.py`` during development.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.engine.backfill import NoBackfill
from tabularium.engine.context import SelectedTier, pack, select_default
from tabularium.engine.engine_context import EngineContext
from tabularium.results import ContextPayload
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock
from tests.support.fake_assessor import EchoAssessor

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _ctx(adapter: MemoryAdapter, cfg: EngineConfig, clock: TestClock) -> EngineContext:
    return EngineContext(
        adapter=adapter,
        embedder=FakeEmbedder(),
        clock=clock,
        cfg=cfg,
        write_lock=asyncio.Lock(),
        assessor=None,
        token_counter=None,
        backfill=NoBackfill(),
    )


def _cfg(**kw: object) -> EngineConfig:
    """Deterministic-importance config: importance == feedback_score."""
    defaults: dict[str, object] = {
        "w_recall": 0.0,
        "w_recency": 0.0,
        "w_feedback": 1.0,
    }
    defaults.update(kw)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


def _fact(
    *,
    scope: str = "proj-a",
    lifecycle: str = "hot",
    feedback_score: float = 0.0,
    validity: str = "valid",
    content: str = "context selection test fact",
    created_at: datetime = EPOCH,
    last_referenced_at: datetime | None = None,
    audience_hint: tuple[str, ...] | None = ("decider",),
) -> Fact:
    """Defaults to ``audience_hint=("decider",)`` (curation block
    for the routing semantics; retargeted off the retired ``"default"``
    role): explicit enumeration only means an unhinted Fact now
    reaches NO role on the push tiers, and almost every test below queries
    ``role="decider"`` to exercise selection mechanics (rank, sort,
    exclusion, concurrency, packing delegation) orthogonal to routing
    itself -- ``"decider"`` is just a stand-in registered role, chosen
    because ``initialize()`` always seeds it (unlike the retired
    ``"default"``). Tests that need a genuinely UNTAGGED Fact -- the
    routing property itself -- pass ``audience_hint=None`` explicitly
    (``TestNoUniversalDefault``, ``test_two_registered_roles_return_identical_payloads``)."""
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        feedback_score=feedback_score,
        created_at=created_at,
        updated_at=created_at,
        last_referenced_at=last_referenced_at,
        audience_hint=audience_hint,
    )


def _vec(dim: int = 4, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _adapter() -> MemoryAdapter:
    return MemoryAdapter(embedding_model="test-model", dim=4, clock=TestClock(EPOCH))


def _tier(tiers: list[SelectedTier], name: str) -> SelectedTier:
    for t in tiers:
        if t.name == name:
            return t
    raise AssertionError(f"no tier named {name!r} in {[t.name for t in tiers]!r}")


# ---------------------------------------------------------------------------
# Rank-merge — the headline property (task step 1 + 5)
# ---------------------------------------------------------------------------


class TestRankMerge:
    """Scope X + global merge by rank within each tier, no per-scope quota."""

    async def test_global_record_outranking_scope_lands_first_hot_tier(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        scope_fact = _fact(scope="proj-a", feedback_score=0.50)
        global_fact = _fact(scope="global", feedback_score=0.90)  # outranks every scope-X record
        for f in (scope_fact, global_fact):
            await adapter.upsert(f, _vec())

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        hot_ids = [e.id for e in _tier(tiers, "hot").entries]

        assert hot_ids == [global_fact.id, scope_fact.id], (
            "a global record outranking every scope-X record must land first — "
            "concatenation (scope-X block then global block) would instead put "
            "scope_fact first"
        )

    async def test_scope_record_outranking_global_lands_first_hot_tier(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        scope_fact = _fact(scope="proj-a", feedback_score=0.90)  # outranks every global record
        global_fact = _fact(scope="global", feedback_score=0.50)
        for f in (scope_fact, global_fact):
            await adapter.upsert(f, _vec())

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        hot_ids = [e.id for e in _tier(tiers, "hot").entries]

        assert hot_ids == [scope_fact.id, global_fact.id], (
            "the reverse case: a scope-X record outranking every global record "
            "must also land first — 'global always wins' scope-priority would "
            "instead put global_fact first"
        )

# ---------------------------------------------------------------------------
# No backfill — active Facts never enter push
# ---------------------------------------------------------------------------


class TestNoBackfill:
    async def test_active_facts_never_appear_in_any_tier(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        actives = [
            _fact(scope="proj-a", lifecycle="active", feedback_score=0.99) for _ in range(8)
        ]
        for f in actives:
            await adapter.upsert(f, _vec())

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")

        assert [t.name for t in tiers] == ["hot", "ember"]
        all_ids = {e.id for t in tiers for e in t.entries}
        assert all_ids == set(), (
            "no number of high-importance active Facts may backfill any tier — "
            "that is Recall's on-demand job, not push's (SPEC §7)"
        )


# ---------------------------------------------------------------------------
# Exclusions — one assertion each
# ---------------------------------------------------------------------------


class TestExclusions:
    async def test_archived_fact_excluded(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        archived = _fact(lifecycle="archived", feedback_score=0.99)
        live_hot = _fact(lifecycle="hot", feedback_score=0.10)
        for f in (archived, live_hot):
            await adapter.upsert(f, _vec())

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        ids = {e.id for e in _tier(tiers, "hot").entries}
        assert archived.id not in ids

    async def test_invalid_record_excluded(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        invalid = _fact(lifecycle="hot", validity="invalid", feedback_score=0.99)
        valid = _fact(lifecycle="hot", validity="valid", feedback_score=0.10)
        for f in (invalid, valid):
            await adapter.upsert(f, _vec())

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        ids = {e.id for e in _tier(tiers, "hot").entries}
        assert invalid.id not in ids
        assert valid.id in ids

    async def test_buried_record_cannot_appear(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        hot = _fact(lifecycle="hot", feedback_score=0.90)
        await adapter.upsert(hot, _vec())
        await adapter.bury(hot.id, reason="test burial — content is gone")

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        ids = {e.id for e in _tier(tiers, "hot").entries}
        assert hot.id not in ids


# ---------------------------------------------------------------------------
# Concurrent bury during select_default's scan_records() round trip
# (``findings-context-latency.md``). ``select_default`` is
# deliberately lock-free (plan.md:62 -- the read-only ``context`` tool must
# not take the write lock). Before that change, each tier paid a scan-then-fetch
# shape (``scan()`` for metadata, one ``get_by_id()`` per surviving
# candidate) with a real unlocked window between the two calls; a concurrent
# ``bury()`` landing in that window made ``get_by_id()`` raise ``KeyError``,
# caught by a per-tier guard that dropped the candidate. The change collapsed
# both calls into a single ``adapter.scan_records()`` round trip that
# returns full records directly -- there is no second call left for a bury
# to race, so the decision below is now settled differently (see the task
# 11 decision recorded in ``engine/context.py``'s module docstring).
# ---------------------------------------------------------------------------


class TestConcurrentBuryDuringSelection:
    async def test_bury_landing_after_the_scan_records_round_trip_does_not_remove_the_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Decision: a bury() landing AFTER select_default's single
        ``scan_records()`` round trip has already captured a record's
        content must NOT retroactively remove it from that tier -- the
        payload reflects the store as of the scan, same as every other
        field already on a selected record (recency, validity, scope).

        ``adapter.scan_records`` is wrapped so the wrapper calls through to
        the REAL implementation first (which, for ``MemoryAdapter``,
        synchronously captures its snapshot list before this wrapper
        regains control) and only THEN pauses -- proving the record's
        content was already captured before the concurrent ``bury()`` runs
        to completion, not merely that the bury lost a race by timing luck.

        Determinism: driven entirely by ``asyncio.Event`` (the gate's
        ``snapshot_taken``) plus a genuine ``asyncio.wait_for`` timeout -- no
        fixed sleep. The bury is fully awaited (not just started) before the
        gate is released.

        Falsify: this is the DELIBERATE behaviour change introduced here
        -- reverting to the old scan-then-fetch shape (or adding a
        second per-candidate existence check after ``scan_records``) would
        turn this test red by making ``hot.id`` disappear from the tier.
        """
        adapter = _adapter()
        cfg = _cfg()
        hot = _fact(lifecycle="hot", feedback_score=0.90)
        await adapter.upsert(hot, _vec())

        real_scan_records = adapter.scan_records
        snapshot_taken = asyncio.Event()
        may_proceed = asyncio.Event()

        async def _gated_scan_records(**kw: Any) -> Any:
            gen = await real_scan_records(**kw)
            snapshot_taken.set()
            await may_proceed.wait()
            return gen

        monkeypatch.setattr(adapter, "scan_records", _gated_scan_records)

        select_task = asyncio.create_task(
            select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        )
        await asyncio.wait_for(snapshot_taken.wait(), timeout=5.0)

        await adapter.bury(hot.id, reason="concurrent bury after scan_records snapshot")

        may_proceed.set()
        tiers = await asyncio.wait_for(select_task, timeout=5.0)

        hot_ids = {e.id for e in _tier(tiers, "hot").entries}
        assert hot.id in hot_ids, (
            "Decision: a bury() landing after scan_records()'s round "
            "trip already captured the record must NOT remove it from the "
            "tier -- the payload is a snapshot as of the scan"
        )

        # The burial itself is real and permanent -- a later get_by_id still
        # raises, this tier's already-captured snapshot is simply unaffected.
        with pytest.raises(KeyError):
            await adapter.get_by_id(hot.id)

    async def test_bury_landing_before_selection_starts_excludes_the_record(self) -> None:
        """The other half of the decision: a record buried BEFORE
        ``select_default`` ever runs is simply absent -- ``bury`` deletes
        the row, so no later ``scan_records()`` round trip can see it. Not
        new behaviour (unaffected by the change), pinned here for contrast with
        the "after" case above."""
        adapter = _adapter()
        cfg = _cfg()
        hot = _fact(lifecycle="hot", feedback_score=0.90)
        await adapter.upsert(hot, _vec())
        await adapter.bury(hot.id, reason="buried before selection runs at all")

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")

        hot_ids = {e.id for e in _tier(tiers, "hot").entries}
        assert hot.id not in hot_ids


# ---------------------------------------------------------------------------
# No per-candidate fetch (the acceptance criterion, pinned
# directly): the collapse removes the ``get_by_id()`` round trip from both
# ``_select_*`` helpers. Wrap it to fail loudly if it is ever called.
# ---------------------------------------------------------------------------


class TestNoPerCandidateFetch:
    async def test_select_default_never_calls_get_by_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        cfg = _cfg()
        hot = _fact(lifecycle="hot", feedback_score=0.9)
        ember = _fact(lifecycle="ember", feedback_score=0.9)
        for rec in (hot, ember):
            await adapter.upsert(rec, _vec())

        async def _forbidden_get_by_id(id: HypomnemaId) -> Any:
            raise AssertionError(
                "select_default must not call get_by_id -- the "
                "scan_records collapse removed the per-candidate round trip"
            )

        monkeypatch.setattr(adapter, "get_by_id", _forbidden_get_by_id)

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")

        assert {e.id for e in _tier(tiers, "hot").entries} == {hot.id}
        assert {e.id for e in _tier(tiers, "ember").entries} == {ember.id}


# ---------------------------------------------------------------------------
# Ember fade floor
# ---------------------------------------------------------------------------


class TestEmberFadeFloor:
    async def test_below_and_at_floor_absent_above_floor_present_and_ranked(self) -> None:
        adapter = _adapter()
        cfg = _cfg(archive_threshold=0.30, revival_threshold=0.35)
        below = _fact(lifecycle="ember", feedback_score=0.20)  # below floor
        at_floor = _fact(lifecycle="ember", feedback_score=0.30)  # == floor -> not "above"
        above_low = _fact(lifecycle="ember", feedback_score=0.40)
        above_high = _fact(lifecycle="ember", feedback_score=0.90)
        for f in (below, at_floor, above_low, above_high):
            await adapter.upsert(f, _vec())

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        ember_ids = [e.id for e in _tier(tiers, "ember").entries]

        assert below.id not in ember_ids
        assert at_floor.id not in ember_ids, "the fade floor is a strict 'above', not 'at or above'"
        assert ember_ids == [above_high.id, above_low.id]


# ---------------------------------------------------------------------------
# Hot tier secondary sort key — last_referenced_at DESC
# ---------------------------------------------------------------------------


class TestHotSort:
    async def test_importance_desc_then_last_referenced_at_desc(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        older = _fact(feedback_score=0.5, last_referenced_at=EPOCH - timedelta(days=10))
        newer = _fact(feedback_score=0.5, last_referenced_at=EPOCH)
        for f in (older, newer):
            await adapter.upsert(f, _vec())

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        ids = [e.id for e in _tier(tiers, "hot").entries]
        assert ids == [newer.id, older.id]


# ---------------------------------------------------------------------------
# Determinism + documented tie-break
# ---------------------------------------------------------------------------


class TestDeterminism:
    async def test_repeated_calls_are_byte_identical(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        facts = [_fact(feedback_score=0.10 * i) for i in range(1, 6)]
        for f in facts:
            await adapter.upsert(f, _vec())
        clock = TestClock(EPOCH)

        first = await select_default(_ctx(adapter, cfg, clock), "proj-a", "decider")
        second = await select_default(_ctx(adapter, cfg, clock), "proj-a", "decider")

        first_ids = [[e.id for e in t.entries] for t in first]
        second_ids = [[e.id for e in t.entries] for t in second]
        assert first_ids == second_ids

    async def test_exact_ties_break_on_ascending_id_string_not_insertion_order(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        # Identical importance (same feedback_score) AND identical recency
        # (both default last_referenced_at=None -> falls back to the same
        # created_at=EPOCH) -- id string is the only remaining differentiator.
        a = _fact(feedback_score=0.5)
        b = _fact(feedback_score=0.5)
        higher, lower = sorted((a, b), key=lambda f: str(f.id), reverse=True)
        # Insert the higher-id one FIRST, to prove the result does not track
        # insertion order.
        await adapter.upsert(higher, _vec())
        await adapter.upsert(lower, _vec())

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        hot_ids = [e.id for e in _tier(tiers, "hot").entries]

        assert hot_ids == [lower.id, higher.id], (
            "documented tie-break: str(id) ascending — never dict/set iteration order"
        )


# ---------------------------------------------------------------------------
# Empty tiers are legal
# ---------------------------------------------------------------------------


class TestEmptyTiersLegal:
    async def test_empty_scope_returns_two_present_empty_tiers(self) -> None:
        adapter = _adapter()
        cfg = _cfg()

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")

        assert [t.name for t in tiers] == ["hot", "ember"]
        for t in tiers:
            assert t.entries == []
            assert t.entries is not None


# ---------------------------------------------------------------------------
# Content is never rewritten
# ---------------------------------------------------------------------------


class TestContentPreserved:
    async def test_content_returned_verbatim(self) -> None:
        adapter = _adapter()
        cfg = _cfg()
        original = "the exact original content string, unmodified"
        f = _fact(feedback_score=0.9, content=original)
        await adapter.upsert(f, _vec())

        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), "proj-a", "decider")
        entry = _tier(tiers, "hot").entries[0]
        assert entry.content == original


# ---------------------------------------------------------------------------
# Delegation — Tabularium._select_default() (core.py's one delegation line)
# ---------------------------------------------------------------------------


class TestEngineDelegation:
    async def test_engine_select_default_delegates_and_returns_tiers(self) -> None:
        adapter = MemoryAdapter(embedding_model="test-model", dim=4)
        embedder = FakeEmbedder(dim=4)
        clock = TestClock(EPOCH)
        cfg = _cfg()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        empty_tiers = await engine._select_default("global", "decider")
        assert isinstance(empty_tiers, list)
        assert all(isinstance(t, SelectedTier) for t in empty_tiers)
        assert [t.name for t in empty_tiers] == ["hot", "ember"]
        assert all(t.entries == [] for t in empty_tiers)

        f = _fact(scope="global", feedback_score=0.9)
        await adapter.upsert(f, _vec())

        tiers = await engine._select_default("global", "decider")
        hot_ids = [e.id for e in _tier(tiers, "hot").entries]
        assert hot_ids == [f.id]


# ===========================================================================
# pack() — the token-budget packer (SPEC §7 locked mechanics)
#
# All tests here use ``FakeTokenCounter`` (cost == len(content) // 4) so
# expected ``budget_used``/``skipped`` values are computed from the fixture
# content, never copied from observed output.
# ===========================================================================


class TestNeverTruncate:
    """The hostile case: a single entry larger than the ENTIRE budget."""

    def test_oversized_entry_is_skipped_whole_not_truncated(self) -> None:
        counter = FakeTokenCounter()
        huge_content = "z" * 400  # cost 100 -- larger than any budget below
        entry = _fact(content=huge_content)
        tiers = [
            SelectedTier(name="hot", entries=[entry]),
            SelectedTier(name="ember", entries=[]),
            SelectedTier(name="decision", entries=[]),
        ]

        payload = pack(tiers, budget=50, counter=counter)

        assert payload.skipped == 1
        assert payload.budget_used == 0
        for tier in payload.tiers:
            # No partial content appears anywhere in the payload -- every
            # tier is simply empty, not holding a sliced fragment of
            # `huge_content`.
            assert tier.entries == []
            assert tier.budget_used == 0


class TestWaterfallSpill:
    """An underfull tier lets the NEXT tier exceed its own nominal share."""

    def test_underfull_tier_lets_next_tier_exceed_its_own_share_by_the_spilled_amount(
        self,
    ) -> None:
        counter = FakeTokenCounter()
        small = _fact(content="a" * 16)  # cost 4
        big = _fact(content="b" * 60)  # cost 15
        tiers = [
            SelectedTier(name="hot", entries=[small]),
            SelectedTier(name="ember", entries=[big]),
        ]
        budget = 20  # even split: share 10 / 10

        payload = pack(tiers, budget=budget, counter=counter)

        hot_tier = next(t for t in payload.tiers if t.name == "hot")
        ember_tier = next(t for t in payload.tiers if t.name == "ember")

        own_share = budget // 2  # == 10
        assert hot_tier.budget_used == 4  # spends only 4 of its own 10-share
        spilled = own_share - hot_tier.budget_used
        assert spilled == 6
        available_to_ember = own_share + spilled  # == 16
        # `ember`'s own share is 10 -- its entry costs 15, which does NOT
        # fit in 10 alone but DOES fit once the 6 spilled units arrive,
        # proving the spill (not merely "more fit").
        assert 15 > own_share
        assert 15 <= available_to_ember
        assert ember_tier.budget_used == 15
        assert ember_tier.budget_used > own_share
        assert payload.skipped == 0


class TestSpillForwardOnly:
    """Spill flows forward only -- a later tier's leftover never retroactively
    admits an entry an earlier tier already skipped."""

    def test_tier_three_leftover_does_not_resurrect_a_skipped_tier_one_entry(
        self,
    ) -> None:
        counter = FakeTokenCounter()
        tier1_entry = _fact(content="a" * 60)  # cost 15 -- doesn't fit tier 1's own 10-share
        tier2_entry = _fact(content="b" * 4)  # cost 1 -- barely spends tier 2's share
        # tier 3 has one entry, but it is oversized enough that it never fits
        # even after everything upstream waterfalls in -- non-empty (curation block:
        # an EMPTY tier gets no share at all, which is a
        # different, separately-tested case) but still ends up skipped, so
        # its whole share (plus everything spilled into it) ends up unused.
        tier3_entry = _fact(content="d" * 400)  # cost 100 -- larger than any available pool here
        tiers = [
            SelectedTier(name="hot", entries=[tier1_entry]),
            SelectedTier(name="ember", entries=[tier2_entry]),
            SelectedTier(name="decision", entries=[tier3_entry]),
        ]
        budget = 30  # shares: 10 / 10 / 10 -- all three tiers are non-empty

        payload = pack(tiers, budget=budget, counter=counter)

        hot_tier = next(t for t in payload.tiers if t.name == "hot")
        decision_tier = next(t for t in payload.tiers if t.name == "decision")

        assert hot_tier.entries == []
        assert hot_tier.skipped == 1
        assert hot_tier.budget_used == 0

        # By the time tier 3 is reached the unused pool is large (tier 1's
        # entire unused 10-share plus tier 2's 9 leftover), far more than
        # the 15-cost entry tier 1 skipped -- proving the surplus never
        # flows backward to reconsider it.
        total_unused = budget - payload.budget_used
        assert total_unused > 15
        assert decision_tier.entries == []
        assert decision_tier.skipped == 1
        assert decision_tier.budget_used == 0


class TestBudgetUsedNeverExceedsBudget:
    """Property: for ANY random entry sizes and budget, ``budget_used`` never
    exceeds ``budget`` -- the shape of bug (off-by-one on the last admitted
    entry) a hand-written case would miss (the acceptance criteria)."""

    @given(
        tier_lengths=st.lists(
            st.lists(st.integers(min_value=0, max_value=400), max_size=6),
            min_size=1,
            max_size=4,
        ),
        budget=st.integers(min_value=0, max_value=2000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_budget_used_never_exceeds_budget(
        self, tier_lengths: list[list[int]], budget: int
    ) -> None:
        counter = FakeTokenCounter()
        tiers = [
            SelectedTier(
                name=f"tier-{i}",
                entries=[_fact(content="a" * length) for length in lengths],
            )
            for i, lengths in enumerate(tier_lengths)
        ]

        payload = pack(tiers, budget=budget, counter=counter)

        assert payload.budget_used <= budget
        for tier in payload.tiers:
            assert tier.budget_used >= 0


class TestGreedyNotOptimal:
    """Greedy is greedy, not optimal -- pins first-fit-in-rank-order so a
    future 'improvement' that reorders entries for a tighter pack has to
    argue with this test, not silently replace the documented behaviour."""

    def test_first_fit_in_rank_order_strands_budget_an_optimal_pack_would_not(
        self,
    ) -> None:
        counter = FakeTokenCounter()
        first = _fact(content="a" * 32)  # cost 8
        second = _fact(content="b" * 28)  # cost 7
        third = _fact(content="c" * 12)  # cost 3
        tiers = [SelectedTier(name="hot", entries=[first, second, third])]

        payload = pack(tiers, budget=10, counter=counter)

        hot_tier = payload.tiers[0]
        # Rank-order first-fit admits `first` (cost 8), leaving only 2 of
        # the 10-budget -- neither `second` (7) nor `third` (3) fits alone
        # in what remains, even though `second` + `third` together (7+3=10)
        # would have used the ENTIRE budget with zero waste under an
        # optimal (reordering) bin-pack. The packer does not reorder.
        assert [e.id for e in hot_tier.entries] == [first.id]
        assert hot_tier.budget_used == 8
        assert hot_tier.skipped == 2


class TestZeroBudget:
    def test_budget_zero_returns_all_tiers_present_and_empty_without_raising(
        self,
    ) -> None:
        counter = FakeTokenCounter()
        tiers = [
            SelectedTier(name="hot", entries=[_fact(content="a" * 40)]),
            SelectedTier(
                name="ember", entries=[_fact(content="b" * 20), _fact(content="c" * 8)]
            ),
            SelectedTier(name="decision", entries=[]),
        ]

        payload = pack(tiers, budget=0, counter=counter)

        assert [t.name for t in payload.tiers] == ["hot", "ember", "decision"]
        assert payload.budget_used == 0
        assert payload.skipped == 3  # every non-empty tier's lone/paired entries
        for tier in payload.tiers:
            assert tier.entries == []


class TestEmptyTierGetsNoShare:
    """The live symptom (curation block): an empty tier no longer
    reserves a share it can never spend, so a non-empty tier packs against
    the FULL budget instead of half of it."""

    def test_empty_ember_no_longer_halves_hots_effective_budget(self) -> None:
        counter = FakeTokenCounter()
        # Approximates the live corpus shape: ten hot atoms costing ~100
        # tokens each (~1000 total -- close to the measured ~1007).
        hot_entries: list[Fact] = [
            _fact(content="a" * 400) for _ in range(10)
        ]  # cost 100 each
        tiers = [
            SelectedTier(name="hot", entries=hot_entries),
            SelectedTier(name="ember", entries=[]),
        ]

        payload = pack(tiers, budget=800, counter=counter)

        hot_tier = next(t for t in payload.tiers if t.name == "hot")
        ember_tier = next(t for t in payload.tiers if t.name == "ember")

        # Pre-fix: hot's share was budget // 2 == 400, packing only 4 of the
        # 10 entries (400 // 100) -- the live-reproduced
        # `budget_used=367 skipped=6 n_entries=4` shape. Post-fix: ember has
        # nothing to spend its share on, so it gets none, and hot packs
        # against the full 800 -- 8 entries instead of 4.
        assert hot_tier.budget_used == 800
        assert len(hot_tier.entries) == 8
        assert hot_tier.skipped == 2
        assert ember_tier.entries == []
        assert ember_tier.budget_used == 0
        assert ember_tier.skipped == 0
        assert payload.budget_used == 800


class TestSingleNonEmptyTierGetsWholeBudget:
    """With exactly one non-empty tier among several, it receives the
    ENTIRE budget -- not a fraction of it split across every tier."""

    def test_lone_non_empty_tier_receives_entire_budget(self) -> None:
        counter = FakeTokenCounter()
        entry = _fact(content="a" * 148)  # cost 37 -- exactly `budget`
        tiers = [
            SelectedTier(name="hot", entries=[]),
            SelectedTier(name="ember", entries=[entry]),
            SelectedTier(name="decision", entries=[]),
        ]
        budget = 37

        payload = pack(tiers, budget=budget, counter=counter)

        hot_tier = next(t for t in payload.tiers if t.name == "hot")
        ember_tier = next(t for t in payload.tiers if t.name == "ember")
        decision_tier = next(t for t in payload.tiers if t.name == "decision")

        # A three-way even split (12/13 each) would strand this cost-37
        # entry with nothing close to enough room. Excluding the two empty
        # tiers hands `ember` the entire 37-budget, and the entry fits
        # exactly.
        assert ember_tier.entries == [entry]
        assert ember_tier.budget_used == 37
        assert ember_tier.skipped == 0
        assert hot_tier.budget_used == 0
        assert hot_tier.skipped == 0
        assert decision_tier.budget_used == 0
        assert decision_tier.skipped == 0
        assert payload.budget_used == 37
        assert payload.skipped == 0


class TestAllTiersEmptyIsNoOp:
    """All tiers empty is a clean no-op -- not a division by zero."""

    def test_all_empty_tiers_return_cleanly_without_dividing_by_zero(self) -> None:
        counter = FakeTokenCounter()
        tiers = [
            SelectedTier(name="hot", entries=[]),
            SelectedTier(name="ember", entries=[]),
        ]

        payload = pack(tiers, budget=800, counter=counter)

        assert [t.name for t in payload.tiers] == ["hot", "ember"]
        assert payload.budget_used == 0
        assert payload.skipped == 0
        for tier in payload.tiers:
            assert tier.entries == []
            assert tier.budget_used == 0
            assert tier.skipped == 0


class TestAllNonEmptySplitByteIdenticalToPreFix:
    """Explicit regression (curation block): when every tier has at
    least one entry, excluding empty tiers from the split changes nothing --
    every tier is a "non-empty index", so the share list is exactly the
    original unconditional ``divmod(budget, len(tiers))`` split. Pinned with
    hand-computed numbers, not merely re-asserting an existing waterfall
    test that happens to use non-empty tiers."""

    def test_all_non_empty_three_tier_split_matches_pre_fix_shares_exactly(self) -> None:
        counter = FakeTokenCounter()
        hot_entry = _fact(content="a" * 44)  # cost 11
        ember_entry = _fact(content="b" * 44)  # cost 11
        decision_entry = _fact(content="c" * 44)  # cost 11
        tiers = [
            SelectedTier(name="hot", entries=[hot_entry]),
            SelectedTier(name="ember", entries=[ember_entry]),
            SelectedTier(name="decision", entries=[decision_entry]),
        ]
        budget = 32  # divmod(32, 3) == (10, 2) -- pre-fix shares [11, 11, 10]

        payload = pack(tiers, budget=budget, counter=counter)

        hot_tier = next(t for t in payload.tiers if t.name == "hot")
        ember_tier = next(t for t in payload.tiers if t.name == "ember")
        decision_tier = next(t for t in payload.tiers if t.name == "decision")

        # hot's pre-fix share is 11 -- its cost-11 entry fits exactly.
        assert hot_tier.budget_used == 11
        assert hot_tier.skipped == 0
        # ember's pre-fix share is also 11 (the remainder's second unit).
        assert ember_tier.budget_used == 11
        assert ember_tier.skipped == 0
        # decision's pre-fix share is 10 -- its cost-11 entry does NOT fit
        # on share alone, and hot/ember spent their shares exactly (no
        # carry reaches decision) -- proving the split itself, not just the
        # total, is byte-identical to the pre-fix formula.
        assert decision_tier.entries == []
        assert decision_tier.budget_used == 0
        assert decision_tier.skipped == 1
        assert payload.budget_used == 22
        assert payload.skipped == 1


class TestPackDeterminism:
    def test_repeated_calls_are_identical(self) -> None:
        counter = FakeTokenCounter()
        tiers = [
            SelectedTier(name="hot", entries=[_fact(content="a" * 40), _fact(content="b" * 12)]),
            SelectedTier(name="ember", entries=[_fact(content="c" * 24)]),
            SelectedTier(name="decision", entries=[]),
        ]

        first = pack(tiers, budget=15, counter=counter)
        second = pack(tiers, budget=15, counter=counter)

        assert [[e.id for e in t.entries] for t in first.tiers] == [
            [e.id for e in t.entries] for t in second.tiers
        ]
        assert first.budget_used == second.budget_used
        assert first.skipped == second.skipped


# ===========================================================================
# Tabularium.context() — the engine method (SPEC §7 / §9.1)
#
# ``context(scope, budget, role) = pack(select_default(scope), budget,
# counter)`` -- this task's locked formula. Every engine here is built with
# an injected ``FakeTokenCounter`` (hermetic lane -- never a real
# ``LitellmTokenCounter``, tests/test_tokens.py's rule).
# ===========================================================================


async def _context_engine(cfg: EngineConfig | None = None) -> Tabularium:
    adapter = MemoryAdapter(embedding_model="test-model", dim=4, clock=TestClock(EPOCH))
    embedder = FakeEmbedder(dim=4)
    engine = Tabularium(
        adapter,
        embedder,
        TestClock(EPOCH),
        cfg if cfg is not None else _cfg(),
        token_counter=FakeTokenCounter(),
    )
    await engine.initialize()
    await adapter.create_scope(Scope("proj-a"))
    return engine


class TestEngineContextPopulated:
    async def test_populated_hot_tier_is_packed_into_the_payload(self) -> None:
        engine = await _context_engine()
        adapter = engine._adapter
        hot = _fact(scope="proj-a", lifecycle="hot", feedback_score=0.9, content="a" * 40)
        await adapter.upsert(hot, _vec())

        payload = await engine.context("proj-a", budget=1000, role="decider")

        assert isinstance(payload, ContextPayload)
        hot_tier = next(t for t in payload.tiers if t.name == "hot")
        assert [e.id for e in hot_tier.entries] == [hot.id]
        assert payload.budget_used == len("a" * 40) // 4
        assert payload.skipped == 0


class _AngleTableEmbedder:
    """Deterministic embedder: exact-string -> vector lookup at a known
    angle, so the invalidating Fact reliably survives the serial probe as
    the sole candidate against the Decision it corrects (lifted from
    ``tests/test_auto_exits.py``)."""

    dim: int = 4
    model: str = "test/context-grounding-invalidation"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


def _angle_vec4(angle_deg: float) -> list[float]:
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a), 0.0, 0.0]


class TestGroundingUnaffectedByInvalidation:
    """P9 invalid-Decision read contract: grounding stays Decision-free and
    byte-identical whether or not a Decision was ever invalidated and
    revalidated — a regression on the standing rule (CONTEXT.md, "Context
    composition"), not a new case. Drives P8's real write path
    (``EchoAssessor("correction")``, ``auto_exit_replacement=True``, a
    Decision save then a Fact save, then ``set_validity(valid=True)``) —
    never sets the invalidation columns by hand."""

    async def test_context_output_unchanged_by_invalidate_and_revalidate(self) -> None:
        decision_text = "the quick brown fox jumps over the lazy dog"
        fact_text = "the quick brown fox jumps over the lazy hound"
        table = {decision_text: _angle_vec4(0), fact_text: _angle_vec4(5)}
        embedder = _AngleTableEmbedder(table)
        clock = TestClock(EPOCH)
        adapter = MemoryAdapter(
            embedding_model=embedder.model, dim=embedder.dim, clock=clock
        )
        cfg = _cfg(candidate_k=1000, gray_floor=0.80, decision_gray_floor=0.70,
                    auto_exit_replacement=True)
        engine = Tabularium(
            adapter, embedder, clock, cfg,
            token_counter=FakeTokenCounter(), assessor=EchoAssessor("correction"),
        )
        await engine.initialize()
        await adapter.create_scope(Scope("proj-a"))

        hot = _fact(scope="proj-a", lifecycle="hot", feedback_score=0.9, content="a" * 40)
        await adapter.upsert(hot, _vec())

        payload_before = await engine.context("proj-a", budget=1000, role="decider")

        decision = await engine.save(decision_text, scope="proj-a", kind="decision")
        assert decision.hypomnema is not None
        fact = await engine.save(fact_text, scope="proj-a")
        assert fact.hypomnema is not None
        assert fact.outcome == "saved"
        decided = await adapter.get_by_id(decision.hypomnema.id)
        assert decided.validity == "invalid"

        revalidated = await engine.set_validity(
            decision.hypomnema.id, valid=True, reason="reviewed"
        )
        assert revalidated.outcome == "revised"

        payload_after = await engine.context("proj-a", budget=1000, role="decider")

        assert payload_before == payload_after
        assert payload_before.model_dump_json() == payload_after.model_dump_json()


class TestEngineContextScopeValidation:
    async def test_unregistered_scope_raises_tool_error(self) -> None:
        engine = await _context_engine()
        with pytest.raises(ToolError, match="Unregistered scope"):
            await engine.context("never-registered", budget=100, role="decider")

    async def test_registered_but_empty_scope_is_not_an_error(self) -> None:
        """§9.2 draws this line: registered-but-empty scope -> all tiers
        present and empty, no error at all -- never confused with an
        unregistered scope.

        Also the regression pin for the cleaner plan
        acceptance criterion: a scope with no Cleaner state gets
        ``cleaner=None`` and every other field on this payload is exactly
        what it was before that field existed (see
        ``tests/test_pass_report.py::TestContextIntegration`` for the fuller
        additive-only proof, including a scope WITH a completed pass)."""
        engine = await _context_engine()

        payload = await engine.context("proj-a", budget=100, role="decider")

        assert [t.name for t in payload.tiers] == ["hot", "ember"]
        for tier in payload.tiers:
            assert tier.entries == []
        assert payload.budget_used == 0
        assert payload.skipped == 0
        assert payload.cleaner is None


class TestEngineContextRoleValidation:
    async def test_unregistered_role_raises_tool_error(self) -> None:
        engine = await _context_engine()
        with pytest.raises(ToolError, match="Unregistered role"):
            await engine.context("proj-a", budget=100, role="never-registered")

    async def test_omitted_role_raises_tool_error_naming_registered_roles(self) -> None:
        """Curation block: ``role`` is now REQUIRED -- omitting it
        (or passing ``role=None`` explicitly) no longer resolves to the
        retired ``"default"`` fallback, it raises a ``ToolError`` naming
        every registered role. This is the exact silent failure that made
        jump the queue: before this fix, a role-less call returned
        an EMPTY payload with no error at all, because no real record was
        ever hinted the old ``"default"`` fallback.

        Falsifiability (step 6): restoring the retired
        ``role if role is not None else _DEFAULT_ROLE`` fallback in
        ``Tabularium.context`` makes this test's ``pytest.raises`` block
        fail (no exception raised) -- see the task verdict for the
        captured failure output.
        """
        engine = await _context_engine()
        with pytest.raises(ToolError, match="requires a role"):
            await engine.context("proj-a", budget=100, role=None)
        with pytest.raises(ToolError, match="requires a role"):
            await engine.context("proj-a", budget=100)

    async def test_omitted_role_error_names_the_four_registered_roles(self) -> None:
        engine = await _context_engine()
        with pytest.raises(ToolError) as exc_info:
            await engine.context("proj-a", budget=100)
        msg = str(exc_info.value)
        for role in ("decider", "builder", "reviewer", "navigator"):
            assert role in msg

    async def test_two_registered_roles_return_identical_payloads(self) -> None:
        """The Curator's configurability is deferred (this plan's locked
        decision) -- with every candidate UNHINTED, every registered role
        resolves to the same (empty) selection: explicit enumeration only
        (curation block) means an unhinted record reaches NO role,
        so ``decider`` and ``implementer`` agree by both being empty, not
        by both being full -- see ``TestNoUniversalDefault`` below for the
        dedicated property. This test pins the unhinted case so the day a
        hinted record makes two roles differ on purpose, this test is the
        thing that has to be updated deliberately, not silently broken."""
        engine = await _context_engine()
        adapter = engine._adapter
        await adapter.create_role("implementer")
        hot = _fact(scope="proj-a", lifecycle="hot", feedback_score=0.9, audience_hint=None)
        ember = _fact(
            scope="proj-a", lifecycle="ember", feedback_score=0.9, audience_hint=None
        )
        for rec in (hot, ember):
            await adapter.upsert(rec, _vec())

        decider_payload = await engine.context("proj-a", budget=1000, role="decider")
        implementer_payload = await engine.context("proj-a", budget=1000, role="implementer")

        assert decider_payload == implementer_payload


# ===========================================================================
# Per-role grounding — routing on push (ADR 0016 amendment), untouched on
# pull (this brief, Scope 3)
# ===========================================================================


class TestUniversality:
    """Explicit enumeration only (ADR 0016's further amendment, curation
    block): "for everyone" is now spelled by naming every role in
    ``audience_hint``, not by omission. A Fact tagged with all four
    registered roles reaches all four. Its mirror -- an untagged Fact
    reaches none of them -- is ``TestNoUniversalDefault`` below; the two
    classes together are this task's rewritten universality regression
    (the old version pinned the now-retired "absence means universal"
    rule).

    Falsifiability (step 6): both breaks below turned this red.
    - Invert ``routes_to``'s return value (``src/tabularium/engine/
      context.py``) — the all-roles-tagged Fact then routes to NO role.
    - Restore the retired ``not rec.audience_hint or`` short-circuit —
      inert here (the Fact is hinted either way), but see
      ``TestNoUniversalDefault`` for the break this predicate change is
      actually pinned against.
    See the run verdict for the captured failure output.
    """

    async def test_fact_tagged_all_four_roles_reaches_all_four_hot_tier(self) -> None:
        engine = await _context_engine()
        adapter = engine._adapter
        roles = ("decider", "builder", "reviewer", "navigator")
        hot = _fact(
            scope="proj-a", lifecycle="hot", feedback_score=0.9, audience_hint=roles
        )
        await adapter.upsert(hot, _vec())

        for role in roles:
            payload = await engine.context("proj-a", budget=1000, role=role)
            hot_ids = {
                e.id for e in next(t for t in payload.tiers if t.name == "hot").entries
            }
            assert hot.id in hot_ids, f"all-roles-tagged hot Fact missing for role={role!r}"

    async def test_fact_tagged_all_four_roles_reaches_all_four_ember_tier(self) -> None:
        engine = await _context_engine()
        adapter = engine._adapter
        roles = ("decider", "builder", "reviewer", "navigator")
        ember = _fact(
            scope="proj-a", lifecycle="ember", feedback_score=0.9, audience_hint=roles
        )
        await adapter.upsert(ember, _vec())

        for role in roles:
            payload = await engine.context("proj-a", budget=1000, role=role)
            ember_ids = {
                e.id for e in next(t for t in payload.tiers if t.name == "ember").entries
            }
            assert ember.id in ember_ids, f"all-roles-tagged ember missing for role={role!r}"


class TestNoUniversalDefault:
    """The mirror property (this task's acceptance criteria): an untagged
    Fact — ``audience_hint`` empty or ``NULL`` — reaches NO registered
    role's payload, on both push tiers. The pre-existing default was the
    opposite (absence meant universal reach); this class pins its
    replacement.

    Falsifiability (step 6): restoring the retired ``not rec.audience_hint
    or`` short-circuit in ``routes_to`` makes the untagged Fact route to
    every role again, and both tests below go red.
    """

    async def test_untagged_hot_fact_reaches_no_registered_role(self) -> None:
        engine = await _context_engine()
        adapter = engine._adapter
        for role in ("decider", "builder", "reviewer", "navigator"):
            await adapter.create_role(role)
        hot = _fact(scope="proj-a", lifecycle="hot", feedback_score=0.9, audience_hint=None)
        await adapter.upsert(hot, _vec())

        for role in ("decider", "builder", "reviewer", "navigator"):
            payload = await engine.context("proj-a", budget=1000, role=role)
            hot_ids = {
                e.id for e in next(t for t in payload.tiers if t.name == "hot").entries
            }
            assert hot.id not in hot_ids, f"untagged hot Fact reached role={role!r}"

    async def test_untagged_ember_reaches_no_registered_role(self) -> None:
        engine = await _context_engine()
        adapter = engine._adapter
        for role in ("decider", "builder", "reviewer", "navigator"):
            await adapter.create_role(role)
        ember = _fact(
            scope="proj-a", lifecycle="ember", feedback_score=0.9, audience_hint=None
        )
        await adapter.upsert(ember, _vec())

        for role in ("decider", "builder", "reviewer", "navigator"):
            payload = await engine.context("proj-a", budget=1000, role=role)
            ember_ids = {
                e.id for e in next(t for t in payload.tiers if t.name == "ember").entries
            }
            assert ember.id not in ember_ids, f"untagged ember reached role={role!r}"


class TestNarrowing:
    """A Fact pinned ``audience_hint=("reviewer",)`` appears for
    ``reviewer`` and is absent for every other registered role — asserted
    against both the hot and ember tiers.

    Falsifiability (step 6): both breaks below turned this red.
    - Invert ``routes_to`` — the hinted Fact then routes to every OTHER
      role and disappears from its own.
    - Hardcode the role forwarded at ``core.py:678`` back to a fixed
      fallback role (reproducing the pre-slice "validated but inert" bug —
      see the module's Problem section) — asking for ``role="reviewer"``
      silently selects as something else and the hinted Fact never appears
      even under its own role.
    See the run verdict for the captured failure output.
    """

    async def test_hinted_fact_reaches_only_its_role_hot_tier(self) -> None:
        engine = await _context_engine()
        adapter = engine._adapter
        await adapter.create_role("reviewer")
        await adapter.create_role("builder")
        hinted = _fact(
            scope="proj-a",
            lifecycle="hot",
            feedback_score=0.9,
            audience_hint=("reviewer",),
        )
        await adapter.upsert(hinted, _vec())

        reviewer = await engine.context("proj-a", budget=1000, role="reviewer")
        decider = await engine.context("proj-a", budget=1000, role="decider")
        builder = await engine.context("proj-a", budget=1000, role="builder")
        navigator = await engine.context("proj-a", budget=1000, role="navigator")

        def _hot_ids(payload: ContextPayload) -> set[HypomnemaId]:
            return {
                e.id for e in next(t for t in payload.tiers if t.name == "hot").entries
            }

        assert hinted.id in _hot_ids(reviewer)
        # "Absent for the other three" (acceptance criteria) -- decider,
        # builder, navigator, the three registered roles besides reviewer.
        assert hinted.id not in _hot_ids(decider)
        assert hinted.id not in _hot_ids(builder)
        assert hinted.id not in _hot_ids(navigator)

    async def test_hinted_fact_reaches_only_its_role_ember_tier(self) -> None:
        engine = await _context_engine()
        adapter = engine._adapter
        await adapter.create_role("reviewer")
        await adapter.create_role("builder")
        hinted = _fact(
            scope="proj-a",
            lifecycle="ember",
            feedback_score=0.9,
            audience_hint=("reviewer",),
        )
        await adapter.upsert(hinted, _vec())

        reviewer = await engine.context("proj-a", budget=1000, role="reviewer")
        decider = await engine.context("proj-a", budget=1000, role="decider")
        builder = await engine.context("proj-a", budget=1000, role="builder")
        navigator = await engine.context("proj-a", budget=1000, role="navigator")

        def _ember_ids(payload: ContextPayload) -> set[HypomnemaId]:
            return {
                e.id for e in next(t for t in payload.tiers if t.name == "ember").entries
            }

        assert hinted.id in _ember_ids(reviewer)
        # "Absent for the other three" (acceptance criteria) -- decider,
        # builder, navigator, the three registered roles besides reviewer.
        assert hinted.id not in _ember_ids(decider)
        assert hinted.id not in _ember_ids(builder)
        assert hinted.id not in _ember_ids(navigator)


class TestPullUntouched:
    """ADR 0016's line that must not move: a hinted Fact is still returned
    by ``recall`` for a caller that is not the hinted role — ``audience_hint``
    never touches pull, only the two push tiers.

    Falsifiability (step 6): inject a filter into ``engine/recall.py`` that
    drops any hit whose ``audience_hint`` is set — the hinted Fact then
    disappears from ``recall``'s hits. Applied and reverted by hand during
    development; see the run verdict for the captured failure output.
    """

    async def test_hinted_fact_still_recallable(self) -> None:
        engine = await _context_engine()
        await engine._adapter.create_role("reviewer")
        content = "a fact hinted to the reviewer role only, pull must not filter it"
        result = await engine.save(content, scope="proj-a", audience_hint=["reviewer"])
        assert result.hypomnema is not None

        recalled = await engine.recall(content, scope="proj-a")
        hit_ids = {h.id for h in recalled.hits}
        assert result.hypomnema.id in hit_ids, (
            "audience_hint must never filter recall — ADR 0016 stays "
            "additive-only on pull"
        )


class TestEngineContextBudgetValidation:
    async def test_negative_budget_raises_value_error(self) -> None:
        """Defense-in-depth for direct (non-tool) callers -- the MCP tool
        itself rejects a negative budget one layer up, at the pydantic/
        FastMCP boundary (see tests/test_server.py)."""
        engine = await _context_engine()
        with pytest.raises(ValueError, match="budget must be >= 0"):
            await engine.context("proj-a", budget=-1, role="decider")
