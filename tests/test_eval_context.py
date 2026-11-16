"""Composition determinism evals — T01 harness extension (SPEC §14, BUILD-PLAN §M2).

The eval line for `context(scope, budget, role) = pack(
select_default(scope), budget, counter)`, minus the profile-swapping half that
the Curator deferral removes. This
lands alongside `tests/test_eval_b5.py`'s recall/dedup floors as the same
substrate: a golden corpus, an injectable `Clock`, event-log replay,
`trajectory = f(event_log, clock)` — extended here from a single record's
projections ([A9]) to a whole tier's ORDERED selection.

Determinism here is a floor, not a measurement (unlike recall@10's tolerance
band): the same store and clock either produce the identical ordered id list,
or the harness is lying about reproducibility. Every test below therefore
asserts an exact ordered id list, never a set or a count.

the amendment (2026-08-01): `select_default` dropped the in-force
Decision tier — grounding ships `hot` + `ember` only; Decisions are pull-only
at session start via `decisions_query`/`decision` (SPEC §7 amendment). This
module's corpus and assertions were updated accordingly: the golden corpus
below covers the two tiers that remain (facets, both scopes, the fade floor,
the documented tie-break) and no longer builds Decision records — a Decision
in the store was never reachable through `select_default` in the first place
(the `hot`/`ember` scans are Fact-only by construction), so a Decision fixture
proved nothing about this module's own selection logic.

Why no `tests/fixtures/frozen_vectors.json`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Task step 1 says to reuse the frozen-vector fixture "where embeddings are
needed rather than generating new ones". `select_default` is query-less (SPEC
§7) — it never calls `vector_search`/`text_search` and never touches an
embedder — so no real embedding is needed anywhere in this module. Every
record is upserted directly through the storage seam with a fixed
placeholder vector (`_vec()`), the same convention `tests/test_context.py`
already established for this exact code path. This is hermetic by
construction, not merely by config: there is no embedder object anywhere in
this file for Ollama/network to reach.

`hot` membership
~~~~~~~~~~~~~~~~
Per the task's CONTEXT (automatic hot-promotion was built and ripped out
2026-07-31, ADR 0020): every `hot`-lifecycle row in this module's corpus is
written directly through the storage seam (`adapter.upsert`), never earned
via a heuristic and never via `set_hot` (which would additionally enforce
`hot_cap`, irrelevant to a selection-composition fixture) — the same
adapter-direct convention `tests/test_context.py` and the plan note
("its tests seed hot rows through the adapter directly rather than earning
them") already use.

Falsifiability
~~~~~~~~~~~~~~
`TestReopenDeterminism`'s corpus deliberately inserts records in an order
that does NOT match their importance/recency/id ranking (see
`_build_corpus`'s "scrambled insertion order" comment) specifically so that a
regression which drops `_select_hot`'s `str(id)` tie-break — falling back to
whatever order `adapter.scan()` happens to yield, which `engine/context.py`'s
own module docstring already documents LanceDB does not guarantee — produces
a WRONG exact-match assertion, not merely a self-inconsistent one. The
mutation was applied by hand against `src/tabularium/engine/context.py`
during development and reverted; it is not committed here. See the task
verdict for the captured failure output.
"""

from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.engine import Tabularium
from tabularium.engine.backfill import NoBackfill
from tabularium.engine.context import SelectedTier, pack, select_default
from tabularium.engine.engine_context import EngineContext
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import ReinforcementEvent

# Adapter-construction pattern reused verbatim from the conformance suite
# (precedent — see tests/test_eval_b5.py / tests/test_hot.py) rather
# than a fourth construction pattern.
from tests.conformance.test_conformance import _make_lancedb_adapter
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock

AnyAdapter = MemoryAdapter | LanceDBAdapter

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
SCOPE_X = Scope("proj-eval")
SCOPE_OTHER = Scope("other-scope")


# ---------------------------------------------------------------------------
# Config / fixture builders
# ---------------------------------------------------------------------------


def _ctx(adapter: AnyAdapter, cfg: EngineConfig, clock: TestClock) -> EngineContext:
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
    """Deterministic-importance config: importance == feedback_score exactly
    (same idiom as tests/test_context.py) plus an explicit fade floor pair so
    the ember facet is unambiguous regardless of EngineConfig's own defaults.
    """
    defaults: dict[str, object] = {
        "w_recall": 0.0,
        "w_recency": 0.0,
        "w_feedback": 1.0,
        "archive_threshold": 0.30,
        "revival_threshold": 0.35,
    }
    defaults.update(kw)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


def _vec(dim: int = 4) -> list[float]:
    return [1.0] * dim


def _fact(
    *,
    scope: str = "proj-eval",
    lifecycle: str = "hot",
    feedback_score: float = 0.0,
    validity: str = "valid",
    content: str = "eval context corpus fact",
    created_at: datetime = EPOCH,
    last_referenced_at: datetime | None = None,
    audience_hint: tuple[str, ...] | None = ("decider",),
) -> Fact:
    """Defaults to ``audience_hint=("decider",)`` (curation block
    for the routing semantics; retargeted off the retired ``"default"``
    role): explicit enumeration only means an unhinted Fact
    reaches NO role on the push tiers, and this module's corpus is
    selected with ``role="decider"`` throughout — tagging it a real,
    always-registered role keeps these determinism/ordering evals
    decoupled from the routing feature itself (pinned in
    ``tests/test_context.py``), without depending on the now-retired
    ``"default"`` fallback."""
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


def _tier(tiers: list[SelectedTier], name: str) -> SelectedTier:
    for t in tiers:
        if t.name == name:
            return t
    raise AssertionError(f"no tier named {name!r} in {[t.name for t in tiers]!r}")


# ---------------------------------------------------------------------------
# The faceted golden corpus (acceptance criterion 1)
#
# Covers: both tiers (the change that dropped the in-force Decision tier — SPEC §7
# amendment 2026-08-01), both scopes (proj-eval + global), records spanning
# the fade floor (archive_threshold=0.30) and the hot thresholds
# (hot_demote_threshold=0.40 / hot_promote_threshold=0.60, EngineConfig
# defaults — dead M3 config today, per the task's CONTEXT, but still the
# documented span the hot tier's importance-ranking must order correctly
# across), one tie requiring the documented str(id) tie-break, and one
# excluded representative per exclusion axis (validity, scope, archived,
# buried) so the corpus also proves what must NOT appear, not only what must.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Corpus:
    hot_a: Fact
    hot_b: Fact
    hot_tie_lower: Fact
    hot_tie_higher: Fact
    hot_c: Fact
    hot_invalid: Fact
    hot_other_scope: Fact
    hot_buried: Fact
    archived: Fact
    ember_below: Fact
    ember_at_floor: Fact
    ember_above_low: Fact
    ember_above_high: Fact
    ember_other_scope: Fact
    ember_invalid: Fact


async def _build_corpus(adapter: AnyAdapter) -> _Corpus:
    """Insert the faceted golden corpus into ``adapter`` and return the
    inserted records (ids come from the records themselves — never a fixed
    literal — so every assertion in this module compares against what was
    actually built, the same convention ``tests/test_context.py`` uses).
    """
    hot_a = _fact(
        scope="global",
        feedback_score=0.80,  # above hot_promote_threshold (0.60)
        last_referenced_at=EPOCH,
        content="hot-a: above hot_promote_threshold, scope=global",
    )
    hot_b = _fact(
        scope=SCOPE_X,
        feedback_score=0.55,  # between hot thresholds (0.40 / 0.60)
        last_referenced_at=EPOCH - timedelta(days=1),
        content="hot-b: between hot thresholds, scope=proj-eval",
    )
    tie_1 = _fact(scope=SCOPE_X, feedback_score=0.50, content="hot-tie candidate 1")
    tie_2 = _fact(scope=SCOPE_X, feedback_score=0.50, content="hot-tie candidate 2")
    hot_tie_lower, hot_tie_higher = sorted((tie_1, tie_2), key=lambda f: str(f.id))
    hot_c = _fact(
        scope="global",
        feedback_score=0.20,  # below hot_demote_threshold (0.40)
        last_referenced_at=EPOCH - timedelta(days=2),
        content="hot-c: below hot_demote_threshold, scope=global",
    )
    hot_invalid = _fact(
        scope=SCOPE_X,
        feedback_score=0.99,
        validity="invalid",
        content="hot-invalid: excluded by validity",
    )
    hot_other_scope = _fact(
        scope=SCOPE_OTHER,
        feedback_score=0.99,
        content="hot-other-scope: excluded, not in the proj-eval read span",
    )
    hot_buried = _fact(
        scope=SCOPE_X,
        feedback_score=0.99,
        content="hot-buried: excluded, content gone",
    )
    archived = _fact(
        scope=SCOPE_X,
        lifecycle="archived",
        feedback_score=0.99,
        content="archived: excluded from every tier structurally",
    )

    ember_below = _fact(
        scope=SCOPE_X,
        lifecycle="ember",
        feedback_score=0.20,  # below the fade floor (0.30)
        content="ember-below: below the fade floor, excluded",
    )
    ember_at_floor = _fact(
        scope="global",
        lifecycle="ember",
        feedback_score=0.30,  # == the fade floor -> not "above", excluded
        content="ember-at-floor: at the fade floor, excluded (strict above)",
    )
    ember_above_low = _fact(
        scope=SCOPE_X,
        lifecycle="ember",
        feedback_score=0.45,
        content="ember-above-low: above the fade floor",
    )
    ember_above_high = _fact(
        scope="global",
        lifecycle="ember",
        feedback_score=0.90,
        content="ember-above-high: above the fade floor",
    )
    ember_other_scope = _fact(
        scope=SCOPE_OTHER,
        lifecycle="ember",
        feedback_score=0.99,
        content="ember-other-scope: excluded, not in the proj-eval read span",
    )
    ember_invalid = _fact(
        scope=SCOPE_X,
        lifecycle="ember",
        feedback_score=0.99,
        validity="invalid",
        content="ember-invalid: excluded by validity",
    )

    vec = _vec()
    # Deliberately scrambled insertion order — NOT importance, recency, or id
    # order. This is the point: a correct `select_default` produces the same
    # exact ordered list no matter what order rows were written in, because
    # its own explicit sort is load-bearing, never a convenience (see
    # engine/context.py's module docstring). The tied pair is inserted
    # higher-id-first specifically so a dropped tie-break (this task's
    # falsifiability mutation) would surface as a wrong exact-match
    # assertion, not a merely-self-inconsistent one.
    facts_in_scrambled_order = [
        hot_c,
        ember_above_high,
        hot_tie_higher,
        hot_tie_lower,
        ember_invalid,
        hot_a,
        ember_below,
        archived,
        hot_other_scope,
        ember_at_floor,
        hot_b,
        ember_other_scope,
        hot_invalid,
        ember_above_low,
        hot_buried,
    ]
    for f in facts_in_scrambled_order:
        await adapter.upsert(f, vec)
    await adapter.bury(hot_buried.id, "eval corpus: buried record must never appear")

    return _Corpus(
        hot_a=hot_a,
        hot_b=hot_b,
        hot_tie_lower=hot_tie_lower,
        hot_tie_higher=hot_tie_higher,
        hot_c=hot_c,
        hot_invalid=hot_invalid,
        hot_other_scope=hot_other_scope,
        hot_buried=hot_buried,
        archived=archived,
        ember_below=ember_below,
        ember_at_floor=ember_at_floor,
        ember_above_low=ember_above_low,
        ember_above_high=ember_above_high,
        ember_other_scope=ember_other_scope,
        ember_invalid=ember_invalid,
    )


def _expected_hot_order(c: _Corpus) -> list[HypomnemaId]:
    """importance DESC (0.80, 0.55, 0.50 tie, 0.50 tie, 0.20), tie -> str(id) asc."""
    return [c.hot_a.id, c.hot_b.id, c.hot_tie_lower.id, c.hot_tie_higher.id, c.hot_c.id]


def _expected_ember_order(c: _Corpus) -> list[HypomnemaId]:
    """importance DESC among strictly-above-floor embers (0.90, 0.45)."""
    return [c.ember_above_high.id, c.ember_above_low.id]


def _all_excluded_ids(c: _Corpus) -> set[HypomnemaId]:
    return {
        c.hot_invalid.id,
        c.hot_other_scope.id,
        c.hot_buried.id,
        c.archived.id,
        c.ember_below.id,
        c.ember_at_floor.id,
        c.ember_other_scope.id,
        c.ember_invalid.id,
    }


# ---------------------------------------------------------------------------
# Criterion 1 + 2: faceted corpus -> exact ordered id list per tier
# ---------------------------------------------------------------------------


class TestFacetedGoldenCorpusOrderedSelection:
    async def test_ordered_selection_exact_per_tier(self, tmp_path: pathlib.Path) -> None:
        adapter = await _make_lancedb_adapter(tmp_path, dim=4)
        corpus = await _build_corpus(adapter)
        cfg = _cfg()
        clock = TestClock(EPOCH)

        tiers = await select_default(_ctx(adapter, cfg, clock), str(SCOPE_X), "decider")

        hot_ids = [e.id for e in _tier(tiers, "hot").entries]
        ember_ids = [e.id for e in _tier(tiers, "ember").entries]

        assert hot_ids == _expected_hot_order(corpus)
        assert ember_ids == _expected_ember_order(corpus)

        all_selected = set(hot_ids) | set(ember_ids)
        assert not (_all_excluded_ids(corpus) & all_selected), (
            "an excluded-facet record leaked into a presented tier"
        )


# ---------------------------------------------------------------------------
# Criterion 3 + 4 + 8: determinism across a store reopen, exact expected
# output — the falsifiability target (see module docstring).
# ---------------------------------------------------------------------------


class TestReopenDeterminism:
    async def test_write_close_reopen_reselect_identical_and_exact(
        self, tmp_path: pathlib.Path
    ) -> None:
        store_path = tmp_path / "eval_context_store"
        adapter1 = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=4
        )
        corpus = await _build_corpus(adapter1)
        cfg = _cfg()

        before = await select_default(
            _ctx(adapter1, cfg, TestClock(EPOCH)), str(SCOPE_X), "decider"
        )
        before_ids = {t.name: [e.id for e in t.entries] for t in before}

        del adapter1  # close

        adapter2 = await LanceDBAdapter.open(
            str(store_path), embedding_model="test-model", dim=4
        )
        after = await select_default(
            _ctx(adapter2, cfg, TestClock(EPOCH)), str(SCOPE_X), "decider"
        )
        after_ids = {t.name: [e.id for e in t.entries] for t in after}

        # Determinism across reopen: identical before/after.
        assert after_ids == before_ids
        # Not merely self-consistent — exactly the application-level-correct
        # order (this is what the falsifiability mutation actually breaks;
        # see the module docstring: reopen alone does not reshuffle
        # LanceDB's physical row order, so a self-consistency-only assertion
        # would NOT catch a dropped tie-break here).
        assert before_ids["hot"] == _expected_hot_order(corpus)
        assert before_ids["ember"] == _expected_ember_order(corpus)


# ---------------------------------------------------------------------------
# Criterion 5: replay determinism — [A9]'s projection-determinism property,
# extended to composition. The SAME event log, replayed through the SAME
# clock, into TWO INDEPENDENT stores (of different adapter implementations,
# for good measure) yields the SAME ordered selection.
# ---------------------------------------------------------------------------

# fact index -> hours-after-EPOCH at which a "dedup" reinforcement event
# fires. Index 0 gets 3 events (recall_count=3), index 1 gets 1 (recall_count
# =1), index 2 gets none (recall_count=0, last_referenced_at stays None ->
# falls back to created_at=EPOCH) — three strictly-different recall_counts so
# the resulting order has no tie to worry about; the tie-break property is
# already covered by the golden-corpus tests above.
_EVENT_HOURS: dict[int, list[int]] = {0: [1, 2, 3], 1: [1], 2: []}


async def _build_replay_shells(adapter: AnyAdapter, ids: list[HypomnemaId]) -> None:
    for fid in ids:
        shell = Fact(
            id=fid,
            type="fact",
            content=f"replay shell {fid}",
            scope=Scope("global"),
            lifecycle="hot",
            validity="valid",
            feedback_score=0.0,
            created_at=EPOCH,
            updated_at=EPOCH,
            # Curation block: explicit enumeration only -- an
            # unhinted shell would reach no role, and this module's
            # selection calls all query role="decider" (retargets
            # this off the retired "default" role).
            audience_hint=("decider",),
        )
        await adapter.upsert(shell, _vec())


async def _replay_event_log(adapter: AnyAdapter, ids: list[HypomnemaId]) -> None:
    """Append ``_EVENT_HOURS`` onto already-upserted shell Facts, then
    recompute + persist projections from the event log — the [A9] idiom
    (``tests/test_eval_b5.py::TestProjectionDeterminism``), applied here to
    drive composition rather than only re-verify a single record's fields.
    """
    for idx, fid in enumerate(ids):
        for hours in _EVENT_HOURS[idx]:
            await adapter.append_event(
                ReinforcementEvent(target_id=fid, at=EPOCH + timedelta(hours=hours), source="dedup")
            )
        events = await adapter.query_events(target_id=fid)
        dedup_events = [e for e in events if e.source in {"dedup", "revival", "compact"}]
        ref_events = [e for e in events if e.source in {"dedup", "revival"}]
        await adapter.update_projections(
            fid,
            recall_count=len(dedup_events),
            last_referenced_at=max((e.at for e in ref_events), default=None),
            feedback_score=0.0,
            decay_due_at=None,
        )


class TestReplayDeterminism:
    async def test_same_event_log_same_clock_same_selection_across_adapters(
        self, tmp_path: pathlib.Path
    ) -> None:
        ids = [new_id(), new_id(), new_id()]
        # w_recall isolates recall_count (event-log-driven) as the only
        # importance term -- a clean, event-count-driven ordering.
        cfg = _cfg(w_recall=1.0, w_recency=0.0, w_feedback=0.0)
        clock = TestClock(EPOCH)

        memory_adapter = MemoryAdapter(embedding_model="test-model", dim=4)
        await _build_replay_shells(memory_adapter, ids)
        await _replay_event_log(memory_adapter, ids)

        lance_adapter = await _make_lancedb_adapter(tmp_path, dim=4)
        await _build_replay_shells(lance_adapter, ids)
        await _replay_event_log(lance_adapter, ids)

        memory_tiers = await select_default(_ctx(memory_adapter, cfg, clock), "global", "decider")
        lance_tiers = await select_default(_ctx(lance_adapter, cfg, clock), "global", "decider")

        # recall_count 3 > 1 > 0 -> ln(1+recall_count) strictly decreasing.
        expected = [ids[0], ids[1], ids[2]]
        memory_hot_ids = [e.id for e in _tier(memory_tiers, "hot").entries]
        lance_hot_ids = [e.id for e in _tier(lance_tiers, "hot").entries]

        assert memory_hot_ids == expected
        assert lance_hot_ids == expected


# ---------------------------------------------------------------------------
# Criterion 6: packing determinism at a fixed budget.
# ---------------------------------------------------------------------------


class TestPackingDeterminism:
    async def test_same_corpus_same_budget_same_skipped_and_budget_used(
        self, tmp_path: pathlib.Path
    ) -> None:
        adapter = await _make_lancedb_adapter(tmp_path, dim=4)
        await _build_corpus(adapter)
        cfg = _cfg()
        tiers = await select_default(_ctx(adapter, cfg, TestClock(EPOCH)), str(SCOPE_X), "decider")
        counter = FakeTokenCounter()

        first = pack(tiers, budget=60, counter=counter)
        second = pack(tiers, budget=60, counter=counter)

        assert first.skipped == second.skipped
        assert first.budget_used == second.budget_used
        assert [[e.id for e in t.entries] for t in first.tiers] == [
            [e.id for e in t.entries] for t in second.tiers
        ]
        for t1, t2 in zip(first.tiers, second.tiers, strict=True):
            assert t1.name == t2.name
            assert t1.skipped == t2.skipped
            assert t1.budget_used == t2.budget_used


# ---------------------------------------------------------------------------
# Full-pipeline sanity: Tabularium.context() == pack(select_default(...))
# composed end to end, twice, byte-identical (criteria 3 + 6 at the public
# engine-method level, not just the two composed functions in isolation).
# ---------------------------------------------------------------------------


class TestEngineContextDeterminism:
    async def test_repeated_context_calls_are_identical(self) -> None:
        adapter = MemoryAdapter(embedding_model="test-model", dim=4)
        embedder = FakeEmbedder(dim=4)
        clock = TestClock(EPOCH)
        cfg = _cfg()
        engine = Tabularium(adapter, embedder, clock, cfg, token_counter=FakeTokenCounter())
        await engine.initialize()
        await adapter.create_scope(SCOPE_X)

        await _build_corpus(adapter)

        first = await engine.context(str(SCOPE_X), budget=500, role="decider")
        second = await engine.context(str(SCOPE_X), budget=500, role="decider")

        assert first == second
