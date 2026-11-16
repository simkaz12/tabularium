"""The Cleaner's falsity axis (2026-08-10 cleaner plan).

Hermetic lane only — ``MemoryAdapter`` + ``FakeJudgeAdapter``/local test doubles + a deterministic
table embedder, no network, no credential, no real ``claude``/``codex`` process and no real litellm
provider call anywhere in this file (same convention as ``test_relevance_axis.py`` /
``test_gate_probe.py`` / ``test_judge_seam.py``).

Falsifiability (Verify steps 1/2 of the task — both breaks target PRODUCTION code in
``src/tabularium/engine/falsity.py``, never a test-local schema or helper):

1. Edge-direction inversion: in ``run_falsity_axis``'s write call, swap
   ``id=candidate.id, superseded_by=newcomer.id`` to
   ``id=newcomer.id, superseded_by=candidate.id`` and re-run ``TestEdgeDirection``. It goes red
   because the stored OLDER record (``O``) comes back unchanged (``validity="valid"``,
   ``superseded_by=None``) while the NEWER record (``N``) is the one left invalid and pointing at
   ``O`` — exactly backwards. ``supersede`` itself enforces no age ordering (only
   kind/self-edge/validity/already-superseded), so this break is NOT absorbed by an incidental
   guard — the call succeeds either way, just with the wrong record in each slot.
2. Global candidate reaches the write path: in ``_generate_pairs``, comment out the
   ``if candidate.scope == _GLOBAL: ... continue`` branch and re-run
   ``TestGlobalCandidateNeverWritten``'s isolation test (the one whose name mentions "span is
   neutralized" — see that class for the exact method name).
   Production ``supersede`` DOES receive ``span=[scope]`` on the real write call (a second,
   independent guard — SPEC-cited cross-scope protection) — but that guard
   is never weakened in production to make this break visible. Instead, the isolation test itself
   monkeypatches ``tabularium.engine.falsity.supersede`` with a spy that strips the ``span``
   kwarg before delegating to the real function, so THIS TEST's assertions are attributable to
   ``_generate_pairs``'s own ``scope == "global"`` classification alone. Under the deliberate break
   the store accepts the write (no exception at all — the isolated span guard cannot fire because
   the spy stripped it), so the test goes red on a plain ``AssertionError`` (a real edge appears in
   the summary) rather than on a ``ToolError`` from the OTHER guard. A companion test,
   ``TestSpanGuardOnWriteCall::test_write_call_passes_span_scoped_to_the_pass``, pins — on
   unmodified production code, no break applied — that the write call really does pass
   ``span=[scope]``, so that guard's continued presence is independently verified rather than
   assumed. Two tests, two guards, neither weakened in production (the generalised rule: when two
   independent guards protect the same property, prove each one separately instead of removing one
   so the other's absence becomes visible).
3. Hot/ember newcomer let through (2026-08-12 fix, N side): revert
   ``_select_falsity_newcomers``'s ``if record.lifecycle not in ("active", "archived") or
   record.id in excluded_ids:`` back to ``if record.lifecycle == "shelved" or record.id in
   excluded_ids:`` and re-run ``TestHotEmberNewcomerExcluded``. Unlike the relevance axis's own
   break 3 (its sibling module), this does NOT raise at all — ``supersede`` has no lifecycle
   guard on the record it writes AS the successor, so a hot/ember newcomer silently supersedes its
   older candidate (confirmed by direct reproduction: the older record ends up
   ``validity="invalid"``, ``superseded_by=<hot Fact's id>``, no exception anywhere). The test goes
   red on a plain ``AssertionError`` (``summary.excluded_shelved_this_pass`` comes back empty
   instead of naming the hot/ember newcomer) — this is the "quieter half" of the same owner
   decision the module docstring's "What is detected but never written" section describes.
4. Hot/ember candidate reaches the write path (2026-08-12 fix, O side): in ``_generate_pairs``,
   comment out the ``if candidate.lifecycle not in ("active", "archived"): ... continue`` branch
   and re-run ``TestGeneratePairs::test_hot_candidate_routed_to_report`` (goes red at the
   ``assert result.viable_pairs == ()`` line — the hot Fact lands in ``viable_pairs`` instead of
   being classified out) and ``TestHotEmberCandidateNeverWritten`` (goes red at
   ``assert summary.edges == ()`` — a real ``FalsityEdge`` is drawn, confirmed by direct
   reproduction to actually supersede the hot Fact in the store). Same "no exception, silent write"
   shape as break 3 — ``supersede`` has no lifecycle guard on the PREDECESSOR either.
5. Split-band auto-apply: make ``_classify_pair`` return
   ``"replace"`` for one ``replace`` vote and one ``coexist`` vote (restoring the pre-change
   policy where the more aggressive judge governed a split) and re-run ``TestBandPolicy``'s two
   split integration tests — both go red because the older record ends up
   ``validity="invalid"``/``superseded_by=<newcomer.id>`` (an irreversible, unattended edge drawn
   on a disagreement) instead of staying ``valid`` with the pair routed to ``contested_pairs``.

Both commands and their captured red output are recorded verbatim in the task verdict, not
duplicated here (matching ``test_relevance_axis.py``'s own convention).
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, FactLifecycle, Scope, new_id
from tabularium.engine import falsity as falsity_mod
from tabularium.engine._errors import ToolError
from tabularium.engine.assessor import ContradictionVerdict, Trace
from tabularium.engine.dispositions import supersede as _real_supersede
from tabularium.engine.dispositions import unsupersede
from tabularium.engine.falsity import (
    FalsityAxisSummary,
    FalsityBandCounts,
    FalsityContestedPair,
    FalsityVerdict,
    _deciding_verdict,
    _generate_pairs,
    _select_falsity_newcomers,
    run_falsity_axis,
)
from tabularium.engine.rubric import Rubric, RubricParameters, load_rubric
from tabularium.engine.verdicts import reconcile_pending_verdicts
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeJudgeAdapter, TestClock
from tests.support.fake_assessor import EchoAssessor
from tests.support.queue import contested_ids, latest_judgment

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIM = 16

_SINCE = datetime(2024, 12, 31, tzinfo=UTC)
_OLD = datetime(2024, 6, 1, tzinfo=UTC)
_NEW = datetime(2025, 1, 2, tzinfo=UTC)
_NEWER_STILL = datetime(2025, 1, 3, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adapter(dim: int = _DIM) -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic", dim=dim)


def _make_clock() -> TestClock:
    return TestClock(_NEW)


def _angle_vec(angle_deg: float, *, offset: int = 0, dim: int = _DIM) -> list[float]:
    """Unit vector at ``angle_deg`` on the 2D subspace starting at ``offset``, zero elsewhere.

    Two vectors on the SAME offset ``theta`` degrees apart have cosine ``cos(theta)`` exactly. Two
    vectors on DIFFERENT, non-overlapping offsets have cosine exactly 0 regardless of angle — used
    to keep two independent pairs from cross-contaminating each other's probe in the multi-batch
    isolation test.
    """
    a = math.radians(angle_deg)
    v = [0.0] * dim
    v[offset] = math.cos(a)
    v[offset + 1] = math.sin(a)
    return v


_COS_5 = math.cos(math.radians(5))


def _fact(
    *,
    content: str,
    scope: str = "proj-a",
    lifecycle: str = "active",
    validity: str = "valid",
    created_at: datetime = _NEW,
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
    )


def _decision(*, content: str, scope: str = "proj-a", created_at: datetime = _OLD) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope(scope),
        lifecycle="in_force",
        validity="valid",
        created_at=created_at,
        updated_at=created_at,
    )


class _TableEmbedder:
    """Deterministic embedder: exact-string -> vector lookup (KeyError on miss) — same device
    ``test_gate_probe.py`` / ``test_fact_supersession.py`` use."""

    dim: int = _DIM
    model: str = "test/table"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


def _v(
    index: int, verdict: str, *, why: str = "because", reason: str | None = None
) -> dict[str, Any]:
    """Build one judge answer payload item. ``reason`` defaults to ``"changed"`` for a
    ``"replace"`` verdict (the axis's pre-A3 hardcoded reason, kept as the default here so every
    existing unanimous-replace fixture stays in agreement without naming a reason explicitly) and
    is omitted for ``"coexist"`` (``FalsityVerdict`` forbids it there)."""
    payload: dict[str, Any] = {"index": index, "verdict": verdict, "why": why}
    if verdict == "replace":
        payload["reason"] = reason if reason is not None else "changed"
    return payload


def _contradiction(
    pick: Literal["coexist", "correction", "changed"], *, confidence: float = 1.0
) -> ContradictionVerdict:
    """Build a scripted ``contradiction`` answer for ``EchoAssessor``'s ``contradiction_script``.
    ``pick`` is the wire's own three-label choice (``coexist``/``correction``/``changed``) —
    collapsed to ``ContradictionVerdict.pick`` here exactly as ``JevAssessor`` collapses it."""
    reason_class: Literal["correction", "changed"] | None = (
        pick if pick in ("correction", "changed") else None
    )
    domain_pick: Literal["coexist", "replace"] = "coexist" if pick == "coexist" else "replace"
    remainder = (1 - confidence) / 2
    odds = dict.fromkeys(("coexist", "correction", "changed"), remainder)
    odds[pick] = confidence
    return ContradictionVerdict(
        pick=domain_pick,
        reason_class=reason_class,
        odds=odds,
        confidence=confidence,
        trace=Trace(
            model="echo", wording_hash="echo", latency_ms=0,
            input_tokens=None, output_tokens=None, request_id=None,
        ),
    )


def _unavailable_contradiction() -> BaseException:
    """``EchoAssessor``'s scripted stand-in for a Cleaner-timeout Assessor call — raised, never
    returned (``ask()`` is what turns a raise into a typed ``Unavailable(reason="timeout")``)."""
    return TimeoutError("assessor timed out")


def _payload(items: list[dict[str, Any]]) -> str:
    return json.dumps({"verdicts": items})


def _small_batch_rubric(batch_cap: int) -> Rubric:
    """The shipped rubric with ``batch_cap`` overridden — lets a test force multiple batches
    without needing dozens of fixture pairs (same device ``test_relevance_axis.py`` uses)."""
    base = load_rubric()
    return Rubric(
        version=base.version,
        parameters=RubricParameters(
            k=base.parameters.k,
            similarity_floor=base.parameters.similarity_floor,
            batch_cap=batch_cap,
            unclear_ceiling=base.parameters.unclear_ceiling,
            pool_limit=base.parameters.pool_limit,
            report_render_cap=base.parameters.report_render_cap,
            revisit_batch_cap=base.parameters.revisit_batch_cap,
            revisit_pass_cap=base.parameters.revisit_pass_cap,
            revisit_min_age_days=base.parameters.revisit_min_age_days,
            revisit_interval_days=base.parameters.revisit_interval_days,
            revisit_stable_multiplier=base.parameters.revisit_stable_multiplier,
            revisit_contested_multiplier=base.parameters.revisit_contested_multiplier,
            revisit_max_interval_days=base.parameters.revisit_max_interval_days,
        ),
        relevance=base.relevance,
        revisit=base.revisit,
        falsity=base.falsity,
        judges=base.judges,
        caseset=base.caseset,
    )


async def _spy_on_supersede(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Wrap the real ``supersede`` so a test can see exactly which ids were written AND
    assert the write lock is released at the moment of the call, while the real write still
    happens (unlike a pure mock, which would leave the store unchanged)."""
    calls: list[dict[str, Any]] = []

    async def spy(**kwargs: Any) -> Any:
        lock = kwargs["lock"]
        assert not lock.locked(), (
            "supersede invoked while the write lock is held -- this would deadlock in "
            "production, since supersede acquires the same lock itself"
        )
        calls.append(kwargs)
        return await _real_supersede(**kwargs)

    monkeypatch.setattr("tabularium.engine.falsity.supersede", spy)
    return calls


# ===========================================================================
# Newcomer pool selection
# ===========================================================================


class TestSelectFalsityNewcomers:
    def test_excludes_decisions(self) -> None:
        fact = _fact(content="an eligible fact")
        decision = _decision(content="an in-force decision", created_at=_NEW)

        pool = _select_falsity_newcomers([fact, decision], excluded_ids=frozenset())

        assert pool.excluded_decisions == (decision.id,)
        assert [f.id for f in pool.newcomers] == [fact.id]

    def test_excludes_global(self) -> None:
        fact = _fact(content="scoped to proj-a")
        global_fact = _fact(content="owner-written", scope="global")

        pool = _select_falsity_newcomers([fact, global_fact], excluded_ids=frozenset())

        assert pool.excluded_global == (global_fact.id,)
        assert [f.id for f in pool.newcomers] == [fact.id]

    def test_excludes_already_shelved_lifecycle(self) -> None:
        fact = _fact(content="still active")
        shelved_fact = _fact(content="already shelved", lifecycle="shelved")

        pool = _select_falsity_newcomers([fact, shelved_fact], excluded_ids=frozenset())

        assert pool.excluded_shelved_this_pass == (shelved_fact.id,)
        assert [f.id for f in pool.newcomers] == [fact.id]

    def test_excludes_hot_lifecycle(self) -> None:
        """Owner decision, 2026-08-12: a hot Fact must never itself become a newcomer N -- lands
        in the SAME ``excluded_shelved_this_pass`` bucket the "already shelved" exclusion already
        used, mirroring ``test_excludes_already_shelved_lifecycle`` exactly."""
        fact = _fact(content="still active")
        hot_fact = _fact(content="pinned hot", lifecycle="hot")

        pool = _select_falsity_newcomers([fact, hot_fact], excluded_ids=frozenset())

        assert pool.excluded_shelved_this_pass == (hot_fact.id,)
        assert [f.id for f in pool.newcomers] == [fact.id]

    def test_excludes_ember_lifecycle(self) -> None:
        """Same owner decision as ``test_excludes_hot_lifecycle``, the other write-time-provisional
        lifecycle."""
        fact = _fact(content="still active")
        ember_fact = _fact(content="write-time-seeded ember", lifecycle="ember")

        pool = _select_falsity_newcomers([fact, ember_fact], excluded_ids=frozenset())

        assert pool.excluded_shelved_this_pass == (ember_fact.id,)
        assert [f.id for f in pool.newcomers] == [fact.id]

    def test_archived_fact_is_eligible_as_newcomer(self) -> None:
        """The widened predicate is ``not in ("active", "archived")``, not ``!= "active"`` -- an
        ``archived`` Fact must stay eligible as a newcomer N, same as it already is on the
        relevance axis (mirrors ``test_relevance_axis.py``'s
        ``test_active_and_archived_facts_are_both_eligible``). This axis's own suite
        used to never pin it, so a future regression to an active-only
        predicate would pass every hot/ember test here while silently excluding archived
        newcomers."""
        archived_fact = _fact(content="cold but still authoritative", lifecycle="archived")

        pool = _select_falsity_newcomers([archived_fact], excluded_ids=frozenset())

        assert pool.excluded_shelved_this_pass == ()
        assert [f.id for f in pool.newcomers] == [archived_fact.id]

    def test_excludes_ids_shelved_this_pass(self) -> None:
        """The relevance axis's ``shelved_ids`` from the SAME pass, passed in as ``excluded_ids`` —
        distinct from a pre-existing ``lifecycle=="shelved"`` record."""
        fact = _fact(content="still active")
        shelved_this_pass = _fact(content="shelved by relevance axis just now", lifecycle="active")

        pool = _select_falsity_newcomers(
            [fact, shelved_this_pass], excluded_ids=frozenset({shelved_this_pass.id})
        )

        assert pool.excluded_shelved_this_pass == (shelved_this_pass.id,)
        assert [f.id for f in pool.newcomers] == [fact.id]

    def test_all_exclusions_apply_independently_in_one_pool(self) -> None:
        fact = _fact(content="eligible")
        decision = _decision(content="decision", created_at=_NEW)
        global_fact = _fact(content="global", scope="global")
        shelved_fact = _fact(content="shelved", lifecycle="shelved")

        pool = _select_falsity_newcomers(
            [fact, decision, global_fact, shelved_fact], excluded_ids=frozenset()
        )

        assert [f.id for f in pool.newcomers] == [fact.id]
        assert pool.excluded_decisions == (decision.id,)
        assert pool.excluded_global == (global_fact.id,)
        assert pool.excluded_shelved_this_pass == (shelved_fact.id,)


# ===========================================================================
# Candidate generation + pair formation
# ===========================================================================


class TestGeneratePairs:
    async def test_older_candidate_forms_viable_pair(self) -> None:
        adapter = _make_adapter()
        newcomer = _fact(content="newer record", created_at=_NEW)
        candidate = _fact(content="older record", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert len(result.viable_pairs) == 1
        n, o, sim = result.viable_pairs[0]
        assert n.id == newcomer.id
        assert o.id == candidate.id
        assert math.isclose(sim, _COS_5, abs_tol=1e-6)
        assert result.decision_candidates == ()
        assert result.global_candidates == ()

    async def test_archived_candidate_forms_viable_pair(self) -> None:
        """The widened predicate is ``not in ("active", "archived")``, not ``!= "active"`` -- an
        ``archived`` Fact must stay eligible as a supersession TARGET (O), same as it already is
        on the relevance axis. This axis's own suite used to never pin it, so a future regression
        to an active-only predicate would pass every
        hot/ember candidate test here while silently excluding archived candidates too."""
        adapter = _make_adapter()
        newcomer = _fact(content="newer record, archived-candidate probe", created_at=_NEW)
        candidate = _fact(
            content="cold but still authoritative", lifecycle="archived", created_at=_OLD
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert len(result.viable_pairs) == 1
        n, o, _sim = result.viable_pairs[0]
        assert n.id == newcomer.id
        assert o.id == candidate.id
        assert result.hot_or_ember_candidates == ()

    async def test_equal_age_candidate_is_discarded(self) -> None:
        adapter = _make_adapter()
        newcomer = _fact(content="newer record, equal-age probe", created_at=_NEW)
        candidate = _fact(content="same-age record", created_at=_NEW)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert result.viable_pairs == ()
        assert result.decision_candidates == ()
        assert result.global_candidates == ()

    async def test_newer_candidate_is_discarded(self) -> None:
        adapter = _make_adapter()
        newcomer = _fact(content="newer record, strictly-newer probe", created_at=_NEW)
        candidate = _fact(content="an even newer record", created_at=_NEWER_STILL)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert result.viable_pairs == ()

    async def test_below_similarity_floor_is_not_a_candidate(self) -> None:
        adapter = _make_adapter()
        newcomer = _fact(content="newer record, far probe", created_at=_NEW)
        candidate = _fact(content="an unrelated older record", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(90))  # cosine 0.0, well below the 0.80 floor
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert result.viable_pairs == ()

    async def test_newcomer_never_matches_itself(self) -> None:
        adapter = _make_adapter()
        newcomer = _fact(content="the only record in the store", created_at=_NEW)
        await adapter.upsert(newcomer, _angle_vec(0))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert result.viable_pairs == ()

    async def test_decision_candidate_routed_to_report(self) -> None:
        adapter = _make_adapter()
        newcomer = _fact(content="newer fact, decision-candidate probe", created_at=_NEW)
        candidate = _decision(content="an in-force decision", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert result.viable_pairs == ()
        assert len(result.decision_candidates) == 1
        entry = result.decision_candidates[0]
        assert entry.newcomer_id == newcomer.id
        assert entry.candidate_id == candidate.id
        assert entry.reason == "candidate_is_decision"
        assert math.isclose(entry.similarity, _COS_5, abs_tol=1e-6)

    async def test_global_candidate_routed_to_report(self) -> None:
        adapter = _make_adapter()
        newcomer = _fact(content="newer fact, global-candidate probe", created_at=_NEW)
        candidate = _fact(content="owner-curated global record", scope="global", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert result.viable_pairs == ()
        assert len(result.global_candidates) == 1
        entry = result.global_candidates[0]
        assert entry.newcomer_id == newcomer.id
        assert entry.candidate_id == candidate.id
        assert entry.reason == "candidate_is_global"

    async def test_hot_candidate_routed_to_report(self) -> None:
        """Owner decision, 2026-08-12: a hot Fact must never be a supersession TARGET either --
        ``probe_filters_for`` does not exclude ``hot``/``ember`` from a probe hit (only ``shelved``
        does), so without this classification the candidate below would become a viable pair."""
        adapter = _make_adapter()
        newcomer = _fact(content="newer fact, hot-candidate probe", created_at=_NEW)
        candidate = _fact(content="a pinned hot record", lifecycle="hot", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert result.viable_pairs == ()
        assert len(result.hot_or_ember_candidates) == 1
        entry = result.hot_or_ember_candidates[0]
        assert entry.newcomer_id == newcomer.id
        assert entry.candidate_id == candidate.id
        assert entry.reason == "candidate_is_hot_or_ember"

    async def test_ember_candidate_routed_to_report(self) -> None:
        """Same owner decision as ``test_hot_candidate_routed_to_report``, the other
        write-time-provisional lifecycle."""
        adapter = _make_adapter()
        newcomer = _fact(content="newer fact, ember-candidate probe", created_at=_NEW)
        candidate = _fact(
            content="a write-time-seeded ember record", lifecycle="ember", created_at=_OLD
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        result = await _generate_pairs(
            [newcomer],
            scope=Scope("proj-a"),
            adapter=adapter,
            embedder=embedder,
            cfg=EngineConfig(),
            rubric=load_rubric(),
        )

        assert result.viable_pairs == ()
        assert len(result.hot_or_ember_candidates) == 1
        entry = result.hot_or_ember_candidates[0]
        assert entry.newcomer_id == newcomer.id
        assert entry.candidate_id == candidate.id
        assert entry.reason == "candidate_is_hot_or_ember"


# ===========================================================================
# Band policy: unanimous / split-is-contested (no unclear on this axis) --,
# superseding earlier decisions's "the more aggressive judge governs the disagreement band".
# ===========================================================================


class TestBandPolicy:
    async def test_unanimous_replace_draws_edge(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="unanimous replace: newer", created_at=_NEW)
        candidate = _fact(content="unanimous replace: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.counts == FalsityBandCounts(replace=1, coexist=0, contested=0)
        assert len(summary.edges) == 1
        edge = summary.edges[0]
        assert edge.id == candidate.id
        assert edge.superseded_by == newcomer.id

    async def test_unanimous_coexist_draws_nothing(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="unanimous coexist: newer", created_at=_NEW)
        candidate = _fact(content="unanimous coexist: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "coexist")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.counts == FalsityBandCounts(replace=0, coexist=1, contested=0)
        assert summary.edges == ()
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid"

    async def test_split_a_replace_b_coexist_is_contested_and_draws_no_edge(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="split A-replace: newer", created_at=_NEW)
        candidate = _fact(content="split A-replace: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A replaces")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "coexist", why="B coexists")])
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.edges == ()
        assert summary.counts == FalsityBandCounts(replace=0, coexist=0, contested=1)
        assert len(summary.contested_pairs) == 1
        pair = summary.contested_pairs[0]
        assert math.isclose(pair.similarity, _COS_5, abs_tol=1e-6)
        assert pair == FalsityContestedPair(
            newcomer_id=newcomer.id,
            candidate_id=candidate.id,
            similarity=pair.similarity,
            judge_a_identity="a",
            judge_a_verdict="replace",
            judge_a_why="A replaces",
            judge_b_identity="b",
            judge_b_verdict="coexist",
            judge_b_why="B coexists",
        )
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid"
        assert stored.superseded_by is None

    async def test_split_a_coexist_b_replace_is_contested_and_draws_no_edge(self) -> None:
        """The reverse split — B replaces, A coexists — must ALSO be contested, never draw an
        edge."""
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="split B-replace: newer", created_at=_NEW)
        candidate = _fact(content="split B-replace: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "coexist", why="A coexists")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "replace", why="B replaces")])
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.edges == ()
        assert len(summary.contested_pairs) == 1
        pair = summary.contested_pairs[0]
        assert math.isclose(pair.similarity, _COS_5, abs_tol=1e-6)
        assert pair == FalsityContestedPair(
            newcomer_id=newcomer.id,
            candidate_id=candidate.id,
            similarity=pair.similarity,
            judge_a_identity="a",
            judge_a_verdict="coexist",
            judge_a_why="A coexists",
            judge_b_identity="b",
            judge_b_verdict="replace",
            judge_b_why="B replaces",
        )
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid"
        assert stored.superseded_by is None


class TestContestedReachesTheUnifiedQueue:
    """M2+ "one unified contested queue" slice (brief tripwire 1: falsity is a PAIR, not a
    record) + verify step "Relevance and falsity contests appear in the queue": a contested
    falsity pair must show a latest ``contested`` review for BOTH participants — newcomer AND
    candidate — never silently collapsed onto one id. Judge detail for each participant lives in the
    SEPARATE ``record_judgments`` table (owner correction, 2026-08-16), each side's row naming
    the OTHER via ``paired_with``.
    """

    async def test_both_pair_participants_have_a_contested_latest_review(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="unified-queue split: newer", created_at=_NEW)
        candidate = _fact(content="unified-queue split: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(
            identity="claude:opus", response=_payload([_v(1, "replace", why="looks stale")])
        )
        judge_b = FakeJudgeAdapter(
            identity="codex:gpt", response=_payload([_v(1, "coexist", why="still holds")])
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )
        assert len(summary.contested_pairs) == 1

        # Falsifiability: before this slice, the falsity axis wrote NOTHING to `record_reviews`
        # for a contested band -- this assertion would fail with an empty/partial list. BOTH
        # sides of the pair must appear, never just one.
        queued = contested_ids(adapter, Scope("proj-a"))
        assert queued == {newcomer.id, candidate.id}

        newcomer_judgment = latest_judgment(adapter, newcomer.id)
        candidate_judgment = latest_judgment(adapter, candidate.id)
        assert newcomer_judgment is not None
        assert candidate_judgment is not None

        for judgment in (newcomer_judgment, candidate_judgment):
            assert judgment.judge_a_identity == "claude:opus"
            assert judgment.judge_a_grade == "replace"
            assert judgment.judge_a_why == "looks stale"
            assert judgment.judge_b_identity == "codex:gpt"
            assert judgment.judge_b_why == "still holds"
            assert math.isclose(judgment.similarity or -1.0, _COS_5, abs_tol=1e-6)

        # Each side names the OTHER, never itself or a forced single "the" target (tripwire 1).
        assert newcomer_judgment.paired_with == candidate.id
        assert candidate_judgment.paired_with == newcomer.id

    async def test_dry_run_writes_neither_review_nor_judgment(self) -> None:
        """Mirrors the revisit/relevance axes' own dry-run contract: judging is identical, only
        the writes are skipped -- extended here to both NEW writes this slice adds."""
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="unified-queue dry run: newer", created_at=_NEW)
        candidate = _fact(content="unified-queue dry run: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace", why="x")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist", why="y")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
            dry_run=True,
        )
        assert len(summary.contested_pairs) == 1
        assert contested_ids(adapter, Scope("proj-a")) == set()
        assert latest_judgment(adapter, newcomer.id) is None
        assert latest_judgment(adapter, candidate.id) is None


class TestDecidingVerdictDefensiveBehavior:
    """``_deciding_verdict`` is only ever called by ``run_falsity_axis`` for a
    ``reason_disagreement``, where BOTH judges voted replace, so ``verdict_a.votes_replace`` is
    always ``True`` at the real call site and the ``verdict_b`` branch of its ternary is DEAD in
    production (see the function's own docstring). This is a direct unit-level pin of that branch,
    bypassing the axis entirely."""

    def test_returns_verdict_b_when_verdict_a_does_not_vote_replace(self) -> None:
        a = FalsityVerdict(index=1, verdict="coexist", why="A's reasoning")
        b = FalsityVerdict(index=1, verdict="replace", why="B's reasoning", reason="changed")

        assert _deciding_verdict(a, b) is b

    def test_returns_verdict_a_when_verdict_a_votes_replace(self) -> None:
        a = FalsityVerdict(index=1, verdict="replace", why="A's reasoning", reason="changed")
        b = FalsityVerdict(index=1, verdict="replace", why="B's reasoning", reason="changed")

        assert _deciding_verdict(a, b) is a


# ===========================================================================
# Edge direction — Verify step 1's deliberate-break target
# ===========================================================================


class TestEdgeDirection:
    async def test_write_uses_older_as_id_and_newer_as_superseded_by(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="direction probe: newer", created_at=_NEW)
        candidate = _fact(content="direction probe: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        # The summary itself pins the direction...
        assert len(summary.edges) == 1
        assert summary.edges[0].id == candidate.id  # the OLDER record is the predecessor
        assert summary.edges[0].superseded_by == newcomer.id  # the NEWER record is the successor

        # ...and so does the actual storage effect, independently of how the summary was built:
        # the older record must be the one left invalid and pointing forward.
        stored_candidate = await adapter.get_by_id(candidate.id)
        stored_newcomer = await adapter.get_by_id(newcomer.id)
        assert stored_candidate.validity == "invalid"
        assert stored_candidate.superseded_by == newcomer.id
        assert stored_newcomer.validity == "valid"
        assert stored_newcomer.superseded_by is None


# ===========================================================================
# Decision candidate — detected, reported, never written
# ===========================================================================


class TestDecisionCandidateNeverWritten:
    async def test_decision_candidate_produces_report_entry_and_no_write(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="decision-candidate integration: newer", created_at=_NEW)
        candidate = _decision(content="an in-force decision, similar enough", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        # Unanimous "replace" -- if the Decision somehow reached the panel and the write path,
        # this is the verdict that would trigger it. It must never get the chance.
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.edges == ()
        assert summary.counts == FalsityBandCounts(replace=0, coexist=0)
        assert len(summary.decision_candidates) == 1
        assert summary.decision_candidates[0].candidate_id == candidate.id
        assert summary.decision_candidates[0].reason == "candidate_is_decision"
        # The Decision never even reaches a batch -- zero viable pairs means zero judge calls.
        assert judge_a.calls == []
        assert judge_b.calls == []

        stored = await adapter.get_by_id(candidate.id)
        assert stored.lifecycle == "in_force"


# ===========================================================================
# Global candidate — detected, reported, never written. Verify step 2's deliberate-break target.
# ===========================================================================


class TestGlobalCandidateNeverWritten:
    async def test_global_candidate_produces_report_entry_and_no_write(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="global-candidate integration: newer", created_at=_NEW)
        candidate = _fact(
            content="owner-curated global record, similar enough", scope="global", created_at=_OLD
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        # Unanimous "replace" -- if `global` somehow reached the panel and the write path, this is
        # the verdict that would trigger it. It must never get the chance: `global` is owner-curated
        # and the Cleaner reads it, never writes to it.
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.edges == ()
        assert summary.counts == FalsityBandCounts(replace=0, coexist=0)
        assert len(summary.global_candidates) == 1
        assert summary.global_candidates[0].candidate_id == candidate.id
        assert summary.global_candidates[0].reason == "candidate_is_global"
        # `global` never even reaches a batch -- zero viable pairs means zero judge calls.
        assert judge_a.calls == []
        assert judge_b.calls == []

        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid"
        assert stored.superseded_by is None

    async def test_generate_pairs_classification_is_the_sole_guard_when_span_is_neutralized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Isolates ``_generate_pairs``'s own ``scope == "global"`` classification from
        ``supersede``'s cross-scope ``span`` guard (dispositions.py) — the SECOND,
        independent guard the real write call also passes (see ``TestSpanGuardOnWriteCall`` for
        that guard's own test). Neither guard is weakened in PRODUCTION here: this test patches
        ``supersede`` with a spy that strips the ``span`` kwarg before delegating to the
        real function, so the span guard cannot fire for the duration of this one test, and
        whatever happens next is attributable to ``_generate_pairs``'s classification alone.

        This is the Verify step 2 deliberate-break target: comment out
        ``_generate_pairs``'s ``if candidate.scope == _GLOBAL: ... continue`` branch and re-run this
        test. With the span guard neutralized here, nothing else stops the write — the store
        genuinely accepts it, and this test goes red on a plain ``AssertionError`` (a real edge
        shows up in ``summary.edges``), not on a ``ToolError`` from a different guard.
        """
        span_stripped_calls: list[dict[str, Any]] = []

        async def span_stripping_spy(**kwargs: Any) -> Any:
            span_stripped_calls.append(kwargs)
            kwargs = dict(kwargs)
            kwargs["span"] = None  # neutralize the OTHER guard so only classification is on trial
            return await _real_supersede(**kwargs)

        monkeypatch.setattr("tabularium.engine.falsity.supersede", span_stripping_spy)

        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="global-candidate isolation probe: newer", created_at=_NEW)
        candidate = _fact(
            content="owner-curated global record, isolation probe",
            scope="global",
            created_at=_OLD,
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        # `supersede` (even with its span guard neutralized) was never even called --
        # `_generate_pairs`'s classification kept the pair from ever reaching the write call.
        assert span_stripped_calls == []
        assert summary.edges == ()
        assert len(summary.global_candidates) == 1
        assert summary.global_candidates[0].candidate_id == candidate.id
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid"


# ===========================================================================
# Hot/ember candidate (O side) — detected, reported, never written. Owner decision, 2026-08-12: the
# Cleaner never touches a hot/ember Fact, including as a supersession target. Unlike the Decision/
# global cases above, `supersede` has NO lifecycle guard of its own on the predecessor -- so
# without `_generate_pairs`'s classification this would be a SILENT supersession, not a crash (see
# `tests/test_relevance_axis.py`'s sibling module for the crash half of the same owner decision).
# ===========================================================================


class TestHotEmberCandidateNeverWritten:
    async def test_hot_candidate_produces_report_entry_and_no_write(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="hot-candidate integration: newer", created_at=_NEW)
        candidate = _fact(
            content="a pinned hot record, similar enough", lifecycle="hot", created_at=_OLD
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        # Unanimous "replace" -- if the hot Fact somehow reached the panel and the write path,
        # this is the verdict that would trigger it. It must never get the chance.
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.edges == ()
        assert summary.counts == FalsityBandCounts(replace=0, coexist=0)
        assert len(summary.hot_or_ember_candidates) == 1
        assert summary.hot_or_ember_candidates[0].candidate_id == candidate.id
        assert summary.hot_or_ember_candidates[0].reason == "candidate_is_hot_or_ember"
        # The hot Fact never even reaches a batch -- zero viable pairs means zero judge calls.
        assert judge_a.calls == []
        assert judge_b.calls == []

        stored = await adapter.get_by_id(candidate.id)
        assert stored.lifecycle == "hot"
        assert stored.validity == "valid"
        assert stored.superseded_by is None

    async def test_ember_candidate_produces_report_entry_and_no_write(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="ember-candidate integration: newer", created_at=_NEW)
        candidate = _fact(
            content="a write-time-seeded ember record, similar enough",
            lifecycle="ember",
            created_at=_OLD,
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.edges == ()
        assert len(summary.hot_or_ember_candidates) == 1
        assert summary.hot_or_ember_candidates[0].candidate_id == candidate.id
        assert summary.hot_or_ember_candidates[0].reason == "candidate_is_hot_or_ember"
        assert judge_a.calls == []
        assert judge_b.calls == []

        stored = await adapter.get_by_id(candidate.id)
        assert stored.lifecycle == "ember"
        assert stored.superseded_by is None


# ===========================================================================
# The write call passes `span=[scope]` — the SECOND, independent guard against a cross-scope
# predecessor. Tested here in isolation from `_generate_pairs`'s own classification, on unmodified
# production code with no break applied, so its continued presence is verified rather than assumed.
# ===========================================================================


class TestSpanGuardOnWriteCall:
    async def test_write_call_passes_span_scoped_to_the_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = await _spy_on_supersede(monkeypatch)

        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="span-guard-on-write-call probe: newer", created_at=_NEW)
        candidate = _fact(content="span-guard-on-write-call probe: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert len(calls) == 1
        assert calls[0]["span"] == [Scope("proj-a")]


# ===========================================================================
# A newcomer shelved by the relevance axis in this same pass draws no edges
# ===========================================================================


class TestShelvedNewcomerExcluded:
    async def test_newcomer_shelved_this_pass_via_excluded_ids_draws_no_edges(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="shelved by relevance axis this pass", created_at=_NEW)
        candidate = _fact(content="an older record it would have replaced", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
            excluded_ids=frozenset({newcomer.id}),
        )

        assert summary.excluded_shelved_this_pass == (newcomer.id,)
        assert summary.edges == ()
        assert judge_a.calls == []
        assert judge_b.calls == []

    async def test_newcomer_already_shelved_lifecycle_draws_no_edges(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(
            content="already shelved before this pass", created_at=_NEW, lifecycle="shelved"
        )
        candidate = _fact(content="an older record it would have replaced", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.excluded_shelved_this_pass == (newcomer.id,)
        assert summary.edges == ()
        assert judge_a.calls == []
        assert judge_b.calls == []


# ===========================================================================
# Hot/ember newcomer (N side) — the OTHER half of the same owner decision, 2026-08-12: a hot/ember
# Fact must never itself become a newcomer either. `supersede` has no lifecycle guard on its
# `superseded_by` side, so without this exclusion it would silently draw the edge.
# ===========================================================================


class TestHotEmberNewcomerExcluded:
    async def test_hot_newcomer_draws_no_edges(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="a pinned hot newcomer", lifecycle="hot", created_at=_NEW)
        candidate = _fact(content="an older record it would have replaced", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.excluded_shelved_this_pass == (newcomer.id,)
        assert summary.edges == ()
        assert judge_a.calls == []
        assert judge_b.calls == []
        stored_candidate = await adapter.get_by_id(candidate.id)
        assert stored_candidate.validity == "valid"
        assert stored_candidate.superseded_by is None

    async def test_ember_newcomer_draws_no_edges(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(
            content="a write-time-seeded ember newcomer", lifecycle="ember", created_at=_NEW
        )
        candidate = _fact(content="an older record it would have replaced", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.excluded_shelved_this_pass == (newcomer.id,)
        assert summary.edges == ()
        assert judge_a.calls == []
        assert judge_b.calls == []
        stored_candidate = await adapter.get_by_id(candidate.id)
        assert stored_candidate.validity == "valid"
        assert stored_candidate.superseded_by is None


# ===========================================================================
# Judge-call failure: nothing written for that batch, failure surfaced
# ===========================================================================


class TestBatchFailure:
    async def test_invalid_output_writes_nothing_and_is_surfaced(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(
            content="would have replaced its candidate if the batch had succeeded",
            created_at=_NEW,
        )
        candidate = _fact(content="would have been replaced", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response="not json at all")

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.edges == ()
        assert summary.counts == FalsityBandCounts(replace=0, coexist=0)
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert failure.pairs == ((newcomer.id, candidate.id),)
        assert failure.judge_a_outcome == "ok"
        assert failure.judge_b_outcome == "invalid_output"
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid"

    async def test_unavailable_writes_nothing_and_is_surfaced(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(
            content="would have replaced its candidate, unavailable probe", created_at=_NEW
        )
        candidate = _fact(
            content="would have been replaced, unavailable probe", created_at=_OLD
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", should_fail=True, fail_reason="401 expired")
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.edges == ()
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert failure.judge_a_outcome == "unavailable"
        assert failure.judge_b_outcome == "ok"
        assert "401 expired" in failure.error
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid"

    async def test_failure_in_one_batch_does_not_affect_another(self) -> None:
        """``batch_cap=1`` forces two separate judge_batch calls (one per pair) — the first batch
        succeeds and draws its edge while the second fails and writes nothing, proving batches are
        independent. The two pairs live on orthogonal vector subspaces so neither probe can pick up
        the other pair's records as a candidate."""
        adapter = _make_adapter()
        clock = _make_clock()

        newcomer_1 = _fact(content="batch 1 -- succeeds: newer", created_at=_NEW)
        candidate_1 = _fact(content="batch 1 -- succeeds: older", created_at=_OLD)
        newcomer_2 = _fact(
            content="batch 2 -- fails: newer", created_at=_NEW.replace(microsecond=1)
        )
        candidate_2 = _fact(content="batch 2 -- fails: older", created_at=_OLD)

        await adapter.upsert(newcomer_1, _angle_vec(0, offset=0))
        await adapter.upsert(candidate_1, _angle_vec(5, offset=0))
        await adapter.upsert(newcomer_2, _angle_vec(0, offset=2))
        await adapter.upsert(candidate_2, _angle_vec(5, offset=2))

        embedder = _TableEmbedder(
            {
                newcomer_1.content: _angle_vec(0, offset=0),
                newcomer_2.content: _angle_vec(0, offset=2),
            }
        )

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))

        class _SequencedJudgeAdapter:
            def __init__(self, *, identity: str, responses: list[str]) -> None:
                self._identity = identity
                self._responses = responses
                self.calls: list[str] = []

            @property
            def identity(self) -> str:
                return self._identity

            def instruction_fingerprint(self) -> str:
                return "not_applicable"

            async def call(self, prompt: str) -> str:
                self.calls.append(prompt)
                return self._responses[len(self.calls) - 1]

        judge_b = _SequencedJudgeAdapter(
            identity="b", responses=[_payload([_v(1, "replace")]), "not json at all"]
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=_small_batch_rubric(batch_cap=1),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.counts == FalsityBandCounts(replace=1, coexist=0)
        assert len(summary.edges) == 1
        assert summary.edges[0].id == candidate_1.id
        assert summary.edges[0].superseded_by == newcomer_1.id

        assert len(summary.failed_batches) == 1
        assert summary.failed_batches[0].pairs == ((newcomer_2.id, candidate_2.id),)

        stored_candidate_1 = await adapter.get_by_id(candidate_1.id)
        stored_candidate_2 = await adapter.get_by_id(candidate_2.id)
        assert stored_candidate_1.validity == "invalid"
        assert stored_candidate_2.validity == "valid"


# ===========================================================================
# Per-call judge telemetry (cleaner-telemetry change) -- mirrors
# ``test_relevance_axis.py``'s own ``TestJudgeCallTelemetry`` exactly, for the falsity axis.
#
# Falsifiability (Verify step 1 of the brief): comment out one of the two
# `calls.append(outcome_X.as_call_record(...))` lines in `run_falsity_axis` (production,
# `src/tabularium/engine/falsity.py`) and re-run
# `TestJudgeCallTelemetry::test_mixed_outcome_batch_records_both_calls_not_just_the_survivor`. It
# goes red because `summary.calls` comes back with only one entry for the batch instead of two.
# ===========================================================================


class TestJudgeCallTelemetry:
    async def test_mixed_outcome_batch_records_both_calls_not_just_the_survivor(self) -> None:
        """One seat fails (invalid JSON), the other succeeds, in the SAME batch. Both calls must
        appear in ``summary.calls`` -- the failing one alongside the surviving one, neither
        dropped (acceptance criterion)."""
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="mixed-outcome batch: newer", created_at=_NEW)
        candidate = _fact(content="mixed-outcome batch: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="claude:sonnet", response=_payload([_v(1, "coexist")]))
        judge_b = FakeJudgeAdapter(identity="codex:gpt", response="not json at all")

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert len(summary.calls) == 2, (
            "one failed call must not suppress the other seat's call for the same batch"
        )
        by_judge = {c.judge: c for c in summary.calls}
        assert by_judge["claude:sonnet"].status == "success"
        assert by_judge["claude:sonnet"].error is None
        assert by_judge["codex:gpt"].status == "invalid_output"
        assert by_judge["codex:gpt"].error is not None
        for record in summary.calls:
            assert record.completion_tokens is None
            assert record.cost is None
            assert record.prompt_tokens > 0

    async def test_two_batches_two_seats_yields_four_call_entries(self) -> None:
        """``batch_cap=1`` forces two separate batches -- two seats x two batches = four call
        entries (acceptance criterion: "two seats x N batches"). Orthogonal vector subspaces keep
        the two pairs from cross-contaminating each other's probe."""
        adapter = _make_adapter()
        clock = _make_clock()

        newcomer_1 = _fact(content="batch 1: newer", created_at=_NEW)
        candidate_1 = _fact(content="batch 1: older", created_at=_OLD)
        newcomer_2 = _fact(content="batch 2: newer", created_at=_NEW.replace(microsecond=1))
        candidate_2 = _fact(content="batch 2: older", created_at=_OLD)
        await adapter.upsert(newcomer_1, _angle_vec(0, offset=0))
        await adapter.upsert(candidate_1, _angle_vec(5, offset=0))
        await adapter.upsert(newcomer_2, _angle_vec(0, offset=2))
        await adapter.upsert(candidate_2, _angle_vec(5, offset=2))
        embedder = _TableEmbedder(
            {
                newcomer_1.content: _angle_vec(0, offset=0),
                newcomer_2.content: _angle_vec(0, offset=2),
            }
        )

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "coexist")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=_small_batch_rubric(batch_cap=1),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert len(summary.calls) == 4
        assert sorted(c.batch_index for c in summary.calls) == [0, 0, 1, 1]
        assert all(c.status == "success" for c in summary.calls)


# ===========================================================================
# The panel: identical batch text
# ===========================================================================


class TestPanelMechanics:
    async def test_both_judges_receive_identical_batch_text(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="panel mechanics: newer", created_at=_NEW)
        candidate = _fact(content="panel mechanics: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "coexist")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist")]))

        await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert len(judge_a.calls) == 1
        assert len(judge_b.calls) == 1
        assert judge_a.calls[0] == judge_b.calls[0]
        assert newcomer.content in judge_a.calls[0]
        assert candidate.content in judge_a.calls[0]


# ===========================================================================
# The write lock is never held across `supersede` — asserted, not just avoided
# ===========================================================================


class TestLockNeverHeldAcrossSupersede:
    async def test_supersede_called_with_lock_released(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = await _spy_on_supersede(monkeypatch)

        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="lock invariant probe: newer", created_at=_NEW)
        candidate = _fact(content="lock invariant probe: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        # The assertion lives INSIDE the spy (see `_spy_on_supersede`) and fires at the
        # moment of every real call, not merely by the absence of an obvious `async with lock:`
        # wrapper in the source. This confirms the assertion path was actually exercised on a
        # genuine write, not vacuously true because no write happened.
        assert len(calls) == 1
        assert calls[0]["predecessor_ids"] == [candidate.id]
        assert calls[0]["successor_id"] == newcomer.id


class _AcquireCountingLock:
    """Wraps a real ``asyncio.Lock``, counting how many times ``acquire`` actually granted the
    lock (not how many times ``async with`` syntax appears in the source): proves the
    Assessor-decided split-supersede write holds ONE
    continuous lock across pending-append/write/stamp, not three separate acquisitions with a
    release in between where a concurrent ``unsupersede`` could land and be silently erased by
    the final stamp."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.acquire_count = 0

    async def acquire(self) -> bool:
        self.acquire_count += 1
        return await self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc_info: object) -> None:
        self.release()


class TestSplitSupersedeHoldsOneContinuousLock:
    async def test_pending_write_and_stamp_share_a_single_acquisition(self) -> None:
        adapter = _make_adapter()
        newcomer = _fact(content="one continuous lock: newer", created_at=_NEW)
        candidate = _fact(content="one continuous lock: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist", why="B")]))
        assessor = EchoAssessor(
            "unrelated", contradiction_script={candidate.content: _contradiction("correction")}
        )
        counting_lock = _AcquireCountingLock()

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
            lock=counting_lock,  # type: ignore[arg-type]
            rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert summary.split_supersede == 1
        # Exactly one acquisition covers the pending-row append, the supersede write itself, and
        # the final stamp for this single pair -- three separate acquisitions (the pre-fix shape)
        # would report 3 here instead.
        assert counting_lock.acquire_count == 1


# ===========================================================================
# Structural guards
# ===========================================================================


class TestStructuralGuards:
    def test_no_bury_call_anywhere_in_the_falsity_module(self) -> None:
        source = (_REPO_ROOT / "src" / "tabularium" / "engine" / "falsity.py").read_text()
        assert ".bury(" not in source
        assert "adapter.bury" not in source

    def test_module_does_not_import_recall(self) -> None:
        """``engine/recall.py`` must never be reached by this axis — it revives archived records
        on a fallback hit, which would turn a sweep's scan into a write."""
        import tabularium.engine.falsity as falsity_mod

        assert "recall" not in falsity_mod.__dict__
        assert not hasattr(falsity_mod, "find_superseded_by")

    def test_summary_is_returned_not_a_report_written(self) -> None:
        """This module returns ``FalsityAxisSummary``; it must never call anything that writes a
        report itself (not this axis's job)."""
        import inspect

        sig = inspect.signature(run_falsity_axis)
        assert sig.return_annotation in ("FalsityAxisSummary", FalsityAxisSummary)


# ===========================================================================
# First pass on an empty pool is a clean no-op
# ===========================================================================


class TestEmptyPool:
    async def test_empty_pool_is_a_clean_no_op(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        embedder = _TableEmbedder({})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        assert summary.newcomer_pool_size == 0
        assert summary.edges == ()
        assert summary.counts == FalsityBandCounts(replace=0, coexist=0)
        assert judge_a.calls == []
        assert judge_b.calls == []


# ===========================================================================
# A refused write skips the pair, never the pass (2026-08-12 crash-loop fix,
# mirroring test_relevance_axis.py's TestWriteRefusalSkipsRecordNotPass)
# ===========================================================================


class TestWriteRefusalSkipsPairNotPass:
    """Reproduces the SAME crash-loop shape ``engine/relevance.py``'s write call had, on this
    axis's own write call. Unlike the relevance axis's reproduction (which needs a genuine TOCTOU
    race), this one is reachable with no concurrent writer at all: two DIFFERENT newcomers in the
    SAME pass both probe back to the SAME older candidate and both get voted "replace" -- the first
    ``supersede`` call succeeds, the second raises ``ToolError`` (`is_fact_supersedable`'s
    "only a valid Fact can be superseded" -- checked before the "already superseded" check, so
    that is the wording actually raised) because the first call already flipped the candidate's
    ``validity``/``superseded_by``. Before this fix that
    second ``ToolError`` was uncaught and propagated out of ``run_falsity_axis`` entirely.
    """

    async def test_second_newcomer_targeting_the_same_candidate_is_refused_first_still_written(
        self,
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()

        candidate = _fact(content="the shared older candidate", created_at=_OLD)
        newcomer_1 = _fact(content="newcomer one, replaces the candidate", created_at=_NEW)
        newcomer_2 = _fact(
            content="newcomer two, also replaces the candidate",
            created_at=_NEW.replace(microsecond=1),
        )
        await adapter.upsert(candidate, _angle_vec(0))
        await adapter.upsert(newcomer_1, _angle_vec(2))
        await adapter.upsert(newcomer_2, _angle_vec(358))  # -2 degrees

        # Both newcomers sit closer to `candidate` (2 degrees away) than to EACH OTHER (4 degrees
        # apart) -- `candidate` wins the backward probe for both, deterministically, rather than
        # one newcomer matching the other.
        embedder = _TableEmbedder(
            {newcomer_1.content: _angle_vec(2), newcomer_2.content: _angle_vec(358)}
        )

        response = _payload([_v(1, "replace"), _v(2, "replace")])
        judge_a = FakeJudgeAdapter(identity="a", response=response)
        judge_b = FakeJudgeAdapter(identity="b", response=response)

        # No `pytest.raises` around this call -- the pass completing at all is the point; an
        # uncaught `ToolError` escaping here would fail this test on its own.
        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=embedder,
            cfg=EngineConfig(),
        )

        # 1. Exactly one edge drawn -- the other pair was refused, not applied.
        assert len(summary.edges) == 1
        assert summary.edges[0].id == candidate.id
        assert summary.counts.replace == 1

        # 2. The refused pair is surfaced -- both ids and a real refusal reason.
        assert len(summary.refused) == 1
        refusal = summary.refused[0]
        assert refusal.candidate_id == candidate.id
        # The first supersession flips the candidate to validity="invalid" in the SAME write as
        # `superseded_by` (`supersede`'s own docstring) -- `is_fact_supersedable` is checked
        # BEFORE the "already superseded" check, so that is the guard the second call actually
        # trips; either wording names a real, non-canned refusal reason.
        assert "not valid" in refusal.reason or "already superseded" in refusal.reason

        # 3. Together the written and refused newcomers account for both -- neither one lost.
        assert {summary.edges[0].superseded_by, refusal.newcomer_id} == {
            newcomer_1.id,
            newcomer_2.id,
        }

        stored_candidate = await adapter.get_by_id(candidate.id)
        assert stored_candidate.validity == "invalid"
        assert stored_candidate.superseded_by in (newcomer_1.id, newcomer_2.id)

    async def test_infrastructure_failure_during_the_write_still_aborts_the_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proves the abort path still fires for a genuine infrastructure failure that is
        NEITHER ``ValueError`` nor a documented CAS-race message (``supersede``
        now converts ANY exception from ``adapter.replace`` into ``ToolError`` chained via
        ``raise ... from exc`` -- never only ``ValueError`` -- because a real storage fault
        (disk, permissions, a Lance panic) is not guaranteed to surface as ``ValueError``).
        The wrapped ``ToolError`` still carries ``__cause__ is not None``, so this axis's own
        catch (``exc.__cause__ is not None`` -> re-raise) still aborts the whole pass rather
        than silently skipping the pair -- the boundary type changed from raw ``RuntimeError``
        to ``ToolError``, the abort-vs-skip PROPERTY did not."""
        adapter = _make_adapter()
        clock = _make_clock()
        candidate = _fact(content="an older candidate", created_at=_OLD)
        newcomer = _fact(content="a newer replacement", created_at=_NEW)
        await adapter.upsert(candidate, _angle_vec(0))
        await adapter.upsert(newcomer, _angle_vec(2))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(2)})

        async def _broken_transition(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("store unavailable")

        monkeypatch.setattr(adapter, "replace", _broken_transition)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        with pytest.raises(ToolError, match="store unavailable"):
            await run_falsity_axis(
                scope=Scope("proj-a"),
                since=_SINCE,
                adapter=adapter,
                clock=clock,
                lock=asyncio.Lock(),
                rubric=load_rubric(),
                judge_a=judge_a,
                judge_b=judge_b,
                embedder=embedder,
                cfg=EngineConfig(),
            )

    async def test_data_integrity_failure_wrapped_by_supersede_still_aborts_the_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``supersede`` (unlike
        ``set_shelved``) converts a ``KeyError``/``ValueError`` escaping ``adapter.transition``
        into ``ToolError`` too, chained via ``raise ... from exc`` (``engine/dispositions.py``). A
        bare ``except ToolError`` would have folded a ``pydantic.ValidationError`` -- a genuine
        data-integrity failure while re-reading the predecessor/successor -- into an ordinary
        refused pair and let the pass complete; this axis's own catch re-raises whenever
        ``exc.__cause__ is not None`` (any in-write failure, not only this one exception subtype
        -- a type-only check misses a plain ``ValueError`` from the codec, see
        the sibling test below), so it must still abort here, uncaught, exactly like the
        plain-``RuntimeError`` infrastructure-failure test above."""
        adapter = _make_adapter()
        clock = _make_clock()
        candidate = _fact(content="an older candidate", created_at=_OLD)
        newcomer = _fact(content="a newer replacement", created_at=_NEW)
        await adapter.upsert(candidate, _angle_vec(0))
        await adapter.upsert(newcomer, _angle_vec(2))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(2)})

        try:
            Fact.model_validate({})
        except ValidationError as caught:
            validation_error = caught
        else:  # pragma: no cover -- Fact has required fields; this should always raise
            pytest.fail("Fact.model_validate({}) unexpectedly did not raise ValidationError")

        async def _transition_raising_validation_error(*args: Any, **kwargs: Any) -> None:
            raise validation_error

        monkeypatch.setattr(adapter, "replace", _transition_raising_validation_error)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        with pytest.raises(ToolError, match="unexpected data-integrity failure"):
            await run_falsity_axis(
                scope=Scope("proj-a"),
                since=_SINCE,
                adapter=adapter,
                clock=clock,
                lock=asyncio.Lock(),
                rubric=load_rubric(),
                judge_a=judge_a,
                judge_b=judge_b,
                embedder=embedder,
                cfg=EngineConfig(),
            )

    async def test_plain_in_write_valueerror_from_supersede_also_aborts_not_just_validationerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An earlier fix distinguished by
        exception SUBTYPE (``isinstance(exc.__cause__, ValidationError)``), which missed that
        ``row_to_record`` (``storage/lancedb_codec.py``) raises a PLAIN ``ValueError`` -- not a
        ``ValidationError`` -- for a corrupt/NULL-required-timestamp row, indistinguishable from
        an ordinary lost CAS race by type alone. The fix now keys off WHETHER the exception came
        from inside the write attempt at all (``exc.__cause__ is not None``), not its subtype --
        this plain ``ValueError`` (chained the same way a real corrupt-row failure would be) must
        abort exactly like the ``ValidationError`` case above, proving the boundary is no longer
        type-specific."""
        adapter = _make_adapter()
        clock = _make_clock()
        candidate = _fact(content="an older candidate", created_at=_OLD)
        newcomer = _fact(content="a newer replacement", created_at=_NEW)
        await adapter.upsert(candidate, _angle_vec(0))
        await adapter.upsert(newcomer, _angle_vec(2))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(2)})

        async def _transition_raising_plain_valueerror(*args: Any, **kwargs: Any) -> None:
            raise ValueError("simulated corrupt row: Required timestamp field 'created_at' is NULL")

        monkeypatch.setattr(adapter, "replace", _transition_raising_plain_valueerror)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        with pytest.raises(ToolError, match="simulated corrupt row"):
            await run_falsity_axis(
                scope=Scope("proj-a"),
                since=_SINCE,
                adapter=adapter,
                clock=clock,
                lock=asyncio.Lock(),
                rubric=load_rubric(),
                judge_a=judge_a,
                judge_b=judge_b,
                embedder=embedder,
                cfg=EngineConfig(),
            )

    async def test_prewrite_refusal_still_causes_no_cause_and_is_skipped(self) -> None:
        """Boundary check on the OTHER side: a refusal `supersede` raises BEFORE ever
        attempting the storage write (``exc.__cause__ is None``) must still be skipped and
        reported, exactly as ``test_second_newcomer_targeting_the_same_candidate_...`` above
        already proves end-to-end -- this test pins the ``__cause__`` contract directly, at the
        unit level, so a future change to `supersede`'s pre-write raises cannot silently
        flip them to chained (`from exc`) without a targeted test catching it."""
        predecessor = _fact(content="already superseded", created_at=_OLD)
        successor = _fact(content="the replacement", created_at=_NEW)
        adapter = _make_adapter()
        await adapter.upsert(predecessor, _angle_vec(0))
        await adapter.upsert(successor, _angle_vec(2))
        await _real_supersede(
            successor_id=successor.id,
            predecessor_ids=[predecessor.id],
            reason="correction",
            note=None,
            adapter=adapter,
            clock=_make_clock(),
            lock=asyncio.Lock(),
        )

        another_successor = _fact(content="a second, later replacement", created_at=_NEWER_STILL)
        await adapter.upsert(another_successor, _angle_vec(4))

        with pytest.raises(ToolError) as excinfo:
            await _real_supersede(
                successor_id=another_successor.id,
                predecessor_ids=[predecessor.id],
                reason="correction",
                note=None,
                adapter=adapter,
                clock=_make_clock(),
                lock=asyncio.Lock(),
            )

        assert excinfo.value.__cause__ is None


# ===========================================================================
# A3 cleaner-integration piece — the Assessor joins the falsity axis (D5/D6/D7/D9/D11/D14/D15).
# ===========================================================================


class TestReasonFromJudges:
    async def test_full_agreement_supersedes_with_the_judges_own_reason_no_call(self) -> None:
        """D6: the write's ``reason`` comes from the judges, never the pre-A3 hardcoded
        ``"changed"`` — proven with an Assessor CONFIGURED but never called, since full agreement
        (same action, same reason) needs no tie-break."""
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="agreement correction: newer", created_at=_NEW)
        candidate = _fact(content="agreement correction: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "replace", reason="correction")])
        )
        assessor = EchoAssessor("unrelated")

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert summary.counts == FalsityBandCounts(replace=1, coexist=0, contested=0)
        assert summary.split_supersede == 0 and summary.reason_by_assessor == 0
        assert assessor.calls == []
        stored = await adapter.get_by_id(candidate.id)
        assert stored.supersession_reason == "correction"


class TestReasonDisagreementTieBreak:
    """D5: both judges vote ``replace`` but name different reasons — an Assessor
    ``contradiction`` call, carrying both judges' votes/reasoning, decides the reason outright."""

    async def _pair(self, adapter: MemoryAdapter) -> tuple[Fact, Fact, _TableEmbedder]:
        newcomer = _fact(content="reason split: newer", created_at=_NEW)
        candidate = _fact(content="reason split: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        return newcomer, candidate, _TableEmbedder({newcomer.content: _angle_vec(0)})

    async def test_assessor_picks_changed_writes_reason_by_assessor(self) -> None:
        adapter = _make_adapter()
        newcomer, candidate, embedder = await self._pair(adapter)
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "replace", why="B", reason="changed")])
        )
        assessor = EchoAssessor(
            "unrelated", contradiction_script={candidate.content: _contradiction("changed")}
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert len(summary.edges) == 1
        assert summary.reason_by_assessor == 1 and summary.split_supersede == 0
        stored = await adapter.get_by_id(candidate.id)
        assert stored.supersession_reason == "changed"

        existing, incoming, panel = assessor.contradiction_calls[0]
        assert existing == candidate.content
        assert incoming == newcomer.content
        assert panel is not None
        assert panel.judge_a.verdict == "replace" and panel.judge_a.reason == "correction"
        assert panel.judge_a.why == "A"
        assert panel.judge_b.verdict == "replace" and panel.judge_b.reason == "changed"
        assert panel.judge_b.why == "B"

        [row] = await adapter.verdicts_for_record(newcomer.id)
        assert row.caller == "cleaner_split"
        assert row.action == "reason_by_assessor"
        assert row.status == "acted"
        assert row.candidate_id == str(candidate.id)
        assert row.new_record_id == str(newcomer.id)
        assert row.effect_ref is not None

    async def test_assessor_picks_coexist_is_contested_no_write(self) -> None:
        """D11: the Assessor supplies no reason for a reason disagreement — a ``coexist`` pick
        against it is forbidden to guess a reason from, so the pair stays contested."""
        adapter = _make_adapter()
        newcomer, candidate, embedder = await self._pair(adapter)
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "replace", why="B", reason="changed")])
        )
        assessor = EchoAssessor(
            "unrelated", contradiction_script={candidate.content: _contradiction("coexist")}
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert summary.edges == ()
        assert summary.counts.contested == 1
        assert summary.split_coexist == 0 and summary.reason_by_assessor == 0
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid" and stored.superseded_by is None

        [row] = await adapter.verdicts_for_record(newcomer.id)
        assert row.action == "contested"
        assert row.actor == "panel"
        assert row.caller == "cleaner_split"

    async def test_assessor_unavailable_is_contested_no_write(self) -> None:
        adapter = _make_adapter()
        newcomer, candidate, embedder = await self._pair(adapter)
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "replace", why="B", reason="changed")])
        )
        assessor = EchoAssessor(
            "unrelated",
            contradiction_script={candidate.content: _unavailable_contradiction()},
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert summary.edges == ()
        assert summary.counts.contested == 1
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid"

        [row] = await adapter.verdicts_for_record(newcomer.id)
        assert row.action == "contested"
        assert row.unavailable_reason == "timeout"


class TestActionSplitTieBreak:
    """D9: one judge votes ``replace``, the other ``coexist`` — an Assessor ``contradiction``
    call decides the pair outright instead of routing it to the contested queue."""

    async def _pair(self, adapter: MemoryAdapter) -> tuple[Fact, Fact, _TableEmbedder]:
        newcomer = _fact(content="action split: newer", created_at=_NEW)
        candidate = _fact(content="action split: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        return newcomer, candidate, _TableEmbedder({newcomer.content: _angle_vec(0)})

    async def test_assessor_picks_correction_writes_split_supersede(self) -> None:
        adapter = _make_adapter()
        newcomer, candidate, embedder = await self._pair(adapter)
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist", why="B")]))
        assessor = EchoAssessor(
            "unrelated", contradiction_script={candidate.content: _contradiction("correction")}
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert len(summary.edges) == 1
        assert summary.split_supersede == 1 and summary.reason_by_assessor == 0
        stored = await adapter.get_by_id(candidate.id)
        assert stored.supersession_reason == "correction"

        existing, incoming, panel = assessor.contradiction_calls[0]
        assert existing == candidate.content
        assert incoming == newcomer.content
        assert panel is not None
        assert panel.judge_a.verdict == "replace"
        assert panel.judge_a.reason == "correction"
        assert panel.judge_a.why == "A"
        assert panel.judge_b.verdict == "coexist"
        assert panel.judge_b.reason is None
        assert panel.judge_b.why == "B"

        [row] = await adapter.verdicts_for_record(newcomer.id)
        assert row.action == "split_supersede"

    async def test_assessor_picks_coexist_writes_split_coexist_both_leave_contested(self) -> None:
        adapter = _make_adapter()
        newcomer, candidate, embedder = await self._pair(adapter)
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist", why="B")]))
        assessor = EchoAssessor(
            "unrelated", contradiction_script={candidate.content: _contradiction("coexist")}
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert summary.edges == ()
        assert summary.counts == FalsityBandCounts(replace=0, coexist=1, contested=0)
        assert summary.split_coexist == 1
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid" and stored.superseded_by is None

        assert contested_ids(adapter, Scope("proj-a")) == set()

        for target in (newcomer, candidate):
            history = await adapter.list_reviews_for(target_id=target.id)
            assert len(history) == 1
            assert history[0].band == "keep"
            assert history[0].outcome == "assessor_resolved"

        existing, incoming, panel = assessor.contradiction_calls[0]
        assert existing == candidate.content
        assert incoming == newcomer.content
        assert panel is not None
        assert panel.judge_a.verdict == "replace"
        assert panel.judge_a.reason == "correction"
        assert panel.judge_a.why == "A"
        assert panel.judge_b.verdict == "coexist"
        assert panel.judge_b.reason is None
        assert panel.judge_b.why == "B"

        [row] = await adapter.verdicts_for_record(newcomer.id)
        assert row.action == "split_coexist"
        assert row.status == "acted"

    async def test_participant_review_append_failure_falls_back_to_contested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolving keep review is the
        effect that clears the pair from the contested queue -- losing even one participant's
        must never be reported as a landed split_coexist."""
        adapter = _make_adapter()
        newcomer, candidate, embedder = await self._pair(adapter)
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist", why="B")]))
        assessor = EchoAssessor(
            "unrelated", contradiction_script={candidate.content: _contradiction("coexist")}
        )

        real_append_review = adapter.append_review
        call_count = {"n": 0}

        async def _broken_once_append_review(*args: Any, **kwargs: Any) -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("review store unavailable")
            await real_append_review(*args, **kwargs)

        monkeypatch.setattr(adapter, "append_review", _broken_once_append_review)

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert summary.edges == ()
        assert summary.split_coexist == 0
        assert summary.counts.contested == 1
        assert len(summary.contested_pairs) == 1
        pair = summary.contested_pairs[0]
        assert pair.newcomer_id == newcomer.id and pair.candidate_id == candidate.id
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid" and stored.superseded_by is None

        # `newcomer`'s own resolving-keep
        # review is the one that failed to append, but `candidate`'s OWN keep review landed
        # successfully before the failure -- both participants must still show contested, not
        # just the one whose write failed outright (the fallback's own timestamp bump is what
        # makes `candidate`'s stale "keep" review lose the tie against its own later "contested"
        # one).
        assert contested_ids(adapter, Scope("proj-a")) == {newcomer.id, candidate.id}

        rows = await adapter.verdicts_for_record(newcomer.id)
        [failed_row] = [r for r in rows if r.action == "split_coexist"]
        assert failed_row.status == "failed"
        [contested_row] = [r for r in rows if r.action == "contested"]
        assert contested_row.status == "acted"

    async def test_assessor_unavailable_writes_todays_contested_rows(self) -> None:
        adapter = _make_adapter()
        newcomer, candidate, embedder = await self._pair(adapter)
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist", why="B")]))
        assessor = EchoAssessor(
            "unrelated",
            contradiction_script={candidate.content: _unavailable_contradiction()},
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert summary.edges == ()
        assert len(summary.contested_pairs) == 1
        assert contested_ids(adapter, Scope("proj-a")) == {newcomer.id, candidate.id}
        [row] = await adapter.verdicts_for_record(newcomer.id)
        assert row.action == "contested" and row.unavailable_reason == "timeout"


class TestSplitSupersedeUnsupersede:
    async def test_unsupersede_on_a_split_supersede_edge_sets_the_override(self) -> None:
        """A2 machinery, unchanged — proven end to end on an edge this axis's OWN split path
        drew, not on a hand-built edge id."""
        adapter = _make_adapter()
        newcomer = _fact(content="split-supersede then undo: newer", created_at=_NEW)
        candidate = _fact(content="split-supersede then undo: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist", why="B")]))
        assessor = EchoAssessor(
            "unrelated", contradiction_script={candidate.content: _contradiction("correction")}
        )

        await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        [row_before] = await adapter.verdicts_for_record(newcomer.id)
        assert row_before.action == "split_supersede"
        assert row_before.override is None

        await unsupersede(
            id=candidate.id, reason="undo the assessor's split-supersede",
            adapter=adapter, clock=_make_clock(), lock=asyncio.Lock(),
        )

        [row_after] = await adapter.verdicts_for_record(newcomer.id)
        assert row_after.override == "unsupersede"
        assert row_after.override_at is not None


class TestGlobalPairAssessor:
    """A global pair the panel replaced but Jev ``breadth`` kept project-specific is reported:
    the ``contradiction`` call is made too — no ``panel``, logged, annotated, never written."""

    async def test_call_is_made_no_panel_row_logged_never_written(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="global assessor pick: newer", created_at=_NEW)
        candidate = _fact(
            content="owner-curated global record, assessor picks a reason",
            scope="global",
            created_at=_OLD,
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))
        assessor = EchoAssessor(
            "unrelated", contradiction_script={candidate.content: _contradiction("changed")}
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert len(judge_a.calls) == 1 and len(judge_b.calls) == 1
        assert summary.edges == ()
        assert len(summary.global_candidates) == 1
        entry = summary.global_candidates[0]
        assert entry.assessor_pick == "changed"

        existing, incoming, panel = assessor.contradiction_calls[0]
        assert existing == candidate.content and incoming == newcomer.content
        assert panel is None

        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid" and stored.superseded_by is None

        [row] = await adapter.verdicts_for_record(candidate.id)
        assert row.action == "global_reported"
        assert row.caller == "cleaner_split"
        assert row.status == "acted"

    async def test_unavailable_pick_is_reported_as_unavailable(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="global assessor unavailable: newer", created_at=_NEW)
        candidate = _fact(
            content="owner-curated global record, assessor unavailable",
            scope="global",
            created_at=_OLD,
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))
        assessor = EchoAssessor(
            "unrelated",
            contradiction_script={candidate.content: _unavailable_contradiction()},
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert summary.global_candidates[0].assessor_pick == "unavailable"
        [row] = await adapter.verdicts_for_record(candidate.id)
        assert row.unavailable_reason == "timeout"

    async def test_no_assessor_configured_leaves_pick_none_no_call(self) -> None:
        """D4/D7: ``assessor is None`` reproduces pre-A3 behaviour for the global report line
        exactly — no call, ``assessor_pick`` stays ``None``."""
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="global no assessor: newer", created_at=_NEW)
        candidate = _fact(
            content="owner-curated global record, no assessor", scope="global", created_at=_OLD
        )
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(),
        )

        assert summary.global_candidates[0].assessor_pick is None
        assert await adapter.verdicts_for_record(candidate.id) == []


class TestMalformedReasonValidation:
    def test_replace_with_no_reason_raises(self) -> None:
        with pytest.raises(ValidationError):
            FalsityVerdict(index=1, verdict="replace", why="x")

    def test_coexist_with_a_reason_raises(self) -> None:
        with pytest.raises(ValidationError):
            FalsityVerdict(index=1, verdict="coexist", why="x", reason="correction")

    async def test_judge_answers_replace_with_no_reason_is_invalid_output(self) -> None:
        """The wire-level equivalent: a judge's own JSON omits ``reason`` on a ``replace``
        verdict — same ``invalid_output`` batch failure a malformed verdict gets today."""
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="malformed reason: newer", created_at=_NEW)
        candidate = _fact(content="malformed reason: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([{"index": 1, "verdict": "replace", "why": "x"}])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(),
        )

        assert summary.edges == ()
        assert len(summary.failed_batches) == 1
        assert summary.failed_batches[0].judge_a_outcome == "invalid_output"


class TestWatermarkAdvancesWithAssessorUnavailable:
    async def test_pool_advances_past_since_when_assessor_is_unavailable_throughout(self) -> None:
        """A batch entirely resolved via an ``Unavailable`` Assessor is neither a failed batch
        nor an unclear/contested band this axis treats specially for watermark purposes — the
        pool-level watermark logic lives in ``clean.py``, this only pins that this axis itself
        reports no failed batches so that logic has nothing stopping it."""
        adapter = _make_adapter()
        clock = _make_clock()
        newcomer = _fact(content="watermark check: newer", created_at=_NEW)
        candidate = _fact(content="watermark check: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist", why="B")]))
        assessor = EchoAssessor(
            "unrelated",
            contradiction_script={candidate.content: _unavailable_contradiction()},
        )

        summary = await run_falsity_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
            embedder=embedder, cfg=EngineConfig(), assessor=assessor,
        )

        assert summary.failed_batches == ()
        assert summary.newcomer_pool_size == 1


class TestSplitSupersedeTornWriteReconciles:
    async def test_stamp_failure_after_supersede_lands_reconciles_to_acted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D14's crash order at the falsity split-supersede site: the pending row lands, the
        real ``supersede`` write lands, the stamp raises — a fresh reconcile pass must still
        find the drawn edge (by ``effect_ref``) and flip the row to ``acted``."""
        adapter = _make_adapter()
        newcomer = _fact(content="torn split-supersede: newer", created_at=_NEW)
        candidate = _fact(content="torn split-supersede: older", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))
        embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "replace", why="A", reason="correction")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "coexist", why="B")]))
        assessor = EchoAssessor(
            "unrelated", contradiction_script={candidate.content: _contradiction("correction")}
        )

        real_stamp_acted_many = falsity_mod.stamp_acted_many  # type: ignore[attr-defined]
        call_count = {"n": 0}

        async def _torn_stamp_acted_many(*args: Any, **kwargs: Any) -> None:
            call_count["n"] += 1
            raise RuntimeError("injected stamp failure")

        monkeypatch.setattr(falsity_mod, "stamp_acted_many", _torn_stamp_acted_many)

        with pytest.raises(RuntimeError, match="injected stamp failure"):
            await run_falsity_axis(
                scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=_make_clock(),
                lock=asyncio.Lock(), rubric=load_rubric(), judge_a=judge_a, judge_b=judge_b,
                embedder=embedder, cfg=EngineConfig(), assessor=assessor,
            )
        assert call_count["n"] == 1

        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "invalid"
        assert stored.superseded_by == newcomer.id

        pending = await adapter.pending_verdicts()
        assert len(pending) == 1
        assert pending[0].action == "split_supersede"
        edge_id = pending[0].effect_ref
        assert edge_id is not None

        monkeypatch.setattr(falsity_mod, "stamp_acted_many", real_stamp_acted_many)
        acted, failed = await reconcile_pending_verdicts(adapter, now=_make_clock().now())
        assert acted == 1 and failed == 0
        [stamped] = await adapter.verdicts_for_record(newcomer.id)
        assert stamped.status == "acted"
        assert stamped.effect_ref == edge_id


# ===========================================================================
# Merge on replace: a `carry` text is appended to N, saved as a record superseding N
# ===========================================================================

_CARRY = "Keep the cache lid on."
_NEWER_TEXT = "merge: deploys are blue-green now"
_OLDER_TEXT = "merge: deploys are in place, keep the cache lid on"
_MERGED_TEXT = _NEWER_TEXT + "\n\n" + _CARRY


def _vc(
    index: int,
    verdict: str,
    *,
    carry: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    entry = _v(index, verdict, reason=reason)
    if carry is not None:
        entry["carry"] = carry
    return entry


class _MergeFixture:
    def __init__(
        self,
        adapter: MemoryAdapter,
        newcomer: Fact,
        candidate: Fact,
        embedder: _TableEmbedder,
    ) -> None:
        self.adapter = adapter
        self.newcomer = newcomer
        self.candidate = candidate
        self.embedder = embedder

    async def run(
        self,
        judge_a: Any,
        judge_b: Any,
        *,
        assessor: Any = None,
        dry_run: bool = False,
        lock: asyncio.Lock | None = None,
    ) -> FalsityAxisSummary:
        return await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=self.adapter,
            clock=_make_clock(),
            lock=lock if lock is not None else asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            embedder=self.embedder,
            cfg=EngineConfig(),
            assessor=assessor,
            dry_run=dry_run,
        )


async def _merge_fixture(
    *, table_extra: dict[str, list[float]] | None = None, third: Fact | None = None
) -> _MergeFixture:
    adapter = _make_adapter()
    await adapter.create_scope(Scope("proj-a"))
    newcomer = _fact(content=_NEWER_TEXT, created_at=_NEW)
    candidate = _fact(content=_OLDER_TEXT, created_at=_OLD)
    await adapter.upsert(newcomer, _angle_vec(0))
    await adapter.upsert(candidate, _angle_vec(5))
    table = {newcomer.content: _angle_vec(0), _MERGED_TEXT: _angle_vec(0, offset=4)}
    if third is not None:
        await adapter.upsert(third, _angle_vec(0, offset=8))
    table.update(table_extra or {})
    return _MergeFixture(adapter, newcomer, candidate, _TableEmbedder(table))


def _replace_panel(
    *, carry_a: str | None = None, carry_b: str | None = None,
    reason_a: str = "changed", reason_b: str = "changed",
) -> tuple[Any, Any]:
    return (
        FakeJudgeAdapter(
            identity="a",
            response=_payload([_vc(1, "replace", carry=carry_a, reason=reason_a)]),
        ),
        FakeJudgeAdapter(
            identity="b",
            response=_payload([_vc(1, "replace", carry=carry_b, reason=reason_b)]),
        ),
    )


async def _only_active_ids(adapter: MemoryAdapter) -> set[Any]:
    active: frozenset[FactLifecycle] = frozenset({"active"})
    return {
        r.id
        async for r in await adapter.scan_records(lifecycle_in=active)
        if r.validity == "valid"
    }


class TestMergeOnReplace:
    async def test_unanimous_replace_with_seat_a_carry_merges(self) -> None:
        fx = await _merge_fixture()
        judge_a, judge_b = _replace_panel(carry_a=_CARRY)

        summary = await fx.run(judge_a, judge_b)

        assert [(e.id, e.superseded_by) for e in summary.edges] == [
            (fx.candidate.id, fx.newcomer.id)
        ]
        assert summary.counts.replace == 1
        assert len(summary.merged) == 1
        triple = summary.merged[0]
        assert (triple.older_id, triple.newer_id) == (fx.candidate.id, fx.newcomer.id)
        assert triple.merged_id is not None
        merged = await fx.adapter.get_by_id(triple.merged_id)
        assert isinstance(merged, Fact)
        assert merged.content == _MERGED_TEXT
        assert merged.source == f"cleaner:merge:{fx.candidate.id}"
        assert merged.lifecycle == "active"
        assert merged.validity == "valid"
        stored_newer = await fx.adapter.get_by_id(fx.newcomer.id)
        assert stored_newer.superseded_by == merged.id
        assert stored_newer.supersession_reason == "elaborated"
        stored_older = await fx.adapter.get_by_id(fx.candidate.id)
        assert stored_older.superseded_by == fx.newcomer.id
        assert summary.merge_reverted == ()

    async def test_only_seat_b_supplying_carry_is_enough_and_seat_a_wins_a_tie(self) -> None:
        b_merged = _NEWER_TEXT + "\n\nB text."
        fx = await _merge_fixture(table_extra={b_merged: _angle_vec(0, offset=4)})
        judge_a, judge_b = _replace_panel(carry_b="B text.")

        summary = await fx.run(judge_a, judge_b)

        assert summary.merged[0].merged_id is not None
        merged = await fx.adapter.get_by_id(summary.merged[0].merged_id)
        assert merged.content == _NEWER_TEXT + "\n\nB text."

        fx = await _merge_fixture()
        judge_a, judge_b = _replace_panel(carry_a=_CARRY, carry_b="B text.")

        summary = await fx.run(judge_a, judge_b)

        assert summary.merged[0].merged_id is not None
        merged = await fx.adapter.get_by_id(summary.merged[0].merged_id)
        assert merged.content == _MERGED_TEXT

    async def test_recall_walks_the_chain_from_the_older_record_to_the_tip(self) -> None:
        from tabularium.engine import Tabularium

        fx = await _merge_fixture(table_extra={_OLDER_TEXT: _angle_vec(5)})
        engine = Tabularium(fx.adapter, fx.embedder, _make_clock(), EngineConfig())
        await engine.initialize()
        judge_a, judge_b = _replace_panel(carry_a=_CARRY)

        summary = await fx.run(judge_a, judge_b)

        tip = summary.merged[0].merged_id
        assert tip is not None
        recalled = await engine.recall(_OLDER_TEXT, scope="proj-a", include_invalid=True)
        pointers = {hit.id: hit.superseded_by for hit in recalled.hits}
        assert pointers[fx.candidate.id] == fx.newcomer.id
        assert pointers[fx.newcomer.id] == tip
        assert pointers[tip] is None
        valid = await engine.recall(_OLDER_TEXT, scope="proj-a")
        assert [hit.id for hit in valid.hits] == [tip]

    async def test_without_carry_nothing_is_merged(self) -> None:
        fx = await _merge_fixture()
        judge_a, judge_b = _replace_panel()

        summary = await fx.run(judge_a, judge_b)

        assert len(summary.edges) == 1
        assert summary.merged == ()
        assert summary.merge_reverted == ()
        assert await _only_active_ids(fx.adapter) == {fx.newcomer.id}

    async def test_carry_on_a_coexist_vote_is_ignored(self) -> None:
        fx = await _merge_fixture()
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_vc(1, "coexist", carry=_CARRY)])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_vc(1, "coexist")]))

        summary = await fx.run(judge_a, judge_b)

        assert summary.edges == ()
        assert summary.merged == ()

    async def test_reason_disagreement_without_assessor_merges(self) -> None:
        fx = await _merge_fixture()
        judge_a, judge_b = _replace_panel(
            carry_a=_CARRY, reason_a="correction", reason_b="changed"
        )

        summary = await fx.run(judge_a, judge_b)

        assert len(summary.edges) == 1
        assert len(summary.merged) == 1
        assert summary.merged[0].merged_id is not None

    async def test_assessor_resolved_reason_disagreement_merges_after_the_lock_is_released(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tabularium.engine.facts import save_internal as real_save

        locked_at_call: list[bool] = []

        async def spy(**kwargs: Any) -> Any:
            locked_at_call.append(kwargs["lock"].locked())
            return await real_save(**kwargs)

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", spy)
        fx = await _merge_fixture()
        judge_a, judge_b = _replace_panel(
            carry_a=_CARRY, reason_a="correction", reason_b="changed"
        )
        assessor = EchoAssessor(
            "unrelated", contradiction_script={_OLDER_TEXT: _contradiction("changed")}
        )

        summary = await fx.run(judge_a, judge_b, assessor=assessor)

        assert summary.reason_by_assessor == 1
        assert len(summary.merged) == 1
        assert locked_at_call == [False]
        assert [call[0] for call in assessor.calls] == ["contradiction"]

    async def test_assessor_resolved_action_split_takes_the_replace_seats_carry(self) -> None:
        fx = await _merge_fixture()
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_vc(1, "coexist", carry="Ignored.")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_vc(1, "replace", carry=_CARRY, reason="changed")])
        )
        assessor = EchoAssessor(
            "unrelated", contradiction_script={_OLDER_TEXT: _contradiction("changed")}
        )

        summary = await fx.run(judge_a, judge_b, assessor=assessor)

        assert summary.split_supersede == 1
        assert len(summary.merged) == 1
        assert summary.merged[0].merged_id is not None
        merged = await fx.adapter.get_by_id(summary.merged[0].merged_id)
        assert merged.content == _MERGED_TEXT

    async def test_a_bounced_merge_undoes_the_edge_and_is_recorded(self) -> None:
        third = _fact(content="merge: a third record that reads the same", created_at=_OLD)
        fx = await _merge_fixture(
            table_extra={_MERGED_TEXT: _angle_vec(0, offset=8)}, third=third
        )
        judge_a, judge_b = _replace_panel(carry_a=_CARRY)

        summary = await fx.run(judge_a, judge_b)

        assert summary.merged == ()
        assert summary.edges == ()
        assert summary.counts.replace == 0
        assert summary.counts.replace == len(summary.edges)
        judged = 1
        assert (
            summary.counts.replace
            + len(summary.merge_reverted)
            + summary.counts.coexist
            + summary.counts.contested
            + len(summary.refused)
            + sum(len(failure.pairs) for failure in summary.failed_batches)
        ) == judged
        assert [(m.older_id, m.newer_id) for m in summary.merge_reverted] == [
            (fx.candidate.id, fx.newcomer.id)
        ]
        assert summary.merge_reverted[0].error.startswith("merged text bounced (")
        assert str(third.id) in summary.merge_reverted[0].error
        stored_older = await fx.adapter.get_by_id(fx.candidate.id)
        stored_newer = await fx.adapter.get_by_id(fx.newcomer.id)
        assert stored_older.validity == "valid"
        assert stored_older.superseded_by is None
        assert stored_newer.validity == "valid"
        assert stored_newer.superseded_by is None
        assert await _only_active_ids(fx.adapter) == {
            fx.candidate.id,
            fx.newcomer.id,
            third.id,
        }

    async def test_a_raising_merge_save_undoes_the_edge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        fx = await _merge_fixture()
        judge_a, judge_b = _replace_panel(carry_a=_CARRY)

        summary = await fx.run(judge_a, judge_b)

        assert summary.edges == ()
        assert summary.counts.replace == 0
        assert [m.error for m in summary.merge_reverted] == ["no disk"]
        stored_older = await fx.adapter.get_by_id(fx.candidate.id)
        assert stored_older.validity == "valid"
        assert stored_older.superseded_by is None

    async def test_a_failed_undo_keeps_the_standing_edge_counted_and_the_row_acted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        fx = await _merge_fixture()

        async def failing_restore(*_: Any, **__: Any) -> None:
            raise RuntimeError("restore failed")

        monkeypatch.setattr(fx.adapter, "restore", failing_restore)
        judge_a, judge_b = _replace_panel(
            carry_a=_CARRY, reason_a="correction", reason_b="changed"
        )
        assessor = EchoAssessor(
            "unrelated", contradiction_script={_OLDER_TEXT: _contradiction("changed")}
        )

        summary = await fx.run(judge_a, judge_b, assessor=assessor)

        assert [(e.id, e.superseded_by) for e in summary.edges] == [
            (fx.candidate.id, fx.newcomer.id)
        ]
        assert summary.counts.replace == len(summary.edges) == 1
        assert summary.reason_by_assessor == 1
        assert summary.merged == ()
        assert summary.merge_reverted == ()
        assert len(summary.merge_revert_failed) == 1
        assert "revert failed" in summary.merge_revert_failed[0].error
        judged = 1
        assert (
            summary.counts.replace
            + len(summary.merge_reverted)
            + summary.counts.coexist
            + summary.counts.contested
            + len(summary.refused)
            + sum(len(failure.pairs) for failure in summary.failed_batches)
        ) == judged
        stored_older = await fx.adapter.get_by_id(fx.candidate.id)
        assert stored_older.superseded_by == fx.newcomer.id
        rows = await fx.adapter.verdicts_for_record(fx.newcomer.id)
        assert [row.status for row in rows] == ["acted"]
        assert rows[0].override is None

    async def test_an_assessor_decided_merge_revert_leaves_every_count_at_the_standing_edges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        fx = await _merge_fixture()
        judge_a, judge_b = _replace_panel(
            carry_a=_CARRY, reason_a="correction", reason_b="changed"
        )
        assessor = EchoAssessor(
            "unrelated", contradiction_script={_OLDER_TEXT: _contradiction("changed")}
        )

        summary = await fx.run(judge_a, judge_b, assessor=assessor)

        assert summary.edges == ()
        assert summary.counts.replace == 0
        assert summary.reason_by_assessor == 0
        assert summary.split_supersede == 0
        assert len(summary.merge_reverted) == 1
        rows = await fx.adapter.verdicts_for_record(fx.newcomer.id)
        assert [row.status for row in rows] == ["failed"]
        assert rows[0].override is None

    async def test_an_already_present_merged_text_leaves_the_edge_standing(self) -> None:
        existing = _fact(content=_MERGED_TEXT, created_at=_OLD)
        fx = await _merge_fixture(third=existing)
        judge_a, judge_b = _replace_panel(carry_a=_CARRY)

        summary = await fx.run(judge_a, judge_b)

        assert len(summary.edges) == 1
        assert summary.merged == ()
        assert summary.merge_reverted == ()
        stored_older = await fx.adapter.get_by_id(fx.candidate.id)
        assert stored_older.superseded_by == fx.newcomer.id

    async def test_dry_run_counts_the_merge_and_writes_nothing(self) -> None:
        fx = await _merge_fixture()
        judge_a, judge_b = _replace_panel(carry_a=_CARRY)

        summary = await fx.run(judge_a, judge_b, dry_run=True)

        assert summary.counts.replace == 1
        assert len(summary.merged) == 1
        assert summary.merged[0].merged_id is None
        stored_older = await fx.adapter.get_by_id(fx.candidate.id)
        assert stored_older.validity == "valid"
        assert await _only_active_ids(fx.adapter) == {fx.candidate.id, fx.newcomer.id}


# ===========================================================================
# Nightly Cleaner under breadth: a project newcomer against a global record
# ===========================================================================


async def _global_pair_run(
    *, breadth_pick: str, scope: str = "proj-a"
) -> tuple[FalsityAxisSummary, MemoryAdapter, Fact, Fact, FakeJudgeAdapter]:
    adapter = _make_adapter()
    newcomer = _fact(content="breadth gate: newer", scope=scope, created_at=_NEW)
    candidate = _fact(content="breadth gate: global record", scope="global", created_at=_OLD)
    await adapter.upsert(newcomer, _angle_vec(0))
    await adapter.upsert(candidate, _angle_vec(5))
    embedder = _TableEmbedder({newcomer.content: _angle_vec(0)})
    judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")]))
    judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")]))

    summary = await run_falsity_axis(
        scope=Scope(scope),
        since=_SINCE,
        adapter=adapter,
        clock=_make_clock(),
        lock=asyncio.Lock(),
        rubric=load_rubric(),
        judge_a=judge_a,
        judge_b=judge_b,
        embedder=embedder,
        cfg=EngineConfig(),
        assessor=EchoAssessor("unrelated", breadth_pick=breadth_pick),
    )

    return summary, adapter, newcomer, candidate, judge_a


class TestGlobalCandidateUnderBreadth:
    async def test_project_specific_newcomer_is_judged_but_draws_no_edge(self) -> None:
        summary, adapter, _newcomer, candidate, judge_a = await _global_pair_run(
            breadth_pick="project_specific"
        )

        assert len(judge_a.calls) == 1
        assert summary.edges == ()
        assert summary.counts.replace == 0
        assert [entry.candidate_id for entry in summary.global_candidates] == [candidate.id]
        assert summary.global_candidates[0].reason == "candidate_is_global"
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "valid"
        assert stored.superseded_by is None

    async def test_owner_wide_newcomer_draws_the_edge_into_global(self) -> None:
        summary, adapter, newcomer, candidate, _judge_a = await _global_pair_run(
            breadth_pick="owner_wide"
        )

        assert [(edge.id, edge.superseded_by) for edge in summary.edges] == [
            (candidate.id, newcomer.id)
        ]
        assert summary.counts.replace == 1
        assert summary.global_candidates == ()
        assert summary.refused == ()
        stored = await adapter.get_by_id(candidate.id)
        assert stored.validity == "invalid"
        assert stored.superseded_by == newcomer.id
        assert stored.supersession_reason == "moved"
        assert stored.supersession_note is not None
        assert stored.supersession_note.startswith("falsity: ")
        assert "(panel), cross-scope into global" in stored.supersession_note

    async def test_unavailable_breadth_fails_the_pair_so_the_watermark_holds(self) -> None:
        adapter = _make_adapter()
        newcomer = _fact(content="breadth gate: newer", created_at=_NEW)
        candidate = _fact(content="breadth gate: global record", scope="global", created_at=_OLD)
        await adapter.upsert(newcomer, _angle_vec(0))
        await adapter.upsert(candidate, _angle_vec(5))

        class _BrokenBreadth(EchoAssessor):
            async def breadth(self, text: str, source_scope: str) -> Any:
                raise RuntimeError("jev down")

        summary = await run_falsity_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=_make_clock(),
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "replace")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "replace")])),
            embedder=_TableEmbedder({newcomer.content: _angle_vec(0)}),
            cfg=EngineConfig(),
            assessor=_BrokenBreadth("unrelated"),
        )

        assert summary.edges == ()
        assert summary.global_candidates == ()
        assert [f.pairs for f in summary.failed_batches] == [((newcomer.id, candidate.id),)]
        assert "breadth unavailable" in summary.failed_batches[0].error

    async def test_global_pass_judges_global_pairs_like_any_scope(self) -> None:
        summary, adapter, newcomer, candidate, _judge_a = await _global_pair_run(
            breadth_pick="project_specific", scope="global"
        )

        assert [(edge.id, edge.superseded_by) for edge in summary.edges] == [
            (candidate.id, newcomer.id)
        ]
        assert summary.excluded_global == ()
        stored = await adapter.get_by_id(candidate.id)
        assert stored.superseded_by == newcomer.id

    def test_global_newcomer_is_excluded_only_outside_a_global_pass(self) -> None:
        record = _fact(content="owner-written", scope="global")

        outside = _select_falsity_newcomers(
            [record], excluded_ids=frozenset(), pass_scope=Scope("proj-a")
        )
        inside = _select_falsity_newcomers(
            [record], excluded_ids=frozenset(), pass_scope=Scope("global")
        )

        assert outside.excluded_global == (record.id,)
        assert [fact.id for fact in inside.newcomers] == [record.id]
