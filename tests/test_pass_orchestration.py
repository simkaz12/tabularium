"""The Cleaner's pass orchestration.

Hermetic lane only — ``MemoryAdapter`` + ``FakeJudgeAdapter``/local test doubles, no network, no
credential, no real ``claude``/``codex`` process and no real litellm provider call anywhere in this
file (same convention as ``test_relevance_axis.py`` / ``test_falsity_axis.py`` /
``test_judge_seam.py``).

Falsifiability (Verify steps 1/2 of the task — both breaks target PRODUCTION code in
``src/tabularium/engine/clean.py``, never a test-local schema or helper):

1. Watermark-advanced-before-verification: reorder ``clean()`` so the ``set_cleaner_state`` call
   that advances the watermark runs BEFORE ``_verify_writes`` is even computed (instead of after,
   gated on it being empty). Re-run
   ``TestVerification::test_write_that_did_not_land_blocks_the_watermark`` — it goes red because
   the stored watermark ends up advanced even though the shelve write never actually landed (the
   test's monkeypatched ``set_shelved`` reports success without touching the adapter).
2. Missing-watermark-defaults-to-epoch: change ``clean()``'s ``if state is None:`` branch so instead
   of initializing to ``clock.now()`` and returning, it sets ``since = datetime.min`` (or any epoch
   sentinel) and falls through into the ordinary pool/axis/verify/advance path. Re-run
   ``TestInitializeFirstPass::test_no_state_initializes_watermark_and_judges_nothing`` — it goes red
   because the fixture store's pre-existing records (born long before the watermark-under-test) get
   swept into the pool and judged, rather than the first pass judging nothing.

Both commands and their captured red output are recorded verbatim in the task verdict, not
duplicated here (matching the sibling axis test files' own convention).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.engine._errors import ToolError
from tabularium.engine.clean import CleanSummary, clean
from tabularium.engine.core import Tabularium
from tabularium.engine.rubric import load_rubric
from tabularium.results import ReviseResult, SupersedeResult
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import CleanerState, ReinforcementEvent
from tests.support.doubles import FakeEmbedder, FakeJudgeAdapter, TestClock

_DIM = 16

_T0 = datetime(2025, 1, 1, tzinfo=UTC)  # the watermark before pass 1
_T1 = datetime(2025, 1, 2, tzinfo=UTC)  # a record born after T0
_T2 = datetime(2025, 1, 3, tzinfo=UTC)  # a record born after T1
_T3 = datetime(2025, 1, 4, tzinfo=UTC)  # a record born after T2 -- beyond a truncated pool_limit
_NOW = datetime(2025, 1, 10, tzinfo=UTC)  # "now" at the moment a pass runs


def _make_adapter(dim: int = _DIM) -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic", dim=dim)


def _make_clock(now: datetime = _NOW) -> TestClock:
    return TestClock(now)


async def _require_scope_always_ok(scope: Scope) -> None:
    """Stand-in for ``Tabularium._require_scope`` — every scope is already registered."""
    del scope


def _vec(dim: int = _DIM, *, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _angle_vec(angle_deg: float, *, offset: int = 0, dim: int = _DIM) -> list[float]:
    """Unit vector at ``angle_deg`` on the 2D subspace starting at ``offset``, zero elsewhere —
    same device ``test_falsity_axis.py`` uses. Two vectors on the SAME offset are related by
    ``cos(angle)``; two vectors on DIFFERENT, non-overlapping offsets are exactly orthogonal
    (cosine 0) regardless of angle, which is what keeps two independent fixtures from
    cross-contaminating each other's falsity probe."""
    a = math.radians(angle_deg)
    v = [0.0] * dim
    v[offset] = math.cos(a)
    v[offset + 1] = math.sin(a)
    return v


class _TableEmbedder:
    """Deterministic embedder: exact-string -> vector lookup (``KeyError`` on miss) — same device
    ``test_falsity_axis.py`` / ``test_gate_probe.py`` use, so the falsity axis's own
    ``embedder.embed(newcomer.content)`` call returns a vector this test controls exactly, rather
    than ``FakeEmbedder``'s hash-based one (which could coincidentally collide between two
    fixture strings at this dimensionality)."""

    dim: int = _DIM
    model: str = "test/table"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


def _fact(
    *,
    content: str,
    scope: str = "proj-a",
    lifecycle: str = "active",
    validity: str = "valid",
    created_at: datetime = _T1,
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


def _v(index: int, grade: str, *, unclear: bool = False, why: str = "because") -> dict[str, Any]:
    return {"index": index, "grade": grade, "unclear": unclear, "why": why}


def _falsity_v(
    index: int, verdict: str, *, why: str = "because", reason: str | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"index": index, "verdict": verdict, "why": why}
    if verdict == "replace":
        payload["reason"] = reason if reason is not None else "changed"
    return payload


def _payload(items: list[dict[str, Any]]) -> str:
    return json.dumps({"verdicts": items})


_ITEM_HEADING_RE = re.compile(r"^## (\d+)\.$", re.MULTILINE)


class _KeepAllJudgeAdapter:
    """Returns ``keep-clear`` for however many records are ACTUALLY in the batch -- counts the
    per-record ``## N.`` headings ``render_batch_prompt`` emits (matched precisely against
    ``^## \\d+\\.$``, never a bare ``"## "`` substring count -- the rubric prose itself is
    markdown with its own ``## `` section headers, e.g. "## The admission gate", which a loose
    substring count would also match), rather than assuming a fixed batch size. Deliberately
    robust to a pool size that might diverge from what a test expects (the ``pool_limit`` wiring
    break isolates the divergence itself, on ``pool_size`` assertions, rather than incidentally
    producing a batch-verdict-count mismatch as a confounding second failure)."""

    def __init__(self, identity: str) -> None:
        self._identity = identity
        self.calls: list[str] = []

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        self.calls.append(prompt)
        count = len(_ITEM_HEADING_RE.findall(prompt))
        return _payload([_v(i, "keep-clear") for i in range(1, count + 1)])


class _InjectingJudgeAdapter:
    """Returns a canned response, but FIRST writes a brand-new record directly into the adapter —
    simulates a record created WHILE the pass is running (mid judge-call), independent of whatever
    ``clean()``'s own pool snapshot already captured before this call was ever made."""

    def __init__(
        self,
        *,
        identity: str,
        response: str,
        adapter: MemoryAdapter,
        inject_fact: Fact,
        inject_vector: list[float],
    ) -> None:
        self._identity = identity
        self._response = response
        self._adapter = adapter
        self._inject_fact = inject_fact
        self._inject_vector = inject_vector
        self.calls: list[str] = []
        self.injected = False

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.injected:
            await self._adapter.upsert(self._inject_fact, self._inject_vector)
            self.injected = True
        return self._response


async def _spy_returning_phantom_shelve(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Monkeypatch ``engine.relevance.set_shelved`` to report success (a plausible
    ``ReviseResult`` with ``lifecycle='shelved'``) WITHOUT calling the real function — the write
    never actually lands on the adapter. Models a write that "returned, no exception raised" but
    did not land (15 verdict files were once lost this way)."""
    calls: list[Any] = []

    async def phantom(**kwargs: Any) -> ReviseResult:
        calls.append(kwargs)
        return ReviseResult(
            id=kwargs["id"], outcome="revised", lifecycle="shelved", validity="valid",
            reason=kwargs["reason"],
        )

    monkeypatch.setattr("tabularium.engine.relevance.set_shelved", phantom)
    return calls


async def _spy_returning_phantom_supersede(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Same device as ``_spy_returning_phantom_shelve``, for the falsity axis's write call."""
    calls: list[Any] = []

    async def phantom(**kwargs: Any) -> SupersedeResult:
        calls.append(kwargs)
        # Return a plausible post-write result WITHOUT ever calling the real
        # replace -- the adapter's stored records are left completely
        # untouched (falsity.py never reads this return value; only the
        # store's own post-write state matters to the verification pass).
        predecessor_id = kwargs["predecessor_ids"][0]
        return SupersedeResult(
            successor_id=kwargs["successor_id"],
            edges=((predecessor_id, "phantom-edge"),),
        )

    monkeypatch.setattr("tabularium.engine.falsity.supersede", phantom)
    return calls


# ===========================================================================
# First pass on an unknown scope: initialize and stop (acceptance criterion)
# ===========================================================================


class TestInitializeFirstPass:
    async def test_no_state_initializes_watermark_and_judges_nothing(self) -> None:
        """Verify step 2's break target: a store with pre-existing records long predating the
        pass must have NONE of them judged on a scope's first pass."""
        adapter = _make_adapter()
        clock = _make_clock()
        # A record that would be swept up if the missing-watermark default were the epoch.
        old_fact = _fact(content="ancient seed record", created_at=datetime(2020, 1, 1, tzinfo=UTC))
        await adapter.upsert(old_fact, _vec())

        assert await adapter.get_cleaner_state(scope=Scope("proj-a")) is None

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
        )

        assert summary.outcome == "initialized"
        assert summary.watermark_before is None
        assert summary.watermark_after == _NOW
        assert summary.pool_size == 0
        assert summary.relevance is None
        assert summary.falsity is None

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _NOW

        # The pre-existing record was never touched -- still active, never shelved.
        stored = await adapter.get_by_id(old_fact.id)
        assert stored.lifecycle == "active"

    async def test_no_state_never_calls_list_born_since(self) -> None:
        """A first pass must never even QUERY the pool -- structural, not just "judges nothing"."""

        adapter = _make_adapter()
        clock = _make_clock()
        calls: list[Any] = []
        original = adapter.list_born_since

        async def spy(**kwargs: Any) -> list[Any]:
            calls.append(kwargs)
            return await original(**kwargs)

        adapter.list_born_since = spy  # type: ignore[method-assign]

        await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
        )

        assert calls == []


# ===========================================================================
# Empty pool: clean no-op (acceptance criterion)
# ===========================================================================


class TestEmptyPool:
    async def test_empty_pool_is_a_no_op_that_completes_and_reports(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
        )

        assert summary.outcome == "empty_pool"
        assert summary.watermark_before == _T0
        assert summary.watermark_after == _T0
        assert summary.pool_size == 0
        assert summary.relevance is None
        assert summary.falsity is None

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T0


# ===========================================================================
# Ordering: relevance before falsity, shelved newcomers excluded from falsity
# (acceptance criterion)
# ===========================================================================


class TestOrdering:
    async def test_newcomer_shelved_by_relevance_draws_no_falsity_edge(self) -> None:
        """The scenario the module docstring describes: a newcomer close enough to an older
        record to be a falsity candidate, but the relevance axis shelves it first -- it must
        never reach the falsity sweep at all."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        older = _fact(
            content="older record, predates the watermark", created_at=_T0 - timedelta(days=1)
        )
        newcomer = _fact(
            content="newcomer, similar to older, born after watermark", created_at=_T1
        )
        await adapter.upsert(older, _vec(seed=1.0))
        await adapter.upsert(newcomer, _vec(seed=1.0))

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.outcome == "completed"
        assert summary.relevance is not None
        assert summary.falsity is not None
        assert summary.relevance.shelved_ids == (newcomer.id,)
        assert newcomer.id in summary.falsity.excluded_shelved_this_pass
        assert summary.falsity.edges == ()

        stored_newcomer = await adapter.get_by_id(newcomer.id)
        assert stored_newcomer.lifecycle == "shelved"
        stored_older = await adapter.get_by_id(older.id)
        assert isinstance(stored_older, Fact)
        assert stored_older.validity == "valid"
        assert stored_older.superseded_by is None

    async def _pass_with_newcomer(
        self, *, judge_a: FakeJudgeAdapter, judge_b: FakeJudgeAdapter
    ) -> tuple[Any, Fact]:
        adapter = _make_adapter()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        older = _fact(
            content="older record, predates the watermark", created_at=_T0 - timedelta(days=1)
        )
        newcomer = _fact(
            content="newcomer, similar to older, born after watermark", created_at=_T1
        )
        await adapter.upsert(older, _vec(seed=1.0))
        await adapter.upsert(newcomer, _vec(seed=1.0))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=_make_clock(),
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )
        return summary, newcomer

    async def test_newcomer_relevance_grades_unclear_draws_no_falsity_edge(self) -> None:
        payload = _payload([_v(1, "keep-lean", unclear=True)])
        summary, newcomer = await self._pass_with_newcomer(
            judge_a=FakeJudgeAdapter(identity="a", response=payload),
            judge_b=FakeJudgeAdapter(identity="b", response=payload),
        )

        assert summary.relevance is not None
        assert summary.relevance.unclear_ids == (newcomer.id,)
        assert summary.falsity is not None
        assert newcomer.id in summary.falsity.excluded_shelved_this_pass
        assert summary.falsity.edges == ()

    async def test_newcomer_whose_shelve_was_refused_draws_no_falsity_edge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def refuse(**_: Any) -> Any:
            raise ToolError("refused")

        monkeypatch.setattr("tabularium.engine.relevance.set_shelved", refuse)

        summary, newcomer = await self._pass_with_newcomer(
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")])),
        )

        assert summary.relevance is not None
        assert [r.id for r in summary.relevance.refused] == [newcomer.id]
        assert summary.falsity is not None
        assert newcomer.id in summary.falsity.excluded_shelved_this_pass
        assert summary.falsity.edges == ()

    async def test_full_pass_relevance_then_falsity_draws_edge_and_advances_watermark(self) -> None:
        """Both born-since records survive relevance (keep); falsity then draws a real edge from
        the newer to the older, and the watermark advances to the newest judged created_at."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        older = _fact(content="older placeholder measurement", created_at=_T1)
        newer = _fact(content="newer corrected measurement", created_at=_T2)
        await adapter.upsert(older, _angle_vec(0))
        await adapter.upsert(newer, _angle_vec(0))
        embedder = _TableEmbedder({older.content: _angle_vec(0), newer.content: _angle_vec(0)})

        rubric = load_rubric()

        class _RoutingJudge:
            """Routes a relevance-shaped payload to keep-both, a falsity-shaped payload to
            replace -- both axes share one judge instance per seat in this test."""

            def __init__(self, identity: str) -> None:
                self._identity = identity
                self.calls: list[str] = []

            @property
            def identity(self) -> str:
                return self._identity

            def instruction_fingerprint(self) -> str:
                return "not_applicable"

            async def call(self, prompt: str) -> str:
                self.calls.append(prompt)
                if "# Cleaner — falsity rubric" in prompt:
                    return _payload([_falsity_v(1, "replace", why="newer measurement supersedes")])
                return _payload(
                    [_v(1, "keep-clear", why="durable"), _v(2, "keep-clear", why="durable")]
                )

        judge_a = _RoutingJudge("a")
        judge_b = _RoutingJudge("b")

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=rubric,
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.outcome == "completed"
        assert summary.relevance is not None
        assert summary.falsity is not None
        assert summary.relevance.shelved_ids == ()
        assert len(summary.falsity.edges) == 1
        edge = summary.falsity.edges[0]
        assert edge.id == older.id
        assert edge.superseded_by == newer.id

        stored_older = await adapter.get_by_id(older.id)
        assert isinstance(stored_older, Fact)
        assert stored_older.validity == "invalid"
        assert stored_older.superseded_by == newer.id

        assert summary.watermark_before == _T0
        assert summary.watermark_after == _T2

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T2


# ===========================================================================
# Hot/ember Facts never crash the pass and never get written — owner decision, 2026-08-12
# ===========================================================================


class TestHotEmberNeverCrashesOrGetsWritten:
    """Reproduces the crash-loop this fix closes, end-to-end. Before it: a hot/ember Fact born in
    the window reached the relevance axis's judging panel (``_select_relevance_pool`` excluded
    only ``lifecycle == "shelved"``), was voted "shelve", and the resulting ``set_shelved`` call
    raised an uncaught ``ToolError`` (it accepts only an ``active``/``archived`` source lifecycle)
    that propagated out of ``clean()`` entirely. Because a failed pass never advances the
    watermark, the NEXT pass reselected the exact same record and died identically — forever. This
    test pins that the pass now completes cleanly instead of crashing."""

    async def test_hot_fact_in_the_pool_does_not_crash_the_pass(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        hot_fact = _fact(
            content="a pinned hot fact born in the window", lifecycle="hot", created_at=_T1
        )
        await adapter.upsert(hot_fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.outcome == "completed"
        assert summary.relevance is not None
        assert summary.relevance.shelved_ids == ()
        # The hot Fact never even reaches a batch -- zero candidates means zero judge calls.
        assert judge_a.calls == []
        assert judge_b.calls == []
        assert summary.watermark_after == _T1

        stored = await adapter.get_by_id(hot_fact.id)
        assert stored.lifecycle == "hot"

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T1

    async def test_ember_fact_in_the_pool_does_not_crash_the_pass(self) -> None:
        """Same reproduction, the other write-time-provisional lifecycle."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        ember_fact = _fact(
            content="a write-time-seeded ember born in the window",
            lifecycle="ember",
            created_at=_T1,
        )
        await adapter.upsert(ember_fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.outcome == "completed"
        assert summary.relevance is not None
        assert summary.relevance.shelved_ids == ()
        assert judge_a.calls == []
        assert judge_b.calls == []
        assert summary.watermark_after == _T1

        stored = await adapter.get_by_id(ember_fact.id)
        assert stored.lifecycle == "ember"

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T1


class _RacingJudgeAdapter:
    """Mirrors ``test_relevance_axis.py``'s own ``_RacingJudgeAdapter`` exactly -- as a side
    effect of its call, promotes ``race_id`` to ``"hot"`` directly on the adapter, reproducing a
    genuine TOCTOU race (another session's concurrent write landing between pool selection and
    this axis's own write loop) rather than monkeypatching the write path itself."""

    identity = "racer"

    def __init__(
        self, *, adapter: MemoryAdapter, clock: TestClock, race_id: HypomnemaId, response: str
    ) -> None:
        self._adapter = adapter
        self._clock = clock
        self._race_id = race_id
        self._response = response
        self.calls: list[str] = []

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        self.calls.append(prompt)
        await self._adapter.transition(self._race_id, "hot", at=self._clock.now())
        return self._response


class TestWriteRefusalDoesNotCrashTheLoop:
    """The GENERALISED sibling of ``TestHotEmberNeverCrashesOrGetsWritten`` above: that fix closed
    the ONE trigger already observed in production (a hot/ember Fact reaching the judge panel).
    This test proves the underlying fragility is closed too -- ANY future write-legality refusal,
    not just that one shape -- by reproducing a refusal ``_select_relevance_pool``'s own exclusion
    cannot prevent: a record promoted to ``hot`` by a genuine TOCTOU race AFTER pool selection
    already ran. Before this fix, this reproduction crashed identically to the hot/ember one:
    ``set_shelved``'s ``ToolError`` propagated out of ``clean()`` uncaught, the pass errored, and
    the watermark never advanced -- a crash loop. This test pins the full ``clean()`` contract:
    the pass completes, the OTHER record still shelves, the refused record is surfaced, and the
    watermark still advances end-to-end (all four -- see this task's own acceptance criteria)."""

    async def test_racing_record_refused_pass_completes_and_watermark_advances(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        raced_fact = _fact(content="promoted to hot mid-pass by another session", created_at=_T1)
        ordinary_fact = _fact(
            content="an unremarkable fact that should still shelve", created_at=_T1
        )
        await adapter.upsert(raced_fact, _vec())
        await adapter.upsert(ordinary_fact, _vec())

        response = _payload([_v(1, "shelve-clear"), _v(2, "shelve-clear")])
        judge_a = _RacingJudgeAdapter(
            adapter=adapter, clock=clock, race_id=raced_fact.id, response=response
        )
        judge_b = FakeJudgeAdapter(identity="b", response=response)

        # No `pytest.raises` around this call -- the pass completing at all, watermark included,
        # is exactly what this test exists to prove.
        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        # 1. The pass completes.
        assert summary.outcome == "completed"
        assert summary.relevance is not None
        # 2. The other record is still shelved.
        assert summary.relevance.shelved_ids == (ordinary_fact.id,)
        # 3. The refused record is surfaced in the report -- id and a real reason.
        assert len(summary.relevance.refused) == 1
        refusal = summary.relevance.refused[0]
        assert refusal.id == raced_fact.id
        assert "shelved" in refusal.reason
        # 4. The watermark advances -- the whole point of this fix.
        assert summary.watermark_after == _T1

        stored_raced = await adapter.get_by_id(raced_fact.id)
        stored_ordinary = await adapter.get_by_id(ordinary_fact.id)
        assert stored_raced.lifecycle == "hot"
        assert stored_ordinary.lifecycle == "shelved"

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T1


# ===========================================================================
# Pass-history append (cleaner-telemetry change): the completed branch
# appends exactly one ``CleanerPassRecord`` to the new table, alongside — never instead of — the
# existing ``set_cleaner_state`` write, which stays unchanged.
# ===========================================================================


class TestPassHistoryAppend:
    async def test_completed_pass_appends_one_row_and_state_write_is_unchanged(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        newcomer = _fact(content="a newcomer that survives both axes", created_at=_T1)
        await adapter.upsert(newcomer, _vec(seed=1.0))

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )
        assert summary.outcome == "completed"

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 1
        record = history[0]
        assert record.scope == Scope("proj-a")
        assert record.outcome == "completed"
        assert record.dry_run is False
        assert record.pass_at == clock.now()
        # payload is opaque JSON text at the storage seam -- just confirm it parses and carries
        # something about this pass, never assert a specific schema (that belongs to a later
        # task, per the task brief).
        payload = json.loads(record.payload)
        assert payload["watermark_after"] == summary.watermark_after.isoformat()

        # The existing set_cleaner_state write is untouched: same fields, same values, watermark
        # included -- this task adds a SECOND, separate write; it does not alter the first one.
        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == summary.watermark_after
        assert state.last_pass_at == clock.now()
        assert state.report is None

    async def test_two_completed_passes_leave_two_history_rows(self) -> None:
        """The property ``cleaner_state``'s upsert violates -- the entire reason the pass-history
        table exists (task brief)."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        first = _fact(content="first pass newcomer", created_at=_T1)
        await adapter.upsert(first, _angle_vec(0, offset=0))

        summary_1 = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )
        assert summary_1.outcome == "completed"

        # Orthogonal offset (see `_angle_vec`'s docstring): guarantees no accidental falsity pair
        # with `first`, independent of the property under test here.
        second = _fact(content="second pass newcomer", created_at=_T2)
        await adapter.upsert(second, _angle_vec(0, offset=2))

        summary_2 = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )
        assert summary_2.outcome == "completed"

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 2

    async def test_blocked_pass_appends_exactly_one_pass_history_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the telemetry change superseded the earlier "blocked appends
        nothing" scope -- every outcome now appends exactly one row, including "blocked" (the
        session-start alarm cares whether the pass RAN, not whether it shelved anything). A bare
        ``clean()`` call supplies no ``build_report`` callback, so the row's payload falls back to
        the same minimal shape every other bare-caller outcome uses."""
        calls = await _spy_returning_phantom_shelve(monkeypatch)
        del calls

        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        newcomer = _fact(
            content="a newcomer the relevance axis will (falsely) shelve", created_at=_T1
        )
        await adapter.upsert(newcomer, _vec(seed=1.0))

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )
        assert summary.outcome == "blocked"

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 1
        record = history[0]
        assert record.outcome == "blocked"
        assert record.dry_run is False
        assert record.pass_at == clock.now()
        payload = json.loads(record.payload)
        assert payload["watermark_after"] == summary.watermark_after.isoformat()
        assert payload["watermark_after"] == _T0.isoformat()  # never moved -- blocked


# ===========================================================================
# Every outcome appends exactly one row (cleaner-telemetry change) --
# "initialized" and "empty_pool" are the two outcomes left at zero rows; "completed" and
# "blocked" (above) already append live, this section adds the two missing branches plus the
# dry-run variant of all four, the cross-pass distinguishability property, and the ordering
# guarantee that a history-append failure never blocks or reverses the watermark write.
# ===========================================================================


class TestEveryOutcomeAppendsOneRow:
    async def test_initialized_pass_appends_one_row(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
        )
        assert summary.outcome == "initialized"

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 1
        record = history[0]
        assert record.outcome == "initialized"
        assert record.dry_run is False
        assert record.pass_at == clock.now()

    async def test_empty_pool_pass_appends_one_row(self) -> None:
        """Falsifiability target (Verify step 1 of the task): remove the
        ``await _append_pass_history(...)`` call from ``clean()``'s ``if not pool:`` branch and
        re-run this test -- it goes red on ``assert len(history) == 1`` (finds ``0``), not on a
        missing symbol or an import error."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
        )
        assert summary.outcome == "empty_pool"

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 1
        record = history[0]
        assert record.outcome == "empty_pool"
        assert record.dry_run is False
        assert record.pass_at == clock.now()

    async def test_the_four_outcomes_are_pairwise_distinguishable(self) -> None:
        """Acceptance criterion: "initialized, empty_pool and blocked rows are distinguishable
        from one another" -- widened here to all four ``clean()``-level outcomes, in one store, by
        the ``outcome`` column alone."""
        adapter = _make_adapter()
        clock = _make_clock()

        # Scope A: no state at all -> "initialized".
        await clean(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )

        # Scope B: watermark exists, nothing born since -> "empty_pool".
        await adapter.set_cleaner_state(
            scope=Scope("proj-b"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        await clean(
            scope="proj-b", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )

        # Scope C: a newcomer, panel unreachable -> "blocked".
        await adapter.set_cleaner_state(
            scope=Scope("proj-c"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        fact_c = _fact(content="a record whose panel cannot be reached", scope="proj-c")
        await adapter.upsert(fact_c, _vec())
        await clean(
            scope="proj-c", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", should_fail=True),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )

        # Scope D: a newcomer, panel keeps it -> "completed".
        await adapter.set_cleaner_state(
            scope=Scope("proj-d"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        fact_d = _fact(content="a record the panel keeps", scope="proj-d")
        await adapter.upsert(fact_d, _vec())
        await clean(
            scope="proj-d", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )

        outcomes = {}
        for scope_name in ("proj-a", "proj-b", "proj-c", "proj-d"):
            rows = await adapter.list_cleaner_passes(scope=Scope(scope_name), limit=10)
            assert len(rows) == 1
            outcomes[scope_name] = rows[0].outcome

        assert outcomes == {
            "proj-a": "initialized",
            "proj-b": "empty_pool",
            "proj-c": "blocked",
            "proj-d": "completed",
        }
        assert len(set(outcomes.values())) == 4  # pairwise distinguishable

    async def test_two_consecutive_passes_leave_two_rows_distinguishable_by_pass_at(self) -> None:
        """Acceptance criterion, the property the old ``cleaner_state`` upsert violates: two
        passes on ONE scope leave two rows with genuinely DIFFERENT ``pass_at`` values (the clock
        is advanced between passes so this cannot pass by coincidence)."""
        adapter = _make_adapter()
        clock = _make_clock()

        summary_1 = await clean(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        assert summary_1.outcome == "initialized"

        clock.advance(timedelta(days=1))

        summary_2 = await clean(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        assert summary_2.outcome == "empty_pool"

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 2
        pass_ats = {row.pass_at for row in history}
        assert len(pass_ats) == 2, "the two rows must carry two DIFFERENT pass_at values"
        outcomes_by_time = {row.pass_at: row.outcome for row in history}
        assert outcomes_by_time[_NOW] == "initialized"
        assert outcomes_by_time[_NOW + timedelta(days=1)] == "empty_pool"


# ===========================================================================
# Dry runs: exactly one write (the history row), flagged, watermark provably unchanged
# (cleaner-telemetry change)
# ===========================================================================


class TestDryRunHistoryRow:
    async def test_dry_run_initialized_appends_a_flagged_row_and_writes_nothing_else(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            dry_run=True,
        )
        assert summary.outcome == "initialized"
        assert summary.dry_run is True

        # The ONE thing a dry run writes: its own history row, flagged.
        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 1
        assert history[0].dry_run is True
        assert history[0].outcome == "initialized"

        # Nothing else: no CleanerState was ever created by this dry run.
        assert await adapter.get_cleaner_state(scope=Scope("proj-a")) is None

    async def test_dry_run_completed_pass_leaves_watermark_unchanged_and_flags_the_row(
        self,
    ) -> None:
        """Falsifiability target (Verify step 2 of the task): hard-code ``dry_run=False`` on the
        ``CleanerPassRecord`` construction inside ``_append_pass_history`` (or pass a literal
        ``False`` instead of the real ``dry_run`` argument at any of ``clean()``'s four call
        sites) and re-run this test -- it goes red on
        ``assert history[0].dry_run is True`` (finds ``False``), not on a missing symbol.

        Also the acceptance criterion's explicit ask: assert and state the watermark value before
        and after a dry run.
        """
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        newcomer = _fact(content="a newcomer a dry run would keep", created_at=_T1)
        await adapter.upsert(newcomer, _vec(seed=1.0))

        watermark_before_dry_run = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert watermark_before_dry_run is not None
        assert watermark_before_dry_run.watermark_at == _T0

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            dry_run=True,
        )
        assert summary.outcome == "completed"
        assert summary.dry_run is True
        # The summary's own "would advance to" value is real -- but the STORE never moved (below).
        assert summary.watermark_after == _T1

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 1
        assert history[0].dry_run is True
        assert history[0].outcome == "completed"

        watermark_after_dry_run = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert watermark_after_dry_run is not None
        assert watermark_after_dry_run.watermark_at == _T0, (
            "a dry run must never move the watermark -- provably unchanged: "
            f"before={watermark_before_dry_run.watermark_at.isoformat()}, "
            f"after={watermark_after_dry_run.watermark_at.isoformat()}"
        )
        assert watermark_before_dry_run.watermark_at == watermark_after_dry_run.watermark_at

        # And the record itself was never shelved/superseded/touched.
        stored = await adapter.get_by_id(newcomer.id)
        assert stored.lifecycle == "active"

    async def test_production_pass_records_dry_run_false(self) -> None:
        """Symmetric to the dry-run test above -- a LIVE pass's row carries ``dry_run=False``."""
        adapter = _make_adapter()
        clock = _make_clock()

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            dry_run=False,
        )
        assert summary.outcome == "initialized"
        assert summary.dry_run is False

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 1
        assert history[0].dry_run is False


# ===========================================================================
# A history-append failure never fails the pass and never blocks the watermark write
# (cleaner-telemetry change acceptance criterion)
# ===========================================================================


class TestHistoryAppendFailureNeverBlocksThePass:
    async def test_initialized_pass_still_writes_the_watermark_when_append_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()

        async def _exploding_append(*, record: Any) -> None:
            raise ConnectionError("pass_history table is unreachable")

        monkeypatch.setattr(adapter, "append_cleaner_pass", _exploding_append)

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
        )

        # The pass itself did not raise, and produced its normal result.
        assert summary.outcome == "initialized"
        assert summary.watermark_after == _NOW

        # The watermark write, which happens BEFORE the (failing) history append, still landed.
        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _NOW

    async def test_completed_pass_still_advances_the_watermark_when_append_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        newcomer = _fact(content="a newcomer kept despite a broken history table", created_at=_T1)
        await adapter.upsert(newcomer, _vec(seed=1.0))

        async def _exploding_append(*, record: Any) -> None:
            raise ConnectionError("pass_history table is unreachable")

        monkeypatch.setattr(adapter, "append_cleaner_pass", _exploding_append)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        # The pass did not raise despite the append blowing up.
        assert summary.outcome == "completed"
        assert summary.watermark_after == _T1

        # The (real) watermark write, which happens BEFORE the failing append, still landed.
        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T1

        # And the record was genuinely kept -- the axis's own write, unaffected by the
        # unrelated table's failure.
        stored = await adapter.get_by_id(newcomer.id)
        assert stored.lifecycle == "active"


# ===========================================================================
# Per-call telemetry reaches the pass-history payload (2026-08-11 cleaner-telemetry plan,
# once a real ``build_report`` callback is supplied (``run_clean_pass``, the production
# path via ``Tabularium.clean()``), ``payload`` carries the SAME enriched ``CleanerReport`` JSON
# already written into ``cleaner_state.report`` -- counts, ids, contradiction reports, batch
# failures, and now one call entry per judge call actually made.
# ===========================================================================


class TestPassHistoryPayloadCarriesCallTelemetry:
    async def test_completed_pass_payload_equals_state_report_and_carries_two_calls(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        newcomer = _fact(content="a newcomer both seats unanimously keep", created_at=_T1)
        await adapter.upsert(newcomer, _vec(seed=1.0))

        judge_a = FakeJudgeAdapter(
            identity="claude:sonnet", response=_payload([_v(1, "keep-clear")])
        )
        judge_b = FakeJudgeAdapter(
            identity="codex:gpt", response=_payload([_v(1, "keep-clear")])
        )

        from tabularium.engine.report import run_clean_pass

        summary = await run_clean_pass(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )
        assert summary.outcome == "completed"

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.report is not None

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 1
        record = history[0]

        # `payload` IS `state.report`, verbatim (cleaner-telemetry change) --
        # one write's worth of content, never a second, divergent schema.
        assert record.payload == state.report

        payload = json.loads(record.payload)
        # Acceptance criterion: "two seats x N batches" -- one relevance batch here.
        relevance_calls = payload["relevance"]["calls"]
        assert len(relevance_calls) == 2
        judges = {c["judge"] for c in relevance_calls}
        assert judges == {"claude:sonnet", "codex:gpt"}
        for call in relevance_calls:
            # Acceptance criterion: seat identity, rubric version, duration, status,
            # error-or-null, prompt tokens -- completion tokens and cost present and null.
            assert call["status"] == "success"
            assert call["error"] is None
            assert call["rubric_version"] == load_rubric().version
            assert call["duration_seconds"] >= 0.0
            assert call["prompt_tokens"] > 0
            assert "completion_tokens" in call and call["completion_tokens"] is None
            assert "cost" in call and call["cost"] is None

        # The report's existing fields survive unchanged in shape alongside the new telemetry.
        assert payload["relevance"]["counts"]["keep"] == 1
        assert payload["relevance"]["shelved_ids"] == []
        assert payload["falsity"]["calls"] == []


# ===========================================================================
# Verification before advance (acceptance criterion; Verify step 1's break target)
# ===========================================================================


class TestVerification:
    async def test_write_that_did_not_land_blocks_the_watermark(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        fact = _fact(content="a record the phantom write pretends to shelve", created_at=_T1)
        await adapter.upsert(fact, _vec())

        phantom_calls = await _spy_returning_phantom_shelve(monkeypatch)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert len(phantom_calls) == 1  # the phantom write really was invoked
        assert summary.outcome == "blocked"
        assert summary.verification_failures == (fact.id,)
        assert summary.watermark_before == _T0
        assert summary.watermark_after == _T0

        # The watermark in the STORE must not have moved either.
        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T0

        # And the record genuinely never transitioned -- the phantom write's whole point.
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"

    async def test_falsity_write_that_did_not_land_blocks_the_watermark(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        older = _fact(content="older placeholder", created_at=_T1)
        newer = _fact(content="newer replacement", created_at=_T2)
        await adapter.upsert(older, _angle_vec(0))
        await adapter.upsert(newer, _angle_vec(0))
        embedder = _TableEmbedder({older.content: _angle_vec(0), newer.content: _angle_vec(0)})

        phantom_calls = await _spy_returning_phantom_supersede(monkeypatch)

        rubric = load_rubric()

        class _RoutingJudge:
            def __init__(self, identity: str) -> None:
                self._identity = identity

            @property
            def identity(self) -> str:
                return self._identity

            def instruction_fingerprint(self) -> str:
                return "not_applicable"

            async def call(self, prompt: str) -> str:
                if "# Cleaner — falsity rubric" in prompt:
                    return _payload([_falsity_v(1, "replace")])
                return _payload(
                    [_v(1, "keep-clear"), _v(2, "keep-clear")]
                )

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=rubric,
            judge_a=_RoutingJudge("a"),
            judge_b=_RoutingJudge("b"),
        )

        assert len(phantom_calls) == 1
        assert summary.outcome == "blocked"
        assert summary.verification_failures == (older.id,)
        assert summary.watermark_after == _T0

        stored_older = await adapter.get_by_id(older.id)
        assert isinstance(stored_older, Fact)
        assert stored_older.validity == "valid"
        assert stored_older.superseded_by is None


# ===========================================================================
# Failed batch / unavailable judge (acceptance criteria)
# ===========================================================================


class TestFailedBatch:
    async def test_unavailable_judge_blocks_the_watermark(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        fact = _fact(content="a record whose panel cannot be reached", created_at=_T1)
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", should_fail=True)
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.outcome == "blocked"
        assert summary.relevance is not None
        assert len(summary.relevance.failed_batches) == 1
        assert summary.relevance.failed_batches[0].judge_a_outcome == "unavailable"
        assert summary.watermark_after == _T0

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T0

    async def test_invalid_judge_output_blocks_the_watermark(self) -> None:
        """A distinct failure shape from "unavailable" -- a response that arrives but fails the
        index protocol -- must ALSO block the watermark (acceptance: "A failed batch...")."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        fact = _fact(content="a record whose panel returns garbage", created_at=_T1)
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response="not json at all")
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.outcome == "blocked"
        assert summary.relevance is not None
        assert len(summary.relevance.failed_batches) == 1
        assert summary.relevance.failed_batches[0].judge_a_outcome == "invalid_output"
        assert summary.watermark_after == _T0


# ===========================================================================
# The fallback pass-history payload carries each failed batch's actual error text, not just a
# count (defect-1 fix, 2026-08-14 judge-call-error-visibility plan). EVERY dry run takes this path
# (`build_report` is never invoked under `dry_run`), and before this fix `fallback_payload`
# carried NOTHING about the relevance/falsity axes at all -- not even a count -- so a dry-run
# batch failure's reason was captured nowhere.
# ===========================================================================


class TestFallbackPayloadCarriesBatchFailureErrors:
    async def test_dry_run_blocked_pass_persists_the_relevance_batch_error_text(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        fact = _fact(content="a record whose panel cannot be reached", created_at=_T1)
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="a", should_fail=True, fail_reason="FakeJudgeAdapter: simulated codex outage"
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            dry_run=True,  # every dry run falls back to `fallback_payload` -- the exact hole
        )
        assert summary.outcome == "blocked"
        assert summary.dry_run is True

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        assert len(history) == 1
        payload = json.loads(history[0].payload)
        failed = payload["relevance_failed_batches"]
        assert len(failed) == 1
        assert failed[0]["batch_index"] == 0
        assert failed[0]["judge_a_outcome"] == "unavailable"
        assert "FakeJudgeAdapter: simulated codex outage" in failed[0]["error"]

    async def test_error_excerpt_is_capped_per_batch(self) -> None:
        """`FALLBACK_ERROR_EXCERPT_CAP` bounds any ONE failed batch's persisted error text.
        Falsifiability: set `FALLBACK_ERROR_EXCERPT_CAP` to a value larger than the fixture's
        error (e.g. 100_000) and re-run -- red, because the persisted excerpt then equals the
        full un-truncated error string instead of being cut to the cap."""
        from tabularium.engine import clean as clean_module

        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        fact = _fact(content="a record whose panel returns a huge error", created_at=_T1)
        await adapter.upsert(fact, _vec())

        huge_error = "X" * 10_000
        judge_a = FakeJudgeAdapter(identity="a", should_fail=True, fail_reason=huge_error)
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            dry_run=True,
        )
        assert summary.outcome == "blocked"

        history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
        payload = json.loads(history[0].payload)
        excerpt = payload["relevance_failed_batches"][0]["error"]
        assert len(excerpt) == clean_module.FALLBACK_ERROR_EXCERPT_CAP
        assert len(excerpt) < len(huge_error)

    def test_total_budget_is_shared_and_the_rest_collapse_to_an_omitted_marker(self) -> None:
        """`_FALLBACK_ERROR_TOTAL_CAP` bounds the SUM of every excerpt in one payload, not just
        one batch's -- a pathological pass with many failed batches must not grow the persisted
        payload unboundedly. Exercises `_capped_entries` directly (no judge panel/adapter
        needed -- the property under test is pure budget arithmetic).

        Falsifiability: change ``if remaining <= 0:`` to ``if False:`` in
        ``_capped_entries`` (never stop early) and re-run -- red, because all 20 failures
        then get a full entry, ``len(entries) == 20`` instead of 9, and no ``omitted_count``
        marker appears anywhere in the result.
        """
        from tabularium.engine.axis_common import BatchFailure
        from tabularium.engine.clean import (
            _FALLBACK_ERROR_TOTAL_CAP,
            FALLBACK_ERROR_EXCERPT_CAP,
            _capped_entries,
        )

        # Each failure's error is EXACTLY the per-error cap -- the budget is spent in whole,
        # predictable increments, so the omission point is exact rather than approximate.
        failures = tuple(
            BatchFailure(
                batch_index=i,
                record_ids=(),
                judge_a_outcome="unavailable",
                judge_b_outcome="ok",
                error="E" * FALLBACK_ERROR_EXCERPT_CAP,
            )
            for i in range(20)
        )

        capped, remaining = _capped_entries(failures, budget=_FALLBACK_ERROR_TOTAL_CAP)
        entries = [entry.model_dump(mode="json") for entry in capped]

        expected_full_entries = _FALLBACK_ERROR_TOTAL_CAP // FALLBACK_ERROR_EXCERPT_CAP
        assert len(entries) == expected_full_entries + 1  # + one trailing omitted-count marker
        assert entries[-1] == {"omitted_count": 20 - expected_full_entries}
        assert remaining == 0
        for entry in entries[:-1]:
            assert entry["batch_index"] in range(expected_full_entries)

    def test_total_budget_is_never_overshot_by_a_partially_fitting_excerpt(self) -> None:
        """The excerpt actually
        stored must be capped to ``min(FALLBACK_ERROR_EXCERPT_CAP, remaining)``, not
        ``FALLBACK_ERROR_EXCERPT_CAP`` alone -- capping only against the per-error limit let the
        last entry that still fit under a positive-but-small ``remaining`` store a FULL 500-char
        excerpt, overshooting the shared budget by up to 499 chars. ``budget=1`` here simulates
        exactly the shape ``fallback_payload`` threads through relevance -> falsity -> revisit:
        an earlier axis has already spent the shared budget down to a small, non-cap-divisible
        remainder before this axis's own call.

        Falsifiability: change ``excerpt_cap = min(FALLBACK_ERROR_EXCERPT_CAP, remaining)`` back
        to ``excerpt_cap = FALLBACK_ERROR_EXCERPT_CAP`` in ``_capped_entries`` and re-run
        -- red, because the stored excerpt then measures 500 chars (not 1) and ``remaining`` comes
        back ``-499`` instead of ``0``, proving the shared budget was overshot.
        """
        from tabularium.engine.axis_common import BatchFailure
        from tabularium.engine.clean import FALLBACK_ERROR_EXCERPT_CAP, _capped_entries

        failures = (
            BatchFailure(
                batch_index=0,
                record_ids=(),
                judge_a_outcome="unavailable",
                judge_b_outcome="ok",
                error="F" * FALLBACK_ERROR_EXCERPT_CAP,
            ),
        )

        capped, remaining = _capped_entries(failures, budget=1)
        entries = [entry.model_dump(mode="json") for entry in capped]

        assert len(entries) == 1
        stored_error = entries[0]["error"]
        assert isinstance(stored_error, str)
        assert len(stored_error) == 1  # capped to what remained, not the full 500
        assert remaining == 0


# ===========================================================================
# Watermark advances to newest FULLY-JUDGED created_at -- concurrent write
# (acceptance criterion; Verify step 3)
# ===========================================================================


class TestConcurrentWrite:
    async def test_record_created_mid_pass_is_judged_by_the_next_pass_not_skipped(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        original = _fact(content="the only record present when the pool was read", created_at=_T1)
        # Orthogonal offsets (see `_angle_vec`'s docstring): guarantees a falsity probe can never
        # accidentally pair `original` with `concurrent`, deterministically, regardless of the
        # concrete embedder-produced vectors -- the property under test (watermark advance) must
        # hold independent of whether a pair happens to form.
        original_vec = _angle_vec(0, offset=0)
        concurrent_vec = _angle_vec(0, offset=2)
        await adapter.upsert(original, original_vec)

        concurrent = _fact(
            content="written mid-pass, after the pool snapshot was already taken", created_at=_T2
        )
        embedder = _TableEmbedder(
            {original.content: original_vec, concurrent.content: concurrent_vec}
        )

        judge_a = _InjectingJudgeAdapter(
            identity="a",
            response=_payload([_v(1, "keep-clear")]),
            adapter=adapter,
            inject_fact=concurrent,
            inject_vector=concurrent_vec,
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary_1 = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        # Pass 1's own pool snapshot predates the injection -- the concurrently-written record is
        # neither counted nor advanced past.
        assert judge_a.injected is True
        assert summary_1.outcome == "completed"
        assert summary_1.pool_size == 1
        assert summary_1.watermark_after == _T1
        assert summary_1.watermark_after < concurrent.created_at

        # Not silently lost -- the NEXT pass picks it up.
        summary_2 = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )

        assert summary_2.outcome == "completed"
        assert summary_2.pool_size == 1
        assert summary_2.relevance is not None
        assert concurrent.id not in summary_2.relevance.excluded_already_shelved
        assert summary_2.watermark_after == _T2

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T2


# ===========================================================================
# pool_limit wiring: truncation advances only to the truncated boundary, and the remainder is
# fully judged by the next pass
# ===========================================================================


class TestPoolLimitTruncationRecovery:
    async def test_truncated_pool_advances_only_to_the_boundary_and_the_remainder_is_judged_next(
        self,
    ) -> None:
        """Three records born since the watermark, ``pool_limit=2`` on pass 1 -- the outer
        snapshot and BOTH axes must all agree on the same truncated pool (previously
        three independently-defaulted ``_DEFAULT_POOL_LIMIT`` constants could diverge). Pass 1
        must advance the watermark only to the truncated boundary's ``created_at`` (record 2's),
        never past record 3, which was never in the pool at all. Pass 2 (no override -- the
        ordinary rubric-sourced default) must then pick up record 3 and fully judge it."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )

        rec1 = _fact(content="record one, oldest of the three", created_at=_T1)
        rec2 = _fact(content="record two, middle -- the truncation boundary", created_at=_T2)
        rec3 = _fact(
            content="record three, newest -- beyond pool_limit=2 on pass 1", created_at=_T3
        )
        # Orthogonal offsets (see `_angle_vec`'s docstring) -- guarantees the falsity axis can
        # never accidentally pair any two of these three, deterministically, regardless of pass.
        vec1, vec2, vec3 = _angle_vec(0, offset=0), _angle_vec(0, offset=2), _angle_vec(0, offset=4)
        await adapter.upsert(rec1, vec1)
        await adapter.upsert(rec2, vec2)
        await adapter.upsert(rec3, vec3)
        embedder = _TableEmbedder(
            {rec1.content: vec1, rec2.content: vec2, rec3.content: vec3}
        )

        # Pass 1: pool_limit=2 truncates the 3-record pool to the 2 OLDEST (rec1, rec2) --
        # `list_born_since` is oldest-first, capped at limit (MemoryAdapter's own contract).
        summary_1 = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=_KeepAllJudgeAdapter("a"),
            judge_b=_KeepAllJudgeAdapter("b"),
            pool_limit=2,
        )

        assert summary_1.outcome == "completed"
        assert summary_1.pool_size == 2
        # The outer snapshot and each axis's own internal re-query agree exactly --
        # neither axis silently judged a different-sized pool than what was snapshotted.
        assert summary_1.relevance is not None
        assert summary_1.relevance.pool_size == 2
        assert summary_1.falsity is not None
        assert summary_1.falsity.newcomer_pool_size == 2
        # The watermark advances only to the TRUNCATED boundary (rec2's created_at) -- never past
        # rec3, which was never part of the pool this pass judged at all.
        assert summary_1.watermark_before == _T0
        assert summary_1.watermark_after == _T2
        assert summary_1.watermark_after < rec3.created_at

        # rec3 was never touched this pass.
        stored_rec3 = await adapter.get_by_id(rec3.id)
        assert stored_rec3.lifecycle == "active"

        # Pass 2: no override -- the ordinary rubric-sourced default. The remainder (rec3) is
        # picked up and FULLY judged (shelved here, proving genuine judging happened, not a
        # silent skip) -- not lost by the truncation on pass 1.
        summary_2 = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
        )

        assert summary_2.outcome == "completed"
        assert summary_2.pool_size == 1
        assert summary_2.relevance is not None
        assert summary_2.relevance.shelved_ids == (rec3.id,)
        assert summary_2.watermark_after == _T3

        stored_rec3_after = await adapter.get_by_id(rec3.id)
        assert stored_rec3_after.lifecycle == "shelved"

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _T3


# ===========================================================================
# scope="global" is refused (acceptance criterion)
# ===========================================================================


class TestGlobalScopeRuns:
    async def test_global_scope_initializes_like_any_scope(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()

        summary = await clean(
            scope="global",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
        )

        assert summary.outcome == "initialized"
        assert await adapter.get_cleaner_state(scope=Scope("global")) is not None


# ===========================================================================
# core.py delegation (acceptance criterion: "clean(scope) exists as an engine arm; core.py
# gains only a delegation line")
# ===========================================================================


class TestCorePyDelegation:
    async def test_tabularium_clean_delegates_to_the_engine_arm(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        engine = Tabularium(adapter, FakeEmbedder(dim=_DIM), clock, EngineConfig())
        await engine.initialize()
        await adapter.create_scope(Scope("proj-a"))

        summary = await engine.clean("proj-a")

        assert isinstance(summary, CleanSummary)
        assert summary.outcome == "initialized"
        assert summary.scope == Scope("proj-a")

    async def test_tabularium_clean_runs_global(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        engine = Tabularium(adapter, FakeEmbedder(dim=_DIM), clock, EngineConfig())
        await engine.initialize()

        summary = await engine.clean("global")

        assert summary.scope == Scope("global")
        assert summary.outcome == "initialized"

    async def test_tabularium_clean_rejects_unregistered_scope(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        engine = Tabularium(adapter, FakeEmbedder(dim=_DIM), clock, EngineConfig())
        await engine.initialize()

        with pytest.raises(ToolError, match="Unregistered scope"):
            await engine.clean("never-registered")


# ===========================================================================
# Revisit axis integration (2026-08-12 cleaner-revisit-pass plan, scope 5) — runs after falsity,
# inside this SAME clean() call, isolated from the birth axes' own outcome/watermark.
# ===========================================================================

_REVISIT_BORN = _NOW - timedelta(days=100)  # well past revisit_min_age_days (30)


async def _give_revisit_evidence(adapter: MemoryAdapter, fact: Fact) -> None:
    await adapter.append_event(
        ReinforcementEvent(
            target_id=fact.id,
            at=_REVISIT_BORN + timedelta(days=1),
            source="dedup",
            matched_from_id=new_id(),
        )
    )


class TestRevisitAxisIntegration:
    async def test_revisit_axis_runs_on_a_scopes_very_first_pass(self) -> None:
        """The birth axes' own initialize-and-stop never sweeps the store on
        a scope's first pass -- but the revisit axis has its own independent cold-start drain
        (decision 4: "the cap only binds during the initial drain") and no watermark to
        initialize, so it must run here too."""
        adapter = _make_adapter()
        clock = _make_clock()
        old_record = _fact(
            content="An old durable gotcha eligible for the revisit rotation.",
            created_at=_REVISIT_BORN,
        )
        await adapter.upsert(old_record, _vec(seed=0.5))
        await _give_revisit_evidence(adapter, old_record)

        assert await adapter.get_cleaner_state(scope=Scope("proj-a")) is None

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.outcome == "initialized"
        assert summary.revisit is not None
        assert summary.revisit.shelved_ids == (old_record.id,)
        stored = await adapter.get_by_id(old_record.id)
        assert stored.lifecycle == "shelved"

        # The birth axes' own watermark contract is untouched -- initialized to `now`, exactly
        # as it would be with no revisit axis at all.
        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.watermark_at == _NOW

    async def test_revisit_axis_runs_as_part_of_a_completed_pass(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        # A newcomer for the birth axes (keeps, so the pass completes cleanly).
        newcomer = _fact(content="a fresh durable lesson worth keeping", created_at=_T1)
        await adapter.upsert(newcomer, _vec())
        # An old, eligible record for the revisit axis, with evidence since its own birth.
        old_record = _fact(
            content="An old durable gotcha eligible for the revisit rotation.",
            created_at=_REVISIT_BORN,
        )
        await adapter.upsert(old_record, _vec(seed=0.5))
        await _give_revisit_evidence(adapter, old_record)

        class _RoutingJudge:
            def __init__(self, identity: str) -> None:
                self._identity = identity
                self.calls: list[str] = []

            @property
            def identity(self) -> str:
                return self._identity

            def instruction_fingerprint(self) -> str:
                return "not_applicable"

            async def call(self, prompt: str) -> str:
                self.calls.append(prompt)
                if "# Cleaner — revisit rubric" in prompt:
                    return _payload([_v(1, "shelve-clear", why="stale now")])
                return _payload([_v(1, "keep-clear", why="durable")])

        judge_a = _RoutingJudge("a")
        judge_b = _RoutingJudge("b")

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.outcome == "completed"
        assert summary.revisit is not None
        assert summary.revisit.shelved_ids == (old_record.id,)
        stored_old = await adapter.get_by_id(old_record.id)
        assert stored_old.lifecycle == "shelved"
        # The birth axis's own newcomer is untouched by the revisit axis's decision.
        stored_newcomer = await adapter.get_by_id(newcomer.id)
        assert stored_newcomer.lifecycle == "active"

        # A review row landed for the revisit-axis candidate.
        assert len(adapter._record_reviews) == 1
        assert adapter._record_reviews[0].target_id == old_record.id

    async def test_revisit_failure_leaves_watermark_and_outcome_untouched_while_birth_axes_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=_T0, report=None),
        )
        newcomer = _fact(content="a fresh durable lesson worth keeping", created_at=_T1)
        await adapter.upsert(newcomer, _vec())

        async def _boom(**kwargs: Any) -> Any:
            raise RuntimeError("kaboom -- revisit axis exploded")

        monkeypatch.setattr("tabularium.engine.clean.run_revisit_axis", _boom)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        # The birth axes completed exactly as they would have without a revisit axis at all.
        assert summary.outcome == "completed"
        assert summary.watermark_after == _T1
        assert summary.relevance is not None
        assert summary.relevance.shelved_ids == ()

        # The revisit axis's own failure is isolated, never blocking the pass or its watermark.
        assert summary.revisit is None
        assert summary.revisit_error is not None
        assert "kaboom" in summary.revisit_error

    async def test_revisit_axis_still_runs_on_an_empty_birth_pool(self) -> None:
        """The revisit axis must NOT be gated
        behind the birth axes having something new to judge -- a quiet scope (nothing born since
        the watermark) can still have old records due for revisit, and the axis is self-limiting
        (decision 2) regardless of whether it finds anything."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_NOW, last_pass_at=_NOW, report=None),
        )

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a"),
            judge_b=FakeJudgeAdapter(identity="b"),
        )

        assert summary.outcome == "empty_pool"
        # The birth axes still see nothing to judge -- but the revisit axis genuinely ran
        # (an empty summary, not `None`, which would mean it never ran at all).
        assert summary.revisit is not None
        assert summary.revisit.pool_size == 0
        assert summary.revisit_error is None

    async def test_revisit_axis_runs_and_shelves_on_an_empty_birth_pool(self) -> None:
        """The positive case: an old, eligible, evidenced record gets shelved by the revisit axis
        even though the birth pool is empty and the pass outcome is `empty_pool`."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_NOW, last_pass_at=_NOW, report=None),
        )
        old_record = _fact(
            content="An old durable gotcha eligible for the revisit rotation.",
            created_at=_REVISIT_BORN,
        )
        await adapter.upsert(old_record, _vec(seed=0.5))
        await _give_revisit_evidence(adapter, old_record)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.outcome == "empty_pool"
        assert summary.revisit is not None
        assert summary.revisit.shelved_ids == (old_record.id,)
        stored = await adapter.get_by_id(old_record.id)
        assert stored.lifecycle == "shelved"
