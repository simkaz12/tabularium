"""The Cleaner's relevance axis (2026-08-10 cleaner plan).

Hermetic lane only — ``MemoryAdapter`` + ``FakeJudgeAdapter``/local test doubles, no network, no
credential, no real ``claude``/``codex`` process and no real litellm provider call anywhere in this
file (same convention as ``test_judge_seam.py`` / ``test_gate_probe.py``).

Falsifiability (Verify steps 1/2 of the task — both breaks target PRODUCTION code in
``src/tabularium/engine/relevance.py``, never a test-local schema or helper):

1. Split-band auto-apply: change ``axis_common.decide_band``'s
   ``if verdict_a.votes_shelve and verdict_b.votes_shelve: return "shelve"`` back to ``or``
   (restoring the pre-change policy where the more aggressive judge governed a split) and
   re-run ``TestSplitBand`` — both pinned tests go red because the stored Fact's ``lifecycle``
   flips to ``"shelved"`` (a real, unattended write happened on a disagreement) instead of staying
   ``"active"`` with the id routed to ``contested_ids``; ``testdecide_band_unit_split_both_
   directions`` also goes red because ``decide_band(shelve, keep)`` comes back ``"shelve"``
   instead of ``"contested"``.
2. Decision let into the pool: comment out ``_select_relevance_pool``'s
   ``if isinstance(record, Decision): ... continue`` branch and re-run
   ``test_decision_in_pool_is_excluded_and_never_reaches_write``. With only that branch removed,
   the Decision does NOT reach a batch or ``set_shelved`` — ``_RelevancePool.candidates`` is
   typed ``tuple[Fact, ...]``, a second, independent structural guard, and pydantic rejects the
   Decision with a ``ValidationError`` at pool-construction time, one step earlier. (Reaching
   ``set_shelved``'s Facts-only ``ToolError`` — a third line of defence — requires ALSO widening
   that field's type, which this deliberate break does not do.) Either way the test performs a
   bare ``await run_relevance_axis(...)`` with no ``pytest.raises`` around it, so whichever
   exception fires is never caught — it surfaces as an uncaught-exception test failure, not a
   silently green run. What the test genuinely proves: a Decision in the born-since pool produces
   no write and no swallowed exception, regardless of which of the two guards catches it.
3. Hot/ember let into the pool (2026-08-12 fix): revert ``_select_relevance_pool``'s
   ``if record.lifecycle not in ("active", "archived"): ...`` back to
   ``if record.lifecycle == "shelved":`` and re-run
   ``TestHotEmberNeverJudgedOrWritten::test_hot_fact_never_reaches_batch_or_write`` (and its ember
   sibling). Unlike break 2 above, this one does NOT hit the pydantic structural guard — a hot/ember
   record is a legitimately-typed ``Fact`` (``lifecycle`` accepts that value), so
   ``_RelevancePool.candidates: tuple[Fact, ...]`` happily accepts it. It reaches the batch, is
   voted "shelve", and ``set_shelved`` (``engine/dispositions.py:1322``, Facts-only-``active``/
   ``archived`` guard) raises ``tabularium.engine._errors.ToolError`` — UNCAUGHT, since
   ``run_relevance_axis`` never wraps that call — so the test fails with the ``ToolError`` itself as
   an uncaught-exception test failure, not an assertion. This is the exact crash this whole fix
   closes: see ``tests/test_pass_orchestration.py``'s ``TestHotEmberNeverCrashesOrGetsWritten`` for
   the same break reproduced one layer up, through ``clean()``.

Both commands and their captured red output are recorded verbatim in the task verdict, not
duplicated here (matching ``test_judge_seam.py``'s / ``test_rubric.py``'s own convention).
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, FactLifecycle, HypomnemaId, Scope, new_id
from tabularium.engine._errors import ToolError
from tabularium.engine.assessor import Trace, Verdict, compute_confidence
from tabularium.engine.axis_common import (
    BandCounts,
    ContestedVerdict,
    RecordVerdict,
    decide_band,
    deciding_verdict,
)
from tabularium.engine.dispositions import set_shelved as _real_set_shelved
from tabularium.engine.facts import save_internal as _real_save_internal
from tabularium.engine.judge import JudgeUnavailable
from tabularium.engine.relevance import (
    RelevanceAxisSummary,
    RelevanceVerdict,
    _select_relevance_pool,
    run_relevance_axis,
)
from tabularium.engine.rubric import Rubric, RubricParameters, load_rubric
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeJudgeAdapter, TestClock
from tests.support.fake_assessor import EchoAssessor
from tests.support.queue import contested_ids, latest_judgment

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SINCE = datetime(2024, 12, 31, tzinfo=UTC)
_NOW = datetime(2025, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adapter(dim: int = 4) -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic", dim=dim)


def _make_clock() -> TestClock:
    return TestClock(_NOW)


def _vec(dim: int = 4, *, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _fact(
    *,
    content: str,
    scope: str = "proj-a",
    lifecycle: str = "active",
    now: datetime = _NOW,
) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity="valid",
        created_at=now,
        updated_at=now,
    )


def _decision(*, content: str, scope: str = "proj-a", now: datetime = _NOW) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope(scope),
        lifecycle="in_force",
        validity="valid",
        created_at=now,
        updated_at=now,
    )


def _v(index: int, grade: str, *, unclear: bool = False, why: str = "because") -> dict[str, Any]:
    return {"index": index, "grade": grade, "unclear": unclear, "why": why}


def _payload(items: list[dict[str, Any]]) -> str:
    return json.dumps({"verdicts": items})


def _payload_missing_unclear(items: list[dict[str, Any]], *, missing_index: int) -> str:
    """Same shape as ``_payload``, except the item at ``missing_index`` has no ``unclear`` key --
    reproduces the live seat-A defect (2026-08-20) exactly: a required, strict field simply
    absent, not malformed JSON."""
    verdicts = []
    for item in items:
        entry = dict(item)
        if entry["index"] == missing_index:
            entry.pop("unclear", None)
        verdicts.append(entry)
    return json.dumps({"verdicts": verdicts})


def _small_batch_rubric(batch_cap: int) -> Rubric:
    """The shipped rubric with ``batch_cap`` overridden — lets a test force multiple batches
    without needing dozens of fixture records."""
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


class _SequencedJudgeAdapter:
    """Returns a different canned response (or raises) on each successive call, in order —
    ``FakeJudgeAdapter`` only ever returns one fixed response, which cannot express "batch 1
    succeeds, batch 2 fails"."""

    def __init__(self, *, identity: str, responses: list[str | Exception]) -> None:
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
        item = self._responses[len(self.calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item


class _SlowJudgeAdapter:
    """Sleeps ``delay`` seconds before returning — proves ``run_relevance_axis`` calls both judges
    concurrently rather than sequentially (elapsed time ~= one delay, not two)."""

    def __init__(self, *, identity: str, response: str, delay: float) -> None:
        self._identity = identity
        self._response = response
        self._delay = delay
        self.calls: list[str] = []

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        self.calls.append(prompt)
        await asyncio.sleep(self._delay)
        return self._response


async def _spy_on_set_shelved(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Wrap the real ``set_shelved`` so a test can see exactly which ids/reasons were written,
    while the real write still happens (unlike a pure mock, which would leave lifecycle
    unchanged)."""
    calls: list[dict[str, Any]] = []

    async def spy(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return await _real_set_shelved(**kwargs)

    monkeypatch.setattr("tabularium.engine.relevance.set_shelved", spy)
    return calls


# ===========================================================================
# Pool selection: three independent exclusions (acceptance criterion)
# ===========================================================================


class TestSelectRelevancePool:
    def test_excludes_decisions(self) -> None:
        fact = _fact(content="an eligible fact")
        decision = _decision(content="an in-force decision")

        pool = _select_relevance_pool([fact, decision])

        assert pool.excluded_decisions == (decision.id,)
        assert [f.id for f in pool.candidates] == [fact.id]

    def test_excludes_global(self) -> None:
        fact = _fact(content="scoped to proj-a")
        global_fact = _fact(content="owner-written", scope="global")

        pool = _select_relevance_pool([fact, global_fact])

        assert pool.excluded_global == (global_fact.id,)
        assert [f.id for f in pool.candidates] == [fact.id]

    def test_excludes_already_shelved(self) -> None:
        fact = _fact(content="still active")
        shelved_fact = _fact(content="already shelved", lifecycle="shelved")

        pool = _select_relevance_pool([fact, shelved_fact])

        assert pool.excluded_already_shelved == (shelved_fact.id,)
        assert [f.id for f in pool.candidates] == [fact.id]

    def test_excludes_hot(self) -> None:
        """Owner decision, 2026-08-12: the Cleaner must not touch a hot Fact at all -- excluded
        from candidacy here, never demoted-then-processed. Lands in the SAME bucket as
        "already shelved" -- see this class's own docstring / the module docstring's pool-selection
        section for why the two grounds are collapsed into one exclusion."""
        fact = _fact(content="still active")
        hot_fact = _fact(content="pinned hot", lifecycle="hot")

        pool = _select_relevance_pool([fact, hot_fact])

        assert pool.excluded_already_shelved == (hot_fact.id,)
        assert [f.id for f in pool.candidates] == [fact.id]

    def test_excludes_ember(self) -> None:
        """Same owner decision as ``test_excludes_hot``, the other write-time-provisional
        lifecycle."""
        fact = _fact(content="still active")
        ember_fact = _fact(content="write-time-seeded ember", lifecycle="ember")

        pool = _select_relevance_pool([fact, ember_fact])

        assert pool.excluded_already_shelved == (ember_fact.id,)
        assert [f.id for f in pool.candidates] == [fact.id]

    def test_hot_and_ember_join_shelved_in_the_same_exclusion_bucket(self) -> None:
        """The lifecycle exclusion SUBSUMES the original "already shelved" exclusion (2026-08-12
        fix) -- shelved, hot and ember all land in the one ``excluded_already_shelved`` bucket,
        since nothing downstream reads it to distinguish "was shelved" from "is hot/ember"."""
        fact = _fact(content="eligible")
        shelved_fact = _fact(content="shelved", lifecycle="shelved")
        hot_fact = _fact(content="hot", lifecycle="hot")
        ember_fact = _fact(content="ember", lifecycle="ember")

        pool = _select_relevance_pool([fact, shelved_fact, hot_fact, ember_fact])

        assert [f.id for f in pool.candidates] == [fact.id]
        assert set(pool.excluded_already_shelved) == {
            shelved_fact.id,
            hot_fact.id,
            ember_fact.id,
        }

    def test_active_and_archived_facts_are_both_eligible(self) -> None:
        """Both legal shelve-source lifecycles (`set_shelved`'s ``active -> shelved`` and
        ``archived -> shelved`` edges) are candidates — only ``shelved`` itself is excluded."""
        active = _fact(content="active", lifecycle="active")
        archived = _fact(content="archived", lifecycle="archived")

        pool = _select_relevance_pool([active, archived])

        assert {f.id for f in pool.candidates} == {active.id, archived.id}
        assert pool.excluded_already_shelved == ()

    def test_all_three_exclusions_apply_independently_in_one_pool(self) -> None:
        fact = _fact(content="eligible")
        decision = _decision(content="decision")
        global_fact = _fact(content="global", scope="global")
        shelved_fact = _fact(content="shelved", lifecycle="shelved")

        pool = _select_relevance_pool([fact, decision, global_fact, shelved_fact])

        assert [f.id for f in pool.candidates] == [fact.id]
        assert pool.excluded_decisions == (decision.id,)
        assert pool.excluded_global == (global_fact.id,)
        assert pool.excluded_already_shelved == (shelved_fact.id,)


# ===========================================================================
# Band policy: unanimous / split / unclear
# ===========================================================================


class TestUnanimousBands:
    async def test_unanimous_shelve_shelves(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="TICKET-B fix: OrderPolicy now gates viewAny on X")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == (fact.id,)
        assert summary.counts == BandCounts(shelve=1, keep=0, unclear=0)
        # 2026-08-12 rubric-gate-retune fix: `verdicts` carries the deciding judge's `why`
        # (both judges used `_v`'s default "because" here) for every shelved id, unconditionally.
        assert summary.verdicts == (
            RecordVerdict(id=fact.id, band="shelve", why="because"),
        )
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "shelved"

    async def test_unanimous_keep_does_nothing(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="a durable, transferable working lesson")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-lean")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert summary.counts == BandCounts(shelve=0, keep=1, unclear=0)
        assert summary.verdicts == ()  # nothing shelved -- nothing to carry a `why` for
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"


class TestSplitBand:
    """Acceptance criterion: pins the split band explicitly, both directions — a split (one
    ``shelve-*``, one ``keep-*``) is ``contested`` and writes NOTHING (a prior decision,
    superseding an earlier decision's "the more aggressive judge always governs"). Renamed from the
    pre-change version of this class, whose two integration tests pinned the opposite
    outcome (a split shelving)."""

    async def test_judge_a_shelve_judge_b_keep_is_contested(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="borderline record, A shelves")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "shelve-lean", why="perishable now")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "keep-clear", why="durable lesson")])
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert summary.contested_ids == (fact.id,)
        assert summary.counts == BandCounts(shelve=0, keep=0, unclear=0, contested=1)
        # Both judges' identity/grade/`why` are carried -- nobody decided, so there is no single
        # deciding judge to name the way a shelve write does.
        assert summary.contested_verdicts == (
            ContestedVerdict(
                id=fact.id,
                judge_a_identity="a",
                judge_a_grade="shelve-lean",
                judge_a_why="perishable now",
                judge_b_identity="b",
                judge_b_grade="keep-clear",
                judge_b_why="durable lesson",
            ),
        )
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"  # untouched -- nothing written

    async def test_judge_a_keep_judge_b_shelve_is_contested(self) -> None:
        """The reverse split — B shelves, A keeps — must ALSO be contested, never shelved."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="borderline record, B shelves")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "keep-lean", why="ambiguous")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "shelve-clear", why="q1 derivable")])
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert summary.contested_ids == (fact.id,)
        assert summary.contested_verdicts == (
            ContestedVerdict(
                id=fact.id,
                judge_a_identity="a",
                judge_a_grade="keep-lean",
                judge_a_why="ambiguous",
                judge_b_identity="b",
                judge_b_grade="shelve-clear",
                judge_b_why="q1 derivable",
            ),
        )
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"

    def testdecide_band_unit_split_both_directions(self) -> None:
        """Direct unit-level pin of the same property, isolated from the storage/judge
        machinery — the deliberate break in Verify step 1 targets ``decide_band`` itself."""
        shelve = RelevanceVerdict(index=1, grade="shelve-clear", unclear=False, why="x")
        keep = RelevanceVerdict(index=1, grade="keep-clear", unclear=False, why="y")

        assert decide_band(shelve, keep) == "contested"
        assert decide_band(keep, shelve) == "contested"
        assert decide_band(keep, keep) == "keep"
        assert decide_band(shelve, shelve) == "shelve"


class TestContestedReachesTheUnifiedQueue:
    """M2+ "one unified contested queue" slice (brief tripwire 2 + verify step "Relevance and
    falsity contests appear in the queue"): a relevance-axis split must be independently
    visible in storage as a latest ``contested`` review -- not just reported in-memory on
    ``RelevanceAxisSummary``.
    """

    async def test_split_band_record_has_a_contested_latest_review(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="borderline record, queue-visible split")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="claude:opus", response=_payload([_v(1, "shelve-lean", why="perishable")])
        )
        judge_b = FakeJudgeAdapter(
            identity="codex:gpt", response=_payload([_v(1, "keep-clear", why="durable")])
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )
        assert summary.contested_ids == (fact.id,)

        # Falsifiability: before this slice, the relevance axis wrote NOTHING to
        # `record_reviews` for a contested band -- this assertion would fail with an empty list.
        queued = contested_ids(adapter, Scope("proj-a"))
        assert queued == {fact.id}

        history = await adapter.list_reviews_for(target_id=fact.id)
        assert len(history) == 1
        review = history[0]
        assert review.band == "contested"

        # Judge identity/grade/`why` live in the SEPARATE `record_judgments` table (owner
        # correction, 2026-08-16), never as fields on `RecordReview` itself.
        judgment = latest_judgment(adapter, fact.id)
        assert judgment is not None
        assert judgment.judge_a_identity == "claude:opus"
        assert judgment.judge_a_grade == "shelve-lean"
        assert judgment.judge_a_why == "perishable"
        assert judgment.judge_b_identity == "codex:gpt"
        assert judgment.judge_b_why == "durable"

    async def test_dry_run_appends_no_review_row(self) -> None:
        """Mirrors the revisit axis's own dry-run contract: judging is identical, only the writes
        are skipped -- extended here to the NEW `record_reviews` write this slice adds."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="borderline record, dry run")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "shelve-lean", why="perishable")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "keep-clear", why="durable")])
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            dry_run=True,
        )
        assert summary.contested_ids == (fact.id,)
        assert await adapter.list_reviews_for(target_id=fact.id) == []
        assert contested_ids(adapter, Scope("proj-a")) == set()
        assert latest_judgment(adapter, fact.id) is None


class TestDecidingVerdict:
    """``deciding_verdict`` is only called for the ``"shelve"`` band, where both judges voted
    shelve, and returns judge A's verdict there."""

    def test_returns_verdict_a_when_both_vote_shelve(self) -> None:
        a = RelevanceVerdict(index=1, grade="shelve-clear", unclear=False, why="A's reasoning")
        b = RelevanceVerdict(index=1, grade="shelve-lean", unclear=False, why="B's reasoning")

        assert decide_band(a, b) == "shelve"
        assert deciding_verdict(a, b) is a


class TestUnclear:
    async def test_unclear_from_judge_a_is_not_acted_on(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="cannot be judged from content alone")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "keep-lean", unclear=True, why="unjudgeable")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert summary.unclear_ids == (fact.id,)
        assert summary.counts == BandCounts(shelve=0, keep=0, unclear=1)
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"

    async def test_unclear_from_judge_b_is_not_acted_on(self) -> None:
        """Same property, the other judge marks unclear — proves it is not A-specific."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="cannot be judged from content alone, reverse")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "keep-lean", unclear=True, why="unjudgeable")])
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert summary.unclear_ids == (fact.id,)
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"


# ===========================================================================
# The panel: identical batch text, run concurrently
# ===========================================================================


class TestPanelMechanics:
    async def test_both_judges_receive_identical_batch_text(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="one record, checked by both judges")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert len(judge_a.calls) == 1
        assert len(judge_b.calls) == 1
        assert judge_a.calls[0] == judge_b.calls[0]
        assert fact.content in judge_a.calls[0]

    async def test_judges_run_concurrently_not_sequentially(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="concurrency probe")
        await adapter.upsert(fact, _vec())

        delay = 0.2
        judge_a = _SlowJudgeAdapter(
            identity="a", response=_payload([_v(1, "keep-clear")]), delay=delay
        )
        judge_b = _SlowJudgeAdapter(
            identity="b", response=_payload([_v(1, "keep-clear")]), delay=delay
        )

        start = time.monotonic()
        await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )
        elapsed = time.monotonic() - start

        # Sequential would take ~2 * delay; concurrent takes ~1 * delay. 1.5x delay is a
        # generous cutoff that stays well clear of scheduler jitter either way.
        assert elapsed < delay * 1.5, f"judges ran sequentially, not concurrently ({elapsed=})"


# ===========================================================================
# The write: reason names the deciding gate question, judge pair, rubric version
# ===========================================================================


class TestShelveReason:
    """Note: a shelve write only ever happens on unanimous agreement now
    (both judges voted ``shelve-*``), so the scenario below is unanimous rather than a split — the
    pre-change version of this class exercised the reason-naming property through a genuine
    split (one judge shelves, one keeps); that split now routes to ``contested`` and writes
    nothing, so it can no longer prove this property at the axis-integration level. The reverse
    -split half of the old class (naming whichever judge actually shelved, not always A) is gone,
    because that branch is unreachable under the current band policy and ``deciding_verdict`` no
    longer has it."""

    async def test_reason_names_deciding_why_judge_pair_and_rubric_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = await _spy_on_set_shelved(monkeypatch)

        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="a record with a clear gate answer")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="claude:sonnet",
            response=_payload([_v(1, "shelve-clear", why="Derivable from git history (q1).")]),
        )
        judge_b = FakeJudgeAdapter(
            identity="codex:gpt-5.1-codex",
            response=_payload([_v(1, "shelve-lean", why="B's own, weaker agreement")]),
        )

        rubric = load_rubric()
        await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=rubric,
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert len(calls) == 1
        reason = calls[0]["reason"]
        assert reason.strip()  # non-empty (acceptance criterion)
        assert "Derivable from git history (q1)." in reason  # the deciding judge's `why`
        assert "claude:sonnet" in reason and "codex:gpt-5.1-codex" in reason  # judge pair
        assert rubric.version in reason  # rubric version


# ===========================================================================
# Judge-call failure: nothing written for that batch, failure surfaced
# ===========================================================================


class TestOversizeUnit:
    async def test_a_single_oversize_fact_fails_its_own_batch_and_the_rest_are_judged(self) -> None:
        from tabularium.engine.judge import PROMPT_CHAR_CEILING

        adapter = _make_adapter()
        big = _fact(content="oversize " + "q" * (PROMPT_CHAR_CEILING + 100))
        normal = _fact(content="a normal sized record")
        await adapter.upsert(big, _vec())
        await adapter.upsert(normal, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=_make_clock(),
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert failure.record_ids == (big.id,)
        assert failure.judge_a_outcome == "unavailable"
        assert failure.judge_b_outcome == "unavailable"
        assert str(PROMPT_CHAR_CEILING) in failure.error
        assert summary.counts.keep == 1
        assert len(judge_a.calls) == 1
        assert len(judge_b.calls) == 1
        assert "a normal sized record" in judge_a.calls[0]
        assert "q" * 1_000 not in judge_a.calls[0]


class TestBatchFailure:
    async def test_invalid_output_writes_nothing_and_is_surfaced(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="would have been shelved if the batch had succeeded")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response="not json at all")

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert summary.unclear_ids == ()
        assert summary.counts == BandCounts(shelve=0, keep=0, unclear=0)
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert failure.record_ids == (fact.id,)
        assert failure.judge_a_outcome == "ok"
        assert failure.judge_b_outcome == "invalid_output"
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"

    async def test_unavailable_writes_nothing_and_is_surfaced(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="would have been shelved if the batch had succeeded")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", should_fail=True, fail_reason="401 expired")
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert failure.judge_a_outcome == "unavailable"
        assert failure.judge_b_outcome == "ok"
        assert "401 expired" in failure.error
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"

    async def test_failure_in_one_batch_does_not_affect_another(self) -> None:
        """batch_cap=1 forces two separate judge_batch calls (one per record) — the first
        batch succeeds and shelves its record while the second fails and writes nothing,
        proving batches are independent."""
        adapter = _make_adapter()
        clock = _make_clock()
        first = _fact(content="batch 1 -- succeeds")
        second = _fact(content="batch 2 -- fails", now=_NOW.replace(microsecond=1))
        await adapter.upsert(first, _vec(seed=1.0))
        await adapter.upsert(second, _vec(seed=2.0))

        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[
                _payload([_v(1, "shelve-clear")]),
                _payload([_v(1, "shelve-clear")]),
            ],
        )
        judge_b = _SequencedJudgeAdapter(
            identity="b",
            responses=[
                _payload([_v(1, "shelve-lean")]),
                "not json at all",
                "not json at all",
                "not json at all",
            ],
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=_small_batch_rubric(batch_cap=1),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == (first.id,)
        assert len(summary.failed_batches) == 1
        assert summary.failed_batches[0].record_ids == (second.id,)
        assert len(judge_a.calls) == 2
        assert len(judge_b.calls) == 4

        stored_first = await adapter.get_by_id(first.id)
        stored_second = await adapter.get_by_id(second.id)
        assert stored_first.lifecycle == "shelved"
        assert stored_second.lifecycle == "active"


class TestSchemaInvalidRetryAndDrop:
    async def test_retry_recovers_and_nothing_is_dropped(self) -> None:
        """A judge whose output fails schema validation once, then validates on retry, drops
        nothing -- no failed batch, both records judged normally."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="seat A drops unclear on the first try, not the second")
        await adapter.upsert(fact, _vec())

        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[
                _payload_missing_unclear([_v(1, "shelve-clear")], missing_index=1),
                _payload([_v(1, "shelve-clear")]),
            ],
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == (fact.id,)
        assert summary.failed_batches == ()
        assert len(judge_a.calls) == 2
        assert len(judge_b.calls) == 1
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "shelved"

    async def test_second_failure_drops_only_that_record_rest_of_batch_proceeds(self) -> None:
        """A batch retry that also fails validation falls back to per-record judging: the record
        whose own single-record call still fails validation is dropped and reported, while the
        rest of the same batch is judged and written normally."""
        adapter = _make_adapter()
        clock = _make_clock()
        keeps_going = _fact(content="judged fine both times, should still get shelved")
        never_validates = _fact(
            content="seat A can never produce a valid verdict for this one",
            now=_NOW.replace(microsecond=1),
        )
        await adapter.upsert(keeps_going, _vec(seed=1.0))
        await adapter.upsert(never_validates, _vec(seed=2.0))

        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[
                _payload_missing_unclear(
                    [_v(1, "shelve-clear"), _v(2, "shelve-clear")], missing_index=2
                ),
                _payload_missing_unclear(
                    [_v(1, "shelve-clear"), _v(2, "shelve-clear")], missing_index=2
                ),
                _payload([_v(1, "shelve-clear")]),
                _payload_missing_unclear([_v(1, "shelve-clear")], missing_index=1),
            ],
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "shelve-clear"), _v(2, "shelve-clear")])
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=_small_batch_rubric(batch_cap=2),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == (keeps_going.id,)
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert failure.record_ids == (never_validates.id,)
        assert failure.judge_a_outcome == "invalid_output"
        assert failure.judge_b_outcome == "ok"
        assert len(judge_a.calls) == 4
        assert len(judge_b.calls) == 1

        stored_kept_going = await adapter.get_by_id(keeps_going.id)
        stored_never_validates = await adapter.get_by_id(never_validates.id)
        assert stored_kept_going.lifecycle == "shelved"
        assert stored_never_validates.lifecycle == "active"

    async def test_hard_failure_on_retry_blocks_the_whole_batch(self) -> None:
        """The retry landing on `judge_unavailable` (not another schema failure) is a DIFFERENT,
        pre-existing failure shape -- the whole batch must fail exactly as it always has, never a
        per-record drop, and no per-record fallback is attempted."""
        adapter = _make_adapter()
        clock = _make_clock()
        first = _fact(content="never gets a decided verdict either")
        second = _fact(content="never gets a decided verdict either, part two")
        await adapter.upsert(first, _vec(seed=1.0))
        await adapter.upsert(second, _vec(seed=2.0))

        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[
                _payload_missing_unclear(
                    [_v(1, "shelve-clear"), _v(2, "shelve-clear")], missing_index=2
                ),
                JudgeUnavailable("seat down between the initial call and the retry"),
            ],
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "shelve-clear"), _v(2, "shelve-clear")])
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=_small_batch_rubric(batch_cap=2),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert set(failure.record_ids) == {first.id, second.id}
        assert failure.judge_a_outcome == "unavailable"
        assert failure.judge_b_outcome == "ok"
        assert len(judge_a.calls) == 2
        assert len(judge_b.calls) == 1

        stored_first = await adapter.get_by_id(first.id)
        stored_second = await adapter.get_by_id(second.id)
        assert stored_first.lifecycle == "active"
        assert stored_second.lifecycle == "active"

    async def test_hard_failure_during_per_record_fallback_blocks_the_whole_batch(self) -> None:
        """The same hard-failure rule applies one tier deeper: a seat going `judge_unavailable`
        mid per-record fallback must still fail the WHOLE batch, not just the records not yet
        attempted -- and must stop spending calls on that seat immediately."""
        adapter = _make_adapter()
        clock = _make_clock()
        first = _fact(content="would have validated fine alone, never gets the chance")
        second = _fact(content="the seat goes down before this one is even attempted")
        await adapter.upsert(first, _vec(seed=1.0))
        await adapter.upsert(second, _vec(seed=2.0))

        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[
                _payload_missing_unclear(
                    [_v(1, "shelve-clear"), _v(2, "shelve-clear")], missing_index=2
                ),
                _payload_missing_unclear(
                    [_v(1, "shelve-clear"), _v(2, "shelve-clear")], missing_index=2
                ),
                JudgeUnavailable("seat goes down partway through the per-record fallback"),
            ],
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "shelve-clear"), _v(2, "shelve-clear")])
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=_small_batch_rubric(batch_cap=2),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert set(failure.record_ids) == {first.id, second.id}
        assert failure.judge_a_outcome == "unavailable"
        assert failure.judge_b_outcome == "ok"
        assert len(judge_a.calls) == 3
        assert len(judge_b.calls) == 1

        stored_first = await adapter.get_by_id(first.id)
        stored_second = await adapter.get_by_id(second.id)
        assert stored_first.lifecycle == "active"
        assert stored_second.lifecycle == "active"

    async def test_initial_hard_failure_on_one_seat_stops_the_others_retry(self) -> None:
        """B is hard-unavailable from its very first call while A's first call is merely
        schema-invalid -- A must never be retried, since the batch is already doomed to fail
        regardless of what A's resolution would have found."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="B is down before A even gets a chance to retry")
        await adapter.upsert(fact, _vec())

        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[_payload_missing_unclear([_v(1, "shelve-clear")], missing_index=1)],
        )
        judge_b = FakeJudgeAdapter(identity="b", should_fail=True, fail_reason="B is down")

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert failure.record_ids == (fact.id,)
        assert failure.judge_a_outcome == "invalid_output"
        assert failure.judge_b_outcome == "unavailable"
        assert len(judge_a.calls) == 1
        assert len(judge_b.calls) == 1
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"

    async def test_hard_failure_resolving_a_stops_b_before_b_is_resolved(self) -> None:
        """A's own resolution (retry) turns hard while B's first call was merely schema-invalid --
        B must never be retried either, since A already dooms the whole batch."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="A goes down on its retry before B is ever resolved")
        await adapter.upsert(fact, _vec())

        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[
                _payload_missing_unclear([_v(1, "shelve-clear")], missing_index=1),
                JudgeUnavailable("seat down on the retry"),
            ],
        )
        judge_b = _SequencedJudgeAdapter(
            identity="b",
            responses=[_payload_missing_unclear([_v(1, "shelve-clear")], missing_index=1)],
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == ()
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert failure.record_ids == (fact.id,)
        assert failure.judge_a_outcome == "unavailable"
        assert failure.judge_b_outcome == "invalid_output"
        assert len(judge_a.calls) == 2
        assert len(judge_b.calls) == 1
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"


# ===========================================================================
# Per-call judge telemetry (cleaner-telemetry change): every `judge_batch`
# call actually made -- both seats, every batch, success or failure alike -- is collected into
# `RelevanceAxisSummary.calls`, never just the failed ones.
#
# Falsifiability (Verify step 1 of the brief): comment out one of the two
# `calls.append(outcome_X.as_call_record(...))` lines in `run_relevance_axis` (production,
# `src/tabularium/engine/relevance.py`) -- or move both `calls.append` calls to AFTER the
# `if judge_a_outcome != "ok" or judge_b_outcome != "ok": ... continue` check, so a failed call
# never reaches the list -- and re-run
# `TestJudgeCallTelemetry::test_mixed_outcome_batch_records_both_calls_not_just_the_survivor``.
# It goes red because `summary.calls` comes back with only ONE entry for the batch instead of
# two, with the failed seat's record silently missing.
# ===========================================================================


class TestJudgeCallTelemetry:
    async def test_mixed_outcome_batch_records_both_calls_not_just_the_survivor(self) -> None:
        """One seat fails (invalid JSON), the other succeeds, in the SAME batch. Both calls must
        appear in `summary.calls` -- the failing one alongside the surviving one, neither
        dropped (acceptance criterion)."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="a record whose panel is split between success and garbage")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="claude:sonnet", response=_payload([_v(1, "keep-clear")])
        )
        judge_b = FakeJudgeAdapter(identity="codex:gpt", response="not json at all")

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert len(summary.calls) == 4, (
            "one failed call must not suppress the other seat's call for the same batch -- "
            "codex:gpt's own retry and per-record fallback (2026-08-26 schema-retry fix) add "
            "three of its own calls to claude:sonnet's one"
        )
        by_judge = {c.judge: c for c in summary.calls}
        assert by_judge["claude:sonnet"].status == "success"
        assert by_judge["claude:sonnet"].error is None
        assert by_judge["codex:gpt"].status == "invalid_output"
        assert by_judge["codex:gpt"].error is not None
        # Both calls are attributed to the same batch and the same rubric prose.
        assert by_judge["claude:sonnet"].batch_index == 0
        assert by_judge["codex:gpt"].batch_index == 0
        assert by_judge["claude:sonnet"].rubric_version == load_rubric().version
        assert by_judge["codex:gpt"].rubric_version == load_rubric().version
        # Completion tokens and cost are present and null on every branch (acceptance criterion).
        for record in summary.calls:
            assert record.completion_tokens is None
            assert record.cost is None
            assert record.prompt_tokens > 0
            assert record.duration_seconds >= 0.0

    async def test_two_batches_two_seats_yields_four_call_entries(self) -> None:
        """`batch_cap=1` forces two separate batches -- two seats x two batches = four call
        entries, both successful here (acceptance criterion: "two seats x N batches")."""
        adapter = _make_adapter()
        clock = _make_clock()
        first = _fact(content="batch 1 record")
        second = _fact(content="batch 2 record", now=_NOW.replace(microsecond=1))
        await adapter.upsert(first, _vec(seed=1.0))
        await adapter.upsert(second, _vec(seed=2.0))

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=_small_batch_rubric(batch_cap=1),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert len(summary.calls) == 4
        assert sorted(c.batch_index for c in summary.calls) == [0, 0, 1, 1]
        assert all(c.status == "success" for c in summary.calls)

    async def test_calls_present_even_when_pass_shelves_and_writes(self) -> None:
        """Telemetry collection is independent of what the batch decided -- a batch that shelves
        a record still records its calls exactly like one that keeps everything."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="a record the panel unanimously shelves")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.shelved_ids == (fact.id,)
        assert len(summary.calls) == 2


# ===========================================================================
# No writes to a Decision, ever -- excluded before batching by `_select_relevance_pool`'s
# explicit isinstance check, with `_RelevancePool.candidates: tuple[Fact, ...]` as a second,
# independent structural guard (a Decision that somehow reached it would fail pydantic
# validation at pool-construction time). `set_shelved`'s own Facts-only `ToolError` is a third
# line of defence behind both. This test (the Verify step 2 deliberate-break target) proves the
# outcome that matters regardless of which guard is live: no write, no swallowed exception.
# ===========================================================================


class TestDecisionNeverWritten:
    async def test_decision_in_pool_is_excluded_and_never_reaches_write(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        decision = _decision(content="an in-force decision born in the window")
        await adapter.upsert(decision, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.excluded_decisions == (decision.id,)
        assert summary.shelved_ids == ()
        # The Decision never even reaches a batch -- a pool of only excluded records means
        # zero batches are formed, so neither judge is ever called.
        assert judge_a.calls == []
        assert judge_b.calls == []

    async def test_eligible_fact_still_processed_when_a_decision_is_mixed_into_the_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = await _spy_on_set_shelved(monkeypatch)

        adapter = _make_adapter()
        clock = _make_clock()
        eligible = _fact(content="eligible fact")
        decision = _decision(content="an in-force decision")
        await adapter.upsert(eligible, _vec(seed=1.0))
        await adapter.upsert(decision, _vec(seed=2.0))

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert [c["id"] for c in calls] == [eligible.id]
        assert summary.excluded_decisions == (decision.id,)
        assert summary.shelved_ids == (eligible.id,)


# ===========================================================================
# Hot/ember Facts never judged or written — owner decision, 2026-08-12
# ===========================================================================


class TestHotEmberNeverJudgedOrWritten:
    """Owner decision, 2026-08-12: the Cleaner must not touch a hot or ember Fact at all -- not
    demoted-then-processed, excluded from candidacy before a judge is ever called. Mirrors
    ``TestDecisionNeverWritten`` above exactly. Before this fix, a hot/ember Fact reached the
    panel, was voted "shelve", and the resulting ``set_shelved`` call raised an uncaught
    ``ToolError`` (it only accepts an ``active``/``archived`` source lifecycle) -- see
    ``tests/test_pass_orchestration.py``'s ``TestHotEmberNeverCrashesOrGetsWritten`` for the
    end-to-end reproduction of that crash."""

    async def test_hot_fact_never_reaches_batch_or_write(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        hot_fact = _fact(content="a pinned hot fact born in the window", lifecycle="hot")
        await adapter.upsert(hot_fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.excluded_already_shelved == (hot_fact.id,)
        assert summary.shelved_ids == ()
        # The hot Fact never even reaches a batch -- a pool of only excluded records means zero
        # batches are formed, so neither judge is ever called.
        assert judge_a.calls == []
        assert judge_b.calls == []
        stored = await adapter.get_by_id(hot_fact.id)
        assert stored.lifecycle == "hot"

    async def test_ember_fact_never_reaches_batch_or_write(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        ember_fact = _fact(
            content="a write-time-seeded ember born in the window", lifecycle="ember"
        )
        await adapter.upsert(ember_fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.excluded_already_shelved == (ember_fact.id,)
        assert summary.shelved_ids == ()
        assert judge_a.calls == []
        assert judge_b.calls == []
        stored = await adapter.get_by_id(ember_fact.id)
        assert stored.lifecycle == "ember"

    async def test_eligible_fact_still_processed_when_a_hot_fact_is_mixed_into_the_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = await _spy_on_set_shelved(monkeypatch)

        adapter = _make_adapter()
        clock = _make_clock()
        eligible = _fact(content="eligible fact")
        hot_fact = _fact(content="pinned hot fact", lifecycle="hot")
        await adapter.upsert(eligible, _vec(seed=1.0))
        await adapter.upsert(hot_fact, _vec(seed=2.0))

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert [c["id"] for c in calls] == [eligible.id]
        assert summary.excluded_already_shelved == (hot_fact.id,)
        assert summary.shelved_ids == (eligible.id,)


# ===========================================================================
# `explicit_pool` (2026-08-12 calibration-harness fix): bypasses `list_born_since` AND
# `_select_relevance_pool`'s exclusions entirely, under `dry_run=True` -- what
# `scripts/calibrate_relevance_rubric.py` needs to re-judge an already-shelved/hot batch.
# ===========================================================================


class TestExplicitPool:
    async def test_hot_fact_reaches_the_judge_via_explicit_pool_and_is_never_written(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        hot_fact = _fact(content="a hot fact fed in explicitly", lifecycle="hot")
        await adapter.upsert(hot_fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
            dry_run=True,
            explicit_pool=[hot_fact],
        )

        # Bypassed pool selection entirely -- reaches the judge, unlike the ordinary
        # pool-selection path (TestHotEmberNeverJudgedOrWritten, which excludes it unjudged).
        assert len(judge_a.calls) == 1
        assert summary.pool_size == 1
        assert summary.excluded_decisions == ()
        assert summary.excluded_global == ()
        assert summary.excluded_already_shelved == ()
        assert summary.shelved_ids == (hot_fact.id,)
        # dry_run=True: no write was ever attempted -- a live write here would have raised
        # ToolError (hot is not a legal `set_shelved` source lifecycle).
        stored = await adapter.get_by_id(hot_fact.id)
        assert stored.lifecycle == "hot"

    async def test_explicit_pool_none_is_unchanged_production_behaviour(self) -> None:
        """``explicit_pool=None`` (the default) must reproduce the ordinary pool-selection path
        exactly -- a hot Fact stays excluded and unjudged, same as
        ``TestHotEmberNeverJudgedOrWritten``."""
        adapter = _make_adapter()
        clock = _make_clock()
        hot_fact = _fact(content="a hot fact, ordinary pool selection", lifecycle="hot")
        await adapter.upsert(hot_fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert judge_a.calls == []
        assert summary.excluded_already_shelved == (hot_fact.id,)

    async def test_explicit_pool_with_dry_run_false_raises_without_calling_the_judge(self) -> None:
        """``explicit_pool`` bypasses
        ``_select_relevance_pool``'s scope/type/lifecycle exclusions entirely, so pairing it with
        ``dry_run=False`` would be a live-write bypass for any future caller -- enforced here as a
        ``ToolError``, checked BEFORE the judge is ever called (never a partial run that queries a
        judge and only then refuses to write)."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="would be a live write if this guard were missing")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        with pytest.raises(ToolError, match="explicit_pool requires dry_run=True"):
            await run_relevance_axis(
                scope=Scope("proj-a"),
                since=_SINCE,
                adapter=adapter,
                clock=clock,
                lock=asyncio.Lock(),
                cfg=EngineConfig(),
                rubric=load_rubric(),
                judge_a=judge_a,
                judge_b=judge_b,
                dry_run=False,
                explicit_pool=[fact],
            )

        assert judge_a.calls == []
        assert judge_b.calls == []


# ===========================================================================
# `full_verdicts` (2026-08-12 calibration-harness fix): both seats' grade/why/unclear for EVERY
# judged record, every band -- unlike `verdicts` (shelve-only) and `contested_verdicts`
# (contested-only).
# ===========================================================================


class TestFullVerdicts:
    async def test_full_verdicts_covers_keep_shelve_unclear_and_contested(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        keep = _fact(content="unanimously kept")
        shelve = _fact(content="unanimously shelved", now=_NOW.replace(microsecond=1))
        unclear = _fact(content="unclear from judge a", now=_NOW.replace(microsecond=2))
        contested = _fact(content="split verdict", now=_NOW.replace(microsecond=3))
        for f, seed in ((keep, 1.0), (shelve, 2.0), (unclear, 3.0), (contested, 4.0)):
            await adapter.upsert(f, _vec(seed=seed))

        judge_a = FakeJudgeAdapter(
            identity="claude:sonnet",
            response=_payload(
                [
                    _v(1, "keep-clear", why="durable"),
                    _v(2, "shelve-clear", why="stale"),
                    _v(3, "keep-lean", unclear=True, why="cannot tell"),
                    _v(4, "shelve-lean", why="A shelves"),
                ]
            ),
        )
        judge_b = FakeJudgeAdapter(
            identity="codex:gpt",
            response=_payload(
                [
                    _v(1, "keep-lean", why="still useful"),
                    _v(2, "shelve-lean", why="also stale"),
                    _v(3, "shelve-clear", why="B thinks shelve"),
                    _v(4, "keep-clear", why="B keeps"),
                ]
            ),
        )

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        by_id = {fv.id: fv for fv in summary.full_verdicts}
        assert len(by_id) == 4

        assert by_id[keep.id].band == "keep"
        assert by_id[keep.id].judge_a_grade == "keep-clear"
        assert by_id[keep.id].judge_a_why == "durable"
        assert by_id[keep.id].judge_b_grade == "keep-lean"
        assert by_id[keep.id].judge_a_unclear is False
        assert by_id[keep.id].judge_b_unclear is False

        assert by_id[shelve.id].band == "shelve"
        assert by_id[unclear.id].band == "unclear"
        assert by_id[unclear.id].judge_a_unclear is True
        assert by_id[contested.id].band == "contested"


# ===========================================================================
# Structural guards
# ===========================================================================


class TestStructuralGuards:
    def test_no_bury_call_anywhere_in_the_relevance_module(self) -> None:
        source = (_REPO_ROOT / "src" / "tabularium" / "engine" / "relevance.py").read_text()
        assert ".bury(" not in source
        assert "adapter.bury" not in source

    def test_summary_is_returned_not_a_report_written(self) -> None:
        """This module returns ``RelevanceAxisSummary``; it must never call anything that
        writes a report itself (not this axis's job)."""
        import inspect

        sig = inspect.signature(run_relevance_axis)
        assert sig.return_annotation in ("RelevanceAxisSummary", RelevanceAxisSummary)


# ===========================================================================
# First pass on an empty pool is a clean no-op
# ===========================================================================


class TestEmptyPool:
    async def test_empty_pool_is_a_clean_no_op(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert summary.pool_size == 0
        assert summary.shelved_ids == ()
        assert summary.counts == BandCounts(shelve=0, keep=0, unclear=0)
        assert judge_a.calls == []
        assert judge_b.calls == []


# ===========================================================================
# A refused write skips the record, never the pass (2026-08-12 crash-loop fix)
# ===========================================================================


class _RacingJudgeAdapter:
    """Behaves like ``FakeJudgeAdapter`` (one fixed response) but, as a side effect of ITS OWN
    call, mutates ``race_id``'s lifecycle directly on the adapter to ``"hot"`` before returning —
    a genuine TOCTOU reproduction of the crash-loop bug, not a monkeypatch of the write path
    itself: ``_select_relevance_pool``'s exclusion already ran (its snapshot was taken before any
    judge is called), and by the time ``run_relevance_axis``'s write loop reaches ``race_id``, the
    concurrent mutation has already landed (``asyncio.gather`` awaits this call to completion
    before the loop below ever runs) — exactly the window another session's ``set_hot`` call would
    occupy in production."""

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


class TestWriteRefusalSkipsRecordNotPass:
    """Reproduces the crash-loop this fix closes, at the axis layer. Before this fix,
    ``set_shelved``'s ``ToolError`` (raised when a record's lifecycle no longer accepts a shelve
    source at write time) was uncaught here and propagated out of ``run_relevance_axis`` entirely.
    """

    async def test_racing_record_is_skipped_other_record_still_shelved_and_surfaced(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()

        raced_fact = _fact(content="promoted to hot mid-pass by another session")
        ordinary_fact = _fact(content="an unremarkable fact that should still shelve")
        await adapter.upsert(raced_fact, _vec())
        await adapter.upsert(ordinary_fact, _vec())

        response = _payload([_v(1, "shelve-clear"), _v(2, "shelve-clear")])
        judge_a = _RacingJudgeAdapter(
            adapter=adapter, clock=clock, race_id=raced_fact.id, response=response
        )
        judge_b = FakeJudgeAdapter(identity="b", response=response)

        # No `pytest.raises` around this call -- proving the pass completes is exactly the point;
        # an uncaught `ToolError` escaping here would fail this test on its own.
        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        # 1. The other record is still shelved.
        assert summary.shelved_ids == (ordinary_fact.id,)
        # 2. The refused record is surfaced -- id and a real refusal reason, not swallowed.
        assert len(summary.refused) == 1
        refusal = summary.refused[0]
        assert refusal.id == raced_fact.id
        assert "shelved" in refusal.reason

        # 3. The store agrees: the raced record was never shelved, the ordinary one really was.
        stored_raced = await adapter.get_by_id(raced_fact.id)
        stored_ordinary = await adapter.get_by_id(ordinary_fact.id)
        assert stored_raced.lifecycle == "hot"
        assert stored_ordinary.lifecycle == "shelved"

    async def test_infrastructure_failure_during_the_write_still_aborts_the_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proves the catch is narrow: an exception ``set_shelved`` does NOT convert to
        ``ToolError`` (its own try/except around the storage calls converts only ``KeyError`` --
        ``engine/dispositions.py``'s ``set_shelved``) must still propagate, uncaught, out of
        ``run_relevance_axis`` -- a real infrastructure failure (store unavailable, disk, embedder)
        must still abort the pass loudly rather than being silently skipped like a legality
        refusal."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="a fact whose shelve write hits a real storage failure")
        await adapter.upsert(fact, _vec())

        async def _broken_append_event(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("store unavailable")

        # `set_shelved` itself only catches `KeyError` around its storage calls (see
        # `engine/dispositions.py`) -- any other exception from `append_event`/`transition` is
        # exactly the "real storage failure" class this axis's narrow `except ToolError` must NOT
        # absorb.
        monkeypatch.setattr(adapter, "append_event", _broken_append_event)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        with pytest.raises(RuntimeError, match="store unavailable"):
            await run_relevance_axis(
                scope=Scope("proj-a"),
                since=_SINCE,
                adapter=adapter,
                clock=clock,
                lock=asyncio.Lock(),
                cfg=EngineConfig(),
                rubric=load_rubric(),
                judge_a=judge_a,
                judge_b=judge_b,
            )


# ---------------------------------------------------------------------------
# A3 cleaner-integration piece: pre-pass, skip switch, split band
# ---------------------------------------------------------------------------


async def _prepass_row_for(adapter: MemoryAdapter, candidate_id: HypomnemaId) -> Any:
    rows = list(adapter._verdicts.values())
    matches = [
        r for r in rows if r.caller == "cleaner_prepass" and r.candidate_id == str(candidate_id)
    ]
    assert len(matches) == 1, f"expected exactly one cleaner_prepass row, got {len(matches)}"
    return matches[0]


class TestPrepassShadow:
    async def test_all_switches_off_writes_one_row_per_newcomer_panel_judges_all(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        f1 = _fact(content="record one")
        f2 = _fact(content="record two")
        f3 = _fact(content="record three")
        for f in (f1, f2, f3):
            await adapter.upsert(f, _vec())

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                "record one": _keep_verdict("keep-clear"),
                "record two": _keep_verdict("keep-clear"),
                "record three": _keep_verdict("keep-clear"),
            },
        )
        response = _payload([_v(1, "keep-clear"), _v(2, "keep-clear"), _v(3, "keep-clear")])
        judge_a = FakeJudgeAdapter(identity="a", response=response)
        judge_b = FakeJudgeAdapter(identity="b", response=response)

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.counts.keep == 3
        rows = list(adapter._verdicts.values())
        prepass_rows = [r for r in rows if r.caller == "cleaner_prepass"]
        assert len(prepass_rows) == 3
        for f in (f1, f2, f3):
            row = await _prepass_row_for(adapter, f.id)
            assert row.action == "panel_keep"
            assert row.actor == "panel"
            assert row.status == "acted"
            assert row.matched is True

    async def test_prepass_timeout_degrades_to_panel_and_batch_still_succeeds(self) -> None:
        """D4: `unavailable` never holds the watermark -- proxied here by the failed_batches
        list staying empty, which is `clean.py`'s own watermark-advance gate."""
        adapter = _make_adapter()
        clock = _make_clock()
        f1 = _fact(content="times out on the pre-pass")
        f2 = _fact(content="answers normally")
        await adapter.upsert(f1, _vec(seed=1.0))
        await adapter.upsert(f2, _vec(seed=2.0))

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                "times out on the pre-pass": TimeoutError("injected"),
                "answers normally": _keep_verdict("keep-clear"),
            },
        )
        response = _payload([_v(1, "keep-clear"), _v(2, "keep-clear")])
        judge_a = FakeJudgeAdapter(identity="a", response=response)
        judge_b = FakeJudgeAdapter(identity="b", response=response)

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.failed_batches == ()
        assert summary.counts.keep == 2
        row = await _prepass_row_for(adapter, f1.id)
        assert row.unavailable_reason == "timeout"
        assert row.pick is None
        assert row.action == "panel_keep"
        assert row.matched is None


def _keep_verdict(grade: str, *, confidence: float = 0.95) -> Verdict:
    peak = confidence
    other = (1.0 - peak) / 3
    odds = {g: (peak if g == grade else other) for g in _RELEVANCE_GRADES}
    return Verdict(
        method="relevance", pick=grade, odds=odds, confidence=compute_confidence(odds),
        trace=Trace(
            model="echo", wording_hash="echo", latency_ms=0,
            input_tokens=None, output_tokens=None, request_id=None,
        ),
    )


_RELEVANCE_GRADES = ["shelve-clear", "shelve-lean", "keep-lean", "keep-clear"]


class TestPrepassSkipSwitch:
    async def test_keep_clear_above_cutoff_skips_the_panel(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        f = _fact(content="a very safe keep")
        await adapter.upsert(f, _vec())

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={"a very safe keep": _keep_verdict("keep-clear", confidence=0.95)},
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([]))
        cfg = EngineConfig(cleaner_prepass_skip_keep=True, cleaner_prepass_skip_cutoff=0.9)

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=cfg, rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.skipped == 1
        assert judge_a.calls == []
        assert judge_b.calls == []
        row = await _prepass_row_for(adapter, f.id)
        assert row.action == "skipped_keep"
        assert row.actor == "engine"
        assert row.matched is True
        # No review row for a skip.
        history = await adapter.list_reviews_for(target_id=f.id)
        assert history == []

    async def test_keep_clear_below_cutoff_still_goes_to_panel(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        f = _fact(content="a less certain keep")
        await adapter.upsert(f, _vec())

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={"a less certain keep": _keep_verdict("keep-clear", confidence=0.85)},
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))
        cfg = EngineConfig(cleaner_prepass_skip_keep=True, cleaner_prepass_skip_cutoff=0.9)

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=cfg, rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.skipped == 0
        assert summary.counts.keep == 1
        assert len(judge_a.calls) == 1

    async def test_shelve_clear_always_goes_to_panel_regardless_of_confidence(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        f = _fact(content="a very certain shelve")
        await adapter.upsert(f, _vec())

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                "a very certain shelve": _keep_verdict("shelve-clear", confidence=0.99)
            },
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))
        cfg = EngineConfig(cleaner_prepass_skip_keep=True, cleaner_prepass_skip_cutoff=0.9)

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=cfg, rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.skipped == 0
        assert len(judge_a.calls) == 1
        assert summary.counts.shelve == 1


class TestRelevanceSplitBand:
    async def test_prepass_keep_lean_resolves_split_keep(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        f = _fact(content="split record leaning keep")
        await adapter.upsert(f, _vec())

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                "split record leaning keep": _keep_verdict("keep-lean", confidence=0.7)
            },
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.split_keep == 1
        assert summary.contested_ids == ()
        row = await _prepass_row_for(adapter, f.id)
        assert row.action == "split_keep"
        assert row.actor == "engine"
        history = await adapter.list_reviews_for(target_id=f.id)
        assert len(history) == 1
        assert history[0].band == "keep"
        assert history[0].outcome == "assessor_resolved"
        # The record leaves the contested set (a prior decision: only a resolving band leaves it).
        assert f.id not in contested_ids(adapter, Scope("proj-a"))

    async def test_split_keep_review_append_failure_falls_back_to_contested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The resolving keep review IS the
        effect that takes the record out of the contested queue, not an optional audit row --
        losing it must never be reported as a landed split_keep."""
        adapter = _make_adapter()
        clock = _make_clock()
        f = _fact(content="split record whose resolving review fails")
        await adapter.upsert(f, _vec())

        real_append_review = adapter.append_review
        call_count = {"n": 0}

        async def _broken_once_append_review(*args: Any, **kwargs: Any) -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("review store unavailable")
            await real_append_review(*args, **kwargs)

        monkeypatch.setattr(adapter, "append_review", _broken_once_append_review)

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                "split record whose resolving review fails": _keep_verdict(
                    "keep-lean", confidence=0.7
                )
            },
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.split_keep == 0
        assert f.id in summary.contested_ids
        assert len(summary.refused) == 1
        assert summary.refused[0].id == f.id
        assert f.id in contested_ids(adapter, Scope("proj-a"))
        row = await _prepass_row_for(adapter, f.id)
        assert row.status == "failed"

    async def test_prepass_shelve_lean_switch_off_annotates_contested(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        f = _fact(content="split record leaning shelve")
        await adapter.upsert(f, _vec())

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                "split record leaning shelve": _keep_verdict("shelve-lean", confidence=0.7)
            },
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(cleaner_split_shelve=False), rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.split_annotated == 1
        assert summary.contested_ids == (f.id,)
        row = await _prepass_row_for(adapter, f.id)
        assert row.action == "split_annotated"
        history = await adapter.list_reviews_for(target_id=f.id)
        assert len(history) == 1
        assert history[0].band == "contested"

    async def test_prepass_shelve_lean_switch_on_shelves_outright(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        f = _fact(content="split record leaning shelve, switch on")
        await adapter.upsert(f, _vec())

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                "split record leaning shelve, switch on": _keep_verdict(
                    "shelve-lean", confidence=0.7
                )
            },
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(cleaner_split_shelve=True), rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.split_shelve == 1
        assert summary.contested_ids == ()
        assert f.id in summary.shelved_ids
        row = await _prepass_row_for(adapter, f.id)
        assert row.action == "split_shelve"
        assert row.actor == "engine"
        stored = await adapter.get_by_id(f.id)
        assert stored.lifecycle == "shelved"

    async def test_prepass_unavailable_on_split_is_panel_contested(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        f = _fact(content="split record with an unavailable pre-pass")
        await adapter.upsert(f, _vec())

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                "split record with an unavailable pre-pass": TimeoutError("injected")
            },
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor,
        )

        assert summary.contested_ids == (f.id,)
        row = await _prepass_row_for(adapter, f.id)
        assert row.action == "panel_contested"
        assert row.matched is None
        assert row.unavailable_reason == "timeout"


class TestNoAssessorConfiguredUnchanged:
    async def test_no_assessor_writes_no_cleaner_rows(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        f = _fact(content="ordinary record, no assessor at all")
        await adapter.upsert(f, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b,
        )

        assert summary.counts.keep == 1
        assert summary.skipped == 0
        rows = list(adapter._verdicts.values())
        assert rows == []


class TestDryRunWithAssessorWritesNoCleanerRows:
    """A simulated pass with an Assessor
    configured must leave NO storage write of any kind (this axis's own pre-existing dry-run
    contract) -- `_write_cleaner_resolved_row` had no `dry_run` gate at all, so every panel band
    (keep/shelve/unclear), the skip filter and the split-annotated fallback all persisted a real,
    unmarked `cleaner_prepass` row even under `dry_run=True`."""

    async def test_keep_shelve_unclear_and_skip_leave_no_verdict_rows(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        keep_fact = _fact(content="dry run keep record")
        shelve_fact = _fact(content="dry run shelve record")
        unclear_fact = _fact(content="dry run unclear record")
        skip_fact = _fact(content="dry run skip record")
        for f in (keep_fact, shelve_fact, unclear_fact, skip_fact):
            await adapter.upsert(f, _vec())

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                # `shelve-clear` never skips regardless of confidence (`_prepass_wants_skip`
                # only ever skips a `keep-clear` pick) -- these three always reach the panel, so
                # the PANEL's own verdicts (below) are what decides their band.
                "dry run keep record": _keep_verdict("shelve-clear"),
                "dry run shelve record": _keep_verdict("shelve-clear"),
                "dry run unclear record": _keep_verdict("shelve-clear"),
                "dry run skip record": _keep_verdict("keep-clear", confidence=0.95),
            },
        )
        # The skip filter removes `skip_fact` from the panel batch entirely -- only the other
        # three (in pool order) reach the judges below.
        response = _payload(
            [
                _v(1, "keep-clear"),
                _v(2, "shelve-clear"),
                _v(3, "keep-clear", unclear=True),
            ]
        )
        judge_a = FakeJudgeAdapter(identity="a", response=response)
        judge_b = FakeJudgeAdapter(identity="b", response=response)
        cfg = EngineConfig(cleaner_prepass_skip_keep=True, cleaner_prepass_skip_cutoff=0.9)

        summary = await run_relevance_axis(
            scope=Scope("proj-a"), since=_SINCE, adapter=adapter, clock=clock,
            lock=asyncio.Lock(), cfg=cfg, rubric=load_rubric(),
            judge_a=judge_a, judge_b=judge_b, assessor=assessor, dry_run=True,
        )

        assert summary.counts.keep == 1
        assert summary.counts.shelve == 1
        assert summary.counts.unclear == 1
        assert summary.skipped == 1
        rows = list(adapter._verdicts.values())
        assert rows == []


# ===========================================================================
# Trim on shelve: a `keep` part of a shelved record is saved as a new Fact
# ===========================================================================


def _vk(index: int, grade: str, keep: str | None = None) -> dict[str, Any]:
    entry = _v(index, grade)
    if keep is not None:
        entry["keep"] = keep
    return entry


async def _trim_run(
    adapter: MemoryAdapter,
    *,
    judge_a: Any,
    judge_b: Any,
    dry_run: bool = False,
    assessor: Any = None,
    cfg: EngineConfig | None = None,
    embedder: Any = "default",
) -> RelevanceAxisSummary:
    from tests.support.doubles import FakeEmbedder

    await adapter.create_scope(Scope("proj-a"))
    return await run_relevance_axis(
        scope=Scope("proj-a"),
        since=_SINCE,
        adapter=adapter,
        clock=_make_clock(),
        lock=asyncio.Lock(),
        cfg=cfg if cfg is not None else EngineConfig(),
        rubric=load_rubric(),
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        dry_run=dry_run,
        embedder=FakeEmbedder(dim=16) if embedder == "default" else embedder,
    )


def _panel(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[Any, Any]:
    return (
        FakeJudgeAdapter(identity="a", response=_payload(a)),
        FakeJudgeAdapter(identity="b", response=_payload(b)),
    )


async def _active_contents(adapter: MemoryAdapter) -> list[str]:
    active: frozenset[FactLifecycle] = frozenset({"active"})
    return [record.content async for record in await adapter.scan_records(lifecycle_in=active)]


class TestTrimOnShelve:
    async def test_seat_a_keep_is_saved_after_the_shelve(self) -> None:
        adapter = _make_adapter(16)
        fact = _fact(content="stale ticket note plus a durable rule")
        await adapter.upsert(fact, _vec(16))
        judge_a, judge_b = _panel(
            [_vk(1, "shelve-clear", "The durable rule.")], [_vk(1, "shelve-lean")]
        )

        summary = await _trim_run(adapter, judge_a=judge_a, judge_b=judge_b)

        assert summary.shelved_ids == (fact.id,)
        assert summary.counts.shelve == 1
        assert len(summary.trimmed) == 1
        pair = summary.trimmed[0]
        assert pair.old_id == fact.id
        assert pair.new_id is not None
        assert (await adapter.get_by_id(fact.id)).lifecycle == "shelved"
        saved = await adapter.get_by_id(pair.new_id)
        assert isinstance(saved, Fact)
        assert saved.content == "The durable rule."
        assert saved.lifecycle == "active"
        assert saved.scope == fact.scope
        assert saved.source == f"cleaner:trim:{fact.id}"

    async def test_only_seat_b_supplying_keep_is_enough(self) -> None:
        adapter = _make_adapter(16)
        fact = _fact(content="stale ticket note plus a durable rule")
        await adapter.upsert(fact, _vec(16))
        judge_a, judge_b = _panel(
            [_vk(1, "shelve-clear")], [_vk(1, "shelve-lean", "B's version of the rule.")]
        )

        summary = await _trim_run(adapter, judge_a=judge_a, judge_b=judge_b)

        assert len(summary.trimmed) == 1
        assert summary.trimmed[0].new_id is not None
        saved = await adapter.get_by_id(summary.trimmed[0].new_id)
        assert saved.content == "B's version of the rule."

    async def test_seat_a_text_wins_when_both_supply_one(self) -> None:
        adapter = _make_adapter(16)
        fact = _fact(content="stale ticket note plus a durable rule")
        await adapter.upsert(fact, _vec(16))
        judge_a, judge_b = _panel(
            [_vk(1, "shelve-clear", "A's version.")], [_vk(1, "shelve-clear", "B's version.")]
        )

        summary = await _trim_run(adapter, judge_a=judge_a, judge_b=judge_b)

        assert summary.trimmed[0].new_id is not None
        assert (await adapter.get_by_id(summary.trimmed[0].new_id)).content == "A's version."

    async def test_without_keep_nothing_new_is_written(self) -> None:
        adapter = _make_adapter(16)
        fact = _fact(content="stale ticket note")
        await adapter.upsert(fact, _vec(16))
        judge_a, judge_b = _panel([_vk(1, "shelve-clear")], [_vk(1, "shelve-clear")])

        summary = await _trim_run(adapter, judge_a=judge_a, judge_b=judge_b)

        assert summary.trimmed == ()
        assert await _active_contents(adapter) == []

    async def test_keep_already_covered_elsewhere_writes_nothing_and_the_shelve_stands(
        self,
    ) -> None:
        adapter = _make_adapter(16)
        fact = _fact(content="stale ticket note plus a durable rule")
        other = _fact(content="The durable rule.", now=_SINCE)
        await adapter.upsert(fact, _vec(16))
        await adapter.upsert(other, _vec(16, seed=-1.0))
        judge_a, judge_b = _panel(
            [_vk(1, "shelve-clear", "The durable rule.")], [_vk(1, "shelve-clear")]
        )

        summary = await _trim_run(adapter, judge_a=judge_a, judge_b=judge_b)

        assert summary.shelved_ids == (fact.id,)
        assert summary.trimmed == ()
        assert summary.refused == ()
        assert (await adapter.get_by_id(fact.id)).lifecycle == "shelved"
        assert await _active_contents(adapter) == ["The durable rule."]

    async def test_a_raising_save_reverts_the_shelve_and_lands_in_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(**_: Any) -> Any:
            raise RuntimeError("embedder exploded")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        adapter = _make_adapter(16)
        fact = _fact(content="stale ticket note plus a durable rule")
        await adapter.upsert(fact, _vec(16))
        judge_a, judge_b = _panel(
            [_vk(1, "shelve-clear", "The durable rule.")], [_vk(1, "shelve-clear")]
        )

        summary = await _trim_run(adapter, judge_a=judge_a, judge_b=judge_b)

        assert (await adapter.get_by_id(fact.id)).lifecycle == "active"
        assert summary.shelved_ids == ()
        assert summary.verdicts == ()
        assert summary.counts.shelve == 0
        assert summary.trimmed == ()
        assert [r.id for r in summary.refused] == [fact.id]
        assert summary.refused[0].reason == (
            "trim save failed: embedder exploded; shelve reverted"
        )

    async def test_assessor_resolved_split_shelve_trims_with_the_shelve_seats_keep(self) -> None:
        adapter = _make_adapter(16)
        fact = _fact(content="split record leaning shelve, with a durable part")
        await adapter.upsert(fact, _vec(16))
        shelve_lean = _keep_verdict("shelve-lean", confidence=0.7)
        assessor = EchoAssessor("unrelated", relevance_script={fact.content: shelve_lean})
        judge_a, judge_b = _panel(
            [_vk(1, "keep-clear")], [_vk(1, "shelve-clear", "The shelve seat's part.")]
        )

        summary = await _trim_run(
            adapter,
            judge_a=judge_a,
            judge_b=judge_b,
            assessor=assessor,
            cfg=EngineConfig(cleaner_split_shelve=True),
        )

        assert summary.split_shelve == 1
        assert fact.id in summary.shelved_ids
        assert len(summary.trimmed) == 1
        assert summary.trimmed[0].new_id is not None
        saved = await adapter.get_by_id(summary.trimmed[0].new_id)
        assert saved.content == "The shelve seat's part."
        assert [call[0] for call in assessor.calls] == ["relevance"]

    async def test_reverted_split_shelve_trim_is_refused_not_shelved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        adapter = _make_adapter(16)
        fact = _fact(content="split record leaning shelve, with a durable part")
        await adapter.upsert(fact, _vec(16))
        shelve_lean = _keep_verdict("shelve-lean", confidence=0.7)
        assessor = EchoAssessor("unrelated", relevance_script={fact.content: shelve_lean})
        judge_a, judge_b = _panel(
            [_vk(1, "shelve-clear", "Part.")], [_vk(1, "keep-clear")]
        )

        summary = await _trim_run(
            adapter,
            judge_a=judge_a,
            judge_b=judge_b,
            assessor=assessor,
            cfg=EngineConfig(cleaner_split_shelve=True),
        )

        assert (await adapter.get_by_id(fact.id)).lifecycle == "active"
        assert summary.shelved_ids == ()
        assert summary.split_shelve == 0
        assert summary.counts.shelve == 0
        assert [r.id for r in summary.refused] == [fact.id]

    async def test_dry_run_counts_the_trim_and_writes_nothing(self) -> None:
        adapter = _make_adapter(16)
        fact = _fact(content="stale ticket note plus a durable rule")
        await adapter.upsert(fact, _vec(16))
        judge_a, judge_b = _panel(
            [_vk(1, "shelve-clear", "The durable rule.")], [_vk(1, "shelve-clear")]
        )

        summary = await _trim_run(adapter, judge_a=judge_a, judge_b=judge_b, dry_run=True)

        assert summary.shelved_ids == (fact.id,)
        assert len(summary.trimmed) == 1
        assert summary.trimmed[0].new_id is None
        assert (await adapter.get_by_id(fact.id)).lifecycle == "active"
        assert await _active_contents(adapter) == [fact.content]

    async def test_without_an_embedder_the_shelve_stands_and_nothing_is_saved(self) -> None:
        adapter = _make_adapter(16)
        fact = _fact(content="stale ticket note plus a durable rule")
        await adapter.upsert(fact, _vec(16))
        judge_a, judge_b = _panel(
            [_vk(1, "shelve-clear", "The durable rule.")], [_vk(1, "shelve-clear")]
        )

        summary = await _trim_run(adapter, judge_a=judge_a, judge_b=judge_b, embedder=None)

        assert summary.shelved_ids == (fact.id,)
        assert summary.trimmed == ()
        assert await _active_contents(adapter) == []

    async def test_the_save_never_runs_while_the_write_lock_is_held(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = _real_save_internal
        locked_at_call: list[bool] = []

        async def spy(**kwargs: Any) -> Any:
            locked_at_call.append(kwargs["lock"].locked())
            return await real(**kwargs)

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", spy)
        adapter = _make_adapter(16)
        fact = _fact(content="stale ticket note plus a durable rule")
        await adapter.upsert(fact, _vec(16))
        judge_a, judge_b = _panel(
            [_vk(1, "shelve-clear", "The durable rule.")], [_vk(1, "shelve-clear")]
        )

        await _trim_run(adapter, judge_a=judge_a, judge_b=judge_b)

        assert locked_at_call == [False]


class _MarkerJudge:
    def __init__(self, *, identity: str, grades: dict[str, str]) -> None:
        self._identity = identity
        self._grades = grades

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        for marker, grade in self._grades.items():
            if marker in prompt:
                return _payload([_v(1, grade)])
        raise AssertionError("prompt carried no known record marker")


class TestShelveCountIncludesEveryStandingShelve:
    async def test_counts_and_the_pool_identity_reconcile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tabularium.engine.cleaner_writes as cleaner_writes_mod

        adapter = _make_adapter()
        facts = {
            marker: _fact(content=f"{marker} durable note")
            for marker in (
                "rec-shelve-1",
                "rec-shelve-2",
                "rec-split-shelve",
                "rec-split-refused",
                "rec-keep",
                "rec-unclear",
            )
        }
        for fact in facts.values():
            await adapter.upsert(fact, _vec())
        refused_id = facts["rec-split-refused"].id

        async def refuse_one(**kwargs: Any) -> Any:
            if kwargs["id"] == refused_id:
                raise ToolError("refused")
            return await _real_set_shelved(**kwargs)

        monkeypatch.setattr(cleaner_writes_mod, "set_shelved", refuse_one)
        shelve_lean = _keep_verdict("shelve-lean", confidence=0.7)
        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                facts["rec-split-shelve"].content: shelve_lean,
                facts["rec-split-refused"].content: shelve_lean,
            },
        )
        grades_a = {
            "rec-shelve-1": "shelve-clear",
            "rec-shelve-2": "shelve-clear",
            "rec-split-shelve": "shelve-clear",
            "rec-split-refused": "shelve-clear",
            "rec-keep": "keep-clear",
            "rec-unclear": "keep-lean",
        }
        grades_b = {**grades_a, "rec-split-shelve": "keep-clear", "rec-split-refused": "keep-clear"}
        judge_a = _MarkerJudge(identity="a", grades=grades_a)
        judge_b = _MarkerJudge(identity="b", grades=grades_b)

        summary = await run_relevance_axis(
            scope=Scope("proj-a"),
            since=_SINCE,
            adapter=adapter,
            clock=_make_clock(),
            lock=asyncio.Lock(),
            cfg=EngineConfig(cleaner_split_shelve=True),
            rubric=_small_batch_rubric(batch_cap=1),
            judge_a=judge_a,
            judge_b=judge_b,
            assessor=assessor,
        )

        assert summary.split_shelve == 1
        assert summary.counts.shelve == 3
        assert summary.counts.shelve == len(summary.shelved_ids)
        assert [r.id for r in summary.refused] == [refused_id]
        refused_not_contested = {r.id for r in summary.refused} - set(summary.contested_ids)
        reconciled = (
            len(summary.excluded_decisions)
            + len(summary.excluded_global)
            + len(summary.excluded_already_shelved)
            + summary.counts.shelve
            + summary.counts.keep
            + summary.counts.unclear
            + summary.counts.contested
            + summary.skipped
            + summary.split_keep
            + sum(len(failure.record_ids) for failure in summary.failed_batches)
            + len(refused_not_contested)
        )
        assert reconciled == summary.pool_size


class TestGlobalPassPool:
    def test_global_records_are_candidates_in_a_global_pass(self) -> None:
        fact = _fact(content="owner-written", scope="global")

        in_global = _select_relevance_pool([fact], pass_scope=Scope("global"))
        in_project = _select_relevance_pool([fact], pass_scope=Scope("proj-a"))

        assert [f.id for f in in_global.candidates] == [fact.id]
        assert in_global.excluded_global == ()
        assert in_project.excluded_global == (fact.id,)

    async def test_global_pass_shelves_a_global_fact_on_unanimous_shelve(self) -> None:
        adapter = _make_adapter()
        fact = _fact(content="stale owner note", scope="global")
        await adapter.upsert(fact, _vec())

        summary = await run_relevance_axis(
            scope=Scope("global"),
            since=_SINCE,
            adapter=adapter,
            clock=_make_clock(),
            lock=asyncio.Lock(),
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
            cfg=EngineConfig(),
        )

        assert summary.shelved_ids == (fact.id,)
        assert (await adapter.get_by_id(fact.id)).lifecycle == "shelved"
