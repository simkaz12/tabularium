"""The Cleaner's revisit axis — the judged half (2026-08-12 cleaner-revisit-pass plan, scope 4).

Hermetic lane only — ``MemoryAdapter`` + ``FakeJudgeAdapter``, no network, no credential, no real
``claude``/``codex`` process and no real litellm provider call anywhere in this file (same
convention as ``test_relevance_axis.py``/``test_falsity_axis.py``).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.domain import Fact, FactLifecycle, Scope, new_id
from tabularium.engine.axis_common import (
    BandCounts,
    ContestedVerdict,
    RecordVerdict,
)
from tabularium.engine.facts import save_internal as _real_save_internal
from tabularium.engine.revisit import RevisitAxisSummary, run_revisit_axis
from tabularium.engine.rubric import Rubric, RubricParameters, load_rubric
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import RecordReview, ReinforcementEvent
from tests.support.doubles import FakeJudgeAdapter, TestClock
from tests.support.queue import latest_judgment

_NOW = datetime(2025, 4, 1, tzinfo=UTC)
_BORN = _NOW - timedelta(days=100)  # well past revisit_min_age_days (30)


def _make_adapter(dim: int = 4) -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic", dim=dim)


def _make_clock(now: datetime = _NOW) -> TestClock:
    return TestClock(now)


def _vec(dim: int = 4, *, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _fact(
    *,
    content: str = "A durable gotcha worth remembering about the deployment pipeline.",
    scope: str = "proj-a",
    lifecycle: str = "active",
    created_at: datetime = _BORN,
) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity="valid",
        created_at=created_at,
        updated_at=created_at,
    )


def _v(index: int, grade: str, *, unclear: bool = False, why: str = "because") -> dict[str, Any]:
    return {"index": index, "grade": grade, "unclear": unclear, "why": why}


def _payload(items: list[dict[str, Any]]) -> str:
    return json.dumps({"verdicts": items})


def _small_revisit_batch_rubric(revisit_batch_cap: int) -> Rubric:
    """The shipped rubric with ``revisit_batch_cap`` overridden — forces multiple batches without
    needing dozens of fixture records (same device the sibling axis test files use for their own
    ``batch_cap``)."""
    base = load_rubric()
    return Rubric(
        version=base.version,
        parameters=RubricParameters(
            k=base.parameters.k,
            similarity_floor=base.parameters.similarity_floor,
            batch_cap=base.parameters.batch_cap,
            unclear_ceiling=base.parameters.unclear_ceiling,
            pool_limit=base.parameters.pool_limit,
            report_render_cap=base.parameters.report_render_cap,
            revisit_batch_cap=revisit_batch_cap,
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


async def _give_evidence(
    adapter: MemoryAdapter, fact: Fact, *, at: datetime = _BORN + timedelta(days=1)
) -> None:
    """Give ``fact`` a lifecycle event since its own birth — the cheapest way to satisfy
    ``select_revisit_pool``'s "no evidence since birth" exclusion so a fixture record reaches a
    judge batch (see ``engine.revisit_pool``'s module docstring)."""
    await adapter.append_event(
        ReinforcementEvent(target_id=fact.id, at=at, source="dedup", matched_from_id=new_id())
    )


async def _run(
    adapter: MemoryAdapter,
    *,
    clock: TestClock,
    judge_a: Any,
    judge_b: Any,
    rubric: Rubric | None = None,
    dry_run: bool = False,
    cfg: Any = None,
    assessor: Any = None,
    embedder: Any = None,
) -> Any:
    from tabularium.config import EngineConfig

    return await run_revisit_axis(
        scope=Scope("proj-a"),
        adapter=adapter,
        clock=clock,
        lock=asyncio.Lock(),
        cfg=cfg if cfg is not None else EngineConfig(),
        rubric=rubric if rubric is not None else load_rubric(),
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        dry_run=dry_run,
        embedder=embedder,
    )


class TestUnanimousBands:
    async def test_unanimous_shelve_writes_and_appends_a_judged_review(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b)

        assert summary.shelved_ids == (fact.id,)
        assert summary.counts == BandCounts(shelve=1, keep=0, unclear=0, contested=0)
        assert summary.verdicts == (RecordVerdict(id=fact.id, band="shelve", why="because"),)
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "shelved"

        assert len(adapter._record_reviews) == 1
        review = adapter._record_reviews[0]
        assert review.target_id == fact.id
        assert review.outcome == "judged"
        assert review.band == "shelve"
        assert review.rubric_version == load_rubric().version

    async def test_unanimous_keep_writes_nothing_but_still_appends_a_review(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-lean")]))

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b)

        assert summary.shelved_ids == ()
        assert summary.counts == BandCounts(shelve=0, keep=1, unclear=0, contested=0)
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"

        assert len(adapter._record_reviews) == 1
        review = adapter._record_reviews[0]
        assert review.target_id == fact.id
        assert review.outcome == "judged"
        assert review.band == "keep"

        # `record_judgments` is a `"contested"`-only writer (M2+ slice) -- a unanimous keep
        # never appends one.
        assert latest_judgment(adapter, fact.id) is None


class TestSplitBand:
    async def test_split_appends_a_contested_review_and_writes_nothing(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "shelve-lean", why="perishable now")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "keep-clear", why="durable lesson")])
        )

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b)

        assert summary.shelved_ids == ()
        assert summary.contested_ids == (fact.id,)
        assert summary.counts == BandCounts(shelve=0, keep=0, unclear=0, contested=1)
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
        assert stored.lifecycle == "active"

        assert len(adapter._record_reviews) == 1
        review = adapter._record_reviews[0]
        assert review.outcome == "judged"
        assert review.band == "contested"

        # Judge identity/grade/`why` land in the SEPARATE `record_judgments` table (M2+ "who
        # judged what and why" slice, owner correction 2026-08-16) -- never as fields on
        # `RecordReview` itself. Falsifiability: before this slice, nothing wrote here at all --
        # this would come back `None`.
        judgment = latest_judgment(adapter, fact.id)
        assert judgment is not None
        assert judgment.judge_a_identity == "a"
        assert judgment.judge_a_grade == "shelve-lean"
        assert judgment.judge_a_why == "perishable now"
        assert judgment.judge_b_identity == "b"
        assert judgment.judge_b_grade == "keep-clear"
        assert judgment.judge_b_why == "durable lesson"
        assert judgment.paired_with is None  # single-record dispute -- nothing to pair against


class TestProtectedRecordNeverShelved:
    async def test_protected_record_is_never_shelved_even_on_unanimous_shelve_verdict(
        self,
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(lifecycle="active")
        await adapter.upsert(fact, _vec())
        # Un-shelved in the past -- structurally protected (decision 9): active lifecycle + a
        # completed shelve-then-unshelve round trip (two `shelved`-source events --
        # a single stray event from a failed shelve attempt must not be enough).
        await adapter.append_event(
            ReinforcementEvent(
                target_id=fact.id, at=_BORN + timedelta(days=1), source="shelved", reason="shelve"
            )
        )
        await adapter.append_event(
            ReinforcementEvent(
                target_id=fact.id, at=_BORN + timedelta(days=2), source="shelved", reason="undo"
            )
        )

        # A judge that would happily shelve it -- must never be consulted for this record at all,
        # since selection removes it before any batch is ever built.
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")]))

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b)

        assert summary.protected_ids == (fact.id,)
        assert summary.shelved_ids == ()
        assert judge_a.calls == []
        assert judge_b.calls == []
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"

        assert len(adapter._record_reviews) == 1
        review = adapter._record_reviews[0]
        assert review.target_id == fact.id
        assert review.outcome == "protected"
        assert review.band is None


class _FailTransitionOnceAdapter(MemoryAdapter):
    """A ``MemoryAdapter`` whose ``transition`` raises once for one specific id/target lifecycle
    pair -- models a real storage-layer failure landing strictly AFTER ``set_shelved``'s own
    audit event append has already succeeded (``set_shelved`` appends its event BEFORE the
    transition call, by design, so this ordering can
    leave a stray event behind on an otherwise-untouched record)."""

    def __init__(self, *, fail_for: object, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fail_for = fail_for
        self._failed_once = False

    async def transition(self, id: object, lifecycle: str, **kwargs: Any) -> None:
        if not self._failed_once and id == self._fail_for and lifecycle == "shelved":
            self._failed_once = True
            raise RuntimeError("simulated storage failure mid-transition")
        await super().transition(id, lifecycle, **kwargs)  # type: ignore[arg-type]


class TestAFailedShelveAttemptDoesNotPermanentlyProtect:
    async def test_a_single_failed_shelve_attempt_does_not_protect_the_record(self) -> None:
        """A single stray `shelved`-source
        event left behind by a failed shelve attempt (event appended, transition raised) must not
        make `is_protected` wrongly, permanently hide an otherwise-ordinary active record from
        the panel forever."""
        adapter = _FailTransitionOnceAdapter(fail_for=None, embedding_model="fake/nomic", dim=4)
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)
        adapter._fail_for = fact.id

        from tabularium.engine.dispositions import set_shelved

        raised = False
        try:
            await set_shelved(
                id=fact.id,
                shelved=True,
                reason="attempted shelve",
                adapter=adapter,
                clock=clock,
                lock=asyncio.Lock(),
            )
        except RuntimeError as exc:
            raised = True
            assert "simulated storage failure" in str(exc)
        assert raised

        # The audit event landed (per `set_shelved`'s own documented "event before transition"
        # ordering) but the lifecycle transition never did.
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"
        events = await adapter.query_events(target_id=fact.id, source="shelved")
        assert len(events) == 1

        # The next revisit pass must still be able to see this record -- not silently and
        # permanently protect it because of one stray event from a failed write.
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))
        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b)

        assert fact.id not in summary.protected_ids
        assert judge_a.calls != []  # actually judged, not silently skipped


class TestBatchFailureLeavesStoreUntouched:
    async def test_bad_index_leaves_the_store_untouched_and_reports_a_failed_batch(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        # Out-of-range index -> JudgeInvalidOutput -> the whole batch fails, nothing written.
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(99, "shelve-clear")]))

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b)

        assert summary.shelved_ids == ()
        assert len(summary.failed_batches) == 1
        failure = summary.failed_batches[0]
        assert failure.record_ids == (fact.id,)
        assert failure.judge_a_outcome == "ok"
        assert failure.judge_b_outcome == "invalid_output"

        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"
        # Nothing in a failed batch is written -- no partial application, no review row either.
        assert adapter._record_reviews == []

    async def test_failure_in_one_batch_does_not_affect_another(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        first = _fact(content="A durable gotcha worth remembering, batch one.")
        second = _fact(content="A durable gotcha worth remembering, batch two.")
        await adapter.upsert(first, _vec(seed=1.0))
        await adapter.upsert(second, _vec(seed=2.0))
        await _give_evidence(adapter, first)
        await _give_evidence(adapter, second)

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

        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[_payload([_v(1, "shelve-clear")]), _payload([_v(1, "shelve-clear")])],
        )
        judge_b = _SequencedJudgeAdapter(
            identity="b",
            responses=[_payload([_v(1, "shelve-lean")]), "not json at all"],
        )

        summary = await _run(
            adapter,
            clock=clock,
            judge_a=judge_a,
            judge_b=judge_b,
            rubric=_small_revisit_batch_rubric(revisit_batch_cap=1),
        )

        assert len(summary.failed_batches) == 1
        stored_first = await adapter.get_by_id(first.id)
        stored_second = await adapter.get_by_id(second.id)
        # Order between `first`/`second` through selection is deterministic (oldest-reviewed/
        # never-reviewed first, tie-broken by created_at) but both were born at the identical
        # `_BORN` instant here -- assert on the SET of outcomes instead of position.
        lifecycles = {stored_first.lifecycle, stored_second.lifecycle}
        assert lifecycles == {"active", "shelved"}
        assert len(adapter._record_reviews) == 1  # only the succeeding batch's record


class TestDryRun:
    async def test_dry_run_writes_nothing_at_all_including_no_review_rows(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b, dry_run=True)

        # Judging still happened -- the summary reports what WOULD have been written.
        assert summary.shelved_ids == (fact.id,)

        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"  # untouched
        assert adapter._record_reviews == []  # no review rows, live or otherwise

    async def test_dry_run_appends_no_protected_or_skipped_review_rows_either(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        protected = _fact(lifecycle="active")
        await adapter.upsert(protected, _vec(seed=1.0))
        # A completed round trip (two events) -- see
        # `TestProtectedRecordNeverShelved` for why a single event is no longer enough.
        await adapter.append_event(
            ReinforcementEvent(
                target_id=protected.id,
                at=_BORN + timedelta(days=1),
                source="shelved",
                reason="shelve",
            )
        )
        await adapter.append_event(
            ReinforcementEvent(
                target_id=protected.id,
                at=_BORN + timedelta(days=2),
                source="shelved",
                reason="undo",
            )
        )
        skipped = _fact(
            content="Nothing has ever happened to this record since it was written.",
            created_at=_BORN,
        )
        await adapter.upsert(skipped, _vec(seed=2.0))

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([]))

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b, dry_run=True)

        assert summary.protected_ids == (protected.id,)
        assert summary.skipped_no_evidence_ids == (skipped.id,)
        assert adapter._record_reviews == []


class _FlakyAppendReviewAdapter(MemoryAdapter):
    """A ``MemoryAdapter`` whose ``append_review`` raises for one specific target id, once --
    everything else (including ``set_shelved``'s own writes) behaves normally. Models the exact
    failure ordering where the lifecycle
    transition landed BEFORE the audit row's append was ever attempted."""

    def __init__(self, *, fail_for: object, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fail_for = fail_for
        self._failed_once = False

    async def append_review(self, review: RecordReview) -> None:
        if not self._failed_once and review.target_id == self._fail_for:
            self._failed_once = True
            raise RuntimeError("simulated review-append failure")
        await super().append_review(review)


class TestReviewAppendFailureIsIsolatedFromTheWrite:
    async def test_a_shelve_survives_in_the_summary_even_when_its_review_append_fails(
        self,
    ) -> None:
        """`set_shelved` lands BEFORE
        `append_review` is attempted -- if the append raises, the shelved id must still be
        observable in the returned summary (and the store), never silently dropped by an
        exception unwinding the whole axis run."""
        adapter = _FlakyAppendReviewAdapter(fail_for=None, embedding_model="fake/nomic", dim=4)
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)
        adapter._fail_for = fact.id

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b)

        # The write landed and is reflected in the summary despite the review-append failure.
        assert summary.shelved_ids == (fact.id,)
        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "shelved"

        # The failure is surfaced, not silently swallowed -- and no phantom review row exists.
        assert len(summary.review_append_failures) == 1
        failure = summary.review_append_failures[0]
        assert failure.target_id == fact.id
        assert failure.outcome == "judged"
        assert "simulated review-append failure" in failure.error
        assert adapter._record_reviews == []

        # Verification still runs and still confirms the write (independent of the review row).
        assert summary.verification_failures == ()


class _InvalidatingJudgeAdapter:
    """Returns a canned response, but FIRST invalidates a target record directly on the adapter --
    simulates a concurrent writer changing the record's eligibility WHILE this batch's judge call
    is in flight, between selection (``list_due_for_review``) and the eventual write. Same device
    as ``test_pass_orchestration.py``'s ``_InjectingJudgeAdapter``."""

    def __init__(
        self, *, identity: str, response: str, adapter: MemoryAdapter, target_id: Any
    ) -> None:
        self._identity = identity
        self._response = response
        self._adapter = adapter
        self._target_id = target_id
        self.calls: list[str] = []
        self._triggered = False

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._triggered:
            self._triggered = True
            await self._adapter.set_validity(self._target_id, False, "concurrent invalidate")
        return self._response


class TestRaceWindowGuardBeforeShelveWrite:
    async def test_a_record_invalidated_mid_pass_is_refused_not_shelved(self) -> None:
        """A fresh re-read immediately before
        the ``set_shelved`` call must catch a record that became ineligible (invalidated,
        superseded, buried) between selection and the write -- `set_shelved` itself only
        re-checks lifecycle, never validity/superseded_by."""
        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        judge_a = _InvalidatingJudgeAdapter(
            identity="a",
            response=_payload([_v(1, "shelve-clear")]),
            adapter=adapter,
            target_id=fact.id,
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b)

        # Not shelved -- refused instead, because the fresh re-read found it already invalid.
        assert summary.shelved_ids == ()
        assert len(summary.refused) == 1
        assert summary.refused[0].id == fact.id

        stored = await adapter.get_by_id(fact.id)
        assert stored.lifecycle == "active"  # unchanged -- never transitioned to shelved
        assert stored.validity == "invalid"  # the concurrent invalidate landed

        # Still gets a review row (decision 15's spirit) -- the panel DID judge it "shelve".
        assert len(adapter._record_reviews) == 1
        review = adapter._record_reviews[0]
        assert review.outcome == "judged"
        assert review.band == "shelve"


class TestAdaptiveIntervalUsesRealHistory:
    async def test_compounding_across_two_passes_reflects_the_first_passs_keep(self) -> None:
        """A second pass's
        ``next_review_due_at`` must compound off the FIRST pass's real, persisted verdict --
        fetched via ``adapter.list_reviews_for``, not computed off an empty history."""
        adapter = _make_adapter()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)
        params = load_rubric().parameters

        judge_a1 = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b1 = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))
        summary1 = await _run(adapter, clock=_make_clock(_NOW), judge_a=judge_a1, judge_b=judge_b1)

        assert summary1.counts.keep == 1
        assert len(adapter._record_reviews) == 1
        review1 = adapter._record_reviews[0]
        assert review1.band == "keep"
        base_due = _NOW + timedelta(days=params.revisit_interval_days)
        assert review1.next_review_due_at == base_due

        # Advance well past the first due date -- the record is due again.
        second_now = base_due + timedelta(days=1)
        judge_a2 = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b2 = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))
        summary2 = await _run(
            adapter, clock=_make_clock(second_now), judge_a=judge_a2, judge_b=judge_b2
        )

        assert summary2.counts.keep == 1
        assert len(adapter._record_reviews) == 2
        review2 = sorted(adapter._record_reviews, key=lambda r: r.reviewed_at)[-1]
        expected_due2 = second_now + timedelta(
            days=params.revisit_interval_days * params.revisit_stable_multiplier
        )
        assert review2.next_review_due_at == expected_due2
        # The compounded interval is strictly longer than the base -- the stable streak mattered.
        assert (review2.next_review_due_at - second_now) > timedelta(
            days=params.revisit_interval_days
        )


# ---------------------------------------------------------------------------
# A3 cleaner-integration piece, D10: reuse the relevance axis's own cleaner_prepass verdict;
# make one fresh call, logged cleaner_split, only when no reusable row exists.
# ---------------------------------------------------------------------------


async def _prepass_row(adapter: MemoryAdapter, fact_id: Any) -> Any:
    rows = await adapter.verdicts_for_record(fact_id)
    matches = [r for r in rows if r.caller == "cleaner_prepass"]
    assert len(matches) == 1
    return matches[0]


class TestRevisitSplitReuse:
    async def test_reuse_makes_no_second_relevance_call(self) -> None:
        from tabularium.engine.verdicts import build_cleaner_resolved_row
        from tests.support.fake_assessor import EchoAssessor

        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        # Seed a pre-existing cleaner_prepass row (as if the relevance axis had already run a
        # pre-pass on this record, picking keep-lean) -- D10's own reuse target.
        prepass_verdict = _keep_lean_verdict()
        existing_row = build_cleaner_resolved_row(
            caller="cleaner_prepass", candidate_id=fact.id, verdict=prepass_verdict,
            action="panel_keep", actor="panel", matched=True, now=clock.now(),
        )
        await adapter.append_verdicts([existing_row])

        assessor = EchoAssessor("unrelated")  # no script needed -- must never be called
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await _run(
            adapter, clock=clock, judge_a=judge_a, judge_b=judge_b, assessor=assessor
        )

        assert assessor.relevance_calls == []
        assert summary.contested_ids == ()
        row = await _prepass_row(adapter, fact.id)
        assert row.action == "split_keep"
        history = await adapter.list_reviews_for(target_id=fact.id)
        assert len(history) == 1
        assert history[0].band == "keep"

    async def test_no_reusable_row_makes_exactly_one_fresh_call_logged_cleaner_split(self) -> None:
        from tests.support.fake_assessor import EchoAssessor

        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="a record that predates A3, no prior cleaner_prepass row")
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                "a record that predates A3, no prior cleaner_prepass row": _keep_lean_verdict()
            },
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await _run(
            adapter, clock=clock, judge_a=judge_a, judge_b=judge_b, assessor=assessor
        )

        assert len(assessor.relevance_calls) == 1
        assert summary.contested_ids == ()
        rows = await adapter.verdicts_for_record(fact.id)
        split_rows = [r for r in rows if r.caller == "cleaner_split"]
        assert len(split_rows) == 1
        assert split_rows[0].action == "split_keep"

    async def test_restamp_updates_action_on_the_existing_row(self) -> None:
        from tabularium.config import EngineConfig
        from tabularium.engine.verdicts import build_cleaner_resolved_row
        from tests.support.fake_assessor import EchoAssessor

        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact()
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        prepass_verdict = _shelve_lean_verdict()
        existing_row = build_cleaner_resolved_row(
            caller="cleaner_prepass", candidate_id=fact.id, verdict=prepass_verdict,
            action="panel_shelve", actor="panel", matched=True, now=clock.now(),
        )
        await adapter.append_verdicts([existing_row])

        assessor = EchoAssessor("unrelated")
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))
        cfg = EngineConfig(cleaner_split_shelve=True)

        summary = await _run(
            adapter, clock=clock, judge_a=judge_a, judge_b=judge_b, assessor=assessor, cfg=cfg
        )

        assert assessor.relevance_calls == []
        assert summary.contested_ids == ()
        assert fact.id in summary.shelved_ids
        row = await _prepass_row(adapter, fact.id)
        assert row.id == existing_row.id  # the SAME row, restamped -- not a second one
        assert row.action == "split_shelve"
        assert row.status == "acted"


class TestSplitKeepCount:
    @pytest.mark.parametrize("dry_run", [False, True])
    async def test_split_keep_resolved_by_the_assessor_is_counted_and_nowhere_else(
        self, dry_run: bool
    ) -> None:
        from tests.support.fake_assessor import EchoAssessor

        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(content="a record the judges split on, kept by the assessor")
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        assessor = EchoAssessor(
            "unrelated", relevance_script={fact.content: _keep_lean_verdict()}
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        summary = await _run(
            adapter,
            clock=clock,
            judge_a=judge_a,
            judge_b=judge_b,
            assessor=assessor,
            dry_run=dry_run,
        )

        assert summary.split_keep == 1
        assert summary.counts == BandCounts()
        assert summary.shelved_ids == ()
        assert summary.unclear_ids == ()
        assert summary.contested_ids == ()
        assert summary.verdicts == ()
        assert summary.refused == ()
        assert summary.failed_batches == ()
        assert _reconciled(summary) == summary.pool_size == 1


class _KeyedJudge:
    """Answers by which record marker the prompt carries, so a multi-batch pass is deterministic
    whatever order selection puts the records in. ``invalidate`` names ``(adapter, id, marker)``:
    the judge invalidates that record just before answering the batch that carries the marker."""

    def __init__(
        self,
        *,
        identity: str,
        grades: dict[str, str],
        broken: tuple[str, ...] = (),
        invalidate: tuple[MemoryAdapter, Any, str] | None = None,
        keeps: dict[str, str] | None = None,
    ) -> None:
        self._identity = identity
        self._grades = grades
        self._broken = broken
        self._invalidate = invalidate
        self._keeps = keeps or {}

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        if self._invalidate is not None and self._invalidate[2] in prompt:
            await self._invalidate[0].set_validity(self._invalidate[1], False, "concurrent")
        for marker in self._broken:
            if marker in prompt:
                return "not json at all"
        for marker, grade in self._grades.items():
            if marker in prompt:
                entry = _v(1, grade)
                if marker in self._keeps:
                    entry["keep"] = self._keeps[marker]
                return _payload([entry])
        raise AssertionError("prompt carried no known record marker")


def _reconciled(summary: RevisitAxisSummary) -> int:
    return (
        len(summary.excluded_global)
        + len(summary.excluded_non_fact)
        + len(summary.protected_ids)
        + len(summary.skipped_no_evidence_ids)
        + len(summary.skipped_too_large_ids)
        + sum(len(failure.record_ids) for failure in summary.failed_batches)
        + summary.counts.shelve
        + summary.counts.keep
        + summary.counts.unclear
        + summary.counts.contested
        + summary.split_keep
        + len(summary.refused)
    )


class TestPoolReconciliation:
    @pytest.mark.parametrize("dry_run", [False, True])
    async def test_every_pool_record_lands_in_exactly_one_counted_term(
        self, dry_run: bool
    ) -> None:
        from tabularium.config import EngineConfig
        from tests.support.fake_assessor import EchoAssessor

        adapter = _make_adapter()
        clock = _make_clock()
        markers = (
            "rec-keep",
            "rec-shelve",
            "rec-splitkeep",
            "rec-splitshelve",
            "rec-contested",
            "rec-failed",
            "rec-refused",
        )
        facts = {marker: _fact(content=f"{marker} durable gotcha") for marker in markers}
        for seed, fact in enumerate(facts.values(), start=1):
            await adapter.upsert(fact, _vec(seed=float(seed)))
            await _give_evidence(adapter, fact)
        no_evidence = _fact(content="rec-silent nothing happened since birth")
        await adapter.upsert(no_evidence, _vec(seed=9.0))

        split = ("shelve-clear", "keep-clear")
        grades_a = {
            "rec-keep": "keep-clear",
            "rec-shelve": "shelve-clear",
            "rec-splitkeep": split[0],
            "rec-splitshelve": split[0],
            "rec-contested": split[0],
            "rec-failed": "shelve-clear",
            "rec-refused": "shelve-clear",
        }
        grades_b = {
            "rec-keep": "keep-clear",
            "rec-shelve": "shelve-lean",
            "rec-splitkeep": split[1],
            "rec-splitshelve": split[1],
            "rec-contested": split[1],
            "rec-failed": "shelve-lean",
            "rec-refused": "shelve-lean",
        }
        judge_a = _KeyedJudge(
            identity="a",
            grades=grades_a,
            invalidate=(adapter, facts["rec-refused"].id, "rec-refused"),
        )
        judge_b = _KeyedJudge(identity="b", grades=grades_b, broken=("rec-failed",))
        assessor = EchoAssessor(
            "unrelated",
            relevance_script={
                facts["rec-splitkeep"].content: _keep_lean_verdict(),
                facts["rec-splitshelve"].content: _shelve_lean_verdict(),
                facts["rec-contested"].content: RuntimeError("assessor unavailable"),
            },
        )

        summary = await _run(
            adapter,
            clock=clock,
            judge_a=judge_a,
            judge_b=judge_b,
            assessor=assessor,
            cfg=EngineConfig(cleaner_split_shelve=True),
            rubric=_small_revisit_batch_rubric(revisit_batch_cap=1),
            dry_run=dry_run,
        )

        assert summary.pool_size == 8
        assert summary.skipped_no_evidence_ids == (no_evidence.id,)
        assert summary.counts.keep == 1
        assert summary.split_keep == 1
        assert summary.counts.contested == 1
        assert summary.contested_ids == (facts["rec-contested"].id,)
        assert summary.counts.unclear == 0
        assert [f.record_ids for f in summary.failed_batches] == [(facts["rec-failed"].id,)]
        assert facts["rec-splitshelve"].id in summary.shelved_ids
        if dry_run:
            assert summary.counts.shelve == 3
            assert summary.refused == ()
        else:
            assert summary.counts.shelve == 2
            assert [r.id for r in summary.refused] == [facts["rec-refused"].id]
            assert facts["rec-refused"].id not in summary.shelved_ids
        assert _reconciled(summary) == summary.pool_size


def _keep_lean_verdict() -> Any:
    from tabularium.engine.assessor import Trace, Verdict, compute_confidence

    odds = {"shelve-clear": 0.1, "shelve-lean": 0.1, "keep-lean": 0.7, "keep-clear": 0.1}
    return Verdict(
        method="relevance", pick="keep-lean", odds=odds, confidence=compute_confidence(odds),
        trace=Trace(
            model="echo", wording_hash="echo", latency_ms=0,
            input_tokens=None, output_tokens=None, request_id=None,
        ),
    )


def _shelve_lean_verdict() -> Any:
    from tabularium.engine.assessor import Trace, Verdict, compute_confidence

    odds = {"shelve-clear": 0.1, "shelve-lean": 0.7, "keep-lean": 0.1, "keep-clear": 0.1}
    return Verdict(
        method="relevance", pick="shelve-lean", odds=odds, confidence=compute_confidence(odds),
        trace=Trace(
            model="echo", wording_hash="echo", latency_ms=0,
            input_tokens=None, output_tokens=None, request_id=None,
        ),
    )


class _HeadingCountingJudge:
    def __init__(self, identity: str) -> None:
        self._identity = identity
        self.calls: list[str] = []

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        import re

        self.calls.append(prompt)
        batch_text = prompt[: prompt.index("# Reply format")]
        count = len(re.findall(r"^## \d+\.$", batch_text, flags=re.MULTILINE))
        return _payload([_v(i, "keep-clear") for i in range(1, count + 1)])


class TestBoundedPrompts:
    async def test_three_million_chars_of_history_is_packed_under_the_ceiling(self) -> None:
        from tabularium.domain import Session
        from tabularium.engine.judge import PROMPT_CHAR_CEILING

        adapter = _make_adapter()
        facts: list[Fact] = []
        total_history = 0
        for i in range(12):
            born = _BORN + timedelta(days=i)
            fact = _fact(content=f"durable gotcha{i:02d} marker{i:02d}", created_at=born)
            facts.append(fact)
            await adapter.upsert(fact, _vec())
            birth_text = "b" * 100_000
            await adapter.append_session(
                Session(
                    id=new_id(),
                    scope="proj-a",
                    started_at=born - timedelta(hours=1),
                    ended_at=born + timedelta(hours=1),
                    content=birth_text,
                )
            )
            total_history += len(birth_text)
            for n in range(5):
                later_text = f"durable gotcha{i:02d} marker{i:02d} " + "l" * 30_000
                await adapter.append_session(
                    Session(
                        id=new_id(),
                        scope="proj-a",
                        started_at=born + timedelta(days=20 + n),
                        ended_at=born + timedelta(days=20 + n, hours=1),
                        content=later_text,
                    )
                )
                total_history += len(later_text)
        assert total_history >= 2_900_000

        judge_a = _HeadingCountingJudge("a")
        judge_b = _HeadingCountingJudge("b")
        summary = await _run(adapter, clock=_make_clock(), judge_a=judge_a, judge_b=judge_b)

        assert judge_a.calls
        for prompt in (*judge_a.calls, *judge_b.calls):
            assert len(prompt) <= PROMPT_CHAR_CEILING
        assert summary.failed_batches == ()
        assert summary.counts.keep == 12
        for fact in facts:
            reviews = await adapter.list_reviews_for(target_id=fact.id)
            assert [r.outcome for r in reviews] == ["judged"]

    async def test_a_record_too_large_alone_gets_a_typed_skip_row_and_leaves_the_pool(
        self,
    ) -> None:
        from tabularium.engine.judge import PROMPT_CHAR_CEILING

        adapter = _make_adapter()
        clock = _make_clock()
        big = _fact(content="durable gotcha99 marker99 " + "q" * (PROMPT_CHAR_CEILING + 100))
        normal = _fact(content="a perfectly ordinary record about the deployment pipeline")
        for fact in (big, normal):
            await adapter.upsert(fact, _vec())
            await _give_evidence(adapter, fact)

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))
        rubric = _small_revisit_batch_rubric(12)

        summary = await _run(adapter, clock=clock, judge_a=judge_a, judge_b=judge_b, rubric=rubric)

        assert summary.failed_batches == ()
        assert summary.skipped_too_large_ids == (big.id,)
        assert summary.counts.keep == 1
        big_reviews = await adapter.list_reviews_for(target_id=big.id)
        assert [(r.outcome, r.band) for r in big_reviews] == [("skipped_too_large", None)]
        normal_reviews = await adapter.list_reviews_for(target_id=normal.id)
        assert [r.outcome for r in normal_reviews] == ["judged"]
        assert len(judge_a.calls) == 1
        assert len(judge_b.calls) == 1
        assert "q" * 1_000 not in judge_a.calls[0]

        due = await adapter.list_due_for_review(
            scope=Scope("proj-a"),
            now=_NOW,
            created_before=_NOW - timedelta(days=rubric.parameters.revisit_min_age_days),
            limit=100,
        )
        assert big.id not in {record.id for record in due}


# ===========================================================================
# Trim on shelve
# ===========================================================================


def _vk(index: int, grade: str, keep: str | None = None) -> dict[str, Any]:
    entry = _v(index, grade)
    if keep is not None:
        entry["keep"] = keep
    return entry


def _embedder() -> Any:
    from tests.support.doubles import FakeEmbedder

    return FakeEmbedder(dim=16)


async def _seed(adapter: MemoryAdapter, content: str, *, seed: float = 1.0) -> Fact:
    await adapter.create_scope(Scope("proj-a"))
    fact = _fact(content=content)
    await adapter.upsert(fact, _vec(16, seed=seed))
    await _give_evidence(adapter, fact)
    return fact


async def _active_contents(adapter: MemoryAdapter) -> list[str]:
    active: frozenset[FactLifecycle] = frozenset({"active"})
    return [record.content async for record in await adapter.scan_records(lifecycle_in=active)]


class TestTrimOnShelve:
    async def test_seat_a_keep_is_saved_after_the_shelve(self) -> None:
        adapter = _make_adapter(16)
        fact = await _seed(adapter, "old rambling note plus a durable rule")
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_vk(1, "shelve-clear", "The durable rule.")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_vk(1, "shelve-lean")]))

        summary = await _run(
            adapter, clock=_make_clock(), judge_a=judge_a, judge_b=judge_b, embedder=_embedder()
        )

        assert summary.shelved_ids == (fact.id,)
        assert summary.counts.shelve == 1
        assert len(summary.trimmed) == 1
        assert summary.trimmed[0].old_id == fact.id
        new_id = summary.trimmed[0].new_id
        assert new_id is not None
        saved = await adapter.get_by_id(new_id)
        assert saved.content == "The durable rule."
        assert saved.lifecycle == "active"
        assert saved.source == f"cleaner:trim:{fact.id}"
        assert (await adapter.get_by_id(fact.id)).lifecycle == "shelved"

    async def test_only_seat_b_supplying_keep_is_enough(self) -> None:
        adapter = _make_adapter(16)
        await _seed(adapter, "old rambling note plus a durable rule")
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_vk(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_vk(1, "shelve-lean", "B's version.")])
        )

        summary = await _run(
            adapter, clock=_make_clock(), judge_a=judge_a, judge_b=judge_b, embedder=_embedder()
        )

        assert summary.trimmed[0].new_id is not None
        assert (await adapter.get_by_id(summary.trimmed[0].new_id)).content == "B's version."

    async def test_keep_already_covered_elsewhere_writes_nothing_and_the_shelve_stands(
        self,
    ) -> None:
        adapter = _make_adapter(16)
        fact = await _seed(adapter, "old rambling note plus a durable rule")
        covering = _fact(content="The durable rule.", created_at=_NOW)
        await adapter.upsert(covering, _vec(16, seed=-1.0))
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_vk(1, "shelve-clear", "The durable rule.")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_vk(1, "shelve-clear")]))
        summary = await _run(
            adapter, clock=_make_clock(), judge_a=judge_a, judge_b=judge_b, embedder=_embedder()
        )

        assert fact.id in summary.shelved_ids
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
        fact = await _seed(adapter, "old rambling note plus a durable rule")
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_vk(1, "shelve-clear", "The durable rule.")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_vk(1, "shelve-clear")]))

        summary = await _run(
            adapter, clock=_make_clock(), judge_a=judge_a, judge_b=judge_b, embedder=_embedder()
        )

        assert (await adapter.get_by_id(fact.id)).lifecycle == "active"
        assert summary.shelved_ids == ()
        assert summary.counts.shelve == 0
        assert summary.trimmed == ()
        assert [r.id for r in summary.refused] == [fact.id]
        assert summary.refused[0].reason == (
            "trim save failed: embedder exploded; shelve reverted"
        )

    async def test_a_reverted_trim_appends_no_review_and_the_record_stays_due(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        adapter = _make_adapter(16)
        fact = await _seed(adapter, "old rambling note plus a durable rule")
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_vk(1, "shelve-clear", "The durable rule.")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_vk(1, "shelve-clear")]))
        clock = _make_clock()

        summary = await _run(
            adapter, clock=clock, judge_a=judge_a, judge_b=judge_b, embedder=_embedder()
        )

        assert [r.id for r in summary.refused] == [fact.id]
        assert adapter._record_reviews == []
        due = await adapter.list_due_for_review(
            scope=Scope("proj-a"), now=clock.now(), created_before=clock.now(), limit=10
        )
        assert [record.id for record in due] == [fact.id]

    @pytest.mark.parametrize("lifecycle", ["active", "archived"])
    async def test_a_reverted_trim_restores_the_lifecycle_and_leaves_no_protection(
        self, lifecycle: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tabularium.engine.revisit_pool import is_protected

        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        adapter = _make_adapter(16)
        await adapter.create_scope(Scope("proj-a"))
        fact = _fact(content="old rambling note plus a durable rule", lifecycle=lifecycle)
        await adapter.upsert(fact, _vec(16, seed=1.0))
        await _give_evidence(adapter, fact)
        clock = _make_clock()

        for _round in range(2):
            judge_a = FakeJudgeAdapter(
                identity="a", response=_payload([_vk(1, "shelve-clear", "The durable rule.")])
            )
            judge_b = FakeJudgeAdapter(identity="b", response=_payload([_vk(1, "shelve-clear")]))
            summary = await _run(
                adapter, clock=clock, judge_a=judge_a, judge_b=judge_b, embedder=_embedder()
            )

            assert [r.id for r in summary.refused] == [fact.id]
            assert summary.protected_ids == ()

        stored = await adapter.get_by_id(fact.id)
        assert isinstance(stored, Fact)
        assert stored.lifecycle == lifecycle
        events = await adapter.query_events(target_id=fact.id)
        assert sum(1 for e in events if e.source == "shelve_reverted") == 2
        assert not is_protected(stored, events)
        due = await adapter.list_due_for_review(
            scope=Scope("proj-a"), now=clock.now(), created_before=clock.now(), limit=10
        )
        assert [record.id for record in due] == [fact.id]

    async def test_an_owner_round_trip_still_protects_the_record(self) -> None:
        from tabularium.engine.dispositions import set_shelved
        from tabularium.engine.revisit_pool import is_protected

        adapter = _make_adapter(16)
        fact = await _seed(adapter, "a record the owner shelved and un-shelved")
        clock = _make_clock()
        lock = asyncio.Lock()
        for shelved in (True, False):
            await set_shelved(
                id=fact.id,
                shelved=shelved,
                reason="owner",
                adapter=adapter,
                clock=clock,
                lock=lock,
            )

        stored = await adapter.get_by_id(fact.id)
        assert isinstance(stored, Fact)
        events = await adapter.query_events(target_id=fact.id)
        assert is_protected(stored, events)

    async def test_assessor_resolved_split_shelve_trims_with_the_shelve_seats_keep(self) -> None:
        from tabularium.config import EngineConfig
        from tests.support.fake_assessor import EchoAssessor

        adapter = _make_adapter(16)
        fact = await _seed(adapter, "split record the assessor shelves, with a durable part")
        assessor = EchoAssessor(
            "unrelated", relevance_script={fact.content: _shelve_lean_verdict()}
        )
        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_vk(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_vk(1, "shelve-clear", "The shelve seat's part.")])
        )

        summary = await _run(
            adapter,
            clock=_make_clock(),
            judge_a=judge_a,
            judge_b=judge_b,
            assessor=assessor,
            cfg=EngineConfig(cleaner_split_shelve=True),
            embedder=_embedder(),
        )

        assert summary.shelved_ids == (fact.id,)
        assert summary.counts.shelve == 1
        assert len(summary.trimmed) == 1
        assert summary.trimmed[0].new_id is not None
        saved = await adapter.get_by_id(summary.trimmed[0].new_id)
        assert saved.content == "The shelve seat's part."
        assert [call[0] for call in assessor.calls] == ["relevance"]

    async def test_dry_run_counts_the_trim_and_writes_nothing(self) -> None:
        adapter = _make_adapter(16)
        fact = await _seed(adapter, "old rambling note plus a durable rule")
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_vk(1, "shelve-clear", "The durable rule.")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_vk(1, "shelve-clear")]))

        summary = await _run(
            adapter,
            clock=_make_clock(),
            judge_a=judge_a,
            judge_b=judge_b,
            dry_run=True,
            embedder=_embedder(),
        )

        assert summary.shelved_ids == (fact.id,)
        assert len(summary.trimmed) == 1
        assert summary.trimmed[0].new_id is None
        assert await _active_contents(adapter) == [fact.content]

    async def test_the_save_never_runs_while_the_write_lock_is_held(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        locked_at_call: list[bool] = []

        async def spy(**kwargs: Any) -> Any:
            locked_at_call.append(kwargs["lock"].locked())
            return await _real_save_internal(**kwargs)

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", spy)
        adapter = _make_adapter(16)
        await _seed(adapter, "old rambling note plus a durable rule")
        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_vk(1, "shelve-clear", "The durable rule.")])
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_vk(1, "shelve-clear")]))

        await _run(
            adapter, clock=_make_clock(), judge_a=judge_a, judge_b=judge_b, embedder=_embedder()
        )

        assert locked_at_call == [False]


class TestPoolReconciliationWithTrim:
    @pytest.mark.parametrize("dry_run", [False, True])
    async def test_a_trim_and_a_reverted_trim_keep_the_identity_whole(
        self, dry_run: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def flaky(**kwargs: Any) -> Any:
            if kwargs["content"] == "the part that cannot be saved":
                raise RuntimeError("no disk")
            return await _real_save_internal(**kwargs)

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", flaky)
        adapter = _make_adapter(16)
        markers = ("rec-keep", "rec-plain", "rec-trim", "rec-revert")
        facts = {}
        for seed, marker in enumerate(markers, start=1):
            facts[marker] = await _seed(adapter, f"{marker} durable gotcha", seed=float(seed))
        grades_a = {
            "rec-keep": "keep-clear",
            "rec-plain": "shelve-clear",
            "rec-trim": "shelve-clear",
            "rec-revert": "shelve-clear",
        }
        grades_b = {**grades_a, "rec-keep": "keep-lean"}
        keeps = {
            "rec-trim": "the part worth keeping",
            "rec-revert": "the part that cannot be saved",
        }
        judge_a = _KeyedJudge(identity="a", grades=grades_a, keeps=keeps)
        judge_b = _KeyedJudge(identity="b", grades=grades_b)

        summary = await _run(
            adapter,
            clock=_make_clock(),
            judge_a=judge_a,
            judge_b=judge_b,
            rubric=_small_revisit_batch_rubric(revisit_batch_cap=1),
            dry_run=dry_run,
            embedder=_embedder(),
        )

        assert summary.pool_size == 4
        assert _reconciled(summary) == summary.pool_size
        assert summary.counts.keep == 1
        if dry_run:
            assert summary.counts.shelve == 3
            assert len(summary.trimmed) == 2
            assert summary.refused == ()
        else:
            assert summary.counts.shelve == 2
            assert [t.old_id for t in summary.trimmed] == [facts["rec-trim"].id]
            assert [r.id for r in summary.refused] == [facts["rec-revert"].id]
            assert (await adapter.get_by_id(facts["rec-revert"].id)).lifecycle == "active"
            assert len(summary.trimmed) <= summary.counts.shelve


class TestGlobalPass:
    async def test_global_pass_has_a_global_due_pool(self) -> None:
        from tabularium.config import EngineConfig

        adapter = _make_adapter()
        clock = _make_clock()
        fact = _fact(scope="global")
        await adapter.upsert(fact, _vec())
        await _give_evidence(adapter, fact)

        summary = await run_revisit_axis(
            scope=Scope("global"),
            adapter=adapter,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
        )

        assert summary.shelved_ids == (fact.id,)
