"""The Cleaner's pass report.

Hermetic lane only (``MemoryAdapter``/``FakeJudgeAdapter``, no network, no credential — same
convention as ``test_pass_orchestration.py``), except ``TestReportSurvivesLanceDBReopen``, which
exercises the real ``LanceDBAdapter`` against a ``tmp_path`` store (no live-Ollama/network call
either — a ``FakeEmbedder`` throughout) specifically to prove the "survives a reopen" acceptance
criterion against production storage, not just the in-memory reference.

Falsifiability (Verify step 1 of the cleaner plan — targets PRODUCTION code in
``src/tabularium/engine/report.py``):

Remove the cap on ``render_context_summary``'s rendering (change ``shown = items[:cap]`` to
``shown = items``) and re-run ``TestBoundedRendering::test_two_hundred_unclear_ids_do_not_produce_
two_hundred_lines`` — it goes red because ``len(summary.owner_attention)`` comes back as 200 instead
of being capped at ``cap + 1`` (the ``cap`` entries shown plus one overflow marker). The command and
its captured red output are recorded verbatim in the task verdict, not duplicated here (matching the
sibling axis test files' own convention).

Falsifiability (Verify step 1 of the 2026-08-11 cleaner-telemetry plan — targets
PRODUCTION code in ``src/tabularium/engine/report.py``):

Point ``load_context_summary`` back at ``adapter.get_cleaner_state(scope=scope)``/``state.report``
(the pre-existing read) and re-run
``TestLoadContextSummary::test_second_pass_changes_what_context_renders`` — it goes red because
that test deliberately runs two outcomes (``"initialized"`` then ``"empty_pool"``) that NEVER
write ``CleanerState.report`` at all (``build_report`` is only ever invoked for
``"completed"``/``"blocked"`` — see ``build_report``'s own gate), so under the old read path
``state.report`` stays ``None`` across both passes and ``load_context_summary`` returns ``None``
both times — the first assertion (``first_render is not None``) fails outright. A test built on
two ``"completed"`` passes instead would NOT go red under this break, by coincidence: the
"one write, not two" composition writes the identical report string into both
``cleaner_state.report`` and the pass-history row's ``payload``, so the old read path would still
appear to "change" between passes even though it is reading the wrong table — this is exactly why
the falsifiability target uses the two outcomes that expose the difference, not the two that paper
over it. The command and its captured red output are recorded verbatim in the task verdict, not
duplicated here.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.engine.axis_common import (
    BandCounts,
    BatchFailure,
    ContestedVerdict,
    RefusedVerdict,
)
from tabularium.engine.clean import FALLBACK_ERROR_EXCERPT_CAP, CleanSummary, clean
from tabularium.engine.core import Tabularium
from tabularium.engine.falsity import (
    FalsityAxisSummary,
    FalsityBandCounts,
    FalsityBatchFailure,
    FalsityContestedPair,
    FalsityRefusedPair,
    FalsityReportEntry,
)
from tabularium.engine.judge import JudgeCallRecord
from tabularium.engine.relevance import RelevanceAxisSummary
from tabularium.engine.report import (
    _CONTESTED_WHY_RENDER_CAP,
    CleanerBatchFailureReport,
    CleanerReport,
    FalsityReportSection,
    RelevanceReportSection,
    RevisitReportSection,
    _capped_persisted_error,
    _owner_attention_items,
    _render_batch_failure_line,
    _truncate_error_tail,
    _truncate_why,
    build_report,
    load_context_summary,
    record_pass_failure,
    render_context_summary,
    render_pass_record,
    run_clean_pass,
)
from tabularium.engine.rubric import load_rubric
from tabularium.results import CleanerContextSummary
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import CleanerPassRecord, CleanerState, ReinforcementEvent
from tests.support.doubles import FakeEmbedder, FakeJudgeAdapter, TestClock

_DIM = 16
_T0 = datetime(2025, 1, 1, tzinfo=UTC)
_T1 = datetime(2025, 1, 2, tzinfo=UTC)
_NOW = datetime(2025, 1, 10, tzinfo=UTC)


def _make_adapter(dim: int = _DIM) -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic", dim=dim)


def _make_clock(now: datetime = _NOW) -> TestClock:
    return TestClock(now)


async def _require_scope_always_ok(scope: Scope) -> None:
    del scope


def _vec(dim: int = _DIM, *, seed: float = 1.0) -> list[float]:
    return [seed] * dim


def _fact(*, content: str, scope: str = "proj-a", created_at: datetime = _T1) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle="active",
        validity="valid",
        created_at=created_at,
        updated_at=created_at,
    )


def _v(index: int, grade: str, *, unclear: bool = False, why: str = "because") -> dict[str, Any]:
    return {"index": index, "grade": grade, "unclear": unclear, "why": why}


def _payload(items: list[dict[str, Any]]) -> str:
    return json.dumps({"verdicts": items})


def _empty_relevance_section() -> RelevanceReportSection:
    return RelevanceReportSection(
        judge_pair=("a", "b"),
        rubric_version="v1",
        counts=BandCounts(),
        shelved_ids=(),
        unclear_ids=(),
        failures=(),
    )


def _empty_falsity_section() -> FalsityReportSection:
    return FalsityReportSection(
        judge_pair=("a", "b"),
        rubric_version="v1",
        counts=FalsityBandCounts(),
        superseded_ids=(),
        global_contradicted=(),
        decision_contradicted=(),
        failures=(),
    )


def _report(
    *,
    relevance: RelevanceReportSection | None = None,
    falsity: FalsityReportSection | None = None,
    outcome: str = "completed",
    verification_failures: tuple[Any, ...] = (),
) -> CleanerReport:
    return CleanerReport(
        scope=Scope("proj-a"),
        pass_at=_NOW,
        outcome=outcome,  # type: ignore[arg-type]
        relevance=relevance if relevance is not None else _empty_relevance_section(),
        falsity=falsity if falsity is not None else _empty_falsity_section(),
        verification_failures=verification_failures,
    )


# ===========================================================================
# build_report — CleanSummary -> CleanerReport (or None)
# ===========================================================================


class TestBuildReportGating:
    async def test_initialized_outcome_has_no_report(self) -> None:
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
        assert build_report(summary, pass_at=_NOW) is None

    async def test_empty_pool_outcome_has_no_report(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
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
        assert build_report(summary, pass_at=_NOW) is None


class TestBuildReportContent:
    async def test_completed_pass_carries_judge_pair_rubric_version_counts_and_ids(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record the panel unanimously shelves")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="claude:sonnet", response=_payload([_v(1, "shelve-clear")])
        )
        judge_b = FakeJudgeAdapter(
            identity="codex:gpt", response=_payload([_v(1, "shelve-clear")])
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
            judge_a=judge_a,
            judge_b=judge_b,
        )
        assert summary.outcome == "completed"

        report = build_report(summary, pass_at=_NOW)
        assert report is not None
        assert report.scope == Scope("proj-a")
        assert report.outcome == "completed"
        assert report.pass_at == _NOW

        # Judge pair + rubric version stamped on the band itself (acceptance criterion: "for
        # every band"), not just once at the top level.
        assert report.relevance.judge_pair == ("claude:sonnet", "codex:gpt")
        assert report.relevance.rubric_version == load_rubric().version
        assert report.falsity.judge_pair == ("claude:sonnet", "codex:gpt")
        assert report.falsity.rubric_version == load_rubric().version

        assert report.relevance.counts.shelve == 1
        assert report.relevance.shelved_ids == (fact.id,)
        assert report.relevance.unclear_ids == ()
        assert report.falsity.counts == FalsityBandCounts()
        assert report.verification_failures == ()

    async def test_unclear_records_the_panel_declined_to_judge(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record neither judge can confidently grade")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "keep-lean", unclear=True)])
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
        )
        assert summary.outcome == "completed"

        report = build_report(summary, pass_at=_NOW)
        assert report is not None
        assert report.relevance.unclear_ids == (fact.id,)
        assert report.relevance.counts.unclear == 1
        # Distinguishable from a shelve/keep -- unclear is its own bucket, never folded in.
        assert report.relevance.shelved_ids == ()

    async def test_blocked_pass_carries_failed_batches_and_verification_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record whose write never lands")
        await adapter.upsert(fact, _vec())

        from tabularium.results import ReviseResult

        async def phantom(**kwargs: Any) -> ReviseResult:
            return ReviseResult(
                id=kwargs["id"], outcome="revised", lifecycle="shelved", validity="valid",
                reason=kwargs["reason"],
            )

        monkeypatch.setattr("tabularium.engine.relevance.set_shelved", phantom)

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

        report = build_report(summary, pass_at=_NOW)
        assert report is not None
        assert report.outcome == "blocked"
        assert report.verification_failures == (fact.id,)


# ===========================================================================
# Revisit axis report section (2026-08-12 cleaner-revisit-pass plan, scope 5) — ISOLATED from
# `outcome`/`verification_failures` above; round-trips through the SAME persisted
# pass-history payload the birth axes already use.
# ===========================================================================


class TestRevisitReportSection:
    async def test_revisit_section_round_trips_through_the_persisted_pass_history_payload(
        self,
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        newcomer = _fact(content="a fresh durable lesson worth keeping")
        await adapter.upsert(newcomer, _vec())
        old_record = _fact(
            content="An old durable gotcha eligible for the revisit rotation.",
            created_at=_NOW - timedelta(days=100),
        )
        await adapter.upsert(old_record, [0.5] * _DIM)
        await adapter.append_event(
            ReinforcementEvent(
                target_id=old_record.id,
                at=_NOW - timedelta(days=99),
                source="dedup",
                matched_from_id=new_id(),
            )
        )

        class _RoutingJudge:
            def __init__(self, identity: str) -> None:
                self._identity = identity

            @property
            def identity(self) -> str:
                return self._identity

            def instruction_fingerprint(self) -> str:
                return "not_applicable"

            async def call(self, prompt: str) -> str:
                if "# Cleaner — revisit rubric" in prompt:
                    return _payload([_v(1, "shelve-clear", why="stale now")])
                return _payload([_v(1, "keep-clear", why="durable")])

        judge_a = _RoutingJudge("a")
        judge_b = _RoutingJudge("b")

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
        assert summary.revisit is not None
        assert summary.revisit.shelved_ids == (old_record.id,)

        rows = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=1)
        assert len(rows) == 1
        payload = json.loads(rows[0].payload)
        report = CleanerReport.model_validate(payload)
        assert isinstance(report.revisit, RevisitReportSection)
        assert report.revisit.shelved_ids == (old_record.id,)
        assert report.revisit.rubric_version == load_rubric().version
        assert report.revisit_error is None
        # ISOLATED: the birth axes' own outcome/verification_failures are untouched by revisit.
        assert report.outcome == "completed"
        assert report.verification_failures == ()

    async def test_revisit_refusal_round_trips_through_the_persisted_rich_report(self) -> None:
        """`RevisitAxisSummary.refused` must survive the RICH `CleanerReport` path (a live
        completed/blocked pass, not the fallback
        payload) -- an earlier `RevisitReportSection` copied every other revisit
        failure channel except this one."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        newcomer = _fact(content="a fresh durable lesson worth keeping")
        await adapter.upsert(newcomer, _vec())
        old_record = _fact(
            content="An old durable gotcha eligible for the revisit rotation.",
            created_at=_NOW - timedelta(days=100),
        )
        await adapter.upsert(old_record, [0.5] * _DIM)
        await adapter.append_event(
            ReinforcementEvent(
                target_id=old_record.id,
                at=_NOW - timedelta(days=99),
                source="dedup",
                matched_from_id=new_id(),
            )
        )

        class _RoutingInvalidatingJudge:
            """Shelves on revisit-rubric prompts, keeps on relevance-only ones -- and, ONLY when
            constructed with ``invalidate=True``, invalidates ``old_record`` directly on the
            adapter the FIRST time it sees a revisit-rubric prompt, simulating a concurrent
            writer racing between selection and the eventual write."""

            def __init__(self, *, identity: str, invalidate: bool) -> None:
                self._identity = identity
                self._invalidate = invalidate
                self._triggered = False

            @property
            def identity(self) -> str:
                return self._identity

            def instruction_fingerprint(self) -> str:
                return "not_applicable"

            async def call(self, prompt: str) -> str:
                if "# Cleaner — revisit rubric" in prompt:
                    if self._invalidate and not self._triggered:
                        self._triggered = True
                        await adapter.set_validity(
                            old_record.id, False, "concurrent invalidate"
                        )
                    return _payload([_v(1, "shelve-clear")])
                return _payload([_v(1, "keep-clear")])

        judge_a = _RoutingInvalidatingJudge(identity="a", invalidate=True)
        judge_b = _RoutingInvalidatingJudge(identity="b", invalidate=False)

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
        assert summary.revisit is not None
        assert summary.revisit.shelved_ids == ()
        assert len(summary.revisit.refused) == 1
        assert summary.revisit.refused[0].id == old_record.id

        rows = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=1)
        payload = json.loads(rows[0].payload)
        report = CleanerReport.model_validate(payload)
        assert report.revisit is not None
        assert len(report.revisit.refused) == 1
        assert report.revisit.refused[0].id == old_record.id

        # And it is visible in owner attention, not silently absorbed.
        summary_view = render_context_summary(report, cap=10)
        assert any("revisit shelve refused" in item for item in summary_view.owner_attention)

    def test_a_pre_this_brief_report_json_with_no_revisit_key_still_parses(self) -> None:
        """Forward-compatibility (matching every other additive field this module already
        defaults): a ``CleanerReport`` persisted before this brief carries no ``revisit``/
        ``revisit_error`` key at all -- ``model_validate`` must still succeed, filling the
        defaults."""
        old_style = {
            "scope": "proj-a",
            "pass_at": _NOW.isoformat(),
            "outcome": "completed",
            "relevance": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {},
                "shelved_ids": [],
                "unclear_ids": [],
                "failures": [],
            },
            "falsity": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {},
                "superseded_ids": [],
                "global_contradicted": [],
                "decision_contradicted": [],
                "failures": [],
            },
            "verification_failures": [],
        }
        report = CleanerReport.model_validate(old_style)
        assert report.revisit is None
        assert report.revisit_error is None

    def test_a_revisit_section_without_pool_size_or_split_keep_still_loads(self) -> None:
        old_style = {
            "scope": "proj-a",
            "pass_at": _NOW.isoformat(),
            "outcome": "completed",
            "relevance": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {},
                "shelved_ids": [],
                "unclear_ids": [],
                "failures": [],
            },
            "falsity": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {},
                "superseded_ids": [],
                "global_contradicted": [],
                "decision_contradicted": [],
                "failures": [],
            },
            "revisit": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {"shelve": 1, "keep": 2},
                "shelved_ids": [],
                "unclear_ids": [],
            },
            "verification_failures": [],
        }
        report = CleanerReport.model_validate(old_style)
        assert report.revisit is not None
        assert report.revisit.pool_size == 0
        assert report.revisit.split_keep == 0

    def test_a_pass_history_payload_without_prompt_chars_still_loads(self) -> None:
        old_call = {
            "batch_index": 0,
            "judge": "a",
            "rubric_version": "v1",
            "duration_seconds": 0.5,
            "status": "success",
            "error": None,
            "prompt_tokens": 10,
            "completion_tokens": None,
            "cost": None,
        }
        old_style = {
            "scope": "proj-a",
            "pass_at": _NOW.isoformat(),
            "outcome": "completed",
            "relevance": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {},
                "shelved_ids": [],
                "unclear_ids": [],
                "failures": [],
                "calls": [old_call],
            },
            "falsity": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {},
                "superseded_ids": [],
                "global_contradicted": [],
                "decision_contradicted": [],
                "failures": [],
            },
            "verification_failures": [],
        }

        report = CleanerReport.model_validate(old_style)

        assert report.relevance is not None
        assert report.relevance.calls[0].prompt_chars == 0


# ===========================================================================
# Contested-rationale persistence + rendering (Cleaner contested-rationale fix): the judges' `why`
# text used to be discarded once a contested band was reported -- `ContestedVerdict`/
# `FalsityContestedPair` already existed on the in-memory axis summaries, but `build_report` never
# threaded them into the persisted `*ReportSection`s, and `_owner_attention_items` rendered only a
# bare id even for the revisit axis, whose rationale WAS already persisted.
# ===========================================================================


def _contested_relevance_summary(
    *, record_id: HypomnemaId, why_a: str = "A's gate clause", why_b: str = "B's gate clause"
) -> RelevanceAxisSummary:
    return RelevanceAxisSummary(
        scope=Scope("proj-a"),
        rubric_version="v1",
        judge_pair=("claude:sonnet", "codex:gpt"),
        pool_size=1,
        excluded_decisions=(),
        excluded_global=(),
        excluded_already_shelved=(),
        counts=BandCounts(contested=1),
        shelved_ids=(),
        unclear_ids=(),
        contested_ids=(record_id,),
        failed_batches=(),
        calls=(),
        contested_verdicts=(
            ContestedVerdict(
                id=record_id,
                judge_a_identity="claude:sonnet",
                judge_a_grade="shelve-lean",
                judge_a_why=why_a,
                judge_b_identity="codex:gpt",
                judge_b_grade="keep-clear",
                judge_b_why=why_b,
            ),
        ),
    )


def _contested_falsity_summary(
    *,
    newcomer_id: HypomnemaId,
    candidate_id: HypomnemaId,
    why_a: str = "what changed per A",
    why_b: str = "what keeps them standing per B",
) -> FalsityAxisSummary:
    return FalsityAxisSummary(
        scope=Scope("proj-a"),
        rubric_version="v1",
        judge_pair=("claude:sonnet", "codex:gpt"),
        newcomer_pool_size=1,
        excluded_decisions=(),
        excluded_global=(),
        excluded_shelved_this_pass=(),
        counts=FalsityBandCounts(contested=1),
        edges=(),
        decision_candidates=(),
        global_candidates=(),
        hot_or_ember_candidates=(),
        failed_batches=(),
        calls=(),
        contested_pairs=(
            FalsityContestedPair(
                newcomer_id=newcomer_id,
                candidate_id=candidate_id,
                similarity=0.91,
                judge_a_identity="claude:sonnet",
                judge_a_verdict="replace",
                judge_a_why=why_a,
                judge_b_identity="codex:gpt",
                judge_b_verdict="coexist",
                judge_b_why=why_b,
            ),
        ),
    )


class TestContestedRationaleSurvivesBuildReportRoundTrip:
    """Falsifiability target (a): remove ``contested_verdicts=relevance.contested_verdicts`` (and
    its falsity sibling) from ``build_report`` in ``src/tabularium/engine/report.py`` and re-run
    this test -- it goes red because ``round_tripped.relevance.contested_verdicts`` (and the
    falsity one) come back empty, so indexing ``[0]`` raises ``IndexError`` rather than merely a
    missing attribute -- the test asserts the actual ``why`` CONTENT survived, not just that the
    field exists."""

    def test_relevance_and_falsity_why_survive_build_report_then_json_round_trip(self) -> None:
        record_id = new_id()
        newcomer_id = new_id()
        candidate_id = new_id()

        summary = CleanSummary(
            scope=Scope("proj-a"),
            outcome="completed",
            watermark_before=_T0,
            watermark_after=_NOW,
            pool_size=1,
            relevance=_contested_relevance_summary(
                record_id=record_id, why_a="RELEVANCE-A-WHY", why_b="RELEVANCE-B-WHY"
            ),
            falsity=_contested_falsity_summary(
                newcomer_id=newcomer_id,
                candidate_id=candidate_id,
                why_a="FALSITY-A-WHY",
                why_b="FALSITY-B-WHY",
            ),
            verification_failures=(),
            reason="completed",
        )

        report = build_report(summary, pass_at=_NOW)
        assert report is not None

        # Straight off build_report, before any JSON round trip.
        assert report.relevance.contested_verdicts[0].judge_a_why == "RELEVANCE-A-WHY"
        assert report.relevance.contested_verdicts[0].judge_b_why == "RELEVANCE-B-WHY"
        assert report.falsity.contested_verdicts[0].judge_a_why == "FALSITY-A-WHY"
        assert report.falsity.contested_verdicts[0].judge_b_why == "FALSITY-B-WHY"

        # The exact persisted shape: JSON string in, JSON string out (what a pass-history row's
        # `payload` column actually stores -- see the module docstring).
        round_tripped = CleanerReport.model_validate_json(report.model_dump_json())
        assert round_tripped.relevance.contested_verdicts[0].id == record_id
        assert round_tripped.relevance.contested_verdicts[0].judge_a_why == "RELEVANCE-A-WHY"
        assert round_tripped.relevance.contested_verdicts[0].judge_b_why == "RELEVANCE-B-WHY"
        assert round_tripped.falsity.contested_verdicts[0].newcomer_id == newcomer_id
        assert round_tripped.falsity.contested_verdicts[0].candidate_id == candidate_id
        assert round_tripped.falsity.contested_verdicts[0].judge_a_why == "FALSITY-A-WHY"
        assert round_tripped.falsity.contested_verdicts[0].judge_b_why == "FALSITY-B-WHY"


class TestContestedRationaleRenderedForAllThreeAxes:
    """Falsifiability target (b): revert the rich rendering in ``_owner_attention_items`` for any
    one of the three axes back to the bare ``f"...: {record_id}"`` line and re-run this test -- the
    corresponding marker assertion below goes red because the marker text genuinely is not in the
    rendered output (not because an attribute is missing -- the report objects below already carry
    the rationale either way)."""

    def test_owner_attention_lines_carry_both_judges_why_for_relevance_falsity_and_revisit(
        self,
    ) -> None:
        relevance_id = new_id()
        newcomer_id = new_id()
        candidate_id = new_id()
        revisit_id = new_id()

        relevance_section = RelevanceReportSection(
            judge_pair=("a", "b"),
            rubric_version="v1",
            counts=BandCounts(contested=1),
            shelved_ids=(),
            unclear_ids=(),
            contested_ids=(relevance_id,),
            contested_verdicts=(
                ContestedVerdict(
                    id=relevance_id,
                    judge_a_identity="claude:sonnet",
                    judge_a_grade="shelve-lean",
                    judge_a_why="RELEVANCE-WHY-A-MARKER",
                    judge_b_identity="codex:gpt",
                    judge_b_grade="keep-clear",
                    judge_b_why="RELEVANCE-WHY-B-MARKER",
                ),
            ),
            failures=(),
        )
        falsity_section = FalsityReportSection(
            judge_pair=("a", "b"),
            rubric_version="v1",
            counts=FalsityBandCounts(contested=1),
            superseded_ids=(),
            global_contradicted=(),
            decision_contradicted=(),
            contested=((newcomer_id, candidate_id),),
            contested_verdicts=(
                FalsityContestedPair(
                    newcomer_id=newcomer_id,
                    candidate_id=candidate_id,
                    similarity=0.9,
                    judge_a_identity="claude:sonnet",
                    judge_a_verdict="replace",
                    judge_a_why="FALSITY-WHY-A-MARKER",
                    judge_b_identity="codex:gpt",
                    judge_b_verdict="coexist",
                    judge_b_why="FALSITY-WHY-B-MARKER",
                ),
            ),
            failures=(),
        )
        revisit_section = RevisitReportSection(
            judge_pair=("a", "b"),
            rubric_version="v1",
            counts=BandCounts(contested=1),
            shelved_ids=(),
            unclear_ids=(),
            contested_ids=(revisit_id,),
            contested_verdicts=(
                ContestedVerdict(
                    id=revisit_id,
                    judge_a_identity="claude:sonnet",
                    judge_a_grade="shelve-clear",
                    judge_a_why="REVISIT-WHY-A-MARKER",
                    judge_b_identity="codex:gpt",
                    judge_b_grade="keep-lean",
                    judge_b_why="REVISIT-WHY-B-MARKER",
                ),
            ),
        )

        report = CleanerReport(
            scope=Scope("proj-a"),
            pass_at=_NOW,
            outcome="completed",
            relevance=relevance_section,
            falsity=falsity_section,
            revisit=revisit_section,
            verification_failures=(),
        )

        summary = render_context_summary(report, cap=10)
        rendered = " ".join(summary.owner_attention)

        assert "RELEVANCE-WHY-A-MARKER" in rendered
        assert "RELEVANCE-WHY-B-MARKER" in rendered
        assert "FALSITY-WHY-A-MARKER" in rendered
        assert "FALSITY-WHY-B-MARKER" in rendered
        assert "REVISIT-WHY-A-MARKER" in rendered
        assert "REVISIT-WHY-B-MARKER" in rendered

        # Both judges' identity and grade/verdict are on the line too, not just the why.
        assert "claude:sonnet" in rendered and "codex:gpt" in rendered
        assert "shelve-lean" in rendered and "keep-clear" in rendered
        assert "replace" in rendered and "coexist" in rendered


class TestContestedWhyTruncatedAtRenderNeverAtPersistence:
    """The grounding line is budget-packed (brief decision): rendering must cap a judge's `why`,
    but persistence must not. Falsifiability: delete the ``_truncate_why`` call inside
    ``_render_contested_line`` (render the raw ``why`` instead) and re-run
    ``test_long_why_is_capped_in_the_rendered_line`` -- it goes red because the 500-character
    marker string then DOES appear verbatim in ``rendered``."""

    def test_long_why_survives_persistence_untruncated_but_is_capped_in_the_rendered_line(
        self,
    ) -> None:
        record_id = new_id()
        long_why = "Q" * 500  # far past any sane render cap
        section = RelevanceReportSection(
            judge_pair=("a", "b"),
            rubric_version="v1",
            counts=BandCounts(contested=1),
            shelved_ids=(),
            unclear_ids=(),
            contested_ids=(record_id,),
            contested_verdicts=(
                ContestedVerdict(
                    id=record_id,
                    judge_a_identity="a",
                    judge_a_grade="shelve-lean",
                    judge_a_why=long_why,
                    judge_b_identity="b",
                    judge_b_grade="keep-clear",
                    judge_b_why="short",
                ),
            ),
            failures=(),
        )

        # Persistence: the full 500-character `why` round-trips untouched.
        round_tripped = RelevanceReportSection.model_validate_json(section.model_dump_json())
        assert round_tripped.contested_verdicts[0].judge_a_why == long_why

        # Rendering: the SAME report's owner-attention line never carries the full 500 characters.
        report = _report(relevance=section)
        summary = render_context_summary(report, cap=10)
        rendered = " ".join(summary.owner_attention)
        assert long_why not in rendered
        assert "Q" * 30 in rendered  # a meaningful prefix still made it through

    def test_embedded_newline_in_why_does_not_break_the_one_line_render_contract(self) -> None:
        """A `why` with an embedded newline (or
        any other whitespace run) must not turn into multiple physical lines inside one
        owner-attention item -- the brief's own "one line per contested record" constraint.
        Falsifiability: drop the ``" ".join(why.split())`` normalization from ``_truncate_why``
        and re-run this test -- it goes red because ``"\\n" in matching[0]`` becomes true."""
        record_id = new_id()
        multiline_why = "first clause\nsecond clause\tthird clause"
        section = RelevanceReportSection(
            judge_pair=("a", "b"),
            rubric_version="v1",
            counts=BandCounts(contested=1),
            shelved_ids=(),
            unclear_ids=(),
            contested_ids=(record_id,),
            contested_verdicts=(
                ContestedVerdict(
                    id=record_id,
                    judge_a_identity="a",
                    judge_a_grade="shelve-lean",
                    judge_a_why=multiline_why,
                    judge_b_identity="b",
                    judge_b_grade="keep-clear",
                    judge_b_why="short",
                ),
            ),
            failures=(),
        )
        report = _report(relevance=section)
        summary = render_context_summary(report, cap=10)

        matching = [item for item in summary.owner_attention if str(record_id) in item]
        assert len(matching) == 1
        assert "\n" not in matching[0]
        assert "\t" not in matching[0]
        assert "first clause second clause third clause" in matching[0]

        # Persistence is untouched by the render-time normalization -- the raw newline survives.
        round_tripped = RelevanceReportSection.model_validate_json(section.model_dump_json())
        assert round_tripped.contested_verdicts[0].judge_a_why == multiline_why

    def test_rendered_why_including_its_ellipsis_never_exceeds_the_cap(self) -> None:
        """The pre-fix ``why[:cap] + "..."``
        could render up to ``cap + 3`` characters -- the ellipsis must be WITHIN the cap, not
        appended after it. Falsifiability: revert ``_truncate_why`` to slice at ``cap`` (not
        ``cap - 3``) before appending ``"..."`` -- ``len(...) == 160`` then fails with 163."""
        long_why = "R" * 500
        truncated = _truncate_why(long_why, cap=160)
        assert len(truncated) == 160
        assert truncated.endswith("...")

        # A `why` at or under the cap is returned verbatim (post whitespace-normalization) --
        # no truncation, no ellipsis, when there is nothing to cut.
        short_why = "a short reason"
        assert _truncate_why(short_why, cap=160) == short_why


class TestOldContestedPayloadsWithoutRationaleStillParse:
    """Falsifiability target (c): remove the ``= ()`` default from either new
    ``contested_verdicts`` field declaration in ``src/tabularium/engine/report.py`` and re-run this
    test -- ``CleanerReport.model_validate`` then raises ``pydantic.ValidationError`` on the missing
    required field instead of defaulting it, and this test goes red on that exception rather than
    on any assertion below."""

    def test_relevance_and_falsity_sections_missing_the_contested_verdicts_key_still_parse(
        self,
    ) -> None:
        relevance_contested_id = str(new_id())
        newcomer_id = str(new_id())
        candidate_id = str(new_id())
        old_json = {
            "scope": "proj-a",
            "pass_at": _NOW.isoformat(),
            "outcome": "completed",
            "relevance": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {"contested": 1},
                "shelved_ids": [],
                "unclear_ids": [],
                "contested_ids": [relevance_contested_id],
                "failures": [],
                # no "contested_verdicts" key at all -- the pre-fix persisted shape.
            },
            "falsity": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {"contested": 1},
                "superseded_ids": [],
                "global_contradicted": [],
                "decision_contradicted": [],
                "contested": [[newcomer_id, candidate_id]],
                "failures": [],
                # no "contested_verdicts" key at all.
            },
            "verification_failures": [],
        }

        report = CleanerReport.model_validate(old_json)
        assert report.relevance.contested_verdicts == ()
        assert report.falsity.contested_verdicts == ()

        # And the renderer degrades gracefully to the old bare-id line rather than crashing --
        # nothing to render rationale FROM on a payload this old.
        summary = render_context_summary(report, cap=10)
        assert any(
            item == f"contested (judges disagreed, no write): {relevance_contested_id}"
            for item in summary.owner_attention
        )
        assert any(
            item
            == (
                "falsity contested (judges disagreed, no edge drawn): "
                f"{newcomer_id} vs {candidate_id}"
            )
            for item in summary.owner_attention
        )


# ===========================================================================
# A relevance-contested newcomer must not draw a falsity edge in the SAME pass: the "contested
# writes nothing"
# guarantee has a hole if the falsity axis is only told to exclude `shelved_ids` -- a newcomer the
# relevance panel disagreed on stays `lifecycle="active"` and eligible, so it could still
# unilaterally supersede an older record via the falsity axis's OWN unanimous vote. Fixed in
# `engine/clean.py` by unioning `contested_ids` into the exclusion set passed as `excluded_ids`.
# ===========================================================================


class _SequencedJudgeAdapter:
    """Returns a different canned response on each successive call, in order -- used here so the
    SAME judge instance can answer the relevance batch (first call) with a relevance-shaped
    payload and, if a regression ever lets the falsity axis reach it (second call), a
    falsity-shaped payload -- proving the fix by what DOESN'T happen (no second call at all) as
    much as by what does. Mirrors the identically-named device in ``test_relevance_axis.py`` /
    ``test_falsity_axis.py``."""

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


class _IdenticalEmbedder:
    """Every text embeds to the SAME unit vector -- forces a real, viable falsity pair (cosine
    1.0) between the newcomer and a pre-existing candidate, regardless of content. Same device
    ``test_fact_supersession.py`` uses for the identical reason."""

    dim: int = _DIM
    model: str = "identical"

    async def embed(self, text: str) -> list[float]:
        del text
        v = [0.0] * self.dim
        v[0] = 1.0
        return v


def _falsity_v(index: int, verdict: str, *, why: str = "because") -> dict[str, Any]:
    return {"index": index, "verdict": verdict, "why": why}


class TestContestedNewcomerExcludedFromFalsity:
    async def test_relevance_split_newcomer_draws_no_falsity_edge_against_older_candidate(
        self,
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        embedder = _IdenticalEmbedder()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )

        # A real, older, live candidate the newcomer below is vector-identical to (cosine 1.0,
        # well above the rubric's similarity_floor) -- born strictly before the newcomer, so it is
        # a genuine falsity-probe hit if the newcomer ever reaches the falsity axis's newcomer pool.
        candidate = _fact(content="an older record the newcomer would replace", created_at=_T0)
        await adapter.upsert(candidate, [1.0] + [0.0] * (_DIM - 1))

        newcomer = _fact(content="a newcomer the relevance panel disagrees on", created_at=_T1)
        await adapter.upsert(newcomer, [1.0] + [0.0] * (_DIM - 1))

        # Relevance: A shelves, B keeps -- a genuine split, resolves to "contested".
        # Falsity: A and B BOTH unanimously "replace" -- the verdict that WOULD draw an
        # edge if the newcomer ever reached a falsity batch; queued as each judge's SECOND
        # response, only consumed if the fix regresses.
        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[
                _payload([_v(1, "shelve-lean", why="A would shelve")]),
                json.dumps({"verdicts": [_falsity_v(1, "replace", why="A would replace")]}),
            ],
        )
        judge_b = _SequencedJudgeAdapter(
            identity="b",
            responses=[
                _payload([_v(1, "keep-clear", why="B would keep")]),
                json.dumps({"verdicts": [_falsity_v(1, "replace", why="B would replace")]}),
            ],
        )

        summary = await clean(
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
        assert summary.outcome == "completed"
        assert summary.relevance is not None
        assert summary.relevance.contested_ids == (newcomer.id,)
        assert summary.relevance.shelved_ids == ()

        # The real acceptance criterion: no falsity batch ever ran against the contested
        # newcomer -- proven both structurally (no edge, no second judge call) and by re-reading
        # the candidate's own storage state.
        assert summary.falsity is not None
        assert summary.falsity.edges == ()
        assert summary.falsity.counts.replace == 0
        assert len(judge_a.calls) == 1, "the falsity axis must never call the judge a second time"
        assert len(judge_b.calls) == 1, "the falsity axis must never call the judge a second time"

        stored_candidate = await adapter.get_by_id(candidate.id)
        assert stored_candidate.validity == "valid"
        assert stored_candidate.superseded_by is None


# ===========================================================================
# Per-call judge telemetry in the report (cleaner-telemetry change): the
# outcomes the ``judge_batch`` produces (``JudgeOutcome``), carried up through each axis's
# own summary (``RelevanceAxisSummary.calls``/``FalsityAxisSummary.calls``), reach
# ``build_report``'s output — nested in each section's own ``calls`` alongside what the report
# already carried (counts, ids, contradiction reports, batch failures).
# ===========================================================================


class TestCallTelemetryInReport:
    async def test_completed_pass_report_carries_one_call_per_seat_per_batch(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record the panel unanimously shelves, telemetry probe")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="claude:sonnet", response=_payload([_v(1, "shelve-clear")])
        )
        judge_b = FakeJudgeAdapter(
            identity="codex:gpt", response=_payload([_v(1, "shelve-clear")])
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
            judge_a=judge_a,
            judge_b=judge_b,
        )
        assert summary.outcome == "completed"

        report = build_report(summary, pass_at=_NOW)
        assert report is not None

        # One relevance batch, two seats -- exactly two call entries; falsity ran with zero
        # candidate pairs -- zero batches, an empty (never absent) list.
        assert len(report.relevance.calls) == 2
        judges = {c.judge for c in report.relevance.calls}
        assert judges == {"claude:sonnet", "codex:gpt"}
        for call in report.relevance.calls:
            assert call.status == "success"
            assert call.error is None
            assert call.completion_tokens is None
            assert call.cost is None
            assert call.prompt_tokens > 0
            assert call.rubric_version == load_rubric().version
            assert call.batch_index == 0
        assert report.falsity.calls == ()

    async def test_blocked_pass_report_carries_the_failed_call_alongside_the_survivor(
        self,
    ) -> None:
        """The acceptance criterion, at the report level: one seat fails, one succeeds, in the
        SAME batch -- both calls appear, neither dropped."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record whose panel is split between success and garbage")
        await adapter.upsert(fact, _vec())

        judge_a = FakeJudgeAdapter(
            identity="claude:sonnet", response=_payload([_v(1, "keep-clear")])
        )
        judge_b = FakeJudgeAdapter(identity="codex:gpt", response="not json at all")

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

        report = build_report(summary, pass_at=_NOW)
        assert report is not None
        assert len(report.relevance.calls) == 4
        by_judge = {c.judge: c for c in report.relevance.calls}
        assert by_judge["claude:sonnet"].status == "success"
        assert by_judge["codex:gpt"].status == "invalid_output"
        assert by_judge["codex:gpt"].error is not None

    async def test_report_round_trips_through_json_with_calls_intact(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record for the JSON round-trip probe")
        await adapter.upsert(fact, _vec())

        summary = await clean(
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
        assert summary.outcome == "completed"

        report = build_report(summary, pass_at=_NOW)
        assert report is not None
        assert len(report.relevance.calls) == 2

        round_tripped = CleanerReport.model_validate_json(report.model_dump_json())
        assert round_tripped == report
        assert round_tripped.relevance.calls == report.relevance.calls

    async def test_instruction_fingerprint_reaches_the_call_record_and_survives_json_round_trip(
        self,
    ) -> None:
        """Owner, at the report level: the value each seat's
        ``instruction_fingerprint()`` returns must reach ``JudgeCallRecord`` off a real ``clean()``
        pass, and must still be there after a full JSON round-trip through
        ``CleanerReport.model_dump_json()``/``model_validate_json()`` -- the exact serialization a
        pass-history row's ``payload`` column stores (``report.py``'s own module docstring)."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record for the instruction-fingerprint survival probe")
        await adapter.upsert(fact, _vec())

        fingerprint_a = "sha256:" + "ab" * 32
        summary = await clean(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(
                identity="a",
                response=_payload([_v(1, "keep-clear")]),
                instruction_fingerprint=fingerprint_a,
            ),
            judge_b=FakeJudgeAdapter(
                identity="b",
                response=_payload([_v(1, "keep-clear")]),
                instruction_fingerprint="absent",
            ),
        )
        assert summary.outcome == "completed"

        report = build_report(summary, pass_at=_NOW)
        assert report is not None
        by_judge = {c.judge: c for c in report.relevance.calls}
        assert by_judge["a"].instruction_fingerprint == fingerprint_a
        assert by_judge["b"].instruction_fingerprint == "absent"

        round_tripped = CleanerReport.model_validate_json(report.model_dump_json())
        rt_by_judge = {c.judge: c for c in round_tripped.relevance.calls}
        assert rt_by_judge["a"].instruction_fingerprint == fingerprint_a
        assert rt_by_judge["b"].instruction_fingerprint == "absent"

    def test_old_call_record_json_missing_instruction_fingerprint_key_still_parses(self) -> None:
        """A ``calls`` entry persisted before this task carries no ``instruction_fingerprint`` key
        at all -- it must still parse, defaulting to the ``"unrecorded"`` sentinel (same
        forward-compatibility contract ``calls: tuple[JudgeCallRecord, ...] = ()`` itself already
        proves one layer up, in ``test_old_stored_report_json_missing_calls_key_still_parses``)."""
        old_call = {
            "batch_index": 0,
            "judge": "a",
            "rubric_version": "v1",
            "duration_seconds": 0.1,
            "status": "success",
            "error": None,
            "prompt_tokens": 1,
            "completion_tokens": None,
            "cost": None,
            # no "instruction_fingerprint" key at all -- the pre-this-task shape.
        }
        record = JudgeCallRecord.model_validate(old_call)
        assert record.instruction_fingerprint == "unrecorded"

    def test_old_stored_report_json_missing_calls_key_still_parses(self) -> None:
        """A pre-existing ``CleanerReport`` JSON string, already sitting in ``cleaner_state.report``
        on a real store, carries no ``calls`` key at all -- it must still parse, with ``calls``
        defaulting to ``()`` (same forward-compatibility contract ``kind``'s own default proves)."""
        old_json = {
            "kind": "report",
            "scope": "proj-a",
            "pass_at": _NOW.isoformat(),
            "outcome": "completed",
            "relevance": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {"shelve": 0, "keep": 1, "unclear": 0},
                "shelved_ids": [],
                "unclear_ids": [],
                "failures": [],
                # no "calls" key at all -- the pre-existing shape.
            },
            "falsity": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {"replace": 0, "coexist": 0},
                "superseded_ids": [],
                "global_contradicted": [],
                "decision_contradicted": [],
                "failures": [],
            },
            "verification_failures": [],
        }
        report = CleanerReport.model_validate(old_json)
        assert report.relevance.calls == ()
        assert report.falsity.calls == ()


# ===========================================================================
# _owner_attention_items / render_context_summary — global/Decision contradiction surfaced
# ===========================================================================


class TestOwnerAttentionItems:
    def test_global_and_decision_contradicted_are_each_present_and_distinguishable(self) -> None:
        global_entry = FalsityReportEntry(
            newcomer_id=new_id(), candidate_id=new_id(), similarity=0.9,
            reason="candidate_is_global",
        )
        decision_entry = FalsityReportEntry(
            newcomer_id=new_id(), candidate_id=new_id(), similarity=0.9,
            reason="candidate_is_decision",
        )
        report = _report(
            falsity=FalsityReportSection(
                judge_pair=("a", "b"),
                rubric_version="v1",
                counts=FalsityBandCounts(),
                superseded_ids=(),
                global_contradicted=(global_entry,),
                decision_contradicted=(decision_entry,),
                failures=(),
            )
        )
        items = _owner_attention_items(report)
        assert any("global contradicted" in item for item in items)
        assert any("Decision contradicted" in item for item in items)
        # Distinguishable -- never collapsed into one bucket.
        global_items = [i for i in items if "global contradicted" in i]
        decision_items = [i for i in items if "Decision contradicted" in i]
        assert len(global_items) == 1
        assert len(decision_items) == 1
        assert global_items != decision_items

    def test_failure_kinds_are_all_surfaced(self) -> None:
        relevance_failure = CleanerBatchFailureReport(
            axis="relevance", batch_index=0, record_ids=(new_id(),),
            judge_a_outcome="unavailable", judge_b_outcome="ok", error="claude CLI exited 1",
        )
        falsity_failure = CleanerBatchFailureReport(
            axis="falsity", batch_index=0, record_ids=(new_id(), new_id()),
            judge_a_outcome="ok", judge_b_outcome="invalid_output", error="not valid JSON",
        )
        write_failure_id = new_id()
        report = _report(
            relevance=RelevanceReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=BandCounts(),
                shelved_ids=(), unclear_ids=(), failures=(relevance_failure,),
            ),
            falsity=FalsityReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=FalsityBandCounts(),
                superseded_ids=(), global_contradicted=(), decision_contradicted=(),
                failures=(falsity_failure,),
            ),
            outcome="blocked",
            verification_failures=(write_failure_id,),
        )
        items = _owner_attention_items(report)
        assert any(
            "relevance batch" in item and "claude CLI exited 1" in item for item in items
        )
        assert any("falsity batch" in item and "not valid JSON" in item for item in items)
        assert any("did not verify" in item and str(write_failure_id) in item for item in items)

    def test_refused_writes_on_both_axes_are_surfaced_with_id_and_reason(self) -> None:
        """2026-08-12 crash-loop fix: a write the store's own legality guard refused must reach
        the owner, id and reason both -- ``RelevanceReportSection.refused``/
        ``FalsityReportSection.refused``, the same mechanism ``contested_ids``/failures already
        use."""
        relevance_refusal = RefusedVerdict(
            id=new_id(), reason="only an active or archived Fact can be shelved"
        )
        falsity_refusal = FalsityRefusedPair(
            newcomer_id=new_id(), candidate_id=new_id(), reason="already superseded"
        )
        report = _report(
            relevance=RelevanceReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=BandCounts(),
                shelved_ids=(), unclear_ids=(), failures=(), refused=(relevance_refusal,),
            ),
            falsity=FalsityReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=FalsityBandCounts(),
                superseded_ids=(), global_contradicted=(), decision_contradicted=(),
                failures=(), refused=(falsity_refusal,),
            ),
        )
        items = _owner_attention_items(report)
        assert any(
            str(relevance_refusal.id) in item and relevance_refusal.reason in item
            for item in items
        )
        assert any(
            str(falsity_refusal.newcomer_id) in item
            and str(falsity_refusal.candidate_id) in item
            and falsity_refusal.reason in item
            for item in items
        )


# ===========================================================================
# Batch-failure error text is bounded on the LIVE path too (2026-08-14). Before this fix,
# `FALLBACK_ERROR_EXCERPT_CAP`/`_FALLBACK_ERROR_TOTAL_
# CAP` (`engine/clean.py`) bounded ONLY `fallback_payload` (every dry run, `"initialized"`,
# `"empty_pool"`) -- `build_report`'s rich `"completed"`/`"blocked"` path carried every
# `*BatchFailure.error` into `CleanerBatchFailureReport` verbatim, and `_owner_attention_items`
# embedded it into a session-start render line with no cap of any kind. Two independent bounds now
# apply: `_capped_persisted_error` (persistence, inside `build_report`) and `_truncate_why` reused
# via `_render_batch_failure_line` (render, inside `_owner_attention_items`) -- deliberately BOTH,
# not just one, because the render-side bound is what protects a `pass_history` row written before
# this fix shipped (persisted uncapped), not only future ones.
# ===========================================================================


class TestBatchFailureErrorBoundedOnTheLivePath:
    def test_capped_persisted_error_keeps_the_tail_not_the_head(self) -> None:
        """Falsifiability: change `_capped_persisted_error` to `error[:cap]` (head, not tail) and
        re-run this test -- it goes red because `TAIL-MARKER` (placed at the very end of the raw
        10,000-character error) no longer survives into the capped 500-character result."""
        raw_error = ("X" * (10_000 - len("TAIL-MARKER"))) + "TAIL-MARKER"
        capped = _capped_persisted_error(raw_error)
        assert len(capped) == FALLBACK_ERROR_EXCERPT_CAP
        assert capped.endswith("TAIL-MARKER")

    def test_short_error_is_returned_unchanged(self) -> None:
        short_error = "claude CLI exited 1: connection refused"
        assert _capped_persisted_error(short_error) == short_error

    def test_build_report_caps_a_long_batch_failure_error_on_every_axis(self) -> None:
        """The live persistence path (`build_report`), not just the fallback path -- constructs a
        real `CleanSummary` (relevance + falsity, both carrying an oversized `failed_batches`
        entry) and asserts what actually lands in the persisted `CleanerReport` is capped.
        Falsifiability: remove the `_capped_persisted_error(...)` wrapper from any one of
        `build_report`'s three `CleanerBatchFailureReport(...)` call sites and re-run -- that
        axis's assertion goes red because its `.error` comes back at the raw 5,000-character
        length instead of 500."""
        oversized_error = ("E" * (5_000 - len("RELEVANCE-TAIL"))) + "RELEVANCE-TAIL"
        relevance = RelevanceAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("a", "b"),
            pool_size=1,
            excluded_decisions=(),
            excluded_global=(),
            excluded_already_shelved=(),
            counts=BandCounts(),
            shelved_ids=(),
            unclear_ids=(),
            failed_batches=(
                BatchFailure(
                    batch_index=0,
                    record_ids=(new_id(),),
                    judge_a_outcome="unavailable",
                    judge_b_outcome="unavailable",
                    error=oversized_error,
                ),
            ),
            calls=(),
        )
        falsity = FalsityAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("a", "b"),
            newcomer_pool_size=1,
            excluded_decisions=(),
            excluded_global=(),
            excluded_shelved_this_pass=(),
            counts=FalsityBandCounts(),
            edges=(),
            decision_candidates=(),
            global_candidates=(),
            hot_or_ember_candidates=(),
            failed_batches=(
                FalsityBatchFailure(
                    batch_index=0,
                    pairs=((new_id(), new_id()),),
                    judge_a_outcome="unavailable",
                    judge_b_outcome="unavailable",
                    error="F" * 5_000,
                ),
            ),
            calls=(),
        )
        summary = CleanSummary(
            scope=Scope("proj-a"),
            outcome="completed",
            watermark_before=_T0,
            watermark_after=_NOW,
            pool_size=1,
            relevance=relevance,
            falsity=falsity,
            reason="completed",
        )

        report = build_report(summary, pass_at=_NOW)
        assert report is not None
        assert len(report.relevance.failures[0].error) == FALLBACK_ERROR_EXCERPT_CAP
        assert report.relevance.failures[0].error.endswith("RELEVANCE-TAIL")
        assert len(report.falsity.failures[0].error) == FALLBACK_ERROR_EXCERPT_CAP

    def test_many_large_failures_produce_a_bounded_owner_attention_render(self) -> None:
        """The reviewer's own reproduction shape (200 failures, ~4001 chars each, 826,736 bytes for
        the error text alone) -- constructed directly at the `CleanerReport` level (bypassing
        `build_report`) so this also proves the render-side bound protects a report whose
        `.error` fields are ALREADY oversized, e.g. a `pass_history` row persisted before this fix
        shipped. Asserts a REAL size bound on the rendered owner-attention text, not just that
        truncation happened somewhere. Falsifiability: revert `_render_batch_failure_line` to embed
        `failure.error` raw (no `_truncate_why` call) and re-run -- the total rendered size balloons
        past the asserted bound by roughly the same 4001/160 ratio the reviewer measured.
        """
        huge_error = "Z" * 4001
        relevance_failures = tuple(
            CleanerBatchFailureReport(
                axis="relevance",
                batch_index=i,
                record_ids=(),
                judge_a_outcome="unavailable",
                judge_b_outcome="unavailable",
                error=huge_error,
            )
            for i in range(100)
        )
        falsity_failures = tuple(
            CleanerBatchFailureReport(
                axis="falsity",
                batch_index=i,
                record_ids=(),
                judge_a_outcome="unavailable",
                judge_b_outcome="unavailable",
                error=huge_error,
            )
            for i in range(100)
        )
        report = _report(
            relevance=RelevanceReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=BandCounts(),
                shelved_ids=(), unclear_ids=(), failures=relevance_failures,
            ),
            falsity=FalsityReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=FalsityBandCounts(),
                superseded_ids=(), global_contradicted=(), decision_contradicted=(),
                failures=falsity_failures,
            ),
            outcome="blocked",
        )

        # render_context_summary's own report_render_cap bounds ITEM COUNT (a pre-existing,
        # separately-tested guarantee) -- pass a generous cap here specifically to isolate and
        # prove the PER-ITEM length bound this fix adds, not lean on the count cap to hide an
        # unbounded per-item length.
        summary = render_context_summary(report, cap=200)
        failure_items = [item for item in summary.owner_attention if "batch" in item]
        assert len(failure_items) == 200  # every failure rendered, none dropped by the count cap

        # The real size bound: were `failure.error` embedded raw, 200 items at 4001 chars each
        # would be ~800 KB (the reviewer's own reproduction, 826,736 bytes). Capped, the whole
        # rendered owner-attention payload must stay in the single-digit KB range.
        total_rendered_bytes = sum(len(item.encode()) for item in summary.owner_attention)
        assert total_rendered_bytes < 50_000, (
            f"owner_attention rendered to {total_rendered_bytes} bytes -- the per-item cap did "
            "not bound the total"
        )
        # And each individual line is itself bounded, not merely small on average.
        for item in failure_items:
            assert len(item) <= _CONTESTED_WHY_RENDER_CAP + len("relevance batch 99 failed: ")

    def test_render_batch_failure_line_truncates_via_the_tail_preserving_convention(
        self,
    ) -> None:
        """Reuses ``_CONTESTED_WHY_RENDER_CAP`` (the established owner-attention line-length bound)
        but NOT ``_truncate_why``'s head-preserving slice direction -- see
        ``_truncate_error_tail``'s own docstring
        for why a batch-failure error must keep its tail, not its head."""
        failure = CleanerBatchFailureReport(
            axis="revisit",
            batch_index=3,
            record_ids=(),
            judge_a_outcome="unavailable",
            judge_b_outcome="unavailable",
            error="W" * 1000,
        )
        line = _render_batch_failure_line(failure)
        expected_error = _truncate_error_tail(failure.error)
        assert line == f"revisit batch 3 failed: {expected_error}"
        assert line.startswith("revisit batch 3 failed: ...")
        assert "W" * 1000 not in line
        # NOT the head-preserving `_truncate_why` shape -- that would end in "..." instead.
        assert not line.endswith("...")

    def test_render_batch_failure_line_preserves_a_marker_at_the_very_end_of_an_oversized_error(
        self,
    ) -> None:
        """The persisted excerpt is already
        tail-capped by ``_capped_persisted_error`` (the live failure signal is expected at the END
        of a joined/echoed diagnostic blob), so the render must keep preserving that same tail, not
        flip to keeping the head. Falsifiability: revert ``_truncate_error_tail`` to
        ``_truncate_why``'s head-preserving shape (slice ``[:cap-3]`` + trailing ``"..."``) and
        re-run -- this goes red because ``END-OF-DIAGNOSTIC-MARKER`` no longer survives."""
        oversized_error = ("noise " * 100) + "END-OF-DIAGNOSTIC-MARKER"
        failure = CleanerBatchFailureReport(
            axis="relevance",
            batch_index=7,
            record_ids=(),
            judge_a_outcome="unavailable",
            judge_b_outcome="unavailable",
            error=oversized_error,
        )
        line = _render_batch_failure_line(failure)
        assert "END-OF-DIAGNOSTIC-MARKER" in line
        assert line.endswith("END-OF-DIAGNOSTIC-MARKER")


# ===========================================================================
# Batch-failure signal in summary_line (2026-08-13 judge-argv-limit fix). Falsifiability: delete
# the `failed_batch_count` clause `render_context_summary` appends after the `revisit_error`
# suffix and re-run these -- both go red because the two compared `summary_line`s (or the two
# owner_attention-vs-summary_line pairs) become indistinguishable again, reproducing the exact
# defect this closes: a pass where every batch's judge call died rendered identically to a
# genuine review-and-keep because band counts were the only signal in `summary_line`.
# ===========================================================================


class TestBatchFailureSignalInSummaryLine:
    def test_all_revisit_batches_failed_is_distinguishable_from_a_genuine_empty_revisit_pass(
        self,
    ) -> None:
        healthy_revisit = RevisitReportSection(
            judge_pair=("a", "b"), rubric_version="v1", counts=BandCounts(),
            shelved_ids=(), unclear_ids=(),
        )
        failure = CleanerBatchFailureReport(
            axis="revisit", batch_index=0, record_ids=(new_id(),),
            judge_a_outcome="unavailable", judge_b_outcome="unavailable",
            error="claude CLI unavailable: [Errno 7] Argument list too long",
        )
        failed_revisit = healthy_revisit.model_copy(update={"failures": (failure,)})

        healthy_report = CleanerReport(
            scope=Scope("proj-a"), pass_at=_NOW, outcome="completed",
            relevance=_empty_relevance_section(), falsity=_empty_falsity_section(),
            revisit=healthy_revisit, verification_failures=(),
        )
        failed_report = healthy_report.model_copy(update={"revisit": failed_revisit})

        healthy_summary = render_context_summary(healthy_report, cap=10)
        failed_summary = render_context_summary(failed_report, cap=10)

        # Both report "0 shelved, 0 contested" for the revisit band -- the failure must show up
        # somewhere else in summary_line, or the two are indistinguishable (the exact defect).
        assert healthy_summary.summary_line != failed_summary.summary_line
        assert "batch" not in healthy_summary.summary_line
        assert "1 batch(es) failed" in failed_summary.summary_line

    def test_batch_failure_count_spans_relevance_and_falsity_too(self) -> None:
        relevance_failure = CleanerBatchFailureReport(
            axis="relevance", batch_index=0, record_ids=(new_id(),),
            judge_a_outcome="unavailable", judge_b_outcome="ok", error="e1",
        )
        falsity_failure = CleanerBatchFailureReport(
            axis="falsity", batch_index=0, record_ids=(new_id(),),
            judge_a_outcome="ok", judge_b_outcome="unavailable", error="e2",
        )
        report = _report(
            relevance=RelevanceReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=BandCounts(),
                shelved_ids=(), unclear_ids=(), failures=(relevance_failure,),
            ),
            falsity=FalsityReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=FalsityBandCounts(),
                superseded_ids=(), global_contradicted=(), decision_contradicted=(),
                failures=(falsity_failure,),
            ),
        )
        summary = render_context_summary(report, cap=10)
        assert "2 batch(es) failed" in summary.summary_line


# ===========================================================================
# Bounded rendering — the falsifiability target (Verify step 1)
# ===========================================================================


class TestBoundedRendering:
    def test_two_hundred_unclear_ids_do_not_produce_two_hundred_lines(self) -> None:
        many_ids = tuple(new_id() for _ in range(200))
        report = _report(
            relevance=RelevanceReportSection(
                judge_pair=("a", "b"), rubric_version="v1",
                counts=BandCounts(unclear=200),
                shelved_ids=(), unclear_ids=many_ids, failures=(),
            )
        )
        cap = 5
        summary = render_context_summary(report, cap=cap)

        assert len(summary.owner_attention) <= cap + 1  # cap entries + one overflow marker
        assert len(summary.owner_attention) != 200
        assert summary.owner_attention[-1].startswith("...and")

    def test_fewer_items_than_the_cap_are_not_truncated(self) -> None:
        report = _report(
            relevance=RelevanceReportSection(
                judge_pair=("a", "b"), rubric_version="v1",
                counts=BandCounts(unclear=2),
                shelved_ids=(), unclear_ids=(new_id(), new_id()), failures=(),
            )
        )
        summary = render_context_summary(report, cap=5)
        assert len(summary.owner_attention) == 2
        assert not summary.owner_attention[-1].startswith("...and")


class TestCallTelemetryNotRendered:
    """Acceptance criterion: call telemetry does not leak into the rendered summary --
    ``render_context_summary`` still only ever bounds ``owner_attention`` by
    ``rubric.parameters.report_render_cap``; ``calls`` is never a source it reads from."""

    def _call(self, judge: str) -> JudgeCallRecord:
        return JudgeCallRecord(
            batch_index=0,
            judge=judge,
            rubric_version="v1",
            duration_seconds=1.23,
            status="success",
            error=None,
            prompt_tokens=42,
            completion_tokens=None,
            cost=None,
        )

    def test_owner_attention_and_summary_line_never_mention_call_telemetry(self) -> None:
        marker_judge = "unique-marker-judge:zz-9999"
        report = _report(
            relevance=RelevanceReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=BandCounts(keep=1),
                shelved_ids=(), unclear_ids=(), failures=(),
                calls=(self._call(marker_judge), self._call("b")),
            ),
            falsity=FalsityReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=FalsityBandCounts(),
                superseded_ids=(), global_contradicted=(), decision_contradicted=(), failures=(),
                calls=(self._call(marker_judge),),
            ),
        )
        summary = render_context_summary(report, cap=5)

        rendered = summary.summary_line + " " + " ".join(summary.owner_attention)
        assert marker_judge not in rendered
        assert "duration_seconds" not in rendered
        assert "prompt_tokens" not in rendered

    def test_owner_attention_items_helper_ignores_calls_too(self) -> None:
        """``_owner_attention_items`` (the uncapped collector ``render_context_summary`` bounds)
        must not source anything from ``calls`` either -- pinned independently of the cap logic."""
        marker_judge = "another-marker-judge:yy-0000"
        report = _report(
            relevance=RelevanceReportSection(
                judge_pair=("a", "b"), rubric_version="v1", counts=BandCounts(),
                shelved_ids=(), unclear_ids=(), failures=(),
                calls=(self._call(marker_judge),),
            ),
        )
        items = _owner_attention_items(report)
        assert items == []


# ===========================================================================
# Exactly one set_cleaner_state call per pass -- the review follow-up's own proof
# obligation. A process death between two writes would leave the watermark
# advanced (or, on "blocked", the row untouched) with the report permanently
# lost -- collapsing to one write removes that window rather than narrowing it.
# ===========================================================================


def _count_set_cleaner_state_calls(adapter: MemoryAdapter) -> list[CleanerState]:
    """Wrap ``adapter.set_cleaner_state`` with a spy that records every call's ``state`` argument,
    while still performing the real write -- returns the list the spy appends to."""
    calls: list[CleanerState] = []
    original = adapter.set_cleaner_state

    async def spy(*, scope: Scope, state: CleanerState) -> None:
        calls.append(state)
        await original(scope=scope, state=state)

    adapter.set_cleaner_state = spy  # type: ignore[method-assign]
    return calls


class TestSingleWrite:
    async def test_completed_pass_writes_state_exactly_once(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record shelved -- single-write proof, completed outcome")
        await adapter.upsert(fact, _vec())
        calls = _count_set_cleaner_state_calls(adapter)

        summary = await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
        )
        assert summary.outcome == "completed"

        assert len(calls) == 1
        # The one write carries BOTH the advanced watermark and the report body -- never a bare
        # `report=None` row later overwritten by a second call.
        assert calls[0].watermark_at == summary.watermark_after
        assert calls[0].report is not None
        # One instant backs both `last_pass_at` on the write and `pass_at` inside the report --
        # never two separate `clock.now()` calls.
        report = CleanerReport.model_validate_json(calls[0].report)
        assert report.pass_at == calls[0].last_pass_at

    async def test_blocked_pass_writes_state_exactly_once(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record whose panel cannot be reached -- single-write, blocked")
        await adapter.upsert(fact, _vec())
        calls = _count_set_cleaner_state_calls(adapter)

        summary = await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", should_fail=True),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )
        assert summary.outcome == "blocked"

        assert len(calls) == 1
        # The watermark did not move -- sourced verbatim from what clean() itself established.
        assert calls[0].watermark_at == _T0
        assert calls[0].watermark_at == summary.watermark_after
        assert calls[0].report is not None
        report = CleanerReport.model_validate_json(calls[0].report)
        assert report.outcome == "blocked"
        assert report.pass_at == calls[0].last_pass_at

    async def test_initialized_pass_writes_state_exactly_once_with_no_report(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        calls = _count_set_cleaner_state_calls(adapter)

        summary = await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        assert summary.outcome == "initialized"

        assert len(calls) == 1
        assert calls[0].report is None

    async def test_empty_pool_pass_writes_state_zero_times(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        calls = _count_set_cleaner_state_calls(adapter)

        summary = await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        assert summary.outcome == "empty_pool"

        assert len(calls) == 0

    async def test_bare_clean_without_build_report_writes_zero_times_on_blocked(self) -> None:
        """A direct ``clean()`` caller supplying no ``build_report`` (this module's own hermetic
        test suite, ``test_pass_orchestration.py``) reproduces the PRE-telemetry contract exactly:
        zero writes on "blocked" -- the callback is what turns this write on, not the outcome
        alone."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record whose panel cannot be reached -- bare clean() call")
        await adapter.upsert(fact, _vec())
        calls = _count_set_cleaner_state_calls(adapter)

        summary = await clean(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", should_fail=True),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )
        assert summary.outcome == "blocked"
        assert len(calls) == 0


# ===========================================================================
# run_clean_pass — persist, overwrite, and distinguish "failed" from "judged nothing"
# ===========================================================================


class TestRunCleanPassPersists:
    async def test_completed_pass_report_survives_a_reopen(self) -> None:
        """"Reopen" for MemoryAdapter means a fresh ``get_cleaner_state`` read, independent of
        whatever in-process reference the write held -- ``TestReportSurvivesLanceDBReopen`` below
        proves the same property against a real, closed-and-reopened LanceDB store."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record the panel shelves, persisted end to end")
        await adapter.upsert(fact, _vec())

        summary = await run_clean_pass(
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
        assert summary.outcome == "completed"

        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.report is not None
        report = CleanerReport.model_validate_json(state.report)
        assert report.relevance.shelved_ids == (fact.id,)
        assert report.outcome == "completed"

    async def test_second_pass_report_replaces_the_first(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact1 = _fact(content="first pass record", created_at=_T1)
        await adapter.upsert(fact1, _vec())

        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
        )
        first_state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert first_state is not None
        assert len(adapter._cleaner_state) == 1

        fact2 = _fact(content="second pass record", created_at=_T1 + timedelta(days=1))
        await adapter.upsert(fact2, _vec())

        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )
        second_state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert second_state is not None
        assert second_state.report != first_state.report

        # One row per scope -- overwritten, never appended (acceptance criterion).
        assert len(adapter._cleaner_state) == 1

        second_report = CleanerReport.model_validate_json(second_state.report)  # type: ignore[arg-type]
        assert second_report.relevance.shelved_ids == ()
        assert second_report.relevance.counts.keep == 1


class TestFailedPassDistinguishableFromJudgedNothing:
    async def test_failed_pass_reports_the_failure_distinctly_from_judging_nothing(
        self,
    ) -> None:
        adapter = _make_adapter()
        clock = _make_clock()

        # Scope A: never touched -- "initialized", judges nothing.
        summary_a = await run_clean_pass(
            scope="scope-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        assert summary_a.outcome == "initialized"
        state_a = await adapter.get_cleaner_state(scope=Scope("scope-a"))
        assert state_a is not None
        assert state_a.report is None

        # Scope B: watermark exists, nothing born since -- "empty_pool", also judges nothing.
        await adapter.set_cleaner_state(
            scope=Scope("scope-b"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        summary_b = await run_clean_pass(
            scope="scope-b", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        assert summary_b.outcome == "empty_pool"
        state_b = await adapter.get_cleaner_state(scope=Scope("scope-b"))
        assert state_b is not None
        assert state_b.report is None

        # Scope C: a real newcomer, but the judge is unreachable -- "blocked", a genuine failure.
        await adapter.set_cleaner_state(
            scope=Scope("scope-c"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact_c = _fact(content="a record whose panel cannot be reached", scope="scope-c")
        await adapter.upsert(fact_c, _vec())
        summary_c = await run_clean_pass(
            scope="scope-c", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", should_fail=True),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )
        assert summary_c.outcome == "blocked"
        state_c = await adapter.get_cleaner_state(scope=Scope("scope-c"))
        assert state_c is not None
        assert state_c.report is not None  # <-- the distinguishing bit

        report_c = CleanerReport.model_validate_json(state_c.report)
        assert report_c.outcome == "blocked"
        assert len(report_c.relevance.failures) == 1
        assert report_c.relevance.failures[0].judge_a_outcome == "unavailable"

        # The watermark did not move on the failed pass -- only the report did.
        assert state_c.watermark_at == _T0


# ===========================================================================
# load_context_summary — the read path context() calls
# ===========================================================================


class TestLoadContextSummary:
    """cleaner-telemetry change: ``load_context_summary`` now reads the newest
    row off the append-only pass history (``list_cleaner_passes(scope=..., limit=1)``) instead of
    ``CleanerState.report`` -- these tests pin the read-side contract post-swap. ``report.py`` no
    longer has ANY reader of ``CleanerState.report`` after this class's rewrite; the deliberate
    break in the task's own Verify step 1 targets ``load_context_summary`` specifically (point the
    read back at ``state.report`` and watch
    ``test_second_pass_changes_what_context_renders`` go red for that reason -- recorded verbatim
    in the task verdict, not duplicated here, matching the sibling falsifiability note above)."""

    async def test_no_state_at_all_returns_none(self) -> None:
        """Acceptance criterion: ``context()`` returns ``cleaner=None`` for a scope with no pass
        history -- genuinely no rows, not merely no ``CleanerState``."""
        adapter = _make_adapter()
        assert await load_context_summary(adapter, scope=Scope("never-cleaned")) is None

    async def test_initialized_pass_renders_distinctly_from_empty_pool_and_blocked(self) -> None:
        """``"initialized"`` is a real pass-history row; that is what makes
        ``context()`` render it instead of ``None`` -- the one deliberate exception to "alarm
        policy unchanged" (a run of consecutive empty passes means the trigger stopped firing)."""
        adapter = _make_adapter()
        clock = _make_clock()
        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        summary = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert summary is not None
        assert summary.outcome == "initialized"
        assert "initialized" in summary.summary_line
        assert "empty_pool" not in summary.summary_line
        assert "blocked" not in summary.summary_line

    async def test_empty_pool_pass_renders_distinctly_from_initialized_and_blocked(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        summary = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert summary is not None
        assert summary.outcome == "empty_pool"
        assert "empty_pool" in summary.summary_line
        assert "initialized" not in summary.summary_line
        assert "blocked" not in summary.summary_line

    async def test_completed_pass_renders_a_summary(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record the panel shelves for the context summary test")
        await adapter.upsert(fact, _vec())

        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
        )

        summary = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert isinstance(summary, CleanerContextSummary)
        assert summary.outcome == "completed"
        assert "1 shelved" in summary.summary_line
        assert summary.pass_at == _NOW

    async def test_blocked_pass_renders_distinctly(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record whose panel cannot be reached -- blocked render probe")
        await adapter.upsert(fact, _vec())

        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", should_fail=True),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )

        summary = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert summary is not None
        assert summary.outcome == "blocked"
        assert "blocked" in summary.summary_line
        assert "initialized" not in summary.summary_line
        assert "empty_pool" not in summary.summary_line

    async def test_errored_row_renders_as_errored(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        await record_pass_failure(
            scope="proj-a", adapter=adapter, clock=clock, error=RuntimeError("kaboom")
        )

        summary = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert summary is not None
        assert summary.outcome == "errored"
        assert "kaboom" in summary.summary_line

    async def test_second_pass_changes_what_context_renders(self) -> None:
        """Acceptance criterion: ``context()`` renders the newest pass row for the scope, and a
        second pass changes what it renders. THIS is the falsifiability target for the task's own
        Verify step 1 (recorded verbatim in the task verdict) -- deliberately uses two outcomes
        (``"initialized"`` then ``"empty_pool"``) that NEVER write ``CleanerState.report`` at all
        (``build_report`` is only ever invoked for ``"completed"``/``"blocked"`` -- see
        ``build_report``'s own gate): under the pre-existing read (``state.report``), BOTH renders
        come back ``None`` (the column never moves for either outcome), so pointing the read back
        there makes THIS test go red -- not the sibling two-completed-passes test below, whose
        ``cleaner_state.report`` also happens to change every pass (the "one write, not two"
        composition writes the identical string into both the row and the pass-history payload),
        so it would keep passing under the old read path by coincidence."""
        adapter = _make_adapter()
        clock = _make_clock()

        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        first_render = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert first_render is not None
        assert first_render.outcome == "initialized"

        clock.advance(timedelta(hours=1))
        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
        )
        second_render = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert second_render is not None
        assert second_render.outcome == "empty_pool"
        assert second_render.summary_line != first_render.summary_line

    async def test_two_completed_passes_produce_different_renders(self) -> None:
        """Regression pin, sibling to the falsifiability target above: a rich ``"completed"``
        render (band counts and all) also changes between passes, not just the bare outcome
        word."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact1 = _fact(content="first pass record", created_at=_T1)
        await adapter.upsert(fact1, _vec())
        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
        )
        first_render = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert first_render is not None
        assert "1 shelved" in first_render.summary_line

        fact2 = _fact(content="second pass record", created_at=_T1 + timedelta(days=1))
        await adapter.upsert(fact2, _vec())
        await run_clean_pass(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")])),
        )
        second_render = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert second_render is not None
        assert second_render.summary_line != first_render.summary_line
        assert "1 shelved" not in second_render.summary_line

    async def test_dry_run_row_renders_with_a_visible_marker(self) -> None:
        """Decision inherited: a dry run's verdicts must never read as though the store did
        them -- the summary line carries a visible marker regardless of outcome, and the rejected
        alternative (filtering dry runs out of the alarm) is not what this does."""
        adapter = _make_adapter()
        clock = _make_clock()
        summary = await clean(
            scope="proj-a", adapter=adapter, embedder=FakeEmbedder(dim=_DIM), clock=clock,
            lock=asyncio.Lock(), cfg=EngineConfig(), require_scope=_require_scope_always_ok,
            dry_run=True,
        )
        assert summary.outcome == "initialized"
        assert summary.dry_run is True
        # The dry run performed no CleanerState write -- only the pass-history row landed.
        assert await adapter.get_cleaner_state(scope=Scope("proj-a")) is None

        rendered = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert rendered is not None
        assert "DRY RUN" in rendered.summary_line
        assert "initialized" in rendered.summary_line


# ===========================================================================
# Revisit activity is rendered on the fallback (initialized/empty_pool) payload too (2026-08-12
# cleaner-revisit-pass plan): the starvation fix made the revisit axis run on EVERY outcome, but
# only the rich CleanerReport path
# rendered anything about it -- these tests pin that the fallback path (fallback_payload /
# _render_uneventful / _render_judged's own fallback branch) no longer drops it.
# ===========================================================================


class TestRevisitActivityRenderedOnFallbackOutcomes:
    async def test_empty_pool_pass_still_renders_a_revisit_shelve(self) -> None:
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_NOW, last_pass_at=_NOW)
        )
        old_record = _fact(
            content="An old durable gotcha eligible for the revisit rotation.",
            created_at=_NOW - timedelta(days=100),
        )
        await adapter.upsert(old_record, [0.5] * _DIM)
        await adapter.append_event(
            ReinforcementEvent(
                target_id=old_record.id,
                at=_NOW - timedelta(days=99),
                source="dedup",
                matched_from_id=new_id(),
            )
        )

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

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
        assert summary.outcome == "empty_pool"
        assert summary.revisit is not None
        assert summary.revisit.shelved_ids == (old_record.id,)

        rendered = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert rendered is not None
        assert rendered.outcome == "empty_pool"
        assert "1 shelved" in rendered.summary_line

    async def test_empty_pool_pass_with_a_failed_revisit_batch_is_distinguishable_in_summary_line(
        self,
    ) -> None:
        """2026-08-13 judge-argv-limit fix, extended to this fallback path: the actually-reported
        incident was a
        ``--dry-run`` pass, which renders
        through ``_render_uneventful``/``_render_revisit_fallback`` -- never through
        ``render_context_summary``'s own band-count clause -- so THIS is the render path the fix
        needs to cover for that incident shape, not only the rich ``CleanerReport`` one.
        Falsifiability: revert the ``suffix +=`` line ``_render_revisit_fallback`` gained for this
        fix and re-run -- red, because ``rendered.summary_line`` goes back to ``"revisit: 0
        shelved, 0 contested"`` — identical to the healthy pass above."""
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_NOW, last_pass_at=_NOW)
        )
        old_record = _fact(
            content="An old durable gotcha whose revisit batch judge call fails.",
            created_at=_NOW - timedelta(days=100),
        )
        await adapter.upsert(old_record, [0.5] * _DIM)
        await adapter.append_event(
            ReinforcementEvent(
                target_id=old_record.id,
                at=_NOW - timedelta(days=99),
                source="dedup",
                matched_from_id=new_id(),
            )
        )

        # Only one seat fails -- the revisit axis marks the whole batch failed and writes nothing
        # from it (mirrors `run_revisit_axis`'s own "judge_a_outcome != 'ok' or judge_b_outcome !=
        # 'ok'" gate), so counts stay all-zero exactly like the healthy pass above.
        judge_a = FakeJudgeAdapter(identity="a", should_fail=True)
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

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
        assert summary.outcome == "empty_pool"
        assert summary.revisit is not None
        assert summary.revisit.shelved_ids == ()
        assert len(summary.revisit.failed_batches) == 1

        rendered = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert rendered is not None
        assert rendered.outcome == "empty_pool"
        # Zero counts, same as the healthy pass above -- the failure must show up elsewhere in
        # summary_line, or the two are indistinguishable (the exact defect this closes).
        assert "0 shelved" in rendered.summary_line
        assert "1 batch(es) failed" in rendered.summary_line

    async def test_initialized_pass_still_renders_a_revisit_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _boom(**kwargs: Any) -> Any:
            raise RuntimeError("kaboom")

        monkeypatch.setattr("tabularium.engine.clean.run_revisit_axis", _boom)

        adapter = _make_adapter()
        clock = _make_clock()

        summary = await run_clean_pass(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
        )
        assert summary.outcome == "initialized"
        assert summary.revisit_error is not None

        rendered = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert rendered is not None
        assert rendered.outcome == "initialized"
        assert "revisit axis errored" in rendered.summary_line
        assert any("kaboom" in item for item in rendered.owner_attention)


# ===========================================================================
# Defect-1 fix (2026-08-14 judge-call-error-visibility plan): persisting a failed batch's error
# text (`fallback_payload`, `engine/clean.py`) is necessary but not sufficient -- the session-
# start grounding render must actually surface it, not leave it inert JSON. These pin the RENDER
# side of the fix; the persistence side is pinned in `test_pass_orchestration.py`'s
# `TestFallbackPayloadCarriesBatchFailureErrors`.
# ===========================================================================


class TestFallbackRenderingSurfacesBatchFailureErrors:
    async def test_revisit_batch_failure_error_reaches_owner_attention(self) -> None:
        """Extends `test_empty_pool_pass_with_a_failed_revisit_batch_is_distinguishable_in_
        summary_line` above (which pins the COUNT reaching `summary_line`) with the actual error
        TEXT reaching `owner_attention` -- the exact incident that motivated this fix ("the only
        artifact anywhere was the integer 4").

        Falsifiability: revert `_render_revisit_fallback`'s ``failed_batch_items = ...`` block back
        to the pre-fix ``if failed: items.append(f"revisit: {failed} batch(es) failed")``-only
        shape and re-run -- red, because the distinctive fail_reason string then appears nowhere
        in ``owner_attention``.
        """
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_NOW, last_pass_at=_NOW)
        )
        old_record = _fact(
            content="An old durable gotcha whose revisit batch judge call fails.",
            created_at=_NOW - timedelta(days=100),
        )
        await adapter.upsert(old_record, [0.5] * _DIM)
        await adapter.append_event(
            ReinforcementEvent(
                target_id=old_record.id,
                at=_NOW - timedelta(days=99),
                source="dedup",
                matched_from_id=new_id(),
            )
        )

        judge_a = FakeJudgeAdapter(
            identity="a", should_fail=True, fail_reason="a distinctive revisit outage message"
        )
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-lean")]))

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
        assert summary.outcome == "empty_pool"

        rendered = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert rendered is not None
        assert any(
            "a distinctive revisit outage message" in item for item in rendered.owner_attention
        )

    async def test_relevance_batch_failure_error_reaches_owner_attention_on_a_bare_caller(
        self,
    ) -> None:
        """A bare ``clean()`` caller (no ``build_report``) falls back to ``fallback_payload`` on
        EVERY outcome, including ``"blocked"`` -- the exact shape a dry run also takes. Before this
        fix, the relevance/falsity axes had NO representation at all in that fallback (not even a
        count); this pins that the render now surfaces the actual error text.
        """
        adapter = _make_adapter()
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record whose relevance panel cannot be reached", created_at=_T1)
        await adapter.upsert(fact, [1.0] * _DIM)

        judge_a = FakeJudgeAdapter(
            identity="a", should_fail=True, fail_reason="a distinctive relevance outage message"
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
            # No `build_report` -- this is the bare-caller fallback path (mirrors every dry run's
            # own gap: `build_report` is never invoked under `dry_run` either).
        )
        assert summary.outcome == "blocked"

        rendered = await load_context_summary(adapter, scope=Scope("proj-a"))
        assert rendered is not None
        assert any(
            "a distinctive relevance outage message" in item for item in rendered.owner_attention
        )


class TestRenderPassRecordCapDelegation:
    """Acceptance criterion: ``render_context_summary``'s cap still bounds ``owner_attention`` when
    reached through the new ``render_pass_record``/``load_context_summary`` route -- a pass with
    200 verdicts does not return 200 lines."""

    def test_completed_row_with_many_verdicts_is_still_capped(self) -> None:
        report = _report(
            relevance=RelevanceReportSection(
                judge_pair=("a", "b"), rubric_version="v1",
                counts=BandCounts(unclear=200),
                shelved_ids=(), unclear_ids=tuple(new_id() for _ in range(200)), failures=(),
            )
        )
        record = CleanerPassRecord(
            scope=Scope("proj-a"), pass_at=_NOW, outcome="completed", dry_run=False,
            payload=report.model_dump_json(),
        )
        summary = render_pass_record(record, cap=5)
        assert len(summary.owner_attention) <= 6  # cap entries + one overflow marker
        assert len(summary.owner_attention) != 200

    def test_blocked_fallback_row_with_many_failed_batches_is_still_capped(self) -> None:
        """The RICH ``CleanerReport`` path above was already capped; the FALLBACK path (a dry run,
        or a bare ``clean()`` caller
        -- ``_render_judged``'s non-``kind:report`` branch) was not. A short-message judge outage
        with many failed batches must not flood ``owner_attention`` either.

        Falsifiability: change ``owner_attention=_cap_owner_attention_items(combined_items,
        cap=cap)`` back to ``owner_attention=combined_items`` in ``_render_judged`` and re-run --
        red, because ``len(summary.owner_attention)`` comes back 20, not capped to 6.
        """
        payload = json.dumps(
            {
                "watermark_before": _T0.isoformat(),
                "watermark_after": _T0.isoformat(),
                "pool_size": 20,
                "reason": "20 batch(es) failed (20 relevance, 0 falsity)",
                "relevance_failed_batches": [
                    {
                        "batch_index": i,
                        "judge_a_outcome": "unavailable",
                        "judge_b_outcome": "ok",
                        "error": "x",
                    }
                    for i in range(20)
                ],
                "falsity_failed_batches": [],
                "revisit": None,
                "revisit_error": None,
            }
        )
        record = CleanerPassRecord(
            scope=Scope("proj-a"), pass_at=_NOW, outcome="blocked", dry_run=True, payload=payload
        )
        summary = render_pass_record(record, cap=5)
        assert len(summary.owner_attention) <= 6  # cap entries + one overflow marker
        assert len(summary.owner_attention) != 20

    def test_uneventful_fallback_row_with_many_revisit_failed_batches_is_still_capped(self) -> None:
        """Same fix, the OTHER fallback branch (``_render_uneventful``, ``"initialized"``/
        ``"empty_pool"``) -- regression
        coverage for fallback render capping on both judged and uneventful pass-history rows.
        """
        payload = json.dumps(
            {
                "watermark_before": None,
                "watermark_after": _NOW.isoformat(),
                "pool_size": 0,
                "reason": "scope has never been judged",
                "relevance_failed_batches": [],
                "falsity_failed_batches": [],
                "revisit": {
                    "pool_size": 20,
                    "counts": {"shelve": 0, "keep": 0, "unclear": 0, "contested": 0},
                    "shelved_ids": [],
                    "contested_ids": [],
                    "protected_count": 0,
                    "skipped_no_evidence_count": 0,
                    "failed_batch_count": 20,
                    "failed_batches": [
                        {
                            "batch_index": i,
                            "judge_a_outcome": "unavailable",
                            "judge_b_outcome": "ok",
                            "error": "x",
                        }
                        for i in range(20)
                    ],
                    "verification_failure_count": 0,
                    "review_append_failure_count": 0,
                    "refused_count": 0,
                },
                "revisit_error": None,
            }
        )
        record = CleanerPassRecord(
            scope=Scope("proj-a"),
            pass_at=_NOW,
            outcome="initialized",
            dry_run=True,
            payload=payload,
        )
        summary = render_pass_record(record, cap=5)
        assert len(summary.owner_attention) <= 6  # cap entries + one overflow marker
        assert len(summary.owner_attention) != 20


# ===========================================================================
# context() integration — the risky criterion: additive only, existing shape pinned
# ===========================================================================


class TestContextIntegration:
    async def _engine(self) -> Tabularium:
        adapter = _make_adapter()
        clock = _make_clock()
        engine = Tabularium(adapter, FakeEmbedder(dim=_DIM), clock, EngineConfig())
        await engine.initialize()
        await adapter.create_scope(Scope("proj-a"))
        return engine

    async def test_scope_with_no_cleaner_state_gets_cleaner_none_and_unchanged_payload_shape(
        self,
    ) -> None:
        """Regression pin (acceptance criterion): existing tiers/budget_used/skipped are exactly
        what a scope with no Cleaner state always returned -- ``cleaner`` is the only addition."""
        engine = await self._engine()

        payload = await engine.context("proj-a", budget=100, role="decider")

        assert [t.name for t in payload.tiers] == ["hot", "ember"]
        for tier in payload.tiers:
            assert tier.entries == []
        assert payload.budget_used == 0
        assert payload.skipped == 0
        assert payload.cleaner is None

    async def test_scope_with_a_completed_pass_surfaces_the_last_pass_summary(self) -> None:
        engine = await self._engine()
        await engine.clean("proj-a")  # "initialized" -- watermark set, nothing judged yet

        adapter = engine._adapter
        fact = _fact(content="a record for the end-to-end context surfacing test")
        await adapter.upsert(fact, _vec())

        # Force the watermark backward so the record above is "born since" -- avoids depending on
        # wall-clock skew between fixture construction and `initialize()`'s own `clock.now()`.
        state = await adapter.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"),
            state=CleanerState(watermark_at=_T0, last_pass_at=state.last_pass_at),
        )

        from tabularium.engine.report import run_clean_pass as _run

        await _run(
            scope="proj-a",
            adapter=adapter,
            embedder=engine._embedder,
            clock=engine._clock,
            lock=engine._write_lock,
            cfg=engine._cfg,
            require_scope=engine._require_scope,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
        )

        payload = await engine.context("proj-a", budget=100, role="decider")
        assert payload.cleaner is not None
        assert payload.cleaner.outcome == "completed"
        assert "1 shelved" in payload.cleaner.summary_line
        # Untouched by the new field's presence.
        assert payload.budget_used == 0
        assert payload.skipped == 0


# ===========================================================================
# LanceDB — "survives a reopen" against real, closed-and-reopened storage
# ===========================================================================


class TestReportSurvivesLanceDBReopen:
    async def test_report_written_by_one_adapter_reads_back_through_a_fresh_open(
        self, tmp_path: pathlib.Path
    ) -> None:
        embedding_model = "fake/nomic-embed-text"
        path = str(tmp_path / "cleaner-report-reopen")

        adapter = await LanceDBAdapter.open(path, embedding_model=embedding_model, dim=_DIM)
        clock = _make_clock()
        await adapter.set_cleaner_state(
            scope=Scope("proj-a"), state=CleanerState(watermark_at=_T0, last_pass_at=_T0)
        )
        fact = _fact(content="a record shelved through the real LanceDB adapter")
        await adapter.upsert(fact, _vec())

        summary = await run_clean_pass(
            scope="proj-a",
            adapter=adapter,
            embedder=FakeEmbedder(dim=_DIM, model=embedding_model),
            clock=clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            require_scope=_require_scope_always_ok,
            rubric=load_rubric(),
            judge_a=FakeJudgeAdapter(identity="a", response=_payload([_v(1, "shelve-clear")])),
            judge_b=FakeJudgeAdapter(identity="b", response=_payload([_v(1, "shelve-clear")])),
        )
        assert summary.outcome == "completed"

        # A FRESH adapter instance against the SAME path -- a genuine reopen, not the same
        # in-process object the write happened through.
        reopened = await LanceDBAdapter.open(path, embedding_model=embedding_model, dim=_DIM)
        state = await reopened.get_cleaner_state(scope=Scope("proj-a"))
        assert state is not None
        assert state.report is not None

        report = CleanerReport.model_validate_json(state.report)
        assert report.outcome == "completed"
        assert report.relevance.shelved_ids == (fact.id,)
        assert report.relevance.counts.shelve == 1
        assert report.relevance.judge_pair == ("a", "b")


# ===========================================================================
# Trim and merge: carried through build_report, the fallback payload, and old payloads
# ===========================================================================


def _trim_merge_summary() -> tuple[CleanSummary, dict[str, Any]]:
    from tabularium.engine.cleaner_writes import MergedTriple, MergeRevertedEntry, TrimmedPair
    from tabularium.engine.revisit import RevisitAxisSummary

    names = ("r_old", "r_new", "v_old", "v_new", "o", "n", "m", "o2", "n2")
    ids = {name: new_id() for name in names}
    relevance = _contested_relevance_summary(record_id=new_id()).model_copy(
        update={"trimmed": (TrimmedPair(old_id=ids["r_old"], new_id=ids["r_new"]),)}
    )
    falsity = _contested_falsity_summary(newcomer_id=new_id(), candidate_id=new_id()).model_copy(
        update={
            "merged": (MergedTriple(older_id=ids["o"], newer_id=ids["n"], merged_id=ids["m"]),),
            "merge_reverted": (
                MergeRevertedEntry(older_id=ids["o2"], newer_id=ids["n2"], error="bounced"),
            ),
            "merge_revert_failed": (
                MergeRevertedEntry(older_id=ids["o"], newer_id=ids["n2"], error="restore failed"),
            ),
        }
    )
    revisit = RevisitAxisSummary(
        scope=Scope("proj-a"),
        rubric_version="v1",
        judge_pair=("a", "b"),
        pool_size=1,
        excluded_global=(),
        excluded_non_fact=(),
        protected_ids=(),
        skipped_no_evidence_ids=(),
        counts=BandCounts(shelve=1),
        shelved_ids=(ids["v_old"],),
        unclear_ids=(),
        contested_ids=(),
        failed_batches=(),
        calls=(),
        verdicts=(),
        contested_verdicts=(),
        verification_failures=(),
        trimmed=(TrimmedPair(old_id=ids["v_old"], new_id=ids["v_new"]),),
    )
    summary = CleanSummary(
        scope=Scope("proj-a"),
        outcome="completed",
        watermark_before=_T0,
        watermark_after=_NOW,
        pool_size=1,
        relevance=relevance,
        falsity=falsity,
        revisit=revisit,
        reason="completed",
    )
    return summary, ids


class TestTrimAndMergeInTheReport:
    def test_trim_and_merge_survive_build_report_and_the_json_round_trip(self) -> None:
        summary, ids = _trim_merge_summary()

        report = build_report(summary, pass_at=_NOW)
        assert report is not None
        round_tripped = CleanerReport.model_validate_json(report.model_dump_json())

        assert [(p.old_id, p.new_id) for p in round_tripped.relevance.trimmed] == [
            (ids["r_old"], ids["r_new"])
        ]
        assert round_tripped.revisit is not None
        assert [(p.old_id, p.new_id) for p in round_tripped.revisit.trimmed] == [
            (ids["v_old"], ids["v_new"])
        ]
        assert [(t.older_id, t.newer_id, t.merged_id) for t in round_tripped.falsity.merged] == [
            (ids["o"], ids["n"], ids["m"])
        ]
        reverted = round_tripped.falsity.merge_reverted
        assert [(m.older_id, m.newer_id, m.error) for m in reverted] == [
            (ids["o2"], ids["n2"], "bounced")
        ]
        failed_undos = round_tripped.falsity.merge_revert_failed
        assert [(m.older_id, m.newer_id, m.error) for m in failed_undos] == [
            (ids["o"], ids["n2"], "restore failed")
        ]

    def test_the_revisit_fallback_payload_carries_trim_counts_and_ids(self) -> None:
        from tabularium.engine.clean import fallback_payload

        summary, ids = _trim_merge_summary()

        payload = json.loads(fallback_payload(summary))

        assert payload["revisit"]["trimmed_count"] == 1
        assert payload["revisit"]["trimmed"] == [
            {"old_id": str(ids["v_old"]), "new_id": str(ids["v_new"])}
        ]

    def test_a_report_json_without_any_trim_or_merge_key_still_loads(self) -> None:
        old_style = {
            "scope": "proj-a",
            "pass_at": _NOW.isoformat(),
            "outcome": "completed",
            "relevance": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {},
                "shelved_ids": [],
                "unclear_ids": [],
                "failures": [],
            },
            "falsity": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {},
                "superseded_ids": [],
                "global_contradicted": [],
                "decision_contradicted": [],
                "failures": [],
            },
            "revisit": {
                "judge_pair": ["a", "b"],
                "rubric_version": "v1",
                "counts": {"shelve": 1},
                "shelved_ids": [],
                "unclear_ids": [],
            },
            "verification_failures": [],
        }

        report = CleanerReport.model_validate(old_style)

        assert report.relevance.trimmed == ()
        assert report.falsity.merged == ()
        assert report.falsity.merge_reverted == ()
        assert report.revisit is not None
        assert report.revisit.trimmed == ()


class TestBlockedStreak:
    @staticmethod
    async def _append(
        adapter: MemoryAdapter, outcome: str, day: int, *, dry_run: bool = False
    ) -> None:
        await adapter.append_cleaner_pass(
            record=CleanerPassRecord(
                scope=Scope("proj-a"),
                pass_at=_T0 + timedelta(days=day),
                outcome=outcome,  # type: ignore[arg-type]
                dry_run=dry_run,
                payload=json.dumps({"error": "x"}),
            )
        )

    async def test_three_blocked_rows_make_a_streak_of_three(self) -> None:
        adapter = _make_adapter()
        await self._append(adapter, "completed", 0)
        for day in (1, 2, 3):
            await self._append(adapter, "blocked", day)

        summary = await load_context_summary(adapter, scope=Scope("proj-a"))

        assert summary is not None
        assert summary.blocked_streak == 3
        assert summary.blocked_since == _T0 + timedelta(days=1)

    async def test_errored_rows_count_and_dry_runs_are_skipped(self) -> None:
        adapter = _make_adapter()
        await self._append(adapter, "blocked", 1)
        await self._append(adapter, "completed", 2, dry_run=True)
        await self._append(adapter, "errored", 3)

        summary = await load_context_summary(adapter, scope=Scope("proj-a"))

        assert summary is not None
        assert summary.blocked_streak == 2

    async def test_one_completed_row_resets_the_streak(self) -> None:
        adapter = _make_adapter()
        for day in (1, 2, 3):
            await self._append(adapter, "blocked", day)
        await self._append(adapter, "completed", 4)

        summary = await load_context_summary(adapter, scope=Scope("proj-a"))

        assert summary is not None
        assert summary.blocked_streak == 0
        assert summary.blocked_since is None
