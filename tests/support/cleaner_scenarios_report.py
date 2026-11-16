"""Report-rendering scenarios for the cleaner golden. Importing the module registers them.

These build the axis summaries by hand so that every kind of owner-attention item is present at
once, which none of the scripted passes does.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Literal
from unittest import mock

from tabularium.domain import Scope
from tabularium.engine import report as report_module
from tabularium.engine.axis_common import (
    BandCounts,
    BatchFailure,
    ContestedVerdict,
    RecordVerdict,
    RefusedVerdict,
)
from tabularium.engine.clean import CleanSummary, fallback_payload
from tabularium.engine.falsity import (
    FalsityAxisSummary,
    FalsityBandCounts,
    FalsityBatchFailure,
    FalsityContestedPair,
    FalsityEdge,
    FalsityRefusedPair,
    FalsityReportEntry,
)
from tabularium.engine.relevance import RelevanceAxisSummary
from tabularium.engine.report import (
    CleanerFailureReport,
    CleanerReport,
    blocked_streak,
    build_report,
    load_context_summary,
    record_pass_failure,
    render_context_summary,
)
from tabularium.engine.revisit import RevisitAxisSummary, RevisitReviewAppendFailure
from tabularium.storage.types import CleanerPassRecord, CleanerState
from tests.support.cleaner_world import (
    AGED,
    PASS_AT,
    SINCE,
    Json,
    World,
    dump_cleaner_state,
    idle_falsity,
    idle_relevance,
    make_world,
    render_all,
    rid,
    scenario,
)

_JUDGES = ("judge-a", "judge-b")


def _why(label: str) -> str:
    return (
        f"{label}: " + "the older record said otherwise, and nothing since has changed that. " * 4
    )


def _error(label: str) -> str:
    return f"{label} failed: " + "diagnostic line that keeps going " * 12 + "THE-LIVE-FAILURE-LINE"


def _relevance(world: World) -> RelevanceAxisSummary:
    return RelevanceAxisSummary(
        scope=world.scope,
        rubric_version=world.rubric.version,
        judge_pair=_JUDGES,
        pool_size=9,
        excluded_decisions=(),
        excluded_global=(),
        excluded_already_shelved=(),
        counts=BandCounts(shelve=1, keep=2, unclear=1, contested=2),
        shelved_ids=(rid(1),),
        unclear_ids=(rid(2),),
        contested_ids=(rid(3), rid(4)),
        failed_batches=(
            BatchFailure(
                batch_index=1,
                record_ids=(rid(5),),
                judge_a_outcome="unavailable",
                judge_b_outcome="ok",
                error=_error("relevance"),
            ),
        ),
        calls=(),
        verdicts=(RecordVerdict(id=rid(1), band="shelve", why="stale"),),
        contested_verdicts=tuple(
            ContestedVerdict(
                id=rid(number),
                judge_a_identity="judge-a",
                judge_a_grade="shelve-clear",
                judge_a_why=_why("A"),
                judge_b_identity="judge-b",
                judge_b_grade="keep-clear",
                judge_b_why="B: still true",
            )
            for number in (3, 4)
        ),
        refused=(RefusedVerdict(id=rid(6), reason="set_shelved refused the record"),),
    )


def _falsity(world: World) -> FalsityAxisSummary:
    return FalsityAxisSummary(
        scope=world.scope,
        rubric_version=world.rubric.version,
        judge_pair=_JUDGES,
        newcomer_pool_size=6,
        excluded_decisions=(),
        excluded_global=(),
        excluded_shelved_this_pass=(),
        counts=FalsityBandCounts(replace=1, coexist=1, contested=2),
        edges=(
            FalsityEdge(
                id=rid(11),
                superseded_by=rid(12),
                similarity=0.96,
                deciding_judge="judge-a",
                why="moved",
            ),
        ),
        decision_candidates=(
            FalsityReportEntry(
                newcomer_id=rid(13),
                candidate_id=rid(14),
                similarity=0.9,
                reason="candidate_is_decision",
            ),
        ),
        global_candidates=(
            FalsityReportEntry(
                newcomer_id=rid(15),
                candidate_id=rid(16),
                similarity=0.9,
                reason="candidate_is_global",
                assessor_pick="changed",
            ),
            FalsityReportEntry(
                newcomer_id=rid(17),
                candidate_id=rid(18),
                similarity=0.9,
                reason="candidate_is_global",
            ),
        ),
        hot_or_ember_candidates=(),
        contested_pairs=tuple(
            FalsityContestedPair(
                newcomer_id=rid(number),
                candidate_id=rid(number + 1),
                similarity=0.96,
                judge_a_identity="judge-a",
                judge_a_verdict="replace",
                judge_a_why=_why("A"),
                judge_b_identity="judge-b",
                judge_b_verdict="coexist",
                judge_b_why="B: both stand",
            )
            for number in (21, 23)
        ),
        failed_batches=(
            FalsityBatchFailure(
                batch_index=0,
                pairs=((rid(25), rid(26)),),
                judge_a_outcome="invalid_output",
                judge_b_outcome="ok",
                error=_error("falsity"),
            ),
        ),
        calls=(),
        refused=(
            FalsityRefusedPair(
                newcomer_id=rid(27), candidate_id=rid(28), reason="not valid any more"
            ),
        ),
    )


def _revisit(world: World) -> RevisitAxisSummary:
    return RevisitAxisSummary(
        scope=world.scope,
        rubric_version=world.rubric.version,
        judge_pair=_JUDGES,
        pool_size=7,
        excluded_global=(),
        excluded_non_fact=(),
        protected_ids=(rid(30),),
        skipped_no_evidence_ids=(rid(31),),
        skipped_too_large_ids=(rid(32),),
        counts=BandCounts(shelve=1, keep=1, unclear=0, contested=2),
        shelved_ids=(rid(33),),
        unclear_ids=(),
        contested_ids=(rid(34), rid(35)),
        failed_batches=(
            BatchFailure(
                batch_index=2,
                record_ids=(rid(36),),
                judge_a_outcome="ok",
                judge_b_outcome="unavailable",
                error=_error("revisit"),
            ),
        ),
        calls=(),
        verdicts=(),
        contested_verdicts=tuple(
            ContestedVerdict(
                id=rid(number),
                judge_a_identity="judge-a",
                judge_a_grade="shelve-clear",
                judge_a_why=_why("A"),
                judge_b_identity="judge-b",
                judge_b_grade="keep-clear",
                judge_b_why="B: still true",
            )
            for number in (34,)
        ),
        verification_failures=(rid(37),),
        review_append_failures=(
            RevisitReviewAppendFailure(
                target_id=rid(38), outcome="judged", error="review store down"
            ),
        ),
        refused=(RefusedVerdict(id=rid(39), reason="not eligible for this caller"),),
    )


def _summary(world: World, *, revisit: bool) -> CleanSummary:
    return CleanSummary(
        scope=world.scope,
        outcome="blocked",
        watermark_before=SINCE,
        watermark_after=SINCE,
        pool_size=9,
        relevance=_relevance(world),
        falsity=_falsity(world),
        verification_failures=(rid(7), rid(8)),
        revisit=_revisit(world) if revisit else None,
        revisit_error=None if revisit else "revisit axis exploded",
        reason="hand-built for the report golden",
    )


@scenario
async def report_renders_every_owner_attention_item_in_frozen_order() -> Json:
    world = await make_world()
    dump: Json = {}
    for label, revisit in (("with_revisit_section", True), ("with_revisit_error", False)):
        summary = _summary(world, revisit=revisit)
        report = build_report(summary, pass_at=PASS_AT)
        assert report is not None
        uncapped = render_context_summary(report, cap=100)
        stripped = json.loads(report.model_dump_json())
        for axis in ("relevance", "falsity", "revisit"):
            if stripped.get(axis) is not None:
                stripped[axis]["contested_verdicts"] = []
        old_shape = CleanerReport.model_validate(stripped)
        dump[label] = {
            "rendered": render_all(world, summary),
            "uncapped_context_summary": uncapped.model_dump(mode="json"),
            "context_summary_without_contested_verdicts": render_context_summary(
                old_shape, cap=100
            ).model_dump(mode="json"),
        }
    return dump


async def _append_row(
    world: World,
    *,
    outcome: Literal["completed", "blocked", "errored"],
    at: datetime,
    dry_run: bool = False,
) -> None:
    summary = CleanSummary(
        scope=world.scope,
        outcome="blocked" if outcome == "blocked" else "completed",
        watermark_before=SINCE,
        watermark_after=SINCE,
        pool_size=3,
        relevance=idle_relevance(world),
        falsity=idle_falsity(world),
        reason=f"synthetic {outcome} row",
        dry_run=dry_run,
    )
    payload = (
        json.dumps({"error": "scheduler crashed"})
        if outcome == "errored"
        else fallback_payload(summary)
    )
    await world.adapter.append_cleaner_pass(
        record=CleanerPassRecord(
            scope=world.scope,
            pass_at=at,
            outcome=outcome,
            dry_run=dry_run,
            payload=payload,
        )
    )


@scenario
async def report_context_summary_counts_the_blocked_and_errored_streak() -> Json:
    world = await make_world()
    stages: Json = {}
    none_yet = await load_context_summary(world.adapter, scope=world.scope, rubric=world.rubric)
    stages["no_history"] = None if none_yet is None else none_yet.model_dump(mode="json")
    rows: tuple[tuple[Literal["completed", "blocked", "errored"], bool], ...] = (
        ("blocked", False),
        ("completed", False),
        ("blocked", False),
        ("blocked", True),
        ("blocked", False),
        ("errored", False),
    )
    for number, (outcome, dry_run) in enumerate(rows):
        await _append_row(world, outcome=outcome, at=AGED + timedelta(days=number), dry_run=dry_run)
        loaded = await load_context_summary(world.adapter, scope=world.scope, rubric=world.rubric)
        assert loaded is not None
        stages[f"after_row_{number}_{outcome}{'_dry' if dry_run else ''}"] = loaded.model_dump(
            mode="json"
        )
    with mock.patch.object(report_module, "load_rubric", lambda: world.rubric):
        defaulted = await load_context_summary(world.adapter, scope=world.scope)
    assert defaulted is not None
    stages["with_the_default_rubric"] = defaulted.model_dump(mode="json")
    listed = await world.adapter.list_cleaner_passes(scope=world.scope, limit=10)
    streak, since = blocked_streak(listed)
    stages["streak"] = {"count": streak, "since": since.isoformat() if since else None}
    stages["pass_history"] = [row.model_dump(mode="json") for row in listed]
    return stages


@scenario
async def report_record_pass_failure_persists_a_failure_report_and_a_history_row() -> Json:
    stages: Json = {}
    for label in (
        "scope_never_judged",
        "scope_with_a_state",
        "state_store_down",
        "history_store_down",
    ):
        world = await make_world()
        if label != "scope_never_judged":
            await world.adapter.set_cleaner_state(
                scope=world.scope,
                state=CleanerState(watermark_at=SINCE, last_pass_at=SINCE, report=None),
            )
        broken_state = label == "state_store_down"
        broken_history = label == "history_store_down"

        async def dead_state(**kwargs: object) -> None:
            raise RuntimeError("state store down")

        async def dead_history(**kwargs: object) -> None:
            raise RuntimeError("history store down")

        with (
            mock.patch.object(
                world.adapter,
                "set_cleaner_state",
                dead_state if broken_state else world.adapter.set_cleaner_state,
            ),
            mock.patch.object(
                world.adapter,
                "append_cleaner_pass",
                dead_history if broken_history else world.adapter.append_cleaner_pass,
            ),
        ):
            await record_pass_failure(
                scope=str(world.scope),
                adapter=world.adapter,
                clock=world.clock,
                error=RuntimeError("scheduler crashed"),
            )
        rows = await world.adapter.list_cleaner_passes(scope=world.scope, limit=10)
        stages[label] = {
            "state": dump_cleaner_state(world.adapter),
            "pass_history": [row.model_dump(mode="json") for row in rows],
        }
    failure = CleanerFailureReport(
        scope=Scope("proj-a"), pass_at=PASS_AT, error="scheduler crashed"
    )
    stages["failure_report_rendered"] = render_context_summary(failure, cap=5).model_dump(
        mode="json"
    )
    return stages
