"""The Cleaner's pass orchestration: order the axes, verify the writes, then move the watermark.

``clean()`` is the entry point; ``engine/report.py``'s ``run_clean_pass`` wraps it. Pool selection,
band policy and the write calls live in ``engine/relevance.py``, ``engine/falsity.py`` and
``engine/revisit.py``. Sequence:

1. Read the ``CleanerState`` for the scope.
2. No state: set the watermark to now and stop. The birth axes do not run and ``list_born_since`` is
   not called, because a first pass must not sweep the whole store. The revisit axis still runs
   (step 6).
3. Take one snapshot of the pool, the records born since the watermark.
4. Empty pool: outcome ``"empty_pool"``, watermark unchanged.
5. Run the relevance axis, then the falsity axis. Falsity leaves out the newcomers that relevance
   shelved, contested, found unclear or refused: a contested newcomer's admission is still
   unresolved, so it must not supersede an older record in the same pass.
6. Run the revisit axis.
7. Re-read every write the birth axes report and check the transition is present.
8. A failed batch on either birth axis, or a write that did not verify, gives ``"blocked"`` and the
   watermark stays put. Otherwise the outcome is ``"completed"`` and the watermark advances.
9. Append a pass-history row, for every outcome.

Revisit axis: ``_run_revisit_axis_isolated`` runs it on every outcome branch, so a scope that rarely
produces new records still gets old records revisited. It is a rotation over old records with no
watermark. Its exceptions and reported failures land in ``CleanSummary.revisit`` and
``revisit_error`` alone; nothing in steps 7 and 8 reads them, so they never change ``outcome`` or
the watermark.

Verification re-reads each id the axes report written (``adapter.get_by_id``) and checks ``lifecycle
== "shelved"`` for a relevance write or ``superseded_by == edge.superseded_by`` for a falsity write.
It does not trust the write call's return value. An id that is now unknown counts as a failure.

Watermark advance: the new value is the newest ``created_at`` in the step 3 snapshot, not
``clock.now()``. A record written while the pass runs is newer than every snapshot record, so the
watermark stops short of it and the next pass picks it up. On ``"blocked"`` the watermark is
``since`` as read at the top of the call. The next pass derives its pool from the same ``since``;
re-judging is safe because ``_select_relevance_pool`` and ``_select_falsity_newcomers`` exclude
records already shelved.

Report: ``build_report`` is an optional ``(CleanSummary, datetime) -> str | None`` callback. For a
``"completed"`` or ``"blocked"`` outcome it is called once, where the pass is about to call
``set_cleaner_state``, with the same ``now`` that becomes ``last_pass_at``. Its return value is
stored in that write's ``report=`` field, so there is no second write and a crash cannot land the
watermark without the report. Without a callback ``report`` is ``None`` and a ``"blocked"`` outcome
writes no ``CleanerState`` at all. The pass-history row carries the report string verbatim when one
was built, else ``fallback_payload``.

NOTE: ``CleanerState.report`` has no production reader; ``context()`` reads the pass history.

Dry run: judges as usual and writes nothing apart from the history row. No watermark initialize, no
axis writes, no verification, no ``CleanerState`` write, and ``build_report`` is not called. The
history row is still appended with ``dry_run=True`` and the fallback payload.

Judge panel: ``judge_a`` and ``judge_b`` default to the rubric's ``seat_a`` and ``seat_b`` through
``build_judge_adapter``. No backstop substitution happens here: an unavailable primary seat fails
its batch, nothing is written and the watermark does not move.

One scope per call. This module writes only the ``CleanerState`` and history rows itself; the axes
make the record writes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import BatchOutcome, Fact, Hypomnema, HypomnemaId, Scope
from tabularium.embedding import Embedder
from tabularium.engine._types import RequireScope
from tabularium.engine.assessor import Assessor
from tabularium.engine.falsity import FalsityAxisSummary, FalsityEdge, run_falsity_axis
from tabularium.engine.filters import validate_and_resolve_scope
from tabularium.engine.judge import JudgeAdapter, build_judge_adapter
from tabularium.engine.pass_payloads import (
    BatchFailureEntry,
    FailedBatchEntry,
    FallbackPayload,
    OmittedMarker,
    RevisitCounts,
    RevisitFallbackPayload,
    TrimmedEntry,
)
from tabularium.engine.relevance import RelevanceAxisSummary, run_relevance_axis
from tabularium.engine.revisit import RevisitAxisSummary, run_revisit_axis
from tabularium.engine.rubric import Rubric, load_rubric
from tabularium.storage.types import CleanerPassRecord, CleanerState, StorageAdapter

_log = logging.getLogger("tabularium")

# `pool_limit` resolves once (the argument, else `Rubric.parameters.pool_limit`) and is passed
# explicitly to both axes, so this module's snapshot and the axes' re-queries use one value.


# ---------------------------------------------------------------------------
# Structured summary — one Cleaner pass, one scope
# ---------------------------------------------------------------------------


class CleanSummary(BaseModel):
    """Structured result of one Cleaner pass.

    This module does not build reports; it stores what an injected ``build_report`` callback
    returns. ``outcome``, ``watermark_*`` and ``verification_failures`` describe the birth axes
    only. ``revisit`` is the revisit axis's summary and ``revisit_error`` the exception text when it
    raised before returning one; at most one of them is set. On a dry run, ``watermark_after`` and
    ``reason`` describe what the pass would do.
    """

    model_config = {"frozen": True}

    scope: Scope
    outcome: Literal["initialized", "empty_pool", "completed", "blocked"]
    watermark_before: datetime | None
    watermark_after: datetime
    pool_size: int
    relevance: RelevanceAxisSummary | None = None
    falsity: FalsityAxisSummary | None = None
    verification_failures: tuple[HypomnemaId, ...] = ()
    # Isolated from the birth axes; see the class docstring.
    revisit: RevisitAxisSummary | None = None
    revisit_error: str | None = None
    reason: str
    # True when the summary describes what a pass would do rather than what it did.
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Verification — re-read every write either axis reports, confirm the transition landed
# ---------------------------------------------------------------------------


async def _verify_writes(
    *,
    adapter: StorageAdapter,
    shelved_ids: Sequence[HypomnemaId],
    edges: Sequence[FalsityEdge],
) -> tuple[HypomnemaId, ...]:
    """Re-read every id either axis reports as written and return the ids whose transition is
    absent.

    A relevance write verifies when a fresh ``get_by_id`` returns a ``Fact`` with ``lifecycle ==
    "shelved"``. A falsity write verifies when it returns a ``Fact`` whose ``superseded_by`` equals
    the reported edge. An id that is now unknown (``KeyError``) fails.
    """
    failures: list[HypomnemaId] = []

    for record_id in shelved_ids:
        try:
            record = await adapter.get_by_id(record_id)
        except KeyError:
            failures.append(record_id)
            continue
        if not (isinstance(record, Fact) and record.lifecycle == "shelved"):
            failures.append(record_id)

    for edge in edges:
        try:
            record = await adapter.get_by_id(edge.id)
        except KeyError:
            failures.append(edge.id)
            continue
        if not (isinstance(record, Fact) and record.superseded_by == edge.superseded_by):
            failures.append(edge.id)

    return tuple(failures)


def _newest_created_at(pool: Sequence[Hypomnema]) -> datetime:
    """The newest ``created_at`` in ``pool``, the candidate watermark.

    It comes from the snapshot this pass judged, not from ``clock.now()``, so a record written
    during the pass stays ahead of the watermark.
    """
    return max(record.created_at for record in pool)


# ---------------------------------------------------------------------------
# Pass-history append: one row per outcome
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Persisted batch-failure error excerpts, capped
# ---------------------------------------------------------------------------
#
# Applies to the fallback payload built by `fallback_payload`, which every dry run takes. Without a
# cap, a repeated judge failure would grow the append-only pass_history row on every dry run. The
# rich report path caps each error itself (`engine.report._capped_persisted_error`, which reuses
# FALLBACK_ERROR_EXCERPT_CAP).
#
# - FALLBACK_ERROR_EXCERPT_CAP bounds one failed batch's error text. A longer error keeps its
#   tail, because the live failure line sits at the end of a joined or echoed diagnostic.
# - _FALLBACK_ERROR_TOTAL_CAP bounds the sum of all excerpts in one payload, across the three
#   axes. Once it is spent, the remaining failures of that axis collapse into one
#   {"omitted_count": N} entry.
FALLBACK_ERROR_EXCERPT_CAP = 500
_FALLBACK_ERROR_TOTAL_CAP = 4000


class _BatchFailureLike(Protocol):
    """The four fields of a batch failure that the fallback payload needs.

    ``BatchFailure`` (relevance and revisit) and ``FalsityBatchFailure`` have no common
    base, so this protocol matches them structurally.
    """

    batch_index: int
    judge_a_outcome: BatchOutcome
    judge_b_outcome: BatchOutcome
    error: str


def _capped_entries(
    failures: Sequence[_BatchFailureLike], *, budget: int
) -> tuple[list[FailedBatchEntry], int]:
    """Render ``failures`` into entries for the fallback pass-history payload.

    Each error keeps its last ``min(FALLBACK_ERROR_EXCERPT_CAP, remaining)`` characters. ``budget``
    is a total shared by the axes of one payload: it is passed in and returned decremented. When it
    reaches zero, the remaining failures collapse into one ``OmittedMarker``, so a reader can tell
    "no more failures" from "budget spent". The slice is never empty, because the loop's
    ``remaining <= 0`` guard has already passed.

    Returns ``(entries, budget_remaining)``.
    """
    entries: list[FailedBatchEntry] = []
    remaining = budget
    for index, failure in enumerate(failures):
        if remaining <= 0:
            entries.append(OmittedMarker(omitted_count=len(failures) - index))
            break
        excerpt_cap = min(FALLBACK_ERROR_EXCERPT_CAP, remaining)
        excerpt = failure.error[-excerpt_cap:]
        remaining -= len(excerpt)
        entries.append(
            BatchFailureEntry(
                batch_index=failure.batch_index,
                judge_a_outcome=failure.judge_a_outcome,
                judge_b_outcome=failure.judge_b_outcome,
                error=excerpt,
            )
        )
    return entries, remaining


def _revisit_fallback_payload(
    summary: CleanSummary, *, budget: int
) -> tuple[RevisitFallbackPayload | None, int]:
    """``summary.revisit`` as a ``RevisitFallbackPayload``, with its failed-batch excerpts drawn
    from ``budget``. Returns ``(None, budget)`` when the revisit axis returned no summary.
    """
    if summary.revisit is None:
        return None, budget

    revisit = summary.revisit
    failed_batches, remaining = _capped_entries(revisit.failed_batches, budget=budget)
    payload = RevisitFallbackPayload(
        pool_size=revisit.pool_size,
        counts=RevisitCounts(
            shelve=revisit.counts.shelve,
            keep=revisit.counts.keep,
            unclear=revisit.counts.unclear,
            contested=revisit.counts.contested,
        ),
        split_keep=revisit.split_keep,
        shelved_ids=tuple(str(i) for i in revisit.shelved_ids),
        contested_ids=tuple(str(i) for i in revisit.contested_ids),
        protected_count=len(revisit.protected_ids),
        skipped_no_evidence_count=len(revisit.skipped_no_evidence_ids),
        skipped_too_large_count=len(revisit.skipped_too_large_ids),
        failed_batch_count=len(revisit.failed_batches),
        failed_batches=tuple(failed_batches),
        verification_failure_count=len(revisit.verification_failures),
        review_append_failure_count=len(revisit.review_append_failures),
        refused_count=len(revisit.refused),
        trimmed_count=len(revisit.trimmed),
        trimmed=tuple(
            TrimmedEntry(
                old_id=str(pair.old_id),
                new_id=str(pair.new_id) if pair.new_id else None,
            )
            for pair in revisit.trimmed
        ),
    )
    return payload, remaining


def fallback_revisit_payload(
    summary: CleanSummary, *, budget: int
) -> tuple[dict[str, object] | None, int]:
    """Plain-JSON summary of ``summary.revisit``, with its failed-batch excerpts drawn from
    ``budget``.

    Returns ``(None, budget)`` when the revisit axis returned no summary because it raised.
    Returns ``(payload, remaining_budget)`` otherwise.
    """
    payload, remaining = _revisit_fallback_payload(summary, budget=budget)
    if payload is None:
        return None, remaining

    return payload.model_dump(mode="json"), remaining


def fallback_payload(summary: CleanSummary) -> str:
    """Minimal JSON payload for a pass-history row when there is no richer report.

    Used for every outcome except a live ``"completed"`` or ``"blocked"`` pass with a
    ``build_report`` callback, dry runs included. Fields: ``watermark_before``, ``watermark_after``,
    ``pool_size``, ``reason``, ``relevance_failed_batches`` and ``falsity_failed_batches`` (empty
    when that axis did not run), ``revisit`` and ``revisit_error``. The capped failed-batch entries
    of the three axes share one ``_FALLBACK_ERROR_TOTAL_CAP`` budget, spent in the order relevance,
    falsity, revisit.
    """
    budget = _FALLBACK_ERROR_TOTAL_CAP
    relevance_failed_batches: list[FailedBatchEntry] = []
    if summary.relevance is not None:
        relevance_failed_batches, budget = _capped_entries(
            summary.relevance.failed_batches, budget=budget
        )
    falsity_failed_batches: list[FailedBatchEntry] = []
    if summary.falsity is not None:
        falsity_failed_batches, budget = _capped_entries(
            summary.falsity.failed_batches, budget=budget
        )
    revisit_payload, budget = _revisit_fallback_payload(summary, budget=budget)
    payload = FallbackPayload(
        watermark_before=summary.watermark_before.isoformat()
        if summary.watermark_before is not None
        else None,
        watermark_after=summary.watermark_after.isoformat(),
        pool_size=summary.pool_size,
        reason=summary.reason,
        relevance_failed_batches=tuple(relevance_failed_batches),
        falsity_failed_batches=tuple(falsity_failed_batches),
        revisit=revisit_payload,
        revisit_error=summary.revisit_error,
    )
    return json.dumps(payload.model_dump(mode="json"))


async def _append_pass_history(
    *,
    adapter: StorageAdapter,
    scope: Scope,
    pass_at: datetime,
    outcome: Literal["initialized", "empty_pool", "completed", "blocked"],
    dry_run: bool,
    payload: str,
) -> None:
    """Append one row to the append-only pass-history table. A failure is logged and not raised.

    Call it after the branch's ``CleanerState`` write, or its deliberate non-write, with the same
    ``pass_at``. The state write has then settled before this runs, so the broad ``except`` here
    cannot hide a real watermark-write failure as a telemetry failure.
    """
    try:
        await adapter.append_cleaner_pass(
            record=CleanerPassRecord(
                scope=scope, pass_at=pass_at, outcome=outcome, dry_run=dry_run, payload=payload
            )
        )
    except Exception:  # deliberately broad -- a telemetry write must never fail the pass
        _log.exception(
            "append_cleaner_pass failed for scope %r outcome %r (dry_run=%s) -- the pass "
            "itself already ran to completion and its watermark (if any) already landed or "
            "was deliberately left alone; this failure affects only the pass-history "
            "telemetry row",
            scope,
            outcome,
            dry_run,
        )


# ---------------------------------------------------------------------------
# Pass context and revisit axis, isolated on every outcome branch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PassContext:
    """The arguments every outcome branch and every axis call of one pass share.

    ``scope`` is the validated scope. ``rubric``, ``judge_a`` and ``judge_b`` are the resolved
    values, with the defaults from ``clean()`` already applied.
    """

    scope: Scope
    adapter: StorageAdapter
    clock: Clock
    lock: asyncio.Lock
    cfg: EngineConfig
    rubric: Rubric
    judge_a: JudgeAdapter
    judge_b: JudgeAdapter
    assessor: Assessor | None
    dry_run: bool
    embedder: Embedder
    require_scope: RequireScope


@dataclass(frozen=True)
class _PassRun:
    """What one pass that had a pool produced, before the outcome is decided."""

    relevance: RelevanceAxisSummary
    falsity: FalsityAxisSummary
    revisit: RevisitAxisSummary | None
    revisit_error: str | None
    verification_failures: tuple[HypomnemaId, ...]


async def _run_revisit_axis_isolated(
    ctx: _PassContext,
) -> tuple[RevisitAxisSummary | None, str | None]:
    """Run the revisit axis and return ``(summary, None)``, or ``(None, str(exc))`` if it raised an
    ``Exception``, which is also logged at ERROR with its traceback.

    ``clean()`` calls it from every outcome branch. The axis is a rotation over old records with no
    watermark, so gating it behind new births would starve it on a quiet scope. The caller threads
    both return values onto ``CleanSummary`` and decides nothing from them.
    """
    try:
        summary = await run_revisit_axis(
            scope=ctx.scope,
            adapter=ctx.adapter,
            clock=ctx.clock,
            lock=ctx.lock,
            cfg=ctx.cfg,
            rubric=ctx.rubric,
            judge_a=ctx.judge_a,
            judge_b=ctx.judge_b,
            assessor=ctx.assessor,
            dry_run=ctx.dry_run,
            embedder=ctx.embedder,
            require_scope=ctx.require_scope,
        )
        return summary, None
    except Exception as exc:  # broad on purpose: the revisit axis is isolated from the pass
        _log.error(
            "revisit axis failed for scope %r: %s -- isolated from this pass's own outcome and "
            "watermark",
            ctx.scope,
            exc,
            exc_info=exc,
        )
        return None, str(exc)


# ---------------------------------------------------------------------------
# The outcome branches
# ---------------------------------------------------------------------------


async def _initialize(ctx: _PassContext) -> CleanSummary:
    """Outcome ``"initialized"``: a scope with no ``CleanerState`` gets its watermark set to now.

    The watermark write is skipped under ``dry_run``. ``list_born_since`` is not called, so a first
    pass never sweeps the store. The revisit axis runs after that write or its skip, and the history
    row is appended last with the same ``now``.
    """
    now = ctx.clock.now()
    if not ctx.dry_run:
        await ctx.adapter.set_cleaner_state(
            scope=ctx.scope,
            state=CleanerState(watermark_at=now, last_pass_at=now, report=None),
        )

    revisit_summary, revisit_error = await _run_revisit_axis_isolated(ctx)
    reason = (
        (
            f"scope {ctx.scope!s} has never been judged; a real pass would initialize the "
            f"watermark to {now.isoformat()} and judge nothing (a first pass never sweeps "
            "the store) — dry run, nothing written"
        )
        if ctx.dry_run
        else (
            f"scope {ctx.scope!s} has never been judged; watermark initialized to "
            f"{now.isoformat()}, nothing judged (a first pass never sweeps the store)"
        )
    )
    summary = CleanSummary(
        scope=ctx.scope,
        outcome="initialized",
        watermark_before=None,
        watermark_after=now,
        pool_size=0,
        revisit=revisit_summary,
        revisit_error=revisit_error,
        reason=reason,
        dry_run=ctx.dry_run,
    )

    await _append_pass_history(
        adapter=ctx.adapter,
        scope=ctx.scope,
        pass_at=now,
        outcome="initialized",
        dry_run=ctx.dry_run,
        payload=fallback_payload(summary),
    )
    return summary


async def _empty_pool(ctx: _PassContext, *, since: datetime) -> CleanSummary:
    """Outcome ``"empty_pool"``: no record was born since ``since``, so the watermark stays put.

    ``now`` is only the history row's ``pass_at``. The revisit axis still runs, since a quiet scope
    can have old records due.
    """
    now = ctx.clock.now()
    revisit_summary, revisit_error = await _run_revisit_axis_isolated(ctx)
    summary = CleanSummary(
        scope=ctx.scope,
        outcome="empty_pool",
        watermark_before=since,
        watermark_after=since,
        pool_size=0,
        revisit=revisit_summary,
        revisit_error=revisit_error,
        reason=(
            f"no records born in {ctx.scope!s} since {since.isoformat()}; nothing to judge, "
            f"watermark unchanged at {since.isoformat()}"
        ),
        dry_run=ctx.dry_run,
    )

    await _append_pass_history(
        adapter=ctx.adapter,
        scope=ctx.scope,
        pass_at=now,
        outcome="empty_pool",
        dry_run=ctx.dry_run,
        payload=fallback_payload(summary),
    )
    return summary


async def _run_birth_axes(
    ctx: _PassContext, *, since: datetime, pool_limit: int
) -> tuple[RelevanceAxisSummary, FalsityAxisSummary]:
    """Run relevance, then falsity without the ids relevance shelved, contested, found unclear or
    refused.

    ``pool_limit`` is passed explicitly so the axes' re-query matches the snapshot ``clean()`` took.
    """
    relevance_summary = await run_relevance_axis(
        scope=ctx.scope,
        since=since,
        adapter=ctx.adapter,
        clock=ctx.clock,
        lock=ctx.lock,
        cfg=ctx.cfg,
        rubric=ctx.rubric,
        judge_a=ctx.judge_a,
        judge_b=ctx.judge_b,
        assessor=ctx.assessor,
        pool_limit=pool_limit,
        dry_run=ctx.dry_run,
        embedder=ctx.embedder,
        require_scope=ctx.require_scope,
    )
    barred_from_falsity = (
        frozenset(relevance_summary.shelved_ids)
        | frozenset(relevance_summary.contested_ids)
        | frozenset(relevance_summary.unclear_ids)
        | frozenset(r.id for r in relevance_summary.refused)
    )
    falsity_summary = await run_falsity_axis(
        scope=ctx.scope,
        since=since,
        adapter=ctx.adapter,
        clock=ctx.clock,
        lock=ctx.lock,
        rubric=ctx.rubric,
        judge_a=ctx.judge_a,
        judge_b=ctx.judge_b,
        embedder=ctx.embedder,
        cfg=ctx.cfg,
        assessor=ctx.assessor,
        excluded_ids=barred_from_falsity,
        pool_limit=pool_limit,
        dry_run=ctx.dry_run,
        require_scope=ctx.require_scope,
    )
    return relevance_summary, falsity_summary


async def _judge_pool(
    ctx: _PassContext,
    *,
    since: datetime,
    pool: Sequence[Hypomnema],
    pool_limit: int,
    build_report: Callable[[CleanSummary, datetime], str | None] | None,
) -> CleanSummary:
    """Outcomes ``"blocked"`` and ``"completed"``: judge the pool snapshot, verify, then decide.

    Order: relevance, falsity, revisit, then verification, which is skipped under ``dry_run``
    because nothing was written and every id would look unchanged. A failed batch or a write that
    did not verify gives ``"blocked"``, else ``"completed"``.
    """
    relevance_summary, falsity_summary = await _run_birth_axes(
        ctx, since=since, pool_limit=pool_limit
    )
    revisit_summary, revisit_error = await _run_revisit_axis_isolated(ctx)
    verification_failures: tuple[HypomnemaId, ...] = (
        ()
        if ctx.dry_run
        else await _verify_writes(
            adapter=ctx.adapter,
            shelved_ids=relevance_summary.shelved_ids,
            edges=falsity_summary.edges,
        )
    )
    run = _PassRun(
        relevance=relevance_summary,
        falsity=falsity_summary,
        revisit=revisit_summary,
        revisit_error=revisit_error,
        verification_failures=verification_failures,
    )

    failed_batch_count = len(relevance_summary.failed_batches) + len(falsity_summary.failed_batches)
    if failed_batch_count or verification_failures:
        return await _finish_blocked(
            ctx, run=run, since=since, pool_size=len(pool), build_report=build_report
        )
    return await _finish_completed(ctx, run=run, since=since, pool=pool, build_report=build_report)


async def _finish_blocked(
    ctx: _PassContext,
    *,
    run: _PassRun,
    since: datetime,
    pool_size: int,
    build_report: Callable[[CleanSummary, datetime], str | None] | None,
) -> CleanSummary:
    """Write the ``"blocked"`` outcome: the watermark stays at ``since``.

    ``now`` is read after the summary is built, because the history row needs a ``pass_at`` either
    way. With ``build_report`` and no ``dry_run`` one ``CleanerState`` write keeps ``watermark_at``
    at ``since`` and stores the report. Otherwise no ``CleanerState`` is written. The history row
    carries the report when one was built, else the fallback payload.
    """
    failed_relevance = len(run.relevance.failed_batches)
    failed_falsity = len(run.falsity.failed_batches)
    reasons: list[str] = []
    if failed_relevance + failed_falsity:
        reasons.append(
            f"{failed_relevance + failed_falsity} batch(es) failed "
            f"({failed_relevance} relevance, {failed_falsity} falsity)"
        )
    if run.verification_failures:
        reasons.append(f"{len(run.verification_failures)} write(s) did not verify on re-read")
    summary = CleanSummary(
        scope=ctx.scope,
        outcome="blocked",
        watermark_before=since,
        watermark_after=since,
        pool_size=pool_size,
        relevance=run.relevance,
        falsity=run.falsity,
        verification_failures=run.verification_failures,
        revisit=run.revisit,
        revisit_error=run.revisit_error,
        reason=f"{'; '.join(reasons)}; watermark unchanged at {since.isoformat()}",
        dry_run=ctx.dry_run,
    )

    now = ctx.clock.now()
    report: str | None = None
    if build_report is not None and not ctx.dry_run:
        report = build_report(summary, now)
        await ctx.adapter.set_cleaner_state(
            scope=ctx.scope,
            state=CleanerState(watermark_at=since, last_pass_at=now, report=report),
        )

    await _append_pass_history(
        adapter=ctx.adapter,
        scope=ctx.scope,
        pass_at=now,
        outcome="blocked",
        dry_run=ctx.dry_run,
        payload=report if report is not None else fallback_payload(summary),
    )
    return summary


async def _finish_completed(
    ctx: _PassContext,
    *,
    run: _PassRun,
    since: datetime,
    pool: Sequence[Hypomnema],
    build_report: Callable[[CleanSummary, datetime], str | None] | None,
) -> CleanSummary:
    """Write the ``"completed"`` outcome: the watermark advances to the newest ``created_at`` in
    ``pool``, never to ``clock.now()``.

    ``now`` is read after the new watermark is computed and is shared by ``build_report`` and
    ``last_pass_at``. Under ``dry_run`` the callback is not called and no ``CleanerState`` is
    written. The history row carries the report when one was built, else the fallback payload.
    """
    new_watermark = _newest_created_at(pool)
    now = ctx.clock.now()
    shelve_count = run.relevance.counts.shelve
    replace_count = run.falsity.counts.replace
    reason = (
        (
            f"pass would complete (dry run, nothing written); watermark would advance from "
            f"{since.isoformat()} to {new_watermark.isoformat()} "
            f"({shelve_count} would shelve, {replace_count} would supersede)"
        )
        if ctx.dry_run
        else (
            f"pass completed; watermark advanced from {since.isoformat()} to "
            f"{new_watermark.isoformat()} ({shelve_count} shelved, {replace_count} superseded)"
        )
    )
    summary = CleanSummary(
        scope=ctx.scope,
        outcome="completed",
        watermark_before=since,
        watermark_after=new_watermark,
        pool_size=len(pool),
        relevance=run.relevance,
        falsity=run.falsity,
        revisit=run.revisit,
        revisit_error=run.revisit_error,
        reason=reason,
        dry_run=ctx.dry_run,
    )

    report: str | None = None
    if not ctx.dry_run:
        report = build_report(summary, now) if build_report is not None else None
        await ctx.adapter.set_cleaner_state(
            scope=ctx.scope,
            state=CleanerState(watermark_at=new_watermark, last_pass_at=now, report=report),
        )

    await _append_pass_history(
        adapter=ctx.adapter,
        scope=ctx.scope,
        pass_at=now,
        outcome="completed",
        dry_run=ctx.dry_run,
        payload=report if report is not None else fallback_payload(summary),
    )
    return summary


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


async def clean(
    *,
    scope: str,
    adapter: StorageAdapter,
    embedder: Embedder,
    clock: Clock,
    lock: asyncio.Lock,
    cfg: EngineConfig,
    require_scope: RequireScope,
    rubric: Rubric | None = None,
    judge_a: JudgeAdapter | None = None,
    judge_b: JudgeAdapter | None = None,
    assessor: Assessor | None = None,
    pool_limit: int | None = None,
    build_report: Callable[[CleanSummary, datetime], str | None] | None = None,
    dry_run: bool = False,
) -> CleanSummary:
    """Run one Cleaner pass over ``scope``. The module docstring has the sequence.

    Args:
        scope: Target scope, one per call. ``"global"`` runs like any other.
        embedder, cfg: Forwarded to the axes.
        clock: Time source, including the initial watermark of an unjudged scope.
        lock: The engine's write lock. The axes take it around each write; this function holds none.
        assessor: Forwarded to the three axes. ``None`` runs them without an Assessor.
        require_scope: Callback that validates scope registration.
        rubric: Defaults to ``load_rubric()``.
        judge_a, judge_b: Default to ``build_judge_adapter`` over ``rubric.judges.seat_a`` and
            ``seat_b``.
        pool_limit: Ceiling on this function's own ``list_born_since`` snapshot, passed on to both
            axes. ``None`` reads ``rubric.parameters.pool_limit``.
        build_report: The callback described in the module docstring.
        dry_run: Writes nothing apart from the history row; see the module docstring.

    Raises:
        ToolError: ``scope`` fails validation or is not registered.
    """
    typed_scope = validate_and_resolve_scope(scope)
    await require_scope(typed_scope)

    resolved_rubric = rubric if rubric is not None else load_rubric()
    ctx = _PassContext(
        scope=typed_scope,
        adapter=adapter,
        clock=clock,
        lock=lock,
        cfg=cfg,
        rubric=resolved_rubric,
        judge_a=judge_a
        if judge_a is not None
        else build_judge_adapter(resolved_rubric.judges.seat_a),
        judge_b=judge_b
        if judge_b is not None
        else build_judge_adapter(resolved_rubric.judges.seat_b),
        assessor=assessor,
        dry_run=dry_run,
        embedder=embedder,
        require_scope=require_scope,
    )
    effective_pool_limit = (
        pool_limit if pool_limit is not None else resolved_rubric.parameters.pool_limit
    )

    state = await adapter.get_cleaner_state(scope=typed_scope)
    if state is None:
        return await _initialize(ctx)

    since = state.watermark_at
    pool = await adapter.list_born_since(scope=typed_scope, since=since, limit=effective_pool_limit)
    if not pool:
        return await _empty_pool(ctx, since=since)

    return await _judge_pool(
        ctx,
        since=since,
        pool=pool,
        pool_limit=effective_pool_limit,
        build_report=build_report,
    )
