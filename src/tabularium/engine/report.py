"""The Cleaner's pass report: what a pass persists, and the bounded rendering of it for
``context()``.

- Persist: ``run_clean_pass`` calls ``clean()`` with a ``build_report`` callback. ``clean()``
  invokes it once, where it is about to write ``CleanerState`` for a ``"completed"`` or
  ``"blocked"`` outcome, and stores the returned JSON in that same ``set_cleaner_state`` write. The
  report and the watermark therefore land together, and the callback receives the same ``now`` that
  becomes ``last_pass_at``. ``run_clean_pass`` and ``build_report`` never call
  ``set_cleaner_state``; only ``record_pass_failure`` does, for a crashed pass. ``clean.py`` never
  learns what a ``CleanerReport`` looks like. ``CleanerState`` holds one row per scope, overwritten
  by each write.
- Surface: ``render_context_summary`` turns a ``CleanerReport`` into the ``CleanerContextSummary``
  that ``context()`` embeds: a summary line plus an owner-attention list capped at
  ``rubric.parameters.report_render_cap``. The cap exists because the summary competes with
  grounding for the session-start budget. The full report stays in the store.
  ``load_context_summary`` reads the newest row of the append-only pass history
  (``list_cleaner_passes(scope=..., limit=...)``) and ``render_pass_record`` renders whichever of
  the five outcomes that row carries.

The persisted models live in ``engine/report_schema.py`` and are re-exported here.

Only ``"completed"`` and ``"blocked"`` outcomes get a ``CleanerReport``. ``"initialized"`` and
``"empty_pool"`` have no judged axis to report on, and ``clean()`` does not call ``build_report``
for them. When a callback is supplied, a ``"blocked"`` pass still writes a report without moving the
watermark, so a failed pass is distinguishable from one that judged nothing.

NOTE: ``CleanerState.report`` is written on each pass and has no production reader; ``context()``
reads the pass history.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import Scope
from tabularium.embedding import Embedder
from tabularium.engine._types import RequireScope
from tabularium.engine.assessor import Assessor
from tabularium.engine.axis_common import ContestedVerdict
from tabularium.engine.clean import FALLBACK_ERROR_EXCERPT_CAP, CleanSummary, clean
from tabularium.engine.falsity import FalsityContestedPair
from tabularium.engine.judge import JudgeAdapter
from tabularium.engine.pass_payloads import (
    ErroredPayload,
    FailedBatchEntry,
    FallbackPayload,
    OmittedMarker,
    PayloadKind,
)
from tabularium.engine.report_schema import (
    CleanerBatchFailureReport as CleanerBatchFailureReport,
)
from tabularium.engine.report_schema import (
    CleanerFailureReport as CleanerFailureReport,
)
from tabularium.engine.report_schema import (
    CleanerReport as CleanerReport,
)
from tabularium.engine.report_schema import (
    FalsityReportSection as FalsityReportSection,
)
from tabularium.engine.report_schema import (
    RelevanceReportSection as RelevanceReportSection,
)
from tabularium.engine.report_schema import (
    RevisitReportSection as RevisitReportSection,
)
from tabularium.engine.rubric import Rubric, load_rubric
from tabularium.results import CleanerContextSummary
from tabularium.storage.types import CleanerPassRecord, CleanerState, StorageAdapter

_log = logging.getLogger("tabularium")


def _capped_persisted_error(error: str, *, cap: int = FALLBACK_ERROR_EXCERPT_CAP) -> str:
    """Tail-truncate one batch failure's ``error`` to ``cap`` characters before it is persisted in a
    ``CleanerBatchFailureReport``.

    This bounds each item's size, not how many items there are. The tail is kept because the live
    failure line sits at the end of an echoed or joined diagnostic. It reuses
    ``FALLBACK_ERROR_EXCERPT_CAP``. ``_truncate_why`` is render-only and keeps the head, which is
    wrong for a diagnostic.
    """
    return error[-cap:]


def build_report(summary: CleanSummary, *, pass_at: datetime) -> CleanerReport | None:
    """Turn one ``CleanSummary`` into a ``CleanerReport``, or ``None`` for ``"initialized"`` and
    ``"empty_pool"``.

    A ``"completed"`` or ``"blocked"`` summary must carry both ``relevance`` and ``falsity``
    (asserted). A missing revisit summary leaves ``revisit_error`` in place of a section; neither
    touches ``outcome``.
    """
    if summary.outcome not in ("completed", "blocked"):
        return None

    relevance = summary.relevance
    falsity = summary.falsity
    assert relevance is not None, "a completed/blocked pass always carries a relevance summary"
    assert falsity is not None, "a completed/blocked pass always carries a falsity summary"

    relevance_failures = tuple(
        CleanerBatchFailureReport(
            axis="relevance",
            batch_index=f.batch_index,
            record_ids=f.record_ids,
            judge_a_outcome=f.judge_a_outcome,
            judge_b_outcome=f.judge_b_outcome,
            error=_capped_persisted_error(f.error),
        )
        for f in relevance.failed_batches
    )
    falsity_failures = tuple(
        CleanerBatchFailureReport(
            axis="falsity",
            batch_index=f.batch_index,
            record_ids=tuple(record_id for pair in f.pairs for record_id in pair),
            judge_a_outcome=f.judge_a_outcome,
            judge_b_outcome=f.judge_b_outcome,
            error=_capped_persisted_error(f.error),
        )
        for f in falsity.failed_batches
    )

    # Revisit is isolated: a `None` summary (the axis raised) becomes `revisit_error`, not an
    # empty section, and neither branch touches `outcome`, which reflects the birth axes alone.
    revisit_section: RevisitReportSection | None = None
    if summary.revisit is not None:
        revisit = summary.revisit
        revisit_failures = tuple(
            CleanerBatchFailureReport(
                axis="revisit",
                batch_index=f.batch_index,
                record_ids=f.record_ids,
                judge_a_outcome=f.judge_a_outcome,
                judge_b_outcome=f.judge_b_outcome,
                error=_capped_persisted_error(f.error),
            )
            for f in revisit.failed_batches
        )
        revisit_section = RevisitReportSection(
            judge_pair=revisit.judge_pair,
            rubric_version=revisit.rubric_version,
            counts=revisit.counts,
            shelved_ids=revisit.shelved_ids,
            unclear_ids=revisit.unclear_ids,
            contested_ids=revisit.contested_ids,
            contested_verdicts=revisit.contested_verdicts,
            protected_count=len(revisit.protected_ids),
            skipped_no_evidence_count=len(revisit.skipped_no_evidence_ids),
            skipped_too_large_count=len(revisit.skipped_too_large_ids),
            failures=revisit_failures,
            calls=revisit.calls,
            verification_failures=revisit.verification_failures,
            review_append_failures=revisit.review_append_failures,
            refused=revisit.refused,
            judgment_append_failures=revisit.judgment_append_failures,
            pool_size=revisit.pool_size,
            split_keep=revisit.split_keep,
            trimmed=revisit.trimmed,
        )

    return CleanerReport(
        scope=summary.scope,
        pass_at=pass_at,
        outcome=summary.outcome,
        relevance=RelevanceReportSection(
            judge_pair=relevance.judge_pair,
            rubric_version=relevance.rubric_version,
            counts=relevance.counts,
            shelved_ids=relevance.shelved_ids,
            unclear_ids=relevance.unclear_ids,
            contested_ids=relevance.contested_ids,
            contested_verdicts=relevance.contested_verdicts,
            failures=relevance_failures,
            calls=relevance.calls,
            refused=relevance.refused,
            judgment_append_failures=relevance.judgment_append_failures,
            skipped=relevance.skipped,
            split_keep=relevance.split_keep,
            split_annotated=relevance.split_annotated,
            split_shelve=relevance.split_shelve,
            trimmed=relevance.trimmed,
        ),
        falsity=FalsityReportSection(
            judge_pair=falsity.judge_pair,
            rubric_version=falsity.rubric_version,
            counts=falsity.counts,
            superseded_ids=tuple(edge.id for edge in falsity.edges),
            global_contradicted=falsity.global_candidates,
            decision_contradicted=falsity.decision_candidates,
            contested=tuple(
                (pair.newcomer_id, pair.candidate_id) for pair in falsity.contested_pairs
            ),
            contested_verdicts=falsity.contested_pairs,
            failures=falsity_failures,
            calls=falsity.calls,
            refused=falsity.refused,
            judgment_append_failures=falsity.judgment_append_failures,
            split_supersede=falsity.split_supersede,
            split_coexist=falsity.split_coexist,
            reason_by_assessor=falsity.reason_by_assessor,
            merged=falsity.merged,
            merge_reverted=falsity.merge_reverted,
            merge_revert_failed=falsity.merge_revert_failed,
        ),
        revisit=revisit_section,
        revisit_error=summary.revisit_error,
        verification_failures=summary.verification_failures,
    )


# ---------------------------------------------------------------------------
# The persist step: compose clean() with a report write
# ---------------------------------------------------------------------------


async def run_clean_pass(
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
    dry_run: bool = False,
) -> CleanSummary:
    """Run ``engine.clean.clean`` with a ``build_report`` callback that serializes this module's
    ``build_report``, so the report is persisted by ``clean()``'s single ``set_cleaner_state``
    write.

    Takes the arguments of ``clean()`` except ``build_report``. ``dry_run`` is forwarded, and
    ``clean()`` does not invoke the callback under it. This function writes nothing itself and
    returns the ``CleanSummary`` from ``clean()``.
    """

    def _build_report_json(summary: CleanSummary, pass_at: datetime) -> str | None:
        """Adapt ``build_report`` to the ``(CleanSummary, datetime) -> str | None`` callback
        ``clean()`` takes.
        """
        report = build_report(summary, pass_at=pass_at)
        return report.model_dump_json() if report is not None else None

    return await clean(
        scope=scope,
        adapter=adapter,
        embedder=embedder,
        clock=clock,
        lock=lock,
        cfg=cfg,
        require_scope=require_scope,
        rubric=rubric,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        pool_limit=pool_limit,
        build_report=_build_report_json,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# The surface step: a bounded rendering for context()
# ---------------------------------------------------------------------------

# Cap on one judge's `why` in a contested owner-attention line. Applied at render time only,
# never at persistence. The `why` fields have a minimum length and no maximum, and the
# session-start grounding is budget-packed, so a contested line has to stay bounded.
_CONTESTED_WHY_RENDER_CAP = 160


def _truncate_why(why: str, *, cap: int = _CONTESTED_WHY_RENDER_CAP) -> str:
    """Render ``why`` as one line of at most ``cap`` characters, including the ellipsis.

    Whitespace runs, newlines included, collapse to single spaces so an owner-attention item stays
    on one line. Render-only: the full ``why`` is what gets persisted.
    """
    normalized = " ".join(why.split())
    if len(normalized) <= cap:
        return normalized
    return normalized[: cap - 3].rstrip() + "..."


def _render_contested_line(
    *,
    label: str,
    subject: str,
    judge_a_identity: str,
    judge_a_grade: str,
    judge_a_why: str,
    judge_b_identity: str,
    judge_b_grade: str,
    judge_b_why: str,
) -> str:
    """Render one contested record or pair as a single owner-attention line: both judges' identity,
    grade and truncated ``why``. The three axes share this shape under different field names, so
    callers pass ``*_grade`` or, for falsity, ``*_verdict`` as the grade.
    """
    return (
        f"{label}: {subject} — "
        f'{judge_a_identity} graded {judge_a_grade} ("{_truncate_why(judge_a_why)}"), '
        f'{judge_b_identity} graded {judge_b_grade} ("{_truncate_why(judge_b_why)}")'
    )


def _truncate_error_tail(error: str, *, cap: int = _CONTESTED_WHY_RENDER_CAP) -> str:
    """Render a batch-failure error as one line of at most ``cap`` characters, keeping the tail
    behind a leading ``...``.

    The persisted error is already tail-capped, so a head-keeping truncation would cut the end that
    carries the failure. Whitespace is collapsed first.
    """
    normalized = " ".join(error.split())
    if len(normalized) <= cap:
        return normalized
    return "..." + normalized[-(cap - 3) :]


def _render_batch_failure_line(failure: CleanerBatchFailureReport) -> str:
    """One failed-batch owner-attention line with ``failure.error`` cut by ``_truncate_error_tail``.

    Truncating at render time also bounds rows persisted without a persistence-time cap.
    """
    return (
        f"{failure.axis} batch {failure.batch_index} failed: "
        f"{_truncate_error_tail(failure.error)}"
    )


def _contested_items(
    *,
    label: str,
    subjects: Sequence[str],
    verdicts: Mapping[str, ContestedVerdict | FalsityContestedPair],
) -> list[str]:
    """One owner-attention line per contested subject, in ``subjects`` order.

    A subject with an entry in ``verdicts`` renders both judges' identity, grade and ``why``; a
    subject without one renders as ``label: subject``, which is how a row persisted without
    ``contested_verdicts`` reads. A falsity pair's grade is its ``*_verdict`` field.
    """
    items: list[str] = []
    for subject in subjects:
        verdict = verdicts.get(subject)
        match verdict:
            case None:
                items.append(f"{label}: {subject}")
            case ContestedVerdict():
                items.append(
                    _render_contested_line(
                        label=label,
                        subject=subject,
                        judge_a_identity=verdict.judge_a_identity,
                        judge_a_grade=verdict.judge_a_grade,
                        judge_a_why=verdict.judge_a_why,
                        judge_b_identity=verdict.judge_b_identity,
                        judge_b_grade=verdict.judge_b_grade,
                        judge_b_why=verdict.judge_b_why,
                    )
                )
            case FalsityContestedPair():
                items.append(
                    _render_contested_line(
                        label=label,
                        subject=subject,
                        judge_a_identity=verdict.judge_a_identity,
                        judge_a_grade=verdict.judge_a_verdict,
                        judge_a_why=verdict.judge_a_why,
                        judge_b_identity=verdict.judge_b_identity,
                        judge_b_grade=verdict.judge_b_verdict,
                        judge_b_why=verdict.judge_b_why,
                    )
                )
    return items


def _revisit_items(revisit: RevisitReportSection) -> list[str]:
    """The revisit section's owner-attention lines: contested records, failed batches, writes that
    did not verify, review rows that failed to append and refused shelves, in that order.
    """
    items = _contested_items(
        label="revisit contested (judges disagreed, no write)",
        subjects=[str(record_id) for record_id in revisit.contested_ids],
        verdicts={str(v.id): v for v in revisit.contested_verdicts},
    )

    for failure in revisit.failures:
        items.append(_render_batch_failure_line(failure))

    for record_id in revisit.verification_failures:
        items.append(f"revisit write did not verify on re-read: {record_id}")

    for append_failure in revisit.review_append_failures:
        items.append(
            f"revisit review row failed to append (write itself landed): "
            f"{append_failure.target_id} — {append_failure.error}"
        )

    for refusal in revisit.refused:
        items.append(
            f"revisit shelve refused (voted shelve, not written): {refusal.id} — {refusal.reason}"
        )

    return items


def _owner_attention_items(report: CleanerReport) -> list[str]:
    """Every owner-attention item across the report, uncapped, over all three axes: unclear,
    contested, refused, global and Decision contradictions, failed batches and writes that did not
    verify.

    A contested item shows both judges' grades when a ``contested_verdicts`` entry exists for it,
    else only the id. ``render_context_summary`` applies the cap.
    """
    items: list[str] = []

    for record_id in report.relevance.unclear_ids:
        items.append(f"unclear (panel declined to judge): {record_id}")

    items.extend(
        _contested_items(
            label="contested (judges disagreed, no write)",
            subjects=[str(record_id) for record_id in report.relevance.contested_ids],
            verdicts={str(v.id): v for v in report.relevance.contested_verdicts},
        )
    )

    for refusal in report.relevance.refused:
        items.append(f"shelve refused (voted shelve, not written): {refusal.id} — {refusal.reason}")

    items.extend(
        _contested_items(
            label="falsity contested (judges disagreed, no edge drawn)",
            subjects=[f"{n} vs {c}" for n, c in report.falsity.contested],
            verdicts={
                f"{v.newcomer_id} vs {v.candidate_id}": v for v in report.falsity.contested_verdicts
            },
        )
    )

    for pair_refusal in report.falsity.refused:
        items.append(
            f"supersede refused (voted replace, not written): "
            f"{pair_refusal.newcomer_id} vs {pair_refusal.candidate_id} — {pair_refusal.reason}"
        )

    for entry in report.falsity.global_contradicted:
        pick_suffix = (
            f" (assessor: {entry.assessor_pick})" if entry.assessor_pick is not None else ""
        )
        items.append(
            f"global contradicted, detected not acted on: "
            f"{entry.newcomer_id} vs {entry.candidate_id}{pick_suffix}"
        )
    for entry in report.falsity.decision_contradicted:
        items.append(
            f"Decision contradicted, detected not acted on: "
            f"{entry.newcomer_id} vs {entry.candidate_id}"
        )

    if report.revisit is not None:
        items.extend(_revisit_items(report.revisit))
    if report.revisit_error is not None:
        items.append(
            f"revisit axis errored (isolated from this pass's outcome): {report.revisit_error}"
        )

    for failure in (*report.relevance.failures, *report.falsity.failures):
        items.append(_render_batch_failure_line(failure))

    for record_id in report.verification_failures:
        items.append(f"write did not verify on re-read: {record_id}")

    return items


def _cap_owner_attention_items(items: Sequence[str], *, cap: int) -> tuple[str, ...]:
    """The first ``cap`` of ``items``, plus an ``"...and N more"`` marker when any were dropped.

    Shared by the rich-report path and both fallback renderers.
    """
    shown = tuple(items[:cap])
    overflow = len(items) - len(shown)
    if overflow > 0:
        shown = (*shown, f"...and {overflow} more")
    return shown


def render_context_summary(
    report: CleanerReport | CleanerFailureReport, *, cap: int
) -> CleanerContextSummary:
    """Render ``report`` to the ``CleanerContextSummary`` ``context()`` embeds.

    A ``CleanerFailureReport`` renders to ``outcome="errored"`` with a single item. For a
    ``CleanerReport``, ``owner_attention`` is capped at ``cap`` entries with an overflow marker.
    """
    if isinstance(report, CleanerFailureReport):
        return CleanerContextSummary(
            scope=report.scope,
            pass_at=report.pass_at,
            outcome="errored",
            summary_line=f"Cleaner errored at {report.pass_at.isoformat()}: {report.error}",
            owner_attention=(f"pass failed: {report.error}",),
        )

    shown = _cap_owner_attention_items(_owner_attention_items(report), cap=cap)

    summary_line = (
        f"Cleaner {report.outcome} at {report.pass_at.isoformat()}: "
        f"{report.relevance.counts.shelve} shelved, "
        f"{report.falsity.counts.replace} superseded, "
        f"{report.relevance.counts.unclear} unclear, "
        f"{report.falsity.counts.coexist} coexist, "
        f"{report.relevance.counts.contested} relevance-contested, "
        f"{report.falsity.counts.contested} falsity-contested"
    )
    # Revisit is appended to the birth-axes clause, never replacing it: `report.outcome` in the
    # preamble describes the birth axes alone.
    if report.revisit is not None:
        summary_line += (
            f"; revisit: {report.revisit.counts.shelve} shelved, "
            f"{report.revisit.counts.contested} contested"
        )
    if report.revisit_error is not None:
        summary_line += "; revisit axis errored"

    # Band counts cannot tell "every batch failed" from "nothing needed action". The attention list
    # also itemizes failed batches but is capped, so this suffix is the always-visible signal.
    failed_batch_count = (
        len(report.relevance.failures)
        + len(report.falsity.failures)
        + (len(report.revisit.failures) if report.revisit is not None else 0)
    )
    if failed_batch_count:
        summary_line += f"; {failed_batch_count} batch(es) failed"

    return CleanerContextSummary(
        scope=report.scope,
        pass_at=report.pass_at,
        outcome=report.outcome,
        summary_line=summary_line,
        owner_attention=tuple(shown),
    )


# ---------------------------------------------------------------------------
# render_pass_record: render the newest pass-history row, whatever its outcome
# ---------------------------------------------------------------------------

_DRY_RUN_MARKER = "[DRY RUN] "


def _render_errored(record: CleanerPassRecord) -> CleanerContextSummary:
    """Render an ``"errored"`` row. ``record_pass_failure`` writes its payload as an
    ``ErroredPayload``.
    """
    payload = ErroredPayload.model_validate_json(record.payload)
    return CleanerContextSummary(
        scope=record.scope,
        pass_at=record.pass_at,
        outcome="errored",
        summary_line=f"Cleaner errored at {record.pass_at.isoformat()}: {payload.error}",
        owner_attention=(f"pass failed: {payload.error}",),
    )


def _render_fallback_failed_batches(
    entries: Sequence[FailedBatchEntry], *, axis_label: str
) -> list[str]:
    """Render capped batch-failure entries from a fallback payload (``relevance_failed_batches``,
    ``falsity_failed_batches`` or the revisit payload's ``failed_batches``) as owner-attention
    lines.

    An ``OmittedMarker``, written when the payload's error-text budget ran out, renders one line.
    """
    items: list[str] = []
    for entry in entries:
        if isinstance(entry, OmittedMarker):
            items.append(
                f"{axis_label}: {entry.omitted_count} more batch failure(s) omitted "
                "(error-text cap reached)"
            )
            continue

        items.append(
            f"{axis_label} batch {entry.batch_index} failed "
            f"(judge_a={entry.judge_a_outcome}, judge_b={entry.judge_b_outcome}): {entry.error}"
        )
    return items


def _render_revisit_fallback(payload: FallbackPayload) -> tuple[str, tuple[str, ...]]:
    """Render the revisit data in a fallback payload as a ``summary_line`` suffix and
    owner-attention lines. Returns ``("", ())`` when the payload has neither ``revisit`` nor
    ``revisit_error``.

    Used by ``_render_uneventful`` and ``_render_judged``'s fallback branch. The failed batch count
    goes in the suffix. Each failed batch gets a line with its error text; a payload without
    ``failed_batches`` gets a count-only line.
    """
    revisit = payload.revisit
    suffix = ""
    items: list[str] = []
    if revisit is not None:
        suffix = f"; revisit: {revisit.counts.shelve} shelved, {revisit.counts.contested} contested"
        for record_id in revisit.contested_ids:
            items.append(f"revisit contested (judges disagreed, no write): {record_id}")
        failed = revisit.failed_batch_count
        if failed:
            suffix += f"; {failed} batch(es) failed"
        failed_batch_items = _render_fallback_failed_batches(
            revisit.failed_batches, axis_label="revisit"
        )
        if failed_batch_items:
            items.extend(failed_batch_items)
        elif failed:
            items.append(f"revisit: {failed} batch(es) failed")
        if revisit.verification_failure_count:
            items.append(
                f"revisit: {revisit.verification_failure_count} write(s) did not verify on re-read"
            )
        if revisit.review_append_failure_count:
            items.append(
                f"revisit: {revisit.review_append_failure_count} review row(s) failed to append"
            )
        if revisit.refused_count:
            items.append(
                f"revisit: {revisit.refused_count} shelve write(s) refused (stale selection)"
            )
    if payload.revisit_error is not None:
        suffix += "; revisit axis errored"
        items.append(
            f"revisit axis errored (isolated from this pass's outcome): {payload.revisit_error}"
        )
    return suffix, tuple(items)


def _render_uneventful(record: CleanerPassRecord, *, cap: int) -> CleanerContextSummary:
    """Render an ``"initialized"`` or ``"empty_pool"`` row.

    No birth axis ran, so ``payload`` is ``fallback_payload``'s shape. The revisit axis does run on
    both outcomes, and its activity is rendered by ``_render_revisit_fallback``. The outcome word is
    in ``summary_line`` so these read differently from ``"blocked"``. ``owner_attention`` is capped
    by ``cap``.
    """
    payload = FallbackPayload.model_validate_json(record.payload)
    revisit_suffix, revisit_items = _render_revisit_fallback(payload)
    return CleanerContextSummary(
        scope=record.scope,
        pass_at=record.pass_at,
        outcome=record.outcome,
        summary_line=f"Cleaner {record.outcome} at {record.pass_at.isoformat()}: {payload.reason}"
        f"{revisit_suffix}",
        owner_attention=_cap_owner_attention_items(revisit_items, cap=cap),
    )


def _render_judged(record: CleanerPassRecord, *, cap: int) -> CleanerContextSummary:
    """Render a ``"completed"`` or ``"blocked"`` row.

    A payload with ``kind == "report"`` is a ``CleanerReport`` and renders through
    ``render_context_summary``. Any other payload is ``fallback_payload``'s shape, from a dry run or
    a ``clean()`` call with no ``build_report``. It renders from ``reason``, the revisit data and,
    on this branch only, each birth axis's capped failed-batch entries. ``owner_attention`` is
    capped by ``cap``.
    """
    if PayloadKind.model_validate_json(record.payload).kind == "report":
        report = CleanerReport.model_validate(json.loads(record.payload))
        return render_context_summary(report, cap=cap)

    payload = FallbackPayload.model_validate_json(record.payload)
    revisit_suffix, revisit_items = _render_revisit_fallback(payload)
    relevance_items = _render_fallback_failed_batches(
        payload.relevance_failed_batches, axis_label="relevance"
    )
    falsity_items = _render_fallback_failed_batches(
        payload.falsity_failed_batches, axis_label="falsity"
    )
    combined_items = (*relevance_items, *falsity_items, *revisit_items)
    return CleanerContextSummary(
        scope=record.scope,
        pass_at=record.pass_at,
        outcome=record.outcome,
        summary_line=f"Cleaner {record.outcome} at {record.pass_at.isoformat()}: {payload.reason}"
        f"{revisit_suffix}",
        owner_attention=_cap_owner_attention_items(combined_items, cap=cap),
    )


def render_pass_record(record: CleanerPassRecord, *, cap: int) -> CleanerContextSummary:
    """Render one pass-history row; the entry point of ``load_context_summary``.

    Dispatches on ``record.outcome`` rather than a ``kind`` field, because ``"initialized"`` and
    ``"empty_pool"`` carry no ``CleanerReport`` and ``"errored"`` carries a smaller payload. The
    newest row renders whatever its outcome. A dry run gets ``"[DRY RUN] "`` prefixed to
    ``summary_line`` for every outcome.
    """
    if record.outcome == "errored":
        rendered = _render_errored(record)
    elif record.outcome in ("initialized", "empty_pool"):
        rendered = _render_uneventful(record, cap=cap)
    else:
        rendered = _render_judged(record, cap=cap)

    if not record.dry_run:
        return rendered
    return rendered.model_copy(update={"summary_line": f"{_DRY_RUN_MARKER}{rendered.summary_line}"})


async def load_context_summary(
    adapter: StorageAdapter, *, scope: Scope, rubric: Rubric | None = None
) -> CleanerContextSummary | None:
    """Render ``scope``'s newest pass-history row, or return ``None`` when the scope has no pass
    history, live or dry. This is the read path ``context()`` uses.

    The newest row renders whatever its outcome, so ``"initialized"`` and ``"empty_pool"`` render
    too: a run of empty passes means the trigger stopped firing. When the newest non-dry rows are
    ``blocked`` or ``errored``, the result carries ``blocked_streak`` and ``blocked_since``.
    """
    rows = await adapter.list_cleaner_passes(scope=scope, limit=_STREAK_LOOKBACK)
    if not rows:
        return None

    resolved_rubric = rubric if rubric is not None else load_rubric()
    rendered = render_pass_record(rows[0], cap=resolved_rubric.parameters.report_render_cap)
    streak, since = blocked_streak(rows)
    if streak == 0:
        return rendered

    return rendered.model_copy(update={"blocked_streak": streak, "blocked_since": since})


_STREAK_LOOKBACK = 200


def blocked_streak(rows: Sequence[CleanerPassRecord]) -> tuple[int, datetime | None]:
    """Count the consecutive newest ``blocked`` or ``errored`` rows of one scope's pass history,
    skipping dry runs. ``rows`` is newest first. Returns the count and the ``pass_at`` of the oldest
    row in the streak, or ``None`` when the count is 0.
    """
    streak = 0
    since: datetime | None = None
    for record in rows:
        if record.dry_run:
            continue
        if record.outcome not in ("blocked", "errored"):
            break
        streak += 1
        since = record.pass_at

    return streak, since


# ---------------------------------------------------------------------------
# Failure persistence: a pass that crashed before clean() wrote its own report
# ---------------------------------------------------------------------------


async def record_pass_failure(
    *, scope: str, adapter: StorageAdapter, clock: Clock, error: Exception
) -> None:
    """Persist that a Cleaner pass crashed for ``scope``. ``Tabularium._run_cleaner_pass`` calls it
    from its ``except`` block, next to the ERROR log line.

    - It never invents a watermark. When the scope has a ``CleanerState``, a new one is written with
      ``watermark_at`` unchanged, ``last_pass_at`` set to now and a ``CleanerFailureReport`` as
      ``report``. When none exists, no state is written: a scope that was never judged stays
      uninitialised, so its next pass is a genuine first pass.
    - It appends an ``"errored"`` pass-history row in a separate guarded block, whether or not a
      state was written. The row needs no watermark. Its payload is ``{"error": str(error)}`` and
      ``dry_run=False``.
    - A storage ``Exception`` in either block is logged at ERROR and not raised. The failure being
      recorded is often the adapter itself. ``CancelledError`` and a failing ``clock.now()``
      propagate.
    """
    now = clock.now()
    typed_scope: Scope | None = None
    try:
        typed_scope = Scope(scope)
        existing = await adapter.get_cleaner_state(scope=typed_scope)
        if existing is not None:
            failure = CleanerFailureReport(scope=typed_scope, pass_at=now, error=str(error))
            await adapter.set_cleaner_state(
                scope=typed_scope,
                state=CleanerState(
                    watermark_at=existing.watermark_at,
                    last_pass_at=now,
                    report=failure.model_dump_json(),
                ),
            )
        # No prior state: no write, since any CleanerState would have to invent a watermark. The
        # pass-history append below still runs.
    except Exception as persist_exc:  # broad on purpose: recording a failure must not raise
        _log.error(
            "record_pass_failure: failed to persist a failure report for scope %r "
            "(original pass error: %s); persist error: %s -- the ERROR log line already "
            "written for the original error is now the only surviving record",
            scope,
            error,
            persist_exc,
            exc_info=persist_exc,
        )

    try:
        history_scope = typed_scope if typed_scope is not None else Scope(scope)
        await adapter.append_cleaner_pass(
            record=CleanerPassRecord(
                scope=history_scope,
                pass_at=now,
                outcome="errored",
                dry_run=False,
                payload=json.dumps(ErroredPayload(error=str(error)).model_dump(mode="json")),
            )
        )
    except Exception as history_exc:  # broad on purpose: recording a failure must not raise
        _log.error(
            "record_pass_failure: failed to append the pass-history row for scope %r "
            "(original pass error: %s); append error: %s",
            scope,
            error,
            history_exc,
            exc_info=history_exc,
        )
