"""The Cleaner's revisit axis: a rotation over old records, judged with evidence in hand.

Selection is deterministic rules in code (``revisit_pool.select_revisit_pool``), not prompt
instructions, so the panel sees only records with mixed evidence. ``engine/revisit_pool.py`` holds
the rule order, the protected rule, the later-session lookup and the history block. A protected
record, a ``skipped_no_evidence`` one and a ``skipped_too_large`` one each get a review row
outside a dry run, so they do not stay at the front of the rotation and block it. The first two are
appended before batching and the third in the batch path. The compounding review interval is
``review_schedule.next_review_due_at``, a pure function of one record's own prior ``RecordReview``
history, which callers fetch with ``adapter.list_reviews_for`` immediately before each write.

Judge failures: a batch where either seat's outcome is not ``success`` writes nothing, including
review rows, and is recorded in ``failed_batches``. There is no retry and no per-record fallback;
only the relevance axis has them. A batch of one record whose prompt is too large for both seats is
recorded in ``skipped_too_large_ids`` with a ``skipped_too_large`` review row.

Band policy is shared with the relevance axis (``axis_common.decide_band``): unanimous shelve writes
via ``set_shelved``, unanimous keep writes nothing, and ``unclear`` from either seat overrides
everything. A split is the panel's ``contested`` band, which writes nothing to the record itself;
when an Assessor is configured, its split resolution can still write (``split_keep`` and
``split_shelve``).

The runner is a short loop over batches. Each batch is judged by ``_run_batch``, and each record
reaches one handler per band (``_handle_unclear``, ``_handle_keep``, ``_handle_contested``,
``_handle_unanimous_shelve``). The handlers fill a ``_Tally`` of single-meaning lists, and the
summary's counts are the lengths of those lists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, assert_never

from pydantic import BaseModel

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import (
    Fact,
    HypomnemaId,
    ReviewBand,
    Scope,
    Session,
    VerdictActor,
)
from tabularium.embedding import Embedder
from tabularium.engine._errors import ToolError
from tabularium.engine._types import RequireScope
from tabularium.engine.admission import ask
from tabularium.engine.assessor import Assessor, Unavailable, Verdict
from tabularium.engine.axis_common import (
    BandCounts,
    BatchFailure,
    ContestedVerdict,
    JudgmentAppendFailure,
    RecordVerdict,
    RefusedVerdict,
    ShelveVerdict,
    append_judgment_safely,
    append_review_safely,
    batch_outcome,
    build_review_row,
    decide_band,
    deciding_verdict,
    run_panel,
    shelve_reason,
)
from tabularium.engine.cleaner_writes import (
    SplitResolution,
    TrimmedPair,
    apply_trim,
    resolve_split_keep_write,
    resolve_split_shelve_write,
)
from tabularium.engine.dispositions import set_shelved
from tabularium.engine.judge import (
    JudgeAdapter,
    JudgeCallRecord,
    JudgeOutcome,
    pack_batches,
    render_batch_prompt,
)
from tabularium.engine.review_schedule import next_review_due_at
from tabularium.engine.revisit_pool import (
    RevisitCandidate,
    RevisitPool,
    assemble_history_block,
    select_revisit_pool,
)
from tabularium.engine.rubric import Rubric
from tabularium.engine.verdicts import (
    CallerName,
    WritePending,
    build_cleaner_pending_row,
    build_cleaner_resolved_row,
    decide_split_action,
    newest_pick_bearing_row,
    restamp_cleaner_row,
    stamp_acted_many,
    verdict_from_cleaner_row,
)
from tabularium.storage.audit import ReviewOutcome
from tabularium.storage.types import (
    RecordJudgment,
    RecordReview,
    ReinforcementEvent,
    StorageAdapter,
    VerdictRow,
)
from tabularium.tokens import LitellmTokenCounter, TokenCounter

# ---------------------------------------------------------------------------
# The judged half: batching, the two seats, the band policy, the writes and the review rows.
#
# Isolation has two layers:
# - Per record: the `set_shelved(..., require_valid_and_unsuperseded=True)` call is wrapped in a
#   narrow `except ToolError`, as in the sibling axes. A record whose eligibility changed between
#   selection and the write is skipped and reported in `RevisitAxisSummary.refused`.
# - Per pass: `engine/clean.py` wraps the whole `run_revisit_axis` call in a broad try/except, so an
#   unanticipated failure is reported in this axis's own section without touching the birth axes'
#   `outcome` or `CleanerState.watermark_at`.
#
# This axis has no watermark: a record a crash left unprocessed stays due, and
# `list_due_for_review` returns it on the next pass.
# ---------------------------------------------------------------------------

#: Ceiling on the per-scope session prefetch (`adapter.search_sessions`) done once per pass to
#: build `sessions_by_scope`. A safety ceiling on a mandatory `limit` argument, not a rubric
#: parameter.
_SESSION_FETCH_LIMIT = 100_000


class RevisitVerdict(ShelveVerdict):
    """One judge's verdict on one record in a revisit batch.

    It has the same shape as ``RelevanceVerdict`` (same four-point scale, ``why`` and ``unclear``),
    because the rendered prompt is ``relevance.prose + revisit.prose`` and the panel answers under
    the same schema. It is its own class so a schema failure is reported under the name
    ``RevisitVerdict``, which lands in ``JudgeCallRecord.error``.
    """


class RevisitReviewAppendFailure(BaseModel):
    """One record whose ``RecordReview`` append failed. The record's band and write decision are not
    lost, since they are already in ``shelved_ids`` and ``counts``; only the audit row is. This is a
    write failure after a verdict, distinct from ``failed_batches``, which are judge-call failures.
    """

    model_config = {"frozen": True}

    target_id: HypomnemaId
    outcome: ReviewOutcome
    error: str


class RevisitAxisSummary(BaseModel):
    """Structured result of one revisit-axis pass. ``engine/clean.py`` reads it to build this axis's
    isolated report section, never to decide the birth axes' ``outcome`` or
    ``CleanerState.watermark_at``.

    Reconciliation: every record in ``pool_size`` lands in exactly one of these terms, so
    ``pool_size == len(excluded_global) + len(excluded_non_fact) + len(protected_ids) +
    len(skipped_no_evidence_ids) + len(skipped_too_large_ids) + <every id across failed_batches> +
    counts.shelve + counts.keep + counts.unclear + counts.contested + split_keep + len(refused)``.
    ``counts.shelve`` already includes an Assessor-resolved split-shelve, which is in
    ``shelved_ids``, so there is no separate term. ``split_keep`` is a split the Assessor resolved
    to keep and is disjoint from ``counts.keep``. ``refused`` is disjoint from ``counts.shelve``: a
    unanimous shelve that ``set_shelved`` refused is in ``refused`` only. A dry run counts the same
    way, except that no write is attempted, so ``refused`` stays empty.

    ``trimmed`` is a subset of ``counts.shelve``, not a new term: shelved records whose judges named
    a surviving part in ``keep`` and whose part was saved as a new Fact. A trim whose save raised or
    bounced un-shelved its record, which is then in ``refused`` and not in ``counts.shelve``.
    """

    model_config = {"frozen": True}

    scope: Scope
    rubric_version: str
    judge_pair: tuple[str, str]
    pool_size: int
    excluded_global: tuple[HypomnemaId, ...]
    excluded_non_fact: tuple[HypomnemaId, ...]
    protected_ids: tuple[HypomnemaId, ...]
    skipped_no_evidence_ids: tuple[HypomnemaId, ...]
    skipped_too_large_ids: tuple[HypomnemaId, ...] = ()
    counts: BandCounts
    shelved_ids: tuple[HypomnemaId, ...]
    unclear_ids: tuple[HypomnemaId, ...]
    contested_ids: tuple[HypomnemaId, ...]
    failed_batches: tuple[BatchFailure, ...]
    calls: tuple[JudgeCallRecord, ...]
    verdicts: tuple[RecordVerdict, ...]
    contested_verdicts: tuple[ContestedVerdict, ...]
    verification_failures: tuple[HypomnemaId, ...]
    review_append_failures: tuple[RevisitReviewAppendFailure, ...] = ()
    refused: tuple[RefusedVerdict, ...] = ()
    # Contested records whose `RecordJudgment` append failed; see
    # `JudgmentAppendFailure`.
    judgment_append_failures: tuple[JudgmentAppendFailure, ...] = ()
    split_keep: int = 0
    trimmed: tuple[TrimmedPair, ...] = ()


async def _verify_shelve_writes(
    *, adapter: StorageAdapter, shelved_ids: Sequence[HypomnemaId]
) -> tuple[HypomnemaId, ...]:
    """Re-read every id this axis reports as shelved and return those whose transition is absent.
    This is the ``shelved_ids`` leg of ``engine/clean.py``'s ``_verify_writes``; this axis draws no
    supersede edges. It is a local copy because ``engine/clean.py`` imports this module, so
    importing the other way would be circular.
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
    return tuple(failures)


def _review_failure(review: RecordReview, error: str) -> RevisitReviewAppendFailure:
    return RevisitReviewAppendFailure(
        target_id=review.target_id, outcome=review.outcome, error=error
    )


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _AxisContext:
    scope: Scope
    adapter: StorageAdapter
    clock: Clock
    lock: asyncio.Lock
    cfg: EngineConfig
    rubric: Rubric
    judge_a: JudgeAdapter
    judge_b: JudgeAdapter
    assessor: Assessor | None
    embedder: Embedder | None
    require_scope: RequireScope | None
    token_counter: TokenCounter
    prose: str
    now: datetime
    dry_run: bool


@dataclass(slots=True)
class _Tally:
    """The lists one pass fills. The band counts and ``split_keep`` in the summary are the lengths
    of ``shelved_ids``, ``keep_ids``, ``unclear_ids``, ``contested_ids`` and ``split_keep_ids``, so
    no count is undone after the fact. ``shelved_ids`` holds both unanimous-panel shelves and
    Assessor split-shelves, and an id enters it only once its write and trim stand. ``pool_size``
    and the excluded, protected and skipped-for-no-evidence ids come from selection, not from here.
    """

    shelved_ids: list[HypomnemaId] = field(default_factory=list)
    unclear_ids: list[HypomnemaId] = field(default_factory=list)
    contested_ids: list[HypomnemaId] = field(default_factory=list)
    keep_ids: list[HypomnemaId] = field(default_factory=list)
    split_keep_ids: list[HypomnemaId] = field(default_factory=list)
    skipped_too_large_ids: list[HypomnemaId] = field(default_factory=list)
    failed_batches: list[BatchFailure] = field(default_factory=list)
    calls: list[JudgeCallRecord] = field(default_factory=list)
    verdicts: list[RecordVerdict] = field(default_factory=list)
    contested_verdicts: list[ContestedVerdict] = field(default_factory=list)
    refused: list[RefusedVerdict] = field(default_factory=list)
    trimmed: list[TrimmedPair] = field(default_factory=list)
    review_append_failures: list[RevisitReviewAppendFailure] = field(default_factory=list)
    judgment_append_failures: list[JudgmentAppendFailure] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PanelRecord:
    record: Fact
    verdict_a: RevisitVerdict
    verdict_b: RevisitVerdict
    judge_a_identity: str
    judge_b_identity: str


@dataclass(frozen=True, slots=True)
class _Selection:
    pool_size: int
    pool: RevisitPool


@dataclass(frozen=True, slots=True)
class _PendingSplit:
    """The pending row for a split write and the adapter call that stores it."""

    row: VerdictRow
    write: WritePending


@dataclass(frozen=True, slots=True)
class _SplitPrepass:
    """The Assessor verdict a split resolves from: the reused ``cleaner_prepass`` row (or ``None``
    when a fresh ``cleaner_split`` call was made), the verdict, and the caller name it is logged
    under.
    """

    reused_row: VerdictRow | None
    verdict: Verdict | Unavailable
    caller: CallerName


# ---------------------------------------------------------------------------
# Review rows
# ---------------------------------------------------------------------------


async def _append_review(
    ctx: _AxisContext,
    tally: _Tally,
    target_id: HypomnemaId,
    *,
    outcome: ReviewOutcome,
    band: ReviewBand | None = None,
    grades: tuple[str, ...] = (),
) -> None:
    await append_review_safely(
        adapter=ctx.adapter,
        review=await build_review_row(
            adapter=ctx.adapter,
            scope=ctx.scope,
            target_id=target_id,
            now=ctx.now,
            rubric=ctx.rubric,
            outcome=outcome,
            band=band,
            grades=grades,
        ),
        failures=tally.review_append_failures,
        to_failure=_review_failure,
    )


async def _append_judged_review(
    ctx: _AxisContext, tally: _Tally, item: _PanelRecord, *, band: ReviewBand
) -> None:
    """The ``judged`` review row for a record the panel decided. A dry run appends none."""
    if ctx.dry_run:
        return

    await _append_review(
        ctx,
        tally,
        item.record.id,
        outcome="judged",
        band=band,
        grades=(item.verdict_a.grade, item.verdict_b.grade),
    )


async def _append_judgment(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    """The judge detail of a contested record goes to the separate append-only ``record_judgments``
    table, not onto the ``RecordReview``. Best-effort: a failure is recorded and does not lose the
    band decision or abort the pass. A dry run appends none.
    """
    if ctx.dry_run:
        return

    await append_judgment_safely(
        adapter=ctx.adapter,
        judgment=RecordJudgment(
            target_id=item.record.id,
            scope=ctx.scope,
            judged_at=ctx.now,
            rubric_version=ctx.rubric.version,
            judge_a_identity=item.judge_a_identity,
            judge_a_grade=item.verdict_a.grade,
            judge_a_why=item.verdict_a.why,
            judge_b_identity=item.judge_b_identity,
            judge_b_grade=item.verdict_b.grade,
            judge_b_why=item.verdict_b.why,
        ),
        failures=tally.judgment_append_failures,
    )


async def _append_pre_batch_reviews(
    ctx: _AxisContext, tally: _Tally, selected: RevisitPool
) -> None:
    """A protected record and a no-evidence record still get a review row, or they would stay at
    the front of the rotation and block it. A dry run appends none.
    """
    if ctx.dry_run:
        return

    for record_id in selected.protected:
        await _append_review(ctx, tally, record_id, outcome="protected")
    for record_id in selected.skipped_no_evidence:
        await _append_review(ctx, tally, record_id, outcome="skipped_no_evidence")


# ---------------------------------------------------------------------------
# Split band: reuses the relevance axis's `cleaner_prepass` verdict when one exists and makes one
# fresh call, logged `cleaner_split`, only when it does not. The write is the shared rule in
# `engine/verdicts.py`.
# ---------------------------------------------------------------------------


async def _split_prepass(ctx: _AxisContext, assessor: Assessor, record: Fact) -> _SplitPrepass:
    """The newest pick-bearing ``cleaner_prepass`` row for ``record`` when one exists, with no
    call. Otherwise one fresh ``relevance`` call, logged ``cleaner_split``.
    """
    existing_rows = await ctx.adapter.verdicts_for_record(record.id)
    reused_row = newest_pick_bearing_row(existing_rows, caller="cleaner_prepass")
    if reused_row is not None:
        return _SplitPrepass(
            reused_row=reused_row,
            verdict=verdict_from_cleaner_row(reused_row),
            caller="cleaner_prepass",
        )

    fresh = await ask(
        lambda: assessor.relevance(record.content),
        method="relevance",
        timeout_s=ctx.cfg.assessor_cleaner_timeout_s,
    )
    return _SplitPrepass(reused_row=None, verdict=fresh, caller="cleaner_split")


def _pending_split_row(
    ctx: _AxisContext,
    record: Fact,
    action: Literal["split_keep", "split_shelve"],
    prepass: _SplitPrepass,
) -> _PendingSplit:
    """The ``_PendingSplit`` for a split write. A reused row is
    restamped ``pending`` and updated in place. Otherwise a fresh pending row is built and
    appended; for ``split_shelve`` it pre-publishes the record's id as ``new_record_id``, so the
    row is crash-safe from the moment it lands pending.
    """
    if prepass.reused_row is not None:
        restamped = restamp_cleaner_row(
            prepass.reused_row, action=action, actor="engine", matched=True, now=ctx.now
        )
        return _PendingSplit(row=restamped, write=ctx.adapter.update_verdicts)

    assert isinstance(prepass.verdict, Verdict)
    fresh = build_cleaner_pending_row(
        caller=prepass.caller,
        candidate_id=record.id,
        new_record_id=record.id if action == "split_shelve" else None,
        verdict=prepass.verdict,
        action=action,
        effect_ref=None,
        now=ctx.now,
    )
    return _PendingSplit(row=fresh, write=ctx.adapter.append_verdicts)


async def _write_split_action(
    ctx: _AxisContext,
    tally: _Tally,
    record: Fact,
    action: Literal["split_keep", "split_shelve"],
    prepass: _SplitPrepass,
) -> SplitResolution:
    """Run the shared split write for ``split_keep`` or ``split_shelve``. ``write`` is
    ``"landed"`` or ``"failed"`` from its result. The shelve re-checks validity and supersession
    under the same lock as the shelve, like this axis's unanimous-shelve write, so a reversal that
    lands while the judges or the Assessor ran cannot slip through.
    """
    pending = _pending_split_row(ctx, record, action, prepass)
    history = tuple(await ctx.adapter.list_reviews_for(target_id=record.id))
    verdict = prepass.verdict
    assert isinstance(verdict, Verdict)

    async def _on_review_append_failure(exc: Exception) -> None:
        tally.review_append_failures.append(
            RevisitReviewAppendFailure(
                target_id=record.id, outcome="assessor_resolved", error=str(exc)
            )
        )

    match action:
        case "split_keep":
            resolved = await resolve_split_keep_write(
                pending_row=pending.row,
                write_pending=pending.write,
                record=record,
                rubric_version=ctx.rubric.version,
                next_review_due_at=next_review_due_at(
                    ctx.now, history=history, params=ctx.rubric.parameters
                ),
                now=ctx.now,
                adapter=ctx.adapter,
                lock=ctx.lock,
                on_review_append_failure=_on_review_append_failure,
            )
        case "split_shelve":
            resolved = await resolve_split_shelve_write(
                pending_row=pending.row,
                write_pending=pending.write,
                record=record,
                reason=(
                    f"Cleaner revisit split: Assessor graded {verdict.pick} "
                    f"(confidence {verdict.confidence:.2f}). Rubric {ctx.rubric.version}."
                ),
                adapter=ctx.adapter,
                clock=ctx.clock,
                lock=ctx.lock,
                require_valid_and_unsuperseded=True,
            )
        case _:
            assert_never(action)
    return SplitResolution(action=action, write="landed" if resolved else "failed")


async def _record_split_row(
    ctx: _AxisContext,
    record: Fact,
    action: Literal["split_annotated", "panel_contested"],
    prepass: _SplitPrepass,
) -> None:
    """Record or restamp the Cleaner's own row for ``split_annotated`` and ``panel_contested``, in
    one lock hold. The caller's contested write still runs.
    """
    actor: VerdictActor = "panel" if action == "panel_contested" else "engine"
    matched = None if action == "panel_contested" else False
    async with ctx.lock:
        if prepass.reused_row is not None:
            restamped = restamp_cleaner_row(
                prepass.reused_row, action=action, actor=actor, matched=matched, now=ctx.now
            )
            await stamp_acted_many(ctx.adapter, [restamped], now=ctx.now)
        else:
            row = build_cleaner_resolved_row(
                caller=prepass.caller,
                candidate_id=record.id,
                verdict=prepass.verdict,
                action=action,
                actor=actor,
                matched=matched,
                now=ctx.now,
            )
            await ctx.adapter.append_verdicts([row])


async def _resolve_revisit_split(ctx: _AxisContext, tally: _Tally, record: Fact) -> SplitResolution:
    """The revisit split band. Returns a ``SplitResolution``; with no Assessor it returns action
    ``"none"`` with ``write`` ``"none"``.

    The verdict comes from ``_split_prepass``. For ``split_keep`` and ``split_shelve``,
    ``_write_split_action`` restamps the existing row ``pending`` (or builds a fresh pending row)
    and runs ``resolve_split_keep_write`` or ``resolve_split_shelve_write``, the latter with
    ``require_valid_and_unsuperseded=True``; ``write`` is ``"landed"`` or ``"failed"`` from its
    result. For ``split_annotated`` and ``panel_contested``, the caller's contested write still runs
    and ``_record_split_row`` only records or restamps the Cleaner's own row, in one lock hold,
    returning ``write`` ``"none"``. Under ``dry_run`` it returns ``"landed"`` for ``split_keep`` and
    ``split_shelve``, ``"none"`` for every other action, and writes nothing.
    """
    assessor = ctx.assessor
    if assessor is None:
        return SplitResolution(action="none", write="none")

    prepass = await _split_prepass(ctx, assessor, record)
    action = decide_split_action(prepass.verdict, cfg=ctx.cfg)

    if ctx.dry_run:
        dry_write: Literal["none", "landed"] = (
            "landed" if action in ("split_keep", "split_shelve") else "none"
        )
        return SplitResolution(action=action, write=dry_write)

    match action:
        case "split_keep" | "split_shelve":
            return await _write_split_action(ctx, tally, record, action, prepass)
        case "split_annotated" | "panel_contested":
            await _record_split_row(ctx, record, action, prepass)
            return SplitResolution(action=action, write="none")
        case "none":
            raise AssertionError("action is 'none' only when no Assessor is configured")
        case _:
            assert_never(action)


# ---------------------------------------------------------------------------
# Per-band handlers
# ---------------------------------------------------------------------------


async def _handle_unclear(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    tally.unclear_ids.append(item.record.id)
    await _append_judged_review(ctx, tally, item, band="unclear")


async def _handle_keep(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    tally.keep_ids.append(item.record.id)
    await _append_judged_review(ctx, tally, item, band="keep")


async def _trim_stands(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> bool:
    """Trim the just-shelved record. ``False`` when the save failed and the shelve was reverted: the
    record is then in ``refused`` and the caller records no shelve.
    """
    trim = await apply_trim(
        record=item.record,
        verdicts=(item.verdict_a, item.verdict_b),
        embedder=ctx.embedder,
        require_scope=ctx.require_scope,
        adapter=ctx.adapter,
        clock=ctx.clock,
        cfg=ctx.cfg,
        lock=ctx.lock,
        dry_run=ctx.dry_run,
    )
    if trim.pair is not None:
        tally.trimmed.append(trim.pair)
    if trim.refusal is not None:
        tally.refused.append(RefusedVerdict(id=item.record.id, reason=trim.refusal))
        return False
    return True


async def _handle_split_shelve(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    """An Assessor-resolved split-shelve whose write landed (under ``dry_run``, would have). The
    Assessor resolution appends no review row, so this handler appends none whether or not the
    trim stands.
    """
    if not await _trim_stands(ctx, tally, item):
        return

    tally.shelved_ids.append(item.record.id)
    tally.verdicts.append(
        RecordVerdict(
            id=item.record.id,
            band="shelve",
            why="Assessor split-shelve vote (revisit axis)",
        )
    )


async def _handle_contested(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    """A split reuses the relevance pre-pass verdict when one exists, else makes a fresh
    ``relevance`` call logged ``cleaner_split``. A landed ``split_keep`` or ``split_shelve``
    resolves the split outright and skips the contested rows below. Any other result, including a
    failed split write, leaves the record contested: it is listed, and unless this is a dry run its
    ``judged``/``contested`` review and its judgment row are appended.
    """
    record = item.record
    split = await _resolve_revisit_split(ctx, tally, record)

    if split.landed and split.action == "split_keep":
        tally.split_keep_ids.append(record.id)
        return

    if split.landed and split.action == "split_shelve":
        await _handle_split_shelve(ctx, tally, item)
        return

    tally.contested_ids.append(record.id)
    tally.contested_verdicts.append(
        ContestedVerdict(
            id=record.id,
            judge_a_identity=item.judge_a_identity,
            judge_a_grade=item.verdict_a.grade,
            judge_a_why=item.verdict_a.why,
            judge_b_identity=item.judge_b_identity,
            judge_b_grade=item.verdict_b.grade,
            judge_b_why=item.verdict_b.why,
        )
    )
    await _append_judged_review(ctx, tally, item, band="contested")
    await _append_judgment(ctx, tally, item)


async def _write_shelve(ctx: _AxisContext, tally: _Tally, item: _PanelRecord, reason: str) -> bool:
    """Shelve the record. ``set_shelved`` re-checks eligibility inside its own lock, atomically
    with its fresh read; a caller-side re-read would leave a window before the write, and only this
    axis passes ``require_valid_and_unsuperseded``. A ``ToolError`` is a pre-write legality refusal
    (an illegal lifecycle, or the stricter validity and superseded guard): this record alone is
    recorded in ``refused`` and ``False`` returns. A storage failure is not a ``ToolError`` and
    propagates. A dry run writes nothing and returns ``True``.
    """
    if ctx.dry_run:
        return True

    try:
        await set_shelved(
            id=item.record.id,
            shelved=True,
            reason=reason,
            adapter=ctx.adapter,
            clock=ctx.clock,
            lock=ctx.lock,
            require_valid_and_unsuperseded=True,
        )
    except ToolError as exc:
        tally.refused.append(RefusedVerdict(id=item.record.id, reason=str(exc)))
        return False
    return True


async def _handle_unanimous_shelve(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    """Shelve, then trim. A ``set_shelved`` refusal is in ``refused`` and the record still gets its
    ``judged``/``shelve`` review, since the panel did decide. A reverted trim is in ``refused`` too
    and gets no review, which leaves the restored record due for the next pass.
    """
    deciding = deciding_verdict(item.verdict_a, item.verdict_b)
    reason = shelve_reason(
        "revisit",
        deciding=deciding,
        deciding_judge=item.judge_a_identity,
        judge_a_identity=item.judge_a_identity,
        judge_b_identity=item.judge_b_identity,
        rubric_version=ctx.rubric.version,
    )

    if await _write_shelve(ctx, tally, item, reason):
        if not await _trim_stands(ctx, tally, item):
            return

        tally.shelved_ids.append(item.record.id)
        tally.verdicts.append(RecordVerdict(id=item.record.id, band="shelve", why=deciding.why))

    await _append_judged_review(ctx, tally, item, band="shelve")


async def _handle_record(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    band = decide_band(item.verdict_a, item.verdict_b)
    match band:
        case "unclear":
            await _handle_unclear(ctx, tally, item)
        case "keep":
            await _handle_keep(ctx, tally, item)
        case "contested":
            await _handle_contested(ctx, tally, item)
        case "shelve":
            await _handle_unanimous_shelve(ctx, tally, item)
        case _:
            assert_never(band)


# ---------------------------------------------------------------------------
# One batch: the panel, the failure gates, the records
# ---------------------------------------------------------------------------


def _is_lone_too_large(
    batch: Sequence[RevisitCandidate],
    outcome_a: JudgeOutcome[RevisitVerdict],
    outcome_b: JudgeOutcome[RevisitVerdict],
) -> bool:
    return (
        len(batch) == 1
        and outcome_a.status == "prompt_too_large"
        and outcome_b.status == "prompt_too_large"
    )


def _batch_failure(
    *,
    batch_index: int,
    batch: Sequence[RevisitCandidate],
    outcome_a: JudgeOutcome[RevisitVerdict],
    outcome_b: JudgeOutcome[RevisitVerdict],
) -> BatchFailure | None:
    """The failure to record when either seat's outcome is not ``ok``, else ``None``. There is no
    retry and no per-record fallback.
    """
    judge_a_outcome = batch_outcome(outcome_a.status)
    judge_b_outcome = batch_outcome(outcome_b.status)
    if judge_a_outcome == "ok" and judge_b_outcome == "ok":
        return None

    errors = [o.error for o in (outcome_a, outcome_b) if o.error is not None]
    return BatchFailure(
        batch_index=batch_index,
        record_ids=tuple(candidate.record.id for candidate in batch),
        judge_a_outcome=judge_a_outcome,
        judge_b_outcome=judge_b_outcome,
        error="; ".join(errors),
    )


async def _skip_too_large(ctx: _AxisContext, tally: _Tally, record: Fact) -> None:
    """A lone record whose prompt both seats found too large is listed and, unless this is a dry
    run, gets a ``skipped_too_large`` review row.
    """
    tally.skipped_too_large_ids.append(record.id)
    if not ctx.dry_run:
        await _append_review(ctx, tally, record.id, outcome="skipped_too_large")


async def _run_batch(
    ctx: _AxisContext,
    tally: _Tally,
    *,
    batch_index: int,
    batch: Sequence[RevisitCandidate],
    contents: Sequence[str],
) -> None:
    """Both seats get the identical prompt over per-record content that already carries its history
    block. Nothing in a failed batch is written, so there is no partial application.
    """
    prompt = render_batch_prompt(
        rubric_prose=ctx.prose, contents=list(contents), item_model=RevisitVerdict
    )
    outcome_a, outcome_b = await run_panel(
        judge_a=ctx.judge_a,
        judge_b=ctx.judge_b,
        prompt=prompt,
        item_model=RevisitVerdict,
        batch_size=len(batch),
        rubric_version=ctx.rubric.version,
        token_counter=ctx.token_counter,
        batch_index=batch_index,
        calls=tally.calls,
    )

    if _is_lone_too_large(batch, outcome_a, outcome_b):
        await _skip_too_large(ctx, tally, batch[0].record)
        return

    failure = _batch_failure(
        batch_index=batch_index, batch=batch, outcome_a=outcome_a, outcome_b=outcome_b
    )
    if failure is not None:
        tally.failed_batches.append(failure)
        return

    for candidate, verdict_a, verdict_b in zip(
        batch, outcome_a.verdicts, outcome_b.verdicts, strict=True
    ):
        item = _PanelRecord(
            record=candidate.record,
            verdict_a=verdict_a,
            verdict_b=verdict_b,
            judge_a_identity=outcome_a.judge,
            judge_b_identity=outcome_b.judge,
        )
        await _handle_record(ctx, tally, item)


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


async def _select_pool(
    *, scope: Scope, adapter: StorageAdapter, now: datetime, rubric: Rubric
) -> _Selection:
    """Read the due pool, each record's events, the scope's sessions, then
    select. The reads happen here, once per pass, so ``select_revisit_pool`` does no I/O.
    """
    created_before = now - timedelta(days=rubric.parameters.revisit_min_age_days)
    raw_pool = await adapter.list_due_for_review(
        scope=scope,
        now=now,
        created_before=created_before,
        limit=rubric.parameters.revisit_pass_cap,
    )

    events_by_id: dict[HypomnemaId, tuple[ReinforcementEvent, ...]] = {}
    for record in raw_pool:
        events_by_id[record.id] = tuple(await adapter.query_events(target_id=record.id))

    sessions = await adapter.search_sessions(
        scopes=[str(scope)], query=None, since=None, until=None, limit=_SESSION_FETCH_LIMIT
    )
    sessions_by_scope: dict[Scope, tuple[Session, ...]] = {scope: tuple(sessions)}

    pool = select_revisit_pool(
        raw_pool,
        pass_scope=scope,
        events_by_id=events_by_id,
        sessions_by_scope=sessions_by_scope,
        params=rubric.parameters,
    )
    return _Selection(pool_size=len(raw_pool), pool=pool)


def _candidate_contents(selection: _Selection) -> list[str]:
    return [
        candidate.record.content
        + assemble_history_block(
            birth_session=candidate.birth_session,
            later_sessions=candidate.later_sessions,
        )
        for candidate in selection.pool.candidates
    ]


def _build_summary(
    ctx: _AxisContext,
    tally: _Tally,
    selection: _Selection,
    verification_failures: tuple[HypomnemaId, ...],
) -> RevisitAxisSummary:
    return RevisitAxisSummary(
        scope=ctx.scope,
        rubric_version=ctx.rubric.version,
        judge_pair=(ctx.judge_a.identity, ctx.judge_b.identity),
        pool_size=selection.pool_size,
        excluded_global=selection.pool.excluded_global,
        excluded_non_fact=selection.pool.excluded_non_fact,
        protected_ids=selection.pool.protected,
        skipped_no_evidence_ids=selection.pool.skipped_no_evidence,
        skipped_too_large_ids=tuple(tally.skipped_too_large_ids),
        counts=BandCounts(
            shelve=len(tally.shelved_ids),
            keep=len(tally.keep_ids),
            unclear=len(tally.unclear_ids),
            contested=len(tally.contested_ids),
        ),
        shelved_ids=tuple(tally.shelved_ids),
        unclear_ids=tuple(tally.unclear_ids),
        contested_ids=tuple(tally.contested_ids),
        failed_batches=tuple(tally.failed_batches),
        calls=tuple(tally.calls),
        verdicts=tuple(tally.verdicts),
        contested_verdicts=tuple(tally.contested_verdicts),
        verification_failures=verification_failures,
        review_append_failures=tuple(tally.review_append_failures),
        refused=tuple(tally.refused),
        judgment_append_failures=tuple(tally.judgment_append_failures),
        split_keep=len(tally.split_keep_ids),
        trimmed=tuple(tally.trimmed),
    )


async def run_revisit_axis(
    *,
    scope: Scope,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
    cfg: EngineConfig,
    rubric: Rubric,
    judge_a: JudgeAdapter,
    judge_b: JudgeAdapter,
    assessor: Assessor | None = None,
    dry_run: bool = False,
    embedder: Embedder | None = None,
    require_scope: RequireScope | None = None,
) -> RevisitAxisSummary:
    """Judge the rotation of old Facts in ``scope`` that are due for another look, with session
    evidence in hand. The module docstring covers selection, the history block and the compounding
    interval.

    Pool: ``adapter.list_due_for_review(scope=scope, now=clock.now(), created_before=now -
    revisit_min_age_days, limit=revisit_pass_cap)``, then ``select_revisit_pool``. Batches are
    packed to ``revisit_batch_cap``, not the birth axes' ``batch_cap``. Both seats get the identical
    prompt, ``relevance.prose + revisit.prose`` over per-record content that already carries its
    history block, and are called concurrently with ``asyncio.gather``.

    Band policy is the relevance axis's. A ``RecordReview`` is appended for each protected,
    ``skipped_no_evidence``, ``skipped_too_large`` and judged record, except that none is appended
    when the Assessor resolved the split (``resolve_split_keep_write`` appends its own keep
    review) or when a trim was reverted, which leaves the restored record due for
    the next pass. A dry run appends no review rows and writes nothing else.

    A review-row append failure does not lose the record's write decision. ``append_review_safely``
    records it in ``review_append_failures``, and a shelve that landed stays in ``shelved_ids`` and
    is still verified by ``_verify_shelve_writes``.

    ``set_shelved`` is called with ``require_valid_and_unsuperseded=True``. A ``ToolError`` from it
    skips the record and records it in ``refused``: its fresh read inside the lock found the record
    no longer eligible (invalidated, superseded, buried, or off ``active``/``archived`` since
    selection). A ``set_shelved`` refusal still gets its ``judged``/``shelve`` ``RecordReview``,
    since the panel did decide. A reverted trim is in ``refused`` with the trim-failure text and
    gets none. Other exceptions propagate to ``engine/clean.py``, which wraps this whole call and
    reports a failure in this axis's section alone.

    ``cfg`` supplies the split band's switch and timeout (``cleaner_split_shelve``,
    ``assessor_cleaner_timeout_s``) and the trim save. This axis makes no gate-probe call.

    ``embedder`` and ``require_scope`` serve trim saves (``engine/cleaner_writes.py``), as in
    ``run_relevance_axis``. A shelved record whose shelve-voting seats name a surviving part in
    ``keep`` is shelved first, then the part is saved as a new Fact. ``embedder=None`` turns the
    save off. Under ``dry_run`` the trim is counted and nothing is saved.
    """
    now = clock.now()
    selection = await _select_pool(scope=scope, adapter=adapter, now=now, rubric=rubric)
    ctx = _AxisContext(
        scope=scope,
        adapter=adapter,
        clock=clock,
        lock=lock,
        cfg=cfg,
        rubric=rubric,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        embedder=embedder,
        require_scope=require_scope,
        token_counter=LitellmTokenCounter(),
        prose=rubric.relevance.prose + "\n\n" + rubric.revisit.prose,
        now=now,
        dry_run=dry_run,
    )
    tally = _Tally()
    await _append_pre_batch_reviews(ctx, tally, selection.pool)

    contents = _candidate_contents(selection)
    groups = pack_batches(
        contents,
        count_cap=rubric.parameters.revisit_batch_cap,
        rubric_prose=ctx.prose,
        item_model=RevisitVerdict,
    )
    for batch_index, group in enumerate(groups):
        await _run_batch(
            ctx,
            tally,
            batch_index=batch_index,
            batch=[selection.pool.candidates[position] for position in group],
            contents=[contents[position] for position in group],
        )

    verification_failures: tuple[HypomnemaId, ...] = ()
    if not dry_run:
        verification_failures = await _verify_shelve_writes(
            adapter=adapter, shelved_ids=tally.shelved_ids
        )

    return _build_summary(ctx, tally, selection, verification_failures)
