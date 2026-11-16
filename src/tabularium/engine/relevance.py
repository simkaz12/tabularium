"""The Cleaner's relevance axis: judge the Facts born since the watermark against the admission
gate, applied retroactively, and shelve the ones that fail it.

Pool selection (``_select_relevance_pool``): ``list_born_since(scope, since)`` returns Facts and
Decisions alike. Three exclusions apply, each reported separately:

- Not a Fact. ``set_shelved`` refuses a Decision, so it is dropped here instead of failing at write
  time.
- ``global`` records, unless the pass itself is a ``global`` pass.
- ``lifecycle`` not in ``("active", "archived")``, the only lifecycles ``set_shelved`` accepts as a
  shelve source. This covers already-shelved Facts and keeps ``hot`` and ``ember`` Facts out of
  candidacy. All three lifecycle grounds land in the ``excluded_already_shelved`` bucket.

The panel: candidates are packed into batches by ``pack_batches`` (rubric ``batch_cap`` and the
prompt ceiling). Both judges receive the identical prompt from ``render_batch_prompt``, with
batch-local integers and no record id, and are called concurrently with ``asyncio.gather``. Neither
sees the other's verdicts.

Band policy:

- Both judges grade ``shelve-*``: shelved.
- Both grade ``keep-*``: untouched, no write.
- One of each: ``contested``. Nothing is written to the record. A contested ``RecordReview`` and
  ``RecordJudgment`` row are appended, except under ``dry_run``.
- ``unclear`` from either judge overrides all of the above and is never acted on.

``contested`` and ``unclear`` are separate signals ("the judges disagreed" and "a judge could not
judge") and are reported separately (``contested_ids``/``contested_verdicts`` and ``unclear_ids``).

NOTE: a completed pass advances the watermark whether or not it found contested records. A contested
id is reported once, on the pass that found it. The watermark moves past the record, so a later
pass's born-since pool does not include it.

Judge failures. A batch where either seat's outcome is ``judge_unavailable``, ``unexpected`` or
``prompt_too_large`` writes nothing and is recorded in ``failed_batches``. An ``invalid_output``
(the response arrived but failed schema validation) is handled more gently: the identical batch call
is retried once, and if that is also invalid the batch is judged one record at a time, so only a
record whose own single-record call is invalid is dropped. A dropped record appears in
``failed_batches`` with narrower ``record_ids``, so it is not lost and the watermark does not
advance past it. Any other failure at any point of the retry or fallback fails the whole batch.
``_resolve_judge_batch`` has the mechanics.

NOTE: only this axis retries invalid judge output. The falsity and revisit axes fail the whole batch
on it.

Every judge call, both seats, every batch, retries and fallback calls included, is collected in
``RelevanceAxisSummary.calls`` as a ``JudgeCallRecord``. ``failed_batches`` names only the calls
that went wrong.

The write: ``set_shelved(id, shelved=True, reason=...)`` from ``engine.dispositions``, or
``resolve_panel_shelve_write`` when an Assessor pre-pass verdict exists. The ``reason`` is built
from the deciding judge's ``why``, the judge pair and the rubric version, because ``set_shelved``
requires a non-empty reason and a generic one would defeat the audit trail. The axis never calls
``bury``, and Decisions and (in a project-scope pass) ``global`` records are excluded from the pool.

A refused write skips the record, not the pass. The pool is a snapshot taken before judging, and a
record's lifecycle can change before the write (promoted to ``hot``, shelved by a concurrent pass).
``set_shelved`` then raises ``ToolError``. The axis catches ``ToolError`` and nothing else around
the shelve write: the record is skipped, left out of ``shelved_ids`` and ``verdicts``, and recorded
in ``RelevanceAxisSummary.refused`` with the refusal text. The rest of the batch and every later
batch continue. Any other exception, such as a storage failure, propagates and aborts the pass.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, assert_never

from pydantic import BaseModel

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import (
    BatchOutcome,
    Decision,
    Fact,
    Hypomnema,
    HypomnemaId,
    ReviewBand,
    Scope,
    VerdictActor,
)
from tabularium.embedding import Embedder
from tabularium.engine._errors import ToolError
from tabularium.engine._types import RequireScope
from tabularium.engine.admission import bounded_ask
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
    resolve_panel_shelve_write,
    resolve_split_keep_write,
    resolve_split_shelve_write,
)
from tabularium.engine.dispositions import set_shelved
from tabularium.engine.filters import GLOBAL
from tabularium.engine.judge import (
    JudgeAdapter,
    JudgeCallRecord,
    JudgeOutcome,
    judge_batch,
    pack_batches,
    render_batch_prompt,
)
from tabularium.engine.review_schedule import next_review_due_at
from tabularium.engine.rubric import Rubric
from tabularium.engine.verdicts import (
    CleanerAction,
    build_cleaner_pending_row,
    build_cleaner_resolved_row,
    decide_split_action,
)
from tabularium.storage.types import RecordJudgment, RecordReview, StorageAdapter
from tabularium.tokens import LitellmTokenCounter, TokenCounter

# `pool_limit` defaults to `Rubric.parameters.pool_limit`, the field `engine/falsity.py` and
# `engine/clean.py` read too.


# ---------------------------------------------------------------------------
# The verdict schema: extends the judge seam's UnclearVerdictBase
# ---------------------------------------------------------------------------


class RelevanceVerdict(ShelveVerdict):
    """One judge's verdict on one record in a relevance batch.

    Kept as its own class so a schema failure is reported under the name ``RelevanceVerdict``,
    which lands in ``JudgeCallRecord.error``. The fields and band arithmetic are those of
    ``ShelveVerdict``.
    """


# ---------------------------------------------------------------------------
# Structured summary: this axis reports, `engine/report.py` builds the report
# ---------------------------------------------------------------------------


class RelevanceFullVerdict(BaseModel):
    """Per-seat detail for every judged candidate in any band: both judges' grade, ``why`` and
    ``unclear`` flag.

    ``verdicts`` and ``contested_verdicts`` are partial (the shelve and contested bands only), and a
    caller that re-judges a labelled batch, such as ``scripts/calibrate_relevance_rubric.py``, needs
    every band. Held in memory only: ``engine/report.py`` does not persist it. Absent for a record
    whose batch failed outright.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    band: ReviewBand
    judge_a_identity: str
    judge_a_grade: str
    judge_a_why: str
    judge_a_unclear: bool
    judge_b_identity: str
    judge_b_grade: str
    judge_b_why: str
    judge_b_unclear: bool


class RelevanceReviewAppendFailure(BaseModel):
    """One contested record whose ``RecordReview`` append failed.

    The contested verdict stands, since it is already in ``contested_ids`` and
    ``contested_verdicts`` before the append is attempted. Only the review row is missing, so
    the record has no ``band="contested"`` review. Mirrors
    ``engine.revisit.RevisitReviewAppendFailure``.
    """

    model_config = {"frozen": True}

    target_id: HypomnemaId
    error: str


class RelevanceAxisSummary(BaseModel):
    """Structured result of one relevance-axis pass. ``engine/clean.py`` reads it to decide whether
    the watermark may advance and to verify writes; ``engine/report.py`` builds the report from it.
    """

    model_config = {"frozen": True}

    scope: Scope
    rubric_version: str
    judge_pair: tuple[str, str]
    pool_size: int
    excluded_decisions: tuple[HypomnemaId, ...]
    excluded_global: tuple[HypomnemaId, ...]
    excluded_already_shelved: tuple[HypomnemaId, ...]
    counts: BandCounts
    shelved_ids: tuple[HypomnemaId, ...]
    unclear_ids: tuple[HypomnemaId, ...]
    # Split verdicts: reported, never written to the record. Distinct from `unclear_ids`.
    contested_ids: tuple[HypomnemaId, ...] = ()
    failed_batches: tuple[BatchFailure, ...]
    # One entry per `judge_batch` call actually made, both seats, every batch, success or
    # failure. Not filtered to failures: a seat degrading toward its timeout only shows as a trend
    # across successful calls.
    calls: tuple[JudgeCallRecord, ...]
    # Id and deciding judge's `why` for every id in `shelved_ids`, same order. See
    # `RecordVerdict`.
    verdicts: tuple[RecordVerdict, ...] = ()
    # Both judges' identity, grade and `why` for every id in `contested_ids`, same order.
    contested_verdicts: tuple[ContestedVerdict, ...] = ()
    # Full per-seat detail for every judged candidate, every band. See `RelevanceFullVerdict`.
    full_verdicts: tuple[RelevanceFullVerdict, ...] = ()
    # Shelve writes the panel decided on that did not stand; see `RefusedVerdict`.
    # Defaulted so older persisted rows still deserialize.
    refused: tuple[RefusedVerdict, ...] = ()
    # Contested records whose `RecordReview` append failed; the contested verdict itself stands.
    review_append_failures: tuple[RelevanceReviewAppendFailure, ...] = ()
    # Contested records whose `RecordJudgment` append failed; see
    # `JudgmentAppendFailure`.
    judgment_append_failures: tuple[JudgmentAppendFailure, ...] = ()
    # Assessor pre-pass and split counts, all 0 when no Assessor is configured. `skipped` (a
    # `keep-clear` pre-pass pick above the skip cutoff, switch on) and `split_keep` (a split the
    # Assessor resolved to keep) are disjoint from `counts.keep`. `counts.shelve` counts every
    # shelve that stands, so `split_shelve` is a sub-count of it; rows persisted before that rule
    # count unanimous shelves only. `split_annotated` is a subset of `counts.contested`: the
    # contested records that carry an Assessor shelve annotation (the switch was off).
    #
    # Counts identity: `pool_size == len(excluded_decisions) + len(excluded_global) +
    # len(excluded_already_shelved) + counts.shelve + counts.keep + counts.unclear +
    # counts.contested + skipped + split_keep + <every id across failed_batches> + <every id in
    # refused that is not also in contested_ids>`. A split-keep whose review append failed is in
    # `refused` and also lands contested; it is counted once, as contested.
    skipped: int = 0
    split_keep: int = 0
    split_annotated: int = 0
    split_shelve: int = 0
    # Shelved records whose judges named a surviving part in `keep`, saved as a new Fact
    # (`engine/cleaner_writes.py`). A subset of `counts.shelve`, not a term in the identity above.
    # A trim that did not stand un-shelved its record: that record is in `refused` and in neither
    # `counts.shelve` nor `split_shelve`.
    trimmed: tuple[TrimmedPair, ...] = ()


# ---------------------------------------------------------------------------
# Pool selection — three independent exclusions
# ---------------------------------------------------------------------------


class _RelevancePool(BaseModel):
    model_config = {"frozen": True}

    candidates: tuple[Fact, ...]
    excluded_decisions: tuple[HypomnemaId, ...]
    excluded_global: tuple[HypomnemaId, ...]
    excluded_already_shelved: tuple[HypomnemaId, ...]


def _select_relevance_pool(
    pool: Sequence[Hypomnema], *, pass_scope: Scope | None = None
) -> _RelevancePool:
    """Filter a born-since pool down to this axis's candidates.

    Checked in this order, so a record matching more than one ground lands in the first bucket:

    1. Not a ``Fact`` (a ``Decision``).
    2. ``scope == "global"``, unless ``pass_scope`` is ``global``.
    3. ``lifecycle`` not in ``("active", "archived")``.

    Everything else is a candidate.
    """
    candidates: list[Fact] = []
    excluded_decisions: list[HypomnemaId] = []
    excluded_global: list[HypomnemaId] = []
    excluded_already_shelved: list[HypomnemaId] = []

    for record in pool:
        if isinstance(record, Decision):
            excluded_decisions.append(record.id)
            continue
        if record.scope == GLOBAL and pass_scope != GLOBAL:
            excluded_global.append(record.id)
            continue
        if record.lifecycle not in ("active", "archived"):
            excluded_already_shelved.append(record.id)
            continue
        candidates.append(record)

    return _RelevancePool(
        candidates=tuple(candidates),
        excluded_decisions=tuple(excluded_decisions),
        excluded_global=tuple(excluded_global),
        excluded_already_shelved=tuple(excluded_already_shelved),
    )


# ---------------------------------------------------------------------------
# Review failures and batch failures
# ---------------------------------------------------------------------------


def _review_failure(review: RecordReview, error: str) -> RelevanceReviewAppendFailure:
    return RelevanceReviewAppendFailure(target_id=review.target_id, error=error)


def _batch_failure(
    *,
    batch_index: int,
    records: Sequence[Fact],
    judge_a_outcome: BatchOutcome,
    judge_b_outcome: BatchOutcome,
    error: str,
) -> BatchFailure:
    """One ``BatchFailure`` covering every id in ``records``: the whole batch for a hard
    failure, or the narrower set that a batch retry and a per-record fallback could not validate.
    """
    return BatchFailure(
        batch_index=batch_index,
        record_ids=tuple(record.id for record in records),
        judge_a_outcome=judge_a_outcome,
        judge_b_outcome=judge_b_outcome,
        error=error,
    )


async def _resolve_judge_batch(
    judge: JudgeAdapter,
    outcome: JudgeOutcome[RelevanceVerdict],
    *,
    batch: Sequence[Fact],
    rubric: Rubric,
    token_counter: TokenCounter,
    batch_index: int,
    calls: list[JudgeCallRecord],
) -> tuple[dict[int, RelevanceVerdict], frozenset[int], JudgeOutcome[RelevanceVerdict] | None]:
    """Resolve one judge's outcome for a batch.

    Returns the verdicts this judge produced, keyed 1-based by position in ``batch``; the positions
    that never got one; and the outcome that was a ``judge_unavailable``, ``unexpected`` or
    ``prompt_too_large`` failure at any point, or ``None``. A non-``None`` outcome means the caller
    treats the whole batch as failed with nothing judged. Only a schema-validation failure
    (``invalid_output``) gets the gentler path.

    - ``success`` returns every verdict.
    - ``invalid_output`` retries the identical batch call once. A retry that is neither ``success``
      nor ``invalid_output`` is a hard failure and returns at once. A retry that is also
      ``invalid_output`` falls back to one call per record (``batch_size=1``).
      ``parse_batch_verdicts`` is all-or-nothing over a response, so re-asking per record is the
      only way to find which item is bad.
    - In the fallback, a record whose single call is ``invalid_output`` is dropped (in the returned
      positions, not in the verdicts) and the loop goes on. Any other non-success single call is a
      hard failure and returns at once, spending no call on the remaining records.

    Every retry and fallback call is appended to ``calls``.
    """
    if outcome.status == "success":
        return dict(enumerate(outcome.verdicts, start=1)), frozenset(), None
    if outcome.status != "invalid_output":
        return {}, frozenset(), outcome

    retried = await judge_batch(
        judge,
        prompt=render_batch_prompt(
            rubric_prose=rubric.relevance.prose,
            contents=[record.content for record in batch],
            item_model=RelevanceVerdict,
        ),
        item_model=RelevanceVerdict,
        batch_size=len(batch),
        rubric_version=rubric.version,
        token_counter=token_counter,
    )
    calls.append(retried.as_call_record(batch_index=batch_index))
    if retried.status == "success":
        return dict(enumerate(retried.verdicts, start=1)), frozenset(), None
    if retried.status != "invalid_output":
        return {}, frozenset(), retried

    verdicts: dict[int, RelevanceVerdict] = {}
    dropped: set[int] = set()
    for pos, record in enumerate(batch, start=1):
        single = await judge_batch(
            judge,
            prompt=render_batch_prompt(
                rubric_prose=rubric.relevance.prose,
                contents=[record.content],
                item_model=RelevanceVerdict,
            ),
            item_model=RelevanceVerdict,
            batch_size=1,
            rubric_version=rubric.version,
            token_counter=token_counter,
        )
        calls.append(single.as_call_record(batch_index=batch_index))
        if single.status == "success":
            verdicts[pos] = single.verdicts[0]
        elif single.status == "invalid_output":
            dropped.add(pos)
        else:
            return verdicts, frozenset(dropped), single
    return verdicts, frozenset(dropped), None


# ---------------------------------------------------------------------------
# Assessor pre-pass and split band
# ---------------------------------------------------------------------------


def _prepass_wants_skip(verdict: Verdict | Unavailable, *, cfg: EngineConfig) -> bool:
    """Whether the Assessor pre-pass lets a record skip the panel: the skip switch is on and the
    pick is ``keep-clear`` at or above the configured cutoff. A ``shelve-clear`` pick always goes to
    the panel, because a shelve needs a judge's ``why``, and ``Unavailable`` never skips.
    """
    if not cfg.cleaner_prepass_skip_keep:
        return False
    if isinstance(verdict, Unavailable):
        return False
    if verdict.pick != "keep-clear":
        return False
    assert cfg.cleaner_prepass_skip_cutoff is not None, (
        "EngineConfig's own validator refuses cleaner_prepass_skip_keep=True with no cutoff"
    )
    return verdict.confidence >= cfg.cleaner_prepass_skip_cutoff


def _panel_matched(verdict: Verdict | Unavailable, *, final_is_keep: bool) -> bool | None:
    """``matched`` for a unanimous panel band: ``None`` when the pre-pass produced no pick
    (``Unavailable``), else whether the pre-pass's keep or shelve direction agrees with the panel.
    """
    if isinstance(verdict, Unavailable):
        return None
    return verdict.pick.startswith("keep") == final_is_keep


async def _write_cleaner_resolved_row(
    *,
    adapter: StorageAdapter,
    lock: asyncio.Lock,
    candidate_id: HypomnemaId,
    verdict: Verdict | Unavailable,
    action: CleanerAction,
    actor: VerdictActor,
    matched: bool | None,
    now: datetime,
    dry_run: bool,
) -> None:
    """Append one already-resolved ``cleaner_prepass`` row (``panel_*``, ``skipped_keep``,
    ``split_annotated``). No Assessor-caused write exists, so the pending, mutate, stamp order does
    not apply.

    A no-op under ``dry_run``: a simulated pass leaves no row that could enter statistics or be
    reused by a live revisit split. ``new_record_id`` is set to ``candidate_id`` for
    ``panel_shelve``, since ``stamp_override`` finds rows by ``effect_ref`` or ``new_record_id``
    only.
    """
    if dry_run:
        return
    row = build_cleaner_resolved_row(
        caller="cleaner_prepass",
        candidate_id=candidate_id,
        new_record_id=candidate_id if action == "panel_shelve" else None,
        verdict=verdict,
        action=action,
        actor=actor,
        matched=matched,
        now=now,
    )
    async with lock:
        await adapter.append_verdicts([row])


async def _resolve_relevance_split(
    *,
    record: Fact,
    prepass_verdict: Verdict | Unavailable | None,
    cfg: EngineConfig,
    rubric: Rubric,
    clock: Clock,
    now: datetime,
    adapter: StorageAdapter,
    lock: asyncio.Lock,
    dry_run: bool,
    review_append_failures: list[RelevanceReviewAppendFailure],
) -> SplitResolution:
    """The relevance split band: reuse the pre-pass verdict, never a second ``relevance`` call.

    Wraps ``verdicts.decide_split_action``, ``resolve_split_keep_write`` and
    ``resolve_split_shelve_write``. This function builds the fresh verdict row; the revisit axis
    restamps an existing row and passes the update write to the same two functions.

    Returns a ``SplitResolution``. ``write`` is ``"landed"`` or ``"failed"`` only for
    ``split_keep`` (the resolving review landed or not) and ``split_shelve`` (``set_shelved`` did
    or did not refuse). For every other action nothing is written here, ``write`` is ``"none"``
    and the caller's contested write runs. Under ``dry_run`` the two write actions report
    ``"landed"``.
    """
    action = decide_split_action(prepass_verdict, cfg=cfg)
    if action != "split_keep" and action != "split_shelve":
        return SplitResolution(action=action, write="none")
    assert prepass_verdict is not None and not isinstance(prepass_verdict, Unavailable)

    if dry_run:
        return SplitResolution(action=action, write="landed")

    pending = build_cleaner_pending_row(
        caller="cleaner_prepass",
        candidate_id=record.id,
        # split_shelve pre-publishes the shelved record's id as new_record_id; split_keep has a
        # single participant and nothing to link a reversal against.
        new_record_id=record.id if action == "split_shelve" else None,
        verdict=prepass_verdict,
        action=action,
        effect_ref=None,
        now=now,
    )
    history = tuple(await adapter.list_reviews_for(target_id=record.id))

    match action:
        case "split_keep":

            async def _on_review_append_failure(exc: Exception) -> None:
                review_append_failures.append(
                    RelevanceReviewAppendFailure(target_id=record.id, error=str(exc))
                )

            resolved = await resolve_split_keep_write(
                pending_row=pending,
                write_pending=adapter.append_verdicts,
                record=record,
                rubric_version=rubric.version,
                next_review_due_at=next_review_due_at(
                    now, history=history, params=rubric.parameters
                ),
                now=now,
                adapter=adapter,
                lock=lock,
                on_review_append_failure=_on_review_append_failure,
            )
        case "split_shelve":
            resolved = await resolve_split_shelve_write(
                pending_row=pending,
                write_pending=adapter.append_verdicts,
                record=record,
                reason=(
                    f"Cleaner relevance split: Assessor graded {prepass_verdict.pick} "
                    f"(confidence {prepass_verdict.confidence:.2f}). Rubric {rubric.version}."
                ),
                adapter=adapter,
                clock=clock,
                lock=lock,
            )
        case _:
            assert_never(action)
    return SplitResolution(action=action, write="landed" if resolved else "failed")


async def _run_prepass(
    *,
    batch: Sequence[Fact],
    assessor: Assessor,
    timeout_s: float,
) -> dict[HypomnemaId, Verdict | Unavailable]:
    """One ``relevance`` call per record in ``batch``, with bounded concurrency (``bounded_ask``),
    before the panel runs. The pre-pass runs whether the skip switch is on or off, so skipping has
    one code path.
    """

    def _factory(record: Fact) -> Callable[[], Awaitable[Verdict]]:
        async def _call() -> Verdict:
            return await assessor.relevance(record.content)

        return _call

    results = await bounded_ask(
        [_factory(record) for record in batch], method="relevance", timeout_s=timeout_s
    )
    return dict(zip((record.id for record in batch), results, strict=True))


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _AxisContext:
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
    now: datetime
    dry_run: bool


@dataclass(slots=True)
class _Tally:
    """The lists one pass fills. The band and split counts in the summary are derived from them, so
    no count is undone after the fact.

    ``shelved_ids`` holds both unanimous-panel shelves and Assessor split-shelves; both really
    shelved the record, so ``clean.py``'s ``_verify_writes`` re-reads both. ``counts.shelve``
    equals ``len(shelved_ids)``; ``split_shelved_ids`` is the subset from an Assessor split, and
    an id enters it only in the statement that puts it in ``shelved_ids``.
    """

    shelved_ids: list[HypomnemaId] = field(default_factory=list)
    split_shelved_ids: list[HypomnemaId] = field(default_factory=list)
    unclear_ids: list[HypomnemaId] = field(default_factory=list)
    contested_ids: list[HypomnemaId] = field(default_factory=list)
    keep_ids: list[HypomnemaId] = field(default_factory=list)
    skipped_ids: list[HypomnemaId] = field(default_factory=list)
    split_keep_ids: list[HypomnemaId] = field(default_factory=list)
    split_annotated_ids: list[HypomnemaId] = field(default_factory=list)
    failed_batches: list[BatchFailure] = field(default_factory=list)
    calls: list[JudgeCallRecord] = field(default_factory=list)
    verdicts: list[RecordVerdict] = field(default_factory=list)
    contested_verdicts: list[ContestedVerdict] = field(default_factory=list)
    full_verdicts: list[RelevanceFullVerdict] = field(default_factory=list)
    refused: list[RefusedVerdict] = field(default_factory=list)
    trimmed: list[TrimmedPair] = field(default_factory=list)
    review_append_failures: list[RelevanceReviewAppendFailure] = field(default_factory=list)
    judgment_append_failures: list[JudgmentAppendFailure] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PanelRecord:
    record: Fact
    verdict_a: RelevanceVerdict
    verdict_b: RelevanceVerdict
    judge_a_identity: str
    judge_b_identity: str
    prepass: Verdict | Unavailable | None


@dataclass(frozen=True, slots=True)
class _PanelVerdicts:
    outcome_a: JudgeOutcome[RelevanceVerdict]
    outcome_b: JudgeOutcome[RelevanceVerdict]
    verdicts_a: dict[int, RelevanceVerdict]
    verdicts_b: dict[int, RelevanceVerdict]
    dropped_a: frozenset[int]
    dropped_b: frozenset[int]


# ---------------------------------------------------------------------------
# Per-band handlers
# ---------------------------------------------------------------------------


async def _write_resolved(
    ctx: _AxisContext,
    item: _PanelRecord,
    *,
    verdict: Verdict | Unavailable,
    action: CleanerAction,
    actor: VerdictActor,
    matched: bool | None,
) -> None:
    await _write_cleaner_resolved_row(
        adapter=ctx.adapter,
        lock=ctx.lock,
        candidate_id=item.record.id,
        verdict=verdict,
        action=action,
        actor=actor,
        matched=matched,
        now=ctx.now,
        dry_run=ctx.dry_run,
    )


async def _handle_unclear(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    tally.unclear_ids.append(item.record.id)
    if item.prepass is not None:
        await _write_resolved(
            ctx,
            item,
            verdict=item.prepass,
            action="panel_unclear",
            actor="panel",
            matched=None,
        )


async def _handle_keep(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    tally.keep_ids.append(item.record.id)
    if item.prepass is not None:
        await _write_resolved(
            ctx,
            item,
            verdict=item.prepass,
            action="panel_keep",
            actor="panel",
            matched=_panel_matched(item.prepass, final_is_keep=True),
        )


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


async def _append_contested_rows(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    record = item.record
    await append_review_safely(
        adapter=ctx.adapter,
        review=await build_review_row(
            adapter=ctx.adapter,
            scope=record.scope,
            target_id=record.id,
            now=ctx.now,
            rubric=ctx.rubric,
            outcome="judged",
            band="contested",
            grades=(item.verdict_a.grade, item.verdict_b.grade),
        ),
        failures=tally.review_append_failures,
        to_failure=_review_failure,
    )
    await append_judgment_safely(
        adapter=ctx.adapter,
        judgment=RecordJudgment(
            target_id=record.id,
            scope=record.scope,
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


async def _record_contested(
    ctx: _AxisContext,
    tally: _Tally,
    item: _PanelRecord,
    *,
    action: CleanerAction | Literal["none"],
    split_keep_write_failed: bool,
) -> None:
    """Nothing is written to the record. Both judges' identity, grade and ``why`` are kept, since
    neither one decided. The review and judgment rows are skipped under ``dry_run``. The resolved
    row is skipped when the resolving keep review failed, because that attempt is already stamped
    ``failed``.
    """
    record = item.record
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
    if not ctx.dry_run:
        await _append_contested_rows(ctx, tally, item)

    if item.prepass is None or split_keep_write_failed:
        return
    assert action != "none", (
        "action is 'none' only when the pre-pass verdict is None (see _resolve_relevance_split), "
        "which this branch has already excluded"
    )
    await _write_resolved(
        ctx,
        item,
        verdict=item.prepass,
        action=action,
        actor="panel" if action == "panel_contested" else "engine",
        matched=(
            None
            if action == "panel_contested"
            else _panel_matched(item.prepass, final_is_keep=False)
        ),
    )


async def _handle_split_shelve(
    ctx: _AxisContext, tally: _Tally, item: _PanelRecord, split: SplitResolution
) -> None:
    record = item.record
    if not split.landed:
        tally.refused.append(RefusedVerdict(id=record.id, reason="split-shelve write refused"))
        return

    if not await _trim_stands(ctx, tally, item):
        return

    assert isinstance(item.prepass, Verdict)
    tally.shelved_ids.append(record.id)
    tally.split_shelved_ids.append(record.id)
    tally.verdicts.append(
        RecordVerdict(
            id=record.id,
            band="shelve",
            why=f"Assessor split-shelve vote (grade {item.prepass.pick})",
        )
    )


async def _handle_contested(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    """A split reuses the pre-pass verdict, with no second ``relevance`` call. The pre-pass verdict
    is ``None`` only when no Assessor is configured; the contested write then runs without a
    verdict row.
    """
    record = item.record
    split = await _resolve_relevance_split(
        record=record,
        prepass_verdict=item.prepass,
        cfg=ctx.cfg,
        rubric=ctx.rubric,
        clock=ctx.clock,
        now=ctx.now,
        adapter=ctx.adapter,
        lock=ctx.lock,
        dry_run=ctx.dry_run,
        review_append_failures=tally.review_append_failures,
    )
    split_keep_write_failed = False
    if split.action == "split_keep":
        if split.landed:
            tally.split_keep_ids.append(record.id)
            return
        split_keep_write_failed = True
        tally.refused.append(RefusedVerdict(id=record.id, reason="split-keep review append failed"))
    if split.action == "split_shelve":
        await _handle_split_shelve(ctx, tally, item, split)
        return
    if split.action == "split_annotated":
        tally.split_annotated_ids.append(record.id)

    await _record_contested(
        ctx, tally, item, action=split.action, split_keep_write_failed=split_keep_write_failed
    )


async def _write_panel_shelve(
    ctx: _AxisContext, tally: _Tally, item: _PanelRecord, reason: str
) -> bool:
    """The store's write-legality guard may refuse the record. Then it is skipped alone, recorded in
    ``refused``, and ``False`` returns. Other exceptions propagate.
    """
    record = item.record
    try:
        if item.prepass is not None:
            await resolve_panel_shelve_write(
                candidate_id=record.id,
                verdict=item.prepass,
                matched=_panel_matched(item.prepass, final_is_keep=False),
                reason=reason,
                now=ctx.now,
                adapter=ctx.adapter,
                clock=ctx.clock,
                lock=ctx.lock,
            )
        else:
            await set_shelved(
                id=record.id,
                shelved=True,
                reason=reason,
                adapter=ctx.adapter,
                clock=ctx.clock,
                lock=ctx.lock,
            )
    except ToolError as exc:
        tally.refused.append(RefusedVerdict(id=record.id, reason=str(exc)))
        return False
    return True


async def _handle_unanimous_shelve(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    record = item.record
    deciding = deciding_verdict(item.verdict_a, item.verdict_b)
    reason = shelve_reason(
        "relevance",
        deciding=deciding,
        deciding_judge=item.judge_a_identity,
        judge_a_identity=item.judge_a_identity,
        judge_b_identity=item.judge_b_identity,
        rubric_version=ctx.rubric.version,
    )
    if not ctx.dry_run and not await _write_panel_shelve(ctx, tally, item, reason):
        return

    if not await _trim_stands(ctx, tally, item):
        return

    tally.shelved_ids.append(record.id)
    tally.verdicts.append(RecordVerdict(id=record.id, band="shelve", why=deciding.why))


async def _handle_record(ctx: _AxisContext, tally: _Tally, item: _PanelRecord) -> None:
    band = decide_band(item.verdict_a, item.verdict_b)
    tally.full_verdicts.append(
        RelevanceFullVerdict(
            id=item.record.id,
            band=band,
            judge_a_identity=item.judge_a_identity,
            judge_a_grade=item.verdict_a.grade,
            judge_a_why=item.verdict_a.why,
            judge_a_unclear=item.verdict_a.unclear,
            judge_b_identity=item.judge_b_identity,
            judge_b_grade=item.verdict_b.grade,
            judge_b_why=item.verdict_b.why,
            judge_b_unclear=item.verdict_b.unclear,
        )
    )
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
# One batch: pre-pass, panel, failure gates
# ---------------------------------------------------------------------------


async def _skip_by_prepass(
    ctx: _AxisContext,
    tally: _Tally,
    batch: Sequence[Fact],
    prepass_by_id: dict[HypomnemaId, Verdict | Unavailable],
) -> list[Fact]:
    """The records that go on to the panel. A record the pre-pass lets skip is counted in
    ``skipped_ids`` and gets its ``skipped_keep`` row, except under ``dry_run``, where the row write
    is a no-op.
    """
    panel_batch: list[Fact] = []
    for record in batch:
        candidate_prepass = prepass_by_id[record.id]
        if not _prepass_wants_skip(candidate_prepass, cfg=ctx.cfg):
            panel_batch.append(record)
            continue

        tally.skipped_ids.append(record.id)
        await _write_cleaner_resolved_row(
            adapter=ctx.adapter,
            lock=ctx.lock,
            candidate_id=record.id,
            verdict=candidate_prepass,
            action="skipped_keep",
            actor="engine",
            matched=True,
            now=ctx.now,
            dry_run=ctx.dry_run,
        )
    return panel_batch


async def _resolve_panel_seats(
    ctx: _AxisContext,
    tally: _Tally,
    *,
    batch_index: int,
    panel_batch: list[Fact],
    outcome_a: JudgeOutcome[RelevanceVerdict],
    outcome_b: JudgeOutcome[RelevanceVerdict],
) -> _PanelVerdicts | None:
    """Both seats' verdicts after the retry and per-record fallback, or ``None`` when either seat
    hard-failed (the whole batch is then recorded in ``failed_batches``).
    """
    verdicts_a, dropped_a, hard_a = await _resolve_judge_batch(
        ctx.judge_a,
        outcome_a,
        batch=panel_batch,
        rubric=ctx.rubric,
        token_counter=ctx.token_counter,
        batch_index=batch_index,
        calls=tally.calls,
    )
    if hard_a is not None:
        tally.failed_batches.append(
            _batch_failure(
                batch_index=batch_index,
                records=panel_batch,
                judge_a_outcome=batch_outcome(hard_a.status),
                judge_b_outcome=batch_outcome(outcome_b.status),
                error=hard_a.error or "",
            )
        )
        return None

    verdicts_b, dropped_b, hard_b = await _resolve_judge_batch(
        ctx.judge_b,
        outcome_b,
        batch=panel_batch,
        rubric=ctx.rubric,
        token_counter=ctx.token_counter,
        batch_index=batch_index,
        calls=tally.calls,
    )
    if hard_b is not None:
        tally.failed_batches.append(
            _batch_failure(
                batch_index=batch_index,
                records=panel_batch,
                judge_a_outcome="invalid_output" if dropped_a else "ok",
                judge_b_outcome=batch_outcome(hard_b.status),
                error=hard_b.error or "",
            )
        )
        return None

    return _PanelVerdicts(
        outcome_a=outcome_a,
        outcome_b=outcome_b,
        verdicts_a=verdicts_a,
        verdicts_b=verdicts_b,
        dropped_a=dropped_a,
        dropped_b=dropped_b,
    )


def _record_dropped(
    tally: _Tally, *, batch_index: int, panel_batch: list[Fact], panel: _PanelVerdicts
) -> None:
    dropped_positions = panel.dropped_a | panel.dropped_b
    if not dropped_positions:
        return
    tally.failed_batches.append(
        _batch_failure(
            batch_index=batch_index,
            records=tuple(panel_batch[pos - 1] for pos in sorted(dropped_positions)),
            judge_a_outcome="invalid_output" if panel.dropped_a else "ok",
            judge_b_outcome="invalid_output" if panel.dropped_b else "ok",
            error=(
                f"{len(dropped_positions)} record(s) never produced a valid verdict from "
                "this judge after a batch retry and a per-record fallback -- dropped from "
                "this pass"
            ),
        )
    )


async def _judge_panel_batch(
    ctx: _AxisContext, tally: _Tally, *, batch_index: int, panel_batch: list[Fact]
) -> _PanelVerdicts | None:
    """Run both seats on ``panel_batch``. ``None`` when the batch failed outright: nothing in a
    failed batch is written, so there is no partial application.
    """
    prompt = render_batch_prompt(
        rubric_prose=ctx.rubric.relevance.prose,
        contents=[record.content for record in panel_batch],
        item_model=RelevanceVerdict,
    )
    outcome_a, outcome_b = await run_panel(
        judge_a=ctx.judge_a,
        judge_b=ctx.judge_b,
        prompt=prompt,
        item_model=RelevanceVerdict,
        batch_size=len(panel_batch),
        rubric_version=ctx.rubric.version,
        token_counter=ctx.token_counter,
        batch_index=batch_index,
        calls=tally.calls,
    )

    usable = ("success", "invalid_output")
    if outcome_a.status not in usable or outcome_b.status not in usable:
        errors = [o.error for o in (outcome_a, outcome_b) if o.error is not None]
        tally.failed_batches.append(
            _batch_failure(
                batch_index=batch_index,
                records=panel_batch,
                judge_a_outcome=batch_outcome(outcome_a.status),
                judge_b_outcome=batch_outcome(outcome_b.status),
                error="; ".join(errors),
            )
        )
        return None

    panel = await _resolve_panel_seats(
        ctx,
        tally,
        batch_index=batch_index,
        panel_batch=panel_batch,
        outcome_a=outcome_a,
        outcome_b=outcome_b,
    )
    if panel is not None:
        _record_dropped(tally, batch_index=batch_index, panel_batch=panel_batch, panel=panel)
    return panel


async def _run_batch(
    ctx: _AxisContext, tally: _Tally, *, batch_index: int, batch: list[Fact]
) -> None:
    prepass_by_id: dict[HypomnemaId, Verdict | Unavailable] = {}
    panel_batch = batch
    if ctx.assessor is not None:
        prepass_by_id = await _run_prepass(
            batch=batch, assessor=ctx.assessor, timeout_s=ctx.cfg.assessor_cleaner_timeout_s
        )
        panel_batch = await _skip_by_prepass(ctx, tally, batch, prepass_by_id)
    if not panel_batch:
        return

    panel = await _judge_panel_batch(ctx, tally, batch_index=batch_index, panel_batch=panel_batch)
    if panel is None:
        return

    dropped_positions = panel.dropped_a | panel.dropped_b
    for pos, record in enumerate(panel_batch, start=1):
        if pos in dropped_positions:
            continue
        item = _PanelRecord(
            record=record,
            verdict_a=panel.verdicts_a[pos],
            verdict_b=panel.verdicts_b[pos],
            judge_a_identity=panel.outcome_a.judge,
            judge_b_identity=panel.outcome_b.judge,
            prepass=prepass_by_id.get(record.id),
        )
        await _handle_record(ctx, tally, item)


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


async def _select_run_pool(
    *,
    scope: Scope,
    since: datetime,
    adapter: StorageAdapter,
    pool_limit: int,
    dry_run: bool,
    explicit_pool: Sequence[Fact] | None,
) -> tuple[int, _RelevancePool]:
    """The pool size and the selected candidates: ``explicit_pool`` as given, or the born-since
    pool after ``_select_relevance_pool``.
    """
    if explicit_pool is None:
        raw_pool = await adapter.list_born_since(scope=scope, since=since, limit=pool_limit)
        return len(raw_pool), _select_relevance_pool(raw_pool, pass_scope=scope)

    if not dry_run:
        raise ToolError(
            "run_relevance_axis: explicit_pool requires dry_run=True -- it bypasses "
            "_select_relevance_pool's scope/type/lifecycle exclusions entirely, so a live write "
            "from it would skip every guard those exclusions exist to enforce"
        )
    selected = _RelevancePool(
        candidates=tuple(explicit_pool),
        excluded_decisions=(),
        excluded_global=(),
        excluded_already_shelved=(),
    )
    return len(explicit_pool), selected


def _build_summary(
    ctx: _AxisContext,
    tally: _Tally,
    *,
    scope: Scope,
    pool_size: int,
    selected: _RelevancePool,
) -> RelevanceAxisSummary:
    return RelevanceAxisSummary(
        scope=scope,
        rubric_version=ctx.rubric.version,
        judge_pair=(ctx.judge_a.identity, ctx.judge_b.identity),
        pool_size=pool_size,
        excluded_decisions=selected.excluded_decisions,
        excluded_global=selected.excluded_global,
        excluded_already_shelved=selected.excluded_already_shelved,
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
        full_verdicts=tuple(tally.full_verdicts),
        refused=tuple(tally.refused),
        review_append_failures=tuple(tally.review_append_failures),
        judgment_append_failures=tuple(tally.judgment_append_failures),
        skipped=len(tally.skipped_ids),
        split_keep=len(tally.split_keep_ids),
        split_annotated=len(tally.split_annotated_ids),
        split_shelve=len(tally.split_shelved_ids),
        trimmed=tuple(tally.trimmed),
    )


async def run_relevance_axis(
    *,
    scope: Scope,
    since: datetime,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
    cfg: EngineConfig,
    rubric: Rubric,
    judge_a: JudgeAdapter,
    judge_b: JudgeAdapter,
    assessor: Assessor | None = None,
    pool_limit: int | None = None,
    dry_run: bool = False,
    explicit_pool: Sequence[Fact] | None = None,
    embedder: Embedder | None = None,
    require_scope: RequireScope | None = None,
) -> RelevanceAxisSummary:
    """Judge the Facts born in ``scope`` since ``since`` and shelve the ones that fail the admission
    gate. The module docstring covers pool selection, the panel and band policy.

    One ``LitellmTokenCounter`` is created for the run and passed to every ``judge_batch`` call. It
    makes no network call.

    ``pool_limit``: ``None`` reads ``rubric.parameters.pool_limit``, the value ``engine/clean.py``
    also uses for its own snapshot.

    ``dry_run``: judging, batching and band decisions are identical, and nothing is written: no
    ``set_shelved``, no verdict rows, no trim save, no contested review or judgment row.
    ``shelved_ids`` still lists what the pass would shelve and ``verdicts`` carries each deciding
    ``why``.

    A shelve write that ``set_shelved`` refuses (``ToolError``) is caught and reported in
    ``refused``; see the module docstring. Any other exception propagates.

    ``embedder`` and ``require_scope`` serve trim saves (``engine/cleaner_writes.py``). A shelved
    record whose shelve-voting judges (seat A first) name a surviving part in ``keep`` has that part
    saved as a new Fact. ``embedder=None`` turns the save off and the shelve stands. The save never
    runs while holding ``lock``.

    ``explicit_pool``: ``None`` uses ``adapter.list_born_since`` and ``_select_relevance_pool``.
    When given, both are skipped and the Facts are judged as they are, with no scope, type or
    lifecycle filtering; ``scope`` and ``since`` play no part in the pool, ``pool_size`` is the
    length of ``explicit_pool`` and every ``excluded_*`` tuple is empty. A calibration harness needs
    this to re-judge an already-shelved batch. Combining it with ``dry_run=False`` raises
    ``ToolError`` before anything else runs, because the write site does not re-check scope, type or
    lifecycle.
    """
    effective_pool_limit = pool_limit if pool_limit is not None else rubric.parameters.pool_limit
    pool_size, selected = await _select_run_pool(
        scope=scope,
        since=since,
        adapter=adapter,
        pool_limit=effective_pool_limit,
        dry_run=dry_run,
        explicit_pool=explicit_pool,
    )
    ctx = _AxisContext(
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
        now=clock.now(),
        dry_run=dry_run,
    )
    tally = _Tally()

    candidate_groups = pack_batches(
        [candidate.content for candidate in selected.candidates],
        count_cap=rubric.parameters.batch_cap,
        rubric_prose=rubric.relevance.prose,
        item_model=RelevanceVerdict,
    )
    for batch_index, group in enumerate(candidate_groups):
        batch = [selected.candidates[position] for position in group]
        await _run_batch(ctx, tally, batch_index=batch_index, batch=batch)

    return _build_summary(ctx, tally, scope=scope, pool_size=pool_size, selected=selected)
