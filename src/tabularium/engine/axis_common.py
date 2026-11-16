"""Pieces the relevance, revisit and falsity axes share: the judge pair call, the status fold, the
relevance/revisit band policy and shelve reason, the best-effort audit appends, the review-row
builder and the summary classes whose fields are identical across axes.

Each axis keeps its own failure classification after ``run_panel``. Only the relevance axis retries
``invalid_output``, so ``run_panel`` returns both outcomes and decides nothing about failure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Literal, Self, assert_never

from pydantic import BaseModel, Field, model_validator

from tabularium.domain import BatchOutcome, HypomnemaId, ReviewBand, Scope
from tabularium.engine.judge import (
    IndexedVerdictBase,
    JudgeAdapter,
    JudgeCallRecord,
    JudgeOutcome,
    JudgeOutcomeStatus,
    UnclearVerdictBase,
    judge_batch,
)
from tabularium.engine.review_schedule import next_review_due_at
from tabularium.engine.rubric import Rubric
from tabularium.storage.audit import ReviewOutcome
from tabularium.storage.types import RecordJudgment, RecordReview, StorageAdapter
from tabularium.tokens import TokenCounter

# ---------------------------------------------------------------------------
# The shelve verdict schema shared by the relevance and revisit panels
# ---------------------------------------------------------------------------


class ShelveVerdict(UnclearVerdictBase):
    """One judge's verdict on one record in a relevance or revisit batch.

    ``grade`` is the rubric's four-point ordered scale, worst to best. The two ``shelve-*`` grades
    and the two ``keep-*`` grades collapse to a boolean vote (``votes_shelve``) for band arithmetic,
    and the grade is kept for the reason string. The inherited ``unclear`` flag is never acted on,
    whatever ``grade`` says (``decide_band``).
    """

    reply_rules = (
        "`keep` goes only on a `shelve-*` grade, and only when one part of the record would pass "
        "the gate on its own. Omit it when nothing of the record survives on its own, and never "
        "put it on a `keep-*` grade.",
    )

    grade: Literal["shelve-clear", "shelve-lean", "keep-lean", "keep-clear"]
    why: str = Field(
        min_length=1, description="The gate question or keep-test clause that decided it."
    )
    keep: str | None = Field(
        default=None,
        description=(
            "The part of this record that still passes the gate on its own, restated as a "
            "standalone record."
        ),
    )

    @property
    def votes_shelve(self) -> bool:
        return self.grade.startswith("shelve")

    @model_validator(mode="after")
    def _normalize_keep(self) -> Self:
        if self.keep is not None:
            self.keep = self.keep.strip() or None
        if self.keep is not None and not self.votes_shelve:
            self.keep = None
        return self


# ---------------------------------------------------------------------------
# Summary classes identical in the relevance and revisit axes (persisted through the report)
# ---------------------------------------------------------------------------


class BandCounts(BaseModel):
    model_config = {"frozen": True}

    shelve: int = 0
    keep: int = 0
    unclear: int = 0
    # A split verdict: the judges did judge and disagreed. Distinct from `unclear`.
    contested: int = 0


class BatchFailure(BaseModel):
    """A group of records this pass never judged; nothing in ``record_ids`` was written.

    ``record_ids`` is the whole batch for a ``judge_unavailable``, ``unexpected`` or
    ``prompt_too_large`` outcome. The relevance axis can also report a subset for
    ``invalid_output``: the records that still failed validation after a batch retry and a
    per-record fallback.
    """

    model_config = {"frozen": True}

    batch_index: int
    record_ids: tuple[HypomnemaId, ...]
    judge_a_outcome: BatchOutcome
    judge_b_outcome: BatchOutcome
    error: str


class RecordVerdict(BaseModel):
    """One shelved record's deciding verdict: id, band and the deciding judge's ``why``.

    Only the ``shelve`` band has a deciding verdict. A keep is unanimous, ``unclear`` is in
    ``unclear_ids``, and a split is contested (``ContestedVerdict``). It is populated under
    ``dry_run`` too, so a dry run can report which records would be shelved and why.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    band: Literal["shelve"]
    why: str


class ContestedVerdict(BaseModel):
    """One contested record's two disagreeing verdicts. Neither judge decided and nothing was
    written, so both judges' identity, grade and ``why`` are kept.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    band: Literal["contested"] = "contested"
    judge_a_identity: str
    judge_a_grade: str
    judge_a_why: str
    judge_b_identity: str
    judge_b_grade: str
    judge_b_why: str


class RefusedVerdict(BaseModel):
    """One record the pass meant to shelve whose write did not stand: not shelved, and surfaced
    here.

    ``reason`` is the ``ToolError`` text for a legality refusal, or a short description when a split
    or trim write did not stand.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    reason: str


class JudgmentAppendFailure(BaseModel):
    """One record whose ``RecordJudgment`` append failed, from a contested verdict (relevance and
    revisit) or a contested pair participant (falsity).

    The contested verdict stands. Only the judge detail in the separate append-only
    ``record_judgments`` table is missing. This is distinct from ``review_append_failures``, which
    concerns ``record_reviews``.
    """

    model_config = {"frozen": True}

    target_id: HypomnemaId
    error: str


# ---------------------------------------------------------------------------
# The judge pair and the status fold
# ---------------------------------------------------------------------------


async def run_panel[V: IndexedVerdictBase](
    *,
    judge_a: JudgeAdapter,
    judge_b: JudgeAdapter,
    prompt: str,
    item_model: type[V],
    batch_size: int,
    rubric_version: str,
    token_counter: TokenCounter,
    batch_index: int,
    calls: list[JudgeCallRecord],
) -> tuple[JudgeOutcome[V], JudgeOutcome[V]]:
    """Call both judges concurrently with the identical ``prompt`` and return their outcomes.

    ``judge_batch`` returns an outcome for any ``Exception``, so the gather needs no
    ``return_exceptions=True``. Both seats' call records are appended to ``calls``, seat A first,
    before this returns, so a caller that skips a failed batch cannot drop a failed call. Failure
    classification is the caller's.
    """
    outcome_a, outcome_b = await asyncio.gather(
        judge_batch(
            judge_a,
            prompt=prompt,
            item_model=item_model,
            batch_size=batch_size,
            rubric_version=rubric_version,
            token_counter=token_counter,
        ),
        judge_batch(
            judge_b,
            prompt=prompt,
            item_model=item_model,
            batch_size=batch_size,
            rubric_version=rubric_version,
            token_counter=token_counter,
        ),
    )

    calls.append(outcome_a.as_call_record(batch_index=batch_index))
    calls.append(outcome_b.as_call_record(batch_index=batch_index))

    return outcome_a, outcome_b


def batch_outcome(status: JudgeOutcomeStatus) -> BatchOutcome:
    """Fold ``judge_batch``'s status onto the ``BatchOutcome`` literal.

    The ``match`` is exhaustive over ``JudgeOutcomeStatus``, so a new status fails mypy at the
    ``assert_never``. ``unexpected`` and ``prompt_too_large`` fold into ``unavailable``: either
    fails the batch the way a diagnosed unavailability does. The error text survives in the batch
    failure's ``error`` and the distinct status in ``JudgeOutcome.status``.
    """
    match status:
        case "success":
            return "ok"
        case "judge_unavailable" | "unexpected" | "prompt_too_large":
            return "unavailable"
        case "invalid_output":
            return "invalid_output"
        case _:
            assert_never(status)


# ---------------------------------------------------------------------------
# Band policy and shelve reason for the relevance and revisit axes
# ---------------------------------------------------------------------------


def decide_band(verdict_a: ShelveVerdict, verdict_b: ShelveVerdict) -> ReviewBand:
    """``unclear`` from either judge wins. Otherwise two shelve votes give ``"shelve"``, two keep
    votes give ``"keep"``, and a split gives ``"contested"``, which writes nothing to the record.
    """
    if verdict_a.unclear or verdict_b.unclear:
        return "unclear"
    if verdict_a.votes_shelve and verdict_b.votes_shelve:
        return "shelve"
    if verdict_a.votes_shelve or verdict_b.votes_shelve:
        return "contested"
    return "keep"


def deciding_verdict(verdict_a: ShelveVerdict, verdict_b: ShelveVerdict) -> ShelveVerdict:
    """The verdict whose ``why`` names the deciding gate question for a shelve write.

    Callers ask only for the ``"shelve"`` band, where both judges voted shelve, so judge A's verdict
    decides and ``verdict_b`` is not consulted.
    """
    return verdict_a


def shelve_reason(
    axis: Literal["relevance", "revisit"],
    *,
    deciding: ShelveVerdict,
    deciding_judge: str,
    judge_a_identity: str,
    judge_b_identity: str,
    rubric_version: str,
) -> str:
    """Build ``set_shelved``'s required non-empty ``reason`` from the deciding judge's ``why``, the
    judge pair and the rubric version. ``axis`` names the calling axis in the stored text.
    """
    return (
        f"Cleaner {axis} axis: {deciding_judge} graded {deciding.grade} — "
        f'"{deciding.why}". Judge panel: {judge_a_identity} + {judge_b_identity}. '
        f"Rubric {rubric_version}."
    )


# ---------------------------------------------------------------------------
# Best-effort audit appends and the review row
# ---------------------------------------------------------------------------


async def append_review_safely[F](
    *,
    adapter: StorageAdapter,
    review: RecordReview,
    failures: list[F],
    to_failure: Callable[[RecordReview, str], F],
) -> None:
    """Append one ``RecordReview``. An ``Exception`` is recorded in ``failures`` as
    ``to_failure(review, str(exc))`` and not raised.

    The caller has already reflected the record's decision in its summary lists, so an audit-row
    failure must not unwind that decision or abort the pass and discard the other records it
    processed. ``to_failure`` is the axis's own failure class because the axes record different
    fields.
    """
    try:
        await adapter.append_review(review)
    except Exception as exc:  # broad on purpose: a failed append is recorded, not raised
        failures.append(to_failure(review, str(exc)))


async def append_judgment_safely(
    *,
    adapter: StorageAdapter,
    judgment: RecordJudgment,
    failures: list[JudgmentAppendFailure],
) -> None:
    """Append one ``RecordJudgment``. An ``Exception`` is recorded in ``failures`` and not raised.
    Same contract as ``append_review_safely``, for the ``record_judgments`` table.
    """
    try:
        await adapter.append_record_judgment(judgment)
    except Exception as exc:  # broad on purpose: a failed append is recorded, not raised
        failures.append(JudgmentAppendFailure(target_id=judgment.target_id, error=str(exc)))


async def build_review_row(
    *,
    adapter: StorageAdapter,
    scope: Scope,
    target_id: HypomnemaId,
    now: datetime,
    rubric: Rubric,
    outcome: ReviewOutcome,
    band: ReviewBand | None,
    grades: tuple[str, ...],
) -> RecordReview:
    """Read ``target_id``'s review history, then build its next ``RecordReview``.

    The history read comes first, then ``next_review_due_at`` computes the due date from it.
    ``scope`` is the row's own scope: the revisit axis passes the pass scope, relevance and falsity
    pass the record's.
    """
    history = tuple(await adapter.list_reviews_for(target_id=target_id))

    return RecordReview(
        target_id=target_id,
        scope=scope,
        reviewed_at=now,
        next_review_due_at=next_review_due_at(now, history=history, params=rubric.parameters),
        outcome=outcome,
        band=band,
        grades=grades,
        rubric_version=rubric.version,
    )
