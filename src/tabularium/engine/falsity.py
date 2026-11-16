"""The Cleaner's falsity axis: a pair-contradiction sweep that draws Fact supersession edges.

Falsity is extrinsic: no judge can tell from "the server listens on 127.0.0.1:8787" alone that it
has gone false. Framed as a pair, a newer record N against an older one O it may have overtaken,
falsity is intrinsic to the text again, and the blind batched panel of the relevance axis
(``engine/relevance.py``) applies.

Candidate generation: for each newcomer N, probe backwards with ``engine.gate.probe_candidates``,
never ``recall()`` (a fallback hit revives an archived record, so scanning would write). The probe
uses the rubric's ``k`` and ``similarity_floor``, valid records only, and the span any read in this
scope uses, the scope plus ``global`` (``engine.filters.read_span``).

Pair formation: a pair is ``(N, O)`` with ``O.created_at < N.created_at``. Direction is structural
and never judged. A candidate that is not strictly older is discarded, neither judged nor reported.

Candidates reported instead of written:

- O is a Decision. Superseding a Decision writes a new record with new content, so an edge cannot be
  drawn between two existing records. Reported as ``candidate_is_decision``, never judged.
- O is in ``global`` and the pass is a project pass. Without an Assessor it is reported as
  ``candidate_is_global`` and never judged. With one, the pair goes through the panel, and a
  non-coexist outcome is written only when Jev ``breadth`` on the newcomer is ``owner_wide``
  (``_global_write_allowed``); otherwise it is reported as ``candidate_is_global``. An unavailable
  ``breadth`` call records the pair in ``failed_batches``. In a ``global`` pass the pair is
  ordinary.
- O's lifecycle is not ``active`` or ``archived`` (for example ``hot`` or ``ember``). Reported as
  ``candidate_is_hot_or_ember``, never judged or superseded. The probe returns such Facts, so this
  module classifies them out.

Adjudication: only a genuine pair reaches a judge. The panel decides whether N replaces O or the two
coexist. The question is replacement, not logical incompatibility, since two consistent records can
still have one replace the other. A split (one ``replace``, one ``coexist``) is ``contested``:
reported, no edge drawn, no judge's vote applied. There is no ``unclear`` on this axis;
``FalsityVerdict`` extends ``IndexedVerdictBase`` directly, and ``_check_unclear_is_strict`` passes
a schema with no ``unclear`` field.

``_classify_pair`` is reason-aware. Two ``coexist`` votes, or two ``replace`` votes with the same
reason, are agreement and need no further call. Two ``replace`` votes with different reasons are a
``reason_disagreement``, and one of each is an ``action_split``. With an Assessor, both are settled
by a fresh ``contradiction`` call that carries both judges' votes and reasoning
(``_resolve_falsity_assessor_call``). Without one, a ``reason_disagreement`` is written with judge
A's reason and an ``action_split`` is contested.

Newcomers the caller bars: ``excluded_ids`` removes records from the newcomer pool. ``clean.py``
passes the relevance axis's shelved, contested, unclear and refused ids, so a newcomer in any of
those groups cannot supersede an older record in the same pass. This module does not import
``engine/relevance.py``.

The write, without an Assessor decision: ``engine.dispositions.supersede(successor_id=N,
predecessor_ids=[O], ...)`` through ``_write_falsity_edge``. N is the successor and O the sole
predecessor, never the reverse; a test pins that direction against the write call. The reason is the
judges' own reason class. A cross-scope write into ``global`` uses reason ``moved`` and a note.
``span`` is ``[scope]``, or ``[scope, global]`` for a pair the breadth gate allowed, matching
``save(kind="fact", supersedes=[...])`` in ``engine/facts.py``. ``supersede``'s own cross-scope
guard is a second line of defence behind the candidate classification, and the two are tested
independently.

The write, after an Assessor decision: ``_arbitrate_replace``, called by
``_resolve_falsity_assessor_call``, holds ``lock`` across the pending verdict row,
``validate_supersede``, ``supersede_write_locked`` and the stamp, so no concurrent writer lands an
override between the write and the stamp. It uses those lock-free pieces because ``supersede``
acquires ``lock`` itself.

Locking: ``supersede`` and the merge save (``cleaner_writes.apply_merge``) take ``lock`` themselves,
and ``asyncio.Lock`` is not reentrant. This module therefore never holds ``lock`` around a call to
either, and never calls ``supersede`` or ``set_shelved`` inside a ``lock`` hold.

Judge failures. A batch where either seat's outcome is not ``success`` (``judge_unavailable``,
``invalid_output``, ``unexpected`` or ``prompt_too_large``) writes nothing and is recorded in
``failed_batches``. Unlike the relevance axis there is no retry and no per-record fallback. Every
call is also collected in ``FalsityAxisSummary.calls``, as in ``RelevanceAxisSummary.calls``.

The axis does not call ``recall()`` or ``bury``, writes no Decision, writes into ``global`` from a
project pass only through the breadth gate, and supersedes no ``hot`` or ``ember`` Fact on either
side of a pair. The newcomer filter and the candidate classification make this structural.

A refused write skips the pair, not the pass. ``supersede`` raises ``ToolError`` for legality
reasons the candidate classification does not pre-empt. The concrete case is two newcomers in one
pass that both replace the same older candidate: the first supersedes it, and the second raises
because only a valid Fact can be superseded. The axis catches the ``ToolError``, draws no edge, does
not count the pair toward ``replace``, and records it in ``FalsityAxisSummary.refused``. The rest of
the batch and every later batch continue.

NOTE: the refusal boundary is ``_is_pre_write_refusal`` (``exc.__cause__ is None``) and is coupled
to exception chaining.
``supersede`` raises its pre-write legality checks bare, but converts a ``KeyError`` or
``ValueError`` from ``adapter.transition`` into a ``ToolError`` chained with ``from exc``. A
``ToolError`` with a cause is re-raised and aborts the pass. That includes an ordinary lost race as
well as a data-integrity failure, and it is the conservative side: a false abort retries on the next
pass, while a false skip would hide corruption. Any exception other than ``ToolError`` also
propagates.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, assert_never
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import (
    AssessorPick,
    BatchOutcome,
    Decision,
    Fact,
    Hypomnema,
    HypomnemaId,
    InvalidationReason,
    Scope,
    SupersessionReason,
)
from tabularium.embedding import Embedder
from tabularium.engine._errors import ToolError
from tabularium.engine._types import RequireScope
from tabularium.engine.admission import ask, bounded_ask
from tabularium.engine.assessor import (
    Assessor,
    ContradictionVerdict,
    PanelDisagreement,
    PanelJudgeAnswer,
    Unavailable,
)
from tabularium.engine.axis_common import (
    JudgmentAppendFailure,
    append_judgment_safely,
    append_review_safely,
    batch_outcome,
    build_review_row,
    run_panel,
)
from tabularium.engine.cleaner_writes import (
    MergeApplied,
    MergedTriple,
    MergeRevertedEntry,
    apply_merge,
    pick_carry,
)
from tabularium.engine.dispositions import (
    supersede,
    supersede_write_locked,
    validate_supersede,
)
from tabularium.engine.filters import GLOBAL, read_span
from tabularium.engine.gate import probe_candidates
from tabularium.engine.judge import (
    IndexedVerdictBase,
    JudgeAdapter,
    JudgeCallRecord,
    pack_batches,
    render_batch_prompt,
)
from tabularium.engine.rubric import Rubric
from tabularium.engine.verdicts import (
    build_cleaner_pending_row,
    build_cleaner_resolved_row,
    stamp_acted_many,
    stamp_failed_many,
)
from tabularium.storage.types import RecordJudgment, RecordReview, StorageAdapter
from tabularium.tokens import LitellmTokenCounter, TokenCounter

# `pool_limit` defaults to `Rubric.parameters.pool_limit`, as in `engine/relevance.py`.


# ---------------------------------------------------------------------------
# The verdict schema: extends IndexedVerdictBase directly (no `unclear` field)
# ---------------------------------------------------------------------------


class FalsityVerdict(IndexedVerdictBase):
    """One judge's verdict on one (newer N, older O) pair in a falsity batch.

    ``verdict`` is the rubric's two-point forced choice, with no ``unclear``. ``why`` names what
    changed between O and N, or what keeps them both standing. ``reason`` is the judge's
    supersession reason class, required exactly when ``verdict == "replace"`` and forbidden
    otherwise. The write uses it, and two ``replace`` votes with different reasons are a
    ``reason_disagreement``.
    """

    reply_rules = (
        "`reason` is required when `verdict` is `replace` and omitted when it is `coexist`. "
        "Use `correction` if the older record was wrong when it was written, `changed` if it was "
        "true then and what it describes has since moved on.",
        "`carry` goes only when `verdict` is `replace`, and only when the older record holds a "
        "separate point that is still true and that the newer record does not say. Omit it "
        "otherwise.",
    )

    verdict: AssessorPick
    reason: InvalidationReason | None = None
    why: str = Field(
        min_length=1, description="What changed between O and N, or what keeps them both standing."
    )
    carry: str | None = Field(
        default=None,
        description=(
            "What of the older record still holds and the newer one does not say, in one or two "
            "sentences that read naturally appended to the newer record."
        ),
    )

    @property
    def votes_replace(self) -> bool:
        return self.verdict == "replace"

    @model_validator(mode="after")
    def _reason_matches_verdict(self) -> FalsityVerdict:
        if self.verdict == "replace" and self.reason is None:
            raise ValueError("verdict='replace' requires reason")
        if self.verdict == "coexist" and self.reason is not None:
            raise ValueError("verdict='coexist' must not carry reason")
        return self

    @model_validator(mode="after")
    def _normalize_carry(self) -> FalsityVerdict:
        if self.carry is not None:
            self.carry = self.carry.strip() or None
        if self.carry is not None and self.verdict != "replace":
            self.carry = None
        return self


# ---------------------------------------------------------------------------
# Structured summary: this axis reports, `engine/report.py` builds the report
# ---------------------------------------------------------------------------


class FalsityBandCounts(BaseModel):
    model_config = {"frozen": True}

    replace: int = 0
    coexist: int = 0
    # A split verdict: the panel judged and disagreed; no edge is drawn either way.
    contested: int = 0


class FalsityEdge(BaseModel):
    """One drawn ``supersede`` edge. The older record's id is ``id`` and the newer's is
    ``superseded_by``, matching the write's argument names so summary fields cannot be mis-mapped
    onto the write call.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    superseded_by: HypomnemaId
    similarity: float
    deciding_judge: str
    why: str


class FalsityReportEntry(BaseModel):
    """A detected pair that was never written.

    ``candidate_is_decision`` and ``candidate_is_hot_or_ember`` pairs are never sent to a judge.
    ``candidate_is_global`` pairs are reported either unjudged (no Assessor) or after the panel
    judged them and the breadth gate refused the write. ``assessor_pick`` is set only on
    ``candidate_is_global`` entries when an Assessor is configured: its ``contradiction`` pick, or
    ``"unavailable"``.
    """

    model_config = {"frozen": True}

    newcomer_id: HypomnemaId
    candidate_id: HypomnemaId
    similarity: float
    reason: Literal["candidate_is_decision", "candidate_is_global", "candidate_is_hot_or_ember"]
    assessor_pick: str | None = None


class FalsityContestedPair(BaseModel):
    """One contested pair: the panel split, so no edge is drawn and the pair is reported instead.
    Both judges' identity, verdict and ``why`` are kept, since neither one decided; this mirrors
    ``RelevanceContestedVerdict``.

    Named like ``FalsityReportEntry`` (``newcomer_id``/``candidate_id``) and not like
    ``FalsityEdge``, because a contested pair is never written.
    """

    model_config = {"frozen": True}

    newcomer_id: HypomnemaId
    candidate_id: HypomnemaId
    similarity: float
    judge_a_identity: str
    judge_a_verdict: str
    judge_a_why: str
    judge_b_identity: str
    judge_b_verdict: str
    judge_b_why: str


class FalsityRefusedPair(BaseModel):
    """One pair that was to be replaced and whose write did not stand: no edge, and reported here.

    ``reason`` is the ``ToolError`` text for a legality refusal (for example two newcomers in one
    pass targeting the same older record), or a short description for an Assessor-path write that
    did not stand. Named like ``FalsityContestedPair`` and not like ``FalsityEdge``.
    """

    model_config = {"frozen": True}

    newcomer_id: HypomnemaId
    candidate_id: HypomnemaId
    reason: str


class FalsityReviewAppendFailure(BaseModel):
    """One record whose ``RecordReview`` append failed, from a contested pair. Mirrors
    ``engine.relevance.RelevanceReviewAppendFailure``.

    The axis appends one row per participant, so the two appends for a pair fail and are reported
    independently; ``target_id`` is the participant this failure concerns.
    """

    model_config = {"frozen": True}

    target_id: HypomnemaId
    error: str


class FalsityBatchFailure(BaseModel):
    """One batch whose judge call failed, or one pair whose ``breadth`` check was unavailable.
    Nothing in it was written.
    """

    model_config = {"frozen": True}

    batch_index: int
    pairs: tuple[tuple[HypomnemaId, HypomnemaId], ...]  # (newcomer_id, candidate_id)
    judge_a_outcome: BatchOutcome
    judge_b_outcome: BatchOutcome
    error: str


class FalsityAxisSummary(BaseModel):
    """Structured result of one falsity-axis pass. ``engine/clean.py`` reads it to decide whether
    the watermark may advance and to verify writes; ``engine/report.py`` builds the report from it.
    """

    model_config = {"frozen": True}

    scope: Scope
    rubric_version: str
    judge_pair: tuple[str, str]
    newcomer_pool_size: int
    excluded_decisions: tuple[HypomnemaId, ...]
    excluded_global: tuple[HypomnemaId, ...]
    excluded_shelved_this_pass: tuple[HypomnemaId, ...]
    counts: FalsityBandCounts
    edges: tuple[FalsityEdge, ...]
    decision_candidates: tuple[FalsityReportEntry, ...]
    global_candidates: tuple[FalsityReportEntry, ...]
    # Candidate O was not active or archived (for example hot or ember): reported, never judged
    # or superseded. Same shape as `decision_candidates` and `global_candidates`.
    hot_or_ember_candidates: tuple[FalsityReportEntry, ...]
    # Split verdicts: judged, disagreed, no edge drawn. The three candidate lists above were never
    # judged.
    contested_pairs: tuple[FalsityContestedPair, ...] = ()
    failed_batches: tuple[FalsityBatchFailure, ...]
    # One entry per `judge_batch` call actually made; mirrors `RelevanceAxisSummary.calls`.
    calls: tuple[JudgeCallRecord, ...]
    # Replace writes the panel decided on that did not stand; see `FalsityRefusedPair`. Defaulted
    # so older persisted rows still deserialize.
    refused: tuple[FalsityRefusedPair, ...] = ()
    # Contested-pair `RecordReview` appends that failed; see `FalsityReviewAppendFailure`.
    review_append_failures: tuple[FalsityReviewAppendFailure, ...] = ()
    # Contested-pair `RecordJudgment` appends that failed; see
    # `JudgmentAppendFailure`.
    judgment_append_failures: tuple[JudgmentAppendFailure, ...] = ()
    # Assessor-caused actions, all 0 when no Assessor is configured. `split_supersede` and
    # `reason_by_assessor` are already counted in `counts.replace` (a real edge is drawn either way,
    # and a reverted merge is in neither); `split_coexist` is already in `counts.coexist`.
    # They are detail, not a second tally.
    split_supersede: int = 0
    split_coexist: int = 0
    reason_by_assessor: int = 0
    # Replace pairs whose judges named a still-true part of O in `carry`, saved as a new record
    # that supersedes N (`engine/cleaner_writes.py`). `merged` is a subset of `counts.replace`.
    # `merge_reverted` are merges whose save bounced or raised: the O -> N edge was undone, both
    # records stayed live, and the pair is absent from `edges` and from `counts.replace`.
    # `merge_revert_failed` are merges whose save failed and whose undo also failed in storage: the
    # edge stands, stays in `edges` and in `counts.replace`, and is never in `merge_reverted`.
    # Outside a dry run `counts.replace == len(edges)`; rows persisted before that rule count a
    # reverted merge in `replace`.
    #
    # Reconciliation over the pairs sent to the judges: `counts.replace + len(merge_reverted) +
    # counts.coexist + counts.contested + len(refused) + <every pair across failed_batches> + <every
    # pair routed to global_candidates by the breadth gate>` equals the number of pairs sent.
    # `newcomer_pool_size == len(excluded_decisions) + len(excluded_global) +
    # len(excluded_shelved_this_pass) + <the newcomers>`.
    merged: tuple[MergedTriple, ...] = ()
    merge_reverted: tuple[MergeRevertedEntry, ...] = ()
    merge_revert_failed: tuple[MergeRevertedEntry, ...] = ()


# ---------------------------------------------------------------------------
# Newcomer pool selection: the relevance axis's exclusions, plus the ids the caller bars
# ---------------------------------------------------------------------------


class _FalsityNewcomerPool(BaseModel):
    model_config = {"frozen": True}

    newcomers: tuple[Fact, ...]
    excluded_decisions: tuple[HypomnemaId, ...]
    excluded_global: tuple[HypomnemaId, ...]
    excluded_shelved_this_pass: tuple[HypomnemaId, ...]


def _select_falsity_newcomers(
    pool: Sequence[Hypomnema],
    *,
    excluded_ids: AbstractSet[HypomnemaId],
    pass_scope: Scope | None = None,
) -> _FalsityNewcomerPool:
    """Filter a born-since pool down to this axis's newcomers N.

    Checked in this order, so a record is counted against the first ground it matches:

    1. Not a ``Fact``. ``supersede`` needs a Fact on both sides.
    2. ``scope == "global"``, unless ``pass_scope`` is ``global``.
    3. ``lifecycle`` not in ``("active", "archived")``, or the id is in ``excluded_ids``. This keeps
       already-shelved, ``hot`` and ``ember`` Facts out, along with the ids the caller barred.
    """
    newcomers: list[Fact] = []
    excluded_decisions: list[HypomnemaId] = []
    excluded_global: list[HypomnemaId] = []
    excluded_shelved: list[HypomnemaId] = []

    for record in pool:
        if isinstance(record, Decision):
            excluded_decisions.append(record.id)
            continue
        if record.scope == GLOBAL and pass_scope != GLOBAL:
            excluded_global.append(record.id)
            continue
        if record.lifecycle not in ("active", "archived") or record.id in excluded_ids:
            excluded_shelved.append(record.id)
            continue
        newcomers.append(record)

    return _FalsityNewcomerPool(
        newcomers=tuple(newcomers),
        excluded_decisions=tuple(excluded_decisions),
        excluded_global=tuple(excluded_global),
        excluded_shelved_this_pass=tuple(excluded_shelved),
    )


# ---------------------------------------------------------------------------
# Candidate generation + pair formation
# ---------------------------------------------------------------------------

# One (newcomer, candidate, similarity) tuple — a viable, judge-bound pair.
_Pair = tuple[Fact, Fact, float]


class _FalsityCandidates(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    viable_pairs: tuple[_Pair, ...]
    decision_candidates: tuple[FalsityReportEntry, ...]
    global_candidates: tuple[FalsityReportEntry, ...]
    hot_or_ember_candidates: tuple[FalsityReportEntry, ...]


async def _generate_pairs(
    newcomers: Sequence[Fact],
    *,
    scope: Scope,
    adapter: StorageAdapter,
    embedder: Embedder,
    cfg: EngineConfig,
    rubric: Rubric,
    assessor: Assessor | None = None,
) -> _FalsityCandidates:
    """For each newcomer N, probe backwards (never ``recall()``) and classify the best candidate.

    One probe per newcomer: valid records only, span ``read_span(scope)``, ``k`` and
    ``similarity_floor`` from the rubric. ``exclude={N.id}`` keeps a newcomer from matching itself.

    Classification order: age (a candidate not strictly older is discarded silently), then Decision,
    then ``global`` (only when there is no Assessor), then lifecycle not in ``("active",
    "archived")``. What remains is a viable pair bound for a judge.
    """
    span = read_span(scope)
    viable_pairs: list[_Pair] = []
    decision_candidates: list[FalsityReportEntry] = []
    global_candidates: list[FalsityReportEntry] = []
    hot_or_ember_candidates: list[FalsityReportEntry] = []

    for newcomer in newcomers:
        vector = await embedder.embed(newcomer.content)
        result = await probe_candidates(
            content=newcomer.content,
            vector=vector,
            span=span,
            k=rubric.parameters.k,
            exclude={newcomer.id},
            validity=frozenset({"valid"}),
            adapter=adapter,
            cfg=cfg,
        )
        if result.best_hit is None or result.best_cosine < rubric.parameters.similarity_floor:
            continue

        candidate = result.best_hit.record
        if not candidate.created_at < newcomer.created_at:
            continue  # discard: not strictly older -- structural, never judged, never reported

        if isinstance(candidate, Decision):
            decision_candidates.append(
                FalsityReportEntry(
                    newcomer_id=newcomer.id,
                    candidate_id=candidate.id,
                    similarity=result.best_cosine,
                    reason="candidate_is_decision",
                )
            )
            continue
        if candidate.scope == GLOBAL and scope != GLOBAL and assessor is None:
            global_candidates.append(
                FalsityReportEntry(
                    newcomer_id=newcomer.id,
                    candidate_id=candidate.id,
                    similarity=result.best_cosine,
                    reason="candidate_is_global",
                )
            )
            continue
        if candidate.lifecycle not in ("active", "archived"):
            # Hot and ember Facts are never touched by the Cleaner, and the probe can return them,
            # so a hot/ember candidate is reported and never becomes a supersession target.
            hot_or_ember_candidates.append(
                FalsityReportEntry(
                    newcomer_id=newcomer.id,
                    candidate_id=candidate.id,
                    similarity=result.best_cosine,
                    reason="candidate_is_hot_or_ember",
                )
            )
            continue

        viable_pairs.append((newcomer, candidate, result.best_cosine))

    return _FalsityCandidates(
        viable_pairs=tuple(viable_pairs),
        decision_candidates=tuple(decision_candidates),
        global_candidates=tuple(global_candidates),
        hot_or_ember_candidates=tuple(hot_or_ember_candidates),
    )


async def _resolve_global_candidates(
    *,
    entries: tuple[FalsityReportEntry, ...],
    assessor: Assessor | None,
    cfg: EngineConfig,
    adapter: StorageAdapter,
    lock: asyncio.Lock,
    now: datetime,
    dry_run: bool,
) -> tuple[FalsityReportEntry, ...]:
    """Annotate ``global`` candidates with the Assessor's pick: one ``contradiction`` call per
    entry, with no ``panel``, bounded by ``admission.bounded_ask``.

    Each entry gets ``assessor_pick`` and, unless ``dry_run``, a ``global_reported`` verdict row.
    Nothing is written to the records here; a write into ``global`` goes through
    ``_global_write_allowed`` in the main loop. Returns ``entries`` unchanged when no Assessor is
    configured.
    """
    if assessor is None or not entries:
        return entries

    newcomers = [await adapter.get_by_id(entry.newcomer_id) for entry in entries]
    older = [await adapter.get_by_id(entry.candidate_id) for entry in entries]

    def _factory(
        newcomer: Fact | Decision, candidate: Fact | Decision
    ) -> Callable[[], Awaitable[ContradictionVerdict]]:
        async def _call() -> ContradictionVerdict:
            return await assessor.contradiction(candidate.content, newcomer.content)

        return _call

    results = await bounded_ask(
        [_factory(n, c) for n, c in zip(newcomers, older, strict=True)],
        method="contradiction",
        timeout_s=cfg.assessor_cleaner_timeout_s,
    )

    resolved: list[FalsityReportEntry] = []
    for entry, result in zip(entries, results, strict=True):
        if isinstance(result, Unavailable):
            pick = "unavailable"
        elif result.pick == "replace":
            assert result.reason_class is not None
            pick = result.reason_class
        else:
            pick = "coexist"
        resolved.append(entry.model_copy(update={"assessor_pick": pick}))
        if not dry_run:
            row = build_cleaner_resolved_row(
                caller="cleaner_split",
                candidate_id=entry.candidate_id,
                new_record_id=entry.newcomer_id,
                verdict=result,
                action="global_reported",
                actor="engine",
                matched=None,
                now=now,
            )
            async with lock:
                await adapter.append_verdicts([row])
    return tuple(resolved)


def _render_pair_content(*, newcomer: Fact, candidate: Fact) -> str:
    """Render one pair as a judge batch item, labelled N and O and never carrying a record id."""
    return (
        f"N (newer, created {newcomer.created_at.isoformat()}):\n{newcomer.content}\n\n"
        f"O (older, created {candidate.created_at.isoformat()}):\n{candidate.content}"
    )


# ---------------------------------------------------------------------------
# Band policy: unanimous applies, a split is contested and draws no edge
# ---------------------------------------------------------------------------


def _deciding_verdict(verdict_a: FalsityVerdict, verdict_b: FalsityVerdict) -> FalsityVerdict:
    """Judge A's verdict when it voted replace, else judge B's.

    Used for the no-Assessor ``reason_disagreement`` write, where both judges voted replace with
    different reasons, so judge A's reason and ``why`` decide. The ``verdict_b`` branch is
    unreachable there.
    """
    return verdict_a if verdict_a.votes_replace else verdict_b


PairClassification = Literal["replace", "coexist", "reason_disagreement", "action_split"]


def _classify_pair(verdict_a: FalsityVerdict, verdict_b: FalsityVerdict) -> PairClassification:
    """The reason-aware classification an Assessor reads.

    - Both ``coexist``: ``"coexist"``, full agreement, no call.
    - Both ``replace`` with the same reason: ``"replace"``, full agreement, no call.
    - Both ``replace`` with different reasons: ``"reason_disagreement"``.
    - One ``replace`` and one ``coexist``: ``"action_split"``.
    """
    if verdict_a.verdict == verdict_b.verdict:
        if verdict_a.verdict == "coexist":
            return "coexist"
        if verdict_a.reason == verdict_b.reason:
            return "replace"
        return "reason_disagreement"
    return "action_split"


def _falsity_matched(result: ContradictionVerdict | Unavailable, *, drew_edge: bool) -> bool | None:
    """``matched`` for a falsity tie-break or split: ``None`` when the Assessor was ``Unavailable``,
    else whether its replace or coexist direction agrees with whether an edge was drawn.
    """
    if isinstance(result, Unavailable):
        return None
    return (result.pick == "replace") == drew_edge


def _review_failure(review: RecordReview, error: str) -> FalsityReviewAppendFailure:
    return FalsityReviewAppendFailure(target_id=review.target_id, error=error)


async def _write_contested_rows(
    *,
    newcomer: Fact,
    candidate: Fact,
    similarity: float,
    judge_a_identity: str,
    judge_b_identity: str,
    verdict_a: FalsityVerdict,
    verdict_b: FalsityVerdict,
    now: datetime,
    rubric: Rubric,
    adapter: StorageAdapter,
    dry_run: bool,
    review_append_failures: list[FalsityReviewAppendFailure],
    judgment_append_failures: list[JudgmentAppendFailure],
) -> None:
    """The contested write: a ``band="contested"`` ``RecordReview`` for each of the two records,
    then a ``RecordJudgment`` for each, carrying ``paired_with`` and ``similarity``. A no-op under
    ``dry_run``.

    Every path that ends contested shares it: an ``action_split`` with no Assessor, an Assessor
    ``coexist`` vote against a ``reason_disagreement``, and an unavailable Assessor.
    """
    if dry_run:
        return
    for target in (newcomer, candidate):
        await append_review_safely(
            adapter=adapter,
            review=await build_review_row(
                adapter=adapter,
                scope=target.scope,
                target_id=target.id,
                now=now,
                rubric=rubric,
                outcome="judged",
                band="contested",
                grades=(verdict_a.verdict, verdict_b.verdict),
            ),
            failures=review_append_failures,
            to_failure=_review_failure,
        )
    for target, paired in ((newcomer, candidate), (candidate, newcomer)):
        await append_judgment_safely(
            adapter=adapter,
            judgment=RecordJudgment(
                target_id=target.id,
                scope=target.scope,
                judged_at=now,
                rubric_version=rubric.version,
                judge_a_identity=judge_a_identity,
                judge_a_grade=verdict_a.verdict,
                judge_a_why=verdict_a.why,
                judge_b_identity=judge_b_identity,
                judge_b_grade=verdict_b.verdict,
                judge_b_why=verdict_b.why,
                paired_with=paired.id,
                similarity=similarity,
            ),
            failures=judgment_append_failures,
        )


def _is_pre_write_refusal(exc: ToolError) -> bool:
    """``True`` when ``supersede`` or ``validate_supersede`` refused before attempting the write,
    which raises a bare ``ToolError`` with no ``__cause__``. ``False`` when the failure happened
    inside the write and was re-raised as a ``ToolError`` chained with ``from``; that one aborts
    the pass.
    """
    return exc.__cause__ is None


@dataclass(frozen=True)
class Refused:
    """A pre-write refusal of a falsity edge; ``reason`` is the ``ToolError`` text."""

    reason: str


@dataclass(frozen=True)
class AssessorResolution:
    """What ``_resolve_falsity_assessor_call`` decided.

    ``action`` is the resolved action and ``verdict`` the Assessor's verdict or ``Unavailable``.

    ``write_refused`` is ``True`` only for ``reason_by_assessor`` and ``split_supersede`` when the
    edge write was refused before the write (the pending row is stamped ``failed``, no edge
    stands); it is ``False`` for those actions when the edge landed (or, in a dry run, would be
    drawn) and ``False`` for every other action.

    ``coexist_attempt_failed`` is ``True`` only when a live ``split_coexist`` write was attempted
    and at least one participant's resolving ``band="keep"`` review failed to append; ``action`` is
    then ``contested``, and one participant's keep review may already sit at ``now``. It is
    ``False`` for every other action and for every other path to ``contested``.
    """

    action: Literal["split_supersede", "split_coexist", "reason_by_assessor", "contested"]
    verdict: ContradictionVerdict | Unavailable
    write_refused: bool = False
    coexist_attempt_failed: bool = False


@dataclass
class _MergeLedger:
    merged: list[MergedTriple] = field(default_factory=list)
    merge_reverted: list[MergeRevertedEntry] = field(default_factory=list)
    merge_revert_failed: list[MergeRevertedEntry] = field(default_factory=list)

    def record(self, applied: MergeApplied | None) -> bool:
        """Append the entry ``applied`` carries to its list; returns ``False`` only when the edge
        was undone and the caller drops it. ``None`` (no carry) and a merge that was already
        covered record nothing and return ``True``.
        """
        if applied is None:
            return True

        if applied.merged is not None:
            self.merged.append(applied.merged)
        if applied.reverted is not None:
            self.merge_reverted.append(applied.reverted)
        if applied.revert_failed is not None:
            self.merge_revert_failed.append(applied.revert_failed)
        return applied.edge_stands


@dataclass(frozen=True)
class _MergeContext:
    embedder: Embedder
    require_scope: RequireScope | None
    adapter: StorageAdapter
    clock: Clock
    cfg: EngineConfig
    lock: asyncio.Lock
    dry_run: bool


async def _merge_if_carried(
    context: _MergeContext,
    newcomer: Fact,
    candidate: Fact,
    verdict_a: FalsityVerdict,
    verdict_b: FalsityVerdict,
) -> MergeApplied | None:
    """Merge the first replace-voting seat's ``carry`` into the newcomer; ``None`` when no
    replace-voting seat supplied one.
    """
    carry = pick_carry(
        [verdict.carry for verdict in (verdict_a, verdict_b) if verdict.votes_replace]
    )
    if carry is None:
        return None

    return await apply_merge(
        newer=newcomer,
        older=candidate,
        carry=carry,
        embedder=context.embedder,
        require_scope=context.require_scope,
        adapter=context.adapter,
        clock=context.clock,
        cfg=context.cfg,
        lock=context.lock,
        dry_run=context.dry_run,
    )


async def _arbitrate_split_coexist(
    *,
    newcomer: Fact,
    candidate: Fact,
    result: ContradictionVerdict,
    rubric: Rubric,
    now: datetime,
    adapter: StorageAdapter,
    lock: asyncio.Lock,
    dry_run: bool,
    review_append_failures: list[FalsityReviewAppendFailure],
) -> AssessorResolution:
    """The Assessor's ``coexist`` on an ``action_split``: no edge, and a resolving
    ``band="keep"`` review for each participant, appended in turn. The pending row, both reviews and
    the stamp share one lock hold. If either review append fails (the other may already have
    landed), the row is stamped ``failed`` and the result is
    ``contested`` with ``coexist_attempt_failed``. A dry run writes nothing.
    """
    if dry_run:
        return AssessorResolution(action="split_coexist", verdict=result)

    pending = build_cleaner_pending_row(
        caller="cleaner_split",
        candidate_id=candidate.id,
        new_record_id=newcomer.id,
        verdict=result,
        action="split_coexist",
        effect_ref=None,
        now=now,
    )
    async with lock:
        await adapter.append_verdicts([pending])
        failures_before = len(review_append_failures)
        for target in (newcomer, candidate):
            await append_review_safely(
                adapter=adapter,
                review=await build_review_row(
                    adapter=adapter,
                    scope=target.scope,
                    target_id=target.id,
                    now=now,
                    rubric=rubric,
                    outcome="assessor_resolved",
                    band="keep",
                    grades=(),
                ),
                failures=review_append_failures,
                to_failure=_review_failure,
            )
        if len(review_append_failures) > failures_before:
            # At least one participant's resolving keep review did not land. Never stamp
            # `acted` for a resolution that did not hold for both: stamp `failed`, and the
            # caller falls back to the contested write for this pair.
            await stamp_failed_many(adapter, [pending], now=now)
            return AssessorResolution(
                action="contested", verdict=result, coexist_attempt_failed=True
            )
        await stamp_acted_many(adapter, [pending], now=now)
    return AssessorResolution(action="split_coexist", verdict=result)


async def _arbitrate_replace(
    *,
    action: Literal["reason_by_assessor", "split_supersede"],
    newcomer: Fact,
    candidate: Fact,
    result: ContradictionVerdict,
    scope: Scope,
    clock: Clock,
    now: datetime,
    adapter: StorageAdapter,
    lock: asyncio.Lock,
    dry_run: bool,
) -> AssessorResolution:
    """The Assessor's ``replace``: the pending row, ``validate_supersede``,
    ``supersede_write_locked`` and the stamp share one lock hold. A pre-write ``ToolError`` (no
    ``__cause__``) stamps the row ``failed`` and returns ``write_refused``; a ``ToolError`` with a
    cause re-raises. A dry run writes nothing.
    """
    assert result.reason_class is not None
    if dry_run:
        return AssessorResolution(action=action, verdict=result)

    edge_id = str(uuid4())
    pending = build_cleaner_pending_row(
        caller="cleaner_split",
        candidate_id=candidate.id,
        new_record_id=newcomer.id,
        verdict=result,
        action=action,
        effect_ref=edge_id,
        now=now,
    )
    # The pending append, the supersede write and the stamp share one lock hold. Separate
    # acquisitions would let a concurrent `unsupersede` land its override between the write and
    # the stamp, and `stamp_acted_many` (built from the stale pre-write `pending`) would erase it.
    async with lock:
        await adapter.append_verdicts([pending])
        try:
            successor, flips = await validate_supersede(
                successor_id=newcomer.id,
                predecessor_ids=[candidate.id],
                reason=_edge_reason(scope, candidate, result.reason_class),
                note=_edge_note(scope, candidate, result.reason_class),
                adapter=adapter,
                span=_write_span(scope, candidate),
                edge_id=edge_id,
            )
            await supersede_write_locked(
                successor=successor,
                successor_id=newcomer.id,
                predecessor_ids=[candidate.id],
                reason=_edge_reason(scope, candidate, result.reason_class),
                flips=flips,
                adapter=adapter,
                now=clock.now(),
            )
        except ToolError as exc:
            if not _is_pre_write_refusal(exc):
                raise  # an in-write failure, not a refusal: abort the pass
            await stamp_failed_many(adapter, [pending], now=clock.now())
            return AssessorResolution(action=action, verdict=result, write_refused=True)
        await stamp_acted_many(adapter, [pending], now=clock.now())
    return AssessorResolution(action=action, verdict=result)


async def _resolve_falsity_assessor_call(
    *,
    kind: Literal["reason_disagreement", "action_split"],
    newcomer: Fact,
    candidate: Fact,
    verdict_a: FalsityVerdict,
    verdict_b: FalsityVerdict,
    assessor: Assessor,
    cfg: EngineConfig,
    scope: Scope,
    rubric: Rubric,
    clock: Clock,
    now: datetime,
    adapter: StorageAdapter,
    lock: asyncio.Lock,
    dry_run: bool,
    review_append_failures: list[FalsityReviewAppendFailure],
) -> AssessorResolution:
    """Arbitrate a ``reason_disagreement`` or ``action_split`` with a fresh ``contradiction`` call
    (caller ``cleaner_split``) that carries both judges' votes and reasoning as ``panel``.

    Returns an ``AssessorResolution``: the resolved action, the Assessor's verdict or
    ``Unavailable``, whether the edge write was refused, and whether a ``split_coexist`` attempt
    failed.

    - ``Unavailable``: ``"contested"``.
    - ``coexist`` on a ``reason_disagreement``: ``"contested"``. The Assessor supplies no reason
      there, and none is guessed.
    - ``coexist`` on an ``action_split``: ``_arbitrate_split_coexist``.
    - ``replace``: ``"reason_by_assessor"`` for a ``reason_disagreement``, else
      ``"split_supersede"``, written by ``_arbitrate_replace``.

    Under ``dry_run`` it makes the call and writes nothing.
    """
    panel = PanelDisagreement(
        judge_a=PanelJudgeAnswer(
            verdict=verdict_a.verdict, reason=verdict_a.reason, why=verdict_a.why
        ),
        judge_b=PanelJudgeAnswer(
            verdict=verdict_b.verdict, reason=verdict_b.reason, why=verdict_b.why
        ),
    )

    async def _call() -> ContradictionVerdict:
        return await assessor.contradiction(candidate.content, newcomer.content, panel=panel)

    result = await ask(_call, method="contradiction", timeout_s=cfg.assessor_cleaner_timeout_s)

    if isinstance(result, Unavailable):
        return AssessorResolution(action="contested", verdict=result)

    if result.pick == "coexist":
        if kind == "reason_disagreement":
            return AssessorResolution(action="contested", verdict=result)
        return await _arbitrate_split_coexist(
            newcomer=newcomer,
            candidate=candidate,
            result=result,
            rubric=rubric,
            now=now,
            adapter=adapter,
            lock=lock,
            dry_run=dry_run,
            review_append_failures=review_append_failures,
        )

    return await _arbitrate_replace(
        action="reason_by_assessor" if kind == "reason_disagreement" else "split_supersede",
        newcomer=newcomer,
        candidate=candidate,
        result=result,
        scope=scope,
        clock=clock,
        now=now,
        adapter=adapter,
        lock=lock,
        dry_run=dry_run,
    )


def _edge_reason(scope: Scope, candidate: Fact, reason: SupersessionReason) -> SupersessionReason:
    if candidate.scope == GLOBAL and scope != GLOBAL:
        return "moved"

    return reason


def _edge_note(scope: Scope, candidate: Fact, reason: SupersessionReason) -> str | None:
    if candidate.scope == GLOBAL and scope != GLOBAL:
        return f"falsity: {reason} (panel), cross-scope into global"

    return None


def _write_span(scope: Scope, candidate: Fact) -> list[Scope]:
    if candidate.scope == GLOBAL and scope != GLOBAL:
        return [scope, GLOBAL]

    return [scope]


async def _global_write_allowed(
    *,
    newcomer: Fact,
    candidate: Fact,
    scope: Scope,
    assessor: Assessor | None,
    cfg: EngineConfig,
) -> Literal["allowed", "gated", "unavailable"]:
    """Whether a write into the candidate's scope may proceed. ``"allowed"`` when the candidate is
    not a ``global`` record seen from a project pass. Otherwise ``"gated"`` without an Assessor or
    when Jev ``breadth`` on the newcomer is not ``owner_wide``, and ``"unavailable"`` when
    ``breadth`` is ``Unavailable``.
    """
    if candidate.scope != GLOBAL or scope == GLOBAL:
        return "allowed"

    if assessor is None:
        return "gated"

    results = await bounded_ask(
        [lambda: assessor.breadth(newcomer.content, str(newcomer.scope))],
        method="breadth",
        timeout_s=cfg.assessor_cleaner_timeout_s,
    )
    result = results[0]
    if isinstance(result, Unavailable):
        return "unavailable"

    return "allowed" if result.pick == "owner_wide" else "gated"


async def _write_falsity_edge(
    *,
    newcomer: Fact,
    candidate: Fact,
    similarity: float,
    reason: InvalidationReason,
    why: str,
    deciding_judge: str,
    scope: Scope,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
    edge_id: str | None = None,
) -> FalsityEdge | Refused:
    """The write: the older record is the predecessor and the newer the successor, never the
    reverse.

    ``supersede`` acquires ``lock`` itself, so the caller must not wrap this call in ``async with
    lock:``. ``span`` is passed as a second guard behind ``_generate_pairs``'s ``global``
    classification.

    Returns the drawn ``FalsityEdge``, or ``Refused`` carrying the refusal text when ``supersede``
    refused before attempting the write (``_is_pre_write_refusal``). Any other exception
    propagates.
    """
    try:
        await supersede(
            successor_id=newcomer.id,
            predecessor_ids=[candidate.id],
            reason=_edge_reason(scope, candidate, reason),
            note=_edge_note(scope, candidate, reason),
            adapter=adapter,
            clock=clock,
            lock=lock,
            span=_write_span(scope, candidate),
            edge_id=edge_id,
        )
    except ToolError as exc:
        if not _is_pre_write_refusal(exc):
            raise
        return Refused(reason=str(exc))
    return FalsityEdge(
        id=candidate.id,
        superseded_by=newcomer.id,
        similarity=similarity,
        deciding_judge=deciding_judge,
        why=why,
    )


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------

_PairKey = tuple[HypomnemaId, HypomnemaId]
_WriteClassification = Literal["replace", "reason_disagreement", "action_split"]


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
    token_counter: TokenCounter
    now: datetime
    dry_run: bool
    merge: _MergeContext


@dataclass(frozen=True, slots=True)
class _JudgedPair:
    """One pair both seats answered: the records, the probe similarity and the two verdicts."""

    newcomer: Fact
    candidate: Fact
    similarity: float
    verdict_a: FalsityVerdict
    verdict_b: FalsityVerdict
    judge_a_identity: str
    judge_b_identity: str
    batch_index: int

    @property
    def key(self) -> _PairKey:
        return (self.newcomer.id, self.candidate.id)


@dataclass(slots=True)
class _Tally:
    """The lists one pass fills. The counts in the summary are derived from them, so no count is
    undone after the fact.

    ``replaced`` holds the pairs whose replace stands. ``_record_replace`` is the only code that
    appends to ``replaced`` or to ``edges``, and a live pass always hands it an edge, so outside a
    dry run ``len(replaced) == len(edges)``. A dry run records a panel replace in ``replaced``
    without an edge. A replace whose merge reverted is in neither list, and one whose merge
    ``revert_failed`` is in both. ``split_superseded`` and ``reason_by_assessor_pairs`` are subsets
    of ``replaced``, appended right after it. ``coexisted`` holds the panel's coexist pairs and the
    Assessor's ``split_coexist`` pairs, and ``split_coexisted`` is the subset from the Assessor.
    """

    edges: list[FalsityEdge] = field(default_factory=list)
    replaced: list[_PairKey] = field(default_factory=list)
    split_superseded: list[_PairKey] = field(default_factory=list)
    reason_by_assessor_pairs: list[_PairKey] = field(default_factory=list)
    coexisted: list[_PairKey] = field(default_factory=list)
    split_coexisted: list[_PairKey] = field(default_factory=list)
    gated_global: list[FalsityReportEntry] = field(default_factory=list)
    contested_pairs: list[FalsityContestedPair] = field(default_factory=list)
    failed_batches: list[FalsityBatchFailure] = field(default_factory=list)
    calls: list[JudgeCallRecord] = field(default_factory=list)
    refused: list[FalsityRefusedPair] = field(default_factory=list)
    review_append_failures: list[FalsityReviewAppendFailure] = field(default_factory=list)
    judgment_append_failures: list[JudgmentAppendFailure] = field(default_factory=list)
    ledger: _MergeLedger = field(default_factory=_MergeLedger)


# ---------------------------------------------------------------------------
# Per-pair handlers
# ---------------------------------------------------------------------------


def _record_replace(tally: _Tally, pair: _JudgedPair, edge: FalsityEdge | None) -> None:
    tally.replaced.append(pair.key)
    if edge is not None:
        tally.edges.append(edge)


def _refused_pair(pair: _JudgedPair, reason: str) -> FalsityRefusedPair:
    return FalsityRefusedPair(
        newcomer_id=pair.newcomer.id, candidate_id=pair.candidate.id, reason=reason
    )


def _contested_pair(pair: _JudgedPair) -> FalsityContestedPair:
    return FalsityContestedPair(
        newcomer_id=pair.newcomer.id,
        candidate_id=pair.candidate.id,
        similarity=pair.similarity,
        judge_a_identity=pair.judge_a_identity,
        judge_a_verdict=pair.verdict_a.verdict,
        judge_a_why=pair.verdict_a.why,
        judge_b_identity=pair.judge_b_identity,
        judge_b_verdict=pair.verdict_b.verdict,
        judge_b_why=pair.verdict_b.why,
    )


async def _merge_stands(ctx: _AxisContext, tally: _Tally, pair: _JudgedPair) -> bool:
    """Merge the pair's carry, if a replace-voting seat supplied one, and record the outcome in
    the ledger. ``False`` only when the merge reverted and the edge was undone.
    """
    applied = await _merge_if_carried(
        ctx.merge, pair.newcomer, pair.candidate, pair.verdict_a, pair.verdict_b
    )
    return tally.ledger.record(applied)


async def _write_panel_replace(
    ctx: _AxisContext,
    tally: _Tally,
    pair: _JudgedPair,
    *,
    reason: InvalidationReason,
    why: str,
    deciding_judge: str,
) -> None:
    """Draw the panel's edge and merge its carry. A refused write is reported in ``refused`` and
    the pair is not recorded as a replace. A dry run draws no edge, ignores the merge result and
    records the replace.
    """
    edge: FalsityEdge | None = None
    if not ctx.dry_run:
        written = await _write_falsity_edge(
            newcomer=pair.newcomer,
            candidate=pair.candidate,
            similarity=pair.similarity,
            reason=reason,
            why=why,
            deciding_judge=deciding_judge,
            scope=ctx.scope,
            adapter=ctx.adapter,
            clock=ctx.clock,
            lock=ctx.lock,
        )
        if isinstance(written, Refused):
            tally.refused.append(_refused_pair(pair, written.reason))
            return
        edge = written

    merge_stands = await _merge_stands(ctx, tally, pair)
    if merge_stands or ctx.dry_run:
        _record_replace(tally, pair, edge)


async def _handle_agreed_replace(ctx: _AxisContext, tally: _Tally, pair: _JudgedPair) -> None:
    """Full agreement on action and reason: no Assessor call and no verdict row. The reason is the
    judges' own.
    """
    reason = pair.verdict_a.reason
    assert reason is not None and reason == pair.verdict_b.reason
    await _write_panel_replace(
        ctx,
        tally,
        pair,
        reason=reason,
        why=pair.verdict_a.why,
        deciding_judge=pair.judge_a_identity,
    )


async def _handle_unarbitrated_reason_disagreement(
    ctx: _AxisContext, tally: _Tally, pair: _JudgedPair
) -> None:
    """No Assessor: judge A's reason wins."""
    deciding = _deciding_verdict(pair.verdict_a, pair.verdict_b)
    deciding_judge = pair.judge_a_identity if deciding is pair.verdict_a else pair.judge_b_identity
    assert deciding.reason is not None
    await _write_panel_replace(
        ctx, tally, pair, reason=deciding.reason, why=deciding.why, deciding_judge=deciding_judge
    )


async def _append_contested_rows(
    ctx: _AxisContext, tally: _Tally, pair: _JudgedPair, *, now: datetime
) -> None:
    await _write_contested_rows(
        newcomer=pair.newcomer,
        candidate=pair.candidate,
        similarity=pair.similarity,
        judge_a_identity=pair.judge_a_identity,
        judge_b_identity=pair.judge_b_identity,
        verdict_a=pair.verdict_a,
        verdict_b=pair.verdict_b,
        now=now,
        rubric=ctx.rubric,
        adapter=ctx.adapter,
        dry_run=ctx.dry_run,
        review_append_failures=tally.review_append_failures,
        judgment_append_failures=tally.judgment_append_failures,
    )


async def _handle_unarbitrated_action_split(
    ctx: _AxisContext, tally: _Tally, pair: _JudgedPair
) -> None:
    """No Assessor: the contested write, no verdict row."""
    tally.contested_pairs.append(_contested_pair(pair))
    await _append_contested_rows(ctx, tally, pair, now=ctx.now)


async def _handle_assessor_replace(
    ctx: _AxisContext, tally: _Tally, pair: _JudgedPair, resolution: AssessorResolution
) -> None:
    """An Assessor ``replace``. A refused write is reported in ``refused``. Otherwise the edge is
    recorded only if the merge has not reverted it; under a dry run the edge is built and the
    merge result is still honoured.
    """
    if resolution.write_refused:
        tally.refused.append(_refused_pair(pair, "split-supersede write refused"))
        return

    result = resolution.verdict
    assert not isinstance(result, Unavailable) and result.reason_class is not None
    edge = FalsityEdge(
        id=pair.candidate.id,
        superseded_by=pair.newcomer.id,
        similarity=pair.similarity,
        deciding_judge="assessor",
        why=f"Assessor: {result.reason_class}",
    )
    if not await _merge_stands(ctx, tally, pair):
        return

    _record_replace(tally, pair, edge)
    if resolution.action == "split_supersede":
        tally.split_superseded.append(pair.key)
    else:
        tally.reason_by_assessor_pairs.append(pair.key)


async def _handle_assessor_contested(
    ctx: _AxisContext, tally: _Tally, pair: _JudgedPair, resolution: AssessorResolution
) -> None:
    """An Assessor call that did not resolve the pair (Unavailable, a coexist vote against a
    reason disagreement, or a split_coexist where one participant's resolving review landed and
    the other's failed). The pair is appended to ``contested_pairs`` in every mode. Outside a dry
    run the contested rows are written, then the Cleaner's own resolved row naming what the
    Assessor said; a dry run writes neither.
    """
    tally.contested_pairs.append(_contested_pair(pair))
    # `coexist_attempt_failed` is set when a split_coexist attempt's review append failed
    # for either participant, so the other's `band="keep"` review may already sit at
    # `now`. Both storage adapters keep the first-inserted review when `reviewed_at` ties,
    # so a contested review at `now` would lose to that stale keep. One microsecond later
    # is unambiguously the newest for both participants. On every other path `now` is used.
    contested_now = (
        ctx.now + timedelta(microseconds=1) if resolution.coexist_attempt_failed else ctx.now
    )
    await _append_contested_rows(ctx, tally, pair, now=contested_now)
    if ctx.dry_run:
        return

    row = build_cleaner_resolved_row(
        caller="cleaner_split",
        candidate_id=pair.candidate.id,
        new_record_id=pair.newcomer.id,
        verdict=resolution.verdict,
        action="contested",
        actor="panel",
        matched=_falsity_matched(resolution.verdict, drew_edge=False),
        now=ctx.now,
    )
    async with ctx.lock:
        await ctx.adapter.append_verdicts([row])


async def _handle_arbitrated(
    ctx: _AxisContext,
    tally: _Tally,
    pair: _JudgedPair,
    *,
    kind: Literal["reason_disagreement", "action_split"],
    assessor: Assessor,
) -> None:
    """With an Assessor, a fresh ``contradiction`` call (caller ``cleaner_split``) carries both
    judges' votes and reasoning.
    """
    resolution = await _resolve_falsity_assessor_call(
        kind=kind,
        newcomer=pair.newcomer,
        candidate=pair.candidate,
        verdict_a=pair.verdict_a,
        verdict_b=pair.verdict_b,
        assessor=assessor,
        cfg=ctx.cfg,
        scope=ctx.scope,
        rubric=ctx.rubric,
        clock=ctx.clock,
        now=ctx.now,
        adapter=ctx.adapter,
        lock=ctx.lock,
        dry_run=ctx.dry_run,
        review_append_failures=tally.review_append_failures,
    )
    match resolution.action:
        case "reason_by_assessor" | "split_supersede":
            await _handle_assessor_replace(ctx, tally, pair, resolution)
        case "split_coexist":
            tally.coexisted.append(pair.key)
            tally.split_coexisted.append(pair.key)
        case "contested":
            await _handle_assessor_contested(ctx, tally, pair, resolution)
        case unreachable:
            assert_never(unreachable)


async def _handle_allowed(
    ctx: _AxisContext, tally: _Tally, pair: _JudgedPair, classification: _WriteClassification
) -> None:
    if classification == "replace":
        await _handle_agreed_replace(ctx, tally, pair)
    elif ctx.assessor is not None:
        await _handle_arbitrated(ctx, tally, pair, kind=classification, assessor=ctx.assessor)
    elif classification == "reason_disagreement":
        await _handle_unarbitrated_reason_disagreement(ctx, tally, pair)
    else:
        await _handle_unarbitrated_action_split(ctx, tally, pair)


def _record_failed_batch(
    tally: _Tally,
    *,
    batch_index: int,
    pairs: tuple[_PairKey, ...],
    judge_a_outcome: BatchOutcome,
    judge_b_outcome: BatchOutcome,
    error: str,
) -> None:
    tally.failed_batches.append(
        FalsityBatchFailure(
            batch_index=batch_index,
            pairs=pairs,
            judge_a_outcome=judge_a_outcome,
            judge_b_outcome=judge_b_outcome,
            error=error,
        )
    )


async def _handle_pair(ctx: _AxisContext, tally: _Tally, pair: _JudgedPair) -> None:
    """Route one judged pair: coexist, the breadth gate, then the replace or arbitration handler."""
    classification = _classify_pair(pair.verdict_a, pair.verdict_b)
    if classification == "coexist":
        tally.coexisted.append(pair.key)
        return

    write_gate = await _global_write_allowed(
        newcomer=pair.newcomer,
        candidate=pair.candidate,
        scope=ctx.scope,
        assessor=ctx.assessor,
        cfg=ctx.cfg,
    )
    match write_gate:
        case "unavailable":
            _record_failed_batch(
                tally,
                batch_index=pair.batch_index,
                pairs=(pair.key,),
                judge_a_outcome="ok",
                judge_b_outcome="ok",
                error="breadth unavailable for a global candidate",
            )
        case "gated":
            tally.gated_global.append(
                FalsityReportEntry(
                    newcomer_id=pair.newcomer.id,
                    candidate_id=pair.candidate.id,
                    similarity=pair.similarity,
                    reason="candidate_is_global",
                )
            )
        case "allowed":
            await _handle_allowed(ctx, tally, pair, classification)
        case unreachable:
            assert_never(unreachable)


async def _run_batch(
    ctx: _AxisContext,
    tally: _Tally,
    *,
    batch_index: int,
    batch: Sequence[_Pair],
    contents: Sequence[str],
) -> None:
    """Judge one batch with both seats. A batch where either seat did not succeed is recorded in
    ``failed_batches`` and nothing in it is written.
    """
    prompt = render_batch_prompt(
        rubric_prose=ctx.rubric.falsity.prose, contents=contents, item_model=FalsityVerdict
    )
    outcome_a, outcome_b = await run_panel(
        judge_a=ctx.judge_a,
        judge_b=ctx.judge_b,
        prompt=prompt,
        item_model=FalsityVerdict,
        batch_size=len(batch),
        rubric_version=ctx.rubric.version,
        token_counter=ctx.token_counter,
        batch_index=batch_index,
        calls=tally.calls,
    )

    judge_a_outcome = batch_outcome(outcome_a.status)
    judge_b_outcome = batch_outcome(outcome_b.status)
    if judge_a_outcome != "ok" or judge_b_outcome != "ok":
        errors = [o.error for o in (outcome_a, outcome_b) if o.error is not None]
        _record_failed_batch(
            tally,
            batch_index=batch_index,
            pairs=tuple((newcomer.id, candidate.id) for newcomer, candidate, _s in batch),
            judge_a_outcome=judge_a_outcome,
            judge_b_outcome=judge_b_outcome,
            error="; ".join(errors),
        )
        return

    for (newcomer, candidate, similarity), verdict_a, verdict_b in zip(
        batch, outcome_a.verdicts, outcome_b.verdicts, strict=True
    ):
        pair = _JudgedPair(
            newcomer=newcomer,
            candidate=candidate,
            similarity=similarity,
            verdict_a=verdict_a,
            verdict_b=verdict_b,
            judge_a_identity=outcome_a.judge,
            judge_b_identity=outcome_b.judge,
            batch_index=batch_index,
        )
        await _handle_pair(ctx, tally, pair)


async def _annotate_global(
    ctx: _AxisContext, entries: tuple[FalsityReportEntry, ...]
) -> tuple[FalsityReportEntry, ...]:
    return await _resolve_global_candidates(
        entries=entries,
        assessor=ctx.assessor,
        cfg=ctx.cfg,
        adapter=ctx.adapter,
        lock=ctx.lock,
        now=ctx.now,
        dry_run=ctx.dry_run,
    )


async def _judge_viable_pairs(
    ctx: _AxisContext, tally: _Tally, viable_pairs: Sequence[_Pair]
) -> None:
    """Pack the viable pairs into batches and judge them in order."""
    pair_contents = [
        _render_pair_content(newcomer=newcomer, candidate=candidate)
        for newcomer, candidate, _similarity in viable_pairs
    ]
    pair_groups = pack_batches(
        pair_contents,
        count_cap=ctx.rubric.parameters.batch_cap,
        rubric_prose=ctx.rubric.falsity.prose,
        item_model=FalsityVerdict,
    )
    for batch_index, group in enumerate(pair_groups):
        await _run_batch(
            ctx,
            tally,
            batch_index=batch_index,
            batch=[viable_pairs[position] for position in group],
            contents=[pair_contents[position] for position in group],
        )


def _build_summary(
    ctx: _AxisContext,
    tally: _Tally,
    *,
    pool_size: int,
    selected: _FalsityNewcomerPool,
    candidates: _FalsityCandidates,
    global_candidates: tuple[FalsityReportEntry, ...],
) -> FalsityAxisSummary:
    return FalsityAxisSummary(
        scope=ctx.scope,
        rubric_version=ctx.rubric.version,
        judge_pair=(ctx.judge_a.identity, ctx.judge_b.identity),
        newcomer_pool_size=pool_size,
        excluded_decisions=selected.excluded_decisions,
        excluded_global=selected.excluded_global,
        excluded_shelved_this_pass=selected.excluded_shelved_this_pass,
        counts=FalsityBandCounts(
            replace=len(tally.replaced),
            coexist=len(tally.coexisted),
            contested=len(tally.contested_pairs),
        ),
        edges=tuple(tally.edges),
        decision_candidates=candidates.decision_candidates,
        global_candidates=global_candidates,
        hot_or_ember_candidates=candidates.hot_or_ember_candidates,
        contested_pairs=tuple(tally.contested_pairs),
        failed_batches=tuple(tally.failed_batches),
        calls=tuple(tally.calls),
        refused=tuple(tally.refused),
        review_append_failures=tuple(tally.review_append_failures),
        judgment_append_failures=tuple(tally.judgment_append_failures),
        split_supersede=len(tally.split_superseded),
        split_coexist=len(tally.split_coexisted),
        reason_by_assessor=len(tally.reason_by_assessor_pairs),
        merged=tuple(tally.ledger.merged),
        merge_reverted=tuple(tally.ledger.merge_reverted),
        merge_revert_failed=tuple(tally.ledger.merge_revert_failed),
    )


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


async def run_falsity_axis(
    *,
    scope: Scope,
    since: datetime,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
    rubric: Rubric,
    judge_a: JudgeAdapter,
    judge_b: JudgeAdapter,
    embedder: Embedder,
    cfg: EngineConfig,
    assessor: Assessor | None = None,
    excluded_ids: AbstractSet[HypomnemaId] = frozenset(),
    pool_limit: int | None = None,
    dry_run: bool = False,
    require_scope: RequireScope | None = None,
) -> FalsityAxisSummary:
    """Judge the Facts born in ``scope`` since ``since`` against older candidates they may have
    overtaken, and draw ``supersede`` edges for the pairs the panel decides replace. The module
    docstring covers candidate generation, pair formation, band policy and the write discipline.

    ``excluded_ids``: ids barred from being newcomers in this pass. ``engine/clean.py`` passes the
    relevance axis's shelved, contested, unclear and refused ids.

    One ``LitellmTokenCounter`` is created for the run and passed to every ``judge_batch`` call.

    ``pool_limit``: ``None`` reads ``rubric.parameters.pool_limit``, the value ``engine/clean.py``
    also uses for its own snapshot.

    ``dry_run``: probing, judging and Assessor calls happen as in a live pass, and nothing is
    written: no ``supersede``, no verdict rows, no contested review or judgment rows, no merge save.
    ``edges`` and ``counts`` still report what would be written, and ``contested_pairs`` is the same
    in both modes.

    ``require_scope`` and ``embedder`` serve the merge save (``engine/cleaner_writes.py``). A
    replace whose replace-voting seats (seat A first) name a still-true part of O in ``carry`` draws
    the O -> N edge as usual, then saves ``N.content + "\\n\\n" + carry`` as a record that
    supersedes N with reason ``elaborated``. The save runs outside any lock hold of this module. A
    bounce or a raise undoes the edge (``merge_reverted``); an undo that fails in storage is
    ``merge_revert_failed`` and the edge stands. Under ``dry_run`` the merge is counted and nothing
    is saved.

    A refused ``supersede`` is skipped and reported in ``refused``; see the module docstring for the
    boundary. Any other exception propagates.
    """
    effective_pool_limit = pool_limit if pool_limit is not None else rubric.parameters.pool_limit
    raw_pool = await adapter.list_born_since(scope=scope, since=since, limit=effective_pool_limit)
    selected = _select_falsity_newcomers(raw_pool, excluded_ids=excluded_ids, pass_scope=scope)
    token_counter = LitellmTokenCounter()
    # Sampled once, before any judging; every contested-pair `RecordReview` row shares it as
    # `reviewed_at`.
    now = clock.now()
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
        token_counter=token_counter,
        now=now,
        dry_run=dry_run,
        merge=_MergeContext(
            embedder=embedder,
            require_scope=require_scope,
            adapter=adapter,
            clock=clock,
            cfg=cfg,
            lock=lock,
            dry_run=dry_run,
        ),
    )

    candidates = await _generate_pairs(
        selected.newcomers,
        scope=scope,
        adapter=adapter,
        embedder=embedder,
        cfg=cfg,
        rubric=rubric,
        assessor=assessor,
    )
    resolved_global_candidates = await _annotate_global(ctx, candidates.global_candidates)

    tally = _Tally()
    await _judge_viable_pairs(ctx, tally, candidates.viable_pairs)
    resolved_gated_global = await _annotate_global(ctx, tuple(tally.gated_global))

    return _build_summary(
        ctx,
        tally,
        pool_size=len(raw_pool),
        selected=selected,
        candidates=candidates,
        global_candidates=(*resolved_global_candidates, *resolved_gated_global),
    )
