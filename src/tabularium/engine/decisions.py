"""Decision arm: ``save_decision`` and the stages it runs.

Differences from the Fact arm: ``supersedes`` takes a single id, ``promoted_from`` is accepted,
the gate floor is ``decision_gray_floor``, a ``Moved`` set never exceeds one candidate, and
``Invalidate`` is never an exit (it applies only to a Fact incoming against a Decision
candidate). The lock discipline and admission order match ``facts.save_internal``.

``save_decision`` validates its flat parameters, then builds a ``DecisionSaveRequest`` and a
``SaveDeps`` (``engine/save_request.py``); the private helpers take those two values instead
of the individual parameters. The stages mirror the Fact arm's: ``_validate_decision_inputs``,
``_validate_lineage``, ``_first_hold``, ``embed_content``, ``_second_hold`` (``_vectorless_hold``
or ``_probe_hold``) and ``assessor_tail``, with the stage helpers shared from ``facts.py`` where
the two arms agree.
Where the Fact arm reads the clock again, several sites here pass an already-read ``now``; each
site keeps the choice it has always made.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Literal
from uuid import uuid4

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import (
    AdmissionStep,
    Decision,
    Fact,
    HypomnemaId,
    PromotedFromSnapshot,
    Scope,
    SupersedesSnapshot,
    SupersessionReason,
    new_id,
)
from tabularium.embedding import Embedder
from tabularium.engine._errors import SupersessionWriteFailed, ToolError, require_text
from tabularium.engine._types import RequirePurpose, RequireRoles, RequireScope
from tabularium.engine.admission import (
    AutoExitSwitches,
    Dispatch,
    ExactHit,
    Invalidate,
    Moved,
    Reinforce,
    Replay,
    Supersede,
    Withheld,
    WriteNew,
    attach_neighbor_verdicts,
    bm25_probe,
    bounce_withheld_update,
    candidate_still_fresh,
    check_exact_key,
    check_replay,
    clean_picture,
    serial_probe,
    upward_promotion_candidates,
)
from tabularium.engine.assessor import Assessor, RelationVerdict, Unavailable
from tabularium.engine.audit_log import log_admission_resolved, log_failed_supersession
from tabularium.engine.backfill import BackfillSignals
from tabularium.engine.dispositions import (
    check_supersedes_reason,
    supersede_legality_span,
)
from tabularium.engine.exits import (
    build_upward_exits,
    is_promotable,
    judge_unavailable_exits,
)
from tabularium.engine.facts import (
    AdmissionAudit,
    AutoExitFrame,
    BounceVerdictName,
    CallerReinforce,
    HoldOutcome,
    JudgedBatch,
    Probe,
    above_floor_candidates,
    admission_result_id,
    assessor_tail,
    begin_caller_reinforce,
    bounce_candidate,
    distinct_exclude_ids,
    embed_content,
    find_vectorless_confirm_distinct_replay,
    floor_for,
    get_or_none,
    pick_matches,
    record_unanswered,
    refresh_moved,
    reinforce,
    resolve_caller_replay_flips,
    row_group,
)
from tabularium.engine.filters import (
    probe_span_for,
    read_span,
    validate_and_resolve_scope,
)
from tabularium.engine.guards import GuardName, guard_check
from tabularium.engine.maintenance import maybe_upkeep
from tabularium.engine.save_request import (
    DecisionSaveRequest,
    ResolveRequest,
    SaveDeps,
    Supersession,
    supersession_from,
)
from tabularium.engine.verdicts import (
    StampFailedOnError,
    begin_caller_exit,
    build_pending_row,
    build_unanswered_rows,
    find_bounce_row,
    stamp_acted_many,
    stamp_acted_many_with_siblings,
)
from tabularium.render import make_supersedes_snapshot
from tabularium.results import (
    BounceNeighbor,
    BounceResolution,
    DecisionBounce,
    HypomnemaRef,
    SaveResult,
)
from tabularium.storage.types import (
    DistinctPair,
    Hit,
    PredecessorFlip,
    StorageAdapter,
    VerdictRow,
)

_log = logging.getLogger("tabularium")

_GLOBAL = Scope("global")


async def _write_decision(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    *,
    vector: list[float] | None,
    now: datetime,
    supersession: Supersession | None,
    reconciles: HypomnemaId | None = None,
    extra_flips: Sequence[PredecessorFlip] = (),
    embedded: bool = True,
    on_written: Callable[[], None] | None = None,
    id: HypomnemaId | None = None,
) -> SaveResult:
    """Construct, validate and write a new Decision.

    Uses ``upsert`` when neither ``supersession`` nor ``extra_flips`` carries anything,
    otherwise one ``adapter.replace`` transaction in which the new row and every flip land
    together or not at all. A failed ``replace`` leaves nothing to compensate for.

    ``id``: pre-minted by an automatic-exit caller before its ``pending`` verdict row is
    written; ``None`` mints one here.

    ``supersession``: the single predecessor with its reason. The automatic-exit and
    caller-replay callers pass ``None`` and carry the predecessor in ``extra_flips`` instead.

    ``extra_flips``: an Assessor-found ``Supersede`` composed with the ``supersession`` edge
    in the same ``replace`` call, each with its own reason class.

    ``vector=None`` with ``embedded=False``: the embedder was unavailable; the row is written
    with a null vector.

    ``on_written``: called right after the adapter write returns and before ``maybe_upkeep``,
    so the outage branch can mark backfill state dirty even if upkeep is cancelled.

    Callers hold the engine's write lock, which is what prevents two concurrent
    ``supersedes=X`` saves from both winning. ``adapter.replace`` re-reads under its own lock
    as a second line of defence for anything the in-process lock cannot see.
    """
    decision = Decision(
        id=id if id is not None else new_id(),
        type="decision",
        content=request.content,
        scope=request.scope,
        source=request.source,
        purpose=request.purpose,
        audience_hint=tuple(request.audience_hint) if request.audience_hint else None,
        lifecycle="in_force",
        supersession_chain=request.supersession_chain,
        promoted_from=request.promoted_snapshots,
        reconciles=reconciles,
        created_at=now,
        updated_at=now,
    )
    flips: list[PredecessorFlip] = []
    if supersession is not None:
        flips.extend(
            PredecessorFlip(
                id=pid,
                reason=supersession.reason,
                note=supersession.note,
                edge_id=str(uuid4()),
            )
            for pid in supersession.supersedes
        )
    flips.extend(extra_flips)
    if flips:
        try:
            await deps.adapter.replace(decision, vector, flips, at=now)
        except Exception as exc:
            log_failed_supersession(
                op="save",
                arm="decision",
                successor_id=decision.id,
                predecessor_ids=[flip.id for flip in flips],
                reason=supersession.reason if supersession is not None else flips[0].reason,
                scope=request.scope,
                error=exc,
            )
            raise SupersessionWriteFailed(f"supersedes: {exc}") from exc
    else:
        await deps.adapter.upsert(decision, vector)

    if on_written is not None:
        on_written()

    await maybe_upkeep(deps.adapter, deps.cfg)

    predecessor_id = supersession.supersedes[0] if supersession is not None else None
    _log.debug(
        "save decision id=%s scope=%s supersedes=%s reconciles=%s",
        decision.id,
        request.scope,
        predecessor_id,
        reconciles,
    )
    outcome: Literal["saved", "reconciled"] = "reconciled" if reconciles else "saved"
    return SaveResult(
        outcome=outcome,
        hypomnema=HypomnemaRef(id=decision.id, kind="decision"),
        embedded=embedded,
    )


async def _write_as_requested(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    *,
    vector: list[float],
    now: datetime,
    reconciles: HypomnemaId | None,
    id: HypomnemaId | None = None,
) -> SaveResult:
    """``_write_decision`` with the request's own ``supersession`` and a live query vector."""
    return await _write_decision(
        request,
        deps,
        vector=vector,
        now=now,
        supersession=request.supersession,
        reconciles=reconciles,
        id=id,
    )


async def _write_vectorless(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    *,
    now: datetime,
    reconciles: HypomnemaId | None,
    id: HypomnemaId | None = None,
) -> SaveResult:
    """``_write_decision`` for the embedder-down path: a null vector and ``backfill.written``."""
    return await _write_decision(
        request,
        deps,
        vector=None,
        now=now,
        supersession=request.supersession,
        reconciles=reconciles,
        embedded=False,
        on_written=deps.backfill.written,
        id=id,
    )


async def _finish_caller_exit(
    deps: SaveDeps, resolve: BounceResolution, row: VerdictRow | None, *, now: datetime
) -> None:
    """Stamp the caller's row, if any, ``acted`` at ``now``; the resolve's candidate is excluded."""
    if row is None:
        return

    assert resolve.bounce_id is not None
    await stamp_acted_many_with_siblings(
        deps.adapter, [row], bounce_id=resolve.bounce_id,
        exclude_candidate_ids={resolve.candidate_id}, now=now,
    )


async def _load_reinforce_target(
    adapter: StorageAdapter, candidate_id: HypomnemaId
) -> Fact | Decision:
    """The record a ``reinforce`` resolve names; ``ToolError`` when it is unknown or buried."""
    try:
        return await adapter.get_by_id(candidate_id)
    except KeyError:
        raise ToolError(
            f"resolve.action='reinforce' (kind='decision') targets unknown/buried "
            f"id={candidate_id!s}. The candidate no longer exists in the store."
        ) from None


def _is_live_decision(record: Fact | Decision) -> bool:
    return (
        isinstance(record, Decision)
        and record.lifecycle == "in_force"
        and record.validity == "valid"
    )


def _out_of_play(record: Fact | Decision) -> bool:
    """A Decision not ``in_force`` or a shelved Fact: the vectorless reinforce writes past it."""
    if isinstance(record, Decision):
        return record.lifecycle != "in_force"

    return record.lifecycle == "shelved"


def _names_promotion(request: DecisionSaveRequest, record: Fact | Decision) -> bool:
    """Whether ``promoted_from`` names ``record`` and it is still a promotable Fact."""
    return (
        isinstance(record, Fact)
        and request.promoted_from is not None
        and record.id in request.promoted_from
        and is_promotable(record)
    )


async def _vectorless_confirm_distinct(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    resolve: BounceResolution,
    now: datetime,
) -> SaveResult:
    candidate_id = resolve.candidate_id
    replay = await find_vectorless_confirm_distinct_replay(
        candidate_id=candidate_id,
        content=request.content,
        span=read_span(request.scope),
        adapter=deps.adapter,
        is_live_match=_is_live_decision,
    )
    if replay is not None:
        return SaveResult(
            outcome="already_saved",
            hypomnema=HypomnemaRef(id=replay.id, kind="decision"),
        )

    bounce_row = await find_bounce_row(
        deps.adapter, bounce_id=resolve.bounce_id, candidate_id=candidate_id
    )
    # An unknown candidate id is allowed here, so an unverifiable match defaults to ``False``.
    matched = False
    if bounce_row is not None:
        cd_candidate = await get_or_none(deps.adapter, candidate_id)
        if cd_candidate is not None:
            matched = pick_matches(
                request, bounce_row, action="confirm_distinct", candidate=cd_candidate
            )
    new_record_id = new_id() if bounce_row is not None else None
    if bounce_row is not None:
        bounce_row = await begin_caller_exit(
            deps.adapter, bounce_row, action="confirm_distinct", matched=matched,
            effect_ref=None,
            new_record_id=str(new_record_id) if new_record_id is not None else None,
            incoming_text=None, now=now,
        )
    async with StampFailedOnError(deps.adapter, lambda: now, row_group(bounce_row)):
        result = await _write_vectorless(
            request, deps, now=now, reconciles=None, id=new_record_id
        )
    if result.hypomnema is not None:
        pair = DistinctPair(
            lower_id=min(result.hypomnema.id, candidate_id, key=str),
            higher_id=max(result.hypomnema.id, candidate_id, key=str),
            source="save",
            judged_at=now,
            similarity=None,
        )
        await deps.adapter.mark_distinct(pair)
    await _finish_caller_exit(deps, resolve, bounce_row, now=now)
    return result


async def _vectorless_reinforce_record(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate: Fact | Decision,
    now: datetime,
) -> SaveResult:
    row = await begin_caller_reinforce(
        request, deps, caller, candidate, effect_now=now, at=lambda: now
    )
    async with StampFailedOnError(deps.adapter, lambda: now, row_group(row)):
        await reinforce(
            target_id=candidate.id,
            source="dedup",
            now=now,
            matched_from_id=None,
            record=candidate,
            adapter=deps.adapter,
            cfg=deps.cfg,
        )
    await maybe_upkeep(deps.adapter, deps.cfg)
    await _finish_caller_exit(deps, caller.resolve, row, now=now)
    return SaveResult(outcome="reinforced", reinforced_id=candidate.id)


async def _vectorless_reinforce(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    resolve: BounceResolution,
    now: datetime,
) -> SaveResult:
    candidate = await _load_reinforce_target(deps.adapter, resolve.candidate_id)
    bounce_row = await find_bounce_row(
        deps.adapter, bounce_id=resolve.bounce_id, candidate_id=resolve.candidate_id
    )
    caller = CallerReinforce(
        resolve=resolve,
        row=bounce_row,
        matched=pick_matches(request, bounce_row, action="reinforce", candidate=candidate),
    )
    if candidate.validity == "invalid":
        return await _write_vectorless(request, deps, now=now, reconciles=candidate.id)

    if _out_of_play(candidate):
        return await _write_vectorless(request, deps, now=now, reconciles=None)

    return await _vectorless_reinforce_record(request, deps, caller, candidate, now)


async def _handle_vectorless_resolve_decision(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    *,
    resolve: BounceResolution,
    now: datetime,
) -> SaveResult:
    """Honor a ``BounceResolution`` while the embedder is unavailable.

    A named candidate needs no query vector, so ``reinforce`` and ``confirm_distinct`` behave
    as in ``_handle_resolve_decision`` minus the best-hit and floor comparison.
    ``save_decision`` validated a ``reinforce`` candidate before the embedder ran; the fresh
    re-read here covers a race since. ``confirm_distinct`` replays idempotently through
    ``find_vectorless_confirm_distinct_replay`` before writing.

    A ``reinforce`` against a Decision that is not ``in_force``, or against a shelved Fact,
    falls through to a fresh vector-less write, matching the live path, where the probe
    excludes both and a stale resolve reaches the "no best hit" write. An invalid candidate
    of either kind is written past as a reconciliation.

    Caller-exit stamping mirrors ``facts._handle_vectorless_resolve``. The invalid,
    not-in-force and shelved fallbacks stay unstamped, as in ``_handle_resolve_decision``.
    Every caller-exit stamp, write and reinforce here uses the passed ``now``.
    """
    if resolve.action == "confirm_distinct":
        return await _vectorless_confirm_distinct(request, deps, resolve, now)

    return await _vectorless_reinforce(request, deps, resolve, now)


async def _resolve_confirm_distinct(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    resolve: BounceResolution,
    candidate: Fact | Decision,
    bounce_row: VerdictRow | None,
    probe: Probe,
    vector: list[float],
) -> SaveResult:
    """Write the new Decision alongside the candidate and register the distinct pair.

    The candidate is left untouched; registering the pair keeps it from bouncing again.
    """
    _log.debug(
        "resolve(decision) confirm_distinct: writing Decision alongside candidate %s",
        candidate.id,
    )
    matched = pick_matches(request, bounce_row, action="confirm_distinct", candidate=candidate)
    new_record_id = new_id() if bounce_row is not None else None
    if bounce_row is not None:
        bounce_row = await begin_caller_exit(
            deps.adapter, bounce_row, action="confirm_distinct", matched=matched,
            effect_ref=None,
            new_record_id=str(new_record_id) if new_record_id is not None else None,
            incoming_text=None, now=probe.now,
        )
    async with StampFailedOnError(deps.adapter, lambda: probe.now, row_group(bounce_row)):
        result = await _write_as_requested(
            request, deps, vector=vector, now=probe.now, reconciles=None, id=new_record_id
        )
    if result.hypomnema is not None:
        pair = DistinctPair(
            lower_id=min(result.hypomnema.id, candidate.id, key=str),
            higher_id=max(result.hypomnema.id, candidate.id, key=str),
            source="save",
            judged_at=probe.now,
            similarity=probe.cosine,
        )
        await deps.adapter.mark_distinct(pair)
        _log.debug(
            "resolve(decision) confirm_distinct: marked pair (%s, %s)",
            pair.lower_id,
            pair.higher_id,
        )
    await _finish_caller_exit(deps, resolve, bounce_row, now=probe.now)
    return result


async def _resolve_reinforce(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate: Fact | Decision,
    probe: Probe,
    vector: list[float],
) -> SaveResult:
    """Reinforce the candidate (agent-adjudicated false alarm); an invalid one is written past.

    An invalid candidate of either kind is never reinforced: the new Decision is written with
    ``reconciles`` set, and the caller's row stays ``unanswered``.
    """
    if candidate.validity == "invalid":
        if isinstance(candidate, Decision):
            _log.debug(
                "resolve(decision) reinforce: candidate %s is invalid; writing novel",
                candidate.id,
            )
        return await _write_as_requested(
            request, deps, vector=vector, now=probe.now, reconciles=candidate.id
        )

    if isinstance(candidate, Decision):
        _log.debug(
            "resolve(decision) reinforce: Decision id=%s — emitting dedup event",
            candidate.id,
        )
    row = await begin_caller_reinforce(
        request, deps, caller, candidate, effect_now=probe.now, at=lambda: probe.now
    )
    async with StampFailedOnError(deps.adapter, lambda: probe.now, row_group(row)):
        await reinforce(
            target_id=candidate.id,
            source="dedup",
            now=probe.now,
            matched_from_id=None,
            record=candidate,
            adapter=deps.adapter,
            cfg=deps.cfg,
            similarity=probe.cosine,
        )
    await maybe_upkeep(deps.adapter, deps.cfg)
    await _finish_caller_exit(deps, caller.resolve, row, now=probe.now)
    return SaveResult(outcome="reinforced", reinforced_id=candidate.id)


async def _handle_resolve_decision(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    *,
    resolve: BounceResolution,
    candidate: Fact | Decision,
    probe: Probe,
    arm_floor: float,
    vector: list[float],
) -> SaveResult:
    """Honor a BounceResolution for a Decision save.

    Called when the candidate is still best, above its arm floor, and a resolve targeting it
    was supplied.

    - ``confirm_distinct``: write the new Decision alongside the candidate.
    - ``reinforce``: reinforce the candidate (agent-adjudicated false alarm). Forbidden for
      invalid candidates.

    ``request.supersession`` is passed to every ``_write_decision`` call. Every caller-exit
    stamp, write and reinforce uses ``probe.now``.
    """
    if probe.cosine < arm_floor:
        _log.debug(
            "resolve(decision): candidate %s below arm floor %.4f; writing novel",
            candidate.id,
            arm_floor,
        )
        return await _write_as_requested(
            request, deps, vector=vector, now=probe.now, reconciles=None
        )

    # The verdict row this resolve names, as in ``facts._handle_resolve``.
    bounce_row = await find_bounce_row(
        deps.adapter, bounce_id=resolve.bounce_id, candidate_id=candidate.id
    )
    if resolve.action == "confirm_distinct":
        return await _resolve_confirm_distinct(
            request, deps, resolve, candidate, bounce_row, probe, vector
        )

    caller = CallerReinforce(
        resolve=resolve,
        row=bounce_row,
        matched=pick_matches(request, bounce_row, action="reinforce", candidate=candidate),
    )
    return await _resolve_reinforce(request, deps, caller, candidate, probe, vector)


async def _resolve_vectorless_save_decision(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    *,
    probe_span: list[Scope],
) -> tuple[SaveResult, list[Fact | Decision]]:
    """Decision-arm mirror of ``facts._resolve_vectorless_save``: BM25-only probe, top five.

    Never calls the Assessor; it returns the neighbour records with an unjudged bounce so the
    caller judges them after its ``async with lock:`` has exited.

    A ``promoted_from`` naming the BM25 primary candidate is a one-shot write-past, mirroring
    the live path's promote bypass; without it, retrying the exit's own ``promoted_from``
    call would bounce forever. ``is_promotable`` is re-checked here rather than trusting the
    earlier unlocked validation.
    """
    neighbors = await bm25_probe(
        content=request.content,
        span=probe_span,
        exclude=request.supersede_ids,
        adapter=deps.adapter,
    )
    if not neighbors or _names_promotion(request, neighbors[0].record):
        result = await _write_vectorless(
            request, deps, now=deps.clock.now(), reconciles=None
        )
        return result, []

    primary = neighbors[0].record
    other_candidates = [
        BounceNeighbor(candidate_id=n.record.id, similarity=None, bm25_rank=n.rank)
        for n in neighbors[1:]
    ]
    bounce = DecisionBounce(
        candidate_id=primary.id,
        candidate_content=primary.content,
        candidate_scope=str(primary.scope),
        similarity=None,
        guard_flag=None,
        verdict="near_dup",
        other_candidates=other_candidates,
        exits=judge_unavailable_exits(primary.id, request.bounce_id),
        judged=None,
        bounce_id=request.bounce_id,
    )
    result = SaveResult(outcome="bounced", bounce=bounce, reason="judge_unavailable")
    return result, [n.record for n in neighbors]


def _admit_replay(replay: Replay, audit: AdmissionAudit) -> SaveResult:
    _log.debug(
        "save(decision) byte-identical replay: content matches "
        "Decision id=%s verbatim; idempotent ack, no write",
        replay.id,
    )
    audit.resolved("replay", "already_saved", str(replay.id))
    return SaveResult(
        outcome="already_saved",
        hypomnema=HypomnemaRef(id=replay.id, kind="decision"),
    )


async def _admit_exact_key_same_or_downward(
    deps: SaveDeps, hit: ExactHit, audit: AdmissionAudit
) -> SaveResult:
    """Reinforce the exact-key match, with a ``raised`` line on failure."""
    record = hit.record
    _log.debug(
        "save(decision) exact-key match (%s): reinforcing "
        "Decision id=%s",
        hit.direction,
        record.id,
    )
    try:
        await reinforce(
            target_id=record.id,
            source="dedup",
            now=deps.clock.now(),
            matched_from_id=None,
            record=record,
            adapter=deps.adapter,
            cfg=deps.cfg,
            similarity=1.0,
        )
        await maybe_upkeep(deps.adapter, deps.cfg)
    except BaseException:
        audit.raised("exact_key", str(record.id))
        raise
    audit.resolved("exact_key", "reinforced", str(record.id))
    return SaveResult(outcome="reinforced", reinforced_id=record.id)


def _bounce_exact_key_upward(
    request: DecisionSaveRequest, hit: ExactHit, audit: AdmissionAudit
) -> SaveResult:
    record = hit.record
    _log.debug(
        "save(decision) exact-key match (upward): bouncing against "
        "Decision id=%s",
        record.id,
    )
    exits = build_upward_exits(
        saved_kind="decision", candidates=[record], bounce_id=request.bounce_id,
        exact_key=True,
    )
    assert len(exits) == 1, "the exact-key upward path offers exactly one exit"
    bounce = DecisionBounce(
        candidate_id=record.id,
        candidate_content=record.content,
        candidate_scope=str(record.scope),
        similarity=1.0,
        guard_flag=None,
        verdict="exact_key",
        other_candidates=[],
        exits=(exits[0],),
        bounce_id=request.bounce_id,
    )
    audit.resolved("exact_key", "bounced", str(record.id))
    return SaveResult(outcome="bounced", bounce=bounce)


async def _admission_steps_1_3_decision(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    *,
    probe_span: list[Scope],
    log_failures: bool = True,
) -> SaveResult | None:
    """Admission steps 1 (replay) and 3 (exact key) for the Decision arm, from either lock hold.

    See ``facts._admission_steps_1_3`` for why the second hold re-runs it, for the step 1 and
    step 3 exclusions and for ``log_failures``. Returns the dispatched ``SaveResult`` on a hit
    (already_saved, reinforced or an upward bounce), or ``None`` on a miss at both steps.
    """
    audit = AdmissionAudit(
        scope=request.scope, span=probe_span, log_raised=log_failures, arm="decision"
    )
    resolve_exclude = request.resolution.exclude_ids
    replay_exclude = resolve_exclude
    admission_exclude = request.supersede_ids | resolve_exclude
    step1 = await check_replay(
        content=request.content,
        kind="decision",
        scope=request.scope,
        exclude=replay_exclude,
        adapter=deps.adapter,
    )
    if isinstance(step1, Replay):
        return _admit_replay(step1, audit)

    step3 = await check_exact_key(
        content=request.content,
        kind="decision",
        scope=request.scope,
        span=probe_span,
        exclude=admission_exclude,
        adapter=deps.adapter,
    )
    if isinstance(step3, ExactHit):
        if step3.direction in ("same", "downward"):
            return await _admit_exact_key_same_or_downward(deps, step3, audit)

        return _bounce_exact_key_upward(request, step3, audit)

    return None


@dataclass(frozen=True, slots=True)
class _HitPool:
    """The probe hits split by what the best-hit selection and the bounce do with them.

    ``viable`` are the hits that are neither the supersedes predecessor, judged distinct from
    a ``confirm_distinct`` candidate, retired nor invalid. ``excluded`` holds the retired and
    invalid ids, which are known only from the hits. ``invalid_candidate`` is the first invalid
    hit that is not the predecessor, a distinct pair or a retired Decision.
    """

    viable: list[Hit]
    excluded: set[HypomnemaId]
    invalid_candidate: Fact | Decision | None


def _split_hits(
    v_hits: list[Hit], supersede_ids: set[HypomnemaId], distinct_exclude: set[HypomnemaId]
) -> _HitPool:
    viable: list[Hit] = []
    excluded: set[HypomnemaId] = set()
    invalid_candidate: Fact | Decision | None = None
    for hit in v_hits:
        cand_id = hit.record.id
        if cand_id in supersede_ids or cand_id in distinct_exclude:
            continue

        rec = hit.record
        if isinstance(rec, Decision) and rec.lifecycle in {"superseded", "revoked"}:
            excluded.add(cand_id)
            continue

        if rec.validity == "invalid":
            if invalid_candidate is None:
                invalid_candidate = rec
            excluded.add(cand_id)
            continue

        viable.append(hit)
    return _HitPool(viable=viable, excluded=excluded, invalid_candidate=invalid_candidate)


@dataclass(frozen=True, slots=True)
class _Survey:
    """The live-vector probe of one hold, with its best hit and that hit's arm floor.

    ``best_hit`` is the first hit that is not the predecessor, a distinct pair, retired or
    invalid, and ``arm_floor`` the floor of its kind (``gray_floor`` without a hit).
    ``probe.v_hits`` holds the viable hits and ``probe.exclude`` the distinct-pair ids, which is
    what ``make_bounce`` takes. ``bounce_hit`` is ``best_hit`` when its cosine reaches the
    floor, else ``None``. ``invalid_candidate`` is the record a below-floor write reconciles.
    """

    probe: Probe
    best_hit: Hit | None
    arm_floor: float
    bounce_hit: Hit | None
    invalid_candidate: Fact | Decision | None


async def _survey_survivors(
    request: DecisionSaveRequest, deps: SaveDeps, probe_span: list[Scope], vector: list[float]
) -> _Survey:
    """Probe the store and pick the best viable hit.

    Probed with only the supersedes predecessor excluded; the distinct-pair read and the
    retired or invalid exclusions need the survivor pool first. Reads the clock once, after
    the probe and the distinct-pair read.
    """
    supersede_ids = request.supersede_ids
    survivors = await serial_probe(
        content=request.content,
        vector=vector,
        span=probe_span,
        k=deps.cfg.probe_k,
        exclude=supersede_ids,
        adapter=deps.adapter,
    )
    v_hits = survivors.hits
    distinct_exclude = await distinct_exclude_ids(
        request.resolution.resolve,
        [(hit.record.id, hit.score) for hit in v_hits],
        deps.adapter,
    )
    pool = _split_hits(v_hits, supersede_ids, distinct_exclude)

    full_exclude = supersede_ids | distinct_exclude | pool.excluded
    best_hit = next((hit for hit in v_hits if hit.record.id not in full_exclude), None)
    best_cosine = best_hit.score if best_hit is not None else 0.0
    now = deps.clock.now()

    arm_floor = (
        floor_for(deps.cfg, best_hit.record) if best_hit is not None else deps.cfg.gray_floor
    )
    return _Survey(
        probe=Probe(cosine=best_cosine, v_hits=pool.viable, exclude=distinct_exclude, now=now),
        best_hit=best_hit,
        arm_floor=arm_floor,
        bounce_hit=best_hit if best_hit is not None and best_cosine >= arm_floor else None,
        invalid_candidate=pool.invalid_candidate,
    )


def _bounce_verdict(candidate: Fact | Decision, guard: GuardName | None) -> BounceVerdictName:
    """A Decision against a Decision is always ``decision_adjacent``; the guard is evidence only."""
    if isinstance(candidate, Decision):
        return "decision_adjacent"

    return "guard_trip" if guard is not None else "near_dup"


async def _redispatch_after_world_moved_decision(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    *,
    vector: list[float],
    probe_span: list[Scope],
) -> SaveResult:
    """Handle ``candidate_changed``: the judged candidate moved while the Assessor ran.

    Mirror of ``facts._redispatch_after_world_moved``. Re-runs survivor selection under this
    lock hold and dispatches on the result. It is narrower than ``_dispatch_candidate``: it never
    replays the promote bypass or a resolve-matching-candidate branch, both of which assume
    the judged candidate is still in play. The below-floor branch logs after the write because
    ``_apply_verdict_decision``'s failure handling already reports a raise.
    """
    survey = await _survey_survivors(request, deps, probe_span, vector)
    probe = survey.probe
    best_hit = survey.best_hit

    if best_hit is None or probe.cosine < survey.arm_floor:
        result = await _write_as_requested(
            request, deps, vector=vector, now=probe.now, reconciles=None
        )
        log_admission_resolved(
            arm="decision", step="withheld", outcome=result.outcome, id="-",
            scope=request.scope, span=probe_span,
        )
        return result

    candidate = best_hit.record
    guard = guard_check(request.content, candidate.content)
    result = bounce_candidate(
        request, deps, probe, candidate, guard, _bounce_verdict(candidate, guard)
    )
    if result.bounce is not None:
        result = result.model_copy(
            update={
                "bounce": result.bounce.model_copy(
                    update=bounce_withheld_update("candidate_changed")
                )
            }
        )
    log_admission_resolved(
        arm="decision", step="withheld", outcome=result.outcome,
        id=admission_result_id(result), scope=request.scope, span=probe_span,
    )
    return result


async def _redispatch_orphaned(
    request: DecisionSaveRequest, deps: SaveDeps, batch: JudgedBatch
) -> SaveResult:
    """Redispatch because the judged candidate moved while the Assessor ran.

    No exit acts on these verdicts, but the redispatch runs its own probe and judge cycle, so
    the batch already obtained is still recorded ``unanswered`` first.
    """
    await record_unanswered(request, deps, batch)
    return await _redispatch_after_world_moved_decision(
        request,
        deps,
        vector=batch.vector,
        probe_span=batch.probe_span,
    )


def _demote_conflicting_exit(
    request: DecisionSaveRequest, fresh: Fact | Decision, exit_: Dispatch | Withheld
) -> Dispatch | Withheld:
    """``WriteNew`` in place of a ``Moved`` or ``Supersede`` that conflicts with the predecessor.

    A flip with ``reason="moved"`` never shares a successor with another flip, because
    ``unsupersede`` deletes the successor and could not undo them separately. So an auto exit
    naming a candidate other than the caller's ``supersedes`` id becomes ``WriteNew`` when
    the caller's ``reason`` is ``"moved"`` or the auto exit is ``Moved``. A caller flip with
    another reason composing with an unrelated auto ``Supersede`` is allowed. Pure: no
    awaits, clock reads or log lines.
    """
    supersession = request.supersession
    if supersession is None:
        return exit_

    auto_target_ids: set[HypomnemaId] = (
        {candidate.id for candidate in exit_.candidates}
        if isinstance(exit_, Moved)
        else {fresh.id} if isinstance(exit_, Supersede)
        else set()
    )
    composes_different_target = (
        bool(auto_target_ids) and supersession.supersedes[0] not in auto_target_ids
    )
    if composes_different_target and (
        supersession.reason == "moved" or isinstance(exit_, Moved)
    ):
        return WriteNew()

    return exit_


async def _bounce_withheld_decision(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    batch: JudgedBatch,
    fresh: Fact | Decision,
    withheld: Withheld,
) -> SaveResult:
    """Record every survivor's verdict ``unanswered`` and return the original bounce updated.

    The update comes from ``bounce_withheld_update``, with the judged neighbours attached.
    When the top candidate is promotable, a second promotable candidate turns the withheld
    reason into ``competing_candidate`` with an upward exit for the top one, and otherwise a
    ``switch_off`` reason replaces the exits with the upward exit.
    """
    await record_unanswered(request, deps, batch)
    log_admission_resolved(
        arm="decision", step="withheld", outcome="bounced",
        id=str(fresh.id), scope=request.scope, span=batch.probe_span,
    )
    other_verdicts = batch.verdicts[1:]
    update = bounce_withheld_update(withheld=withheld.reason, verdict=batch.verdict)
    update["other_candidates"] = attach_neighbor_verdicts(
        batch.result_bounce.other_candidates, batch.candidate_ids[1:], other_verdicts
    )
    moved_candidates = upward_promotion_candidates(
        incoming_kind="decision", typed_scope=request.scope,
        top_verdict=batch.verdict, top_candidate=fresh,
        other_verdicts=other_verdicts, other_candidates=batch.other_judged_candidates,
    )
    top_is_promotable = bool(moved_candidates) and moved_candidates[0].id == fresh.id
    if top_is_promotable:
        if len(moved_candidates) > 1:
            update["withheld"] = "competing_candidate"
            update["exits"] = build_upward_exits(
                saved_kind="decision",
                candidates=[fresh],
                bounce_id=request.bounce_id,
            )
        elif withheld.reason == "switch_off":
            update["exits"] = build_upward_exits(
                saved_kind="decision",
                candidates=moved_candidates,
                bounce_id=request.bounce_id,
            )
    return SaveResult(
        outcome="bounced",
        bounce=batch.result_bounce.model_copy(update=update),
    )


@dataclass(frozen=True, slots=True)
class _AutoWrite:
    """What an automatic exit that writes a successor hands to ``_commit_auto_write``.

    ``rows`` are the pending rows for the acted-on candidates, written together with
    ``sibling_rows``. ``flips`` go to ``_write_decision`` as ``extra_flips``. ``caller_rows``
    are the caller-replay rows stamped with the write.
    """

    new_record_id: HypomnemaId
    rows: list[VerdictRow]
    sibling_rows: list[VerdictRow]
    flips: list[PredecessorFlip]
    caller_rows: list[VerdictRow]
    step: Literal["auto_write", "auto_supersede"]


async def _commit_auto_write(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    frame: AutoExitFrame,
    write: _AutoWrite,
) -> SaveResult:
    """Append the pending rows, write the successor, stamp the rows ``acted`` and log the exit.

    The pending rows and the caller-replay rows are stamped ``failed`` if the write raises an
    ``Exception``; a cancellation stamps nothing. The caller-replay rows are stamped ``acted``
    after the exit's own rows. The write uses ``frame.now``; each stamp reads the clock.
    """
    await deps.adapter.append_verdicts([*write.rows, *write.sibling_rows])
    async with StampFailedOnError(deps.adapter, deps.clock.now, write.rows, write.caller_rows):
        result = await _write_decision(
            request,
            deps,
            vector=frame.batch.vector,
            now=frame.now,
            supersession=None,
            reconciles=None,
            extra_flips=write.flips,
            id=write.new_record_id,
        )
    await stamp_acted_many(deps.adapter, write.rows, now=deps.clock.now())
    if write.caller_rows:
        assert request.resolution.bounce_id is not None
        await stamp_acted_many_with_siblings(
            deps.adapter, write.caller_rows, bounce_id=request.resolution.bounce_id,
            exclude_candidate_ids=request.supersede_ids,
            now=deps.clock.now(),
        )
    log_admission_resolved(
        arm="decision", step=write.step, outcome=result.outcome,
        id=admission_result_id(result), scope=request.scope, span=frame.batch.probe_span,
    )
    return result


async def _auto_reinforce(
    request: DecisionSaveRequest, deps: SaveDeps, frame: AutoExitFrame
) -> SaveResult:
    batch = frame.batch
    fresh = frame.fresh
    effect_ref = f"{fresh.id!s}@{frame.now.isoformat()}"
    row = build_pending_row(
        candidate_id=fresh.id, similarity=batch.judged_similarity, verdict=batch.verdict,
        action="reinforce", effect_ref=effect_ref, new_record_id=None, now=frame.now,
        incoming_text=request.content,
    )
    await deps.adapter.append_verdicts([row, *frame.sibling_rows])
    async with StampFailedOnError(deps.adapter, deps.clock.now, [row]):
        await reinforce(
            target_id=fresh.id, source="dedup", now=frame.now, matched_from_id=None,
            record=fresh, adapter=deps.adapter, cfg=deps.cfg,
            similarity=batch.judged_similarity,
        )
    await stamp_acted_many(deps.adapter, [row], now=deps.clock.now())
    await maybe_upkeep(deps.adapter, deps.cfg)
    log_admission_resolved(
        arm="decision", step="auto_reinforce", outcome="reinforced",
        id=str(fresh.id), scope=request.scope, span=batch.probe_span,
    )
    return SaveResult(outcome="reinforced", reinforced_id=fresh.id)


async def _auto_write_new(
    request: DecisionSaveRequest, deps: SaveDeps, frame: AutoExitFrame
) -> SaveResult:
    batch = frame.batch
    admission_hit = await _admission_steps_1_3_decision(
        request,
        deps,
        probe_span=batch.probe_span,
        log_failures=False,
    )
    if admission_hit is not None:
        # A concurrent admission already won; the verdicts are still recorded.
        await record_unanswered(request, deps, batch)
        return admission_hit

    new_record_id = new_id()
    caller_flips, caller_rows = await resolve_caller_replay_flips(
        request,
        deps,
        new_record_id=new_record_id,
        now=frame.now,
    )
    row = build_pending_row(
        candidate_id=frame.fresh.id, similarity=batch.judged_similarity, verdict=batch.verdict,
        action="write_new", effect_ref=None, new_record_id=str(new_record_id), now=frame.now,
    )
    write = _AutoWrite(
        new_record_id=new_record_id,
        rows=[row],
        sibling_rows=frame.sibling_rows,
        flips=caller_flips,
        caller_rows=caller_rows,
        step="auto_write",
    )
    return await _commit_auto_write(request, deps, frame, write)


async def _auto_supersede(
    request: DecisionSaveRequest, deps: SaveDeps, frame: AutoExitFrame, supersede: Supersede
) -> SaveResult:
    batch = frame.batch
    fresh = frame.fresh
    new_record_id = new_id()
    caller_flips, caller_rows = await resolve_caller_replay_flips(
        request,
        deps,
        new_record_id=new_record_id,
        now=frame.now,
    )
    caller_flip_ids = {flip.id for flip in caller_flips}
    edge_id = str(uuid4())
    auto_flips: list[PredecessorFlip] = (
        []
        if fresh.id in caller_flip_ids
        else [PredecessorFlip(
            id=fresh.id, reason=supersede.reason, note=None, edge_id=edge_id,
        )]
    )
    row = build_pending_row(
        candidate_id=fresh.id, similarity=batch.judged_similarity, verdict=batch.verdict,
        action="supersede", effect_ref=edge_id, new_record_id=str(new_record_id), now=frame.now,
    )
    write = _AutoWrite(
        new_record_id=new_record_id,
        rows=[row],
        sibling_rows=frame.sibling_rows,
        flips=[*caller_flips, *auto_flips],
        caller_rows=caller_rows,
        step="auto_supersede",
    )
    return await _commit_auto_write(request, deps, frame, write)


async def _auto_moved(
    request: DecisionSaveRequest, deps: SaveDeps, frame: AutoExitFrame, moved: Moved
) -> SaveResult:
    batch = frame.batch
    new_record_id = new_id()
    caller_flips, caller_rows = await resolve_caller_replay_flips(
        request,
        deps,
        new_record_id=new_record_id,
        now=frame.now,
    )
    caller_flip_ids = {flip.id for flip in caller_flips}
    similarity_by_id = dict(zip(batch.candidate_ids, batch.similarities, strict=True))
    verdict_by_id = dict(zip(batch.candidate_ids, batch.verdicts, strict=True))
    moved_flips: list[PredecessorFlip] = []
    moved_rows: list[VerdictRow] = []
    claimed_ids: set[HypomnemaId] = set()
    for target in moved.candidates:
        claimed_ids.add(target.id)
        if target.id in caller_flip_ids:
            continue

        target_edge_id = str(uuid4())
        moved_flips.append(
            PredecessorFlip(id=target.id, reason="moved", note=None, edge_id=target_edge_id)
        )
        target_verdict = verdict_by_id[target.id]
        assert isinstance(target_verdict, RelationVerdict)
        moved_rows.append(
            build_pending_row(
                candidate_id=target.id, similarity=similarity_by_id.get(target.id),
                verdict=target_verdict, action="supersede", effect_ref=target_edge_id,
                new_record_id=str(new_record_id), now=frame.now,
            )
        )
    claimed_strs = {str(cid) for cid in claimed_ids}
    unclaimed_sibling_rows = [
        r for r in frame.sibling_rows if r.candidate_id not in claimed_strs
    ]
    write = _AutoWrite(
        new_record_id=new_record_id,
        rows=moved_rows,
        sibling_rows=unclaimed_sibling_rows,
        flips=[*caller_flips, *moved_flips],
        caller_rows=caller_rows,
        step="auto_supersede",
    )
    return await _commit_auto_write(request, deps, frame, write)


async def _run_auto_exit(
    request: DecisionSaveRequest, deps: SaveDeps, frame: AutoExitFrame, exit_: Dispatch
) -> SaveResult:
    if isinstance(exit_, Reinforce):
        return await _auto_reinforce(request, deps, frame)

    if isinstance(exit_, WriteNew):
        return await _auto_write_new(request, deps, frame)

    if isinstance(exit_, Supersede):
        return await _auto_supersede(request, deps, frame, exit_)

    if isinstance(exit_, Moved):
        return await _auto_moved(request, deps, frame, exit_)

    assert isinstance(exit_, Invalidate)
    raise AssertionError(
        "unreachable: legal_exit never returns Invalidate for incoming_kind='decision'"
    )


async def _apply_verdict_decision(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    *,
    verdict: RelationVerdict,
    verdicts: Sequence[RelationVerdict | Unavailable],
    candidate_ids: Sequence[HypomnemaId],
    similarities: Sequence[float | None],
    judged_candidate: Fact | Decision,
    judged_similarity: float | None,
    other_judged_candidates: Sequence[Fact | Decision],
    result_bounce: DecisionBounce,
    vector: list[float],
    probe_span: list[Scope],
) -> SaveResult:
    """Dispatch a judged bounce through ``clean_picture``: the Decision arm's automatic-exit path.

    See ``facts._apply_verdict`` for verdict-row shape, neighbour ``judged`` attachment,
    ``effect_ref`` and the ``Exception``-not-``BaseException`` stamping sites. This arm never
    reaches ``Invalidate``, because ``legal_exit`` returns it only for a Fact incoming.

    Re-takes ``deps.lock`` and re-reads ``judged_candidate`` before deciding, so
    ``candidate_still_fresh`` is checked inside this hold.

    ``request.supersession`` names a single id; a caller replay goes through
    ``resolve_caller_replay_flips``, which builds one flip for it.

    ``clean_picture`` caps ``Moved`` at one candidate here, so the ``Moved`` branch composes a
    single candidate. Two or more survivors reach ``Withheld`` as ``competing_candidate``, and
    the bounce's recommended exit names only the top survivor, the one shape a single-id
    ``supersedes`` can replay. ``_demote_conflicting_exit`` applies the rule that keeps a
    ``moved`` flip from sharing a successor with another flip.
    """
    switches = AutoExitSwitches.from_config(deps.cfg)
    batch = JudgedBatch(
        verdict=verdict,
        verdicts=verdicts,
        candidate_ids=candidate_ids,
        similarities=similarities,
        judged_candidate=judged_candidate,
        judged_similarity=judged_similarity,
        other_judged_candidates=other_judged_candidates,
        result_bounce=result_bounce,
        vector=vector,
        probe_span=probe_span,
    )

    async with deps.lock:
        fresh_candidate = await get_or_none(deps.adapter, judged_candidate.id)
        if not candidate_still_fresh(judged=judged_candidate, fresh=fresh_candidate):
            return await _redispatch_orphaned(request, deps, batch)

        assert fresh_candidate is not None

        exit_ = clean_picture(
            incoming_kind="decision",
            typed_scope=request.scope,
            top_verdict=verdict,
            top_candidate=fresh_candidate,
            other_verdicts=verdicts[1:],
            other_candidates=other_judged_candidates,
            switches=switches,
        )
        exit_ = _demote_conflicting_exit(request, fresh_candidate, exit_)

        if isinstance(exit_, Withheld):
            return await _bounce_withheld_decision(request, deps, batch, fresh_candidate, exit_)

        if isinstance(exit_, Moved):
            refreshed = await refresh_moved(deps, fresh_candidate, exit_)
            if refreshed is None:
                return await _redispatch_orphaned(request, deps, batch)

            exit_ = refreshed

        now = deps.clock.now()
        # Every verdict that reached the Assessor gets a row: siblings of the acted-on
        # survivor are written ``unanswered`` with ``bounce_id=None``.
        sibling_rows = (
            build_unanswered_rows(
                bounce_id=None, candidate_ids=candidate_ids[1:], similarities=similarities[1:],
                verdicts=verdicts[1:], now=now, start_rank=1,
            )
            if len(candidate_ids) > 1
            else []
        )
        frame = AutoExitFrame(
            batch=batch, fresh=fresh_candidate, now=now, sibling_rows=sibling_rows
        )
        return await _run_auto_exit(request, deps, frame, exit_)


async def _check_supersedes_target(
    adapter: StorageAdapter,
    supersedes: HypomnemaId | None,
    *,
    reason: SupersessionReason | None,
    span: list[Scope],
    probe_span: list[Scope],
) -> tuple[SupersedesSnapshot, ...]:
    """The lineage snapshot chain for ``supersedes``, or ``()`` when none is named.

    The target must be an in-force Decision inside the read span. The read span applies to
    every reason except ``moved``, whose upward retry is checked against the wider probe span
    (``supersede_legality_span``).
    """
    if supersedes is None:
        return ()

    try:
        target = await adapter.get_by_id(supersedes)
    except KeyError:
        raise ToolError(
            f"supersedes: unknown id={supersedes!s} (buried or does not exist)"
        ) from None
    if not isinstance(target, Decision):
        raise ToolError(
            f"supersedes: id={supersedes!s} is a Fact, not a Decision; "
            "supersession targets must be in-force Decisions (SPEC §6)"
        )
    if target.lifecycle != "in_force":
        raise ToolError(
            f"supersedes: Decision id={supersedes!s} is not in_force "
            f"(lifecycle={target.lifecycle!r}); chains are linear — only "
            "in-force Decisions may be superseded (SPEC §6)"
        )
    legality_span = supersede_legality_span(reason=reason, read_span=span, probe_span=probe_span)
    if target.scope not in legality_span:
        raise ToolError(
            f"supersedes: Decision id={supersedes!s} belongs to scope "
            f"{target.scope!r} which is outside the caller's read span "
            f"{[str(s) for s in legality_span]!r}; cross-scope supersession is not allowed"
        )
    return (*target.supersession_chain, make_supersedes_snapshot(target))


async def _snapshot_promoted_fact(
    adapter: StorageAdapter, eid: HypomnemaId, span: list[Scope]
) -> PromotedFromSnapshot:
    """The birth-frozen snapshot of one promoted Fact; ``ToolError`` when it cannot be promoted."""
    try:
        target_fact = await adapter.get_by_id(eid)
    except KeyError:
        raise ToolError(
            f"promoted_from: unknown id={eid!s} (buried or does not exist)"
        ) from None
    if not isinstance(target_fact, Fact):
        raise ToolError(
            f"promoted_from: id={eid!s} is a Decision, not a Fact; "
            "only Facts may be promoted (SPEC §8.4)"
        )
    if not is_promotable(target_fact):
        raise ToolError(
            f"promoted_from: Fact id={eid!s} has lifecycle="
            f"{target_fact.lifecycle!r}; only active, hot or ember Facts "
            "may be promoted — archived and shelved Facts are not "
            "(SPEC §8.4)"
        )
    if target_fact.scope not in span:
        raise ToolError(
            f"promoted_from: Fact id={eid!s} belongs to scope "
            f"{target_fact.scope!r} which is outside the caller's read span "
            f"{[str(s) for s in span]!r}; cross-scope promotion is not allowed"
        )
    # recurrence_count counts the Fact's ``source="dedup"`` events.
    dedup_events = await adapter.query_events(target_id=eid, source="dedup")
    return PromotedFromSnapshot(
        id=eid,
        content=target_fact.content,
        valence=target_fact.valence,
        recurrence_count=len(dedup_events),
    )


async def _snapshot_promoted_from(
    adapter: StorageAdapter, promoted_from: list[HypomnemaId] | None, span: list[Scope]
) -> tuple[PromotedFromSnapshot, ...]:
    """One snapshot per ``promoted_from`` id, checked one after another in the given order."""
    if not promoted_from:
        return ()

    snaps: list[PromotedFromSnapshot] = []
    for eid in promoted_from:
        snaps.append(await _snapshot_promoted_fact(adapter, eid, span))
    return tuple(snaps)


@dataclass(frozen=True, slots=True)
class _Lineage:
    """The birth-frozen lineage snapshots the unlocked pre-embed validation took."""

    supersession_chain: tuple[SupersedesSnapshot, ...]
    promoted_snapshots: tuple[PromotedFromSnapshot, ...]


async def _validate_lineage(
    adapter: StorageAdapter,
    typed_scope: Scope,
    *,
    supersedes: HypomnemaId | None,
    reason: SupersessionReason | None,
    promoted_from: list[HypomnemaId] | None,
    resolve: BounceResolution | None,
) -> _Lineage:
    """The unlocked pre-embed validation, in the order its adapter calls are pinned.

    Reads ``list_scopes`` first, then checks the ``supersedes`` target, each ``promoted_from``
    Fact, a ``reinforce`` resolve's target, and finally the reason grouping. It holds no
    lock, so it only shapes the snapshots; ``adapter.replace`` enforces the set-exactly-once
    invariant on ``superseded_by``.
    """
    span = read_span(typed_scope)
    registered = await adapter.list_scopes()
    probe_span = probe_span_for(typed_scope, registered)

    chain = await _check_supersedes_target(
        adapter, supersedes, reason=reason, span=span, probe_span=probe_span
    )
    promoted_snapshots = await _snapshot_promoted_from(adapter, promoted_from, span)

    # A ``reinforce`` resolve must name an existing, unburied target; otherwise ToolError.
    if resolve is not None and resolve.action == "reinforce":
        await _load_reinforce_target(adapter, resolve.candidate_id)

    # Reason and scope validation, before the embedder.
    await check_supersedes_reason(
        predecessor_ids=[supersedes] if supersedes is not None else [],
        reason=reason,
        successor_scope=typed_scope,
        adapter=adapter,
    )
    return _Lineage(supersession_chain=chain, promoted_snapshots=promoted_snapshots)


async def _validate_decision_inputs(
    *,
    content: str,
    scope: str,
    purpose: str | None,
    audience_hint: list[str] | None,
    require_scope: RequireScope,
    require_purpose: RequirePurpose,
    require_roles: RequireRoles,
) -> Scope:
    """The validate stage: reject empty content, resolve the scope and run the three gates.

    Takes the entry function's own validators and inputs one for one, so it has seven
    parameters. Returns the validated scope.
    """
    require_text(content, "content must be non-empty", error=ValueError)

    typed_scope = validate_and_resolve_scope(scope)
    await require_scope(typed_scope)
    await require_purpose(purpose)
    await require_roles(audience_hint)
    return typed_scope


async def _first_hold(request: DecisionSaveRequest, deps: SaveDeps) -> SaveResult | None:
    """Hold 1: admission steps 1-3. A hit returns from inside the hold, so it never embeds."""
    async with deps.lock:
        # Re-read under the lock so a scope registered since the pre-lock fetch is seen.
        registered = await deps.adapter.list_scopes()
        probe_span = probe_span_for(request.scope, registered)
        # Runs whether or not ``resolve`` is set; the exclusions remove the resolve's own
        # candidate.
        return await _admission_steps_1_3_decision(request, deps, probe_span=probe_span)


async def _vectorless_hold(
    request: DecisionSaveRequest, deps: SaveDeps, probe_span: list[Scope]
) -> HoldOutcome:
    """Hold 2 with no vector: a caller's ``resolve`` is honored, else a BM25-only probe runs.

    Both paths can raise from inside a write, hence ``BaseException`` so a cancellation still
    gets its one admission line. The BM25 path returns neighbour records, which become
    ``outage_candidates`` for the Assessor after the lock exits.
    """
    audit = AdmissionAudit(scope=request.scope, span=probe_span, arm="decision")
    resolve = request.resolution.resolve
    outage_candidates: list[Fact | Decision] = []
    if resolve is not None:
        try:
            result = await _handle_vectorless_resolve_decision(
                request,
                deps,
                resolve=resolve,
                now=deps.clock.now(),
            )
        except BaseException:
            audit.raised("resolve", str(resolve.candidate_id))
            raise
    else:
        try:
            result, outage_candidates = await _resolve_vectorless_save_decision(
                request, deps, probe_span=probe_span
            )
        except BaseException:
            audit.raised("vectorless", "-")
            raise
    landed_id = result.hypomnema.id if result.hypomnema is not None else None
    _log.info("vectorless_write kind=decision id=%s trigger=save", landed_id)
    # Outage outcomes share this harvest point, as in ``facts._vectorless_hold``.
    outage_step: Literal["resolve", "bm25_only", "vectorless"] = (
        "resolve"
        if resolve is not None
        else ("bm25_only" if result.outcome == "bounced" else "vectorless")
    )
    audit.resolved(outage_step, result.outcome, admission_result_id(result))
    return HoldOutcome(
        result=result,
        vector=None,
        probe_span=probe_span,
        outage_candidates=outage_candidates,
        assessor_candidates=[],
        assessor_similarities=[],
        assessor_similarity=None,
        pending_admission_log=None,
    )


@dataclass(frozen=True, slots=True)
class _AssessorPick:
    """What the Assessor call needs after the lock exits: the candidates and their scores.

    ``candidates`` is the primary followed by every other viable hit above its own arm floor,
    ``similarities`` their scores in the same order. The Assessor reads them only on a bounce.
    """

    candidates: list[Fact | Decision]
    similarities: list[float | None]
    similarity: float | None


def _assessor_pick(deps: SaveDeps, survey: _Survey, primary: Hit) -> _AssessorPick:
    candidate = primary.record
    candidates: list[Fact | Decision] = [
        candidate,
        *above_floor_candidates(
            survey.probe.v_hits,
            primary_id=candidate.id,
            exclude=survey.probe.exclude,
            cfg=deps.cfg,
        ),
    ]
    score_by_id = {hit.record.id: hit.score for hit in survey.probe.v_hits}
    return _AssessorPick(
        candidates=candidates,
        similarities=[score_by_id.get(record.id) for record in candidates],
        similarity=survey.probe.cosine,
    )


async def _promote_bypass(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    survey: _Survey,
    candidate: Fact | Decision,
    vector: list[float],
) -> SaveResult:
    """One-shot write-past for a ``promoted_from`` that names this bounce's own candidate.

    No resolve and no distinct pair: the engine already adjudicated the pair. Caller replay,
    ``save(promoted_from=[cid], bounce_id=B)``, stamps the row the earlier bounce wrote for
    this candidate. The write uses the survey's ``now``; the row's ``begin_caller_exit`` and
    ``acted`` stamp read the clock.
    """
    _log.debug(
        "save(decision) promote bypass: candidate %s named in "
        "promoted_from; writing past the bounce, no distinct-pair "
        "registration",
        candidate.id,
    )
    resolve_bounce_id = request.resolution.bounce_id
    promote_row = (
        await find_bounce_row(
            deps.adapter, bounce_id=resolve_bounce_id, candidate_id=candidate.id
        )
        if resolve_bounce_id is not None
        else None
    )
    promote_new_id = new_id() if promote_row is not None else None
    promote_matched = pick_matches(
        request, promote_row, action="promoted_from", candidate=candidate
    )
    if promote_row is not None:
        promote_row = await begin_caller_exit(
            deps.adapter, promote_row, action="promoted_from",
            matched=promote_matched, effect_ref=None,
            new_record_id=str(promote_new_id),
            incoming_text=None, now=deps.clock.now(),
        )
    async with StampFailedOnError(deps.adapter, deps.clock.now, row_group(promote_row)):
        result = await _write_as_requested(
            request, deps, vector=vector, now=survey.probe.now, reconciles=None,
            id=promote_new_id,
        )
    if promote_row is not None:
        assert resolve_bounce_id is not None
        await stamp_acted_many_with_siblings(
            deps.adapter, [promote_row], bounce_id=resolve_bounce_id,
            exclude_candidate_ids={candidate.id}, now=deps.clock.now(),
        )
    return result


async def _dispatch_candidate(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    survey: _Survey,
    candidate: Fact | Decision,
    vector: list[float],
) -> SaveResult:
    """The promote bypass, a resolve naming this candidate, or a bounce against it.

    A ``promoted_from`` naming the candidate is checked first, and ``is_promotable`` is
    re-checked here since the Fact may have changed since the unlocked validation. Only the
    primary candidate qualifies; a ``promoted_from`` naming anyone else bounces normally.
    """
    if _names_promotion(request, candidate):
        return await _promote_bypass(request, deps, survey, candidate, vector)

    guard = guard_check(request.content, candidate.content)
    verdict = _bounce_verdict(candidate, guard)
    resolve = request.resolution.resolve
    if resolve is not None and resolve.candidate_id == candidate.id:
        return await _handle_resolve_decision(
            request,
            deps,
            resolve=resolve,
            candidate=candidate,
            probe=survey.probe,
            arm_floor=survey.arm_floor,
            vector=vector,
        )

    _log.debug(
        "save(decision) bounce: candidate id=%s cosine=%.4f verdict=%s",
        candidate.id,
        survey.probe.cosine,
        verdict,
    )
    return bounce_candidate(request, deps, survey.probe, candidate, guard, verdict)


def _eligible_step(request: DecisionSaveRequest, candidate: Fact | Decision) -> AdmissionStep:
    """The ``resolve`` step when a resolve targets this candidate, else ``survivors``."""
    resolve = request.resolution.resolve
    if resolve is not None and resolve.candidate_id == candidate.id:
        return "resolve"

    return "survivors"


async def _dispatch_eligible(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    audit: AdmissionAudit,
    survey: _Survey,
    candidate: Fact | Decision,
    step: AdmissionStep,
    vector: list[float],
) -> SaveResult:
    """``_dispatch_candidate`` with one admission line per save when it raises.

    The promote bypass and ``_handle_resolve_decision`` can both raise from
    ``_write_decision``; ``BaseException`` covers cancellation.
    """
    try:
        return await _dispatch_candidate(request, deps, survey, candidate, vector)
    except BaseException:
        audit.raised(step, str(candidate.id))
        raise


async def _replay_caller_supersede(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    row: VerdictRow,
    survey: _Survey,
    vector: list[float],
    reconciles: HypomnemaId | None,
) -> SaveResult:
    """Caller supersede replay for the single predecessor, as in ``facts._write_below_floor``.

    ``save(supersedes=X, bounce_id=B)`` moves the row the earlier bounce wrote for ``X`` from
    ``unanswered`` to ``pending`` before the write, stamps it ``acted`` after, and draws the
    edge through ``extra_flips`` with a pre-minted edge id. The edge id is minted before the
    new record id.
    """
    supersession = request.supersession
    resolve_bounce_id = request.resolution.bounce_id
    assert resolve_bounce_id is not None
    assert supersession is not None
    predecessor_id = supersession.supersedes[0]

    predecessor = await get_or_none(deps.adapter, predecessor_id)
    matched = predecessor is not None and pick_matches(
        request, row, action="supersede", candidate=predecessor
    )
    edge_id = str(uuid4())
    new_record_id = new_id()
    row = await begin_caller_exit(
        deps.adapter, row, action="supersede",
        matched=matched, effect_ref=edge_id,
        new_record_id=str(new_record_id), incoming_text=None,
        now=deps.clock.now(),
    )
    async with StampFailedOnError(deps.adapter, deps.clock.now, [row]):
        result = await _write_decision(
            request,
            deps,
            vector=vector,
            now=survey.probe.now,
            supersession=None,
            reconciles=reconciles,
            extra_flips=[
                PredecessorFlip(
                    id=predecessor_id,
                    reason=supersession.reason,
                    note=supersession.note,
                    edge_id=edge_id,
                )
            ],
            id=new_record_id,
        )
    await stamp_acted_many_with_siblings(
        deps.adapter, [row], bounce_id=resolve_bounce_id,
        exclude_candidate_ids={predecessor_id}, now=deps.clock.now(),
    )
    return result


async def _write_below_floor(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    audit: AdmissionAudit,
    survey: _Survey,
    vector: list[float],
) -> SaveResult:
    """Write a novel Decision when the best hit is missing or under its arm floor.

    The admission line is written before the write, unlike other branches, so it fires even if
    ``_write_decision`` raises. ``excluded_only`` means the only similar candidate was the named
    predecessor; otherwise ``below_floor``. An invalid candidate seen in the probe becomes the
    record the new Decision reconciles. A caller supersede replay goes through
    ``_replay_caller_supersede`` once an unanswered row exists for the predecessor.
    """
    invalid = survey.invalid_candidate
    reconciles = invalid.id if invalid is not None else None
    audit.resolved("excluded_only" if request.supersede_ids else "below_floor", "saved", "-")

    supersession = request.supersession
    resolve_bounce_id = request.resolution.bounce_id
    supersede_row = (
        await find_bounce_row(
            deps.adapter,
            bounce_id=resolve_bounce_id,
            candidate_id=supersession.supersedes[0],
        )
        if resolve_bounce_id is not None and supersession is not None
        else None
    )
    if supersede_row is None:
        return await _write_as_requested(
            request, deps, vector=vector, now=survey.probe.now, reconciles=reconciles
        )

    return await _replay_caller_supersede(request, deps, supersede_row, survey, vector, reconciles)


async def _probe_hold(
    request: DecisionSaveRequest,
    deps: SaveDeps,
    probe_span: list[Scope],
    vector: list[float],
) -> HoldOutcome | SaveResult:
    """Hold 2 with a vector: steps 1 and 3 again, then the probe and its dispatch.

    A hit at steps 1 and 3 returns its ``SaveResult`` and skips the Assessor. Otherwise the
    survey picks the best viable hit. At or above its arm floor it is dispatched (promote
    bypass, resolve or bounce) and the ``HoldOutcome`` carries a ``pending_admission_log`` for
    it; below the floor, or with no hit, a novel Decision is written and none is carried.
    """
    audit = AdmissionAudit(scope=request.scope, span=probe_span, arm="decision")
    admission_hit = await _admission_steps_1_3_decision(request, deps, probe_span=probe_span)
    if admission_hit is not None:
        return admission_hit

    survey = await _survey_survivors(request, deps, probe_span, vector)
    bounce_hit = survey.bounce_hit
    pick = _AssessorPick(candidates=[], similarities=[], similarity=None)
    pending_admission_log: tuple[AdmissionStep, SaveResult] | None = None
    if bounce_hit is None:
        result = await _write_below_floor(request, deps, audit, survey, vector)
    else:
        candidate = bounce_hit.record
        pick = _assessor_pick(deps, survey, bounce_hit)
        step = _eligible_step(request, candidate)
        result = await _dispatch_eligible(request, deps, audit, survey, candidate, step, vector)
        pending_admission_log = (step, result)

    return HoldOutcome(
        result=result,
        vector=vector,
        probe_span=probe_span,
        outage_candidates=[],
        assessor_candidates=pick.candidates,
        assessor_similarities=pick.similarities,
        assessor_similarity=pick.similarity,
        pending_admission_log=pending_admission_log,
    )


async def _second_hold(
    request: DecisionSaveRequest, deps: SaveDeps, vector: list[float] | None
) -> HoldOutcome | SaveResult:
    """Hold 2: the probe through the terminal write, vector-less or with the query vector."""
    async with deps.lock:
        # Re-read again: the embed round trip sits between the two fetches.
        registered = await deps.adapter.list_scopes()
        probe_span = probe_span_for(request.scope, registered)
        if vector is None:
            return await _vectorless_hold(request, deps, probe_span)

        return await _probe_hold(request, deps, probe_span, vector)


async def save_decision(
    *,
    content: str,
    scope: str,
    source: str | None,
    purpose: str | None,
    audience_hint: list[str] | None,
    supersedes: HypomnemaId | None,
    promoted_from: list[HypomnemaId] | None,
    resolve: BounceResolution | None,
    adapter: StorageAdapter,
    embedder: Embedder,
    clock: Clock,
    cfg: EngineConfig,
    lock: asyncio.Lock,
    require_scope: RequireScope,
    require_purpose: RequirePurpose,
    require_roles: RequireRoles,
    backfill: BackfillSignals,
    reason: SupersessionReason | None = None,
    note: str | None = None,
    assessor: Assessor | None = None,
    resolve_bounce_id: str | None = None,
) -> SaveResult:
    """Create a new Decision, optionally superseding an existing one.

    Runs the same dedup gate as the Fact arm and the same lock discipline as
    ``facts.save_internal``: two write-lock holds with the embed call between them, steps 1-3
    and ``list_scopes`` re-read in the second hold, no lock across an Assessor call, and a
    fresh ``candidate_still_fresh`` re-read afterwards. The held lock also covers the
    ``adapter.replace`` inside ``_write_decision``, so two concurrent ``supersedes=X`` saves
    cannot both read X as unsuperseded. The pre-embed ``supersedes`` and ``promoted_from``
    validation is unlocked and only shapes the birth-frozen lineage snapshot; the
    set-exactly-once invariant on ``superseded_by`` is enforced inside ``adapter.replace``.

    The stages run in this order: ``_validate_decision_inputs``, ``_validate_lineage``,
    ``_first_hold``, ``embed_content``, ``_second_hold`` and ``assessor_tail``, which also
    writes the final admission line.

    ``resolve_bounce_id``: the public ``save`` tool's ``bounce_id``; see ``save_internal``.

    The request value is built after ``check_supersedes_reason``; building it validates
    nothing.

    ``supersedes`` skips the bounce for the named target, since naming it is the resolution.
    ``promoted_from`` naming the bounce's own candidate is a one-shot write-past: the
    candidate stays live, no distinct pair is registered and no ``resolve`` is needed. A
    different candidate above its floor, or a ``promoted_from`` naming anyone else, bounces
    normally.

    ``reason`` and ``note`` are validated once before the embedder by
    ``check_supersedes_reason``: required whenever ``supersedes`` is not ``None``, forbidden
    otherwise. They are forwarded to every ``_write_decision`` call, which mints the edge id
    and lands it in the same ``adapter.replace`` transaction as the new row.

    ``vector is None``: the embedder was unavailable and the similarity probe cannot run. A
    supplied ``resolve`` is honored by ``_handle_vectorless_resolve_decision``; without one,
    the content lands as a vector-less Decision that backfill fills in later.

    ``pending_admission_log`` works as in ``facts.save_internal``, including its two narrow
    ``except BaseException`` guards around the Assessor tail.
    """
    typed_scope = await _validate_decision_inputs(
        content=content,
        scope=scope,
        purpose=purpose,
        audience_hint=audience_hint,
        require_scope=require_scope,
        require_purpose=require_purpose,
        require_roles=require_roles,
    )

    # One id for every bounce this call can build.
    bounce_id = str(uuid4())

    lineage = await _validate_lineage(
        adapter,
        typed_scope,
        supersedes=supersedes,
        reason=reason,
        promoted_from=promoted_from,
        resolve=resolve,
    )

    request = DecisionSaveRequest(
        content=content,
        scope=typed_scope,
        source=source,
        purpose=purpose,
        audience_hint=audience_hint,
        supersession=supersession_from(
            [supersedes] if supersedes is not None else None, reason, note
        ),
        supersession_chain=lineage.supersession_chain,
        promoted_snapshots=lineage.promoted_snapshots,
        promoted_from=promoted_from,
        resolution=ResolveRequest(resolve=resolve, bounce_id=resolve_bounce_id),
        bounce_id=bounce_id,
    )
    deps = SaveDeps(adapter=adapter, clock=clock, cfg=cfg, lock=lock, backfill=backfill)

    admission_hit = await _first_hold(request, deps)
    if admission_hit is not None:
        return admission_hit

    vector = await embed_content(embedder, content, backfill)

    outcome = await _second_hold(request, deps, vector)
    if isinstance(outcome, SaveResult):
        return outcome

    return await assessor_tail(
        request,
        deps,
        assessor=assessor,
        outcome=outcome,
        bounce_type=DecisionBounce,
        apply_verdict=partial(_apply_verdict_decision, request, deps),
    )
