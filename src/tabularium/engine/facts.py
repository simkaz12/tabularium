"""Fact save arm: ``save_internal``, ``_dispatch_match``, ``_handle_resolve``, ``make_bounce``,
``_write_fact`` and the ``reinforce`` composition helper.

``decisions`` imports ``HoldOutcome``, ``above_floor_candidates``, ``admission_result_id``,
``assessor_tail``, ``distinct_exclude_ids``, ``find_vectorless_confirm_distinct_replay``,
``make_bounce``, ``reinforce`` and ``resolve_caller_replay_flips`` from here; ``recall`` imports
``reinforce``; ``core`` imports ``save_internal``.

Admission steps 1-3 (replay, ``supersedes`` / ``resolve`` exclusion, exact content key) live
in ``engine/admission.py`` and run inside the first write-lock hold, before the embedder; see
``save_internal``.

``save_internal`` validates its flat parameters, then builds a ``FactSaveRequest`` and a
``SaveDeps`` (``engine/save_request.py``); the private helpers below it take those two values
instead of the individual parameters.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Literal, Protocol
from uuid import uuid4

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import (
    AdmissionAction,
    AdmissionStep,
    Asserter,
    Decision,
    Fact,
    HypomnemaId,
    Kind,
    SaveLifecycle,
    Scope,
    SupersessionReason,
    Valence,
    new_id,
)
from tabularium.embedding import Embedder, EmbedderUnavailable
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
    from_verdict,
    relation_verdicts,
    scope_direction,
    serial_probe,
    to_assessor_candidates,
    upward_promotion_candidates,
)
from tabularium.engine.assessor import Assessor, RelationVerdict, Unavailable
from tabularium.engine.audit_log import AdmissionStep as LogStep
from tabularium.engine.audit_log import log_admission_resolved, log_failed_supersession
from tabularium.engine.backfill import BackfillSignals
from tabularium.engine.dispositions import (
    check_supersedes_reason,
    supersede_legality_span,
)
from tabularium.engine.ember_recurrence import record_ember_recurrence
from tabularium.engine.exits import build_exits, build_upward_exits, judge_unavailable_exits
from tabularium.engine.filters import (
    probe_filters_for,
    probe_span_for,
    read_span,
    validate_and_resolve_scope,
)
from tabularium.engine.guards import GuardName, guard_check
from tabularium.engine.maintenance import maybe_upkeep
from tabularium.engine.projections import compute_projections
from tabularium.engine.save_request import (
    FactSaveRequest,
    ResolveRequest,
    SaveDeps,
    SaveRequest,
    Supersession,
    supersession_from,
)
from tabularium.engine.verdicts import (
    StampFailedOnError,
    begin_caller_exit,
    build_pending_row,
    build_unanswered_rows,
    caller_exit_matches_pick,
    find_bounce_row,
    stamp_acted_many,
    stamp_acted_many_with_siblings,
)
from tabularium.results import (
    BounceExit,
    BounceNeighbor,
    BounceResolution,
    DecisionBounce,
    FactBounce,
    HypomnemaRef,
    SaveResult,
)
from tabularium.storage.content_key import content_key
from tabularium.storage.types import (
    DecisionInvalidation,
    DistinctPair,
    Hit,
    PredecessorFlip,
    ReinforcementEvent,
    StorageAdapter,
    VerdictRow,
)

_log = logging.getLogger("tabularium")

_GLOBAL = Scope("global")


async def distinct_exclude_ids(
    resolve: BounceResolution | None,
    fused_ids: list[tuple[HypomnemaId, float]],
    adapter: StorageAdapter,
) -> set[HypomnemaId]:
    """Ids already judged distinct from ``resolve.candidate_id``, to exclude from best-hit pick.

    Applies only when ``resolve.action == "confirm_distinct"``; otherwise returns an empty
    set. A ``reinforce`` resolve must re-run the gate like a fresh probe: excluding the
    candidate the content actually matches would reinforce the stale ``resolve.candidate_id``
    instead of bouncing again. A plain save has no id to consult the pair registry with.
    Shared by both arms.
    """
    exclude: set[HypomnemaId] = set()
    if resolve is not None and resolve.action == "confirm_distinct":
        for cand_id, _ in fused_ids:
            if await adapter.is_distinct(resolve.candidate_id, cand_id):
                exclude.add(cand_id)
    return exclude



async def find_vectorless_confirm_distinct_replay(
    *,
    candidate_id: HypomnemaId,
    content: str,
    span: list[Scope],
    adapter: StorageAdapter,
    is_live_match: Callable[[Fact | Decision], bool],
) -> Fact | Decision | None:
    """Find the oldest live row that replays a vector-less ``confirm_distinct`` resolve.

    Admission step 1 is keyed on scope and content, not a candidate id, so it does not cover
    an explicit ``resolve``. Without this check every retry during an embedder outage would
    write another vector-less row.

    Uses ``scan_records`` rather than ``text_search`` because BM25 top-``k`` can rank the
    duplicate below other rows or lose a tie to ``candidate_id`` itself. The full scan is
    acceptable because only a ``confirm_distinct`` resolve during an outage reaches it.

    A match has exactly ``content`` and satisfies ``is_distinct(candidate_id, match.id)``;
    the registry has no reverse lookup by one id. The scan uses
    ``probe_filters_for(span).lifecycle_in`` so shelved Facts stay ineligible, as in the live
    probe. Ties resolve to the lowest ``(created_at, str(id))``.
    """
    matches: list[Fact | Decision] = []
    lifecycle_in = probe_filters_for(span).lifecycle_in
    async for rec in await adapter.scan_records(lifecycle_in=lifecycle_in):
        if rec.id == candidate_id or rec.scope not in span or rec.content != content:
            continue
        if not is_live_match(rec):
            continue
        if await adapter.is_distinct(candidate_id, rec.id):
            matches.append(rec)
    if not matches:
        return None
    return min(matches, key=lambda rec: (rec.created_at, str(rec.id)))


# Reinforced, revived and already_saved outcomes name a record the gate matched, not content
# this call wrote, so no supersession edge is drawn for the caller's ``supersedes``; the reason
# below is returned instead of dropping the request silently.

def _no_edge_reason(supersession: Supersession | None) -> str | None:
    if supersession is None:
        return None
    named = ", ".join(str(pid) for pid in supersession.supersedes)
    return (
        f"supersedes named [{named}] but no edge was drawn: this outcome names a "
        "record the gate matched, not content this call wrote — retry deliberately "
        "with supersede(...) once you know what this call actually resolved to"
    )


def admission_result_id(result: SaveResult) -> str:
    """The id ``log_admission_resolved`` reports for a built ``SaveResult``.

    The fresh row on a write, the reinforced row on a reinforce or revival, the candidate on
    a bounce, else ``"-"`` for a novel write with no matched candidate.
    """
    if result.hypomnema is not None:
        return str(result.hypomnema.id)
    if result.reinforced_id is not None:
        return str(result.reinforced_id)
    if result.bounce is not None:
        return str(result.bounce.candidate_id)
    return "-"


BounceVerdictName = Literal["near_dup", "decision_adjacent", "guard_trip", "exact_key"]


def make_bounce(
    record: Fact | Decision,
    cosine: float,
    guard: GuardName | None,
    verdict: BounceVerdictName,
    *,
    saved_kind: Kind,
    saved_scope: Scope,
    v_hits: list[Hit],
    content: str,
    exclude: set[HypomnemaId],
    cfg: EngineConfig,
    bounce_id: str,
) -> SaveResult:
    """Construct a bounced SaveResult from a candidate record.

    ``other_candidates`` is computed here, only when a bounce is actually built.
    ``saved_kind`` selects ``FactBounce`` or ``DecisionBounce`` (the kind being saved, never
    the candidate's) and feeds ``build_exits`` together with ``saved_scope``. ``bounce_id`` is
    minted once per save call and carried onto the bounce and into every exit's ``how``.
    """
    other_candidates = _bounce_neighbors(
        v_hits, content=content, primary_id=record.id, exclude=exclude, cfg=cfg
    )
    exits = build_exits(
        saved_kind=saved_kind, candidate=record, saved_scope=saved_scope, bounce_id=bounce_id
    )
    assert len(exits) in (2, 3)
    exits_tuple: tuple[BounceExit, BounceExit] | tuple[BounceExit, BounceExit, BounceExit] = (
        (exits[0], exits[1]) if len(exits) == 2 else (exits[0], exits[1], exits[2])
    )
    bounce: FactBounce | DecisionBounce
    if saved_kind == "fact":
        bounce = FactBounce(
            candidate_id=record.id,
            candidate_content=record.content,
            candidate_scope=str(record.scope),
            similarity=cosine,
            guard_flag=guard,
            verdict=verdict,
            other_candidates=other_candidates,
            exits=exits_tuple,
            bounce_id=bounce_id,
        )
    else:
        bounce = DecisionBounce(
            candidate_id=record.id,
            candidate_content=record.content,
            candidate_scope=str(record.scope),
            similarity=cosine,
            guard_flag=guard,
            verdict=verdict,
            other_candidates=other_candidates,
            exits=exits_tuple,
            bounce_id=bounce_id,
        )
    return SaveResult(outcome="bounced", bounce=bounce)


def _bounce_neighbors(
    v_hits: list[Hit],
    *,
    content: str,
    primary_id: HypomnemaId,
    exclude: set[HypomnemaId],
    cfg: EngineConfig,
) -> list[BounceNeighbor]:
    """Other above-floor candidates from the same probe, with no extra store query.

    Each neighbour is tested against the floor of its own kind (``decision_gray_floor`` for
    Decisions, ``gray_floor`` for Facts). The primary candidate and ``exclude`` are dropped.
    Content is omitted (see ``BounceNeighbor``). Sorted by descending similarity with the id
    string as tie-break, for a deterministic envelope.
    """
    neighbors: list[BounceNeighbor] = []
    for hit in v_hits:
        rec = hit.record
        if rec.id == primary_id or rec.id in exclude:
            continue
        arm_floor = cfg.decision_gray_floor if isinstance(rec, Decision) else cfg.gray_floor
        if hit.score < arm_floor:
            continue
        guard = None if isinstance(rec, Decision) else guard_check(content, rec.content)
        neighbors.append(
            BounceNeighbor(candidate_id=rec.id, similarity=hit.score, guard_flag=guard)
        )
    # The id tie-break is explicit because adapters do not promise an order for equal scores.
    # ``or 0.0`` only narrows the type: similarity is always a float on this path.
    neighbors.sort(key=lambda n: (-(n.similarity or 0.0), str(n.candidate_id)))
    return neighbors


def above_floor_candidates(
    v_hits: list[Hit],
    *,
    primary_id: HypomnemaId,
    exclude: set[HypomnemaId],
    cfg: EngineConfig,
) -> list[Fact | Decision]:
    """The population ``_bounce_neighbors`` computes, as full records for the Assessor call."""
    records: list[Fact | Decision] = []
    for hit in v_hits:
        rec = hit.record
        if rec.id == primary_id or rec.id in exclude:
            continue
        arm_floor = cfg.decision_gray_floor if isinstance(rec, Decision) else cfg.gray_floor
        if hit.score < arm_floor:
            continue
        records.append(rec)
    return records


async def _revive_archived_fact(
    *,
    record: Fact,
    now: datetime,
    adapter: StorageAdapter,
    cfg: EngineConfig,
) -> None:
    """Reinforce with ``source="revival"``, then transition archived to active.

    The order means a failed transition leaves the record archived with a harmless revival
    event rather than active with stale projections. Used by the gate's archived-hit dispatch
    and by the ``confirm_distinct`` replay guard.
    """
    await reinforce(
        target_id=record.id,
        source="revival",
        now=now,
        matched_from_id=None,
        record=record,
        adapter=adapter,
        cfg=cfg,
    )
    await adapter.transition(record.id, "active", at=now)
    await maybe_upkeep(adapter, cfg)


async def reinforce(
    *,
    target_id: HypomnemaId,
    source: Literal["dedup", "revival", "compact"],
    now: datetime,
    matched_from_id: HypomnemaId | None,
    record: Fact | Decision,
    adapter: StorageAdapter,
    cfg: EngineConfig,
    similarity: float | None = None,
) -> None:
    """Append a reinforcement event and recompute projections from the full event log.

    Accepts Fact and Decision targets. Decisions have no decay or feedback fields, so those
    are passed to ``update_projections`` as neutral defaults and only the shared counters
    change. ``record`` is the pre-reinforce state, read for ``created_at`` in the decay
    formula (Facts only).

    ``similarity`` is the serial probe's cosine for this target, passed through from a caller
    that computed it. It is meaningful only for ``source="dedup"`` and is otherwise ``None``.
    """
    event = ReinforcementEvent(
        target_id=target_id,
        at=now,
        source=source,
        matched_from_id=matched_from_id,
        value=None,
        similarity=similarity,
    )
    await adapter.append_event(event)

    events = await adapter.query_events(target_id=target_id)
    projections = compute_projections(record, events, cfg)

    await adapter.update_projections(
        target_id,
        recall_count=projections.recall_count,
        last_referenced_at=projections.last_referenced_at,
        feedback_score=projections.feedback_score,
        decay_due_at=projections.decay_due_at,
    )


def _new_fact(
    request: FactSaveRequest,
    *,
    id: HypomnemaId | None,
    reconciles: HypomnemaId | None,
    provisional: bool,
    now: datetime,
) -> Fact:
    return Fact(
        id=id if id is not None else new_id(),
        type="fact",
        content=request.content,
        scope=request.scope,
        source=request.source,
        purpose=request.purpose,
        audience_hint=tuple(request.audience_hint) if request.audience_hint else None,
        lifecycle=request.lifecycle,
        valence=request.valence,
        reconciles=reconciles,
        created_at=now,
        updated_at=now,
        provisional=provisional,
    )


async def _replace_fact(
    request: FactSaveRequest,
    deps: SaveDeps,
    fact: Fact,
    vector: list[float] | None,
    *,
    supersession: Supersession | None,
    flips: list[PredecessorFlip],
    invalidations: Sequence[DecisionInvalidation],
    now: datetime,
) -> None:
    """Write ``fact`` and its flips in one ``adapter.replace`` call.

    A failing ``replace`` is logged as a failed supersession and raised as
    ``SupersessionWriteFailed``, chained to the adapter's exception. The logged reason is the
    request's ``supersession`` reason, else the first flip's, else the first invalidation's.
    """
    try:
        await deps.adapter.replace(fact, vector, flips, invalidations=invalidations, at=now)
    except Exception as exc:
        log_reason = (
            supersession.reason
            if supersession is not None
            else flips[0].reason
            if flips
            else invalidations[0].reason
        )
        log_failed_supersession(
            op="save",
            arm="fact",
            successor_id=fact.id,
            predecessor_ids=[flip.id for flip in flips],
            reason=log_reason,
            scope=request.scope,
            error=exc,
        )
        raise SupersessionWriteFailed(f"supersedes: {exc}") from exc


async def _write_fact(
    request: FactSaveRequest,
    deps: SaveDeps,
    *,
    vector: list[float] | None,
    reconciles: HypomnemaId | None,
    supersession: Supersession | None,
    provisional: bool = False,
    extra_flips: Sequence[PredecessorFlip] = (),
    invalidations: Sequence[DecisionInvalidation] = (),
    embedded: bool = True,
    on_written: Callable[[], None] | None = None,
    id: HypomnemaId | None = None,
) -> SaveResult:
    """Construct, validate and write a new Fact.

    The row's content, scope, source, purpose, audience hint, lifecycle and valence come from
    ``request``; the adapter, clock and config from ``deps``.

    Uses ``upsert`` when ``supersession`` is ``None`` and ``extra_flips`` and ``invalidations``
    are both empty, otherwise a single ``adapter.replace`` transaction.

    ``id``: pre-minted by an automatic-exit caller before its ``pending`` verdict row is
    written, so the row's ``new_record_id`` names the successor before the store changes.
    ``None`` mints one here.

    ``supersession``: when set, one edge id is minted per predecessor, and the edges land in
    the same transaction as the successor row and every other flip; a failure leaves nothing
    written. The automatic-exit callers pass ``None`` and carry the caller's predecessors in
    ``extra_flips`` instead.

    ``extra_flips``: flips beyond those built from ``supersession``, each with its own reason
    class, composed into the same ``replace`` call.

    ``invalidations``: Decisions this Fact successor invalidates in the same ``replace``
    transaction. No supersession edge is drawn from a Fact to a Decision.

    ``vector=None`` with ``embedded=False``: the embedder was unavailable; the row is written
    with a null vector. Only the plain ``save`` outage branch passes ``embedded=False``.

    ``on_written``: called right after the adapter write returns and before ``maybe_upkeep``,
    so the outage branch can mark backfill state dirty even if upkeep is cancelled.

    ``provisional``: ``True`` only for a new ember written by ``engine.embers.save_ember``
    for an agent with no ``audience_hint``.
    """
    now = deps.clock.now()
    fact = _new_fact(request, id=id, reconciles=reconciles, provisional=provisional, now=now)
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
    if flips or invalidations:
        await _replace_fact(
            request,
            deps,
            fact,
            vector,
            supersession=supersession,
            flips=flips,
            invalidations=invalidations,
            now=now,
        )
    else:
        await deps.adapter.upsert(fact, vector)
    if on_written is not None:
        on_written()
    await maybe_upkeep(deps.adapter, deps.cfg)
    outcome: Literal["saved", "reconciled"] = (
        "reconciled" if reconciles is not None else "saved"
    )
    _log.debug(
        "save %s id=%s scope=%s lifecycle=%s",
        outcome,
        fact.id,
        request.scope,
        request.lifecycle,
    )
    return SaveResult(
        outcome=outcome,
        hypomnema=HypomnemaRef(id=fact.id, kind="fact"),
        embedded=embedded,
    )


@dataclass(frozen=True, slots=True)
class AdmissionAudit:
    """Writes the ``admission_resolved`` lines of one save, for its arm, scope and probe span.

    ``arm`` is the saved kind and defaults to ``"fact"``. ``resolved`` always writes.
    ``raised`` writes only while ``log_raised`` is set.
    """

    scope: Scope
    span: list[Scope]
    log_raised: bool = True
    arm: Kind = "fact"

    def resolved(self, step: LogStep, outcome: str, record_id: str) -> None:
        log_admission_resolved(
            arm=self.arm,
            step=step,
            outcome=outcome,
            id=record_id,
            scope=self.scope,
            span=self.span,
        )

    def raised(self, step: LogStep, record_id: str) -> None:
        if self.log_raised:
            self.resolved(step, "raised", record_id)


@dataclass(frozen=True, slots=True)
class Probe:
    """What the live-vector probe found: its hits, the ids it excludes and the hold's clock read.

    ``cosine`` is the best non-excluded hit's score, or ``0.0`` when there is none. ``v_hits``
    and ``exclude`` go to ``make_bounce`` untouched.
    """

    cosine: float
    v_hits: list[Hit]
    exclude: set[HypomnemaId]
    now: datetime


def floor_for(cfg: EngineConfig, record: Fact | Decision) -> float:
    return cfg.decision_gray_floor if isinstance(record, Decision) else cfg.gray_floor


async def get_or_none(adapter: StorageAdapter, record_id: HypomnemaId) -> Fact | Decision | None:
    try:
        return await adapter.get_by_id(record_id)
    except KeyError:
        return None


def bounce_candidate(
    request: SaveRequest,
    deps: SaveDeps,
    probe: Probe,
    record: Fact | Decision,
    guard: GuardName | None,
    verdict: BounceVerdictName,
) -> SaveResult:
    return make_bounce(
        record,
        probe.cosine,
        guard,
        verdict,
        saved_kind=request.kind,
        saved_scope=request.scope,
        v_hits=probe.v_hits,
        content=request.content,
        exclude=probe.exclude,
        cfg=deps.cfg,
        bounce_id=request.bounce_id,
    )


async def _write_as_requested(
    request: FactSaveRequest,
    deps: SaveDeps,
    *,
    vector: list[float],
    reconciles: HypomnemaId | None,
) -> SaveResult:
    return await _write_fact(
        request,
        deps,
        vector=vector,
        reconciles=reconciles,
        supersession=request.supersession,
    )


async def _dispatch_decision_candidate(
    request: FactSaveRequest,
    deps: SaveDeps,
    probe: Probe,
    record: Decision,
    *,
    vector: list[float],
    provisional: bool,
) -> SaveResult:
    """Bounce at the lower ``decision_gray_floor`` after the ordinary guard check, else write."""
    if probe.cosine >= deps.cfg.decision_gray_floor:
        guard = guard_check(request.content, record.content)
        _log.debug("save bounce decision (cosine=%.4f, id=%s)", probe.cosine, record.id)
        return bounce_candidate(request, deps, probe, record, guard, "decision_adjacent")

    return await _write_fact(
        request,
        deps,
        vector=vector,
        reconciles=None,
        supersession=request.supersession,
        provisional=provisional,
    )


async def _dispatch_archived_candidate(
    request: FactSaveRequest,
    deps: SaveDeps,
    probe: Probe,
    record: Fact,
    guard: GuardName | None,
) -> SaveResult:
    """Revive an archived Fact in place, or bounce on a guard trip.

    Reinforce runs before the transition, so a failed transition leaves the record archived
    with a harmless event, not active with stale projections. A guard trip bounces instead, so
    contradictory content is not reactivated.
    """
    if guard is not None:
        return bounce_candidate(request, deps, probe, record, guard, "guard_trip")

    _log.debug("save dedup-revival (cosine=%.4f, id=%s)", probe.cosine, record.id)
    await _revive_archived_fact(record=record, now=probe.now, adapter=deps.adapter, cfg=deps.cfg)
    return SaveResult(
        outcome="revived", reinforced_id=record.id,
        reason=_no_edge_reason(request.supersession),
    )


async def _dispatch_ember_candidate(
    request: FactSaveRequest,
    deps: SaveDeps,
    probe: Probe,
    record: Fact,
    guard: GuardName | None,
    roles: list[str] | None,
) -> SaveResult:
    """Reinforce an ember on canonical-key equality; any other match bounces.

    A guard trip bounces first. The recurrence update runs after the generic reinforce, never
    before, and the role retag shares its reignite and ``probe.now``.
    """
    _log.debug("save ember-recurrence (cosine=%.4f, id=%s)", probe.cosine, record.id)
    if guard is not None:
        return bounce_candidate(request, deps, probe, record, guard, "guard_trip")

    if content_key(request.content) != content_key(record.content):
        return bounce_candidate(request, deps, probe, record, None, "near_dup")

    await reinforce(
        target_id=record.id,
        source="dedup",
        now=probe.now,
        matched_from_id=None,
        record=record,
        adapter=deps.adapter,
        cfg=deps.cfg,
        similarity=probe.cosine,
    )
    suggestion = await record_ember_recurrence(
        record=record, asserted_by=request.asserted_by, adapter=deps.adapter, cfg=deps.cfg,
    )
    if roles is not None:
        await deps.adapter.transition(
            record.id,
            "ember",
            at=probe.now,
            audience_hint=tuple(roles),
            provisional=False if request.asserted_by == "owner" else None,
        )
    await maybe_upkeep(deps.adapter, deps.cfg)
    return SaveResult(
        outcome="reinforced",
        reinforced_id=record.id,
        escalation_suggestion=suggestion,
        reason=_no_edge_reason(request.supersession),
    )


async def _dispatch_match(
    request: FactSaveRequest,
    deps: SaveDeps,
    *,
    best_hit: Hit,
    probe: Probe,
    vector: list[float],
    roles: list[str] | None,
    provisional: bool,
) -> SaveResult:
    """Dispatch on the best candidate returned by the dedup gate.

    Called from ``save_internal`` and from ``_handle_resolve`` when the named candidate is no
    longer the best match, so invalid records, Decisions, guard trips and revival are handled
    identically on both paths.

    ``probe.v_hits`` and ``probe.exclude`` go to ``make_bounce`` untouched and feed
    ``other_candidates``, which is computed only when this call bounces.

    ``roles`` and ``provisional`` are explicit because the resolve path passes ``None`` and
    ``False`` whatever the request carries.

    ``request.asserted_by`` matters only to the ember-recurrence branch, which forwards it to
    ``record_ember_recurrence``.

    ``roles`` matters only to the ember-recurrence branch, the one place this dispatch
    changes an existing record's ``audience_hint``. It overwrites rather than unions, and
    ``None`` leaves the hint untouched. The retag shares the reinforcement's lock hold and
    ``probe.now``, so one call reignites and retags. Other branches write a new record, which
    uses ``audience_hint``, or have no live record to retag.

    ``provisional`` is the owner-verdict-outstanding mark from ``save_ember``. It is
    carried only by the two branches that write a new record: reconciliation past an invalid match,
    and a novel write below ``decision_gray_floor``. The ember-recurrence branch clears it only
    through an owner-asserted ``roles`` retag.
    """
    record = best_hit.record

    if record.validity == "invalid":
        _log.debug("save reconciled past invalid candidate id=%s", record.id)
        return await _write_fact(
            request,
            deps,
            vector=vector,
            reconciles=record.id,
            supersession=request.supersession,
            provisional=provisional,
        )

    if isinstance(record, Decision):
        return await _dispatch_decision_candidate(
            request, deps, probe, record, vector=vector, provisional=provisional
        )

    guard = guard_check(request.content, record.content)

    if isinstance(record, Fact) and record.lifecycle == "archived":
        return await _dispatch_archived_candidate(request, deps, probe, record, guard)

    if isinstance(record, Fact) and record.lifecycle == "ember":
        return await _dispatch_ember_candidate(request, deps, probe, record, guard, roles)

    if isinstance(record, Fact) and record.lifecycle in {"active", "hot"}:
        if guard is not None:
            return bounce_candidate(request, deps, probe, record, guard, "guard_trip")

        return bounce_candidate(request, deps, probe, record, None, "near_dup")

    _log.warning(
        "save: unhandled dispatch arm for record id=%s lifecycle=%s; bouncing",
        record.id,
        record.lifecycle,
    )
    return bounce_candidate(request, deps, probe, record, None, "near_dup")


# Automatic exits: after the Assessor call (made with the lock released) the arm re-takes the
# lock, re-reads the judged candidate and dispatches on ``legal_exit``.


async def _redispatch_after_world_moved(
    request: FactSaveRequest,
    deps: SaveDeps,
    *,
    vector: list[float],
    probe_span: list[Scope],
) -> SaveResult:
    """Handle ``candidate_changed``: the judged candidate moved while the Assessor ran.

    Re-runs survivor selection under the current lock hold through ``_dispatch_match``, so
    candidate-kind rules stay in one place, and writes through or returns a fresh bounce with
    no verdict of its own. The below-floor branch logs after the write rather than before,
    because ``_apply_verdict``'s failure handling already reports a raise under the step
    being resolved and an earlier log would duplicate it.
    """
    survivors = await serial_probe(
        content=request.content, vector=vector, span=probe_span, k=deps.cfg.probe_k,
        exclude=request.supersede_ids, adapter=deps.adapter,
    )
    v_hits = survivors.hits
    distinct_exclude = await distinct_exclude_ids(
        request.resolution.resolve, [(hit.record.id, hit.score) for hit in v_hits], deps.adapter
    )
    candidate_exclude = distinct_exclude | request.supersede_ids
    best_hit = next((hit for hit in v_hits if hit.record.id not in candidate_exclude), None)
    best_cosine = best_hit.score if best_hit is not None else 0.0
    arm_floor = (
        floor_for(deps.cfg, best_hit.record) if best_hit is not None else deps.cfg.gray_floor
    )
    now = deps.clock.now()

    if best_hit is None or best_cosine < arm_floor:
        result = await _write_fact(
            request,
            deps,
            vector=vector,
            reconciles=None,
            supersession=request.supersession,
            provisional=request.provisional,
        )
        log_admission_resolved(
            arm="fact", step="withheld", outcome=result.outcome, id="-",
            scope=request.scope, span=probe_span,
        )
        return result

    result = await _dispatch_match(
        request,
        deps,
        best_hit=best_hit,
        probe=Probe(cosine=best_cosine, v_hits=v_hits, exclude=candidate_exclude, now=now),
        vector=vector,
        roles=request.roles,
        provisional=request.provisional,
    )
    if result.outcome == "bounced" and result.bounce is not None:
        result = result.model_copy(
            update={
                "bounce": result.bounce.model_copy(
                    update=bounce_withheld_update("candidate_changed")
                )
            }
        )
    log_admission_resolved(
        arm="fact", step="withheld", outcome=result.outcome,
        id=admission_result_id(result), scope=request.scope, span=probe_span,
    )
    return result


async def resolve_caller_replay_flips(
    request: SaveRequest,
    deps: SaveDeps,
    *,
    new_record_id: HypomnemaId,
    now: datetime,
) -> tuple[list[PredecessorFlip], list[VerdictRow]]:
    """Build the supersession flips for every id a caller-replay ``save(supersedes=[...])`` names.

    Flips are built here rather than through ``_write_fact`` / ``_write_decision``, which
    mint the edge id inside the write, too late to pre-commit a verdict row to it. When a row
    exists under ``request.resolution.bounce_id`` for an id it moves ``unanswered`` to
    ``pending`` with the same ``effect_ref`` before the write. An id with no matching row
    still gets its flip. A request without a ``supersession`` returns two empty lists.

    Returns ``(flips, pending_rows)``; the caller stamps ``pending_rows`` ``acted`` or
    ``failed`` once the write lands or raises.
    """
    supersession = request.supersession
    if supersession is None:
        return [], []

    resolve_bounce_id = request.resolution.bounce_id
    flips: list[PredecessorFlip] = []
    pending_rows: list[VerdictRow] = []
    for pid in supersession.supersedes:
        edge_id = str(uuid4())
        flips.append(
            PredecessorFlip(
                id=pid, reason=supersession.reason, note=supersession.note, edge_id=edge_id,
            )
        )
        if resolve_bounce_id is None:
            continue
        row = await find_bounce_row(deps.adapter, bounce_id=resolve_bounce_id, candidate_id=pid)
        if row is None:
            continue
        try:
            candidate = await deps.adapter.get_by_id(pid)
        except KeyError:
            candidate = None
        matched = (
            caller_exit_matches_pick(
                row, action="supersede", incoming_kind=request.kind, candidate=candidate,
                direction=scope_direction(
                    incoming_scope=request.scope, candidate_scope=candidate.scope
                ),
            )
            if candidate is not None
            else False
        )
        pending_rows.append(
            await begin_caller_exit(
                deps.adapter, row, action="supersede", matched=matched, effect_ref=edge_id,
                new_record_id=str(new_record_id), incoming_text=None, now=now,
            )
        )
    return flips, pending_rows


@dataclass(frozen=True, slots=True)
class JudgedBatch:
    """The Assessor batch an automatic exit acts on.

    ``verdicts``, ``candidate_ids`` and ``similarities`` run in parallel, primary first, and
    ``verdict`` is ``verdicts[0]``. ``judged_candidate`` and ``judged_similarity`` are the
    primary as the probe saw it. ``result_bounce`` is the bounce this save built before the
    Assessor ran.
    """

    verdict: RelationVerdict
    verdicts: Sequence[RelationVerdict | Unavailable]
    candidate_ids: Sequence[HypomnemaId]
    similarities: Sequence[float | None]
    judged_candidate: Fact | Decision
    judged_similarity: float | None
    other_judged_candidates: Sequence[Fact | Decision]
    result_bounce: FactBounce | DecisionBounce
    vector: list[float]
    probe_span: list[Scope]


@dataclass(frozen=True, slots=True)
class AutoExitFrame:
    """A judged batch whose primary was re-read fresh, with the exit's clock read and sibling rows.

    ``sibling_rows`` are built for the survivors after the primary, ``unanswered`` and with no
    ``bounce_id``. The exit appends them with its own rows; ``Moved`` drops the ones it claims.
    """

    batch: JudgedBatch
    fresh: Fact | Decision
    now: datetime
    sibling_rows: list[VerdictRow]


@dataclass(frozen=True, slots=True)
class _AutoWrite:
    """What an automatic exit that writes a successor hands to ``_commit_auto_write``.

    ``rows`` are the pending rows for the acted-on candidates, written together with
    ``sibling_rows``. ``flips`` and ``invalidations`` go to ``_write_fact`` as ``extra_flips``
    and ``invalidations``. ``caller_rows`` are the caller-replay rows stamped with the write.
    """

    new_record_id: HypomnemaId
    rows: list[VerdictRow]
    sibling_rows: list[VerdictRow]
    flips: list[PredecessorFlip]
    invalidations: Sequence[DecisionInvalidation]
    caller_rows: list[VerdictRow]
    step: Literal["auto_write", "auto_supersede", "auto_invalidate"]


async def record_unanswered(request: SaveRequest, deps: SaveDeps, batch: JudgedBatch) -> None:
    rows = build_unanswered_rows(
        bounce_id=request.bounce_id,
        candidate_ids=batch.candidate_ids,
        similarities=batch.similarities,
        verdicts=batch.verdicts,
        now=deps.clock.now(),
    )
    await deps.adapter.append_verdicts(rows)


async def _redispatch_orphaned(
    request: FactSaveRequest, deps: SaveDeps, batch: JudgedBatch
) -> SaveResult:
    """Redispatch because the judged candidate moved while the Assessor ran.

    No exit acts on these verdicts, but the redispatch runs its own probe and judge cycle, so
    the batch already obtained is still recorded ``unanswered`` first.
    """
    await record_unanswered(request, deps, batch)
    return await _redispatch_after_world_moved(
        request,
        deps,
        vector=batch.vector,
        probe_span=batch.probe_span,
    )


async def _bounce_withheld(
    request: FactSaveRequest,
    deps: SaveDeps,
    batch: JudgedBatch,
    fresh: Fact | Decision,
    withheld: Withheld,
) -> SaveResult:
    """Record every survivor's verdict ``unanswered`` and return the original bounce updated.

    The update comes from ``bounce_withheld_update``, with the judged neighbours attached and,
    for ``switch_off`` when upward candidates exist, the upward exits.
    """
    await record_unanswered(request, deps, batch)
    log_admission_resolved(
        arm="fact", step="withheld", outcome="bounced",
        id=str(fresh.id), scope=request.scope, span=batch.probe_span,
    )
    other_verdicts = batch.verdicts[1:]
    update = bounce_withheld_update(withheld=withheld.reason, verdict=batch.verdict)
    update["other_candidates"] = attach_neighbor_verdicts(
        batch.result_bounce.other_candidates, batch.candidate_ids[1:], other_verdicts
    )
    if withheld.reason == "switch_off":
        # The switch-off upward bounce still lists ``moved`` first and ``confirm_distinct``
        # second, computed from raw verdicts because ``clean_picture`` withholds before
        # composing when the switch is off.
        moved_candidates = upward_promotion_candidates(
            incoming_kind="fact", typed_scope=request.scope,
            top_verdict=batch.verdict, top_candidate=fresh,
            other_verdicts=other_verdicts, other_candidates=batch.other_judged_candidates,
        )
        if moved_candidates:
            update["exits"] = build_upward_exits(
                saved_kind="fact", candidates=moved_candidates, bounce_id=request.bounce_id,
            )
    return SaveResult(
        outcome="bounced",
        bounce=batch.result_bounce.model_copy(update=update),
    )


async def refresh_moved(
    deps: SaveDeps, fresh: Fact | Decision, moved: Moved
) -> Moved | None:
    """The moved set with every member re-read, or ``None`` when one of them changed.

    Every other moved-set member is re-checked with ``candidate_still_fresh`` before the
    write: ``adapter.replace`` checks existence, kind and live state, never content, so a
    sibling edited during the Assessor call must not be retired on a verdict about its old
    text. ``fresh`` stands for the first member.
    """
    fresh_moved: list[Fact | Decision] = [fresh]
    for candidate in moved.candidates[1:]:
        fresh_member = await get_or_none(deps.adapter, candidate.id)
        if not candidate_still_fresh(judged=candidate, fresh=fresh_member):
            return None

        assert fresh_member is not None
        fresh_moved.append(fresh_member)
    return Moved(candidates=tuple(fresh_moved))


async def _commit_auto_write(
    request: FactSaveRequest,
    deps: SaveDeps,
    batch: JudgedBatch,
    write: _AutoWrite,
) -> SaveResult:
    """Append the pending rows, write the successor, stamp the rows ``acted`` and log the exit.

    The pending rows and the caller-replay rows are stamped ``failed`` if the write raises an
    ``Exception``; a cancellation stamps nothing. The caller-replay rows are stamped ``acted``
    after the exit's own rows.
    """
    await deps.adapter.append_verdicts([*write.rows, *write.sibling_rows])
    async with StampFailedOnError(deps.adapter, deps.clock.now, write.rows, write.caller_rows):
        result = await _write_fact(
            request,
            deps,
            vector=batch.vector,
            reconciles=None,
            supersession=None,
            provisional=request.provisional,
            extra_flips=write.flips,
            invalidations=write.invalidations,
            id=write.new_record_id,
        )
    await stamp_acted_many(deps.adapter, write.rows, now=deps.clock.now())
    if write.caller_rows:
        assert request.resolution.bounce_id is not None
        await stamp_acted_many_with_siblings(
            deps.adapter, write.caller_rows, bounce_id=request.resolution.bounce_id,
            exclude_candidate_ids=request.supersede_ids, now=deps.clock.now(),
        )
    log_admission_resolved(
        arm="fact", step=write.step, outcome=result.outcome,
        id=admission_result_id(result), scope=request.scope, span=batch.probe_span,
    )
    return result


async def _auto_reinforce(
    request: FactSaveRequest, deps: SaveDeps, frame: AutoExitFrame
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
        arm="fact", step="auto_reinforce", outcome="reinforced",
        id=str(fresh.id), scope=request.scope, span=batch.probe_span,
    )
    return SaveResult(
        outcome="reinforced",
        reinforced_id=fresh.id,
        reason=_no_edge_reason(request.supersession),
    )


async def _auto_write_new(
    request: FactSaveRequest, deps: SaveDeps, frame: AutoExitFrame
) -> SaveResult:
    batch = frame.batch
    admission_hit = await _admission_steps_1_3(
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
        invalidations=(),
        caller_rows=caller_rows,
        step="auto_write",
    )
    return await _commit_auto_write(request, deps, batch, write)


async def _auto_supersede(
    request: FactSaveRequest, deps: SaveDeps, frame: AutoExitFrame, supersede: Supersede
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
        invalidations=(),
        caller_rows=caller_rows,
        step="auto_supersede",
    )
    return await _commit_auto_write(request, deps, batch, write)


async def _auto_moved(
    request: FactSaveRequest, deps: SaveDeps, frame: AutoExitFrame, moved: Moved
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
        invalidations=(),
        caller_rows=caller_rows,
        step="auto_supersede",
    )
    return await _commit_auto_write(request, deps, batch, write)


async def _auto_invalidate(
    request: FactSaveRequest, deps: SaveDeps, frame: AutoExitFrame, invalidate: Invalidate
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
    invalidation = DecisionInvalidation(id=fresh.id, reason=invalidate.reason)
    row = build_pending_row(
        candidate_id=fresh.id, similarity=batch.judged_similarity, verdict=batch.verdict,
        action="invalidate", effect_ref=str(fresh.id),
        new_record_id=str(new_record_id), now=frame.now,
    )
    write = _AutoWrite(
        new_record_id=new_record_id,
        rows=[row],
        sibling_rows=frame.sibling_rows,
        flips=caller_flips,
        invalidations=[invalidation],
        caller_rows=caller_rows,
        step="auto_invalidate",
    )
    return await _commit_auto_write(request, deps, batch, write)


async def _run_auto_exit(
    request: FactSaveRequest, deps: SaveDeps, frame: AutoExitFrame, exit_: Dispatch
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
    return await _auto_invalidate(request, deps, frame, exit_)


async def _apply_verdict(
    request: FactSaveRequest,
    deps: SaveDeps,
    *,
    verdict: RelationVerdict,
    verdicts: Sequence[RelationVerdict | Unavailable],
    candidate_ids: Sequence[HypomnemaId],
    similarities: Sequence[float | None],
    judged_candidate: Fact | Decision,
    judged_similarity: float | None,
    other_judged_candidates: Sequence[Fact | Decision],
    result_bounce: FactBounce,
    vector: list[float],
    probe_span: list[Scope],
) -> SaveResult:
    """Dispatch a judged bounce through ``clean_picture``: the Fact arm's automatic-exit path.

    Re-takes ``deps.lock`` and re-reads ``judged_candidate`` before deciding, so the top candidate
    is always current; other survivors enter ``clean_picture`` at their probe-time snapshots.

    ``WriteNew`` reruns ``_admission_steps_1_3`` with ``log_failures=False`` (its caller,
    ``assessor_tail``, owns the log) because the Assessor call is outside the lock and a second
    identical save can land meanwhile.

    Verdict rows: a ``Withheld`` exit writes every survivor's verdict ``unanswered`` under
    ``bounce_id`` in this lock hold, and each judged neighbour on
    ``result_bounce.other_candidates`` gets its ``judged`` block. An automatic exit writes
    ``pending`` rows for the acted-on candidates only, in the same hold, then stamps them
    ``acted`` or ``failed``; it has no ``bounce_id``. Ids are minted before the row is
    written and edge ids before ``adapter.replace``. ``Moved`` writes one row per composed
    predecessor, all sharing the successor id.

    ``effect_ref`` is persisted: ``"<candidate id>@<now isoformat>"`` for reinforce, the edge
    id for supersede, the candidate id for invalidate, ``None`` for write_new.

    Caller replay (``request.resolution.bounce_id`` with a ``supersession``): when
    ``resolve_caller_replay_flips``
    already covers the judged candidate, its flip draws the supersession edge and the automatic
    path adds none for that id; otherwise the automatic path draws it.

    The stamping sites catch ``Exception``, not ``BaseException``, so a cancellation does not
    stamp ``failed``.
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
            incoming_kind="fact",
            typed_scope=request.scope,
            top_verdict=verdict,
            top_candidate=fresh_candidate,
            other_verdicts=verdicts[1:],
            other_candidates=other_judged_candidates,
            switches=switches,
        )

        if isinstance(exit_, Withheld):
            return await _bounce_withheld(request, deps, batch, fresh_candidate, exit_)

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


@dataclass(frozen=True, slots=True)
class CallerReinforce:
    """A caller's ``reinforce`` resolve and the verdict row it names.

    ``row`` is ``None`` for an unknown, absent or already-resolved ``bounce_id``. ``matched``
    says whether the row's recorded pick equals ``reinforce``; it is ``False`` without a row.
    """

    resolve: BounceResolution
    row: VerdictRow | None
    matched: bool


def pick_matches(
    request: SaveRequest,
    row: VerdictRow | None,
    *,
    action: AdmissionAction,
    candidate: Fact | Decision,
) -> bool:
    if row is None:
        return False

    return caller_exit_matches_pick(
        row, action=action, incoming_kind=request.kind, candidate=candidate,
        direction=scope_direction(
            incoming_scope=request.scope, candidate_scope=candidate.scope
        ),
    )


def row_group(row: VerdictRow | None) -> list[VerdictRow] | None:
    return None if row is None else [row]


async def _load_reinforce_target(
    adapter: StorageAdapter, candidate_id: HypomnemaId
) -> Fact | Decision:
    try:
        return await adapter.get_by_id(candidate_id)
    except KeyError:
        raise ToolError(
            f"resolve.action='reinforce' targets unknown/buried id={candidate_id!s}. "
            "The candidate no longer exists in the store."
        ) from None


async def begin_caller_reinforce(
    request: SaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate: Fact | Decision,
    *,
    effect_now: datetime,
    at: Callable[[], datetime],
) -> VerdictRow | None:
    """Move the caller's row ``unanswered -> pending`` once a branch is committed to reinforcing.

    These branches write a reconciliation or fresh row instead and leave the row ``unanswered``;
    stamping first would later mark that success ``failed``. They are an invalid candidate on
    both paths of both arms, a shelved Fact on the vectorless path of both arms, and a Decision
    not in force on the Decision arm's vectorless path only.
    ``effect_now`` fixes the ``effect_ref``; ``at`` is read only when a row exists, for the
    row's timestamp. Returns the pending row, or ``None`` without one.
    """
    if caller.row is None:
        return None

    effect_ref = f"{candidate.id!s}@{effect_now.isoformat()}"
    return await begin_caller_exit(
        deps.adapter, caller.row, action="reinforce", matched=caller.matched,
        effect_ref=effect_ref, new_record_id=None, incoming_text=request.content, now=at(),
    )


async def _finish_caller_exit(
    deps: SaveDeps, resolve: BounceResolution, row: VerdictRow | None
) -> None:
    if row is None:
        return

    assert resolve.bounce_id is not None
    await stamp_acted_many_with_siblings(
        deps.adapter, [row], bounce_id=resolve.bounce_id,
        exclude_candidate_ids={resolve.candidate_id}, now=deps.clock.now(),
    )


async def _write_vectorless(
    request: FactSaveRequest, deps: SaveDeps, *, reconciles: HypomnemaId | None
) -> SaveResult:
    return await _write_fact(
        request,
        deps,
        vector=None,
        reconciles=reconciles,
        supersession=request.supersession,
        embedded=False,
        on_written=deps.backfill.written,
    )


async def _vectorless_confirm_distinct(
    request: FactSaveRequest,
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
        is_live_match=lambda rec: isinstance(rec, Fact) and rec.validity == "valid",
    )
    if replay is not None:
        if isinstance(replay, Fact) and replay.lifecycle == "archived":
            await _revive_archived_fact(
                record=replay, now=now, adapter=deps.adapter, cfg=deps.cfg
            )
        return SaveResult(
            outcome="already_saved",
            hypomnema=HypomnemaRef(id=replay.id, kind="fact"),
            reason=_no_edge_reason(request.supersession),
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
    async with StampFailedOnError(deps.adapter, deps.clock.now, row_group(bounce_row)):
        result = await _write_fact(
            request,
            deps,
            vector=None,
            reconciles=None,
            supersession=request.supersession,
            embedded=False,
            on_written=deps.backfill.written,
            id=new_record_id,
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
    await _finish_caller_exit(deps, resolve, bounce_row)
    return result


async def _vectorless_reinforce_decision(
    request: FactSaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate: Decision,
    now: datetime,
) -> SaveResult:
    row = await begin_caller_reinforce(
        request, deps, caller, candidate, effect_now=now, at=lambda: now
    )
    async with StampFailedOnError(deps.adapter, deps.clock.now, row_group(row)):
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
    await _finish_caller_exit(deps, caller.resolve, row)
    return SaveResult(
        outcome="reinforced", reinforced_id=candidate.id,
        reason=_no_edge_reason(request.supersession),
    )


async def _vectorless_revive_fact(
    request: FactSaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate: Fact,
    now: datetime,
) -> SaveResult:
    row = await begin_caller_reinforce(
        request, deps, caller, candidate, effect_now=now, at=lambda: now
    )
    async with StampFailedOnError(deps.adapter, deps.clock.now, row_group(row)):
        await reinforce(
            target_id=candidate.id,
            source="revival",
            now=now,
            matched_from_id=None,
            record=candidate,
            adapter=deps.adapter,
            cfg=deps.cfg,
        )
        await deps.adapter.transition(candidate.id, "active", at=now)
    await maybe_upkeep(deps.adapter, deps.cfg)
    await _finish_caller_exit(deps, caller.resolve, row)
    return SaveResult(
        outcome="revived", reinforced_id=candidate.id,
        reason=_no_edge_reason(request.supersession),
    )


async def _vectorless_reinforce_fact(
    request: FactSaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate: Fact,
    now: datetime,
) -> SaveResult:
    row = await begin_caller_reinforce(
        request, deps, caller, candidate, effect_now=now, at=lambda: now
    )
    async with StampFailedOnError(deps.adapter, deps.clock.now, row_group(row)):
        await reinforce(
            target_id=candidate.id,
            source="dedup",
            now=now,
            matched_from_id=None,
            record=candidate,
            adapter=deps.adapter,
            cfg=deps.cfg,
        )
    suggestion = None
    if candidate.lifecycle == "ember":
        suggestion = await record_ember_recurrence(
            record=candidate,
            asserted_by=request.asserted_by,
            adapter=deps.adapter,
            cfg=deps.cfg,
        )
    await maybe_upkeep(deps.adapter, deps.cfg)
    await _finish_caller_exit(deps, caller.resolve, row)
    return SaveResult(
        outcome="reinforced",
        reinforced_id=candidate.id,
        escalation_suggestion=suggestion,
        reason=_no_edge_reason(request.supersession),
    )


async def _vectorless_reinforce(
    request: FactSaveRequest,
    deps: SaveDeps,
    resolve: BounceResolution,
    now: datetime,
) -> SaveResult:
    candidate_id = resolve.candidate_id
    # ``reinforce`` requires an existing candidate; an unknown id is a caller error.
    candidate_record = await _load_reinforce_target(deps.adapter, candidate_id)

    reinforce_bounce_row = await find_bounce_row(
        deps.adapter, bounce_id=resolve.bounce_id, candidate_id=candidate_id
    )
    caller = CallerReinforce(
        resolve=resolve,
        row=reinforce_bounce_row,
        matched=pick_matches(
            request, reinforce_bounce_row, action="reinforce", candidate=candidate_record
        ),
    )

    if isinstance(candidate_record, Decision):
        if candidate_record.validity == "invalid":
            return await _write_vectorless(request, deps, reconciles=candidate_record.id)

        return await _vectorless_reinforce_decision(request, deps, caller, candidate_record, now)

    assert isinstance(candidate_record, Fact)
    if candidate_record.validity == "invalid":
        return await _write_vectorless(request, deps, reconciles=candidate_record.id)

    if candidate_record.lifecycle == "archived":
        return await _vectorless_revive_fact(request, deps, caller, candidate_record, now)

    if candidate_record.lifecycle == "shelved":
        return await _write_vectorless(request, deps, reconciles=None)

    return await _vectorless_reinforce_fact(request, deps, caller, candidate_record, now)


async def _handle_vectorless_resolve(
    request: FactSaveRequest,
    deps: SaveDeps,
    *,
    resolve: BounceResolution,
    now: datetime,
) -> SaveResult:
    """Honor a ``BounceResolution`` while the embedder is unavailable.

    A named candidate needs no query vector, so ``reinforce`` and ``confirm_distinct`` behave
    as in ``_handle_resolve`` minus the best-hit and floor comparison. ``reinforce`` never
    writes a new row. ``confirm_distinct`` replays idempotently through
    ``find_vectorless_confirm_distinct_replay``, otherwise writes the vector-less row and
    registers the distinct pair; an unknown candidate id is not an error, since the pair is
    recorded against the id the caller named.

    A ``reinforce`` against a shelved Fact falls through to a fresh vector-less write, matching
    the live path, where the probe excludes shelved Facts and a stale resolve reaches the
    "no best hit" write.

    Caller-exit stamping mirrors ``_handle_resolve`` minus the probe. A verbatim
    ``confirm_distinct`` replay finds no ``unanswered`` row, so it stamps nothing. The
    invalid-candidate and shelved fallbacks stay unstamped, as in ``_handle_resolve``.
    """
    if resolve.action == "confirm_distinct":
        return await _vectorless_confirm_distinct(request, deps, resolve, now)

    return await _vectorless_reinforce(request, deps, resolve, now)


async def _resolve_overtaken(
    request: FactSaveRequest,
    deps: SaveDeps,
    resolve: BounceResolution,
    best_hit: Hit | None,
    probe: Probe,
    vector: list[float],
) -> SaveResult:
    """The named candidate is no longer the best match: write novel or re-dispatch the new best."""
    if best_hit is None:
        return await _write_as_requested(request, deps, vector=vector, reconciles=None)

    # Below its arm floor the new best candidate is treated as novel: a stale resolve
    # must not mutate on a weak match.
    new_arm_floor = floor_for(deps.cfg, best_hit.record)
    if probe.cosine < new_arm_floor:
        _log.debug(
            "resolve stale: new best %s below arm floor %.4f; writing novel",
            best_hit.record.id,
            new_arm_floor,
        )
        return await _write_as_requested(request, deps, vector=vector, reconciles=None)

    # Candidate overtaken: dispatch on the new best hit through the full matrix.
    _log.debug(
        "resolve: candidate %s no longer best match (now %s); re-dispatching",
        resolve.candidate_id,
        best_hit.record.id,
    )
    return await _dispatch_match(
        request,
        deps,
        best_hit=best_hit,
        probe=probe,
        vector=vector,
        roles=None,
        provisional=False,
    )


async def _resolve_confirm_distinct(
    request: FactSaveRequest,
    deps: SaveDeps,
    resolve: BounceResolution,
    best_record: Fact | Decision,
    bounce_row: VerdictRow | None,
    probe: Probe,
    vector: list[float],
) -> SaveResult:
    candidate_id = resolve.candidate_id
    matched = pick_matches(
        request, bounce_row, action="confirm_distinct", candidate=best_record
    )
    new_record_id = new_id() if bounce_row is not None else None
    if bounce_row is not None:
        bounce_row = await begin_caller_exit(
            deps.adapter, bounce_row, action="confirm_distinct", matched=matched,
            effect_ref=None,
            new_record_id=str(new_record_id) if new_record_id is not None else None,
            incoming_text=None, now=deps.clock.now(),
        )
    async with StampFailedOnError(deps.adapter, deps.clock.now, row_group(bounce_row)):
        result = await _write_fact(
            request,
            deps,
            vector=vector,
            reconciles=None,
            supersession=request.supersession,
            id=new_record_id,
        )
    if result.hypomnema is not None:
        pair = DistinctPair(
            lower_id=min(result.hypomnema.id, candidate_id, key=str),
            higher_id=max(result.hypomnema.id, candidate_id, key=str),
            source="save",
            judged_at=probe.now,
            similarity=probe.cosine,
        )
        await deps.adapter.mark_distinct(pair)
        _log.debug(
            "resolve confirm_distinct: wrote id=%s, marked pair (%s, %s)",
            result.hypomnema.id,
            pair.lower_id,
            pair.higher_id,
        )
    await _finish_caller_exit(deps, resolve, bounce_row)
    return result


async def _resolve_reinforce_decision(
    request: FactSaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate: Decision,
    probe: Probe,
    vector: list[float],
) -> SaveResult:
    # A Decision invalidated since the bounce goes to reconciliation instead of being
    # reinforced; the named row stays ``unanswered``.
    if candidate.validity == "invalid":
        _log.debug(
            "resolve reinforce: Decision id=%s is invalid; reconciling",
            candidate.id,
        )
        return await _write_as_requested(request, deps, vector=vector, reconciles=candidate.id)

    # An explicit resolve may reinforce a Decision (``source="dedup"``). The gate itself
    # never auto-reinforces Decisions.
    _log.debug(
        "resolve reinforce: Decision id=%s — emitting dedup event (§4.4)",
        candidate.id,
    )
    row = await begin_caller_reinforce(
        request, deps, caller, candidate, effect_now=probe.now, at=deps.clock.now
    )
    async with StampFailedOnError(deps.adapter, deps.clock.now, row_group(row)):
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
    await _finish_caller_exit(deps, caller.resolve, row)
    return SaveResult(
        outcome="reinforced", reinforced_id=candidate.id,
        reason=_no_edge_reason(request.supersession),
    )


async def _resolve_revive_fact(
    request: FactSaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate: Fact,
    probe: Probe,
) -> SaveResult:
    row = await begin_caller_reinforce(
        request, deps, caller, candidate, effect_now=probe.now, at=deps.clock.now
    )
    async with StampFailedOnError(deps.adapter, deps.clock.now, row_group(row)):
        await reinforce(
            target_id=candidate.id,
            source="revival",
            now=probe.now,
            matched_from_id=None,
            record=candidate,
            adapter=deps.adapter,
            cfg=deps.cfg,
        )
        await deps.adapter.transition(candidate.id, "active", at=probe.now)
    await maybe_upkeep(deps.adapter, deps.cfg)
    await _finish_caller_exit(deps, caller.resolve, row)
    return SaveResult(
        outcome="revived", reinforced_id=candidate.id,
        reason=_no_edge_reason(request.supersession),
    )


async def _resolve_reinforce_fact(
    request: FactSaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate: Fact,
    probe: Probe,
) -> SaveResult:
    """Reinforce an ember, active, hot or shelved Fact with ``source="dedup"``, lifecycle untouched.

    A shelved record reaches this branch because the candidate is loaded by id, not through
    the probe; it is reinforced, never revived.
    """
    row = await begin_caller_reinforce(
        request, deps, caller, candidate, effect_now=probe.now, at=deps.clock.now
    )
    async with StampFailedOnError(deps.adapter, deps.clock.now, row_group(row)):
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
    suggestion = None
    if candidate.lifecycle == "ember":
        # Same recurrence update as ``_dispatch_match``, so the counters agree across
        # both entry points.
        suggestion = await record_ember_recurrence(
            record=candidate,
            asserted_by=request.asserted_by,
            adapter=deps.adapter,
            cfg=deps.cfg,
        )
    await maybe_upkeep(deps.adapter, deps.cfg)
    await _finish_caller_exit(deps, caller.resolve, row)
    return SaveResult(
        outcome="reinforced",
        reinforced_id=candidate.id,
        escalation_suggestion=suggestion,
        reason=_no_edge_reason(request.supersession),
    )


async def _resolve_reinforce(
    request: FactSaveRequest,
    deps: SaveDeps,
    caller: CallerReinforce,
    candidate_record: Fact | Decision,
    probe: Probe,
    vector: list[float],
) -> SaveResult:
    if isinstance(candidate_record, Decision):
        return await _resolve_reinforce_decision(
            request, deps, caller, candidate_record, probe, vector
        )

    if isinstance(candidate_record, Fact):
        # An invalid record is never reinforced; it goes to reconciliation.
        if candidate_record.validity == "invalid":
            _log.debug(
                "resolve reinforce: candidate %s is invalid; reconciling",
                caller.resolve.candidate_id,
            )
            return await _write_as_requested(
                request, deps, vector=vector, reconciles=candidate_record.id
            )

        if candidate_record.lifecycle == "archived":
            return await _resolve_revive_fact(request, deps, caller, candidate_record, probe)

        return await _resolve_reinforce_fact(request, deps, caller, candidate_record, probe)

    raise ToolError(
        "resolve.action='reinforce' on unexpected record type for "
        f"id={caller.resolve.candidate_id!s}"
    )


async def _handle_resolve(
    request: FactSaveRequest,
    deps: SaveDeps,
    *,
    resolve: BounceResolution,
    best_hit: Hit | None,
    probe: Probe,
    vector: list[float],
) -> SaveResult:
    """Handle a bounce resolution against the gate result already computed.

    If the named candidate is still the best match, honor the action; otherwise return a fresh
    bounce. ``probe.v_hits`` and ``probe.exclude`` are passed through to the re-dispatch's
    ``_dispatch_match``. ``request.bounce_id`` is this save's own id, used for any bounce built
    here, and is distinct from ``resolve.bounce_id``, the prior bounce being resolved, whose
    verdict row is stamped through ``find_bounce_row``.

    ``request.supersession`` is passed to every ``_write_fact`` call below and read by
    ``_dispatch_match``; each branch that writes may draw the edges.
    """
    candidate_id = resolve.candidate_id

    candidate_record: Fact | Decision | None
    if resolve.action == "reinforce":
        candidate_record = await _load_reinforce_target(deps.adapter, candidate_id)
    else:
        candidate_record = await get_or_none(deps.adapter, candidate_id)

    candidate_still_best = (
        best_hit is not None
        and best_hit.record.id == candidate_id
    )

    if not candidate_still_best:
        return await _resolve_overtaken(request, deps, resolve, best_hit, probe, vector)

    # Below the arm floor the gate would have passed as novel, so the resolve is stale: write novel.
    assert best_hit is not None  # guaranteed by candidate_still_best
    arm_floor = floor_for(deps.cfg, best_hit.record)
    if probe.cosine < arm_floor:
        _log.debug(
            "resolve: candidate %s still best but below arm floor %.4f; writing novel",
            candidate_id,
            arm_floor,
        )
        return await _write_as_requested(request, deps, vector=vector, reconciles=None)

    # The verdict row this resolve names, or None for an unknown, absent or already-resolved
    # bounce_id.
    bounce_row = await find_bounce_row(
        deps.adapter, bounce_id=resolve.bounce_id, candidate_id=candidate_id
    )

    if resolve.action == "confirm_distinct":
        return await _resolve_confirm_distinct(
            request, deps, resolve, best_hit.record, bounce_row, probe, vector
        )

    assert candidate_record is not None  # guaranteed above; kept for type narrowing
    caller = CallerReinforce(
        resolve=resolve,
        row=bounce_row,
        matched=pick_matches(
            request, bounce_row, action="reinforce", candidate=best_hit.record
        ),
    )
    return await _resolve_reinforce(request, deps, caller, candidate_record, probe, vector)


async def _resolve_vectorless_save(
    request: FactSaveRequest,
    deps: SaveDeps,
    *,
    probe_span: list[Scope],
) -> tuple[SaveResult, list[Fact | Decision]]:
    """Embedder down, no ``resolve``: BM25-only probe, top five, no cosine floor.

    With no candidates the vector-less row is written as a plain ``SaveResult``. With
    candidates it returns an unjudged bounce (``reason="judge_unavailable"``,
    ``similarity=None``, each neighbour's ``bm25_rank`` set) plus the neighbour records, so the
    caller can run ``relation_verdicts`` after this function's ``async with lock:`` has exited.
    This helper never calls the Assessor: a manual release and reacquire of the shared
    ``asyncio.Lock`` is unsafe under cancellation, since a failed reacquire lets the caller's
    own ``async with`` release a lock another task now holds.
    """
    neighbors = await bm25_probe(
        content=request.content,
        span=probe_span,
        exclude=request.supersede_ids,
        adapter=deps.adapter,
    )
    if not neighbors:
        result = await _write_fact(
            request,
            deps,
            vector=None,
            reconciles=None,
            supersession=request.supersession,
            provisional=request.provisional,
            embedded=False,
            on_written=deps.backfill.written,
        )
        return result, []

    primary = neighbors[0].record
    other_candidates = [
        BounceNeighbor(candidate_id=n.record.id, similarity=None, bm25_rank=n.rank)
        for n in neighbors[1:]
    ]
    bounce = FactBounce(
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


async def _admit_replay(
    request: FactSaveRequest, deps: SaveDeps, replay: Replay, audit: AdmissionAudit
) -> SaveResult:
    existing = await deps.adapter.get_by_id(replay.id)
    if isinstance(existing, Fact) and existing.lifecycle == "archived":
        _log.debug(
            "save byte-identical replay: content matches archived "
            "Fact id=%s verbatim; reviving archived->active before "
            "ack (SPEC §4.5 [A6] amendment) rather than leaving it "
            "dormant",
            existing.id,
        )
        try:
            await _revive_archived_fact(
                record=existing, now=deps.clock.now(), adapter=deps.adapter, cfg=deps.cfg
            )
        except BaseException:
            audit.raised("replay", str(replay.id))
            raise
    else:
        _log.debug(
            "save byte-identical replay: content matches Fact "
            "id=%s verbatim; idempotent ack, no write",
            existing.id,
        )
    audit.resolved("replay", "already_saved", str(replay.id))
    return SaveResult(
        outcome="already_saved",
        hypomnema=HypomnemaRef(id=replay.id, kind="fact"),
        reason=_no_edge_reason(request.supersession),
    )


async def _admit_exact_key_same_or_downward(
    request: FactSaveRequest, deps: SaveDeps, hit: ExactHit, audit: AdmissionAudit
) -> SaveResult:
    """Revive an archived exact-key match, or reinforce it, with a ``raised`` line on failure."""
    record = hit.record
    if isinstance(record, Fact) and record.lifecycle == "archived":
        _log.debug(
            "save exact-key match (%s): reviving archived Fact "
            "id=%s (archived->active) before reinforcing",
            hit.direction,
            record.id,
        )
        try:
            await _revive_archived_fact(
                record=record,
                now=deps.clock.now(),
                adapter=deps.adapter,
                cfg=deps.cfg,
            )
        except BaseException:
            audit.raised("exact_key", str(record.id))
            raise
        audit.resolved("exact_key", "revived", str(record.id))
        return SaveResult(
            outcome="revived",
            reinforced_id=record.id,
            reason=_no_edge_reason(request.supersession),
        )

    _log.debug(
        "save exact-key match (%s): reinforcing Fact id=%s",
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
    return SaveResult(
        outcome="reinforced",
        reinforced_id=record.id,
        reason=_no_edge_reason(request.supersession),
    )


def _bounce_exact_key_upward(
    request: FactSaveRequest, hit: ExactHit, audit: AdmissionAudit
) -> SaveResult:
    record = hit.record
    _log.debug(
        "save exact-key match (upward): bouncing against Fact id=%s",
        record.id,
    )
    exits = build_upward_exits(
        saved_kind="fact",
        candidates=[record],
        bounce_id=request.bounce_id,
        exact_key=True,
    )
    assert len(exits) == 1, "the exact-key upward path offers exactly one exit"
    bounce = FactBounce(
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


async def _admission_steps_1_3(
    request: FactSaveRequest,
    deps: SaveDeps,
    *,
    probe_span: list[Scope],
    log_failures: bool = True,
) -> SaveResult | None:
    """Admission steps 1 (replay) and 3 (exact key) for the Fact arm, from either lock hold.

    The first hold calls this before the embedder runs. The second re-runs it before
    ``serial_probe`` because a punctuation-only save tokenizes to nothing, so BM25 never
    returns a concurrent twin and the probe alone would let two racing saves of the same
    content both land.

    Step 1 excludes only the resolve candidate (``request.resolution``), never the supersedes
    ids: naming a row's own byte-identical content as its predecessor is still a replay. Step 3
    excludes both, so a live resolve's own candidate reaches ``_handle_resolve`` while a
    different row's exact-key match still intercepts.

    ``log_failures`` defaults on: the log line here is the only one on a raise at the two
    lock-hold call sites. ``_apply_verdict``'s ``WriteNew`` recheck passes ``False`` because
    its caller already logs once on any failure.

    Returns the dispatched ``SaveResult`` on a hit (already_saved, reinforced, revived or an
    upward bounce), or ``None`` on a miss at both steps.
    """
    audit = AdmissionAudit(scope=request.scope, span=probe_span, log_raised=log_failures)
    resolve_exclude = request.resolution.exclude_ids
    replay_exclude = resolve_exclude
    admission_exclude = request.supersede_ids | resolve_exclude
    step1 = await check_replay(
        content=request.content,
        kind="fact",
        scope=request.scope,
        exclude=replay_exclude,
        adapter=deps.adapter,
    )
    if isinstance(step1, Replay):
        return await _admit_replay(request, deps, step1, audit)

    step3 = await check_exact_key(
        content=request.content,
        kind="fact",
        scope=request.scope,
        span=probe_span,
        exclude=admission_exclude,
        adapter=deps.adapter,
    )
    if isinstance(step3, ExactHit):
        if step3.direction in ("same", "downward"):
            return await _admit_exact_key_same_or_downward(request, deps, step3, audit)

        return _bounce_exact_key_upward(request, step3, audit)

    return None


@dataclass(frozen=True, slots=True)
class HoldOutcome:
    """What a save's second lock hold leaves for ``assessor_tail``.

    ``result`` is the hold's SaveResult. ``vector`` is the query embedding, ``None`` when the
    embedder was unavailable. ``outage_candidates`` is non-empty only when the outage path built
    an unjudged bounce. ``assessor_candidates``, ``assessor_similarities`` and
    ``assessor_similarity`` describe the survivors of a live-vector bounce, primary first.
    ``pending_admission_log`` is the admission line still owed, or ``None``.
    """

    result: SaveResult
    vector: list[float] | None
    probe_span: list[Scope]
    outage_candidates: list[Fact | Decision]
    assessor_candidates: list[Fact | Decision]
    assessor_similarities: list[float | None]
    assessor_similarity: float | None
    pending_admission_log: tuple[AdmissionStep, SaveResult] | None


class ApplyVerdict[BounceT](Protocol):
    """An arm's ``_apply_verdict`` with its request and collaborators already bound."""

    async def __call__(
        self,
        *,
        verdict: RelationVerdict,
        verdicts: Sequence[RelationVerdict | Unavailable],
        candidate_ids: Sequence[HypomnemaId],
        similarities: Sequence[float | None],
        judged_candidate: Fact | Decision,
        judged_similarity: float | None,
        other_judged_candidates: Sequence[Fact | Decision],
        result_bounce: BounceT,
        vector: list[float],
        probe_span: list[Scope],
    ) -> SaveResult: ...


async def _judge_outage_candidates(
    request: SaveRequest,
    deps: SaveDeps,
    assessor: Assessor | None,
    outcome: HoldOutcome,
    result: SaveResult,
    result_bounce: FactBounce | DecisionBounce,
) -> SaveResult:
    """Judge the unjudged outage bounce's neighbours and attach the verdicts to the bounce.

    Rows are appended under a short ``deps.lock`` hold for every verdict except ``absent``. A
    successful top verdict opens the full exit set and drops ``reason``.
    """
    outage_candidates = outcome.outage_candidates
    outage_pairs = [
        (candidate, None) for candidate in to_assessor_candidates(outage_candidates)
    ]
    outage_verdicts = await relation_verdicts(
        assessor, request.content, outage_pairs, timeout_s=deps.cfg.assessor_save_timeout_s,
    )
    # ``absent`` writes no row, so a keyless install accumulates none; ``timeout`` and
    # ``error`` are real failed attempts and still write one.
    loggable = [
        (candidate, verdict)
        for candidate, verdict in zip(outage_candidates, outage_verdicts, strict=True)
        if not (isinstance(verdict, Unavailable) and verdict.reason == "absent")
    ]
    if loggable:
        outage_rows = build_unanswered_rows(
            bounce_id=request.bounce_id,
            candidate_ids=[candidate.id for candidate, _ in loggable],
            similarities=[None] * len(loggable),
            verdicts=[verdict for _, verdict in loggable],
            now=deps.clock.now(),
        )
        async with deps.lock:
            await deps.adapter.append_verdicts(outage_rows)
    top_outage = outage_verdicts[0]
    # A sibling survivor's verdict reaches its ``other_candidates`` entry as on the
    # live-vector path; only the primary's exit set is never auto-dispatched here.
    outage_neighbors = attach_neighbor_verdicts(
        result_bounce.other_candidates,
        [candidate.id for candidate in outage_candidates[1:]],
        outage_verdicts[1:],
    )
    if isinstance(top_outage, RelationVerdict):
        exits = build_exits(
            saved_kind=request.kind, candidate=outage_candidates[0], saved_scope=request.scope,
            bounce_id=request.bounce_id,
        )
        exits_tuple = (
            (exits[0], exits[1]) if len(exits) == 2 else (exits[0], exits[1], exits[2])
        )
        return result.model_copy(
            update={
                "reason": None,
                "bounce": result_bounce.model_copy(
                    update={
                        "judged": {"relation": from_verdict(top_outage)},
                        "exits": exits_tuple,
                        "other_candidates": outage_neighbors,
                    }
                ),
            }
        )

    return result.model_copy(
        update={
            "bounce": result_bounce.model_copy(
                update={
                    "judged": {"relation": from_verdict(top_outage)},
                    "other_candidates": outage_neighbors,
                }
            ),
        }
    )


async def _ask_assessor(
    request: SaveRequest,
    deps: SaveDeps,
    assessor: Assessor,
    outcome: HoldOutcome,
) -> Sequence[RelationVerdict | Unavailable]:
    """Judge the live-vector survivors outside any lock.

    If the call raises, including by cancellation, the admission line still pending from the
    hold is logged as ``raised`` before the exception propagates.
    """
    candidate_pairs = list(
        zip(
            to_assessor_candidates(outcome.assessor_candidates),
            outcome.assessor_similarities,
            strict=True,
        )
    )
    try:
        return await relation_verdicts(
            assessor, request.content, candidate_pairs,
            timeout_s=deps.cfg.assessor_save_timeout_s,
        )
    except BaseException:
        if outcome.pending_admission_log is not None:
            step, logged_result = outcome.pending_admission_log
            log_admission_resolved(
                arm=request.kind,
                step=step,
                outcome="raised",
                id=admission_result_id(logged_result),
                scope=request.scope,
                span=outcome.probe_span,
            )
        raise


async def _apply_top_verdict[BounceT: (FactBounce, DecisionBounce)](
    request: SaveRequest,
    outcome: HoldOutcome,
    top: RelationVerdict,
    verdicts: Sequence[RelationVerdict | Unavailable],
    result_bounce: BounceT,
    apply_verdict: ApplyVerdict[BounceT],
) -> SaveResult:
    """Run the arm's automatic-exit path.

    If it raises, the step being resolved is logged as ``raised`` before the exception propagates.
    """
    vector = outcome.vector
    assert vector is not None, "the survivors path only runs with a live vector"
    assert outcome.pending_admission_log is not None
    raised_step = outcome.pending_admission_log[0]
    assessor_candidates = outcome.assessor_candidates
    try:
        return await apply_verdict(
            verdict=top,
            verdicts=verdicts,
            candidate_ids=[candidate.id for candidate in assessor_candidates],
            similarities=outcome.assessor_similarities,
            judged_candidate=assessor_candidates[0],
            judged_similarity=outcome.assessor_similarity,
            other_judged_candidates=assessor_candidates[1:],
            result_bounce=result_bounce,
            vector=vector,
            probe_span=outcome.probe_span,
        )
    except BaseException:
        log_admission_resolved(
            arm=request.kind,
            step=raised_step,
            outcome="raised",
            id=str(assessor_candidates[0].id),
            scope=request.scope,
            span=outcome.probe_span,
        )
        raise


async def _attach_unavailable_verdict(
    request: SaveRequest,
    deps: SaveDeps,
    outcome: HoldOutcome,
    verdicts: Sequence[RelationVerdict | Unavailable],
    result: SaveResult,
    result_bounce: FactBounce | DecisionBounce,
) -> SaveResult:
    """Record every survivor's verdict ``unanswered``; the top ``Unavailable`` rides the bounce."""
    assessor_candidates = outcome.assessor_candidates
    unanswered_rows = build_unanswered_rows(
        bounce_id=request.bounce_id,
        candidate_ids=[candidate.id for candidate in assessor_candidates],
        similarities=outcome.assessor_similarities,
        verdicts=verdicts,
        now=deps.clock.now(),
    )
    async with deps.lock:
        await deps.adapter.append_verdicts(unanswered_rows)
    # The top candidate's ``Unavailable`` verdict never reaches ``apply_verdict``, but
    # a sibling can have a real verdict from the same batch; attach it.
    updated_neighbors = attach_neighbor_verdicts(
        result_bounce.other_candidates,
        [candidate.id for candidate in assessor_candidates[1:]],
        verdicts[1:],
    )
    return result.model_copy(
        update={
            "bounce": result_bounce.model_copy(
                update={
                    "judged": {"relation": from_verdict(verdicts[0])},
                    "other_candidates": updated_neighbors,
                }
            ),
        }
    )


async def assessor_tail[BounceT: (FactBounce, DecisionBounce)](
    request: SaveRequest,
    deps: SaveDeps,
    *,
    assessor: Assessor | None,
    outcome: HoldOutcome,
    bounce_type: type[BounceT],
    apply_verdict: ApplyVerdict[BounceT],
) -> SaveResult:
    """Judge a bounce with the Assessor, apply the verdict and flush the admission log.

    Shared by both arms after their second lock hold. ``request.kind`` is the arm named in the
    admission log and in the exit set, ``bounce_type`` is the bounce class that arm builds, and
    ``apply_verdict`` is that arm's automatic-exit path.

    The Assessor call runs outside any lock, on a bounce only. An ``Unavailable`` top verdict
    rides on the bounce as evidence and never changes ``result.outcome``. Non-empty
    ``outcome.outage_candidates`` means the outage path built an unjudged bounce without trying
    the Assessor; it is judged here regardless of ``assessor`` (``relation_verdicts`` reports
    ``absent``), so the bounce always carries a typed ``judged["relation"]``. A successful
    verdict opens the full exit set and drops ``reason``. The live-vector survivors branch is
    gated on ``assessor is not None``.

    ``append_verdicts`` is called under a short ``deps.lock`` hold, with ``deps.clock.now()``
    read before the hold. ``apply_verdict`` takes its own hold. Two narrow ``except
    BaseException`` guards log the admission line as ``raised`` and re-raise: the first flushes
    ``outcome.pending_admission_log`` when ``relation_verdicts`` is cancelled, the second logs
    under the step being resolved when ``apply_verdict`` raises. A successful ``apply_verdict``
    clears the pending line, since it has logged its own. Any line still pending is logged
    last.
    """
    result = outcome.result
    pending_admission_log = outcome.pending_admission_log

    if result.outcome == "bounced" and result.bounce is not None:
        result_bounce = result.bounce
        if outcome.outage_candidates:
            result = await _judge_outage_candidates(
                request, deps, assessor, outcome, result, result_bounce
            )
        elif assessor is not None and outcome.assessor_candidates:
            verdicts = await _ask_assessor(request, deps, assessor, outcome)
            top = verdicts[0]
            assert isinstance(result_bounce, bounce_type)
            if isinstance(top, RelationVerdict):
                result = await _apply_top_verdict(
                    request, outcome, top, verdicts, result_bounce, apply_verdict
                )
                pending_admission_log = None
            else:
                result = await _attach_unavailable_verdict(
                    request, deps, outcome, verdicts, result, result_bounce
                )

    if pending_admission_log is not None:
        step, logged_result = pending_admission_log
        log_admission_resolved(
            arm=request.kind,
            step=step,
            outcome=logged_result.outcome,
            id=admission_result_id(logged_result),
            scope=request.scope,
            span=outcome.probe_span,
        )

    return result


@dataclass(frozen=True, slots=True)
class _Survey:
    """The second hold's live-vector probe, with what the Assessor call needs afterwards.

    ``best_hit`` is the first hit not in ``probe.exclude``, and ``arm_floor`` the floor of its
    kind (``gray_floor`` without a hit). ``assessor_candidates`` is the best hit followed by the
    other survivors above their own arm floor, and ``assessor_similarities`` their scores; the
    Assessor reads them only on a bounce.
    """

    probe: Probe
    best_hit: Hit | None
    arm_floor: float
    assessor_candidates: list[Fact | Decision]
    assessor_similarities: list[float | None]
    assessor_similarity: float | None


async def _validate_inputs(
    *,
    content: str,
    scope: str,
    purpose: str | None,
    audience_hint: list[str] | None,
    roles: list[str] | None,
    require_scope: RequireScope,
    require_purpose: RequirePurpose,
    require_roles: RequireRoles,
) -> Scope:
    """The validate stage: reject empty content, resolve the scope and run the three gates.

    Takes the entry function's own validators and inputs one for one, so it has eight
    parameters. Returns the validated scope.
    """
    require_text(content, "content must be non-empty", error=ValueError)

    typed_scope = validate_and_resolve_scope(scope)
    await require_scope(typed_scope)
    await require_purpose(purpose)
    await require_roles(audience_hint)
    await require_roles(roles)
    return typed_scope


async def _check_supersession_inputs(
    adapter: StorageAdapter,
    typed_scope: Scope,
    *,
    supersedes: list[HypomnemaId] | None,
    reason: SupersessionReason | None,
) -> None:
    """Reason and scope validation, before the embedder.

    The read span applies to every reason except ``moved``, whose upward retry names a
    predecessor outside a global save's read span, so it is checked against the wider probe
    span (``supersede_legality_span``). Reads ``list_scopes`` outside any hold.
    """
    span = read_span(typed_scope)
    registered = await adapter.list_scopes()
    probe_span = probe_span_for(typed_scope, registered)
    await check_supersedes_reason(
        predecessor_ids=supersedes or [],
        reason=reason,
        successor_scope=typed_scope,
        adapter=adapter,
        span=supersede_legality_span(reason=reason, read_span=span, probe_span=probe_span),
    )


async def _first_hold(request: FactSaveRequest, deps: SaveDeps) -> SaveResult | None:
    """Hold 1: admission steps 1-3. A hit returns from inside the hold, so it never embeds."""
    async with deps.lock:
        # Re-read under the lock: the pre-lock fetch only feeds the legality checks, and a scope
        # registered since must still be seen by the exact-key direction rule.
        registered = await deps.adapter.list_scopes()
        probe_span = probe_span_for(request.scope, registered)
        # Runs whether or not ``resolve`` is set. ``replay_exclude`` and ``admission_exclude``
        # remove the resolve's own candidate, which ``_handle_resolve`` re-gates fresh; a
        # different row's exact-key match must still be caught, or a stale ``reinforce`` target
        # could fall through to the probe and duplicate it.
        return await _admission_steps_1_3(request, deps, probe_span=probe_span)


async def embed_content(
    embedder: Embedder, content: str, backfill: BackfillSignals
) -> list[float] | None:
    """Embed outside the lock. ``None`` when the embedder is unavailable.

    Shared by both arms; ``backfill.embedded()`` runs only after a successful embed."""
    try:
        vector = await embedder.embed(content)
    except EmbedderUnavailable:
        return None

    backfill.embedded()
    return vector


async def _vectorless_hold(
    request: FactSaveRequest, deps: SaveDeps, probe_span: list[Scope]
) -> HoldOutcome:
    """Hold 2 with no vector: a caller's ``resolve`` is honored, else a BM25-only probe runs.

    Both paths can raise from inside a write, hence ``BaseException`` so a cancellation still
    gets its one admission line. The BM25 path returns neighbour records, which become
    ``outage_candidates`` for the Assessor after the lock exits.
    """
    audit = AdmissionAudit(scope=request.scope, span=probe_span)
    resolve = request.resolution.resolve
    outage_candidates: list[Fact | Decision] = []
    if resolve is not None:
        try:
            result = await _handle_vectorless_resolve(
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
            result, outage_candidates = await _resolve_vectorless_save(
                request, deps, probe_span=probe_span
            )
        except BaseException:
            audit.raised("vectorless", "-")
            raise
    landed_id = (
        result.hypomnema.id
        if result.hypomnema is not None
        else None
    )
    _log.info("vectorless_write kind=fact id=%s trigger=save", landed_id)
    # Outage outcomes share this harvest point: an explicit ``resolve`` names its own step;
    # otherwise a BM25 probe with candidates bounces and an empty one is a vectorless write.
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


async def _survey_survivors(
    request: FactSaveRequest, deps: SaveDeps, probe_span: list[Scope], vector: list[float]
) -> _Survey:
    """Probe the store and pick the best non-excluded hit.

    Probed with only the supersedes predecessors excluded; the distinct-pair exclusion needs
    the survivor pool first. ``v_hits`` is ordered by cosine descending, so the first
    non-excluded hit wins. Reads the clock once, after the probe and the distinct-pair read.
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

    # ``candidate_exclude`` joins the supersedes ids and the distinct ids; it feeds both
    # candidate selection and bounce building.
    candidate_exclude = distinct_exclude | supersede_ids

    now = deps.clock.now()

    best_hit = next((hit for hit in v_hits if hit.record.id not in candidate_exclude), None)
    best_cosine = best_hit.score if best_hit is not None else 0.0
    # Captured for the Assessor call after the lock releases: every survivor above its
    # own arm floor, primary first, as in ``_bounce_neighbors``. Used only on a bounce.
    assessor_candidates = (
        [best_hit.record, *above_floor_candidates(
            v_hits, primary_id=best_hit.record.id, exclude=candidate_exclude, cfg=deps.cfg,
        )]
        if best_hit is not None
        else []
    )
    score_by_id = {hit.record.id: hit.score for hit in v_hits}
    return _Survey(
        probe=Probe(cosine=best_cosine, v_hits=v_hits, exclude=candidate_exclude, now=now),
        best_hit=best_hit,
        arm_floor=(
            floor_for(deps.cfg, best_hit.record) if best_hit is not None else deps.cfg.gray_floor
        ),
        assessor_candidates=assessor_candidates,
        assessor_similarities=[score_by_id.get(record.id) for record in assessor_candidates],
        assessor_similarity=best_hit.score if best_hit is not None else None,
    )


async def _resolve_in_hold(
    request: FactSaveRequest,
    deps: SaveDeps,
    audit: AdmissionAudit,
    survey: _Survey,
    vector: list[float],
    resolve: BounceResolution,
) -> SaveResult:
    """``_handle_resolve`` with one admission line per save: ``_write_fact`` can raise from inside,
    and ``BaseException`` covers cancellation."""
    try:
        return await _handle_resolve(
            request,
            deps,
            resolve=resolve,
            best_hit=survey.best_hit,
            probe=survey.probe,
            vector=vector,
        )
    except BaseException:
        audit.raised("resolve", str(resolve.candidate_id))
        raise


async def _write_below_floor(
    request: FactSaveRequest,
    deps: SaveDeps,
    audit: AdmissionAudit,
    survey: _Survey,
    vector: list[float],
) -> SaveResult:
    """Write a novel Fact when the best hit is missing or under its arm floor.

    The admission line is written before the write, unlike other branches, so it fires even if
    ``_write_fact`` raises. ``excluded_only`` means the only similar candidate was a named
    ``supersedes`` predecessor; otherwise ``below_floor``. A caller supersede replay,
    ``save(supersedes=[...], bounce_id=B)``, stamps every named predecessor that has an
    unanswered row under ``B``; ids are minted before ``begin_caller_exit`` and edge ids before
    ``adapter.replace``.
    """
    _log.debug(
        "save pass (cosine=%.4f < arm_floor=%.4f)",
        survey.probe.cosine,
        survey.arm_floor,
    )
    audit.resolved("excluded_only" if request.supersede_ids else "below_floor", "saved", "-")
    novel_new_id = new_id()
    caller_flips, caller_rows = await resolve_caller_replay_flips(
        request,
        deps,
        new_record_id=novel_new_id,
        now=deps.clock.now(),
    )
    async with StampFailedOnError(deps.adapter, deps.clock.now, caller_rows):
        result = await _write_fact(
            request,
            deps,
            vector=vector,
            reconciles=None,
            supersession=None if caller_rows else request.supersession,
            provisional=request.provisional,
            extra_flips=caller_flips if caller_rows else (),
            id=novel_new_id,
        )
    if caller_rows:
        resolve_bounce_id = request.resolution.bounce_id
        assert resolve_bounce_id is not None
        await stamp_acted_many_with_siblings(
            deps.adapter, caller_rows, bounce_id=resolve_bounce_id,
            exclude_candidate_ids=request.supersede_ids, now=deps.clock.now(),
        )
    return result


async def _dispatch_in_hold(
    request: FactSaveRequest,
    deps: SaveDeps,
    audit: AdmissionAudit,
    best_hit: Hit,
    survey: _Survey,
    vector: list[float],
) -> SaveResult:
    """``_dispatch_match`` with the raise-path logging of ``_resolve_in_hold``.

    ``_dispatch_match`` can reach ``_write_fact`` (reconciliation and below-decision-floor
    branches).
    """
    try:
        return await _dispatch_match(
            request,
            deps,
            best_hit=best_hit,
            probe=survey.probe,
            vector=vector,
            roles=request.roles,
            provisional=request.provisional,
        )
    except BaseException:
        audit.raised("survivors", str(best_hit.record.id))
        raise


async def _probe_hold(
    request: FactSaveRequest,
    deps: SaveDeps,
    probe_span: list[Scope],
    vector: list[float],
) -> HoldOutcome | SaveResult:
    """Hold 2 with a vector: steps 1 and 3 again, then the probe and its dispatch.

    A concurrent identical save can land between the holds and the probe may not see it (see
    ``_admission_steps_1_3``); the exclusions keep a live resolve's own candidate out of that
    check. A hit there returns its ``SaveResult`` and skips the Assessor. Otherwise the
    ``HoldOutcome`` carries a ``pending_admission_log`` for the ``resolve`` and ``survivors``
    results, whose automatic exits may log their own line, and none for a novel write.
    """
    audit = AdmissionAudit(scope=request.scope, span=probe_span)
    admission_hit = await _admission_steps_1_3(request, deps, probe_span=probe_span)
    if admission_hit is not None:
        return admission_hit

    survey = await _survey_survivors(request, deps, probe_span, vector)
    resolve = request.resolution.resolve
    best_hit = survey.best_hit
    pending_admission_log: tuple[AdmissionStep, SaveResult] | None
    if resolve is not None:
        result = await _resolve_in_hold(request, deps, audit, survey, vector, resolve)
        pending_admission_log = ("resolve", result)
    elif best_hit is None or survey.probe.cosine < survey.arm_floor:
        result = await _write_below_floor(request, deps, audit, survey, vector)
        pending_admission_log = None
    else:
        result = await _dispatch_in_hold(request, deps, audit, best_hit, survey, vector)
        pending_admission_log = ("survivors", result)

    return HoldOutcome(
        result=result,
        vector=vector,
        probe_span=probe_span,
        outage_candidates=[],
        assessor_candidates=survey.assessor_candidates,
        assessor_similarities=survey.assessor_similarities,
        assessor_similarity=survey.assessor_similarity,
        pending_admission_log=pending_admission_log,
    )


async def _second_hold(
    request: FactSaveRequest, deps: SaveDeps, vector: list[float] | None
) -> HoldOutcome | SaveResult:
    """Hold 2: the probe through the terminal write, vector-less or with the query vector."""
    async with deps.lock:
        # Re-read again: the embed round trip sits between the two fetches.
        registered = await deps.adapter.list_scopes()
        probe_span = probe_span_for(request.scope, registered)
        if vector is None:
            return await _vectorless_hold(request, deps, probe_span)

        return await _probe_hold(request, deps, probe_span, vector)


async def save_internal(
    *,
    content: str,
    scope: str,
    source: str | None,
    purpose: str | None,
    audience_hint: list[str] | None,
    lifecycle: SaveLifecycle,
    valence: Valence | None,
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
    asserted_by: Asserter = "agent",
    roles: list[str] | None = None,
    provisional: bool = False,
    supersedes: list[HypomnemaId] | None = None,
    reason: SupersessionReason | None = None,
    note: str | None = None,
    assessor: Assessor | None = None,
    resolve_bounce_id: str | None = None,
) -> SaveResult:
    """Shared implementation of the Fact path of ``save`` and of ``ember``.

    After validating its parameters the arm builds a ``FactSaveRequest`` and a ``SaveDeps``
    and hands those to its helpers; building them validates nothing. The stages run in this
    order: ``_validate_inputs``, ``_check_supersession_inputs``, ``_first_hold``, ``embed_content``,
    ``_second_hold`` and ``assessor_tail``, which also writes the final admission line.

    Lock discipline: the arm takes ``lock`` twice. The first hold runs admission steps 1-3
    (a hit returns from inside it, never embedding) and releases on a miss. The embed call
    runs between the holds, because holding the write lock across an Ollama round trip would
    serialize every save behind the slowest embedding. The second hold covers the probe
    through the terminal write. A concurrent identical save can land in the gap, so steps 1-3
    and ``list_scopes`` are re-read inside the second hold; only ``vector`` and the validated
    ``typed_scope`` cross the gap. No lock is held across an Assessor call: a bounce is judged
    after the holds exit, and the candidate is re-read fresh (``candidate_still_fresh``) in a
    re-taken hold before any automatic exit.

    ``resolve_bounce_id``: the public ``save`` tool's ``bounce_id``, the prior bounce a
    ``supersedes`` replay names. It is distinct from ``bounce_id``, this call's own id for
    any bounce it builds. ``ember`` always leaves it ``None``.

    ``supersedes`` / ``reason`` / ``note``: only plain ``save`` passes them. Named ids are
    excluded from candidate selection and from a bounce's ``other_candidates``, so a save
    never bounces off its own predecessor. ``reason`` is required when ``supersedes`` is
    non-empty and forbidden otherwise, and the grouping is validated before the embedder
    runs. The successor row and every predecessor flip land in one ``adapter.replace``
    transaction; a failed ``replace`` raises ``ToolError`` and nothing is written.

    ``asserted_by``: only ``ember`` passes a non-default value. A plain ``save`` that matches
    an ember is an agent call, never an owner-asserted recurrence.

    ``roles``: only ``ember`` passes it. It goes through the same ``require_roles`` gate as
    ``audience_hint`` and is used only by ``_dispatch_match``'s ember-recurrence branch.

    ``provisional``: only ``save_ember`` sets it ``True``, for a new agent-authored ember
    without ``audience_hint``. It is forwarded to whichever fresh-write path runs and ignored
    by reinforce and revival branches, which mutate an existing record.

    ``vector is None``: the embedder was unavailable. ``serial_probe`` needs a query vector
    and is never run with a fake one. A caller-supplied ``resolve`` names its own target, so
    ``_handle_vectorless_resolve`` honors it. Without ``resolve``, ``bm25_probe`` runs, and an
    empty result writes through ``_write_fact`` as a vector-less row that backfill fills in
    later.

    ``backfill``: ``backfill.written()`` is called on the ``vector is None`` branch.
    ``backfill.embedded()`` is called once after a successful embed and schedules a pass only
    when the store is already dirty.

    ``pending_admission_log``: a "survivors" or "resolve" bounce that reaches the Assessor may
    still be dispatched to an automatic exit, which logs its own final line. The original
    step's line is therefore held and flushed at the bottom of ``assessor_tail``, which owns the
    two narrow ``except BaseException`` guards around the Assessor call and ``_apply_verdict``.
    """
    typed_scope = await _validate_inputs(
        content=content,
        scope=scope,
        purpose=purpose,
        audience_hint=audience_hint,
        roles=roles,
        require_scope=require_scope,
        require_purpose=require_purpose,
        require_roles=require_roles,
    )

    # One id for every bounce this call can build, whichever path builds it.
    bounce_id = str(uuid4())

    await _check_supersession_inputs(
        adapter, typed_scope, supersedes=supersedes, reason=reason
    )

    request = FactSaveRequest(
        content=content,
        scope=typed_scope,
        source=source,
        purpose=purpose,
        audience_hint=audience_hint,
        lifecycle=lifecycle,
        valence=valence,
        asserted_by=asserted_by,
        roles=roles,
        provisional=provisional,
        supersession=supersession_from(supersedes, reason, note),
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
        bounce_type=FactBounce,
        apply_verdict=partial(_apply_verdict, request, deps),
    )
