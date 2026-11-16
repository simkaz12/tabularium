"""Admission check order shared by both save arms, run before the dedup probe.

1. ``check_replay``: a byte-identical live row in the same scope is a replay. The arm
   acks ``already_saved`` and never reinforces.
2. Named ``supersedes`` / ``resolve`` predecessors are passed as ``exclude`` to every later
   match.
3. ``check_exact_key``: the canonical content key matches a row in the probe span.
   Same-scope and downward matches reinforce silently; an upward match (a global save
   restating a project row) always bounces.
4. ``serial_probe``: vector top-``probe_k``, with BM25 narrowing inside it. Each survivor at
   or above the arm's own floor goes to the Assessor, one concurrent ``relation`` call each,
   verdicts in probe order. The top survivor's verdict drives dispatch through
   ``legal_exit``. ``bm25_probe`` is the embedder-down variant: BM25 only, top five, no
   cosine floor.

Steps 1-3 run inside the arm's first write-lock hold, before the embedder; a hit returns from
inside that hold without embedding. A miss releases the lock, the arm embeds, re-takes the
lock and runs the probe. No lock is held across an Assessor call: the arm releases the second
hold before ``relation_verdicts`` and re-reads the candidate afterwards
(``candidate_still_fresh``).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Literal, TypedDict

from tabularium.config import EngineConfig
from tabularium.domain import (
    Decision,
    Fact,
    HypomnemaId,
    InvalidationReason,
    Kind,
    Scope,
    SupersessionReason,
)
from tabularium.engine.assessor import (
    Assessor,
    AssessorCandidate,
    JevError,
    Method,
    NoReasonClass,
    Relation,
    RelationVerdict,
    Unavailable,
    Verdict,
)
from tabularium.engine.audit_log import log_assessor_call
from tabularium.engine.filters import probe_filters_for
from tabularium.results import (
    BounceExit,
    BounceNeighbor,
    JudgedUnavailable,
    JudgedVerdict,
    WithheldReason,
)
from tabularium.storage.content_key import content_key
from tabularium.storage.types import Hit, StorageAdapter

_GLOBAL = Scope("global")

#: Fixed width of the embedder-down probe; not a config knob.
_BM25_OUTAGE_TOP_N = 5

Direction = Literal["same", "downward", "upward"]


def from_verdict(v: Verdict | Unavailable) -> JudgedVerdict | JudgedUnavailable:
    """Project an engine-side ``Verdict`` or ``Unavailable`` onto its wire shape, sans ``trace``."""
    if isinstance(v, Unavailable):
        return JudgedUnavailable(reason=v.reason, detail=v.detail)
    reason_class = v.reason_class if isinstance(v, RelationVerdict) else None
    return JudgedVerdict(
        pick=v.pick, odds=v.odds, confidence=v.confidence, reason_class=reason_class
    )

# Population for steps 1 and 3: same kind, valid, and these lifecycles. Embers are excluded
# (recurrence has its own path in ``_dispatch_match``). ``shelved`` is excluded because
# shelving withdraws the gate's defence of a Fact, so re-saving its content must succeed.
_FACT_ADMISSION_LIFECYCLE: frozenset[str] = frozenset({"active", "hot", "archived"})
_DECISION_ADMISSION_LIFECYCLE: frozenset[str] = frozenset({"in_force"})


@dataclass(frozen=True, slots=True)
class Replay:
    """Step 1 hit: a byte-identical live row already exists."""

    id: HypomnemaId


@dataclass(frozen=True, slots=True)
class ExactHit:
    """Step 3 hit: the canonical content key matched a live row in the span."""

    record: Fact | Decision
    direction: Direction


@dataclass(frozen=True, slots=True)
class Miss:
    """Neither step matched — continue to the embedder and the probe."""


def _kind_matches(record: Fact | Decision, kind: Kind) -> bool:
    return record.type == kind


def _in_population(record: Fact | Decision, *, kind: Kind) -> bool:
    """The admission population shared by steps 1 and 3."""
    if record.validity != "valid" or not _kind_matches(record, kind):
        return False
    allowed = (
        _FACT_ADMISSION_LIFECYCLE if kind == "fact" else _DECISION_ADMISSION_LIFECYCLE
    )
    return record.lifecycle in allowed


async def check_replay(
    *,
    content: str,
    kind: Kind,
    scope: Scope,
    exclude: AbstractSet[HypomnemaId],
    adapter: StorageAdapter,
) -> Replay | Miss:
    """Step 1: a byte-identical live row in ``scope`` alone, never the wider probe span.

    With several matches the oldest wins, ordered by ``(created_at, str(id))``.
    """
    key = content_key(content)
    candidates = await adapter.find_by_content_key(key, [scope])
    live = [
        c
        for c in candidates
        if c.id not in exclude and c.content == content and _in_population(c, kind=kind)
    ]
    if not live:
        return Miss()
    winner = min(live, key=lambda c: (c.created_at, str(c.id)))
    return Replay(id=winner.id)


def scope_direction(*, incoming_scope: Scope, candidate_scope: Scope) -> Direction:
    if candidate_scope == incoming_scope:
        return "same"
    if incoming_scope == _GLOBAL:
        return "upward"
    return "downward"


async def check_exact_key(
    *,
    content: str,
    kind: Kind,
    scope: Scope,
    span: list[Scope],
    exclude: AbstractSet[HypomnemaId],
    adapter: StorageAdapter,
) -> ExactHit | Miss:
    """Step 3: the canonical content key matches a live row in the probe span.

    The winner is the same-scope row, then ``global``, then the oldest by
    ``(created_at, str(id))`` within that tier. ``direction`` comes from the winner's scope
    against ``scope``.
    """
    key = content_key(content)
    candidates = await adapter.find_by_content_key(key, span)
    live = [c for c in candidates if c.id not in exclude and _in_population(c, kind=kind)]
    if not live:
        return Miss()

    def _tier(record: Fact | Decision) -> int:
        if record.scope == scope:
            return 0
        if record.scope == _GLOBAL:
            return 1
        return 2

    winner = min(live, key=lambda c: (_tier(c), c.created_at, str(c.id)))
    return ExactHit(
        record=winner,
        direction=scope_direction(incoming_scope=scope, candidate_scope=winner.scope),
    )


@dataclass(frozen=True, slots=True)
class Survivors:
    """The serial probe's output: vector-leg hits that also appear in the BM25 leg.

    ``exclude`` is already dropped and hits are ordered by cosine descending. No gray floor
    is applied; each arm applies the floor for its own candidate kind.
    """

    hits: list[Hit]


@dataclass(frozen=True, slots=True)
class BM25Neighbor:
    """One embedder-down probe hit; BM25 rank stands in for the cosine that does not exist."""

    record: Fact | Decision
    rank: int


async def serial_probe(
    *,
    content: str,
    vector: list[float],
    span: list[Scope],
    k: int,
    exclude: AbstractSet[HypomnemaId],
    adapter: StorageAdapter,
) -> Survivors:
    """Intersect the vector top-``k`` with the BM25 top-``k`` by id, without re-ranking by BM25.

    A BM25-only hit is lost and a vector-only hit is not a survivor. Survivors keep the
    vector-leg cosine and are ordered by it descending, so the best candidate is ``hits[0]``.
    """
    filters = probe_filters_for(span)
    v_hits = await adapter.vector_search(vector, filters=filters, k=k)
    t_hits = await adapter.text_search(content, filters=filters, k=k)
    text_ids = {hit.record.id for hit in t_hits}
    survivors = [
        hit for hit in v_hits if hit.record.id in text_ids and hit.record.id not in exclude
    ]
    survivors.sort(key=lambda hit: -hit.score)
    return Survivors(hits=survivors)


async def bm25_probe(
    *,
    content: str,
    span: list[Scope],
    exclude: AbstractSet[HypomnemaId],
    adapter: StorageAdapter,
) -> list[BM25Neighbor]:
    """BM25-only probe for when the embedder is down: top five, no cosine floor.

    Fetches ``len(exclude)`` extra hits so dropping ``exclude`` still leaves five. Rank is
    1-indexed over the post-exclude survivors.
    """
    filters = probe_filters_for(span)
    hits = await adapter.text_search(content, filters=filters, k=_BM25_OUTAGE_TOP_N + len(exclude))
    survivors = [hit for hit in hits if hit.record.id not in exclude][:_BM25_OUTAGE_TOP_N]
    return [BM25Neighbor(record=hit.record, rank=i + 1) for i, hit in enumerate(survivors)]


def to_assessor_candidates(records: Sequence[Fact | Decision]) -> list[AssessorCandidate]:
    """Project domain records to the Assessor's own candidate shape."""
    return [
        AssessorCandidate(
            id=str(record.id),
            content=record.content,
            kind=record.type,
            scope=str(record.scope),
            created_at=record.created_at,
        )
        for record in records
    ]


async def ask[T](
    coro_factory: Callable[[], Awaitable[T]] | None,
    *,
    method: Method,
    timeout_s: float,
) -> T | Unavailable:
    """Run one Assessor call under ``timeout_s``, mapping every failure to ``Unavailable``.

    Logs one ``assessor_call`` DEBUG line per call. ``coro_factory`` is ``None`` when no
    Assessor is configured, giving reason ``absent`` without a call. ``NotImplementedError``
    re-raises. For other exceptions ``detail`` is the first 200 characters of
    ``JevError.message``, the fixed string ``"no_reason_class"`` for ``NoReasonClass``, and
    the exception class name otherwise.
    """
    if coro_factory is None:
        log_assessor_call(method=method, outcome="unavailable", reason="absent", latency_ms=None)
        return Unavailable(method=method, reason="absent", detail=None, latency_ms=None)

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(coro_factory(), timeout=timeout_s)
    except NotImplementedError:
        raise
    except TimeoutError:
        latency_ms = int((time.monotonic() - start) * 1000)
        log_assessor_call(
            method=method, outcome="unavailable", reason="timeout", latency_ms=latency_ms
        )
        return Unavailable(method=method, reason="timeout", detail=None, latency_ms=latency_ms)
    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        if isinstance(exc, JevError):
            detail = exc.message[:200]
        elif isinstance(exc, NoReasonClass):
            detail = "no_reason_class"
        else:
            detail = type(exc).__name__
        log_assessor_call(
            method=method, outcome="unavailable", reason="error", latency_ms=latency_ms
        )
        return Unavailable(method=method, reason="error", detail=detail, latency_ms=latency_ms)

    latency_ms = int((time.monotonic() - start) * 1000)
    log_assessor_call(method=method, outcome="verdict", reason=None, latency_ms=latency_ms)
    return result


async def relation_verdicts(
    assessor: Assessor | None,
    incoming: str,
    survivors: Sequence[tuple[AssessorCandidate, float | None]],
    *,
    timeout_s: float,
    neighbours: Sequence[AssessorCandidate] | None = None,
) -> list[RelationVerdict | Unavailable]:
    """One concurrent ``relation`` call per survivor, verdicts in probe order.

    ``survivors`` pairs each candidate with its probe similarity, ``None`` on the
    embedder-down path. ``neighbours`` is passed on each call minus the survivor being judged;
    with none left the keyword is omitted.
    """

    def _factory(
        candidate: AssessorCandidate, similarity: float | None
    ) -> Callable[[], Awaitable[RelationVerdict]]:
        async def _call() -> RelationVerdict:
            assert assessor is not None
            others = [n for n in neighbours or () if n.id != candidate.id]
            if not others:
                return await assessor.relation(incoming, candidate, similarity=similarity)
            return await assessor.relation(
                incoming, candidate, similarity=similarity, neighbours=others
            )

        return _call

    calls = [
        ask(
            _factory(candidate, similarity) if assessor is not None else None,
            method="relation",
            timeout_s=timeout_s,
        )
        for candidate, similarity in survivors
    ]
    return list(await asyncio.gather(*calls))


#: Maximum concurrent Cleaner Assessor calls, a Jev concurrency ceiling rather than a tunable.
_CLEANER_FANOUT_LIMIT = 8


async def bounded_ask[T](
    factories: Sequence[Callable[[], Awaitable[T]]],
    *,
    method: Method,
    timeout_s: float,
) -> list[T | Unavailable]:
    """Run every factory through ``ask``, at most ``_CLEANER_FANOUT_LIMIT`` at a time.

    The semaphore is held for a call's full duration, including its timeout.
    """
    semaphore = asyncio.Semaphore(_CLEANER_FANOUT_LIMIT)

    async def _bounded(factory: Callable[[], Awaitable[T]]) -> T | Unavailable:
        async with semaphore:
            return await ask(factory, method=method, timeout_s=timeout_s)

    return list(await asyncio.gather(*(_bounded(factory) for factory in factories)))


@dataclass(frozen=True, slots=True)
class AutoExitSwitches:
    """Per-relation automatic-exit switches: a typed view of ``EngineConfig``'s four bools."""

    restatement: bool
    replacement: bool
    follow_up: bool
    distinct: bool

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> AutoExitSwitches:
        """Read the four ``auto_exit_*`` switches off ``cfg`` as it stands now."""
        return cls(
            restatement=cfg.auto_exit_restatement,
            replacement=cfg.auto_exit_replacement,
            follow_up=cfg.auto_exit_follow_up,
            distinct=cfg.auto_exit_distinct,
        )

    def for_relation(self, relation: Relation) -> bool:
        if relation == "restatement":
            return self.restatement
        if relation == "replacement":
            return self.replacement
        if relation == "follow_up":
            return self.follow_up
        return self.distinct


@dataclass(frozen=True, slots=True)
class Reinforce:
    """Legal automatic exit: reinforce the candidate in place, ``source="dedup"``."""


@dataclass(frozen=True, slots=True)
class WriteNew:
    """Legal automatic exit: write the incoming content as a new row, registering no pair."""


@dataclass(frozen=True, slots=True)
class Supersede:
    """Legal automatic exit: supersede the candidate with the incoming content."""

    reason: SupersessionReason


@dataclass(frozen=True, slots=True)
class Invalidate:
    """Legal automatic exit: a Fact successor invalidates the Decision candidate."""

    reason: InvalidationReason


AutoExit = Reinforce | WriteNew | Supersede | Invalidate


@dataclass(frozen=True, slots=True)
class Promote:
    """Legal per-candidate exit: a same-kind restatement of an incoming ``global`` save, direction
    upward, with its relation's switch on.

    Never dispatched alone. ``clean_picture`` composes every survivor's ``Promote`` into one
    ``Moved`` set.
    """


@dataclass(frozen=True, slots=True)
class Moved:
    """The composed upward-promotion dispatch: every legally promotable survivor, in rank order.

    ``build_upward_exits`` names them in the same order. On the Decision arm the set never
    exceeds one candidate, because ``save(supersedes=...)`` takes one id there and
    ``unsupersede`` deletes the successor; ``clean_picture`` withholds instead. The Fact arm
    has no cap.
    """

    candidates: tuple[Fact | Decision, ...]


Dispatch = AutoExit | Moved


@dataclass(frozen=True, slots=True)
class Withheld:
    """No automatic exit taken. The verdict rides as evidence only and ``reason`` is a
    ``WithheldReason`` token.
    """

    reason: WithheldReason


def candidate_still_fresh(*, judged: Fact | Decision, fresh: Fact | Decision | None) -> bool:
    """Whether ``fresh``, a re-read of ``judged.id`` after the Assessor call, still matches what the
    Assessor was shown.

    ``False`` when ``fresh`` is gone, its canonical content key changed, or it left its kind's
    admission population since the judged snapshot. A candidate already outside the
    population at judge time is not a transition; ``legal_exit`` reports it as
    ``candidate_retired``.
    """
    if fresh is None:
        return False
    if content_key(fresh.content) != content_key(judged.content):
        return False
    if _in_population(judged, kind=judged.type) and not _in_population(fresh, kind=fresh.type):
        return False
    return True


class BounceUpdate(TypedDict, total=False):
    """The keys a ``model_copy(update=...)`` payload on a bounce may carry.

    ``bounce_withheld_update`` always sets ``judged`` and ``withheld``. The save arms add
    ``other_candidates`` and ``exits`` to the returned dict before applying it.
    """

    judged: dict[str, JudgedVerdict | JudgedUnavailable] | None
    withheld: WithheldReason
    other_candidates: list[BounceNeighbor]
    exits: tuple[BounceExit, ...]


def bounce_withheld_update(
    withheld: WithheldReason, *, verdict: RelationVerdict | None = None
) -> BounceUpdate:
    """The ``model_copy(update=...)`` payload attaching ``withheld`` to a bounce, plus
    ``judged["relation"]`` when a verdict was judged against the candidate the bounce names.
    """
    judged = {"relation": from_verdict(verdict)} if verdict is not None else None
    return {"judged": judged, "withheld": withheld}


def legal_exit(
    *,
    relation: Relation,
    reason_class: SupersessionReason | None,
    incoming_kind: Kind,
    candidate: Fact | Decision,
    direction: Direction,
    switches: AutoExitSwitches,
) -> AutoExit | Promote | Withheld:
    """Intersect a relation verdict with the legal-exit matrix.

    Check order: retired candidate, switch, kind pair, direction, reason class. A retired
    candidate with its switch off reports ``candidate_retired``; a Decision save against a
    Fact reports ``decision_against_fact`` before any scope or reason-class question.

    ``candidate`` must be a fresh read, since its lifecycle and validity decide the retired
    check. ``clean_picture`` also calls this for non-top survivors with their probe-time
    snapshots.

    A ``replacement`` needs a ``reason_class`` other than ``moved``. Same-kind gives
    ``Supersede``. An incoming Fact against a Decision candidate gives ``Withheld`` for
    ``elaborated`` and ``Invalidate`` for ``correction`` or ``changed``. ``follow_up`` and
    ``distinct`` ask no direction or reason-class question and give ``WriteNew``.

    A same-kind ``restatement`` at ``direction="upward"`` returns ``Promote``, which
    ``clean_picture`` composes into a ``Moved`` set.
    """
    candidate_kind: Kind = candidate.type

    if not _in_population(candidate, kind=candidate_kind):
        return Withheld(reason="candidate_retired")

    if not switches.for_relation(relation):
        return Withheld(reason="switch_off")

    if relation == "restatement":
        if incoming_kind != candidate_kind:
            return Withheld(reason="cross_kind_restatement")
        if direction == "upward":
            return Promote()
        return Reinforce()

    if relation == "replacement":
        if incoming_kind == "decision" and candidate_kind == "fact":
            return Withheld(reason="decision_against_fact")
        if direction != "same":
            return Withheld(reason="cross_scope_replacement")
        if reason_class is None or reason_class == "moved":
            return Withheld(reason="no_reason_class")
        if incoming_kind == candidate_kind:
            return Supersede(reason=reason_class)
        if reason_class == "elaborated":
            return Withheld(reason="elaborated_against_decision")
        if reason_class == "correction":
            return Invalidate(reason="correction")
        return Invalidate(reason="changed")

    if incoming_kind == "decision" and candidate_kind == "fact":
        return Withheld(reason="decision_against_fact")
    return WriteNew()



def clean_picture(
    *,
    incoming_kind: Kind,
    typed_scope: Scope,
    top_verdict: RelationVerdict,
    top_candidate: Fact | Decision,
    other_verdicts: Sequence[RelationVerdict | Unavailable],
    other_candidates: Sequence[Fact | Decision],
    switches: AutoExitSwitches,
) -> Dispatch | Withheld:
    """Decide the dispatch for the top survivor given every other survivor's verdict.

    ``legal_exit`` on the top survivor decides first. ``Withheld`` propagates. A single
    ``AutoExit`` dispatches unless another survivor's verdict is ``restatement`` or
    ``replacement``, which withholds ``competing_candidate``; ``Unavailable``, ``distinct``
    and ``follow_up`` never block. ``Promote`` gathers every other survivor whose own
    ``legal_exit`` is also ``Promote`` into one ``Moved`` set in rank order. Any other
    survivor still judged ``restatement`` or ``replacement`` withholds
    ``competing_candidate``, except a retired one, which ``legal_exit`` already reports as
    ``candidate_retired``.

    On the Decision arm a ``Moved`` set of two or more candidates withholds
    ``competing_candidate``, because a shared successor could not be undone. The Fact arm has
    no cap.

    ``top_candidate`` must be the fresh read taken under the lock after
    ``candidate_still_fresh``. ``other_candidates`` and ``other_verdicts`` are the probe-time
    snapshots, positionally paired and never re-read. ``adapter.replace`` re-checks every
    composed predecessor under the lock, so a stale one surfaces as a write failure handled
    by the arm.
    """
    top_direction = scope_direction(incoming_scope=typed_scope, candidate_scope=top_candidate.scope)
    top_exit = legal_exit(
        relation=top_verdict.pick,
        reason_class=top_verdict.reason_class,
        incoming_kind=incoming_kind,
        candidate=top_candidate,
        direction=top_direction,
        switches=switches,
    )

    if isinstance(top_exit, Withheld):
        return top_exit

    if not isinstance(top_exit, Promote):
        for other_verdict in other_verdicts:
            if (
                isinstance(other_verdict, RelationVerdict)
                and other_verdict.pick in ("restatement", "replacement")
            ):
                return Withheld(reason="competing_candidate")
        return top_exit

    moved: list[Fact | Decision] = [top_candidate]
    for other_verdict, other_candidate in zip(other_verdicts, other_candidates, strict=True):
        if not isinstance(other_verdict, RelationVerdict):
            continue
        other_direction = scope_direction(
            incoming_scope=typed_scope, candidate_scope=other_candidate.scope
        )
        other_exit = legal_exit(
            relation=other_verdict.pick,
            reason_class=other_verdict.reason_class,
            incoming_kind=incoming_kind,
            candidate=other_candidate,
            direction=other_direction,
            switches=switches,
        )
        if isinstance(other_exit, Promote):
            moved.append(other_candidate)
        elif other_verdict.pick in ("restatement", "replacement"):
            return Withheld(reason="competing_candidate")
    if incoming_kind == "decision" and len(moved) > 1:
        return Withheld(reason="competing_candidate")
    return Moved(candidates=tuple(moved))


def upward_promotion_candidates(
    *,
    incoming_kind: Kind,
    typed_scope: Scope,
    top_verdict: RelationVerdict | Unavailable,
    top_candidate: Fact | Decision,
    other_verdicts: Sequence[RelationVerdict | Unavailable],
    other_candidates: Sequence[Fact | Decision],
) -> list[Fact | Decision]:
    """Survivors that would compose a ``Moved`` dispatch if every switch were on.

    Uses raw verdict picks, ignoring switch state. It builds the recommended exit of a
    switch-off upward bounce and is never a dispatch decision. Rank order, top first when it
    qualifies.
    """

    def _is_upward_promotable(
        verdict: RelationVerdict | Unavailable, candidate: Fact | Decision
    ) -> bool:
        if not isinstance(verdict, RelationVerdict) or verdict.pick != "restatement":
            return False
        if candidate.type != incoming_kind:
            return False
        return (
            scope_direction(incoming_scope=typed_scope, candidate_scope=candidate.scope) == "upward"
        )

    members = [top_candidate] if _is_upward_promotable(top_verdict, top_candidate) else []
    members.extend(
        candidate
        for verdict, candidate in zip(other_verdicts, other_candidates, strict=True)
        if _is_upward_promotable(verdict, candidate)
    )
    return members


def attach_neighbor_verdicts(
    neighbors: Sequence[BounceNeighbor],
    candidate_ids: Sequence[HypomnemaId],
    verdicts: Sequence[RelationVerdict | Unavailable],
) -> list[BounceNeighbor]:
    """Attach each judged neighbour's ``relation`` odds, keyed by candidate id.

    ``other_candidates`` is built by ``facts._bounce_neighbors`` before any Assessor call.
    Neighbours the Assessor never saw keep ``judged=None``.
    """
    by_id = dict(zip(candidate_ids, verdicts, strict=True))
    return [
        neighbor.model_copy(
            update={"judged": {"relation": from_verdict(by_id[neighbor.candidate_id])}}
        )
        if neighbor.candidate_id in by_id
        else neighbor
        for neighbor in neighbors
    ]
