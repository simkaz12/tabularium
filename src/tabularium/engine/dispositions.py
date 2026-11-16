"""The disposition family: ``set_validity``, ``bury``, ``revoke``, ``set_hot``, ``set_audience``,
``set_shelved``, ``supersede`` and ``unsupersede``, plus the shared supersession validation.

Arms: ``set_validity`` and ``bury`` apply to Facts and Decisions; ``revoke`` is Decision-only;
``set_hot``, ``set_audience``, ``set_shelved`` and ``supersede`` are Fact-only; ``unsupersede``
dispatches on the predecessor's own kind. After cheap argument checks each runs its
read-check-write body under the engine's single write lock, with two exceptions in where the
lock starts: ``supersede`` runs ``validate_supersede`` before the lock and only
``supersede_write_locked`` under it, and ``unsupersede`` checks that the id exists before taking
the lock and does the rest under it.

Decision-arm supersession is ``save(kind="decision", supersedes=...)`` in ``decisions.py``: it
replaces the predecessor with a new record carrying frozen lineage, so it is not an edge between
two stored records. ``supersede`` here draws a forward pointer between two existing Facts.
``save(kind="fact", supersedes=[...])`` shares ``check_supersedes_reason`` with it but writes the
new successor and the flips itself through ``adapter.replace``.

``unsupersede`` repairs either arm's forward edge. Fact arm: it clears ``superseded_by`` and
restores ``validity="valid"``, both rows staying live. Decision arm: the successor must be
``in_force`` and must not be the successor of any other predecessor, since ``adapter.restore``
deletes it. Write order under the lock: the undo event first (keyed by the predecessor's
``supersession_edge_id``, so a retry after a crash finds it), then on the Decision arm the
successor's tombstone carrying ``restores``, then ``adapter.restore``.

Re-ops:

- same-value ``set_validity`` returns ``outcome="unchanged"`` after one ``get_by_id`` read, with
  no record write and no override stamp;
- re-``bury`` returns ``outcome="already_buried"`` with the first call's ``tombstone_id``;
- re-``revoke`` returns ``outcome="unchanged"`` without moving ``updated_at``;
- ``set_hot`` and ``set_audience`` return the unchanged record with no write when the requested
  state already holds (see each docstring);
- ``unsupersede`` is not a re-op: a second call on the same edge is refused because
  ``superseded_by`` is already ``None``. Only a retry after a crash is safe.

``bury`` tells "already buried" from "never existed" through ``adapter.get_tombstone(id)``, which
never raises, rather than parsing the adapters' ``KeyError`` messages.

Hot tier
~~~~~~~~
``hot`` is always-on: a hot record has no decay cutoff and is eligible for every session-start
payload, subject to ``context``'s scope, validity and role routing and to token packing. Promotion
is therefore an owner action and the engine never fills the tier on its own. ``set_hot`` is the
only promotion path (``active`` or ``ember`` to ``hot``) and the only deliberate demotion path
(``hot`` to ``active``, never back to ``ember``). A hot Fact that is buried or invalidated leaves
the tier through ``select_default``'s filters without any change to its ``lifecycle``.

``hot_cap`` is a payload-size guardrail counted per ``(scope, role)``. Promoting past a role's cap
raises ``ToolError`` naming the role and every Fact currently hot for it in that scope; nothing is
ever evicted silently. Occupants are the hot Facts in the same scope that
``engine.context.routes_to`` sends to the role, the predicate the push tiers route with. The scan
does not filter on validity, so an invalid hot Fact counts toward the cap although
``select_default`` skips it. ``cfg.hot_cap_global`` governs the ``global`` scope and
``cfg.hot_cap_scope`` every other scope.

``set_audience`` sets ``audience_hint`` and clears ``provisional`` without touching ``lifecycle``.
It refuses a hot Fact so that ``set_hot``, which re-checks the cap, stays the only writer of a hot
Fact's audience.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NoReturn
from uuid import uuid4

import pydantic

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import (
    Decision,
    Fact,
    HypomnemaId,
    Kind,
    Scope,
    SupersessionReason,
    UnsupersedeShape,
    Validity,
)
from tabularium.engine._errors import ToolError, require_text
from tabularium.engine._types import RequireRoles
from tabularium.engine.audit_log import (
    log_failed_supersession,
    log_unsupersede_completed,
    log_unsupersede_failed,
)
from tabularium.engine.context import routes_to
from tabularium.engine.filters import GLOBAL, PUSH_HOT_LIFECYCLE
from tabularium.engine.lookups import get_fact, get_record, unknown_id
from tabularium.engine.verdicts import stamp_override
from tabularium.results import BuryResult, ReviseResult, SupersedeResult, UnsupersedeResult
from tabularium.storage.content_key import content_hash
from tabularium.storage.types import (
    PredecessorFlip,
    ReinforcementEvent,
    StorageAdapter,
    UnsupersedeEvent,
)

_log = logging.getLogger("tabularium")


async def set_validity(
    *,
    id: HypomnemaId,
    valid: bool,
    reason: str,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
) -> ReviseResult:
    """Flip the validity bit on a Fact or Decision.

    Never calls ``transition`` and never touches ``updated_at``: validity is metadata about frozen
    content, and ``updated_at`` is the time of the last lifecycle transition.

    ``clock`` supplies two timestamps: the ``RevalidationEvent`` written by ``valid=True`` on a
    Decision that a Fact invalidated, and the ``stamp_override`` call after a revision (on both the
    revalidate path and the ordinary path). The revalidate call goes through ``adapter.revalidate``
    first, which returns ``None`` and writes nothing when the Decision carries no Fact-drawn
    invalidation, and the ordinary path then applies.

    Returns:
        ``ReviseResult`` with ``outcome="unchanged"`` (already at that validity, no record write)
        or ``"revised"``.

    Raises:
        ToolError: unknown or buried id (also when a race buries the record mid-call); empty
                   ``reason``; ``valid=True`` on a Fact whose ``superseded_by`` is set. That check
                   runs before the adapter is called and holds on both adapters; use
                   ``unsupersede`` to clear the pointer.
    """
    require_text(reason, "set_validity: reason is required and must be non-empty (SPEC §9.1)")

    async with lock:
        record = await get_record(adapter, id)

        target_validity: Validity = "valid" if valid else "invalid"
        if record.validity == target_validity:
            # Not a stamp_override site: a record already at the requested validity may have
            # reached it by an unrelated cause, and stamping would inflate the override count.
            return ReviseResult(
                id=id,
                outcome="unchanged",
                lifecycle=record.lifecycle,
                validity=record.validity,
                reason=reason,
            )

        if valid and isinstance(record, Fact) and record.superseded_by is not None:
            raise ToolError(
                f"set_validity: Fact id={id!s} is superseded (by "
                f"{record.superseded_by!s}) and cannot be re-validated via "
                "set_validity(valid=True) — the supersession pointer and "
                "the validity flip were set together in one transition; "
                "use unsupersede(id, "
                "reason=...) instead, which clears the pointer and restores "
                "validity together in one call"
            )

        if valid and isinstance(record, Decision):
            now = clock.now()
            event = await adapter.revalidate(id, reason, at=now)
            if event is not None:
                updated = await adapter.get_by_id(id)
                await stamp_override(adapter, ref=str(id), tool="set_validity", now=now)
                return ReviseResult(
                    id=id,
                    outcome="revised",
                    lifecycle=updated.lifecycle,
                    validity=updated.validity,
                    reason=reason,
                )

        try:
            await adapter.set_validity(id, valid, reason)
        except KeyError:
            # Buried between the existence check and the write: report it as unknown.
            raise unknown_id(id) from None
        updated = await get_record(adapter, id)
        await stamp_override(adapter, ref=str(id), tool="set_validity", now=clock.now())
        return ReviseResult(
            id=id,
            outcome="revised",
            lifecycle=updated.lifecycle,
            validity=updated.validity,
            reason=reason,
        )


async def bury(
    *,
    id: HypomnemaId,
    reason: str,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
) -> BuryResult:
    """Bury a Fact or Decision: drop content, vector and text-index row, write a tombstone.

    Irreversible. The reversible alternatives are ``set_validity(valid=False)`` and, for Facts,
    archival. A re-bury returns ``outcome="already_buried"`` with the first call's
    ``tombstone_id`` and writes no second tombstone. ``clock`` is unused: the adapter stamps the
    tombstone itself.

    Returns:
        ``BuryResult`` with ``outcome="buried"`` or ``"already_buried"``.

    Raises:
        ToolError: unknown id; empty ``reason``.
    """
    del clock  # unused: the adapter stamps the tombstone's own timestamp
    require_text(reason, "bury: reason is required and must be non-empty (SPEC §9.1)")

    async with lock:
        try:
            tombstone_id = await adapter.bury(id, reason)
        except KeyError:
            tombstone = await adapter.get_tombstone(id)
            if tombstone is None:
                raise unknown_id(id) from None
            # Scrub again on a retry: a crash after adapter.bury and before the first scrub
            # would otherwise leave incoming_text behind.
            await adapter.scrub_verdict_text(id)
            return BuryResult(
                buried_id=id,
                tombstone_id=tombstone.id,
                outcome="already_buried",
                reason=reason,
            )

        await adapter.scrub_verdict_text(id)

        return BuryResult(
            buried_id=id,
            tombstone_id=tombstone_id,
            outcome="buried",
            reason=reason,
        )


async def revoke(
    *,
    id: HypomnemaId,
    reason: str,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
) -> ReviseResult:
    """Revoke an in-force Decision.

    ``in_force`` to ``revoked`` is a permanent edge; revoking a ``superseded`` Decision is illegal.
    Re-revoking a ``revoked`` Decision returns ``outcome="unchanged"`` without moving
    ``updated_at``. ``clock`` stamps ``updated_at`` on a genuine revocation.

    Returns:
        ``ReviseResult`` with ``outcome="unchanged"`` or ``"revised"``.

    Raises:
        ToolError: unknown or buried id (also when a race buries the record mid-call); empty
                   ``reason``; id names a Fact; the Decision is already ``superseded``.
    """
    require_text(reason, "revoke: reason is required and must be non-empty (SPEC §9.1)")

    async with lock:
        record = await get_record(adapter, id)

        if not isinstance(record, Decision):
            raise ToolError(
                f"revoke: id={id!s} is a Fact, not a Decision; "
                "revoke is Decision-only (SPEC §9.2)"
            )

        if record.lifecycle == "revoked":
                # Idempotent retry: stamp so a committed reversal keeps its override stamp.
            await stamp_override(adapter, ref=str(id), tool="revoke", now=clock.now())
            return ReviseResult(
                id=id,
                outcome="unchanged",
                lifecycle=record.lifecycle,
                validity=record.validity,
                reason=reason,
            )

        if record.lifecycle == "superseded":
            raise ToolError(
                f"revoke: Decision id={id!s} is already superseded; "
                "both Decision edges originate at in_force (SPEC §1.2) — "
                "a superseded Decision cannot be revoked"
            )

        now = clock.now()
        try:
            await adapter.transition(id, "revoked", at=now)
        except KeyError:
            # Buried between the existence check and the write: report it as unknown.
            raise unknown_id(id) from None
        updated = await get_record(adapter, id)
        await stamp_override(adapter, ref=str(id), tool="revoke", now=now)
        return ReviseResult(
            id=id,
            outcome="revised",
            lifecycle=updated.lifecycle,
            validity=updated.validity,
            reason=reason,
        )


async def check_supersedes_reason(
    *,
    predecessor_ids: Sequence[HypomnemaId],
    reason: SupersessionReason | None,
    successor_scope: Scope,
    adapter: StorageAdapter,
    span: list[Scope] | None = None,
) -> None:
    """Validate ``reason`` and the scope grouping of ``predecessor_ids`` (``save``, ``supersede``).

    Runs before the embedder and the lock, so it is informational: ``replace`` re-checks each
    predecessor fresh under the lock. ``span`` (``save``'s Fact arm) requires every predecessor's
    scope to be a member, checked against the record already fetched for the grouping check;
    ``None`` is unrestricted, which is how ``supersede`` calls it.

    Raises:
        ToolError: ``predecessor_ids`` empty and ``reason`` not ``None``, or the reverse; an id
                   unknown or buried; a predecessor's scope outside ``span``; the list mixing
                   scopes relative to ``successor_scope`` (split it into two calls); ``reason``
                   ``"moved"`` against an all-same-scope list, or any other class against an
                   all-cross-scope list.
    """
    if not predecessor_ids:
        if reason is not None:
            raise ToolError("reason must not be set without supersedes/predecessor_ids")
        return
    if reason is None:
        raise ToolError("reason is required when supersedes/predecessor_ids is non-empty")

    same_scope: list[HypomnemaId] = []
    cross_scope: list[HypomnemaId] = []
    for pid in predecessor_ids:
        try:
            predecessor = await adapter.get_by_id(pid)
        except KeyError:
            raise ToolError(
                f"supersedes: unknown id={pid!s} (buried or does not exist)"
            ) from None
        if span is not None and predecessor.scope not in span:
            raise ToolError(
                f"supersedes: Fact id={pid!s} belongs to scope "
                f"{predecessor.scope!r} which is outside the caller's read "
                f"span {[str(s) for s in span]!r}; cross-scope supersession "
                "beyond the read span is not allowed"
            )
        if predecessor.scope == successor_scope:
            same_scope.append(pid)
        else:
            cross_scope.append(pid)

    if same_scope and cross_scope:
        raise ToolError(
            "supersedes: predecessors must be all same-scope or all "
            f"cross-scope relative to the successor's scope {successor_scope!r} "
            f"— split into two calls; same-scope={[str(p) for p in same_scope]!r}, "
            f"cross-scope={[str(p) for p in cross_scope]!r}"
        )
    if cross_scope and reason != "moved":
        raise ToolError(
            f"supersedes: reason={reason!r} is refused for a cross-scope "
            "predecessor — cross-scope supersession requires reason='moved'"
        )
    if same_scope and reason == "moved":
        raise ToolError(
            "supersedes: reason='moved' is refused for a same-scope "
            "predecessor — 'moved' is for cross-scope supersession only"
        )


def supersede_legality_span(
    *,
    reason: SupersessionReason | None,
    read_span: list[Scope],
    probe_span: list[Scope],
) -> list[Scope]:
    """The scope-isolation span for ``save``'s ``supersedes`` legality check.

    ``reason="moved"`` is legal across the wider admission probe span, because a ``global`` save
    must be able to name a same-content predecessor outside ``[global]``. Every other reason keeps
    the read-span-only guard.
    """
    return probe_span if reason == "moved" else read_span


async def validate_supersede(
    *,
    successor_id: HypomnemaId,
    predecessor_ids: Sequence[HypomnemaId],
    reason: SupersessionReason,
    note: str | None,
    adapter: StorageAdapter,
    span: list[Scope] | None,
    edge_id: str | None,
) -> tuple[Fact, list[PredecessorFlip]]:
    """The lock-free half of ``supersede``: every pre-lock check.

    Split out so a caller holding ``lock`` across its own pending-verdict-row write and
    ``supersede_write_locked`` need not re-acquire it. Returns the fresh ``successor`` and the
    ``flips`` that ``supersede_write_locked`` writes; raises the refusals ``supersede`` documents.
    """
    if not predecessor_ids:
        raise ToolError("supersede: predecessor_ids must be non-empty")
    if successor_id in predecessor_ids:
        raise ToolError(f"supersede: a Fact cannot supersede itself (id={successor_id!s})")

    try:
        successor = await adapter.get_by_id(successor_id)
    except KeyError:
        raise ToolError(f"supersede: unknown/buried successor_id={successor_id!s}") from None
    if not isinstance(successor, Fact):
        raise ToolError(
            f"supersede: successor_id={successor_id!s} is a Decision, not a "
            "Fact; supersede is Facts-only"
        )
    if successor.superseded_by is not None:
        raise ToolError(
            f"supersede: successor_id={successor_id!s} is itself already "
            f"superseded (by {successor.superseded_by!s}) — cannot name an "
            "already-superseded record as a successor"
        )
    if successor.validity != "valid" or successor.lifecycle == "shelved":
        raise ToolError(
            f"supersede: successor_id={successor_id!s} is not eligible "
            f"(validity={successor.validity!r}, lifecycle={successor.lifecycle!r}) "
            "— only a valid, unretired Fact may take a new supersession edge"
        )

    await check_supersedes_reason(
        predecessor_ids=predecessor_ids,
        reason=reason,
        successor_scope=successor.scope,
        adapter=adapter,
    )

    for pid in predecessor_ids:
        try:
            predecessor = await adapter.get_by_id(pid)
        except KeyError:
            raise ToolError(f"supersede: unknown/buried id={pid!s}") from None
        if not isinstance(predecessor, Fact):
            raise ToolError(
                f"supersede: id={pid!s} is a Decision, not a Fact; "
                "supersede is Facts-only"
            )
        if span is not None and predecessor.scope not in span:
            raise ToolError(
                f"supersede: Fact id={pid!s} belongs to scope "
                f"{predecessor.scope!r} which is outside the caller's read "
                f"span {[str(s) for s in span]!r}; cross-scope supersession "
                "is not allowed"
            )
        if predecessor.superseded_by is not None:
            raise ToolError(
                f"supersede: Fact id={pid!s} is already superseded "
                f"(superseded_by={predecessor.superseded_by!s})"
            )
        if predecessor.validity != "valid":
            raise ToolError(
                f"supersede: Fact id={pid!s} is not valid (validity="
                f"{predecessor.validity!r}) — only a valid Fact can be superseded"
            )

    if edge_id is not None and len(predecessor_ids) != 1:
        raise ToolError(
            f"supersede: edge_id is only valid with a single predecessor, got "
            f"{len(predecessor_ids)}"
        )

    flips = [
        PredecessorFlip(
            id=pid, reason=reason, note=note,
            edge_id=edge_id if edge_id is not None else str(uuid4()),
        )
        for pid in predecessor_ids
    ]
    return successor, flips


async def supersede_write_locked(
    *,
    successor: Fact,
    successor_id: HypomnemaId,
    predecessor_ids: Sequence[HypomnemaId],
    reason: SupersessionReason,
    flips: list[PredecessorFlip],
    adapter: StorageAdapter,
    now: datetime,
) -> SupersedeResult:
    """The write itself. The caller must already hold the write lock; this never acquires it.

    Split out of ``supersede`` so a caller can hold one lock across its pending-verdict append,
    this write and its own stamp. Separate acquisitions would let a concurrent ``unsupersede``
    land between the write and the stamp and be erased by it.
    """
    try:
        await adapter.replace(successor, None, flips, at=now)
    except Exception as exc:
        log_failed_supersession(
            op="supersede",
            arm="fact",
            successor_id=successor_id,
            predecessor_ids=predecessor_ids,
            reason=reason,
            scope=successor.scope,
            error=exc,
        )
        # ValidationError subclasses ValueError; keep a data-integrity failure from reading as a
        # lost CAS race.
        if isinstance(exc, pydantic.ValidationError):
            raise ToolError(
                f"supersede: unexpected data-integrity failure while "
                f"writing the edge for successor_id={successor_id!s} "
                f"(NOT a lost race — this is not one of the documented "
                f"CAS-race messages): {exc}"
            ) from exc
        raise ToolError(f"supersede: {exc}") from exc

    return SupersedeResult(
        successor_id=successor_id,
        edges=tuple((flip.id, flip.edge_id) for flip in flips),
    )


async def supersede(
    *,
    successor_id: HypomnemaId,
    predecessor_ids: Sequence[HypomnemaId],
    reason: SupersessionReason,
    note: str | None,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
    span: list[Scope] | None = None,
    edge_id: str | None = None,
) -> SupersedeResult:
    """Draw a forward pointer from one or more existing Facts to an existing Fact, in one records
    transaction. Facts only, on both sides.

    ``span``: when given, every predecessor's scope must be in it, checked here before the lock.
    ``None`` is unrestricted, as for the ``tabularium supersede`` CLI subcommand.

    ``edge_id``: pre-minted by a caller that names the edge in a ``pending`` verdict row before
    this runs, so a crash between the two leaves reconcile a real id to look for. Single
    predecessor only. ``None`` mints one.

    ``reason``, ``note`` and the same-scope-or-all-cross-scope grouping are validated by
    ``check_supersedes_reason``. The successor must be a valid, unretired Fact, rechecked fresh by
    ``adapter.replace`` under the same lock hold as every predecessor. On a ``replace`` failure
    ``log_failed_supersession`` fires once, then ``ToolError``; nothing is partially written.

    Returns:
        ``SupersedeResult`` with the successor and one ``(predecessor_id, edge_id)`` pair per
        predecessor, in the caller's order.

    Raises:
        ToolError: ``predecessor_ids`` empty or containing ``successor_id``; unknown or buried
                   successor or predecessor; the successor or any predecessor is a Decision; a
                   predecessor outside ``span``; the ``check_supersedes_reason`` refusals; the
                   successor invalid or retired, or any predecessor stale, at ``replace``'s fresh
                   check; ``edge_id`` with more than one predecessor.
    """
    successor, flips = await validate_supersede(
        successor_id=successor_id,
        predecessor_ids=predecessor_ids,
        reason=reason,
        note=note,
        adapter=adapter,
        span=span,
        edge_id=edge_id,
    )
    now = clock.now()
    async with lock:
        return await supersede_write_locked(
            successor=successor,
            successor_id=successor_id,
            predecessor_ids=predecessor_ids,
            reason=reason,
            flips=flips,
            adapter=adapter,
            now=now,
        )


@dataclass(frozen=True, slots=True)
class _UnsupersedeEdge:
    """The edge being undone, as read from the predecessor under the write lock."""

    predecessor_id: HypomnemaId
    successor_id: HypomnemaId
    edge_id: str
    arm: Kind
    withdrawn_reason: SupersessionReason | None
    withdrawn_note: str | None
    reason: str
    at: datetime


def _unsupersede_edge(record: Fact | Decision, *, reason: str, clock: Clock) -> _UnsupersedeEdge:
    """Refuse a record with no edge to undo, then read the edge and ``clock.now()`` as its instant.

    Raises ``ToolError`` when ``superseded_by`` is ``None`` or ``supersession_edge_id`` is
    missing. Does no storage I/O.
    """
    if record.superseded_by is None:
        raise ToolError(
            f"unsupersede: id={record.id!s} is not currently superseded — "
            "nothing to undo (never superseded, or already undone)"
        )
    if record.supersession_edge_id is None:
        raise ToolError(
            f"unsupersede: id={record.id!s} is superseded but carries no "
            "supersession_edge_id — this should not happen once the "
            "startup edge-id population pass has run; retry after a "
            "server restart"
        )

    return _UnsupersedeEdge(
        predecessor_id=record.id,
        successor_id=record.superseded_by,
        edge_id=record.supersession_edge_id,
        arm="fact" if isinstance(record, Fact) else "decision",
        withdrawn_reason=record.supersession_reason,
        withdrawn_note=record.supersession_note,
        reason=reason,
        at=clock.now(),
    )


async def _check_decision_successor(adapter: StorageAdapter, edge: _UnsupersedeEdge) -> Decision:
    """Return the Decision successor of ``edge`` after the checks that make deleting it safe.

    Raises ``ToolError`` when the successor is unknown or buried, is not ``in_force``, or is also
    the successor of another Decision.
    """
    id = edge.predecessor_id
    successor_id = edge.successor_id
    try:
        successor = await adapter.get_by_id(successor_id)
    except KeyError:
        raise ToolError(
            f"unsupersede: id={id!s}'s successor id={successor_id!s} "
            "is unknown or buried — cannot verify it is in_force"
        ) from None
    assert isinstance(successor, Decision)
    if successor.lifecycle != "in_force":
        raise ToolError(
            f"unsupersede: successor id={successor_id!s} is not "
            f"in_force (lifecycle={successor.lifecycle!r}) — a "
            "superseded, revoked or buried successor has its own "
            "history the delete would erase"
        )

    # adapter.restore deletes the successor, which would orphan any other predecessor's
    # superseded_by pointer, so a shared successor is refused.
    sibling_predecessor_ids: list[HypomnemaId] = []
    async for other in await adapter.scan_records(lifecycle_in=None):
        if (
            other.id != id
            and isinstance(other, Decision)
            and other.superseded_by == successor_id
        ):
            sibling_predecessor_ids.append(other.id)
    if sibling_predecessor_ids:
        raise ToolError(
            f"unsupersede: successor id={successor_id!s} also carries "
            f"{[str(s) for s in sibling_predecessor_ids]!r} as live "
            f"predecessors — undoing id={id!s} alone would delete the "
            "successor out from under them; not supported yet"
        )

    return successor


@contextmanager
def _logged_unsupersede_failure(
    edge: _UnsupersedeEdge, *, passthrough: tuple[type[Exception], ...] = ()
) -> Iterator[None]:
    """Log a write failure inside the block with ``log_unsupersede_failed`` and re-raise it as
    ``ToolError("unsupersede: <error>")`` chained from the original.

    An exception of a ``passthrough`` type propagates unchanged and unlogged. A ``BaseException``
    that is not an ``Exception`` is never caught.
    """
    try:
        yield
    except passthrough:
        raise
    except Exception as exc:
        log_unsupersede_failed(
            op="unsupersede",
            arm=edge.arm,
            predecessor_id=edge.predecessor_id,
            successor_id=edge.successor_id,
            edge_id=edge.edge_id,
            reason=edge.reason,
            error=exc,
        )
        raise ToolError(f"unsupersede: {exc}") from exc


async def _append_unsupersede_event(adapter: StorageAdapter, edge: _UnsupersedeEdge) -> None:
    """Append the undo event, keyed by the edge id, as the first write of an undo."""
    event = UnsupersedeEvent(
        id=uuid4(),
        edge_id=edge.edge_id,
        predecessor_id=edge.predecessor_id,
        successor_id=edge.successor_id,
        arm=edge.arm,
        withdrawn_reason=edge.withdrawn_reason,
        withdrawn_note=edge.withdrawn_note,
        reason=edge.reason,
        at=edge.at,
    )
    with _logged_unsupersede_failure(edge):
        await adapter.append_unsupersede(event)


async def _plant_restore_tombstone(
    adapter: StorageAdapter, edge: _UnsupersedeEdge, successor: Decision
) -> None:
    """Make the successor carry a tombstone whose ``restores`` names the predecessor.

    Plants one when there is none. Keeps an existing one whose ``restores_content_hash`` matches
    the successor's content, withdraws and replants a stale one, and raises ``ToolError`` for one
    that restores a different predecessor. A storage failure is logged and raised as ``ToolError``.
    """
    successor_id = edge.successor_id
    with _logged_unsupersede_failure(edge, passthrough=(ToolError,)):
        existing_tombstone = await adapter.get_tombstone(successor_id)
        if existing_tombstone is None:
            await adapter.plant_tombstone(
                successor_id,
                f"unsupersede: {edge.reason}",
                restores=edge.predecessor_id,
            )
        elif existing_tombstone.restores != edge.predecessor_id:
            raise ToolError(
                f"unsupersede: id={edge.predecessor_id!s}'s successor id={successor_id!s} already "
                f"carries a tombstone that does not authorise this restore "
                f"(restores={existing_tombstone.restores!s}) — refusing before "
                "the destructive transaction"
            )
        elif existing_tombstone.restores_content_hash != content_hash(successor.content):
            await adapter.withdraw_tombstone(successor_id)
            await adapter.plant_tombstone(
                successor_id,
                f"unsupersede: {edge.reason}",
                restores=edge.predecessor_id,
            )


async def _restore_predecessor(adapter: StorageAdapter, edge: _UnsupersedeEdge) -> None:
    """Run ``adapter.restore`` for the edge, the last write of an undo."""
    with _logged_unsupersede_failure(edge):
        await adapter.restore(
            edge.predecessor_id, successor_id=edge.successor_id, edge_id=edge.edge_id, at=edge.at
        )


async def unsupersede(
    *,
    id: HypomnemaId,
    reason: str,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
    skip_override_stamp: bool = False,
) -> UnsupersedeResult:
    """Undo a supersession on either arm, dispatching on the predecessor's own kind.

    Fact arm: the pointer clears and validity restores on the predecessor; both rows stay live.
    Decision arm: the predecessor returns to ``in_force`` and the successor is deleted.

    Write order, under ``lock``: the undo event, keyed by the predecessor's current
    ``supersession_edge_id`` and idempotent on it. On the Decision arm the successor's tombstone
    carrying ``restores`` is planted next. An existing tombstone is kept when its
    ``restores_content_hash`` matches the successor's current content; if it does not, it is
    withdrawn and replanted. A live call always completes this way, whereas the startup repair
    pass backs off. ``adapter.restore`` runs last.

    ``skip_override_stamp`` leaves the verdict rows to the caller; the Cleaner passes it when
    reverting an edge it just drew, so a self-revert is not counted as an owner override.

    ``withdrawn_reason`` is ``None`` for an edge drawn before the reason column existed.

    Args:
        id:      The superseded predecessor to restore.
        reason:  Audit-trail reason for the undo, carried on the undo event and not on the row.
        clock:   Stamps the undo event and the records transaction with the same instant.

    Returns:
        ``UnsupersedeResult`` naming which shape ran.

    Raises:
        ToolError: unknown or buried id; empty ``reason``; ``superseded_by`` already ``None``
                   (never superseded or already undone; this is a refusal, not a re-op); no
                   ``supersession_edge_id``; on the Decision arm, a successor that is unknown,
                   buried or not ``in_force``, that has other live predecessors, or that already
                   carries a tombstone whose ``restores`` names a different predecessor; a
                   storage failure at any write step, logged by ``log_unsupersede_failed``.
    """
    require_text(reason, "unsupersede: reason is required and must be non-empty (SPEC §9.1)")

    await get_record(adapter, id)

    async with lock:
        record = await get_record(adapter, id)
        edge = _unsupersede_edge(record, reason=reason, clock=clock)
        successor = (
            await _check_decision_successor(adapter, edge) if edge.arm == "decision" else None
        )

        await _append_unsupersede_event(adapter, edge)
        if successor is not None:
            await _plant_restore_tombstone(adapter, edge, successor)
        await _restore_predecessor(adapter, edge)

        shape: UnsupersedeShape = (
            "fact_both_live" if edge.arm == "fact" else "decision_restored"
        )
        log_unsupersede_completed(
            arm=edge.arm,
            predecessor_id=id,
            successor_id=edge.successor_id,
            edge_id=edge.edge_id,
            shape=shape,
        )
        if not skip_override_stamp:
            await stamp_override(adapter, ref=edge.edge_id, tool="unsupersede", now=edge.at)
        return UnsupersedeResult(
            predecessor_id=id,
            successor_id=edge.successor_id,
            edge_id=edge.edge_id,
            arm=edge.arm,
            shape=shape,
            withdrawn_reason=edge.withdrawn_reason,
            withdrawn_note=edge.withdrawn_note,
            at=edge.at,
        )


async def _refuse_unspecified_hot_roles(adapter: StorageAdapter) -> NoReturn:
    registered = await adapter.list_roles()
    raise ToolError(
        "set_hot(hot=True) requires roles naming the Fact's audience "
        "explicitly — an unspecified audience is a caller bug (owner "
        "decision 2026-08-02: \"no caller has no role — bounce back to "
        "get it\"), not a state this tool leaves to whatever "
        "audience_hint the Fact happened to capture at write time. "
        "Pass roles=[...] to route it, or roles=[] to pin it hot while "
        f"deliberately reaching no role. Registered roles: {registered!r}"
    )


async def _hot_target_fact(adapter: StorageAdapter, id: HypomnemaId) -> Fact:
    """Fetch the record for ``set_hot`` and refuse a Decision. The caller holds the write lock."""
    record = await get_record(adapter, id)

    if not isinstance(record, Fact):
        raise ToolError(
            f"set_hot: id={id!s} is a Decision, not a Fact; "
            "set_hot is Facts-only — the hot tier holds only Facts "
            "(SPEC §1.2 / §8.5)"
        )

    return record


async def _check_hot_cap(
    *, record: Fact, roles: list[str], adapter: StorageAdapter, cfg: EngineConfig
) -> None:
    """Raise ``ToolError`` when promoting ``record`` into any of ``roles`` would exceed its cap.

    Counts, per role, the hot Facts in the record's own scope that route to the role, excluding
    the record itself. Every role is checked before the caller writes. An empty ``roles``
    scans nothing and passes. The caller holds the write lock.
    """
    if not roles:
        return

    cap = cfg.hot_cap_global if record.scope == GLOBAL else cfg.hot_cap_scope
    scoped_hot: list[Fact] = []
    async for candidate in await adapter.scan_records(lifecycle_in=PUSH_HOT_LIFECYCLE):
        if (
            isinstance(candidate, Fact)
            and candidate.scope == record.scope
            and candidate.id != record.id
        ):
            scoped_hot.append(candidate)
    for role in roles:
        occupants = [c.id for c in scoped_hot if routes_to(c, role)]
        if len(occupants) >= cap:
            raise ToolError(
                f"set_hot: promoting id={record.id!s} into role={role!r} "
                f"would exceed hot_cap={cap} for "
                f"(scope={record.scope!r}, role={role!r}). "
                f"Currently hot for role={role!r} in "
                f"scope={record.scope!r}: "
                f"{[str(o) for o in occupants]}. hot_cap is a "
                "payload-size guardrail, not a contest — the "
                "owner chooses what leaves (set_hot(<id>, "
                "hot=False), retag it off this role, or bury it) "
                "before promoting another; nothing is evicted "
                "automatically, and this whole promotion is "
                "rejected — no role is partially tagged even "
                "when other requested roles still have room."
            )


async def _write_hot_lifecycle(
    *,
    id: HypomnemaId,
    target: Literal["hot", "active"],
    audience_hint: tuple[str, ...] | None,
    provisional: bool | None,
    adapter: StorageAdapter,
    clock: Clock,
) -> Fact:
    """Transition ``id`` to ``target`` at ``clock.now()`` and return the re-read Fact.

    A KeyError from the transition (buried since the caller's read) is reported as an unknown id.
    """
    now = clock.now()
    try:
        await adapter.transition(
            id, target, at=now, audience_hint=audience_hint, provisional=provisional
        )
    except KeyError:
        # Buried between the checks and the write: report it as unknown.
        raise unknown_id(id) from None

    return await get_fact(adapter, id)


async def _promote(
    *,
    record: Fact,
    roles: list[str],
    adapter: StorageAdapter,
    clock: Clock,
    cfg: EngineConfig,
) -> Fact:
    """Promote ``record`` to ``hot`` with ``roles`` as its audience, clearing ``provisional``.

    Returns ``record`` unchanged, with no write, when it is already hot with exactly ``roles``.
    Otherwise refuses a lifecycle other than ``hot``, ``active`` or ``ember``, runs
    ``_check_hot_cap`` and writes. The caller holds the write lock.
    """
    if record.lifecycle == "hot":
        # Reachable only with the exact roles the Fact already carries; different roles fall
        # through to the hot_cap check and a write.
        if tuple(roles) == (record.audience_hint or ()):
            return record
    elif record.lifecycle not in ("active", "ember"):
        raise ToolError(
            f"set_hot: Fact id={record.id!s} has lifecycle={record.lifecycle!r}; "
            "only an active or ember Fact can be promoted to hot "
            "(archived Facts are not promotable via set_hot; nor are "
            "shelved Facts — ADR 0021 — un-shelve first via "
            "set_shelved(id, shelved=False, reason=...), then set_hot)"
        )

    await _check_hot_cap(record=record, roles=roles, adapter=adapter, cfg=cfg)

    return await _write_hot_lifecycle(
        id=record.id,
        target="hot",
        audience_hint=tuple(roles),
        provisional=False,
        adapter=adapter,
        clock=clock,
    )


async def _demote(
    *,
    record: Fact,
    roles: list[str] | None,
    adapter: StorageAdapter,
    clock: Clock,
) -> Fact:
    """Demote a hot ``record`` to ``active``, leaving ``provisional`` as it is.

    An active record with ``roles=None`` is returned unchanged with no write. An active record
    with ``roles`` is retagged through a same-lifecycle transition without re-arming. A hot
    record's demotion re-arms ``escalation_baseline`` to the owner recurrence count unless it
    already equals it. Refuses any other lifecycle. The caller holds the write lock.
    """
    if record.lifecycle == "active":
        if roles is None:
            return record
    elif record.lifecycle != "hot":
        raise ToolError(
            f"set_hot: Fact id={record.id!s} has lifecycle={record.lifecycle!r}; "
            "only a hot Fact can be demoted to active via "
            "set_hot(hot=False)"
        )

    updated = await _write_hot_lifecycle(
        id=record.id,
        target="active",
        audience_hint=tuple(roles) if roles is not None else None,
        provisional=None,
        adapter=adapter,
        clock=clock,
    )

    if (
        record.lifecycle == "hot"
        and updated.escalation_baseline != updated.owner_recurrence_count
    ):
        # Re-arm the ladder: a future owner-asserted crossing must reach escalation_threshold
        # from the current count. Skipped when the baseline is already current.
        await adapter.update_recurrence(
            record.id,
            owner_recurrence_count=updated.owner_recurrence_count,
            agent_recurrence_count=updated.agent_recurrence_count,
            escalation_baseline=updated.owner_recurrence_count,
        )
        updated = await get_fact(adapter, record.id)

    return updated


async def set_hot(
    *,
    id: HypomnemaId,
    hot: bool,
    roles: list[str] | None = None,
    adapter: StorageAdapter,
    clock: Clock,
    cfg: EngineConfig,
    lock: asyncio.Lock,
    require_roles: RequireRoles,
) -> Fact:
    """Promote an ``active`` or ``ember`` Fact to ``hot``, or demote a ``hot`` Fact to ``active``.

    Owner-curated. Promoting an ember is the Foreground exit of the escalation ladder; the
    suggestion is advisory and no recurrence threshold is enforced. Facts only.

    Edges: ``hot=True`` needs the Fact ``active`` or ``ember``; ``hot=False`` needs it ``hot`` and
    always lands on ``active``, never ``ember``. Archived and shelved Facts cannot be promoted.

    ``roles`` is the Fact's new ``audience_hint``; it overwrites and never unions, and is
    validated with ``require_roles`` before any write. ``hot=True`` requires it: ``roles=None``
    raises before the record is fetched. ``roles=[]`` is legal and means hot but reaching no role.
    With ``hot=False`` it is optional: ``None`` leaves the hint and ``[]`` clears it.

    No-write re-ops, which return the unchanged record: ``hot=True`` on a hot Fact whose current
    ``audience_hint`` equals ``roles``, and ``hot=False`` on an active Fact with ``roles=None``.
    Otherwise an already-in-state call is a role-only retag through a same-lifecycle
    ``transition``. A promote-side retag re-runs the ``hot_cap`` check; a demote-side retag does
    not and does not re-arm the baseline.

    Promotion checks ``hot_cap`` for every role in ``roles`` before any write, so one violation
    rejects the whole promotion. The Fact itself is excluded from the occupant counts, so a retag
    into a role it already holds never trips that role's cap.

    Promotion clears ``provisional``. Demotion leaves it. Demotion resets
    ``escalation_baseline`` to the current ``owner_recurrence_count``, skipped when already
    current, so the next owner-asserted crossing fires from the new baseline.

    Args:
        id:            Fact id.
        hot:           ``True`` to promote, ``False`` to demote.
        roles:         Registered roles to persist as ``audience_hint``; see above.
        clock:         Stamps ``updated_at`` on a genuine transition.
        cfg:           Supplies ``hot_cap_global`` and ``hot_cap_scope``.
        require_roles: Role-registry gate; raises ``ToolError`` on an unregistered role.

    Returns:
        The Fact after the call: unchanged on a no-write re-op, otherwise re-read after the
        ``transition`` (and again after the baseline re-arm write).

    Raises:
        ToolError: unknown or buried id; id names a Decision; the requested edge is illegal for
                   the Fact's lifecycle; promoting into a role past its ``hot_cap``; an
                   unregistered role in ``roles``; ``hot=True`` with ``roles=None``.
    """
    await require_roles(roles)
    if hot and roles is None:
        await _refuse_unspecified_hot_roles(adapter)

    async with lock:
        record = await _hot_target_fact(adapter, id)

        if hot:
            assert roles is not None
            return await _promote(record=record, roles=roles, adapter=adapter, clock=clock, cfg=cfg)

        return await _demote(record=record, roles=roles, adapter=adapter, clock=clock)


async def set_audience(
    *,
    id: HypomnemaId,
    roles: list[str],
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
    require_roles: RequireRoles,
) -> Fact:
    """Set a Fact's ``audience_hint`` and clear ``provisional``; ``lifecycle`` is never touched.

    The curation path for "this is for builders and reviewers, but not for every session's
    payload": it rules on the audience without promoting into the hot tier. Facts only.

    A hot Fact is refused with a ``ToolError`` naming ``set_hot``, which re-checks ``hot_cap``
    before persisting a new hint; a second, uncapped route to the field would let a caller exceed
    the cap. ``set_hot`` stays the only writer of ``lifecycle="hot"``.

    ``roles`` is required and overwrites the stored hint, never unions. ``[]`` is legal and
    means this Fact reaches no role. ``provisional`` is always set ``False`` by a genuine call.
    A call whose ``roles`` equal the current ``audience_hint`` on a Fact already
    ``provisional=False`` is a no-op returning the unchanged record; any other call writes.

    Args:
        id:            Fact id.
        roles:         Registered roles to persist as ``audience_hint``.
        clock:         Stamps ``updated_at`` on a genuine transition.
        require_roles: Role-registry gate; raises ``ToolError`` on an unregistered role.

    Returns:
        The Fact after the call: unchanged on the no-op, otherwise re-read after ``transition``.

    Raises:
        ToolError: ``roles`` is ``None`` (checked before the record is fetched); unknown or
                   buried id; id names a Decision; the Fact is ``hot``; an unregistered role in
                   ``roles``.
    """
    await require_roles(roles)
    if roles is None:
        # An untyped caller could pass None, and require_roles treats falsy input as nothing to
        # validate; fail with ToolError instead of a TypeError from tuple(roles) below.
        registered = await adapter.list_roles()
        raise ToolError(
            "set_audience requires roles naming the Fact's audience "
            "explicitly -- there is no \"leave the hint untouched\" mode, "
            "since the whole point of this tool is an explicit ruling. "
            "Pass roles=[...] to rule on it, or roles=[] to rule that it "
            f"reaches no role. Registered roles: {registered!r}"
        )

    async with lock:
        record = await get_record(adapter, id)

        if not isinstance(record, Fact):
            raise ToolError(
                f"set_audience: id={id!s} is a Decision, not a Fact; "
                "set_audience is Facts-only — `provisional` does not exist "
                "on a Decision (SPEC §1.1)"
            )

        if record.lifecycle == "hot":
            raise ToolError(
                f"set_audience: Fact id={id!s} is hot; retag a hot Fact's "
                "audience via set_hot(id, hot=True, roles=[...]) instead, "
                "which re-checks hot_cap before persisting the new hint. "
                "set_audience never touches hot membership and must not "
                "become a second, uncapped path to the same field."
            )

        target_hint = tuple(roles)
        if (record.audience_hint or ()) == target_hint and record.provisional is False:
            return record

        now = clock.now()
        try:
            await adapter.transition(
                id,
                record.lifecycle,
                at=now,
                audience_hint=target_hint,
                provisional=False,
            )
        except KeyError:
            # Buried between the checks and the write: report it as unknown.
            raise unknown_id(id) from None
        updated = await get_fact(adapter, id)

        return updated


async def _shelvable_fact(adapter: StorageAdapter, id: HypomnemaId) -> Fact:
    """Fetch the record for ``set_shelved`` and refuse a Decision. The caller holds the lock."""
    record = await get_record(adapter, id)

    if not isinstance(record, Fact):
        raise ToolError(
            f"set_shelved: id={id!s} is a Decision, not a Fact; "
            "set_shelved is Facts-only — Decisions have their own arm "
            "and their own three lifecycle states (SPEC §1.1, ADR 0021)"
        )

    return record


def _unchanged_shelve_result(record: Fact, reason: str) -> ReviseResult:
    return ReviseResult(
        id=record.id,
        outcome="unchanged",
        lifecycle=record.lifecycle,
        validity=record.validity,
        reason=reason,
    )


def _check_shelvable(record: Fact) -> None:
    """Refuse a lifecycle other than ``active`` or ``archived``."""
    if record.lifecycle not in ("active", "archived"):
        raise ToolError(
            f"set_shelved: Fact id={record.id!s} has lifecycle={record.lifecycle!r}; "
            "only an active or archived Fact can be shelved — demote a hot "
            "Fact via set_hot(id, hot=False) first, and an ember cannot be "
            "shelved directly (ADR 0021 names no such edge)"
        )


def _check_strict_shelve_guard(record: Fact) -> None:
    """Refuse a Fact that is not ``valid`` or is superseded, for callers with the stricter guard."""
    if record.validity != "valid" or record.superseded_by is not None:
        superseded_by_repr = (
            str(record.superseded_by) if record.superseded_by is not None else None
        )
        raise ToolError(
            f"set_shelved: Fact id={record.id!s} is not eligible for this caller's stricter "
            f"shelve guard (validity={record.validity!r}, "
            f"superseded_by={superseded_by_repr!r}) "
            "— superseded/invalid Facts are left alone (decision 17), checked here, "
            "inside this same lock, atomically with the fresh read above"
        )


def _check_unshelvable(record: Fact) -> None:
    """Refuse a lifecycle other than ``shelved``."""
    if record.lifecycle != "shelved":
        raise ToolError(
            f"set_shelved: Fact id={record.id!s} has lifecycle={record.lifecycle!r}; "
            "only a shelved Fact can be un-shelved via "
            "set_shelved(id, shelved=False, reason=...) — it was never "
            "shelved by this tool, so un-shelving it does not apply"
        )


def _shelve_event(
    id: HypomnemaId, *, at: datetime, source: Literal["shelved", "shelve_reverted"], reason: str
) -> ReinforcementEvent:
    return ReinforcementEvent(
        target_id=id,
        at=at,
        source=source,
        matched_from_id=None,
        value=None,
        reason=reason,
    )


async def _write_shelve_event_first(
    *,
    id: HypomnemaId,
    target: Literal["shelved", "active"],
    reason: str,
    adapter: StorageAdapter,
    clock: Clock,
) -> datetime:
    """Append a ``shelved`` event, then transition ``id`` to ``target``; return the shared
    instant. A failed transition leaves the event. The shelve and un-shelve writes both use it.
    """
    now = clock.now()
    event = _shelve_event(id, at=now, source="shelved", reason=reason)
    try:
        # Event first, transition second: see the set_shelved docstring.
        await adapter.append_event(event)
        await adapter.transition(id, target, at=now)
    except KeyError:
        # Buried between the existence check and the write: report it as unknown.
        raise unknown_id(id) from None

    return now


async def _revert_shelve(
    record: Fact,
    *,
    restore_lifecycle: Literal["active", "archived"],
    reason: str,
    adapter: StorageAdapter,
    clock: Clock,
) -> datetime:
    """Undo a Cleaner shelve: transition to ``restore_lifecycle`` first, then append a
    ``shelve_reverted`` event. Returns the write instant.

    A failed transition leaves no ``shelve_reverted`` event.
    """
    id = record.id
    now = clock.now()
    event = _shelve_event(id, at=now, source="shelve_reverted", reason=reason)
    try:
        await adapter.transition(id, restore_lifecycle, at=now)
        await adapter.append_event(event)
    except KeyError:
        # Buried between the existence check and the write: report it as unknown.
        raise unknown_id(id) from None

    return now


async def set_shelved(
    *,
    id: HypomnemaId,
    shelved: bool,
    reason: str,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
    require_valid_and_unsuperseded: bool = False,
    skip_override_stamp: bool = False,
    cleaner_revert: bool = False,
    restore_lifecycle: Literal["active", "archived"] = "active",
) -> ReviseResult:
    """Shelve or un-shelve a Fact. Facts only.

    ``shelved`` is the one lifecycle the dedup-probe filter (``probe_filters_for``) never sees, so
    a later save of the same content succeeds. It stays in ``primary_recall_filters``, so ordinary
    ``recall`` still returns it, and a hit on it is never revived.

    Edges: ``active -> shelved`` and ``archived -> shelved`` (``shelved=True``), and
    ``shelved -> active`` (``shelved=False``), which always lands on ``active`` unless
    ``cleaner_revert`` is set. Anything else raises ``ToolError``: ``hot`` or ``ember`` cannot be
    shelved (demote a hot Fact with ``set_hot`` first), and ``shelved=False`` needs a ``shelved``
    or ``active`` Fact. ``shelved=True`` on a ``shelved`` Fact, or ``shelved=False`` on an
    ``active`` one, returns ``outcome="unchanged"`` with no record write and no event. The
    already-``shelved`` case still calls ``stamp_override`` (which may update verdict rows) unless
    ``skip_override_stamp`` is set.

    ``reason`` is persisted as a ``ReinforcementEvent`` with ``source`` ``"shelved"``, which no
    projection formula counts. The event is appended before the ``transition`` so that a failed
    transition leaves the audit event. The exception is ``shelve_reverted``, which transitions
    first, so that a failed transition leaves no event cancelling the shelve's own ``shelved``
    event in revisit's protection count.

    ``skip_override_stamp``: when ``True``, never touch ``stamp_override``. The Cleaner's
    split-shelve write pre-publishes its verdict row's ``new_record_id`` before calling here, and
    the stamp would mistake that write for an owner reversal. The caller stamps the row ``acted``
    itself inside its own lock hold.

    ``cleaner_revert``: set only by the Cleaner's undo of a shelve it just wrote. The un-shelve
    appends ``source="shelve_reverted"``, which revisit's protection check does not count as an
    owner round trip, and lands on ``restore_lifecycle`` instead of ``active``. Ignored when
    ``shelved=True``.

    ``require_valid_and_unsuperseded``: when ``True`` and ``shelved=True``, also requires
    ``validity == "valid"`` and ``superseded_by is None``, checked inside this lock against the
    fresh read, raising ``ToolError`` before any write. The revisit axis passes it, because a
    check before the lock would leave a window for another writer.

    Args:
        id:      Fact id.
        shelved: ``True`` to shelve, ``False`` to un-shelve.
        reason:  Audit-trail reason, non-empty.
        clock:   Stamps the event and the ``transition`` with the same instant.

    Returns:
        ``ReviseResult`` with ``outcome="unchanged"`` or ``"revised"``. ``validity`` is the
        record's own.

    Raises:
        ToolError: unknown or buried id (also when a race buries the record mid-call); empty
                   ``reason``; id names a Decision; the edge is illegal for the Fact's
                   lifecycle; the ``require_valid_and_unsuperseded`` guard fails.
    """
    require_text(reason, "set_shelved: reason is required and must be non-empty (SPEC §9.1)")

    async with lock:
        record = await _shelvable_fact(adapter, id)

        if shelved and record.lifecycle == "shelved":
            # Idempotent retry: stamp so a committed reversal keeps its override stamp.
            if not skip_override_stamp:
                await stamp_override(adapter, ref=str(id), tool="set_shelved", now=clock.now())
            return _unchanged_shelve_result(record, reason)

        if not shelved and record.lifecycle == "active":
            # Not a stamp_override site: active is the default state, so reaching it does not
            # mean this tool ran, and stamping would inflate the override count.
            return _unchanged_shelve_result(record, reason)

        if shelved:
            _check_shelvable(record)
            if require_valid_and_unsuperseded:
                _check_strict_shelve_guard(record)
            now = await _write_shelve_event_first(
                id=id, target="shelved", reason=reason, adapter=adapter, clock=clock
            )
        elif cleaner_revert:
            _check_unshelvable(record)
            now = await _revert_shelve(
                record,
                restore_lifecycle=restore_lifecycle,
                reason=reason,
                adapter=adapter,
                clock=clock,
            )
        else:
            _check_unshelvable(record)
            now = await _write_shelve_event_first(
                id=id, target="active", reason=reason, adapter=adapter, clock=clock
            )

        updated = await get_fact(adapter, id)
        if not skip_override_stamp:
            await stamp_override(adapter, ref=str(id), tool="set_shelved", now=now)

        return ReviseResult(
            id=id,
            outcome="revised",
            lifecycle=updated.lifecycle,
            validity=updated.validity,
            reason=reason,
        )
