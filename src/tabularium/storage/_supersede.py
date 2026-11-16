"""Guarded-mutation resolvers for supersession: the forward edge (``replace``)
and its two-arm undo (``restore``).

Each adapter re-reads every referent a mutation touches immediately before the
write, under its own concurrency primitive (``LanceDBAdapter._supersede_lock``;
``MemoryAdapter``'s single yield point plus post-yield re-fetch). These
functions take those fresh reads and decide whether the write is still legal
and what it looks like, so neither adapter keeps its own copy of the rules.

- ``resolve_replace_successor`` / ``resolve_replace_predecessor`` /
  ``resolve_invalidation_target``: the forward edge drawn by
  ``StorageAdapter.replace``. One call may carry many predecessors against one
  successor; each is resolved independently and the caller stops at the first
  illegal one (all-or-nothing).
- ``resolve_unsupersede_write``: the reverse edge. Re-reads only the
  predecessor; the successor's eligibility (a Decision successor must be
  ``in_force``) is the engine's check, made before ``restore`` is called.
- ``resolve_unsupersede_append``: idempotency rule for ``append_unsupersede``.
- ``resolve_unsupersede_repair``: the pure decision behind
  ``StorageAdapter.repair_unsupersedes`` (skip, restore, plant then restore,
  or withdraw), with an optional warning for anomalies.
- ``run_unsupersede_repair``: the loop behind ``repair_unsupersedes``, calling
  ``resolve_unsupersede_repair`` per event; the adapter supplies a
  ``RepairIO`` for the reads and writes.
- ``resolve_withdraw_tombstone``: guard behind ``withdraw_tombstone``; a
  tombstone gives up its restore authority only while it is still pending and
  its buried row is still present (and ``in_force`` for a Decision).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from tabularium.domain import Decision, Fact, HypomnemaId, Kind
from tabularium.storage.content_key import content_hash
from tabularium.storage.types import (
    DecisionInvalidation,
    PredecessorFlip,
    RepairReport,
    Tombstone,
    UnsupersedeEvent,
)

_log = logging.getLogger("tabularium")


def resolve_replace_successor(
    *,
    successor: Fact | Decision,
    existing: Fact | Decision | None,
    vector: list[float] | None,
    buried: bool = False,
) -> Fact | Decision:
    """Validate the successor half of a ``replace`` call and return the record to write.

    ``existing`` must be a read taken immediately before the write, inside the
    caller's lock. ``None`` means the fresh lookup found no live row. ``buried`` is
    the same fresh check's answer to "was this id ever written and then buried";
    burial is the only deletion path, so it separates a new id from one a
    concurrent ``bury()`` removed.

    - ``existing is None`` and ``buried``: raises ``ValueError`` (lost a race
      against ``bury()``). Otherwise the successor is new and ``successor`` is
      returned unchanged. ``vector`` may be ``None`` there: the embedder was
      unavailable and the caller's dirty mark lets backfill fill it later.
    - ``existing is not None``: an already-live successor gaining edges. It must
      have the same type as ``successor`` and ``validity="valid"``; a Fact must
      have ``superseded_by is None`` and ``lifecycle != "shelved"``; a Decision
      must be ``in_force``. Raises ``ValueError`` naming the id otherwise, and
      returns ``existing``, never the caller's possibly stale ``successor``.
    """
    if existing is None:
        if buried:
            raise ValueError(
                f"replace: successor id={successor.id!s} does not exist or "
                "is buried (lost a concurrent race)"
            )
        return successor

    if type(existing) is not type(successor):
        raise ValueError(
            f"replace: successor id={successor.id!s} changed type between "
            f"the caller's read and this write (was {type(successor).__name__}, "
            f"now {type(existing).__name__}) — lost a concurrent race"
        )
    # Arm checks run before the generic validity check: a Fact's validator makes
    # ``superseded_by is not None`` imply ``validity == "invalid"``, so validity
    # first would mask the more specific "already superseded" message.
    if isinstance(existing, Fact):
        if existing.superseded_by is not None:
            raise ValueError(
                f"replace: successor id={successor.id!s} is itself already "
                f"superseded (by {existing.superseded_by!s}) — cannot name an "
                "already-superseded record as a successor"
            )
        if existing.lifecycle == "shelved":
            raise ValueError(
                f"replace: successor id={successor.id!s} is shelved — only "
                "a valid, unretired Fact may take a new supersession edge"
            )
    else:
        assert isinstance(existing, Decision)
        if existing.lifecycle != "in_force":
            raise ValueError(
                f"replace: successor id={successor.id!s} is not in_force "
                f"(lifecycle={existing.lifecycle!r}) — only an in_force "
                "Decision may take a new supersession edge"
            )
    if existing.validity != "valid":
        raise ValueError(
            f"replace: successor id={successor.id!s} is not valid "
            f"(validity={existing.validity!r}) — only a valid, unretired "
            "record may take a new supersession edge"
        )
    return existing


def resolve_replace_predecessor(
    *,
    predecessor: Fact | Decision | None,
    predecessor_id: HypomnemaId,
    successor_id: HypomnemaId,
    successor_type: type[Fact] | type[Decision],
    flip: PredecessorFlip,
    at: datetime,
) -> Fact | Decision:
    """Validate one predecessor of a ``replace`` call and return the record to write.

    ``predecessor`` must be a fresh read taken inside the caller's lock; ``None``
    means no live row.

    Raises ``ValueError`` naming ``predecessor_id`` when the predecessor does not
    exist, names itself as successor, differs in type from the successor, or is not
    supersedable (a Fact must be valid and not superseded; a Decision must be
    ``in_force``).

    The result carries the flip fields (``superseded_by``, ``supersession_reason``,
    ``supersession_note``, ``supersession_edge_id``, ``updated_at=at``) plus the
    arm change (a Fact becomes ``validity="invalid"``; a Decision becomes
    ``lifecycle="superseded"`` with ``validity`` untouched), layered onto the fresh
    read via ``model_validate``.
    """
    if predecessor is None:
        raise ValueError(
            f"replace: predecessor id={predecessor_id!s} does not exist or "
            "is buried (lost a concurrent race)"
        )
    if predecessor_id == successor_id:
        raise ValueError(
            f"replace: a record cannot supersede itself (id={predecessor_id!s})"
        )
    if not isinstance(predecessor, successor_type):
        raise ValueError(
            f"replace: predecessor id={predecessor_id!s} is a "
            f"{type(predecessor).__name__}, not a {successor_type.__name__} "
            "— both sides of one edge must be the same arm"
        )

    flip_fields: dict[str, object] = {
        "superseded_by": successor_id,
        "supersession_reason": flip.reason,
        "supersession_note": flip.note,
        "supersession_edge_id": flip.edge_id,
        "updated_at": at,
    }
    if isinstance(predecessor, Fact):
        if predecessor.superseded_by is not None:
            raise ValueError(
                f"replace: Fact id={predecessor_id!s} is already superseded "
                f"(by {predecessor.superseded_by!s}) — cannot supersede twice"
            )
        if predecessor.validity != "valid":
            raise ValueError(
                f"replace: Fact id={predecessor_id!s} is not valid "
                f"(validity={predecessor.validity!r}) — only a valid Fact "
                "can be superseded"
            )
        flip_fields["validity"] = "invalid"
    else:
        assert isinstance(predecessor, Decision)
        if predecessor.superseded_by is not None:
            raise ValueError(
                f"replace: Decision id={predecessor_id!s} is already "
                f"superseded (by {predecessor.superseded_by!s}) — cannot "
                "supersede twice"
            )
        if predecessor.lifecycle != "in_force":
            raise ValueError(
                f"replace: Decision id={predecessor_id!s} is not in_force "
                f"(lifecycle={predecessor.lifecycle!r}) — cannot supersede"
            )
        flip_fields["lifecycle"] = "superseded"

    return type(predecessor).model_validate(
        {**predecessor.model_dump(mode="python"), **flip_fields}
    )


def resolve_invalidation_target(
    *,
    target: Fact | Decision | None,
    target_id: HypomnemaId,
    successor_id: HypomnemaId,
    successor_type: type[Fact] | type[Decision],
    invalidation: DecisionInvalidation,
    flip_ids: frozenset[HypomnemaId],
) -> Decision:
    """Validate one invalidation target of a ``replace`` call and return the Decision to write.

    ``target`` must be a fresh read taken inside the caller's lock; ``None`` means
    no live row.

    Raises ``ValueError`` naming ``target_id`` when the target does not exist,
    ``successor_type`` is not ``Fact`` (only a Fact successor may invalidate a
    Decision), the target is not a Decision, it is not ``in_force`` and valid, or
    it is also in ``flip_ids`` (invalidation draws no edge, so a target cannot also
    be a flip predecessor).

    The result has ``validity="invalid"`` and ``invalidated_by`` /
    ``invalidation_reason`` set from ``invalidation``; every other field comes from
    the fresh read.
    """
    if target is None:
        raise ValueError(
            f"replace: invalidation target id={target_id!s} does not exist or "
            "is buried (lost a concurrent race)"
        )
    if successor_type is not Fact:
        raise ValueError(
            f"replace: invalidation target id={target_id!s} named, but the "
            f"successor is a {successor_type.__name__}, not a Fact — only a "
            "Fact successor may invalidate a Decision"
        )
    if not isinstance(target, Decision):
        raise ValueError(
            f"replace: invalidation target id={target_id!s} is a "
            f"{type(target).__name__}, not a Decision"
        )
    if target_id in flip_ids:
        raise ValueError(
            f"replace: invalidation target id={target_id!s} is also named by a "
            "predecessor flip in this same call — invalidation draws no edge, "
            "so a target can never be a flip predecessor too"
        )
    if target.lifecycle != "in_force":
        raise ValueError(
            f"replace: invalidation target id={target_id!s} is not in_force "
            f"(lifecycle={target.lifecycle!r}) — only an in_force Decision may "
            "be invalidated"
        )
    if target.validity != "valid":
        raise ValueError(
            f"replace: invalidation target id={target_id!s} is not valid "
            f"(validity={target.validity!r}) — only a valid Decision may be "
            "invalidated"
        )

    return Decision.model_validate(
        {
            **target.model_dump(mode="python"),
            "validity": "invalid",
            "invalidated_by": str(successor_id),
            "invalidation_reason": invalidation.reason,
        }
    )


def resolve_unsupersede_write(
    *,
    predecessor: Fact | Decision | None,
    predecessor_id: HypomnemaId,
    successor_id: HypomnemaId,
    edge_id: str,
    at: datetime,
) -> Fact | Decision:
    """Validate an undo (``StorageAdapter.restore``) against a fresh predecessor read.

    ``predecessor`` must be a fresh read taken inside the caller's lock; ``None``
    means no live row. Only the predecessor is read: the reverse direction writes
    only its row, and the successor's eligibility is the engine's check.

    Raises ``ValueError`` naming ``predecessor_id`` when the predecessor does not
    exist, its current ``superseded_by`` is not ``successor_id``, or its current
    ``supersession_edge_id`` is not ``edge_id`` (a later generation of the edge, or
    a concurrent second restore won).

    The result clears ``superseded_by``, ``supersession_reason``,
    ``supersession_note`` and ``supersession_edge_id``, sets ``updated_at=at``, and
    reverses the arm change (Fact ``validity="valid"``; Decision
    ``lifecycle="in_force"``, ``validity`` as it was). It is layered onto the
    fresh read via ``model_validate``.
    """
    if predecessor is None:
        raise ValueError(
            f"restore: predecessor id={predecessor_id!s} does not exist or "
            "is buried (lost a concurrent race)"
        )
    if predecessor.superseded_by != successor_id:
        raise ValueError(
            f"restore: predecessor id={predecessor_id!s} superseded_by="
            f"{predecessor.superseded_by!s} does not match the expected "
            f"successor_id={successor_id!s} — lost a concurrent race, or a "
            "later generation of this predecessor's edge"
        )
    if predecessor.supersession_edge_id != edge_id:
        raise ValueError(
            f"restore: predecessor id={predecessor_id!s} supersession_edge_id="
            f"{predecessor.supersession_edge_id!r} does not match the "
            f"expected edge_id={edge_id!r} — a later generation of this "
            "predecessor's edge, or a concurrent second restore already "
            "won the race"
        )

    flip_fields: dict[str, object] = {
        "superseded_by": None,
        "supersession_reason": None,
        "supersession_note": None,
        "supersession_edge_id": None,
        "updated_at": at,
    }
    if isinstance(predecessor, Fact):
        flip_fields["validity"] = "valid"
    else:
        assert isinstance(predecessor, Decision)
        flip_fields["lifecycle"] = "in_force"

    return type(predecessor).model_validate(
        {**predecessor.model_dump(mode="python"), **flip_fields}
    )


def resolve_unsupersede_append(
    *, existing: UnsupersedeEvent, event: UnsupersedeEvent
) -> UnsupersedeEvent:
    """Decide what ``append_unsupersede`` returns when a row already exists for ``event.edge_id``.

    ``existing`` is a retry of the same undo only when predecessor, successor and
    arm all match ``event``; otherwise raises ``ValueError`` rather than keep a row
    that would leave the caller's undo without a findable event.

    A retry whose ``existing.refused_at`` is set comes back with ``refused_at``
    cleared: a fresh live ``unsupersede`` reopens a refused edge. The caller
    compares ``refused_at`` on the result with ``existing`` to decide whether a
    write is needed.
    """
    if (
        existing.predecessor_id != event.predecessor_id
        or existing.successor_id != event.successor_id
        or existing.arm != event.arm
    ):
        raise ValueError(
            f"append_unsupersede: edge_id={event.edge_id!r} is already recorded "
            f"for predecessor={existing.predecessor_id!s} "
            f"successor={existing.successor_id!s} arm={existing.arm!r} — refusing "
            f"to reuse it for predecessor={event.predecessor_id!s} "
            f"successor={event.successor_id!s} arm={event.arm!r}"
        )
    if existing.refused_at is not None:
        return existing.model_copy(update={"refused_at": None})
    return existing


UnsupersedeRepairAction = Literal["skip", "restore", "plant_then_restore", "withdraw"]


@dataclass(frozen=True, slots=True)
class RepairDecision:
    """The action ``resolve_unsupersede_repair`` chose for one undo event, and a warning to log."""

    action: UnsupersedeRepairAction
    warning: str | None


def resolve_unsupersede_repair(
    *,
    event: UnsupersedeEvent,
    predecessor: Fact | Decision | None,
    successor: Fact | Decision | None,
    tombstone: Tombstone | None,
    refused_successors_this_pass: frozenset[HypomnemaId] = frozenset(),
) -> RepairDecision:
    """Decide the next repair action for one undo event from fresh reads.

    ``predecessor`` and ``successor`` are ``None`` when missing or buried.
    ``tombstone`` is the successor's current tombstone, if any.
    ``refused_successors_this_pass`` holds successor ids already withdrawn earlier
    in the same repair pass: ``replace`` lets one successor gain several
    predecessors, and without this set a later event against the same successor
    would see "no tombstone" and delete the content that the earlier event just
    protected.

    A successor read is needed because a failed ``plant_tombstone`` / ``restore``
    leaves the server alive, and the successor can be superseded or revoked before
    a retry. The successor's row carries no marker of a pending undo.

    A set ``event.refused_at`` short-circuits before every other successor,
    lifecycle, tombstone or hash check: ``"withdraw"`` if a tombstone naming this
    predecessor is still present (crash recovery), else ``"skip"``. Only a fresh
    live ``append_unsupersede`` clears it.

    Returns a ``RepairDecision`` of ``(action, warning)``:

    - ``"skip"``: nothing to do. The predecessor is gone; its current edge is not
      this event's (a later generation); the event's ``arm`` disagrees with the
      predecessor's type; the event is durably refused with no tombstone left;
      the successor is in ``refused_successors_this_pass``; the successor is a
      Decision no longer ``in_force`` (the same ``lifecycle`` check also rejects
      a Fact id, since the two arms' ``lifecycle`` literals are disjoint); the
      successor is gone with no tombstone naming this predecessor; the tombstone
      carries no ``restores`` (a legacy tombstone never authorises a restore); or
      it names a different predecessor.
    - ``"restore"``: call ``StorageAdapter.restore``. The Fact arm always lands
      here. The Decision arm lands here when the successor is ``in_force`` and a
      matching tombstone already exists.
    - ``"plant_then_restore"``: Decision arm, not refused, successor ``in_force``,
      no tombstone yet: plant one (``restores=predecessor.id``) before restoring.
    - ``"withdraw"``: Decision arm. Either (1) a fresh refusal: the tombstone
      names this predecessor but its ``restores_content_hash`` no longer matches
      ``content_hash(successor.content)`` (the successor was edited while the undo
      was pending, or the tombstone has no hash); the caller calls
      ``mark_unsupersede_refused`` first, then ``withdraw_tombstone``; or (2)
      crash recovery: the event is already refused and its tombstone remains, so
      the caller only withdraws. The caller tells them apart by
      ``event.refused_at``.

    ``warning`` is set when the edge id matches but ``superseded_by`` differs from
    the event's ``successor_id``, the arm disagrees, the tombstone names another
    predecessor, or the hash mismatch fires (an edit raced the undo; not
    corruption, but worth an operator's attention). The same-pass skip also carries
    a warning. Crash-recovery ``"withdraw"`` carries none. The caller logs it.
    """
    if predecessor is None:
        return RepairDecision("skip", None)
    if predecessor.supersession_edge_id != event.edge_id:
        return RepairDecision("skip", None)
    if predecessor.superseded_by != event.successor_id:
        return RepairDecision(
            "skip",
            (
                f"unsupersede_repair: predecessor id={predecessor.id!s} edge_id="
                f"{event.edge_id!r} matches this undo event, but its current "
                f"superseded_by={predecessor.superseded_by!s} does not match "
                f"the event's successor_id={event.successor_id!s} — skipping, "
                "data inconsistency suspected"
            ),
        )
    predecessor_arm: Kind = (
        "fact" if isinstance(predecessor, Fact) else "decision"
    )
    if event.arm != predecessor_arm:
        return RepairDecision(
            "skip",
            (
                f"unsupersede_repair: event edge_id={event.edge_id!r} declares "
                f"arm={event.arm!r} but predecessor id={predecessor.id!s} is a "
                f"{type(predecessor).__name__} — skipping, data inconsistency "
                "suspected"
            ),
        )
    if predecessor_arm == "fact":
        return RepairDecision("restore", None)

    assert isinstance(predecessor, Decision)
    if event.refused_at is not None:
        if tombstone is not None and tombstone.restores == predecessor.id:
            return RepairDecision("withdraw", None)
        return RepairDecision("skip", None)
    if event.successor_id in refused_successors_this_pass:
        return RepairDecision(
            "skip",
            (
                f"unsupersede_repair: successor id={event.successor_id!s} was "
                f"already refused earlier in this repair pass — skipping "
                f"predecessor id={predecessor.id!s}'s edge_id={event.edge_id!r} "
                "rather than re-deriving an action from its now-stale "
                "tombstone state"
            ),
        )
    if successor is None:
        return RepairDecision("skip", None)
    if successor.lifecycle != "in_force":
        return RepairDecision("skip", None)
    if tombstone is None:
        return RepairDecision("plant_then_restore", None)
    if tombstone.restores is None:
        return RepairDecision("skip", None)
    if tombstone.restores != predecessor.id:
        return RepairDecision(
            "skip",
            (
                f"unsupersede_repair: successor id={event.successor_id!s}'s "
                f"tombstone restores={tombstone.restores!s} does not match "
                f"predecessor id={predecessor.id!s} — skipping, data "
                "inconsistency suspected"
            ),
        )
    if tombstone.restores_content_hash != content_hash(successor.content):
        return RepairDecision(
            "withdraw",
            (
                f"unsupersede_repair: successor id={event.successor_id!s}'s "
                f"content changed while predecessor id={predecessor.id!s}'s "
                f"undo (edge_id={event.edge_id!r}) was pending — refusing the "
                "restore and withdrawing the stale tombstone; the edit wins"
            ),
        )
    return RepairDecision("restore", None)


class RepairIO(Protocol):
    """What ``run_unsupersede_repair`` needs from an adapter: reads and the four repair writes."""

    async def list_events(self) -> list[UnsupersedeEvent]: ...

    async def load_record(self, id: HypomnemaId) -> Fact | Decision | None:
        """The live record, or ``None`` when it is missing or buried."""
        ...

    async def load_tombstone(self, buried_id: HypomnemaId) -> Tombstone | None: ...

    async def mark_refused(self, edge_id: str) -> None: ...

    async def withdraw_tombstone(self, buried_id: HypomnemaId) -> None: ...

    async def plant_tombstone(
        self, buried_id: HypomnemaId, reason: str, *, restores: HypomnemaId
    ) -> None: ...

    async def restore(self, event: UnsupersedeEvent) -> None: ...


async def run_unsupersede_repair(io: RepairIO) -> RepairReport:
    """Replay every recorded undo event once and count how each one ended.

    Each event is decided by ``resolve_unsupersede_repair`` from fresh reads, then
    skipped, withdrawn (``refused``) or restored (``completed``, planting the
    tombstone first when the decision says so). A successor withdrawn earlier in
    the pass is passed back into later decisions. The events are the snapshot
    ``io.list_events`` returns.
    """
    completed = 0
    skipped = 0
    refused = 0
    refused_successors_this_pass: set[HypomnemaId] = set()
    for event in await io.list_events():
        predecessor = await io.load_record(event.predecessor_id)
        successor = await io.load_record(event.successor_id)
        tombstone = await io.load_tombstone(event.successor_id)
        decision = resolve_unsupersede_repair(
            event=event,
            predecessor=predecessor,
            successor=successor,
            tombstone=tombstone,
            refused_successors_this_pass=frozenset(refused_successors_this_pass),
        )
        if decision.warning is not None:
            _log.warning(decision.warning)

        if decision.action == "skip":
            skipped += 1
            continue

        if decision.action == "withdraw":
            if event.refused_at is None:
                await io.mark_refused(event.edge_id)
            await io.withdraw_tombstone(event.successor_id)
            refused_successors_this_pass.add(event.successor_id)
            refused += 1
            continue

        if decision.action == "plant_then_restore":
            assert predecessor is not None
            await io.plant_tombstone(
                event.successor_id,
                f"unsupersede: {event.reason}",
                restores=predecessor.id,
            )
        await io.restore(event)
        completed += 1

    return RepairReport(completed=completed, skipped=skipped, refused=refused)


def resolve_withdraw_tombstone(
    *,
    buried_id: HypomnemaId,
    tombstone: Tombstone | None,
    buried: Fact | Decision | None,
) -> None:
    """Validate a ``withdraw_tombstone`` call against fresh reads of the tombstone and row.

    Raises ``KeyError`` (nothing written) unless all hold: a tombstone exists for
    ``buried_id``; it carries a real ``restores`` (never a legacy tombstone from
    ordinary ``bury()``); and ``buried`` (a fresh ``get_by_id``, ``None`` on
    ``KeyError``) is present and, for a Decision, still ``in_force``. A completed
    burial is permanent and withdrawing its marker cannot undo it.
    """
    if tombstone is None or tombstone.restores is None:
        raise KeyError(
            f"withdraw_tombstone: id={buried_id!s} carries no pending-undo "
            "tombstone to withdraw"
        )
    if buried is None:
        raise KeyError(
            f"withdraw_tombstone: id={buried_id!s}'s burial is already "
            "complete — a completed undo's tombstone is permanent history, "
            "never withdrawable"
        )
    if isinstance(buried, Decision) and buried.lifecycle != "in_force":
        raise KeyError(
            f"withdraw_tombstone: id={buried_id!s} is no longer in_force "
            f"(lifecycle={buried.lifecycle!r}) — refusing to withdraw its "
            "tombstone"
        )
