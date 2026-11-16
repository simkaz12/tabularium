"""Typed request and collaborator values that the save path threads between its helpers.

``facts.save_internal`` and ``decisions.save_decision`` take flat keyword parameters. After
their validation they build one request value and one ``SaveDeps``, and every private helper
below them takes those two instead of a long parameter list. Constructing any value here does
no validation; the entry functions have already validated every field.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import (
    Asserter,
    HypomnemaId,
    Kind,
    PromotedFromSnapshot,
    SaveLifecycle,
    Scope,
    SupersedesSnapshot,
    SupersessionReason,
    Valence,
)
from tabularium.engine.backfill import BackfillSignals
from tabularium.results import BounceResolution
from tabularium.storage.types import StorageAdapter


@dataclass(frozen=True, slots=True)
class Supersession:
    """Predecessors a save names, with the ``reason`` and ``note`` that apply to each.

    ``supersedes`` holds at least one id when the value is built by ``supersession_from``, and
    ``reason`` is always set, so a reader never has to check the pairing. A save that names no
    predecessor carries ``None`` instead of a ``Supersession``.
    """

    supersedes: tuple[HypomnemaId, ...]
    reason: SupersessionReason
    note: str | None


def supersession_from(
    supersedes: Sequence[HypomnemaId] | None,
    reason: SupersessionReason | None,
    note: str | None,
) -> Supersession | None:
    """Pair the predecessors with their reason, or return ``None`` when none are named.

    Call it after ``check_supersedes_reason`` accepted the arguments. That check rejects both a
    non-empty ``supersedes`` without a ``reason`` and a ``reason`` without ``supersedes``, so
    on accepted input the two are either both present or both absent. ``note`` is dropped
    together with an empty ``supersedes``, as no edge exists to carry it.
    """
    if not supersedes or reason is None:
        return None

    return Supersession(supersedes=tuple(supersedes), reason=reason, note=note)


def superseded_ids(supersession: Supersession | None) -> set[HypomnemaId]:
    """The predecessor ids as a fresh set; empty when ``supersession`` is ``None``."""
    return set(supersession.supersedes) if supersession is not None else set()


@dataclass(frozen=True, slots=True)
class ResolveRequest:
    """What the caller said about an earlier bounce.

    ``resolve`` is the ``BounceResolution`` the caller supplied, if any. ``bounce_id`` is the
    prior bounce that a ``supersedes`` or ``promoted_from`` replay names; it is distinct from
    the id of the bounce the current save may build, which lives on the request as
    ``bounce_id``.
    """

    resolve: BounceResolution | None
    bounce_id: str | None

    @property
    def exclude_ids(self) -> set[HypomnemaId]:
        """The resolve candidate as a fresh set; empty without a ``resolve``."""
        return {self.resolve.candidate_id} if self.resolve is not None else set()


@dataclass(frozen=True, slots=True)
class FactSaveRequest:
    """One Fact save, after its entry function validated it.

    ``scope`` is the validated scope. ``bounce_id`` is the id of any bounce this save builds.
    ``lifecycle``, ``valence``, ``asserted_by``, ``roles`` and ``provisional`` are the ember
    inputs; a plain ``save`` leaves them at their defaults.
    """

    kind: ClassVar[Kind] = "fact"

    content: str
    scope: Scope
    source: str | None
    purpose: str | None
    audience_hint: list[str] | None
    lifecycle: SaveLifecycle
    valence: Valence | None
    asserted_by: Asserter
    roles: list[str] | None
    provisional: bool
    supersession: Supersession | None
    resolution: ResolveRequest
    bounce_id: str

    @property
    def supersede_ids(self) -> set[HypomnemaId]:
        """The named predecessors as a fresh set."""
        return superseded_ids(self.supersession)


@dataclass(frozen=True, slots=True)
class DecisionSaveRequest:
    """One Decision save, after its entry function validated it.

    ``supersession`` names at most one predecessor. ``supersession_chain`` and
    ``promoted_snapshots`` are the lineage snapshots the unlocked pre-embed validation took.
    ``promoted_from`` is the caller's list of Fact ids as given. ``bounce_id`` is the id of any
    bounce this save builds.
    """

    kind: ClassVar[Kind] = "decision"

    content: str
    scope: Scope
    source: str | None
    purpose: str | None
    audience_hint: list[str] | None
    supersession: Supersession | None
    supersession_chain: tuple[SupersedesSnapshot, ...]
    promoted_snapshots: tuple[PromotedFromSnapshot, ...]
    promoted_from: list[HypomnemaId] | None
    resolution: ResolveRequest
    bounce_id: str

    @property
    def supersede_ids(self) -> set[HypomnemaId]:
        """The named predecessor as a fresh set."""
        return superseded_ids(self.supersession)


SaveRequest = FactSaveRequest | DecisionSaveRequest


@dataclass(frozen=True, slots=True)
class SaveDeps:
    """The collaborators the save helpers use, fixed for one save call.

    Built from the entry function's parameters on every call, so a ``cfg`` swapped by
    ``reconfigure()`` between two saves reaches the second one.
    """

    adapter: StorageAdapter
    clock: Clock
    cfg: EngineConfig
    lock: asyncio.Lock
    backfill: BackfillSignals
