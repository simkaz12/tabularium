"""The ``Tabularium`` engine: all memory semantics above the storage seam.

The storage adapter carries mechanisms and the embedder carries provider routing; the MCP server
calls into this class.

Dedup gate
~~~~~~~~~~
Every ``save``/``ember`` runs the dedup gate before writing. The best candidate is classified
and dispatched:

  valid active/hot Fact  bounce (guard trip or near_dup). An exact content-key match is caught
                         earlier, at admission, before the probe runs.
  valid ember            recurrence: reinforce with ``source="dedup"``, then split the count by
                         ``asserted_by``. An owner-asserted event crossing
                         ``escalation_threshold`` surfaces an escalation suggestion naming both
                         exits (codify, foreground); an agent-asserted event only keeps the
                         ember warm.
  valid archived Fact    revival: in-place ``archived`` to ``active`` via ``adapter.transition``
                         plus reinforcement (``source="revival"``).
  Decision               bounce eagerly (lower ``decision_gray_floor``); never auto-reinforce.
  guard trip             downgrade to a bounce and surface the differing token.
  invalid (either arm)   non-blocking: the valid save always writes, the new record carries
                         ``reconciles=<invalid_id>``, outcome ``reconciled``.

``save(resolve=BounceResolution(...))`` re-runs the gate. If the named candidate is still the best
match, ``confirm_distinct`` writes and calls ``mark_distinct`` (idempotent) and ``reinforce``
reinforces the candidate. If it is no longer the best match the result is a fresh bounce. An
unknown or buried id with ``action="reinforce"`` is a ``ToolError``.

Reinforcement and projections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reinforcement appends a ``ReinforcementEvent`` and recomputes projections through
``update_projections``; the formulas live above the seam in ``engine.projections`` and the
adapter stores the values verbatim. A fallback-pass recall hit is reinforced with
``source="revival"`` and moves to ``active`` only when importance reaches ``revival_threshold``;
an explicit ``include_archived=True`` read never reinforces or revives.

Registries
~~~~~~~~~~
``purpose`` must be a registered facet value (``None`` is allowed) and ``audience_hint`` entries
must be registered roles. ``initialize()`` registers the five purpose values and the four roles
``decider``, ``builder``, ``reviewer`` and ``navigator``, and deletes the retired ``default``
role, so a store that still holds one loses it at the next start. ``context()`` requires ``role``;
a caller supplying none gets a ``ToolError`` naming the registered roles.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from datetime import datetime, timedelta

from tabularium.clock import Clock, SystemClock
from tabularium.config import EngineConfig
from tabularium.domain import (
    Asserter,
    Fact,
    HypomnemaId,
    Kind,
    Scope,
    SupersessionReason,
    Valence,
)
from tabularium.embedding import Embedder
from tabularium.engine._errors import ToolError
from tabularium.engine._types import ALL_ROLES
from tabularium.engine.assessor import Assessor
from tabularium.engine.backfill import Backfiller
from tabularium.engine.clean import CleanSummary
from tabularium.engine.context import SelectedTier, pack, select_default
from tabularium.engine.decision_read import decision_view, decisions_query
from tabularium.engine.decisions import save_decision
from tabularium.engine.dispositions import (
    bury,
    revoke,
    set_audience,
    set_hot,
    set_shelved,
    set_validity,
    supersede,
    unsupersede,
)
from tabularium.engine.embers import save_ember
from tabularium.engine.engine_context import EngineContext
from tabularium.engine.facts import save_internal
from tabularium.engine.feedback import feedback
from tabularium.engine.filters import GLOBAL, validate_and_resolve_scope
from tabularium.engine.recall import recall
from tabularium.engine.report import load_context_summary, record_pass_failure
from tabularium.engine.report import run_clean_pass as _run_clean_pass
from tabularium.engine.startup import run_startup_passes
from tabularium.results import (
    BounceResolution,
    BuryResult,
    ContextPayload,
    DecisionLink,
    DecisionView,
    FeedbackResult,
    RecallResult,
    ReviseResult,
    SaveResult,
    SupersedeResult,
    UnsupersedeResult,
)
from tabularium.storage.types import (
    StorageAdapter,
)
from tabularium.tokens import LitellmTokenCounter, TokenCounter

_log = logging.getLogger("tabularium")

_DEFAULT_PURPOSES = ("constraint", "convention", "preference", "pitfall", "reference")
# Roles seeded by initialize(). ``navigator`` receives no payload of its own, but routing is
# subtractive, so a scout can only be kept away from an ember if it is nameable in some other
# atom's audience_hint. ``ALL_ROLES`` in ``engine._types`` is the single tuple shared with
# ``engine.embers.save_ember``, whose creation-time default must never be a live
# ``list_roles()`` snapshot.

# Removed by initialize() on every call.
_RETIRED_DEFAULT_ROLE = "default"


class Tabularium:
    """The Tabularium engine: pydantic in, pydantic out.

    Constructor args:
      adapter       a ``StorageAdapter``.
      embedder      an ``Embedder``.
      clock         a ``Clock``; ``SystemClock`` when omitted.
      config        ``EngineConfig``; defaults when omitted.
      token_counter a ``TokenCounter`` for ``context()`` packing. When omitted, a
                    ``LitellmTokenCounter`` is built on first use inside ``context()``, never at
                    construction.
      assessor      optional ``Assessor`` that both save arms pass to
                    ``admission.relation_verdicts`` on a bounce; see ``engine.admission`` for
                    how an unavailable assessor surfaces.

    Await ``initialize()`` after construction to seed the registries::

        engine = Tabularium(adapter, embedder, clock, config)
        await engine.initialize()
    """

    def __init__(
        self,
        adapter: StorageAdapter,
        embedder: Embedder,
        clock: Clock | None = None,
        config: EngineConfig | None = None,
        *,
        token_counter: TokenCounter | None = None,
        assessor: Assessor | None = None,
    ) -> None:
        self._adapter = adapter
        self._embedder = embedder
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._cfg = config if config is not None else EngineConfig()
        self._assessor: Assessor | None = assessor
        self._token_counter: TokenCounter | None = token_counter
        # Per-scope in-flight Cleaner pass. It is the single-flight gate (a second trigger for a
        # scope already present is a no-op) and the strong reference that keeps the task from
        # being garbage collected; a done-callback in ``_run_single_flight_pass`` pops the entry.
        self._cleaner_tasks: dict[str, asyncio.Task[CleanSummary | None]] = {}
        self._nightly_task: asyncio.Task[None] | None = None
        self._next_nightly: datetime | None = None
        # The single global write lock. One process and one event loop mean writes only race
        # between ``await`` points. It is one lock rather than per-record locks because write
        # volume is low and a global lock is trivially correct. Held from the dedup candidate
        # read through the terminal write of each arm, and never across an embed call.
        self._write_lock: asyncio.Lock = asyncio.Lock()
        # One Backfiller per engine; ``initialize()`` sets ``dirty`` before the startup sweep.
        self._backfiller: Backfiller = Backfiller(adapter, embedder, self._write_lock)

    async def initialize(self) -> None:
        """Seed the global scope, five purpose facets and four roles; delete the ``default`` role.

        Idempotent. ``create_role`` is a no-op on a registered name, and ``delete_role`` is a
        no-op when the role is absent, so no existence check precedes it. Then runs
        ``engine.startup.run_startup_passes`` (content keys, edge ids, undo repair, verdict
        reconciliation) before returning.

        With ``cfg.read_only`` it returns at once: no registry write, no startup pass, no
        backfill and no nightly scheduler, whatever ``enable_digest_cleaner`` says. The store
        must already hold the seeded registries.
        """
        if self._cfg.read_only:
            return

        await self._adapter.create_scope(GLOBAL)
        for purpose in _DEFAULT_PURPOSES:
            await self._adapter.create_facet(purpose)
        for role in ALL_ROLES:
            await self._adapter.create_role(role)
        await self._adapter.delete_role(_RETIRED_DEFAULT_ROLE)

        await run_startup_passes(self._adapter, _log, lock=self._write_lock, clock=self._clock)

        self._backfiller.dirty = True
        self._backfiller.schedule(trigger="startup")
        self._start_nightly_scheduler()

    @property
    def pending_backfill(self) -> Awaitable[None]:
        """Await this to block until no backfill pass is running and no rerun is pending."""
        return self._backfiller.pending()


    async def save(
        self,
        content: str,
        *,
        kind: Kind = "fact",
        scope: str,
        source: str | None = None,
        purpose: str | None = None,
        audience_hint: list[str] | None = None,
        resolve: BounceResolution | None = None,
        supersedes: HypomnemaId | list[HypomnemaId] | None = None,
        promoted_from: list[HypomnemaId] | None = None,
        reason: SupersessionReason | None = None,
        note: str | None = None,
        bounce_id: str | None = None,
    ) -> SaveResult:
        """Save a Fact or Decision.

        Args:
            content:        Record content, non-empty.
            kind:           ``"fact"`` (default) or ``"decision"``.
            scope:          Target scope, registered.
            source:         Optional provenance string.
            purpose:        Registered facet value or None.
            audience_hint:  Registered role names.
            resolve:        Bounce resolution adjudication.
            supersedes:     A list of Fact ids when ``kind="fact"``; a single in-force Decision id
                            when ``kind="decision"``. The other shape is rejected.
            promoted_from:  Ember Fact ids whose snapshots are frozen into this Decision;
                            requires ``kind="decision"``.
            reason:         Required when ``supersedes`` is non-empty and forbidden otherwise:
                            ``"correction"``, ``"changed"`` or ``"elaborated"`` for a same-scope
                            predecessor, ``"moved"`` for a cross-scope one. A multi-id
                            ``supersedes`` must be all same-scope or all cross-scope relative to
                            ``scope``; a mixed list is refused before the embedder runs.
            note:           Optional free text stored beside ``reason`` on every predecessor's
                            row in the same transaction.
            bounce_id:      The ``bounce_id`` of a prior bounce, copied when replaying its
                            ``supersede`` or ``promoted_from`` exit (``reinforce`` and
                            ``confirm_distinct`` carry theirs on ``resolve``). Omitting it
                            leaves that bounce's verdict rows ``unanswered``.

        Returns:
            ``SaveResult`` with outcome ``saved | reinforced | revived | reconciled | bounced |
            already_saved``. On the Fact arm ``supersedes`` is all-or-nothing: any stale
            predecessor at write time fails the whole call and nothing lands.

        Raises:
            ToolError: kind not in {"fact", "decision"}; unregistered scope, purpose or role;
                       unknown or buried id with ``resolve.action="reinforce"``; a list
                       ``supersedes`` with kind="decision" or a bare id with kind="fact";
                       ``promoted_from`` with kind="fact" or containing a non-ember id;
                       ``supersedes`` naming a non-in-force or non-Decision id (Decision arm);
                       ``reason`` missing or present without a matching ``supersedes``; a
                       mixed-scope ``supersedes`` list; a ``reason`` class illegal for the scope
                       relation; any stale predecessor at ``replace``'s fresh check.
            ValueError: empty content.
        """
        # Validate kind first: ``Literal`` is not enforced at runtime for in-process callers, and
        # the supersedes/promoted_from guards below would misreport a bad kind.
        if kind not in ("fact", "decision"):
            raise ToolError(
                f"Invalid kind {kind!r}: must be 'fact' or 'decision'"
            )

        if promoted_from is not None and kind != "decision":
            raise ToolError(
                "promoted_from requires kind='decision' (SPEC §9.1)"
            )

        if kind == "decision":
            if isinstance(supersedes, list):
                raise ToolError(
                    "supersedes: kind='decision' takes a single id, not a "
                    "list — a list is the Fact arm's shape "
                    "(save(kind='fact', supersedes=[...]))"
                )
            return await save_decision(
                content=content,
                scope=scope,
                source=source,
                purpose=purpose,
                audience_hint=audience_hint,
                supersedes=supersedes,
                promoted_from=promoted_from,
                resolve=resolve,
                adapter=self._adapter,
                embedder=self._embedder,
                clock=self._clock,
                cfg=self._cfg,
                lock=self._write_lock,
                require_scope=self._require_scope,
                require_purpose=self._require_purpose,
                require_roles=self._require_roles,
                reason=reason,
                note=note,
                backfill=self._backfiller,
                assessor=self._assessor,
                resolve_bounce_id=bounce_id,
            )

        if supersedes is not None and not isinstance(supersedes, list):
            raise ToolError(
                "supersedes: kind='fact' requires a LIST of ids, e.g. "
                "supersedes=[<id>] — a bare id is the Decision arm's shape "
                "(save(kind='decision', supersedes=<id>))"
            )
        return await save_internal(
            content=content,
            scope=scope,
            source=source,
            purpose=purpose,
            audience_hint=audience_hint,
            lifecycle="active",
            valence=None,
            resolve=resolve,
            supersedes=supersedes,
            adapter=self._adapter,
            embedder=self._embedder,
            clock=self._clock,
            cfg=self._cfg,
            lock=self._write_lock,
            require_scope=self._require_scope,
            require_purpose=self._require_purpose,
            require_roles=self._require_roles,
            reason=reason,
            note=note,
            backfill=self._backfiller,
            assessor=self._assessor,
            resolve_bounce_id=bounce_id,
        )

    async def ember(
        self,
        content: str,
        *,
        scope: str,
        valence: Valence,
        asserted_by: Asserter = "agent",
        source: str | None = None,
        purpose: str | None = None,
        audience_hint: list[str] | None = None,
        roles: list[str] | None = None,
    ) -> SaveResult:
        """Save an ember Fact: lifecycle ``ember``, ``valence`` set, same gate as ``save``.

        Args:
            content:       Record content.
            scope:         Target scope, registered.
            valence:       ``"correction"`` or ``"guideline"``.
            asserted_by:   ``"owner"`` or ``"agent"`` (default). Recorded, never verified. Only an
                           owner-asserted reinforcement advances ``owner_recurrence_count`` and
                           can cross ``escalation_threshold``.
            source:        Optional provenance string.
            purpose:       Registered facet value or None.
            audience_hint: Registered roles, used only when this call creates a new ember. A falsy
                           value (omitted or ``[]``) with ``asserted_by="owner"`` raises. With
                           ``asserted_by="agent"`` it defaults to the four canonical roles and
                           the Fact is marked ``provisional``. A non-empty hint is honoured
                           verbatim and never provisional.
            roles:         Registered roles that retag an existing ember's ``audience_hint`` when
                           this call's gate reinforces it (overwrites, never unions); ``None``
                           leaves the stored hint. No effect on a new write.

        Returns:
            ``SaveResult`` as for ``save``. ``escalation_suggestion`` is set on a ``reinforced``
            outcome exactly when this call crossed the escalation threshold.

        Raises:
            ToolError: unregistered scope, purpose, or role in ``audience_hint``/``roles``;
                       ``asserted_by="owner"`` with a falsy ``audience_hint``.
        """
        return await save_ember(
            content=content,
            scope=scope,
            valence=valence,
            asserted_by=asserted_by,
            source=source,
            purpose=purpose,
            audience_hint=audience_hint,
            roles=roles,
            adapter=self._adapter,
            embedder=self._embedder,
            clock=self._clock,
            cfg=self._cfg,
            lock=self._write_lock,
            require_scope=self._require_scope,
            require_purpose=self._require_purpose,
            require_roles=self._require_roles,
            backfill=self._backfiller,
            assessor=self._assessor,
        )


    async def decision(self, id: HypomnemaId) -> DecisionView:
        """Return the full view of a single Decision.

        Raises:
            ToolError: unknown or buried id; id names a Fact.
        """
        return await decision_view(id=id, adapter=self._adapter, clock=self._clock)


    async def decisions_query(
        self,
        query: str,
        *,
        scope: str,
        limit: int = 10,
        include_superseded: bool = False,
        include_invalid: bool = False,
    ) -> list[DecisionLink]:
        """Hybrid retrieval restricted to the Decision arm.

        ``include_superseded`` widens the lifecycle filter from ``in_force`` to include
        ``superseded`` and ``revoked``; ``include_invalid`` widens validity to include invalidated
        Decisions, as ``recall`` does.

        Returns:
            ``DecisionLink`` triage objects sorted by RRF score with the importance boost.

        Raises:
            ToolError: unregistered scope; empty query.
        """
        return await decisions_query(
            query=query,
            scope=scope,
            limit=limit,
            include_superseded=include_superseded,
            include_invalid=include_invalid,
            adapter=self._adapter,
            embedder=self._embedder,
            clock=self._clock,
            cfg=self._cfg,
            require_scope=self._require_scope,
        )


    async def set_validity(
        self, id: HypomnemaId, valid: bool, reason: str
    ) -> ReviseResult:
        """Flip the validity bit on a Fact or Decision.

        Returns:
            ``ReviseResult`` with outcome ``"unchanged"`` (same value again) or ``"revised"``.

        Raises:
            ToolError: unknown or buried id.
        """
        return await set_validity(
            id=id,
            valid=valid,
            reason=reason,
            adapter=self._adapter,
            clock=self._clock,
            lock=self._write_lock,
        )

    async def bury(self, id: HypomnemaId, reason: str) -> BuryResult:
        """Bury a Fact or Decision; the one destructive tool.

        Returns:
            ``BuryResult`` with outcome ``"buried"`` or ``"already_buried"``.

        Raises:
            ToolError: unknown id; empty ``reason``.
        """
        return await bury(
            id=id,
            reason=reason,
            adapter=self._adapter,
            clock=self._clock,
            lock=self._write_lock,
        )

    async def revoke(self, id: HypomnemaId, reason: str) -> ReviseResult:
        """Revoke an in-force Decision.

        Returns:
            ``ReviseResult`` with outcome ``"unchanged"`` (already revoked) or ``"revised"``.

        Raises:
            ToolError: unknown or buried id; id names a Fact; the Decision is already
                       ``superseded``.
        """
        return await revoke(
            id=id,
            reason=reason,
            adapter=self._adapter,
            clock=self._clock,
            lock=self._write_lock,
        )

    async def supersede(
        self,
        successor_id: HypomnemaId,
        predecessor_ids: list[HypomnemaId],
        reason: SupersessionReason,
        note: str | None = None,
    ) -> SupersedeResult:
        """Draw a forward pointer from existing Facts to an existing Fact in one transaction.

        Facts only.

        This is how to finish the "split by scope" continuation that ``save``'s mixed-scope
        refusal prescribes: save once with the same-scope group, then attach the cross-scope group
        with ``reason="moved"``. The ``tabularium supersede`` CLI subcommand calls
        ``engine.dispositions.supersede`` directly and needs no embedder.

        The successor must be valid and unretired, rechecked fresh by ``adapter.replace`` under
        the same lock hold as every predecessor check. The call is all-or-nothing: any stale
        predecessor at write time fails the whole call.

        Args:
            successor_id:     The Fact id every predecessor will point at.
            predecessor_ids:  The Fact ids being superseded.
            reason:           Required; the class and scope rules of ``save(supersedes=...)``.
            note:             Optional free text stored beside ``reason`` on every predecessor.

        Returns:
            ``SupersedeResult`` with the successor and one ``(predecessor_id, edge_id)`` pair per
            predecessor.

        Raises:
            ToolError: ``predecessor_ids`` empty; a self-edge; unknown or buried successor or
                       predecessor; either side is a Decision; the reason and scope checks
                       ``save`` also enforces; the successor invalid or retired, or any
                       predecessor stale, at ``replace``'s fresh check.
        """
        return await supersede(
            successor_id=successor_id,
            predecessor_ids=predecessor_ids,
            reason=reason,
            note=note,
            adapter=self._adapter,
            clock=self._clock,
            lock=self._write_lock,
        )

    async def unsupersede(self, id: HypomnemaId, reason: str) -> UnsupersedeResult:
        """Undo a supersession on either arm, dispatching on the predecessor's own kind.

        Fact arm: clears ``superseded_by`` and restores ``validity="valid"``; both rows stay live.
        Decision arm: restores the predecessor to ``in_force`` and deletes the successor, which
        must be ``in_force``. Write order: the undo event, then (Decision arm) the successor's
        tombstone carrying ``restores``, then ``adapter.restore`` in one records transaction.
        The undo event is keyed by the predecessor's edge id, so a retry after a crash finds it.

        Not a re-op: a second call on the same edge is refused because ``superseded_by`` is
        already ``None``.

        Args:
            id:     The superseded predecessor to restore.
            reason: Required audit-trail reason, carried on the undo event and not on the row.

        Returns:
            ``UnsupersedeResult`` naming the shape that ran (``"fact_both_live"`` or
            ``"decision_restored"``).

        Raises:
            ToolError: unknown or buried id; empty ``reason``; the row is not currently
                       superseded; on the Decision arm, the successor is unknown, buried or not
                       ``in_force``, has other live predecessors, or already carries a tombstone
                       restoring a different predecessor.
        """
        return await unsupersede(
            id=id,
            reason=reason,
            adapter=self._adapter,
            clock=self._clock,
            lock=self._write_lock,
        )

    async def set_hot(
        self, id: HypomnemaId, hot: bool, roles: list[str] | None = None
    ) -> Fact:
        """Promote an ``active`` or ``ember`` Fact to ``hot``, or demote a hot one to ``active``.

        Owner-curated and Facts-only. Demotion always lands on ``active`` and re-arms the Fact's
        ``escalation_baseline`` when it is not already current. ``roles`` becomes the Fact's
        ``audience_hint``, the routing key for the push tiers; it overwrites and never unions.

        Args:
            id:    Fact id.
            hot:   ``True`` to promote, ``False`` to demote.
            roles: Registered roles to set as ``audience_hint``. Required when ``hot=True``;
                   ``[]`` is legal and means hot but reaching no role. Optional when
                   ``hot=False``: ``None`` leaves the hint, ``[]`` clears it.

        Returns:
            The Fact after the call, unchanged on an idempotent re-op (see
            ``dispositions.set_hot``).

        Raises:
            ToolError: unknown or buried id; id names a Decision; the requested edge is illegal
                       for the Fact's lifecycle; promoting into a role past its per-(scope, role)
                       ``hot_cap`` (names the role and every Fact currently hot for it); an
                       unregistered role in ``roles``; ``hot=True`` with ``roles=None``.
        """
        return await set_hot(
            id=id,
            hot=hot,
            roles=roles,
            adapter=self._adapter,
            clock=self._clock,
            cfg=self._cfg,
            lock=self._write_lock,
            require_roles=self._require_roles,
        )

    async def set_audience(self, id: HypomnemaId, roles: list[str]) -> Fact:
        """Rule on a Fact's audience without promoting it into the hot tier.

        Sets ``audience_hint`` and clears ``provisional``; ``lifecycle`` is never touched. Facts
        only. A ``hot`` Fact is refused: retag it with ``set_hot(id, hot=True, roles=[...])``,
        which re-checks ``hot_cap``.

        Args:
            id:    Fact id.
            roles: Registered roles to persist as ``audience_hint``; required, overwrites and
                   never unions. ``[]`` is legal and means the Fact reaches no role.

        Returns:
            The Fact after the call: unchanged (no adapter write) when ``roles`` equal the current
            ``audience_hint`` and ``provisional`` is already ``False``, otherwise re-read after the
            ``transition``.

        Raises:
            ToolError: ``roles`` is ``None``; unknown or buried id; id names a Decision; the Fact
                       is ``hot``; an unregistered role in ``roles``.
        """
        return await set_audience(
            id=id,
            roles=roles,
            adapter=self._adapter,
            clock=self._clock,
            lock=self._write_lock,
            require_roles=self._require_roles,
        )

    async def set_shelved(
        self, id: HypomnemaId, shelved: bool, reason: str
    ) -> ReviseResult:
        """Shelve or un-shelve a Fact.

        A shelved Fact stays visible to ``recall`` and is never revived by a hit, but the dedup
        probe does not see it, so a later save of the same content succeeds. Facts only.

        Edges: ``active -> shelved`` and ``archived -> shelved`` (``shelved=True``);
        ``shelved -> active`` (``shelved=False``). ``hot`` and ``ember`` Facts cannot be shelved
        directly. ``reason`` is persisted as an event. Re-ops return ``outcome="unchanged"`` with
        no write.

        Args:
            id:      Fact id.
            shelved: ``True`` to shelve, ``False`` to un-shelve.
            reason:  Required audit-trail reason.

        Returns:
            ``ReviseResult`` with outcome ``"unchanged"`` or ``"revised"``.

        Raises:
            ToolError: unknown or buried id; empty ``reason``; id names a Decision; the edge is
                       illegal for the Fact's current lifecycle.
        """
        return await set_shelved(
            id=id,
            shelved=shelved,
            reason=reason,
            adapter=self._adapter,
            clock=self._clock,
            lock=self._write_lock,
        )


    async def feedback(
        self, id: HypomnemaId, value: float, reason: str | None = None
    ) -> FeedbackResult:
        """Record a feedback event on a Fact and recompute its projections.

        ``value`` must be in ``[0, 1]`` and is rejected, never clamped. ``reason`` is stored on the
        event as given.

        Returns:
            ``FeedbackResult`` with the new ``feedback_score`` and recomputed ``importance``.

        Raises:
            ToolError: value outside [0, 1]; unknown or buried id; id names a Decision.
        """
        return await feedback(
            id=id,
            value=value,
            reason=reason,
            adapter=self._adapter,
            clock=self._clock,
            cfg=self._cfg,
            lock=self._write_lock,
        )


    async def recall(
        self,
        query: str,
        *,
        scope: str,
        limit: int = 10,
        include_invalid: bool = False,
        include_archived: bool = False,
    ) -> RecallResult:
        """Recall memories by hybrid search over ``[scope, global]``.

        ``include_invalid`` adds records with validity ``invalid``. ``include_archived`` is an
        explicit history query that never revives.

        Returns:
            ``RecallResult`` with hits, ``degraded`` flag, reason and ``legs_used``.

        Raises:
            ToolError: unregistered scope.
        """
        return await recall(
            query=query,
            scope=scope,
            limit=limit,
            include_invalid=include_invalid,
            include_archived=include_archived,
            ctx=self._context(),
        )


    async def context(
        self,
        scope: str,
        budget: int,
        role: str | None = None,
    ) -> ContextPayload:
        """Compose the session-start payload: ``pack(select_default(scope, role), budget)``.

        Selection is the fixed ``hot`` then ``ember`` tiers, filtered by ``role``: a record reaches
        a role only when its ``audience_hint`` names it, and a falsy hint reaches no role.
        Decisions are pull-only (``decisions_query``, ``decision``) and are not a tier.

        ``role`` is required. The parameter type allows ``None`` so a role-less call raises a
        ``ToolError`` naming the registered roles, after the scope checks and before role registry
        validation, rather than a ``TypeError``. ``server.py``'s ``context`` tool forwards the
        role unchecked and lets that error propagate.

        Args:
            scope:  Read scope, registered; expands to ``[scope, global]``.
            budget: Total token budget. ``pack`` raises ``ValueError`` on a negative budget; the
                    MCP layer rejects it first.
            role:   Registered role name, the routing key for both tiers.

        Returns:
            ``ContextPayload`` with the packed ``hot`` and ``ember`` tiers, the payload-level
            ``budget_used`` and ``skipped``, a ``cleaner`` rendering of the scope's newest Cleaner
            pass (``None`` when the scope has never had one).

        Raises:
            ToolError: unregistered scope; role omitted; unregistered role.
            ValueError: negative budget (direct callers only).
        """
        typed_scope = validate_and_resolve_scope(scope)
        await self._require_scope(typed_scope)
        if role is None:
            registered = await self._adapter.list_roles()
            raise ToolError(
                "context() requires a role; none was given — an unspecified "
                "audience is a caller bug (owner decision 2026-08-02: \"no "
                "caller has no role — bounce back to get it\"), not a state "
                "resolved to a default. Pass role=<name>. "
                f"Registered roles: {registered!r}"
            )
        await self._require_roles([role])

        tiers = await self._select_default(scope, role)

        payload = pack(tiers, budget, self._resolve_token_counter())

        cleaner_summary = await load_context_summary(self._adapter, scope=typed_scope)

        return payload.model_copy(update={"cleaner": cleaner_summary})

    def _resolve_token_counter(self) -> TokenCounter:
        """Return the engine's ``TokenCounter``, building and caching a ``LitellmTokenCounter`` on
        first use when none was injected."""
        if self._token_counter is None:
            self._token_counter = LitellmTokenCounter()
        return self._token_counter


    async def _select_default(self, scope: str, role: str) -> list[SelectedTier]:
        """Delegate to ``engine.context.select_default``; no tier or ranking logic lives here.

        Returns:
            The ``hot`` and ``ember`` ``SelectedTier`` entries, role-filtered, each possibly empty.
        """
        return await select_default(self._context(), scope, role)

    def _next_nightly_after(self, now: datetime) -> datetime:
        """The first local ``cleaner_nightly_at`` instant strictly after ``now``."""
        hour, minute = (int(part) for part in self._cfg.cleaner_nightly_at.split(":"))
        local_now = now.astimezone()
        candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= local_now:
            candidate = (candidate.replace(tzinfo=None) + timedelta(days=1)).astimezone()

        return candidate

    def _start_nightly_scheduler(self) -> None:
        if not self._cfg.enable_digest_cleaner:
            return

        if self._nightly_task is not None and not self._nightly_task.done():
            return

        self._next_nightly = self._next_nightly_after(self._clock.now())
        self._nightly_task = asyncio.create_task(self._nightly_loop())

    async def _nightly_loop(self) -> None:
        """Sleep until the next ``cleaner_nightly_at``, run the tick, repeat.

        A tick failure is logged at ERROR and never ends the loop.
        """
        while True:
            assert self._next_nightly is not None
            wait = (self._next_nightly - self._clock.now()).total_seconds()
            await asyncio.sleep(max(wait, 1.0))
            try:
                await self._nightly_tick()
            except Exception as exc:
                _log.error("nightly Cleaner tick failed: %s", exc, exc_info=exc)
                self._next_nightly = self._next_nightly_after(self._clock.now())

    async def _nightly_tick(self) -> bool:
        """Run one pass per registered scope, in sequence, when the clock has reached the next
        scheduled instant; then schedule the following night. Returns whether passes ran."""
        if self._next_nightly is None or self._clock.now() < self._next_nightly:
            return False

        self._next_nightly = self._next_nightly_after(self._clock.now())
        for scope in await self._adapter.list_scopes():
            await self._run_single_flight_pass(str(scope))

        return True

    async def _run_single_flight_pass(self, scope: str) -> None:
        """Run a Cleaner pass for ``scope`` as a task kept in ``self._cleaner_tasks``. A scope
        already present there has a pass in flight, so this call is a no-op for it. The task
        body never raises (``_run_cleaner_pass``)."""
        if scope in self._cleaner_tasks:
            return

        def _clear(_task: asyncio.Task[CleanSummary | None]) -> None:
            self._cleaner_tasks.pop(scope, None)

        task = asyncio.create_task(self._run_cleaner_pass(scope))
        self._cleaner_tasks[scope] = task
        task.add_done_callback(_clear)
        await task

    async def _run_cleaner_pass(self, scope: str) -> CleanSummary | None:
        """The task body run by ``_run_single_flight_pass``. Returns ``None`` after a failure.

        Runs ``self.clean(scope)``, which goes through ``engine.report.run_clean_pass`` so the
        pass persists its own report. No MCP tool wraps ``clean``; the callers that reach a pass
        are this task and the ``tabularium clean`` CLI, which calls ``run_clean_pass`` directly.

        ``_run_single_flight_pass`` awaits this task, so the nightly tick blocks on each scope in
        turn; the catch below keeps one scope's failure from reaching the tick. Any ``Exception``
        is logged at ERROR with its traceback and persisted best-effort through
        ``report.record_pass_failure``, so ``context()`` can tell a crashed pass from a scope
        never cleaned. That persist attempt has its own ``try`` because the adapter, the likely
        cause of the failure, is also what it needs.
        """
        try:
            return await self.clean(scope)
        except Exception as exc:
            _log.error(
                "detached Cleaner pass failed for scope %r: %s", scope, exc, exc_info=exc
            )
            try:
                await record_pass_failure(
                    scope=scope, adapter=self._adapter, clock=self._clock, error=exc
                )
            except Exception as persist_exc:
                _log.error(
                    "record_pass_failure itself raised while persisting a failure "
                    "report for scope %r: %s",
                    scope,
                    persist_exc,
                    exc_info=persist_exc,
                )
            return None

    async def clean(self, scope: str) -> CleanSummary:
        """Run one Cleaner pass over ``scope`` and persist its report.

        Delegates to ``engine.report.run_clean_pass``; ``engine.clean.clean`` holds the pass
        itself. A pass runs the relevance axis, the falsity axis, then the revisit axis. A revisit
        failure never changes ``CleanSummary.outcome`` or the watermark.

        Args:
            scope: Target scope, exactly one per call; ``"global"`` runs like any other.

        Returns:
            ``CleanSummary``; its ``outcome`` is ``initialized``, ``empty_pool``, ``completed`` or
            ``blocked``.

        Raises:
            ToolError: unregistered scope.
        """
        return await _run_clean_pass(
            scope=scope,
            adapter=self._adapter,
            embedder=self._embedder,
            clock=self._clock,
            lock=self._write_lock,
            cfg=self._cfg,
            require_scope=self._require_scope,
            assessor=self._assessor,
        )

    def _context(self) -> EngineContext:
        """Build the context for one call from the engine's current collaborators."""
        return EngineContext(
            adapter=self._adapter,
            embedder=self._embedder,
            clock=self._clock,
            cfg=self._cfg,
            write_lock=self._write_lock,
            assessor=self._assessor,
            token_counter=self._token_counter,
            backfill=self._backfiller,
        )

    async def _require_scope(self, scope: Scope) -> None:
        """Raise ``ToolError`` if scope is not registered."""
        await self._context().require_scope(scope)

    async def _require_purpose(self, purpose: str | None) -> None:
        """Raise ``ToolError`` if purpose is non-None and not a registered facet."""
        await self._context().require_purpose(purpose)

    async def _require_roles(self, audience_hint: list[str] | None) -> None:
        """Raise ``ToolError`` if any audience_hint entry is not a registered role."""
        await self._context().require_roles(audience_hint)
