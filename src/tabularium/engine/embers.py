"""The ember save arm.

Embers are Facts with ``lifecycle="ember"``: corrections and guidelines that recur until they
are codified into a Decision. ``save_ember`` delegates to ``facts.save_internal``;
``facts._dispatch_match`` calls ``ember_recurrence.record_ember_recurrence`` when a save
reinforces an ember.
"""

from __future__ import annotations

import asyncio

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import Asserter, Valence
from tabularium.embedding import Embedder
from tabularium.engine._errors import ToolError
from tabularium.engine._types import (
    ALL_ROLES,
    RequirePurpose,
    RequireRoles,
    RequireScope,
)
from tabularium.engine.assessor import Assessor
from tabularium.engine.backfill import BackfillSignals
from tabularium.engine.facts import save_internal
from tabularium.results import SaveResult
from tabularium.storage.types import StorageAdapter


async def save_ember(
    *,
    content: str,
    scope: str,
    valence: Valence,
    asserted_by: Asserter = "agent",
    source: str | None,
    purpose: str | None,
    audience_hint: list[str] | None,
    adapter: StorageAdapter,
    embedder: Embedder,
    clock: Clock,
    cfg: EngineConfig,
    lock: asyncio.Lock,
    require_scope: RequireScope,
    require_purpose: RequirePurpose,
    require_roles: RequireRoles,
    backfill: BackfillSignals,
    roles: list[str] | None = None,
    assessor: Assessor | None = None,
) -> SaveResult:
    """Save an ember Fact through ``save_internal`` with ``lifecycle="ember"``.

    ``valence`` is required. ``lock`` is passed through to ``save_internal``.

    ``asserted_by`` is recorded, never verified. An agent reinforcement bumps
    ``agent_recurrence_count`` and keeps the ember warm; only ``"owner"`` advances
    ``owner_recurrence_count`` and can arm the rung-2 escalation.

    ``roles`` retags the ``audience_hint`` of an existing ember when this call's dedup gate
    reinforces it. It overwrites rather than unions, and ``None`` leaves the stored hint
    untouched. It has no effect on any other dispatch outcome.

    Audience rule, checked before scope, purpose and role validation, the embed call and the
    dedup gate: no ember is created reaching nobody. A falsy ``audience_hint`` (``None`` or
    ``[]``) is treated the same on both paths:

    - ``asserted_by="owner"`` raises ``ToolError`` naming the registered roles.
    - ``asserted_by="agent"`` takes every role in ``engine._types.ALL_ROLES`` and writes the
      Fact with ``provisional=True``, meaning an owner verdict on the audience is outstanding.
      ``provisional`` is a persisted field because an all-roles hint cannot distinguish
      "owner said everyone" from "agent guessed".

    A non-empty ``audience_hint`` is kept verbatim and never marked provisional. The rule
    applies even when the gate ends up reinforcing an existing ember, where ``audience_hint``
    is unused. ``roles=[]`` on the retag path is a separate operation and still clears the
    stored hint.
    """
    provisional = False
    if not audience_hint:
        if asserted_by == "owner":
            registered = await adapter.list_roles()
            raise ToolError(
                'ember(asserted_by="owner") requires a non-empty audience_hint '
                "naming who this correction is for — an ember is authored, not "
                "inferred, and the author is right there to say who it is for "
                "(curation block, SPEC §8.4 amendment: no ember can be "
                "born reaching nobody, by either authorship path — there is no "
                "'deliberately reach nobody' option at creation time). Pass "
                "audience_hint=[...] naming one or more registered roles. "
                f"Registered roles: {registered!r}"
            )
        # NOTE: ALL_ROLES is the fixed canonical tuple; a live list_roles() read could tag the
        # ember with stray roles. require_roles still checks the tuple against the registry.
        audience_hint = list(ALL_ROLES)
        provisional = True

    return await save_internal(
        content=content,
        scope=scope,
        source=source,
        purpose=purpose,
        audience_hint=audience_hint,
        lifecycle="ember",
        valence=valence,
        asserted_by=asserted_by,
        provisional=provisional,
        resolve=None,
        adapter=adapter,
        embedder=embedder,
        clock=clock,
        cfg=cfg,
        lock=lock,
        require_scope=require_scope,
        require_purpose=require_purpose,
        require_roles=require_roles,
        roles=roles,
        backfill=backfill,
        assessor=assessor,
    )
