"""The collaborators one engine call runs against, and the registry checks that need them."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import Scope
from tabularium.embedding import Embedder
from tabularium.engine._errors import ToolError
from tabularium.engine.assessor import Assessor
from tabularium.engine.backfill import BackfillSignals
from tabularium.storage.types import StorageAdapter
from tabularium.tokens import TokenCounter


@dataclass(frozen=True)
class EngineContext:
    """A snapshot of the engine's collaborators, built by ``Tabularium._context()`` for one call.

    Built by ``Tabularium._context()`` on each ``recall``, ``context`` and ``clean`` call and on
    each registry check, so a collaborator reassigned on the engine is read at call time, not at
    construction. ``assessor`` and ``token_counter`` are ``None`` when the engine was given none.
    The ``require_*`` methods read the registries from ``adapter``.
    """

    adapter: StorageAdapter
    embedder: Embedder
    clock: Clock
    cfg: EngineConfig
    write_lock: asyncio.Lock
    assessor: Assessor | None
    token_counter: TokenCounter | None
    backfill: BackfillSignals

    async def require_scope(self, scope: Scope) -> None:
        """Raise ``ToolError`` if scope is not registered."""
        registered = await self.adapter.list_scopes()
        if scope not in registered:
            raise ToolError(
                f"Unregistered scope {scope!r}: "
                "register with ``tabularium scope add <name>`` before use. "
                f"Registered scopes: {registered!r}"
            )

    async def require_purpose(self, purpose: str | None) -> None:
        """Raise ``ToolError`` if purpose is non-None and not a registered facet."""
        if purpose is None:
            return
        registered = await self.adapter.list_facets()
        if purpose not in registered:
            raise ToolError(
                f"Unregistered purpose {purpose!r}: "
                "register with ``tabularium facet add <name>`` before use. "
                f"Registered facets: {registered!r}"
            )

    async def require_roles(self, audience_hint: list[str] | None) -> None:
        """Raise ``ToolError`` if any audience_hint entry is not a registered role."""
        if not audience_hint:
            return
        registered = await self.adapter.list_roles()
        for role in audience_hint:
            if role not in registered:
                raise ToolError(
                    f"Unregistered role {role!r} in audience_hint: "
                    "register with ``tabularium role add <name>`` before use. "
                    f"Registered roles: {registered!r}"
                )
