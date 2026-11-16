"""Type aliases and the role constants shared by engine modules; imports nothing from the engine."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from tabularium.domain import Scope

RequireScope = Callable[[Scope], Awaitable[None]]
RequirePurpose = Callable[[str | None], Awaitable[None]]
RequireRoles = Callable[[list[str] | None], Awaitable[None]]

# The four audience roles. Single source for ``Tabularium.initialize()`` seeding and for
# ``save_ember``'s creation-time default, so a new ember reaches exactly these four and never a
# live registry snapshot that may carry operator-added or historical roles.
_DECIDER_ROLE = "decider"
_BUILDER_ROLE = "builder"
_REVIEWER_ROLE = "reviewer"
_NAVIGATOR_ROLE = "navigator"
ALL_ROLES = (_DECIDER_ROLE, _BUILDER_ROLE, _REVIEWER_ROLE, _NAVIGATOR_ROLE)
