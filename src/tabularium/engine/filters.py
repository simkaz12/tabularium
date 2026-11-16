"""Scope validation, read and probe spans, and the lifecycle filters for dedup probes and recall.

Pure functions: no I/O.
"""

from __future__ import annotations

from tabularium.domain import FactLifecycle, Scope, Validity, parse_scope
from tabularium.engine._errors import ToolError
from tabularium.storage.types import Filters

GLOBAL = Scope("global")


def validate_and_resolve_scope(raw: str) -> Scope:
    """Validate and convert a raw scope string to a ``Scope`` newtype.

    Raises:
        ToolError: if the string does not match the scope pattern.
    """
    try:
        return parse_scope(raw)
    except ValueError:
        raise ToolError(
            f"Invalid scope {raw!r}: must match ^[a-z][a-z0-9-]{{0,31}}$"
        ) from None


def read_span(scope: Scope) -> list[Scope]:
    """``[global]`` for ``global``, else ``[scope, global]``."""
    if scope == GLOBAL:
        return [GLOBAL]
    return [scope, GLOBAL]


def probe_span_for(scope: Scope, registered: list[Scope]) -> list[Scope]:
    """Scopes a row saved into ``scope`` will later be read back from.

    A named scope probes ``[scope, global]``. ``global`` is in every scope's read span, so a
    ``global`` save probes every registered scope.
    """
    if scope == GLOBAL:
        return list(registered)
    return [scope, GLOBAL]


PUSH_HOT_LIFECYCLE: frozenset[FactLifecycle] = frozenset({"hot"})
PUSH_EMBER_LIFECYCLE: frozenset[FactLifecycle] = frozenset({"ember"})


def probe_filters_for(span: list[Scope]) -> Filters:
    """Dedup-probe filters: every lifecycle except ``shelved``, both validities.

    A shelved Fact is invisible to the probe, so a later save of the same content succeeds,
    while ``primary_recall_filters`` still returns it. ``archived`` stays visible: it is revivable
    and deduped against; ``shelved`` is neither.
    """
    return Filters(
        scopes=span,
        validity=frozenset({"valid", "invalid"}),
        lifecycle_in=frozenset(
            {"active", "hot", "ember", "archived", "in_force", "superseded", "revoked"}
        ),
    )


def _recall_validity(
    include_invalid: bool,
) -> frozenset[Validity]:
    """``{"valid", "invalid"}`` when ``include_invalid``, else ``{"valid"}``."""
    both: frozenset[Validity] = frozenset({"valid", "invalid"})
    valid_only: frozenset[Validity] = frozenset({"valid"})
    return both if include_invalid else valid_only


def primary_recall_filters(span: list[Scope], include_invalid: bool) -> Filters:
    """Primary recall pass: ``active``, ``hot``, ``ember`` and ``shelved``.

    The primary pass is a pure read: it reinforces nothing. A shelved hit is never revived,
    because revival draws only from ``archived_only_filters``.
    """
    return Filters(
        scopes=span,
        validity=_recall_validity(include_invalid),
        lifecycle_in=frozenset({"active", "hot", "ember", "shelved"}),
    )


def fallback_recall_filters(span: list[Scope], include_invalid: bool) -> Filters:
    """Primary filters widened to ``archived``, for an explicit ``include_archived=True`` query.

    Keeps ``shelved`` so the result is a superset of the default pass. An explicit
    ``include_archived=True`` never revives, so the shelved hits here are never revival candidates.
    """
    return Filters(
        scopes=span,
        validity=_recall_validity(include_invalid),
        lifecycle_in=frozenset({"active", "hot", "ember", "shelved", "archived"}),
    )


def archived_only_filters(span: list[Scope], include_invalid: bool) -> Filters:
    """Archived-only supplement for the auto-widen path of ``recall``.

    Searching archived records alone keeps the bounded ``candidate_k`` from being used up by
    active hits already found. ``shelved`` is excluded: the primary pass already returned it, and
    it must never be a revival candidate.
    """
    return Filters(
        scopes=span,
        validity=_recall_validity(include_invalid),
        lifecycle_in=frozenset({"archived"}),
    )
