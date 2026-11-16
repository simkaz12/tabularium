"""Push-side default selection and the token-budget packer for the session-start payload.

``select_default`` is query-less, token-blind and push-only: it selects and ranks records and
never rewrites, truncates or summarizes content. Tiers, in presentation order:

1. hot Facts, ranked by ``importance DESC, last_referenced_at DESC``;
2. embers strictly above the fade floor, ranked by importance. The floor is
   ``cfg.archive_threshold``; there is no ember-specific knob.

Decisions are pull-only at session start (``decisions_query``, ``decision``); there is no
Decision tier and no flag to enable one. General ``active`` Facts never reach the payload.

Rules:

- Each tier is one ranked list over the whole read span (``read_span``), sorted by its own key
  with no per-scope quota or priority.
- Excluded: ``invalid`` records, and any lifecycle outside the tier's own ``scan_records``
  filter. Buried records never appear in a scan.
- Every sort key ends in ``str(id)`` ascending. ``scan_records`` guarantees no order across
  adapters, so this explicit sort is load-bearing.
- Role routing: a record reaches a role only when ``audience_hint`` names that role. A falsy hint
  reaches no role.
- Each tier costs one ``adapter.scan_records`` round trip and no per-candidate ``get_by_id``.
  Scope and validity are applied in Python on the yielded records.
- ``select_default`` takes no lock. A record buried after its tier's scan returned can still
  appear in that tier's output.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from tabularium.config import EngineConfig
from tabularium.domain import Fact, Scope
from tabularium.engine.engine_context import EngineContext
from tabularium.engine.filters import (
    PUSH_EMBER_LIFECYCLE,
    PUSH_HOT_LIFECYCLE,
    read_span,
    validate_and_resolve_scope,
)
from tabularium.engine.scoring import compute_importance
from tabularium.results import ContextPayload, PackedTier
from tabularium.storage.types import StorageAdapter
from tabularium.tokens import TokenCounter

_VALID = "valid"


class SelectedTier(BaseModel):
    """One ranked push tier; ``entries`` is in final presentation order and may be empty.

    Content is carried verbatim as the full Fact record.
    """

    model_config = {"frozen": True}

    name: str
    entries: list[Fact] = Field(default_factory=list)


def _ref_at(record: Fact) -> datetime:
    """``last_referenced_at``, falling back to ``created_at``."""
    return record.last_referenced_at if record.last_referenced_at is not None else record.created_at


def routes_to(rec: Fact, role: str) -> bool:
    """Whether ``rec`` reaches ``role``'s push payload: only when ``audience_hint`` names the role.

    A falsy hint (``None`` or empty) routes to no role; "for everyone" means naming all four.
    Both push tiers use this one rule. Pull (``recall``) never calls it.
    """
    return role in (rec.audience_hint or ())


async def select_default(ctx: EngineContext, scope: str, role: str) -> list[SelectedTier]:
    """Select the two push tiers, ``hot`` then ``ember``, for ``role`` over ``read_span(scope)``.

    Always returns both tiers in that order, each independently ranked and possibly empty. Every
    candidate's importance is computed against one ``ctx.clock.now()``. ``role`` is already
    validated by the caller; ``ctx.cfg.archive_threshold`` is the ember fade floor.

    Raises:
        ToolError: invalid scope.
    """
    typed_scope = validate_and_resolve_scope(scope)
    span = read_span(typed_scope)
    now = ctx.clock.now()

    hot_entries = await _select_hot(ctx.adapter, ctx.cfg, span, now, role)
    ember_entries = await _select_embers(ctx.adapter, ctx.cfg, span, now, role)

    return [
        SelectedTier(name="hot", entries=list(hot_entries)),
        SelectedTier(name="ember", entries=list(ember_entries)),
    ]


async def _select_hot(
    adapter: StorageAdapter,
    cfg: EngineConfig,
    span: list[Scope],
    now: datetime,
    role: str,
) -> list[Fact]:
    """Hot Facts in the span, role-routed, by importance desc, recency desc, then ``str(id)``."""
    scored: list[tuple[Fact, float]] = []
    async for rec in await adapter.scan_records(lifecycle_in=PUSH_HOT_LIFECYCLE):
        if not isinstance(rec, Fact) or rec.scope not in span or rec.validity != _VALID:
            continue
        if not routes_to(rec, role):
            continue
        scored.append((rec, compute_importance(rec, now=now, cfg=cfg)))

    scored.sort(key=lambda item: (-item[1], -_ref_at(item[0]).timestamp(), str(item[0].id)))
    return [rec for rec, _imp in scored]


async def _select_embers(
    adapter: StorageAdapter,
    cfg: EngineConfig,
    span: list[Scope],
    now: datetime,
    role: str,
) -> list[Fact]:
    """Embers in the span, role-routed, with importance strictly above the fade floor.

    Ordered by importance desc, then ``str(id)``.
    """
    scored: list[tuple[Fact, float]] = []
    async for rec in await adapter.scan_records(lifecycle_in=PUSH_EMBER_LIFECYCLE):
        if not isinstance(rec, Fact) or rec.scope not in span or rec.validity != _VALID:
            continue
        if not routes_to(rec, role):
            continue
        imp = compute_importance(rec, now=now, cfg=cfg)
        if imp <= cfg.archive_threshold:
            continue
        scored.append((rec, imp))

    scored.sort(key=lambda item: (-item[1], str(item[0].id)))
    return [rec for rec, _imp in scored]


def pack(
    tiers: list[SelectedTier],
    budget: int,
    counter: TokenCounter,
) -> ContextPayload:
    """Pack ``tiers`` into ``budget`` tokens: greedy whole-entry fill, forward-only spill.

    Pure: no I/O, clock or adapter. Tiers and entries are never reordered, and content is never
    truncated: an entry that does not fit is skipped whole and counted in that tier's ``skipped``,
    and the next entry in the tier is still tried.

    The budget is split as equally as possible across the tiers that have at least one entry; the
    first ``budget % n`` of them get one extra unit, so shares sum to ``budget``. A tier with no
    entries gets share 0. Each tier then spends its share plus the unused budget carried forward
    from earlier tiers. Spill never flows backward. ``counter.count`` runs once per entry
    considered, on ``entry.content``.

    Returns:
        A ``ContextPayload`` with one ``PackedTier`` per input tier, in order, plus the summed
        ``budget_used`` and ``skipped``. An empty ``tiers`` returns an empty payload.

    Raises:
        ValueError: if ``budget`` is negative.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")

    tier_count = len(tiers)
    if tier_count == 0:
        return ContextPayload(tiers=[], budget_used=0, skipped=0)

    non_empty_indices = [i for i, tier in enumerate(tiers) if tier.entries]
    shares = [0] * tier_count
    if non_empty_indices:
        base_share, remainder = divmod(budget, len(non_empty_indices))
        for rank, i in enumerate(non_empty_indices):
            shares[i] = base_share + (1 if rank < remainder else 0)

    packed_tiers: list[PackedTier] = []
    carry = 0
    for tier, share in zip(tiers, shares, strict=True):
        available = share + carry
        used = 0
        skipped = 0
        entries: list[Fact] = []
        for entry in tier.entries:
            cost = counter.count(entry.content)
            if used + cost <= available:
                entries.append(entry)
                used += cost
            else:
                skipped += 1
        packed_tiers.append(
            PackedTier(name=tier.name, entries=entries, budget_used=used, skipped=skipped)
        )
        carry = available - used

    return ContextPayload(
        tiers=packed_tiers,
        budget_used=sum(t.budget_used for t in packed_tiers),
        skipped=sum(t.skipped for t in packed_tiers),
    )
