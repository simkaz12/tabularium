"""Decision read path: ``load_revalidation_records``, ``decision_view`` and ``decisions_query``.

None of these writes through the save path. ``decisions_query`` searches and ranks with
``recall.search_legs`` and ``recall.fuse_and_boost``, passing ``decision_importance`` as the
importance rule.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import Decision, DecisionLifecycle, Fact, HypomnemaId, Validity
from tabularium.embedding import Embedder, EmbedderUnavailable
from tabularium.engine._errors import ToolError, require_text
from tabularium.engine._types import RequireScope
from tabularium.engine.filters import read_span, validate_and_resolve_scope
from tabularium.engine.read_events import log_read_events
from tabularium.engine.recall import fuse_and_boost, search_legs
from tabularium.engine.scoring import compute_importance
from tabularium.render import derive_title, render_decision
from tabularium.results import DecisionLink, DecisionView, RevalidationRecord
from tabularium.storage.types import Filters, StorageAdapter

_log = logging.getLogger("tabularium")


async def load_revalidation_records(
    decision_id: HypomnemaId,
    adapter: StorageAdapter,
) -> list[RevalidationRecord]:
    """Load a Decision's revalidation history as ``RevalidationRecord``s, oldest first.

    ``decision_view``, ``render_decision`` and ``cli._cmd_sync`` share this mapping, so an
    exported render matches ``engine.decision(id)``.
    """
    events = await adapter.list_revalidations(decision_id)
    return [
        RevalidationRecord(
            withdrawn_cause=event.withdrawn_cause,
            withdrawn_reason=event.withdrawn_reason,
            reason=event.reason,
            at=event.at,
        )
        for event in sorted(events, key=lambda e: e.at)
    ]


async def decision_view(
    *,
    id: HypomnemaId,
    adapter: StorageAdapter,
    clock: Clock,
) -> DecisionView:
    """Return the full view of a single Decision.

    Logs a ``ReadEvent(source="decision-view")`` once the id is confirmed to be a live
    Decision, and not for an unknown or buried id or a Fact id, since nothing was delivered.
    """
    try:
        record = await adapter.get_by_id(id)
    except KeyError:
        raise ToolError(
            f"decision: unknown or buried id={id!s}"
        ) from None

    if isinstance(record, Fact):
        raise ToolError(
            f"decision: id={id!s} is a Fact, not a Decision; "
            "use recall to retrieve Facts"
        )

    # Follow ``superseded_by`` to the in-force tip.
    tip: HypomnemaId | None = None
    if record.superseded_by is not None:
        current_id: HypomnemaId | None = record.superseded_by
        for _ in range(1000):  # bound against cycles in corrupt data
            if current_id is None:
                break
            try:
                hop = await adapter.get_by_id(current_id)
            except KeyError:
                break
            if not isinstance(hop, Decision):
                break
            if hop.superseded_by is None:
                tip = hop.id
                break
            current_id = hop.superseded_by

    revalidations = await load_revalidation_records(record.id, adapter)

    await log_read_events([record.id], "decision-view", adapter=adapter, clock=clock)

    return DecisionView(
        id=record.id,
        content=record.content,
        lifecycle=record.lifecycle,
        validity=record.validity,
        supersedes=(
            record.supersession_chain[-1].id
            if record.supersession_chain
            else None
        ),
        superseded_by=record.superseded_by,
        tip=tip,
        promoted_from=(
            [snap.id for snap in record.promoted_from]
            if record.promoted_from
            else None
        ),
        invalidated_by=record.invalidated_by,
        invalidation_reason=record.invalidation_reason,
        revalidations=revalidations,
        recall_count=record.recall_count,
        last_referenced_at=record.last_referenced_at,
        purpose=record.purpose,
        created_at=record.created_at,
        rendered=render_decision(record, revalidations=revalidations),
    )


def decision_importance(record: Fact | Decision, now: datetime, cfg: EngineConfig) -> float:
    """Importance for ``decisions_query`` ranking.

    A Fact scores ``compute_importance``. A Decision scores from its recall count and recency
    alone, with no feedback or decay term.
    """
    if isinstance(record, Fact):
        return compute_importance(record, now=now, cfg=cfg)

    ref_at = (
        record.last_referenced_at
        if record.last_referenced_at is not None
        else record.created_at
    )
    delta_days = (now - ref_at).total_seconds() / 86400.0
    recency = math.exp(-delta_days / cfg.half_life_days)
    return (
        cfg.w_recall * math.log1p(record.recall_count)
        + cfg.w_recency * recency
    )


async def decisions_query(
    *,
    query: str,
    scope: str,
    limit: int,
    include_superseded: bool,
    include_invalid: bool,
    adapter: StorageAdapter,
    embedder: Embedder,
    clock: Clock,
    cfg: EngineConfig,
    require_scope: RequireScope,
) -> list[DecisionLink]:
    """Hybrid retrieval restricted to the Decision arm."""
    require_text(query, "query must be non-empty", error=ValueError)

    typed_scope = validate_and_resolve_scope(scope)
    await require_scope(typed_scope)

    span = read_span(typed_scope)

    if include_superseded:
        lifecycle_set: frozenset[DecisionLifecycle] = frozenset(
            {"in_force", "superseded", "revoked"}
        )
    else:
        lifecycle_set = frozenset({"in_force"})

    # Valid-only by default, as in ``recall``; ``include_invalid`` widens it.
    if include_invalid:
        validity_set: frozenset[Validity] = frozenset(
            {"valid", "invalid"}
        )
    else:
        validity_set = frozenset({"valid"})

    filters = Filters(
        scopes=span,
        validity=validity_set,
        lifecycle_in=lifecycle_set,
    )

    candidate_k = cfg.candidate_k

    # Without an embedder only the BM25 leg runs.
    vector: list[float] | None = None
    try:
        vector = await embedder.embed(query)
    except EmbedderUnavailable:
        _log.warning("decisions_query degraded: embedder unavailable; BM25 only")

    v_hits, t_hits = await search_legs(
        query, vector, filters=filters, k=candidate_k, adapter=adapter
    )

    ranked = fuse_and_boost(
        v_hits,
        t_hits,
        clock=clock,
        cfg=cfg,
        importance=decision_importance,
    )
    if ranked is None:
        return []

    records = ranked.records
    links: list[DecisionLink] = []
    for rid, _ in ranked.final_scores[:limit]:
        rec = records.get(rid)
        if rec is None or not isinstance(rec, Decision):
            continue
        links.append(
            DecisionLink(
                uri=f"decision://{rec.id!s}",
                id=rec.id,
                title=derive_title(rec.content),
                lifecycle=rec.lifecycle,
                validity=rec.validity,
                superseded_by=rec.superseded_by,
                scope=str(rec.scope),
            )
        )

    return links
