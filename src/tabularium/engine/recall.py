"""Hybrid recall: vector and BM25 legs fused with RRF, importance boost, archived fallback."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, HypomnemaId, Scope
from tabularium.embedding import EmbedderUnavailable
from tabularium.engine._errors import require_text
from tabularium.engine.engine_context import EngineContext
from tabularium.engine.facts import reinforce
from tabularium.engine.filters import (
    archived_only_filters,
    fallback_recall_filters,
    primary_recall_filters,
    read_span,
    validate_and_resolve_scope,
)
from tabularium.engine.read_events import log_read_events
from tabularium.engine.scoring import compute_importance, fuse_rrf
from tabularium.results import RecalledHit, RecallResult
from tabularium.storage.types import Filters, Hit, StorageAdapter

_log = logging.getLogger("tabularium")


@dataclass(frozen=True)
class _Retrieved:
    v_hits: list[Hit]
    t_hits: list[Hit]
    legs_used: list[str]
    archived_hits: list[Hit]


@dataclass(frozen=True)
class Ranked:
    final_scores: list[tuple[HypomnemaId, float]]
    records: dict[HypomnemaId, Fact | Decision]
    now: datetime


ImportanceRule = Callable[[Fact | Decision, datetime, EngineConfig], float]


async def search_legs(
    query: str,
    vector: list[float] | None,
    *,
    filters: Filters,
    k: int,
    adapter: StorageAdapter,
) -> tuple[list[Hit], list[Hit]]:
    """Text leg alone when ``vector`` is ``None``, otherwise both legs gathered."""
    if vector is None:
        t_hits = await adapter.text_search(query, filters=filters, k=k)
        return [], t_hits

    v_hits, t_hits = await asyncio.gather(
        adapter.vector_search(vector, filters=filters, k=k),
        adapter.text_search(query, filters=filters, k=k),
    )
    return v_hits, t_hits


async def _retrieve(
    query: str,
    vector: list[float] | None,
    *,
    span: list[Scope],
    candidate_k: int,
    include_invalid: bool,
    include_archived: bool,
    adapter: StorageAdapter,
    cfg: EngineConfig,
) -> _Retrieved:
    """Run the primary search, then the archived-only supplement when the primary pass is thin.

    The supplement runs only when ``include_archived`` is false and fewer than
    ``cfg.min_active_results`` distinct ids came back. Its hits are appended to both legs and
    also returned separately as ``archived_hits``. ``vector=None`` searches the text leg alone.
    """
    if include_archived:
        primary_filters = fallback_recall_filters(span, include_invalid)
    else:
        primary_filters = primary_recall_filters(span, include_invalid)

    v_hits, t_hits = await search_legs(
        query, vector, filters=primary_filters, k=candidate_k, adapter=adapter
    )
    legs_used = ["text"] if vector is None else ["vector", "text"]

    archived_hits: list[Hit] = []
    if not include_archived:
        primary_ids = {h.record.id for h in v_hits} | {h.record.id for h in t_hits}
        if len(primary_ids) < cfg.min_active_results:
            arch_filters = archived_only_filters(span, include_invalid)
            fb_v, fb_t = await search_legs(
                query, vector, filters=arch_filters, k=candidate_k, adapter=adapter
            )
            archived_hits = list(fb_v) + list(fb_t)
            v_hits = v_hits + fb_v
            t_hits = t_hits + fb_t

    return _Retrieved(
        v_hits=v_hits,
        t_hits=t_hits,
        legs_used=legs_used,
        archived_hits=archived_hits,
    )


def fact_importance(record: Fact | Decision, now: datetime, cfg: EngineConfig) -> float:
    """``compute_importance`` for a Fact; 0.0 for a Decision, which takes no decay and no boost."""
    if isinstance(record, Fact):
        return compute_importance(record, now=now, cfg=cfg)

    return 0.0


def fuse_and_boost(
    v_hits: list[Hit],
    t_hits: list[Hit],
    *,
    clock: Clock,
    cfg: EngineConfig,
    importance: ImportanceRule,
) -> Ranked | None:
    """RRF-fuse the legs and boost by normalised importance, best first; ``None`` when empty.

    ``importance`` scores each fused record at the time read from ``clock``. The clock is read
    only when there is something to rank. Equal boosted scores keep fused order.
    """
    fused = fuse_rrf(v_hits, t_hits, k=cfg.rrf_k)

    records: dict[HypomnemaId, Fact | Decision] = {}
    for h in v_hits:
        records[h.record.id] = h.record
    for h in t_hits:
        if h.record.id not in records:
            records[h.record.id] = h.record

    if not fused:
        return None

    now = clock.now()

    raw_importance: dict[HypomnemaId, float] = {}
    for rid, _rrf in fused:
        raw_importance[rid] = importance(records[rid], now, cfg)

    max_imp = max(raw_importance.values()) if raw_importance else 0.0

    final_scores: list[tuple[HypomnemaId, float]] = []
    for rid, rrf in fused:
        imp_norm = (raw_importance[rid] / max_imp) if max_imp > 0.0 else 0.0
        boosted = rrf * (1.0 + cfg.beta * imp_norm)
        final_scores.append((rid, boosted))

    final_scores.sort(key=lambda t: t[1], reverse=True)

    return Ranked(final_scores=final_scores, records=records, now=now)


async def _revive_archived(
    ranked: Ranked,
    archived_hits: list[Hit],
    *,
    limit: int,
    adapter: StorageAdapter,
    lock: asyncio.Lock,
    cfg: EngineConfig,
) -> None:
    """Reinforce and reactivate archived top-``limit`` hits, replacing their entries in ``records``.

    The only place a read mutates record state. Every candidate came from the unlocked search and
    may have been buried, invalidated or revived since, so each is re-read with ``get_by_id``
    under ``lock`` before any decision, and the fresh record replaces the searched one. A record
    buried mid-recall is dropped from ``records``. Invalid and non-Fact records are never
    reinforced. A record that is no longer archived is skipped too: another caller already
    transitioned it, and reinforcing again would write a second revival event.
    The transition to active fires only when importance reaches
    ``cfg.revival_threshold``.
    """
    records = ranked.records
    now = ranked.now
    archived_ids = {h.record.id for h in archived_hits}

    async with lock:
        for rid, _ in ranked.final_scores[:limit]:
            if rid not in archived_ids:
                continue
            try:
                live_rec = await adapter.get_by_id(rid)
            except KeyError:
                _log.debug(
                    "recall fallback revival: id=%s was buried "
                    "concurrently; skipping revival and dropping the hit",
                    rid,
                )
                records.pop(rid, None)
                continue
            records[rid] = live_rec
            if not isinstance(live_rec, Fact):
                continue
            if live_rec.validity != "valid":
                continue
            if live_rec.lifecycle != "archived":
                continue
            await reinforce(
                target_id=rid,
                source="revival",
                now=now,
                matched_from_id=None,
                record=live_rec,
                adapter=adapter,
                cfg=cfg,
            )
            try:
                updated_rec = await adapter.get_by_id(rid)
            except KeyError:
                records.pop(rid, None)
                continue
            if isinstance(updated_rec, Fact):
                records[rid] = updated_rec
                imp = compute_importance(updated_rec, now=now, cfg=cfg)
                if imp >= cfg.revival_threshold:
                    await adapter.transition(rid, "active", at=now)
                    _log.debug(
                        "recall fallback revival: id=%s importance=%.4f"
                        " >= revival_threshold=%.4f",
                        rid,
                        imp,
                        cfg.revival_threshold,
                    )
                    try:
                        records[rid] = await adapter.get_by_id(rid)
                    except KeyError:
                        records.pop(rid, None)


async def recall(
    *,
    query: str,
    scope: str,
    limit: int,
    include_invalid: bool,
    include_archived: bool,
    ctx: EngineContext,
) -> RecallResult:
    """Recall memories over ``read_span(scope)`` by hybrid search.

    The search, fusion and boost phases are lock-free reads. Only the revival loop takes
    ``ctx.write_lock``. Hits captured before it may be stale, so the loop re-reads each candidate
    with ``get_by_id`` under the lock before touching it.

    ``include_archived`` is an explicit history query that never reinforces or revives. With
    ``cfg.read_only`` no recall revives either, because revival writes events, projections and
    lifecycle; the ``read_events`` append stays.
    ``ctx.backfill.embedded()`` is called once after a successful embed, to schedule a backfill pass
    when the store is known dirty. An unavailable embedder degrades to the text leg alone.

    Returns:
        ``RecallResult`` with the hits, the ``degraded`` flag and reason, and ``legs_used``.

    Raises:
        ValueError: blank ``query``.
        ToolError: invalid or unregistered scope.
    """
    require_text(query, "query must be non-empty", error=ValueError)

    typed_scope = validate_and_resolve_scope(scope)
    await ctx.require_scope(typed_scope)

    adapter = ctx.adapter
    clock = ctx.clock
    cfg = ctx.cfg

    span = read_span(typed_scope)
    candidate_k = max(limit, cfg.candidate_k)

    degraded = False
    degraded_reason: str | None = None
    vector: list[float] | None = None

    try:
        vector = await ctx.embedder.embed(query)
    except EmbedderUnavailable as exc:
        degraded = True
        degraded_reason = exc.reason
        _log.warning("recall degraded (embedder unavailable): %s", exc.reason)
    else:
        ctx.backfill.embedded()

    retrieved = await _retrieve(
        query,
        vector,
        span=span,
        candidate_k=candidate_k,
        include_invalid=include_invalid,
        include_archived=include_archived,
        adapter=adapter,
        cfg=cfg,
    )
    legs_used = retrieved.legs_used

    ranked = fuse_and_boost(
        retrieved.v_hits,
        retrieved.t_hits,
        clock=clock,
        cfg=cfg,
        importance=fact_importance,
    )
    if ranked is None:
        return RecallResult(
            hits=[],
            degraded=degraded,
            reason=degraded_reason,
            legs_used=legs_used,
        )

    may_revive = not include_archived and not cfg.read_only
    if may_revive and retrieved.archived_hits:
        await _revive_archived(
            ranked,
            retrieved.archived_hits,
            limit=limit,
            adapter=adapter,
            lock=ctx.write_lock,
            cfg=cfg,
        )

    records = ranked.records
    hits: list[RecalledHit] = []
    for rid, score in ranked.final_scores[:limit]:
        rec = records.get(rid)
        if rec is None:
            continue
        hits.append(
            RecalledHit(
                id=rid,
                kind=rec.type,
                content=rec.content,
                lifecycle=rec.lifecycle,
                validity=rec.validity,
                superseded_by=rec.superseded_by,
                score=score,
                scope=str(rec.scope),
                purpose=rec.purpose,
            )
        )

    # One read event per returned hit, none for candidates trimmed by the limit. Unlocked and
    # never fails the call.
    await log_read_events([h.id for h in hits], "recall", adapter=adapter, clock=clock)

    return RecallResult(
        hits=hits,
        degraded=degraded,
        reason=degraded_reason,
        legs_used=legs_used,
    )
