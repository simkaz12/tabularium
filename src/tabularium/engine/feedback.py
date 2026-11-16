"""The ``feedback`` arm: append a ``source="feedback"`` event and recompute the Fact's projections.

- ``value`` is in ``[0, 1]`` (bad about 0.0, neutral about 0.5, helpful about 1.0) and is
  rejected, never clamped, outside it. The score term is non-negative, so feedback erodes a prior
  boost toward 0 but cannot push a record below its pre-feedback floor or force archival; an
  explicit neutral is a mild positive signal.
- Fact-only: a Decision id is a category ``ToolError``.
- Feedback and references stay independent. All four projections are recomputed from the full
  event log with the same pure helpers ``facts.reinforce`` uses; ``compute_recall_count`` and
  ``compute_last_referenced_at`` ignore ``feedback`` events, so a feedback event never changes
  ``recall_count`` or ``last_referenced_at``.
- This module builds its own ``ReinforcementEvent`` rather than calling ``facts.reinforce``,
  whose sources carry no ``value`` or ``reason``.
- The whole read-check-append-recompute-write body runs under the engine's single write lock, so
  two concurrent calls on one record cannot overwrite each other's projections from a stale log.
"""

from __future__ import annotations

import asyncio

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId
from tabularium.engine._errors import ToolError
from tabularium.engine.lookups import get_fact, get_record
from tabularium.engine.projections import compute_projections
from tabularium.engine.scoring import compute_importance
from tabularium.results import FeedbackResult
from tabularium.storage.types import ReinforcementEvent, StorageAdapter


async def feedback(
    *,
    id: HypomnemaId,
    value: float,
    reason: str | None,
    adapter: StorageAdapter,
    clock: Clock,
    cfg: EngineConfig,
    lock: asyncio.Lock,
) -> FeedbackResult:
    """Append a feedback event for Fact ``id`` and recompute its projections under ``lock``.

    ``reason`` is stored on the event as given (``None`` stays ``None``). ``cfg`` supplies the
    composite weights for ``importance`` and ``decay_due_at``.

    Returns:
        ``FeedbackResult`` with the new ``feedback_score`` and recomputed ``importance``.

    Raises:
        ToolError: ``value`` outside ``[0, 1]``; unknown or buried id (including a record buried
            between the check and the re-read); id names a Decision.
    """
    if not (0.0 <= value <= 1.0):
        raise ToolError(
            f"feedback: value={value!r} must be in [0, 1] "
            "(anchors bad≈0.0 · neutral≈0.5 · helpful≈1.0 — SPEC §8.1 [A8]); "
            "rejected, never silently clamped"
        )

    async with lock:
        record = await get_record(adapter, id)

        if not isinstance(record, Fact):
            raise ToolError(
                f"feedback: id={id!s} is a Decision, not a Fact; "
                "feedback is Facts-only — a Decision's standing is governed "
                "by supersession and validity, not usefulness scoring "
                "(SPEC §9.2)"
            )

        now = clock.now()
        event = ReinforcementEvent(
            target_id=id,
            at=now,
            source="feedback",
            matched_from_id=None,
            value=value,
            reason=reason,
        )
        await adapter.append_event(event)

        events = await adapter.query_events(target_id=id)
        projections = compute_projections(record, events, cfg)

        await adapter.update_projections(
            id,
            recall_count=projections.recall_count,
            last_referenced_at=projections.last_referenced_at,
            feedback_score=projections.feedback_score,
            decay_due_at=projections.decay_due_at,
        )

        updated = await get_fact(adapter, id)

        importance = compute_importance(updated, now=now, cfg=cfg)

    return FeedbackResult(
        id=id,
        feedback_score=projections.feedback_score,
        importance=importance,
        reason=reason,
    )
