"""Pure functions deriving a Fact's projections from its reinforcement events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact
from tabularium.storage.types import ReinforcementEvent


def compute_ewma(events: list[ReinforcementEvent], alpha: float = 0.3) -> float:
    """EWMA of ``source="feedback"`` event values, starting from 0.0 and applied in the given order.

    Other sources do not touch the score.
    """
    score = 0.0
    for ev in events:
        if ev.source == "feedback" and ev.value is not None:
            score = alpha * ev.value + (1.0 - alpha) * score
    return score


def compute_recall_count(events: list[ReinforcementEvent]) -> int:
    """Count ``dedup``, ``revival`` and ``compact`` events; ``feedback`` events do not count."""
    return sum(
        1 for ev in events if ev.source in {"dedup", "revival", "compact"}
    )


def compute_last_referenced_at(
    events: list[ReinforcementEvent],
) -> datetime | None:
    """Latest ``at`` among ``dedup`` and ``revival`` events, or ``None`` when there are none.

    ``compact`` is excluded: an offline sweep is not a reference. Callers treat ``None`` as
    ``created_at``.
    """
    return max(
        (ev.at for ev in events if ev.source in {"dedup", "revival"}),
        default=None,
    )


def compute_decay_due_at(
    recall_count: int,
    feedback_score: float,
    last_referenced_at: datetime | None,
    created_at: datetime,
    cfg: EngineConfig,
) -> datetime | None:
    """Time at which the importance composite crosses ``archive_threshold``, or ``None``.

    With ``constant = w_recall*ln(1+recall_count) + w_feedback*feedback_score`` the crossing is
    ``ref + half_life * -ln((archive_threshold - constant) / w_recency)`` days, where ``ref`` is
    ``last_referenced_at`` or ``created_at``.

    ``None`` means no future crossing: ``constant`` already reaches the threshold, ``w_recency``
    is not positive, or recency alone cannot hold the record above the threshold (it is already
    past the crossing). Nothing here archives synchronously.
    """
    constant = (
        cfg.w_recall * math.log1p(recall_count)
        + cfg.w_feedback * feedback_score
    )
    if constant >= cfg.archive_threshold:
        return None

    remaining = cfg.archive_threshold - constant
    if cfg.w_recency <= 0.0 or remaining <= 0.0:
        return None

    ratio = remaining / cfg.w_recency
    if ratio >= 1.0:
        return None

    delta_days = -cfg.half_life_days * math.log(ratio)
    if delta_days < 0:
        return None

    ref_at = last_referenced_at if last_referenced_at is not None else created_at
    return ref_at + timedelta(days=delta_days)


@dataclass(frozen=True)
class Projections:
    """The four event-derived fields written through ``update_projections``."""

    recall_count: int
    last_referenced_at: datetime | None
    feedback_score: float
    decay_due_at: datetime | None


def compute_projections(
    record: Fact | Decision,
    events: list[ReinforcementEvent],
    cfg: EngineConfig,
) -> Projections:
    """All four projections for ``record`` from its full event log, in log order.

    A Fact with at least one event gets ``decay_due_at`` from ``compute_decay_due_at``; a Fact
    with no events gets ``None``, because running the formula would manufacture drift on every
    Fact that was never reinforced. A
    Decision has no feedback or decay, so it gets ``feedback_score=0.0`` and
    ``decay_due_at=None``. ``recall_count`` and ``last_referenced_at`` are computed for both.
    """
    recall_count = compute_recall_count(events)
    last_referenced_at = compute_last_referenced_at(events)

    if not isinstance(record, Fact):
        return Projections(
            recall_count=recall_count,
            last_referenced_at=last_referenced_at,
            feedback_score=0.0,
            decay_due_at=None,
        )

    feedback_score = compute_ewma(events)
    decay_due_at: datetime | None = None
    if events:
        decay_due_at = compute_decay_due_at(
            recall_count=recall_count,
            feedback_score=feedback_score,
            last_referenced_at=last_referenced_at,
            created_at=record.created_at,
            cfg=cfg,
        )

    return Projections(
        recall_count=recall_count,
        last_referenced_at=last_referenced_at,
        feedback_score=feedback_score,
        decay_due_at=decay_due_at,
    )
