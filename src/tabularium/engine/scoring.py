"""RRF fusion and the importance formula. Pure functions: no I/O, no adapter calls."""

from __future__ import annotations

import math
from datetime import datetime

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId
from tabularium.storage.types import Hit


def _rrf_score(rank: int, k: int) -> float:
    """Single-leg RRF contribution: 1 / (k + rank)."""
    return 1.0 / (k + rank)


def fuse_rrf(
    vector_hits: list[Hit],
    text_hits: list[Hit],
    k: int,
) -> list[tuple[HypomnemaId, float]]:
    """Fuse two retrieval legs with RRF.

    Returns ``(id, rrf_score)`` pairs, highest score first. A candidate seen by one leg
    contributes 0 from the other.
    """
    scores: dict[HypomnemaId, float] = {}

    for hit in vector_hits:
        rid = hit.record.id
        scores[rid] = scores.get(rid, 0.0) + _rrf_score(hit.rank, k)

    for hit in text_hits:
        rid = hit.record.id
        scores[rid] = scores.get(rid, 0.0) + _rrf_score(hit.rank, k)

    return sorted(scores.items(), key=lambda t: t[1], reverse=True)


def compute_importance(
    record: Fact,
    *,
    now: datetime,
    cfg: EngineConfig,
) -> float:
    """Importance of a Fact, computed on demand and never stored.

    Recency decays from ``last_referenced_at``, falling back to ``created_at``.
    """
    ref_at = record.last_referenced_at if record.last_referenced_at is not None \
        else record.created_at
    delta_days = (now - ref_at).total_seconds() / 86400.0
    recency = math.exp(-delta_days / cfg.half_life_days)
    return (
        cfg.w_recall * math.log1p(record.recall_count)
        + cfg.w_recency * recency
        + cfg.w_feedback * record.feedback_score
    )
