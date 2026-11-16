"""Hybrid dedup probe: vector and text legs fused with RRF, best candidate by raw cosine.

``probe_candidates`` is used by the Cleaner's falsity sweep. The save arms probe through
``admission.serial_probe`` instead.

Selection is anchored on the vector leg's raw cosine, never on the fused score. The fused
order only drives iteration. Does not import ``engine.recall`` or ``engine.decisions``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Set as AbstractSet

from pydantic import BaseModel

from tabularium.config import EngineConfig
from tabularium.domain import HypomnemaId, Scope, Validity
from tabularium.engine.filters import probe_filters_for
from tabularium.engine.scoring import fuse_rrf
from tabularium.storage.types import Filters, Hit, StorageAdapter


class ProbeResult(BaseModel):
    """Both legs, the fused order and the best candidate from one probe.

    ``best_hit`` and ``best_cosine`` come from ``select_best_candidate`` applied to
    ``exclude`` alone.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    v_hits: list[Hit]
    t_hits: list[Hit]
    fused_ids: list[tuple[HypomnemaId, float]]
    best_hit: Hit | None
    best_cosine: float


def select_best_candidate(
    fused_ids: list[tuple[HypomnemaId, float]],
    v_hits: list[Hit],
    exclude: AbstractSet[HypomnemaId],
) -> tuple[Hit | None, float]:
    """Pick the candidate with the highest vector-leg cosine, iterating in fused order.

    A candidate present only in the text leg has no vector hit and is never selected.
    Ties keep the earlier candidate: only a strictly greater cosine replaces the best.
    Returns ``(None, 0.0)`` when nothing qualifies.
    """
    best_cosine = 0.0
    best_hit: Hit | None = None
    for cand_id, _ in fused_ids:
        if cand_id in exclude:
            continue
        v_hit = next((h for h in v_hits if h.record.id == cand_id), None)
        if v_hit is not None and v_hit.score > best_cosine:
            best_cosine = v_hit.score
            best_hit = v_hit
    return best_hit, best_cosine


async def probe_candidates(
    *,
    content: str,
    vector: list[float],
    span: list[Scope],
    k: int,
    exclude: AbstractSet[HypomnemaId] = frozenset(),
    validity: frozenset[Validity] | None = None,
    adapter: StorageAdapter,
    cfg: EngineConfig,
) -> ProbeResult:
    """Run both legs concurrently at ``k`` per leg, fuse with ``cfg.rrf_k``, and pick the best.

    Probes both validities by default so a save can revive or reconcile an invalid record.
    ``validity`` overrides that, for example ``frozenset({"valid"})`` for valid-only
    candidates.
    """
    probe_filters: Filters = probe_filters_for(span)
    if validity is not None:
        probe_filters = Filters(
            scopes=probe_filters.scopes,
            validity=validity,
            lifecycle_in=probe_filters.lifecycle_in,
        )

    v_hits, t_hits = await asyncio.gather(
        adapter.vector_search(vector, filters=probe_filters, k=k),
        adapter.text_search(content, filters=probe_filters, k=k),
    )
    fused_ids = fuse_rrf(v_hits, t_hits, k=cfg.rrf_k)
    best_hit, best_cosine = select_best_candidate(fused_ids, v_hits, exclude)

    return ProbeResult(
        v_hits=v_hits,
        t_hits=t_hits,
        fused_ids=fused_ids,
        best_hit=best_hit,
        best_cosine=best_cosine,
    )
