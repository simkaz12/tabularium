"""Storage maintenance policy and the projection rebuild.

Everything here goes through the ``StorageAdapter`` seam, never LanceDB directly.

``maybe_upkeep`` triggers on measured debt (``num_small_fragments``), not a write counter, which
would have to be persisted and goes wrong when the store is maintained out of band. It runs
synchronously on the save call that crosses the threshold, so there is no background task or
scheduler.

``rebuild_projections`` recomputes ``recall_count``, ``last_referenced_at``, ``feedback_score``
and ``decay_due_at`` for every live record from the event log alone, with the same pure helpers
the live write path uses (``engine.projections``). Report mode (default) writes nothing; repair
mode persists each drifted record through ``update_projections``.

The event log is fetched once and grouped by ``target_id`` in Python; records are streamed one at
a time through ``scan_records(lifecycle_in=None)``.

``rebuild_projections`` takes no lock, so a long rebuild does not stall saves:

- A record buried between the scan and its repair write makes ``update_projections`` raise
  ``KeyError``. That is caught and logged at DEBUG; the drift stays reported and the record does
  not count toward ``repaired``.
- A repair write can overwrite a concurrent writer's own ``update_projections`` (such as
  ``feedback``) on the same id with this rebuild's stale values. The event log stays
  authoritative, so the next repair pass heals it. Report mode cannot hit this.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, HypomnemaId
from tabularium.engine.projections import compute_projections
from tabularium.results import ProjectionDrift, ProjectionRebuildReport
from tabularium.storage.types import ReinforcementEvent, StorageAdapter

_log = logging.getLogger("tabularium")


async def maybe_upkeep(adapter: StorageAdapter, cfg: EngineConfig) -> None:
    """Run ``adapter.upkeep()`` when ``num_small_fragments`` reaches ``cfg.maintenance_threshold``.

    Reads ``maintenance_debt()`` once and compares an absolute count (the debt signal floors near 1
    on a healthy store).

    A failure in either call is logged at WARNING and swallowed, because the
    record that triggered the call is already committed and a maintenance error must not reach the
    save path's returned verdict.
    """
    try:
        debt = await adapter.maintenance_debt()
        if debt.num_small_fragments >= cfg.maintenance_threshold:
            await adapter.upkeep()
    except Exception as exc:
        _log.warning(
            "maintenance policy failed (threshold=%d): %s",
            cfg.maintenance_threshold,
            exc,
            exc_info=exc,
        )


def _record_drift(
    record: Fact | Decision,
    *,
    recall_count: int,
    last_referenced_at: datetime | None,
    feedback_score: float,
    decay_due_at: datetime | None,
) -> ProjectionDrift | None:
    """A ``ProjectionDrift`` naming only the fields where ``record`` disagrees, or ``None``."""
    drift_recall_count: tuple[int, int] | None = None
    if record.recall_count != recall_count:
        drift_recall_count = (record.recall_count, recall_count)

    drift_last_referenced_at: tuple[datetime | None, datetime | None] | None = None
    if record.last_referenced_at != last_referenced_at:
        drift_last_referenced_at = (record.last_referenced_at, last_referenced_at)

    drift_feedback_score: tuple[float, float] | None = None
    drift_decay_due_at: tuple[datetime | None, datetime | None] | None = None
    if isinstance(record, Fact):
        if record.feedback_score != feedback_score:
            drift_feedback_score = (record.feedback_score, feedback_score)
        if record.decay_due_at != decay_due_at:
            drift_decay_due_at = (record.decay_due_at, decay_due_at)

    if (
        drift_recall_count is None
        and drift_last_referenced_at is None
        and drift_feedback_score is None
        and drift_decay_due_at is None
    ):
        return None

    return ProjectionDrift(
        id=record.id,
        kind=record.type,
        recall_count=drift_recall_count,
        last_referenced_at=drift_last_referenced_at,
        feedback_score=drift_feedback_score,
        decay_due_at=drift_decay_due_at,
    )


async def rebuild_projections(
    adapter: StorageAdapter,
    cfg: EngineConfig,
    *,
    repair: bool = False,
) -> ProjectionRebuildReport:
    """Recompute every live record's projections from the event log and report drift.

    ``repair=False`` (default) writes nothing. ``repair=True`` also persists each drifted record's
    recomputed values through ``update_projections``. See the module docstring for the locking
    behaviour.

    Expected values come from ``compute_projections``: a Fact with zero events expects
    ``decay_due_at=None``, and a Decision expects ``feedback_score=0.0`` and ``decay_due_at=None``.

    Returns:
        ``ProjectionRebuildReport`` with ``scanned``, one ``drifted`` entry per record with a
        disagreeing field, and ``repaired``, the writes applied (always 0 in report mode).
    """
    all_events = await adapter.query_events()
    events_by_target: dict[HypomnemaId, list[ReinforcementEvent]] = defaultdict(list)
    for ev in all_events:
        events_by_target[ev.target_id].append(ev)

    scanned = 0
    drifted: list[ProjectionDrift] = []
    repaired = 0

    async for record in await adapter.scan_records(lifecycle_in=None):
        scanned += 1
        events = events_by_target.get(record.id, [])
        projections = compute_projections(record, events, cfg)

        drift = _record_drift(
            record,
            recall_count=projections.recall_count,
            last_referenced_at=projections.last_referenced_at,
            feedback_score=projections.feedback_score,
            decay_due_at=projections.decay_due_at,
        )
        if drift is None:
            continue
        drifted.append(drift)

        if repair:
            try:
                await adapter.update_projections(
                    record.id,
                    recall_count=projections.recall_count,
                    last_referenced_at=projections.last_referenced_at,
                    feedback_score=projections.feedback_score,
                    decay_due_at=projections.decay_due_at,
                )
                repaired += 1
            except KeyError:
                _log.debug(
                    "rebuild_projections: id=%s was buried between the scan "
                    "and its repair write; skipping (drift stays reported)",
                    record.id,
                )

    return ProjectionRebuildReport(
        mode="repair" if repair else "report",
        scanned=scanned,
        drifted=drifted,
        repaired=repaired,
    )
