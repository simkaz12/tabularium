"""The ``tabularium doctor`` report.

Doctor observes and never polices: nothing it computes feeds an engine decision and no save is
rejected or rewritten because of it. It reads only through the ``StorageAdapter`` seam.

Sections:

1. Facet coverage: share of live records carrying a ``purpose``, the value distribution and
   ``audience_hint`` frequency. ``scanned`` is the denominator for every ratio.
2. Dead vocabulary: registered purposes no policy reads, and registered purposes unused in the
   scanned data. The data half comes from the same scan as section 1.
3. Rows awaiting embedding: the size of ``adapter.backfill_targets()``, the list the backfill
   pass consumes. Rows saved while the embedder was down carry ``vector=None`` until then.
4. Projection drift: ``maintenance.rebuild_projections`` in report mode (``repair=False``).
"""

from __future__ import annotations

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact
from tabularium.engine.maintenance import rebuild_projections
from tabularium.results import (
    AwaitingEmbeddingReport,
    DeadVocabularyReport,
    DoctorReport,
    FacetCoverageReport,
)
from tabularium.storage.types import StorageAdapter


async def build_doctor_report(
    adapter: StorageAdapter, cfg: EngineConfig
) -> DoctorReport:
    """Build the doctor report.

    One ``scan_records(lifecycle_in=None)`` pass (buried excluded, archived included) feeds both
    the facet coverage and the data half of dead vocabulary, so the two cover the same population.
    """
    registered_facets = await adapter.list_facets()

    scanned = 0
    fact_count = 0
    decision_count = 0
    with_purpose = 0
    with_audience_hint = 0
    purpose_counts: dict[str, int] = dict.fromkeys(registered_facets, 0)

    async for record in await adapter.scan_records(lifecycle_in=None):
        scanned += 1
        if isinstance(record, Fact):
            fact_count += 1
        elif isinstance(record, Decision):
            decision_count += 1

        if record.purpose is not None:
            with_purpose += 1
            purpose_counts[record.purpose] = purpose_counts.get(record.purpose, 0) + 1
        if record.audience_hint:
            with_audience_hint += 1

    facet_coverage = FacetCoverageReport(
        scanned=scanned,
        fact_count=fact_count,
        decision_count=decision_count,
        with_purpose=with_purpose,
        purpose_counts=purpose_counts,
        with_audience_hint=with_audience_hint,
    )

    registered_sorted = sorted(registered_facets)
    dead = sorted(set(registered_facets))
    unused_in_data = sorted(
        name for name in registered_facets if purpose_counts.get(name, 0) == 0
    )
    dead_vocabulary = DeadVocabularyReport(
        registered=registered_sorted,
        dead=dead,
        unused_in_data=unused_in_data,
    )

    awaiting_embedding = AwaitingEmbeddingReport(count=len(await adapter.backfill_targets()))

    projection_drift = await rebuild_projections(adapter, cfg, repair=False)

    return DoctorReport(
        facet_coverage=facet_coverage,
        dead_vocabulary=dead_vocabulary,
        awaiting_embedding=awaiting_embedding,
        projection_drift=projection_drift,
    )
