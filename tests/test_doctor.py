"""``tabularium doctor`` — the full SPEC §13 report (M2 slice B).

Dual-parameterized over both storage adapters (memory / lancedb) via the
``adapter`` fixture, reusing the conformance suite's adapter-construction
helpers — same precedent ``tests/test_projection_rebuild.py`` and
``tests/test_engine_feedback.py`` already establish.

Falsifiability (task brief's explicit requirement): every count asserted
here is hand-computed from the seeded fixture, never merely "the call did
not raise" — a wrong sum, a wrong denominator, or a swapped dead/unused
list would fail these tests, not just a missing method.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, Scope, new_id
from tabularium.engine.doctor import build_doctor_report
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
    _reopen_at_model,
)
from tests.support.doubles import TestClock, reopen_memory_at

AnyDoctorAdapter = MemoryAdapter | LanceDBAdapter

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DIM = 4


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _fact(
    *,
    scope: str = "proj-a",
    lifecycle: str = "active",
    validity: str = "valid",
    content: str = "doctor test fact",
    purpose: str | None = None,
    audience_hint: tuple[str, ...] | None = None,
    created_at: datetime = EPOCH,
    **kw: Any,
) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        purpose=purpose,
        audience_hint=audience_hint,
        created_at=created_at,
        updated_at=created_at,
        **kw,
    )


def _decision(
    *,
    scope: str = "proj-a",
    lifecycle: str = "in_force",
    validity: str = "valid",
    content: str = "doctor test decision",
    purpose: str | None = None,
    created_at: datetime = EPOCH,
    **kw: Any,
) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        purpose=purpose,
        created_at=created_at,
        updated_at=created_at,
        **kw,
    )


def _vec(seed: float = 1.0, dim: int = _DIM) -> list[float]:
    import math

    v = [seed] * dim
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


@pytest.fixture(params=["memory", "lancedb"])
async def adapter(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> AnyDoctorAdapter:
    """A fresh adapter, parameterized over both {"memory", "lancedb"}."""
    if request.param == "memory":
        return _make_memory_adapter(dim=_DIM)
    return await _make_lancedb_adapter(tmp_path, dim=_DIM)


# ===========================================================================
# Section 1 — facet coverage
# ===========================================================================


class TestFacetCoverage:
    async def test_empty_store_reports_zero_not_an_error(
        self, adapter: AnyDoctorAdapter
    ) -> None:
        report = await build_doctor_report(adapter, EngineConfig())
        fc = report.facet_coverage
        assert fc.scanned == 0
        assert fc.fact_count == 0
        assert fc.decision_count == 0
        assert fc.with_purpose == 0
        assert fc.with_audience_hint == 0

    async def test_matches_hand_computed_coverage(
        self, adapter: AnyDoctorAdapter
    ) -> None:
        """5 Facts + 2 Decisions, mixed purpose/audience_hint — every
        reported number is checked against a value computed by hand from
        the exact same seed, not merely "some number came back".
        """
        for name in ("constraint", "convention", "pitfall", "preference", "reference"):
            await adapter.create_facet(name)
        await adapter.create_role("implementer")

        # 3 Facts carry "constraint", 1 carries "pitfall", 1 carries none.
        facts = [
            _fact(content="f1", purpose="constraint", audience_hint=("implementer",)),
            _fact(content="f2", purpose="constraint"),
            _fact(content="f3", purpose="constraint"),
            _fact(content="f4", purpose="pitfall", audience_hint=("implementer",)),
            _fact(content="f5", purpose=None),
        ]
        for f in facts:
            await adapter.upsert(f, _vec(seed=hash(f.content) % 97 + 1))

        # 1 Decision carries "convention", 1 carries none.
        decisions = [
            _decision(content="d1", purpose="convention"),
            _decision(content="d2", purpose=None),
        ]
        for d in decisions:
            await adapter.upsert(d, _vec(seed=hash(d.content) % 97 + 2))

        report = await build_doctor_report(adapter, EngineConfig())
        fc = report.facet_coverage

        assert fc.scanned == 7  # denominator: 5 facts + 2 decisions
        assert fc.fact_count == 5
        assert fc.decision_count == 2
        assert fc.with_purpose == 5  # 3 constraint + 1 pitfall + 1 convention
        assert fc.purpose_counts["constraint"] == 3
        assert fc.purpose_counts["pitfall"] == 1
        assert fc.purpose_counts["convention"] == 1
        assert fc.purpose_counts["preference"] == 0
        assert fc.purpose_counts["reference"] == 0
        assert fc.with_audience_hint == 2  # f1 and f4


# ===========================================================================
# Section 2 — dead-vocabulary flags
# ===========================================================================


class TestDeadVocabulary:
    async def test_all_registered_values_are_dead_today(
        self, adapter: AnyDoctorAdapter
    ) -> None:
        """No code path in this build branches on a purpose value (the
        Curator — the only design consumer — is deferred). This is a
        structural, code-level fact, independent of any saved data: even a
        value that IS used in the data (below) is still reported dead.
        """
        for name in ("constraint", "convention"):
            await adapter.create_facet(name)
        await adapter.upsert(
            _fact(content="uses constraint", purpose="constraint"), _vec(1.0)
        )

        report = await build_doctor_report(adapter, EngineConfig())
        dv = report.dead_vocabulary
        assert dv.registered == ["constraint", "convention"]
        assert dv.dead == ["constraint", "convention"]

    async def test_unused_in_data_is_a_distinct_axis_from_dead(
        self, adapter: AnyDoctorAdapter
    ) -> None:
        """``constraint`` has a live occurrence (data-level: NOT unused) but
        is still ``dead`` (code-level: no policy row consumes it — same as
        every other registered value). ``convention`` has neither.
        """
        for name in ("constraint", "convention"):
            await adapter.create_facet(name)
        await adapter.upsert(
            _fact(content="uses constraint", purpose="constraint"), _vec(1.0)
        )

        report = await build_doctor_report(adapter, EngineConfig())
        dv = report.dead_vocabulary

        assert "constraint" in dv.dead
        assert "constraint" not in dv.unused_in_data
        assert "convention" in dv.dead
        assert "convention" in dv.unused_in_data


# ===========================================================================
# Section 3 — park counts [A7]
# ===========================================================================


class TestAwaitingEmbedding:
    async def test_empty_store_reports_zero(self, adapter: AnyDoctorAdapter) -> None:
        report = await build_doctor_report(adapter, EngineConfig())
        assert report.awaiting_embedding.count == 0

    async def test_up_to_date_rows_report_zero(self, adapter: AnyDoctorAdapter) -> None:
        await adapter.upsert(_fact(content="up to date"), _vec(1.0))
        await adapter.upsert(_fact(content="also up to date"), _vec(2.0))

        report = await build_doctor_report(adapter, EngineConfig())
        assert report.awaiting_embedding.count == 0

    async def test_matches_hand_computed_backfill_target_count(
        self, adapter: AnyDoctorAdapter, tmp_path: pathlib.Path
    ) -> None:
        """A row stamped with a stale ``embedding_model`` counts as awaiting
        embedding — the same target list ``backfill_targets()`` (and, once
        the backfill pass exists, the pass itself) consumes.  Expressed via
        a second adapter view opened at a different model (the same
        ``_reopen_at_model`` idiom the conformance suite's ``TestBackfill``
        uses), never by mutating adapter internals.
        """
        clock = TestClock(EPOCH)
        stale = _fact(content="needs re-embedding")
        await adapter.upsert(stale, _vec(1.0))

        if isinstance(adapter, MemoryAdapter):
            adapter_v2: AnyDoctorAdapter = reopen_memory_at(
                adapter, embedding_model="nomic-v2"
            )
        else:
            assert isinstance(adapter, LanceDBAdapter)
            adapter_v2 = await _reopen_at_model(
                adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock, dim=_DIM
            )
        # Written AFTER reopening — stamped at the now-current model, so it
        # is not itself a target.
        await adapter_v2.upsert(_fact(content="up to date"), _vec(2.0))

        report = await build_doctor_report(adapter_v2, EngineConfig())
        assert report.awaiting_embedding.count == 1
        targets = await adapter_v2.backfill_targets()
        assert stale.id in {t.id for t in targets}


# ===========================================================================
# Section 4 — projection drift [A9]
# ===========================================================================


class TestProjectionDrift:
    async def test_clean_store_reports_zero_drift(
        self, adapter: AnyDoctorAdapter
    ) -> None:
        await adapter.upsert(_fact(content="clean"), _vec(1.0))
        report = await build_doctor_report(adapter, EngineConfig())
        pd = report.projection_drift
        assert pd.mode == "report"
        assert pd.scanned == 1
        assert pd.drifted == []
        assert pd.repaired == 0

    async def test_surfaces_a_seeded_drift_and_never_writes(
        self, adapter: AnyDoctorAdapter
    ) -> None:
        """Corrupt a stored ``recall_count`` directly via
        ``update_projections`` — bypassing the normal write path, exactly
        the "materialized state disagrees with the log" condition this
        section exists to detect. ``doctor`` must surface it AND must never
        write (report mode, always — ``doctor`` observes, never repairs).
        """
        f = _fact(content="will drift")
        await adapter.upsert(f, _vec(1.0))
        clean = _fact(content="stays clean")
        await adapter.upsert(clean, _vec(2.0))

        await adapter.update_projections(
            f.id,
            recall_count=99,
            last_referenced_at=None,
            feedback_score=0.0,
            decay_due_at=None,
        )

        report = await build_doctor_report(adapter, EngineConfig())
        pd = report.projection_drift

        assert pd.mode == "report"
        drifted_ids = {d.id for d in pd.drifted}
        assert f.id in drifted_ids
        assert clean.id not in drifted_ids
        assert pd.repaired == 0

        # Never-writes postcondition: the corrupted value is still there.
        still_corrupted = await adapter.get_by_id(f.id)
        assert still_corrupted.recall_count == 99
