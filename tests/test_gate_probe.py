"""Shared dedup-gate probe (``engine/gate.py``) — 2026-08-10 cleaner plan.

Coverage:
  - Fact arm: candidate selection through the extracted probe is unchanged —
    raw cosine wins even when it diverges from fused RRF rank (the exact
    thing the gate's inline code did, and the exact thing the extraction
    must not silently swap for the read-path ranking).
  - Decision arm: same property, same fixture shape.
  - ``probe_candidates``'s ``validity`` override: omitted reproduces the
    gate's both-validities default; passed as ``frozenset({"valid"})``
    excludes an invalid near-neighbour entirely (the falsity sweep's use).
  - ``recall()`` / ``decisions_query()`` are untouched by this extraction —
    asserted structurally (neither module imports ``engine.gate``, and
    their own hybrid-retrieval code paths are unchanged).

Hermetic lane only — ``MemoryAdapter`` + ``FakeEmbedder``/table embedders, no
network, no disk (same convention as ``test_engine*.py``).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, Scope, new_id
from tabularium.engine import Tabularium
from tabularium.engine.gate import probe_candidates
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import TestClock

# ---------------------------------------------------------------------------
# Shared fixtures (angle-vector pattern lifted from test_engine_m1.py /
# test_engine_b2.py — a controlled, non-hash-based way to place records at
# known cosine similarities to a probe vector).
# ---------------------------------------------------------------------------

_DIM = 16


def _make_adapter(dim: int = _DIM) -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic", dim=dim)


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"candidate_k": 1000, "gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


def _angle_vec(angle_deg: float, dim: int = _DIM) -> list[float]:
    """Unit vector at ``angle_deg`` from axis 0, zero-padded to ``dim``.

    Two angle-parameterised vectors ``theta`` apart have cosine ``cos(theta)``
    exactly — lets a fixture place candidates at known, distinct raw-cosine
    similarities to a single probe vector.
    """
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


class _TableEmbedder:
    """Deterministic embedder: exact-string -> vector lookup."""

    dim: int = _DIM
    model: str = "test/table"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


def _fact(
    *,
    content: str,
    scope: str = "global",
    lifecycle: str = "active",
    validity: str = "valid",
    now: datetime,
) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
    )


def _decision(
    *,
    content: str,
    scope: str = "global",
    lifecycle: str = "in_force",
    validity: str = "valid",
    now: datetime,
) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Serial-probe fixture (P7 admission-order piece):
# vector top-k, BM25 narrows inside it by SET INTERSECTION — never a fused
# rank, never a BM25 re-rank.
#
#   Candidate VECTOR_ONLY — 5deg from the probe vector (cosine ~0.9962),
#     content shares NO tokens with the probe query -> in the vector top-k,
#     but absent from the BM25 leg entirely -> NOT a survivor (a vector hit
#     absent from BM25 is dropped).
#   Candidate SURVIVOR — 8deg (cosine ~0.9903), content DOES share tokens
#     with the probe query -> in BOTH legs' top-k -> the one survivor.
#   Candidate BM25_ONLY_OUTSIDE_TOPK — 40deg (cosine ~0.766), content also
#     shares tokens with the probe query (BM25 would rank it), but at
#     ``probe_k=2`` it is the THIRD-closest vector — outside the vector
#     leg's top-k entirely -> NOT a survivor (a BM25 hit outside the vector
#     top-k is lost, never re-ranked in on BM25 alone).
#
# With ``probe_k=2``: vector top-2 = {VECTOR_ONLY, SURVIVOR}; BM25 top-2 (the
# only two candidates with any token overlap) = {SURVIVOR, BM25_ONLY_OUTSIDE_TOPK}.
# Intersection = {SURVIVOR} — exactly one candidate clears both legs.
# ---------------------------------------------------------------------------

_PROBE_QUERY = "widget frobnicate gizmo assembly"
_CAND_VECTOR_ONLY_CONTENT = "alpha unrelated filler content zzzqxyz"
_CAND_SURVIVOR_CONTENT = "widget frobnicate gizmo maintenance notes"
_CAND_BM25_ONLY_OUTSIDE_TOPK_CONTENT = "widget frobnicate gizmo outlier report"

_COS_SURVIVOR = math.cos(math.radians(8))

# Ordering fixture: two candidates that BOTH survive (shared tokens, both
# inside the vector top-k) at different cosines — the bounce must name the
# higher one, proving survivors are ordered by cosine, not by BM25 score or
# insertion order.
_CAND_HIGHER_CONTENT = "koala eucalyptus habitat range"
_CAND_LOWER_CONTENT = "koala eucalyptus habitat territory"
_ORDERING_QUERY = "koala eucalyptus habitat survey"
_COS_HIGHER = math.cos(math.radians(5))
_COS_LOWER = math.cos(math.radians(15))

# Retained for ``TestProbeCandidatesValidityOverride`` below, which pins
# ``gate.probe_candidates`` itself (unaffected by the serial-probe rewrite
# above — falsity.py/edit.py still call it directly) against this same
# angle-5/30 fixture shape.
_COS_A = math.cos(math.radians(5))
_COS_B = math.cos(math.radians(30))


class TestFactArmSerialProbeRule:
    async def test_survivor_is_the_one_candidate_in_both_legs(self) -> None:
        """A vector hit absent from BM25 is dropped; a BM25 hit outside the
        vector top-k is dropped; only the intersection survives. This is
        also the fixture the Verify step's deliberate break (re-rank by
        BM25 score instead of intersecting) flips."""
        adapter = _make_adapter()
        clock = _make_clock()
        now = clock.now()

        vector_only = _fact(content=_CAND_VECTOR_ONLY_CONTENT, now=now)
        survivor = _fact(content=_CAND_SURVIVOR_CONTENT, now=now)
        bm25_only = _fact(content=_CAND_BM25_ONLY_OUTSIDE_TOPK_CONTENT, now=now)
        await adapter.upsert(vector_only, _angle_vec(5))
        await adapter.upsert(survivor, _angle_vec(8))
        await adapter.upsert(bm25_only, _angle_vec(40))

        cfg = _make_config(gray_floor=0.0, probe_k=2)
        embedder = _TableEmbedder({_PROBE_QUERY: _angle_vec(0)})
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save(_PROBE_QUERY, scope="global")

        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.candidate_id == survivor.id
        assert r.bounce.similarity is not None
        assert math.isclose(r.bounce.similarity, _COS_SURVIVOR, abs_tol=1e-6)

    async def test_survivors_order_by_cosine(self) -> None:
        """Two candidates both clear both legs — the bounce names the
        higher-cosine one."""
        adapter = _make_adapter()
        clock = _make_clock()
        now = clock.now()

        higher = _fact(content=_CAND_HIGHER_CONTENT, now=now)
        lower = _fact(content=_CAND_LOWER_CONTENT, now=now)
        await adapter.upsert(higher, _angle_vec(5))
        await adapter.upsert(lower, _angle_vec(15))

        cfg = _make_config(gray_floor=0.0, probe_k=10)
        embedder = _TableEmbedder({_ORDERING_QUERY: _angle_vec(0)})
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save(_ORDERING_QUERY, scope="global")

        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.candidate_id == higher.id
        assert r.bounce.similarity is not None
        assert math.isclose(r.bounce.similarity, _COS_HIGHER, abs_tol=1e-6)


class TestDecisionArmSerialProbeRule:
    async def test_survivor_is_the_one_candidate_in_both_legs(self) -> None:
        """Same serial-probe rule, Decision arm, at its own floor."""
        adapter = _make_adapter()
        clock = _make_clock()
        now = clock.now()

        vector_only = _decision(content=_CAND_VECTOR_ONLY_CONTENT, now=now)
        survivor = _decision(content=_CAND_SURVIVOR_CONTENT, now=now)
        bm25_only = _decision(content=_CAND_BM25_ONLY_OUTSIDE_TOPK_CONTENT, now=now)
        await adapter.upsert(vector_only, _angle_vec(5))
        await adapter.upsert(survivor, _angle_vec(8))
        await adapter.upsert(bm25_only, _angle_vec(40))

        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.0, probe_k=2)
        embedder = _TableEmbedder({_PROBE_QUERY: _angle_vec(0)})
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save(_PROBE_QUERY, kind="decision", scope="global")

        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.candidate_id == survivor.id
        assert r.bounce.similarity is not None
        assert math.isclose(r.bounce.similarity, _COS_SURVIVOR, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# probe_candidates: validity override
# ---------------------------------------------------------------------------


class TestProbeCandidatesValidityOverride:
    async def _seed(self, adapter: MemoryAdapter, now: datetime) -> tuple[Fact, Fact]:
        """An invalid near-neighbour closer to the probe than the valid one."""
        invalid_fact = _fact(
            content="invalid near neighbour", validity="invalid", now=now
        )
        valid_fact = _fact(content="valid further neighbour", now=now)
        await adapter.upsert(invalid_fact, _angle_vec(5))  # closer -> higher cosine
        await adapter.upsert(valid_fact, _angle_vec(30))
        return invalid_fact, valid_fact

    async def test_omitted_override_reproduces_both_validities_default(self) -> None:
        """Omitting ``validity`` reproduces the gate's default: the invalid
        near-neighbour is visible and wins on raw cosine (unaffected by
        omission — existing callers see byte-identical behaviour)."""
        adapter = _make_adapter()
        clock = _make_clock()
        now = clock.now()
        invalid_fact, _valid_fact = await self._seed(adapter, now)

        cfg = _make_config()
        result = await probe_candidates(
            content=_PROBE_QUERY,
            vector=_angle_vec(0),
            span=[Scope("global")],
            k=cfg.candidate_k,
            adapter=adapter,
            cfg=cfg,
        )

        assert result.best_hit is not None
        assert result.best_hit.record.id == invalid_fact.id
        assert math.isclose(result.best_cosine, _COS_A, abs_tol=1e-6)

    async def test_valid_only_override_excludes_invalid_neighbour(self) -> None:
        """``validity=frozenset({"valid"})`` yields valid-only candidates —
        the invalid near-neighbour the gate would return (and the falsity
        sweep must not) is excluded from the probe entirely, not merely
        deprioritized: the valid, farther candidate wins instead."""
        adapter = _make_adapter()
        clock = _make_clock()
        now = clock.now()
        _invalid_fact, valid_fact = await self._seed(adapter, now)

        cfg = _make_config()
        result = await probe_candidates(
            content=_PROBE_QUERY,
            vector=_angle_vec(0),
            span=[Scope("global")],
            k=cfg.candidate_k,
            validity=frozenset({"valid"}),
            adapter=adapter,
            cfg=cfg,
        )

        assert result.best_hit is not None
        assert result.best_hit.record.id == valid_fact.id
        assert math.isclose(result.best_cosine, _COS_B, abs_tol=1e-6)
        assert all(h.record.validity == "valid" for h in result.v_hits)
        assert all(h.record.validity == "valid" for h in result.t_hits)


# ---------------------------------------------------------------------------
# recall() / decisions_query() untouched (out-of-scope guarantee)
# ---------------------------------------------------------------------------


class TestReadPathsUntouched:
    def test_recall_module_does_not_import_gate(self) -> None:
        """``engine/recall.py`` must never depend on the gate's probe —
        different filters, RRF x importance ranking, and a write side effect
        (archived-hit revival) that would turn a sweep's scan into a write."""
        import tabularium.engine.recall as recall_mod

        assert "gate" not in recall_mod.__dict__
        assert not hasattr(recall_mod, "probe_candidates")
        assert not hasattr(recall_mod, "select_best_candidate")

    def test_decisions_query_does_not_use_gate_probe(self) -> None:
        """``decisions_query`` builds its own ``Filters`` and ranks through
        ``recall.fuse_and_boost`` (RRF x importance path) — it must not route
        through ``probe_candidates``/``select_best_candidate``."""
        import inspect

        from tabularium.engine.decision_read import decisions_query

        source = inspect.getsource(decisions_query)
        assert "probe_candidates" not in source
        assert "select_best_candidate" not in source
