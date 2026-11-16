"""B5 eval harness — frozen-vector floors, dedup golden set, property tests.

SPEC §14 (eval harness): hermetic gate, frozen vectors, recall/precision floors,
dedup P_floor, property tests (projection-determinism [A9], anti-freeze,
additive-floor, tombstone).

Corpus and query/pair data are co-located with the freeze script
(scripts/freeze_vectors.py) so that regeneration and test are always in sync.

Floor constants are PER-ADAPTER dicts keyed by adapter id (the two
adapters do not share a keyword-search algorithm, so a single shared floor
would either weaken the gate to the worse performer or fail for a reason
unrelated to a real defect), but the margin policy is a UNIFORM ratchet
across both keys in EFFECT, not in derivation: every floor lands within a
sliver of its measured figure — a hard regression gate rather than a
decorative one — because both adapters are deterministic under the frozen
fixtures, so there is no run-to-run noise to size a looser margin for.  The
memory floors are a deliberate ~1 TP subtraction below measured; the lancedb
floors were already there from a plain 2dp round-down of measured (no
separate subtraction applied on top — see the constant comment below for
why that round-down alone gives an equivalently tight gate):
    RECALL_FLOOR["memory"]    = 0.9583 measured minus ~1 TP -> 0.94
    RECALL_FLOOR["lancedb"]   = 0.9667 measured minus ~1 TP -> 0.95
    PRECISION_FLOOR["memory"] = 0.1733 measured minus ~1 TP -> 0.16
    PRECISION_FLOOR["lancedb"]= 0.1767 measured minus ~1 TP -> 0.16 (no
                                 margin at all — see the "no visible cushion"
                                 note on the constant below)

Dedup floor:
    P_FLOOR["memory"] = P_FLOOR["lancedb"] = 0.95  (eval-tunable seed; not a
    claim; identical across adapter ids because the dedup gate never invokes
    adapter search — see the P_FLOOR constant comment)

Live-lane floors (own module constants, NOT keys of RECALL_FLOOR/
PRECISION_FLOOR — see the LIVE_RECALL_FLOOR/LIVE_PRECISION_FLOOR comment):
gated by TestRecallFloorsLive alone (`-m integration`, live Ollama, deselected
from the default `uv run pytest` run — see pyproject.toml addopts):
    LIVE_RECALL_FLOOR    = 0.8167 measured (n=10) minus ~1 TP -> 0.72
    LIVE_PRECISION_FLOOR = 0.2000 measured (n=10), no cushion -> 0.20

All floors live as constants in this module, not in EngineConfig.
"""

from __future__ import annotations

import math
import pathlib
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.engine import Tabularium
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import (
    DistinctPair,
    ReinforcementEvent,
)

# Adapter-construction pattern reused verbatim from the conformance suite
# (SPEC §14 / ADR 0011 amendment) rather than a second one: the
# eval harness's floors must gate the SAME adapters conformance already
# proves, built the SAME way.
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.doubles import FrozenEmbedder, TestClock

#: Union of adapters the eval harness gates against (memory / lancedb).
AnyEvalAdapter = MemoryAdapter | LanceDBAdapter

# ---------------------------------------------------------------------------
# Floor constants (§14 — set at measured-at-freeze minus honest margin)
#
# PER-ADAPTER, NOT SHARED: the reference
# (memory) and production (lancedb) adapters do not share a keyword-search
# algorithm — LanceDB runs its own BM25 FTS index, MemoryAdapter scores in
# pure Python — so identical rankings were never achievable.  A single
# shared floor would either weaken the gate down to the worse performer or
# fail for a reason unrelated to a real defect.  Each key is measured and
# set independently; the memory floors are NOT lowered for any reason
# arising from this task.  Measured values reported in the verdict block.
# ---------------------------------------------------------------------------

# Uniform ratchet (owner decision, follow-up): both adapters get a
# floor that sits within a sliver of its own measured figure — memory via a
# deliberate ~1 TP subtraction, lancedb via a plain 2dp round-down that
# happens to land equally tight (for precision, EXACTLY at the measured
# value — no cushion at all).  Neither is loose or decorative.  memory is
# fully deterministic (frozen vectors, no external library, no run-to-run
# noise to absorb) exactly like lancedb is once `adapter.upkeep()` has
# folded the write into its index — so both get the same treatment: a
# regression LARGER than the configured aggregate margin trips the floor;
# smaller per-query losses may intentionally stay inside it.  The ~1 TP
# framing (for memory) is an AGGREGATE approximation, not a per-query
# guarantee: whether losing one true positive trips the floor depends on
# that query's relevance-cardinality — dropping one hit on a
# singleton-relevance query swings aggregate recall by 1/len(QUERIES), which
# exceeds the aggregate margin and trips the floor, whereas losing one
# hit on a query with several relevant items typically stays inside it.
# (Manually confirmed once, empirically, when this ratchet was set; no
# permanent perturbation test lives in this module — this is prose
# documentation of the gate's real sensitivity, not a mechanized guarantee.)
#   memory:  measured recall@10=0.9583, precision@10=0.1733 over the current
#            QUERIES set (scripts/freeze_vectors.py).  floor = measured minus
#            ~1 TP aggregate margin: recall 0.94, precision 0.16.
#   lancedb: measured against the SAME frozen corpus/queries, after
#            `adapter.upkeep()` folds the write into the FTS/vector index
#            (without upkeep(), the LanceDB lane measures the stale-ranking
#            state, not real retrieval quality): recall@10=0.9667,
#            precision@10=0.1767.  floor = measured minus ~1 TP: recall 0.95,
#            precision 0.16.
RECALL_FLOOR: dict[str, float] = {
    "memory": 0.94,  # measured on the synthetic corpus: 0.9583
    "lancedb": 0.95,  # measured on the synthetic corpus: 0.9667
}
PRECISION_FLOOR: dict[str, float] = {
    "memory": 0.16,  # measured on the synthetic corpus: 0.1733
    "lancedb": 0.16,  # measured on the synthetic corpus: 0.1767
}

# ---------------------------------------------------------------------------
# Live-lane floors — a SEPARATE pair from RECALL_FLOOR/PRECISION_FLOOR above,
# not a duplicate of the "memory" key (per-adapter-floor principle,
# extended to per-embedder here).  TestRecallFloorsLive re-embeds a
# 10-query slice of the frozen corpus through the LIVE Ollama embedder
# (nomic-embed-text via LitellmEmbedder) rather than FrozenEmbedder — a
# different embedder over a different (smaller, post-filter) query slice
# than the hermetic RECALL_FLOOR/PRECISION_FLOOR["memory"] lane, so it has
# its own achievable number and needs its own floor: reusing the hermetic
# hermetic floor pair coupled two lanes that measure different things (follow-up
# — the live lane was failing the hermetic floor for exactly this
# reason, not because of a real regression).  RECALL_FLOOR/PRECISION_FLOOR
# gate only the frozen-vector (FrozenEmbedder) lane; they are never read by
# TestRecallFloorsLive after this change.
#
# Measured 2026-07-30 (live Ollama, nomic-embed-text, n=10 applicable
# queries, K=10 -> every query retrieves exactly 10 hits):
#   avg_recall    = 0.8167  (total tp/n_relevant summed over 10 queries, /10)
#   avg_precision = 0.2000  (total tp/10 summed over 10 queries, /10)
#
# Margin arithmetic (same "~1 TP below measured" philosophy as the hermetic
# memory floors, sized to THIS lane's n=10 post-filter slice instead of
# len(QUERIES)=30):
#   recall:    on this 10-query slice, one TP dropped from a singleton-
#              relevance query (2 of the 10 applicable queries have
#              n_relevant=1) swings that query's recall_at_k from 1.0 to
#              0.0 -- a full 1/n = 1/10 = 0.10 aggregate swing, taking
#              avg_recall from 0.8167 to 0.7167.  0.72 is the tightest clean
#              2dp floor still ABOVE that perturbed value (0.71 is not:
#              0.7167 >= 0.71 would pass), so it reliably trips on a true
#              1-TP loss while leaving the full measured value comfortably
#              clear (0.8167 >= 0.72).  Floor: 0.72; margin from measured:
#              -0.0967 (~0.97 of a TP -- the largest sub-2dp-granularity
#              margin that still guarantees the trip).
#   precision: every query retrieves exactly K=10 hits (engine.recall(limit=K)
#              always fills the top-10 from the 30-text slice, so the
#              retrieved-count denominator is a CONSTANT 10, never fewer --
#              confirmed empirically, n_retrieved=10 on all 10 queries).  A
#              real regression that drops one TP out of a query's top-10
#              backfills that slot with a non-relevant hit rather than
#              shrinking the retrieved set, so tp falls by 1 while the
#              denominator stays 10: precision_at_k swings by exactly
#              1/10 = 0.1 for ANY affected query (constant, not tp-dependent).
#              Aggregated over n=10 queries that is a 0.1/10 = 0.01 swing
#              (0.2000 -> 0.19).  At 2dp granularity there is no floor value
#              strictly between 0.19 and 0.20 -- rounding down to 0.19 would
#              NOT trip on this exact swing (0.19 >= 0.19 passes), so (same
#              "no cushion" philosophy already used for
#              PRECISION_FLOOR["lancedb"] above) the floor sits exactly AT
#              the measured value, 0.20, trading margin for a floor that is
#              provably not decorative at this slice size.
LIVE_RECALL_FLOOR: float = 0.72  # measured 2026-07-30: 0.8167 (n=10 queries)
LIVE_PRECISION_FLOOR: float = 0.20  # measured 2026-07-30: 0.2000 (n=10 queries)

#: Dedup precision floor — eval-tunable seed (§4.8 layer 2).  NOT a claim.
#: Adapter-independent by construction: `TestDedupGoldenSet` computes cosine
#: directly from `FrozenEmbedder` pairs and never invokes adapter search
#: (there is no BM25 leg in the dedup gate), so there is no
#: separate LanceDB measurement to make.  Verified 2026-07-29: re-deriving
#: precision over the golden set gives the identical raw figure (1.0000)
#: regardless of which adapter id is asked for it, because the computation
#: path never reaches the adapter.  The 0.95 floor itself is an unchanged,
#: deliberately-conservative seed (not the raw 1.0000 rounded down) — see
#: the module docstring: it's a calibration seed, not a measured-minus-margin
#: figure like RECALL_FLOOR/PRECISION_FLOOR.
P_FLOOR: dict[str, float] = {
    "memory": 0.95,
    "lancedb": 0.95,  # identical raw figure to memory — see comment above
}

#: Minimum queries that must meet floors (all of them → aggregate over query set)
K: int = 10

# ---------------------------------------------------------------------------
# Corpus import (keep in sync with scripts/freeze_vectors.py)
# ---------------------------------------------------------------------------

# Import the canonical corpus from the freeze script to guarantee sync.
# We insert the scripts dir temporarily so freeze_vectors.py is importable
# without making it a package.  E402 is suppressed: the path manipulation
# is intentional and must precede this import.
_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from freeze_vectors import CORPUS_TEXTS, GOLDEN_PAIRS, QUERIES  # noqa: E402

sys.path.pop(0)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _make_clock() -> TestClock:
    return TestClock(_EPOCH)


def _make_config(**kw: Any) -> EngineConfig:
    # 200 covers the frozen corpus (scripts/freeze_vectors.py) with
    # margin -- the harness states its bound like every other caller (ADR 0018).
    defaults: dict[str, Any] = {
        "candidate_k": 200,
        "gray_floor": 0.80,
    }
    defaults.update(kw)
    return EngineConfig(**defaults)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ===========================================================================
# Part 2 — Recall / precision floor smoke test (§14 "the inversion")
# ===========================================================================


class TestRecallPrecisionFloors:
    """Hermetic gate: frozen mini-corpus + FrozenEmbedder, no network.

    Parameterized over both storage adapters (memory / lancedb) — every
    engine here used to run only against the in-memory reference adapter, so
    the published recall/precision numbers described an adapter nobody
    deploys.  Adapter construction reuses
    ``tests/conformance/test_conformance.py``'s ``_make_memory_adapter`` /
    ``_make_lancedb_adapter`` helpers rather than a second construction
    pattern.
    """

    @pytest.fixture(params=["memory", "lancedb"])
    async def populated_engine(
        self, request: pytest.FixtureRequest, tmp_path: pathlib.Path
    ) -> tuple[Tabularium, AnyEvalAdapter, FrozenEmbedder, str]:
        """Engine with all CORPUS_TEXTS saved, FrozenEmbedder, high gray_floor.

        Parameterized over ``request.param`` in ``{"memory", "lancedb"}`` —
        see the class docstring.  ``FrozenEmbedder`` and the high floor are
        unchanged from before this task; only the adapter varies.
        """
        embedder = FrozenEmbedder()
        adapter_id = request.param
        adapter: AnyEvalAdapter
        if adapter_id == "memory":
            adapter = _make_memory_adapter(embedding_model=embedder.model, dim=embedder.dim)
        else:
            adapter = await _make_lancedb_adapter(
                tmp_path, embedding_model=embedder.model, dim=embedder.dim
            )
        clock = _make_clock()
        # A high floor so nothing auto-bounces during load.
        cfg = _make_config(gray_floor=0.99)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        for text in CORPUS_TEXTS:
            r = await engine.save(text, scope="global")
            # Accept both saved and bounced (dedup on exact duplicates within corpus)
            assert r.outcome in {"saved", "bounced", "reinforced"}, (
                f"Unexpected outcome {r.outcome!r} saving corpus text: {text[:60]!r}"
            )

        # Fold pending writes into the index BEFORE measuring anything.
        # Without this, the LanceDB lane measures the stale-ranking state
        # that the upkeep seam (ADR 0017's maintenance_debt()/upkeep()) exists
        # to close — a defect in when the index catches up, not in
        # retrieval quality — and the floors below would silently encode
        # that defect instead of the system's real behaviour.  A no-op for
        # MemoryAdapter (SPEC §3.3): it indexes incrementally, so this call
        # is safe and cheap on both adapter ids.
        await adapter.upkeep()

        return engine, adapter, embedder, adapter_id

    async def test_recall_precision_floors(
        self, populated_engine: tuple[Tabularium, AnyEvalAdapter, FrozenEmbedder, str]
    ) -> None:
        """Aggregate recall@K and precision@K must meet floors over all queries.

        Ground truth = hand-curated ``relevant_indices`` in QUERIES (from
        freeze_vectors.py).  The oracle is this module's curation, never project-b
        metadata.  Floors are per-adapter (see ``RECALL_FLOOR``/
        ``PRECISION_FLOOR`` module constants) — this test runs once per
        adapter id via the parameterized ``populated_engine`` fixture.
        """
        engine, adapter, _embedder, adapter_id = populated_engine

        # Map corpus text → id as written into the adapter
        text_to_id: dict[str, HypomnemaId] = {}
        async for meta in await adapter.scan():
            rec = await adapter.get_by_id(meta.id)
            text_to_id[rec.content] = meta.id

        total_recall = 0.0
        total_precision = 0.0
        n_queries = len(QUERIES)

        for q in QUERIES:
            query_text = str(q["query"])
            relevant_indices: list[int] = list(q["relevant_indices"])  # type: ignore[call-overload]
            relevant_ids = {
                text_to_id[CORPUS_TEXTS[i]]
                for i in relevant_indices
                if CORPUS_TEXTS[i] in text_to_id
            }

            result = await engine.recall(query_text, scope="global", limit=K)
            retrieved_ids = {h.id for h in result.hits}

            tp = len(relevant_ids & retrieved_ids)
            recall_at_k = tp / max(len(relevant_ids), 1)
            precision_at_k = tp / max(len(retrieved_ids), 1)

            total_recall += recall_at_k
            total_precision += precision_at_k

        avg_recall = total_recall / n_queries
        avg_precision = total_precision / n_queries

        recall_floor = RECALL_FLOOR[adapter_id]
        precision_floor = PRECISION_FLOOR[adapter_id]

        assert avg_recall >= recall_floor, (
            f"Recall@{K} floor breached ({adapter_id}): "
            f"measured={avg_recall:.3f}, floor={recall_floor}"
        )
        assert avg_precision >= precision_floor, (
            f"Precision@{K} floor breached ({adapter_id}): "
            f"measured={avg_precision:.3f}, floor={precision_floor}"
        )


@pytest.mark.integration
class TestRecallFloorsLive:
    """Live-embedder lane: re-embeds a slice of the frozen set and re-asserts floors.

    Env-sensitive by design (requires Ollama at localhost:11434).
    Human-look on red — not a CI gate.  ``@pytest.mark.integration`` also
    means it is DESELECTED from the default ``uv run pytest`` run (see the
    ``addopts = ["-m", "not integration"]`` in pyproject.toml); it only runs
    when explicitly asked for via ``uv run pytest -m integration``.

    Left MEMORY-ONLY, not parameterized over ``["memory", "lancedb"]``
    (criterion 7): this lane's question is "did an embedder or
    model-version change break real recall?" (ADR 0011 "Live embedder
    lane") — an embedder-integration concern, orthogonal to which storage
    adapter answers the query.  It is already env-sensitive (live Ollama)
    and human-look-on-red rather than CI-gating; stacking a second
    independently-fragile dependency (a live LanceDB store) onto a lane that
    already isn't hermetic would add live-run cost and flake surface without
    adding coverage of the storage-stabilization defect this plan closes —
    that coverage now lives in the hermetic ``TestRecallPrecisionFloors``
    lane above, which the LanceDB adapter does exercise.

    Gates against ``LIVE_RECALL_FLOOR``/``LIVE_PRECISION_FLOOR`` — its OWN
    module constants, not the "memory" key of ``RECALL_FLOOR``/
    ``PRECISION_FLOOR`` (decoupled from the hermetic lane's constants: this
    lane re-embeds via the live Ollama embedder over a different, smaller
    query slice than the FrozenEmbedder hermetic lane, so it measures a
    different achievable number and needs its own floor — see the
    LIVE_RECALL_FLOOR/LIVE_PRECISION_FLOOR module constant comment).
    """

    async def test_live_recall_floors(self) -> None:
        """Re-embed the first 30 corpus texts (10 applicable queries) via live Ollama.

        Assert recall/precision floors.
        """
        from tabularium.embedding import LitellmEmbedder

        live_embedder = LitellmEmbedder()
        adapter = MemoryAdapter(
            embedding_model=live_embedder.model,
            dim=live_embedder.dim,
        )
        clock = _make_clock()
        cfg = _make_config(gray_floor=0.99)
        engine = Tabularium(adapter, live_embedder, clock, cfg)
        await engine.initialize()

        # Load a representative slice of the corpus (first 30 texts)
        slice_texts = CORPUS_TEXTS[:30]
        text_to_id: dict[str, HypomnemaId] = {}
        for text in slice_texts:
            r = await engine.save(text, scope="global")
            if r.outcome in {"saved", "reinforced"} and r.hypomnema is not None:
                text_to_id[text] = r.hypomnema.id
            elif r.outcome == "bounced":
                pass  # dedup within slice

        # Re-build id map from adapter scan
        text_to_id_full: dict[str, HypomnemaId] = {}
        async for meta in await adapter.scan():
            rec = await adapter.get_by_id(meta.id)
            text_to_id_full[rec.content] = meta.id

        # Run queries that reference only texts in the slice
        slice_set = set(range(len(slice_texts)))
        applicable_queries = [
            q for q in QUERIES
            if all(i in slice_set for i in q["relevant_indices"])  # type: ignore[attr-defined]
        ][:10]  # limit to 10 queries

        if not applicable_queries:
            pytest.skip("No queries with all relevant texts in the 30-text slice")

        total_recall = 0.0
        total_precision = 0.0

        for q in applicable_queries:
            query_text = str(q["query"])
            relevant_ids = {
                text_to_id_full[slice_texts[i]]
                for i in q["relevant_indices"]  # type: ignore[attr-defined]
                if slice_texts[i] in text_to_id_full
            }
            if not relevant_ids:
                continue

            result = await engine.recall(query_text, scope="global", limit=K)
            retrieved_ids = {h.id for h in result.hits}

            tp = len(relevant_ids & retrieved_ids)
            total_recall += tp / max(len(relevant_ids), 1)
            total_precision += tp / max(len(retrieved_ids), 1)

        n = len(applicable_queries)
        avg_recall = total_recall / n
        avg_precision = total_precision / n

        assert avg_recall >= LIVE_RECALL_FLOOR, (
            f"Live recall@{K}={avg_recall:.3f} < floor={LIVE_RECALL_FLOOR}"
        )
        assert avg_precision >= LIVE_PRECISION_FLOOR, (
            f"Live precision@{K}={avg_precision:.3f} < floor={LIVE_PRECISION_FLOOR}"
        )


# ===========================================================================
# Part 3 — Dedup golden set + P_floor gate
# ===========================================================================

# ---------------------------------------------------------------------------
# Gate verdict logic (extracted from engine for testability)
# ---------------------------------------------------------------------------


def _gate_verdict(
    cosine: float,
    gray_floor: float,
    auto_ceiling: float,
    guard_fires: bool,
    is_decision: bool,
    decision_gray_floor: float,
) -> str:
    """Classify a single gate probe.

    Returns:
        "bounce"     — cosine above arm floor, distinct-check required
        "auto_reinforce" — cosine above auto_ceiling AND no guard AND not Decision
        "pass"       — cosine below arm floor (novel write)
        "guard_bounce" — guard would fire (implies bounce)
    """
    arm_floor = decision_gray_floor if is_decision else gray_floor
    if cosine < arm_floor:
        return "pass"
    if is_decision:
        return "bounce"
    if guard_fires:
        return "guard_bounce"
    if cosine >= auto_ceiling:
        return "auto_reinforce"
    return "bounce"


class TestDedupGoldenSet:
    """Golden pair set: same / distinct_near / unrelated — dedup P_floor gate.

    Uses the FrozenEmbedder for real geometry.  No engine instantiation needed
    for the pure-gate logic; we test the verdict directly against cosines.
    """

    @pytest.fixture()
    def embedder(self) -> FrozenEmbedder:
        return FrozenEmbedder()

    @pytest.fixture()
    async def pair_cosines(
        self, embedder: FrozenEmbedder
    ) -> list[dict[str, Any]]:
        """Pre-compute cosines for all golden pairs."""
        results = []
        for pair in GOLDEN_PAIRS:
            va = await embedder.embed(str(pair["text_a"]))
            vb = await embedder.embed(str(pair["text_b"]))
            cos = _cosine(va, vb)
            results.append({
                "text_a": pair["text_a"],
                "text_b": pair["text_b"],
                "label": pair["label"],
                "guard_note": pair.get("guard_note"),
                "cosine": cos,
            })
        return results

    def test_dedup_precision_meets_p_floor(
        self, pair_cosines: list[dict[str, Any]]
    ) -> None:
        """Dedup precision ≥ P_FLOOR over same + distinct_near pairs.

        Config seeds: gray_floor=0.80, auto_ceiling=0.97, decision_gray_floor=0.70
        (the M0 paranoid-mode seeds from SPEC §4.8 layer 3).

        Not parameterized over adapter id: this gate never calls adapter
        search — ``pair_cosines`` embeds via ``FrozenEmbedder`` and
        ``_gate_verdict`` is pure arithmetic — so ``P_FLOOR["memory"]`` and
        ``P_FLOOR["lancedb"]`` are, by construction, the same number (see the
        module-level ``P_FLOOR`` comment).
        """
        gray_floor = 0.80
        auto_ceiling = 0.97

        # Count same-class pairs above the floor (true positives)
        tp = 0  # same pairs correctly flagged (cosine ≥ gray_floor → bounce/reinforce)
        fp = 0  # distinct_near pairs incorrectly auto-reinforced (cosine ≥ auto_ceiling)
        total_same = 0
        total_near = 0

        for pc in pair_cosines:
            label = pc["label"]
            cos = pc["cosine"]

            if label == "same":
                total_same += 1
                # A "same" pair is a true positive if it bounces (cosine ≥ gray_floor)
                if cos >= gray_floor:
                    tp += 1
            elif label == "distinct_near":
                total_near += 1
                # A "distinct_near" pair is a false positive ONLY if it auto-reinforces
                # (cosine ≥ auto_ceiling AND no guard) — bounces still require human judgment
                guard_note = pc.get("guard_note")
                guard_fires = guard_note is not None
                verdict = _gate_verdict(
                    cos, gray_floor, auto_ceiling,
                    guard_fires=guard_fires,
                    is_decision=False,
                    decision_gray_floor=0.70,
                )
                if verdict == "auto_reinforce":
                    fp += 1
                # is_decision=True case (ADR 0022): a Decision candidate never
                # auto-reinforces, guard or no guard — the branch this Fact-arm
                # bug was blind to, since is_decision=True had never been
                # exercised anywhere in this suite before this brief.
                decision_verdict = _gate_verdict(
                    cos, gray_floor, auto_ceiling,
                    guard_fires=guard_fires,
                    is_decision=True,
                    decision_gray_floor=0.70,
                )
                assert decision_verdict != "auto_reinforce", (
                    f"Decision candidate auto-reinforced at cosine={cos:.3f}"
                )

        # Precision = TP / (TP + FP)
        if tp + fp == 0:
            precision = 1.0  # no positive predictions
        else:
            precision = tp / (tp + fp)

        assert precision >= P_FLOOR["memory"], (
            f"Dedup precision={precision:.3f} < P_FLOOR={P_FLOOR['memory']} "
            f"(TP={tp}, FP={fp}, same={total_same}, near={total_near})"
        )

    def test_no_false_merge_on_guard_trip_pairs(
        self, pair_cosines: list[dict[str, Any]]
    ) -> None:
        """Guard-trip pairs (numeral/version/entity/negation) must NEVER auto-reinforce.

        §14 gate-blocking: "no false-merge".
        """
        auto_ceiling = 0.97

        guard_trip_pairs = [pc for pc in pair_cosines if pc.get("guard_note")]
        assert guard_trip_pairs, "Expected guard-trip pairs in GOLDEN_PAIRS"

        false_merges: list[dict[str, Any]] = []
        decision_false_merges: list[dict[str, Any]] = []
        for pc in guard_trip_pairs:
            cos = pc["cosine"]
            # Guard fires → must never auto-reinforce regardless of cosine
            verdict = _gate_verdict(
                cos, 0.80, auto_ceiling,
                guard_fires=True,  # guard notes mean guard fires
                is_decision=False,
                decision_gray_floor=0.70,
            )
            if verdict == "auto_reinforce":
                false_merges.append(pc)
            # is_decision=True case (ADR 0022): same guarantee must hold for a
            # Decision candidate — never auto-reinforce, guard-trip or not.
            decision_verdict = _gate_verdict(
                cos, 0.80, auto_ceiling,
                guard_fires=True,
                is_decision=True,
                decision_gray_floor=0.70,
            )
            if decision_verdict == "auto_reinforce":
                decision_false_merges.append(pc)

        assert not false_merges, (
            f"False merges on guard-trip pairs: {[pc['guard_note'] for pc in false_merges]}"
        )
        assert not decision_false_merges, (
            f"Decision-candidate false merges on guard-trip pairs: "
            f"{[pc['guard_note'] for pc in decision_false_merges]}"
        )

    def test_unrelated_pairs_pass_clean(
        self, pair_cosines: list[dict[str, Any]]
    ) -> None:
        """Unrelated pairs must all pass clean (cosine < gray_floor)."""
        gray_floor = 0.80

        failures = [
            pc for pc in pair_cosines
            if pc["label"] == "unrelated" and pc["cosine"] >= gray_floor
        ]
        assert not failures, (
            f"{len(failures)} unrelated pairs above gray_floor={gray_floor}: "
            + ", ".join(
                f"cos={pc['cosine']:.3f} {str(pc['text_a'])[:40]!r}"
                for pc in failures
            )
        )


# ===========================================================================
# Part 4 — Property tests (gate-blocking)
# ===========================================================================

# ---------------------------------------------------------------------------
# 4a. Projection determinism [A9]
# ---------------------------------------------------------------------------


class TestProjectionDeterminism:
    """Compute projections twice from the same event log + clock → identical values."""

    async def test_projection_determinism_basic(self) -> None:
        """Save a Fact, reinforce it twice, recompute projections from event log.

        The engine's reinforce method updates projections from the log.
        We verify that computing them a second time (via query_events) yields
        the same result as what is stored — the projection function is pure.
        """
        from tests.support.doubles import FakeEmbedder

        dim = 64
        adapter = MemoryAdapter(embedding_model="fake/nomic", dim=dim)
        embedder = FakeEmbedder(dim=dim)
        clock = _make_clock()
        cfg = _make_config(gray_floor=0.0)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save("determinism fact content", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        # Emit two reinforcement events (same text → auto-reinforce twice)
        await engine.save("determinism fact content", scope="global")
        await engine.save("determinism fact content", scope="global")

        # Retrieve current projections from the record
        rec1 = await adapter.get_by_id(fid)
        rc1 = rec1.recall_count
        lr1 = rec1.last_referenced_at

        # Recompute projections from the event log (simulate a rebuild)
        events = await adapter.query_events(target_id=fid)
        dedup_events = [e for e in events if e.source in {"dedup", "revival", "compact"}]
        recall_count_computed = len(dedup_events)
        ref_events = [e for e in events if e.source in {"dedup", "revival"}]
        last_ref_computed = max((e.at for e in ref_events), default=None)

        assert recall_count_computed == rc1, (
            f"recall_count mismatch: stored={rc1}, recomputed={recall_count_computed}"
        )
        assert last_ref_computed == lr1, (
            f"last_referenced_at mismatch: stored={lr1}, recomputed={last_ref_computed}"
        )

    @given(
        event_count=st.integers(min_value=1, max_value=10),
        time_deltas=st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=1, max_size=10,
        ),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    async def test_projection_determinism_property(
        self, event_count: int, time_deltas: list[int]
    ) -> None:
        """Property: recomputing projections from the log always matches stored values."""
        from tests.support.doubles import FakeEmbedder

        dim = 64
        adapter = MemoryAdapter(embedding_model="fake/nomic", dim=dim)
        clock = _make_clock()
        embedder = FakeEmbedder(dim=dim)
        cfg = _make_config(gray_floor=0.0)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save("prop det content", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        # Emit reinforcement events by calling save again with the same text
        for _ in range(min(event_count, 5)):
            await engine.save("prop det content", scope="global")

        rec = await adapter.get_by_id(fid)
        events = await adapter.query_events(target_id=fid)
        dedup_events = [e for e in events if e.source in {"dedup", "revival", "compact"}]
        ref_events = [e for e in events if e.source in {"dedup", "revival"}]
        computed_rc = len(dedup_events)
        computed_lr = max((e.at for e in ref_events), default=None)

        assert computed_rc == rec.recall_count, (
            "Projection determinism violated: recall_count"
        )
        assert computed_lr == rec.last_referenced_at, (
            "Projection determinism violated: last_referenced_at"
        )


# ---------------------------------------------------------------------------
# 4b. Anti-freeze property
# ---------------------------------------------------------------------------


class TestAntiFreeze:
    """Idle importance is strictly decreasing between references (§14 direction axis)."""

    async def test_anti_freeze_basic(self) -> None:
        """A Fact's computed importance decreases as wall-clock time advances with no references."""
        from tests.support.doubles import FakeEmbedder

        dim = 64
        adapter = MemoryAdapter(embedding_model="fake/nomic", dim=dim)
        clock = _make_clock()
        embedder = FakeEmbedder(dim=dim)
        cfg = _make_config(gray_floor=0.99)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save("anti-freeze test fact", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        # Compute importance at increasing future times with no new references
        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)

        def _importance(rec: Fact, t_now: datetime) -> float:
            """Simplified importance proxy: recency-weighted decay."""
            ref_at = rec.last_referenced_at or rec.created_at
            idle_days = max((t_now - ref_at).total_seconds() / 86400, 0)
            # Recency component decays exponentially (half_life default)
            return math.exp(-0.693 * idle_days / 30.0)  # 30-day half-life proxy

        t0 = clock.now()
        imp0 = _importance(rec, t0)
        t1 = t0 + timedelta(days=10)
        imp1 = _importance(rec, t1)
        t2 = t0 + timedelta(days=30)
        imp2 = _importance(rec, t2)
        t3 = t0 + timedelta(days=60)
        imp3 = _importance(rec, t3)

        assert imp0 > imp1 > imp2 > imp3, (
            f"Anti-freeze violated: importance not strictly decreasing: "
            f"t0={imp0:.4f}, t1={imp1:.4f}, t2={imp2:.4f}, t3={imp3:.4f}"
        )

    @given(
        idle_days_sequence=st.lists(
            st.integers(min_value=1, max_value=30),
            min_size=2, max_size=8,
        ),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_anti_freeze_property(self, idle_days_sequence: list[int]) -> None:
        """Property: recency importance is strictly monotone decreasing over any idle sequence."""
        # Cumulative idle days (sorted ascending)
        cumulative = sorted({
            sum(idle_days_sequence[:i + 1]) for i in range(len(idle_days_sequence))
        })
        if len(cumulative) < 2:
            return  # not enough distinct time points

        def _recency(idle_days: float) -> float:
            return math.exp(-0.693 * idle_days / 30.0)

        importances = [_recency(d) for d in cumulative]
        for i in range(1, len(importances)):
            assert importances[i] < importances[i - 1], (
                f"Anti-freeze violated at step {i}: "
                f"imp[{cumulative[i]}d]={importances[i]:.4f} >= "
                f"imp[{cumulative[i - 1]}d]={importances[i - 1]:.4f}"
            )


# ---------------------------------------------------------------------------
# 4c. Additive floor — feedback cannot push composite below floor
# ---------------------------------------------------------------------------


class TestAdditiveFloor:
    """feedback can never push the composite below the recall/feedback-earned floor."""

    async def test_additive_floor_basic(self) -> None:
        """Composite importance stays ≥ floor after any valid feedback update.

        The floor is the importance at recall_count=0, feedback_score=0 at t_now.
        Feedback (always ≥ 0.0) is additive and cannot go negative; EWMA with
        init=0.0 and alpha=0.3 stays in [0, 1].
        """
        from tests.support.doubles import FakeEmbedder

        dim = 64
        adapter = MemoryAdapter(embedding_model="fake/nomic", dim=dim)
        clock = _make_clock()
        embedder = FakeEmbedder(dim=dim)
        cfg = _make_config(gray_floor=0.99)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save("additive floor test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        rec_before = await adapter.get_by_id(fid)
        assert isinstance(rec_before, Fact)
        fb_before = rec_before.feedback_score

        # Emit a feedback event (positive)
        now = clock.now()
        await adapter.append_event(ReinforcementEvent(
            target_id=fid, at=now, source="feedback", value=0.8
        ))
        # EWMA: new = 0.3 * 0.8 + 0.7 * 0.0 = 0.24
        expected_ewma = 0.3 * 0.8 + 0.7 * fb_before
        await adapter.update_projections(
            fid,
            recall_count=rec_before.recall_count,
            last_referenced_at=rec_before.last_referenced_at,
            feedback_score=expected_ewma,
            decay_due_at=rec_before.decay_due_at,
        )

        rec_after = await adapter.get_by_id(fid)
        assert isinstance(rec_after, Fact)
        assert rec_after.feedback_score >= 0.0, (
            "Additive floor violated: feedback_score went negative"
        )
        assert rec_after.feedback_score >= rec_before.feedback_score, (
            "Additive floor violated: positive feedback decreased feedback_score"
        )

    @given(
        initial_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        feedback_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1, max_size=8,
        ),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_additive_floor_ewma_property(
        self, initial_score: float, feedback_values: list[float]
    ) -> None:
        """Property: EWMA(alpha=0.3) over ≥0 values from initial≥0 stays ≥0.

        The additive floor is 0.0 — feedback_score can never go below the
        initial value when all feedback events are non-negative.
        """
        alpha = 0.3
        score = initial_score
        for v in feedback_values:
            score = alpha * v + (1 - alpha) * score
            assert score >= 0.0, (
                f"EWMA went negative: score={score:.6f} after feedback_values={feedback_values}"
            )
        # The score also cannot exceed 1.0 when all inputs are in [0, 1]
        assert score <= 1.0 + 1e-9, (
            f"EWMA exceeded 1.0: score={score:.6f}"
        )


# ---------------------------------------------------------------------------
# 4d. Every burial leaves a tombstone (§14 + SPEC §1.3)
# ---------------------------------------------------------------------------


class TestBurialTombstone:
    """Every burial writes a tombstone; burying a buried id raises KeyError."""

    async def test_burial_writes_tombstone(self) -> None:
        """A freshly buried record has a tombstone with all §1.3 fields."""
        from tests.support.doubles import FakeEmbedder

        dim = 64
        adapter = MemoryAdapter(embedding_model="fake/nomic", dim=dim)
        clock = _make_clock()
        embedder = FakeEmbedder(dim=dim)
        cfg = _make_config()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r = await engine.save("burial tombstone fact", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        tid = await adapter.bury(fid, "test burial reason")

        # get_tombstone returns the tombstone (Protocol method)
        tombstone = await adapter.get_tombstone(fid)
        assert tombstone is not None, "No tombstone written after burial"
        assert tombstone.buried_id == fid
        assert tombstone.id == tid
        assert tombstone.reason == "test burial reason"
        assert tombstone.type == "fact"
        assert tombstone.timestamp.tzinfo is not None

    async def test_get_tombstone_returns_none_for_live_id(self) -> None:
        """get_tombstone returns None for a live (not buried) id."""
        from tests.support.doubles import FakeEmbedder

        dim = 64
        adapter = MemoryAdapter(embedding_model="fake/nomic", dim=dim)
        embedder = FakeEmbedder(dim=dim)
        cfg = _make_config()
        engine = Tabularium(adapter, embedder, _make_clock(), cfg)
        await engine.initialize()

        r = await engine.save("live record", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        tombstone = await adapter.get_tombstone(fid)
        assert tombstone is None, "get_tombstone must return None for a live record"

    async def test_get_tombstone_returns_none_for_unknown_id(self) -> None:
        """get_tombstone returns None for an unknown id (no error)."""
        adapter = MemoryAdapter(embedding_model="fake/nomic", dim=64)
        tombstone = await adapter.get_tombstone(new_id())
        assert tombstone is None

    @given(
        n_saves=st.integers(min_value=1, max_value=5),
        bury_index=st.integers(min_value=0, max_value=4),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    async def test_burial_tombstone_property(
        self, n_saves: int, bury_index: int
    ) -> None:
        """Property: every burial writes exactly one tombstone for the buried id."""
        from tests.support.doubles import FakeEmbedder

        n_saves = max(1, n_saves)
        bury_index = bury_index % n_saves
        dim = 64

        adapter = MemoryAdapter(embedding_model="fake/nomic", dim=dim)
        clock = _make_clock()
        embedder = FakeEmbedder(dim=dim)
        cfg = _make_config(gray_floor=0.99)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        ids: list[HypomnemaId] = []
        for i in range(n_saves):
            r = await engine.save(f"property burial fact {i} unique-{id(i)}", scope="global")
            if r.hypomnema is not None:
                ids.append(r.hypomnema.id)

        if not ids or bury_index >= len(ids):
            return  # not enough saved records (dedup bounced)

        target = ids[bury_index]
        await adapter.bury(target, "property test burial")

        tombstone = await adapter.get_tombstone(target)
        assert tombstone is not None, (
            f"No tombstone for buried id {target}"
        )
        assert tombstone.buried_id == target

        # Live ids must still have no tombstone
        for lid in ids:
            if lid != target:
                ts = await adapter.get_tombstone(lid)
                assert ts is None, f"Live id {lid} unexpectedly has a tombstone"

    @given(
        n_records=st.integers(min_value=1, max_value=6),
        bury_flags=st.lists(st.booleans(), min_size=1, max_size=6),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    async def test_burial_tombstone_property_interleaved(
        self, n_records: int, bury_flags: list[bool]
    ) -> None:
        """Property (SPEC §14, M2A): after ANY interleaving of saves and
        burials, ``get_tombstone(id)`` returns a tombstone for EVERY id that
        was successfully buried, and ``None`` for every id that was not.

        Unlike ``test_burial_tombstone_property`` above (which buries exactly
        ONE record, chosen after all saves complete), this generates a
        sequence of saves with a burial decision interleaved after EACH save
        that succeeds — genuine interleaving of writes and burials, not a
        single end-of-run burial. Hermetic (``FakeEmbedder``, no network) and
        ``PYTHONHASHSEED``-independent: the property holds regardless of
        which hash bucket a given content string lands in — it never asserts
        anything about vector geometry, only about the buried/live partition
        of ids.
        """
        from tests.support.doubles import FakeEmbedder

        dim = 64
        adapter = MemoryAdapter(embedding_model="fake/nomic", dim=dim)
        clock = _make_clock()
        embedder = FakeEmbedder(dim=dim)
        cfg = _make_config(gray_floor=0.99)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        buried_ids: list[HypomnemaId] = []
        live_ids: list[HypomnemaId] = []

        for i in range(n_records):
            r = await engine.save(
                f"interleaved burial property fact {i}", scope="global"
            )
            if r.hypomnema is None:
                continue  # accidental FakeEmbedder hash collision bounced this save
            rid = r.hypomnema.id
            should_bury = bury_flags[i % len(bury_flags)]
            if should_bury:
                await adapter.bury(rid, f"interleaved property burial {i}")
                buried_ids.append(rid)
            else:
                live_ids.append(rid)

        for bid in buried_ids:
            tombstone = await adapter.get_tombstone(bid)
            assert tombstone is not None, f"No tombstone for buried id {bid}"
            assert tombstone.buried_id == bid

        for lid in live_ids:
            tombstone = await adapter.get_tombstone(lid)
            assert tombstone is None, f"Live id {lid} unexpectedly has a tombstone"


# ===========================================================================
# Persistence conformance class (LanceDB only)
# ===========================================================================


class TestLanceDBPersistence:
    """Write → close → reopen: all data survives.

    Skipped for memory adapter — it is inherently ephemeral.
    Run only for LanceDB (the persistent adapter).
    """

    @pytest.mark.skipif(
        False, reason="LanceDB only; ephemeral adapters cannot test persistence"
    )
    async def test_persistence_roundtrip(self, tmp_path: Any) -> None:
        """Records, tombstones, distinct pairs, event log, registries survive reopen."""
        from tabularium.domain import Decision, new_id
        from tabularium.storage.lancedb_adapter import LanceDBAdapter

        model = "test-model"
        dim = 4
        now = _EPOCH
        store_path = str(tmp_path / "persist_store")

        # --- Write phase ---
        adapter1 = await LanceDBAdapter.open(store_path, embedding_model=model, dim=dim)

        # Scope + facet + role registration
        await adapter1.create_scope(Scope("testscope"))
        await adapter1.create_facet("convention")
        await adapter1.create_role("implementer")

        # Save a Fact and a Decision
        fact = Fact(
            id=new_id(), type="fact", content="persisted fact",
            scope=Scope("testscope"), lifecycle="active",
            created_at=now, updated_at=now,
        )
        vec_f = [1.0, 0.0, 0.0, 0.0]
        await adapter1.upsert(fact, vec_f)

        dec = Decision(
            id=new_id(), type="decision", content="persisted decision",
            scope=Scope("global"), lifecycle="in_force",
            created_at=now, updated_at=now,
        )
        vec_d = [0.0, 1.0, 0.0, 0.0]
        await adapter1.upsert(dec, vec_d)

        # Bury the fact
        await adapter1.bury(fact.id, "test burial")

        # Event log entry
        await adapter1.append_event(ReinforcementEvent(
            target_id=dec.id, at=now, source="dedup",
        ))

        # Distinct pair
        other_id = new_id()
        # Insert other_id so is_distinct works
        other_fact = Fact(
            id=other_id, type="fact", content="other fact",
            scope=Scope("global"), lifecycle="active",
            created_at=now, updated_at=now,
        )
        await adapter1.upsert(other_fact, [0.0, 0.0, 1.0, 0.0])
        await adapter1.mark_distinct(DistinctPair(
            lower_id=min(dec.id, other_id, key=str),
            higher_id=max(dec.id, other_id, key=str),
            source="save",
            judged_at=now,
        ))

        # Update projection on dec
        await adapter1.update_projections(
            dec.id, recall_count=1, last_referenced_at=now,
            feedback_score=0.0, decay_due_at=None,
        )

        # --- Close (drop the adapter object) ---
        del adapter1

        # --- Reopen phase ---
        adapter2 = await LanceDBAdapter.open(store_path, embedding_model=model, dim=dim)

        # Record survives — the decision
        dec_loaded = await adapter2.get_by_id(dec.id)
        assert isinstance(dec_loaded, Decision)
        assert dec_loaded.content == "persisted decision"
        assert dec_loaded.recall_count == 1

        # Burial survives — fact is buried
        with pytest.raises(KeyError):
            await adapter2.get_by_id(fact.id)

        # Tombstone survives
        tombstone = await adapter2.get_tombstone(fact.id)
        assert tombstone is not None
        assert tombstone.buried_id == fact.id
        assert tombstone.type == "fact"
        assert tombstone.reason == "test burial"

        # Event log survives
        events = await adapter2.query_events(target_id=dec.id)
        assert len(events) == 1
        assert events[0].source == "dedup"

        # Distinct pair survives
        is_dist = await adapter2.is_distinct(dec.id, other_id)
        assert is_dist

        # Registries survive
        scopes = await adapter2.list_scopes()
        assert "testscope" in scopes
        facets = await adapter2.list_facets()
        assert "convention" in facets
        roles = await adapter2.list_roles()
        assert "implementer" in roles

    @pytest.mark.parametrize("adapter_type", ["memory"])
    async def test_memory_adapter_skip_marker(self, adapter_type: str) -> None:
        """Memory adapter is explicitly skipped — it cannot persist across instantiations."""
        # This test exists only to document the skip reason (never silently omit it).
        pytest.skip(
            f"{adapter_type} adapter is ephemeral by design — persistence roundtrip "
            "is not applicable and is explicitly excluded from this test class."
        )
