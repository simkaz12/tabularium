"""β / RRF-`k` ablation — offline calibration harness (M2 slice B).

Offline-only: never imported by the gate or any production code.

Sweeps ``beta`` (the post-fusion importance boost weight, SPEC §5:
``final = RRF(v, t) · (1 + β · importance_norm)``) and ``rrf_k`` (the RRF
fusion constant) over the T01 frozen recall corpus/query set
(``scripts/freeze_vectors.py``) and reports recall@10 / precision@10 for
every ``(beta, rrf_k)`` combination, plus a repeated-baseline noise-floor
measurement at the current seed defaults.

Hermetic and reproducible: ``MemoryAdapter`` + ``FrozenEmbedder`` (frozen
real nomic-embed-text geometry, no network) + ``TestClock`` (fixed epoch) —
the same hermetic construction ``tests/test_eval_b5.py::TestRecallPrecisionFloors``
uses for its "memory" adapter lane. PYTHONHASHSEED-independent: no set/dict
iteration order feeds into any score or ranking decision (RRF fusion sorts
on numeric score; MemoryAdapter's own leg orderings are deterministic given
frozen vectors and a fixed corpus insertion order).

The corpus is populated into ONE ``MemoryAdapter`` a single time; every
``(beta, rrf_k)`` sweep point wraps a fresh ``Tabularium`` engine around that
SAME populated adapter/embedder/clock with only ``EngineConfig`` differing —
so every combination exercises the real production ``engine.recall()`` path
(filters, candidate fan-out, RRF fusion, importance boost) rather than a
reimplementation of the scoring math.

This script writes ONLY a raw-data JSON artifact (measurement, not a
recommendation to apply) — no config default is read from or written to
``src/tabularium/config.py``. The script follows the style of ``calibrate_thresholds.py`` (data
here,
judgment there).

Usage::

    uv run python scripts/calibrate_beta_rrf.py

Output:
  - Summary tables to stdout (noise floor, full grid, Pareto frontier)
  - calibration/beta-rrf-ablation.json
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from datetime import UTC, datetime, timedelta

# ---------------------------------------------------------------------------
# Import corpus/query fixtures from the freeze script (kept in sync, same
# import pattern as scripts/calibrate_thresholds.py).
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))
from freeze_vectors import CORPUS_TEXTS, QUERIES  # noqa: E402

sys.path.remove(str(_SCRIPTS_DIR))

from tests.support.doubles import FrozenEmbedder, TestClock  # noqa: E402

from tabularium.config import EngineConfig  # noqa: E402
from tabularium.domain import HypomnemaId  # noqa: E402
from tabularium.engine import Tabularium  # noqa: E402
from tabularium.storage.memory import MemoryAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_PATH = pathlib.Path(__file__).parent.parent / "calibration" / "beta-rrf-ablation.json"

_EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

#: recall@K / precision@K, matching tests/test_eval_b5.py's K.
K = 10

#: Sweep ranges. beta MUST include 0.0 (the "importance boost does nothing
#: useful" null hypothesis has to be on the chart). rrf_k MUST include 60
#: (the seeded default, SPEC §5 / config.py). Both ranges also include the
#: current EngineConfig defaults as a grid point so "seed" is never
#: extrapolated.
BETA_RANGE: list[float] = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
RRF_K_RANGE: list[int] = [10, 30, 60, 100, 200]

DEFAULT_BETA: float = EngineConfig.model_fields["beta"].default
DEFAULT_RRF_K: int = EngineConfig.model_fields["rrf_k"].default

#: Repeats for the noise-floor measurement at the seed default point.
NOISE_FLOOR_REPEATS = 5

#: Per-record clock advance for the SECOND (staggered-importance) experiment
#: below -- see that section's docstring for why the primary sweep needs it.
STAGGER_STEP = timedelta(hours=12)


# ---------------------------------------------------------------------------
# Corpus population (once) + per-combo measurement
# ---------------------------------------------------------------------------


async def _build_populated_adapter(
    *, stagger_step: timedelta | None = None
) -> tuple[MemoryAdapter, FrozenEmbedder, TestClock]:
    """Populate one ``MemoryAdapter`` with ``CORPUS_TEXTS``, built once.

    Reused (read-only) by every sweep point in ``_measure`` below. Returns
    the ``TestClock`` instance used during population -- when
    ``stagger_step`` is given, the clock is advanced by that step between
    each save (so records carry genuinely different ``last_referenced_at``
    values) and the returned clock is left at its FINAL value, so a caller
    measuring recall "now" (immediately after population) sees the same
    recency spread a real corpus accumulated over time would have. When
    ``stagger_step`` is ``None`` the clock never advances (every record
    shares one instant), matching ``tests/test_eval_b5.py``'s hermetic gate
    construction exactly.
    """
    embedder = FrozenEmbedder()
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim)
    clock = TestClock(_EPOCH)
    # A high floor so nothing bounces during corpus load (same pattern as
    # tests/test_eval_b5.py::populated_engine).
    load_cfg = EngineConfig(gray_floor=0.99, candidate_k=200)
    engine = Tabularium(adapter, embedder, clock, load_cfg)
    await engine.initialize()

    for text in CORPUS_TEXTS:
        r = await engine.save(text, scope="global")
        assert r.outcome in {"saved", "bounced", "reinforced"}, (
            f"Unexpected outcome {r.outcome!r} saving corpus text: {text[:60]!r}"
        )
        if stagger_step is not None:
            clock.advance(stagger_step)
    await adapter.upkeep()  # no-op for MemoryAdapter; parity with the lancedb lane
    return adapter, embedder, clock


async def _text_to_id_map(adapter: MemoryAdapter) -> dict[str, HypomnemaId]:
    """content -> id, via ``scan_records()`` directly (no per-candidate ``get_by_id``)."""
    return {rec.content: rec.id async for rec in await adapter.scan_records()}


async def _measure(
    adapter: MemoryAdapter,
    embedder: FrozenEmbedder,
    clock: TestClock,
    text_to_id: dict[str, HypomnemaId],
    beta: float,
    rrf_k: int,
) -> dict[str, float]:
    """Recall@K / precision@K over ``QUERIES`` at one ``(beta, rrf_k)`` point.

    Wraps a FRESH ``Tabularium``/``cfg`` around the already-populated
    adapter — population is a one-time cost; only ``cfg`` (and thus the RRF
    fusion + importance-boost math inside ``engine.recall()``) varies per
    call. Uses the real production ``recall()`` path end to end, not a
    reimplementation of the scoring formula. ``clock`` is the SAME instance
    ``_build_populated_adapter`` returned -- reusing it (rather than a fresh
    ``TestClock(_EPOCH)``) means "now" for the importance-boost calculation
    is consistent with whatever recency spread population left behind.
    """
    cfg = EngineConfig(beta=beta, rrf_k=rrf_k, candidate_k=200)
    engine = Tabularium(adapter, embedder, clock, cfg)
    await engine.initialize()  # idempotent; registries already exist on the shared adapter

    total_recall = 0.0
    total_precision = 0.0
    n_queries = len(QUERIES)

    for q in QUERIES:
        query_text = str(q["query"])
        relevant_indices: list[int] = list(q["relevant_indices"])  # type: ignore[call-overload]
        relevant_ids = {
            text_to_id[CORPUS_TEXTS[i]] for i in relevant_indices if CORPUS_TEXTS[i] in text_to_id
        }

        result = await engine.recall(query_text, scope="global", limit=K)
        retrieved_ids = {h.id for h in result.hits}

        tp = len(relevant_ids & retrieved_ids)
        total_recall += tp / max(len(relevant_ids), 1)
        total_precision += tp / max(len(retrieved_ids), 1)

    return {
        "avg_recall": total_recall / n_queries,
        "avg_precision": total_precision / n_queries,
    }


def _pareto_frontier(grid: list[dict[str, float]]) -> list[dict[str, float]]:
    """Non-dominated points on (avg_recall, avg_precision), best-first."""
    frontier: list[dict[str, float]] = []
    for g in grid:
        dominated = any(
            o["avg_recall"] >= g["avg_recall"]
            and o["avg_precision"] >= g["avg_precision"]
            and (o["avg_recall"], o["avg_precision"]) != (g["avg_recall"], g["avg_precision"])
            for o in grid
        )
        if not dominated:
            frontier.append(g)
    frontier.sort(key=lambda g: (-g["avg_recall"], -g["avg_precision"]))
    return frontier


async def _run_grid(
    adapter: MemoryAdapter,
    embedder: FrozenEmbedder,
    clock: TestClock,
    text_to_id: dict[str, HypomnemaId],
    betas: list[float],
    rrf_ks: list[int],
) -> list[dict[str, float]]:
    """Sweep the full ``betas x rrf_ks`` grid, printing a table as it goes."""
    print(f"{'beta':>6}  {'rrf_k':>6}  {'recall@10':>10}  {'precision@10':>13}")
    grid: list[dict[str, float]] = []
    for beta in betas:
        for rrf_k in rrf_ks:
            m = await _measure(adapter, embedder, clock, text_to_id, beta, rrf_k)
            grid.append({"beta": beta, "rrf_k": float(rrf_k), **m})
            marker = " <- seed" if (beta, rrf_k) == (DEFAULT_BETA, DEFAULT_RRF_K) else ""
            print(
                f"{beta:>6}  {rrf_k:>6}  {m['avg_recall']:>10.4f}  "
                f"{m['avg_precision']:>13.4f}{marker}"
            )
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print(
        f"Populating hermetic MemoryAdapter corpus "
        f"({len(CORPUS_TEXTS)} texts, {len(QUERIES)} queries)..."
    )
    adapter, embedder, clock = await _build_populated_adapter()
    text_to_id = await _text_to_id_map(adapter)
    print("Corpus populated (uniform timestamps -- matches tests/test_eval_b5.py exactly).\n")

    # --- Noise floor: repeat the seed default point N times ---
    print(
        f"=== Noise floor: {NOISE_FLOOR_REPEATS}x repeats at seed defaults "
        f"(beta={DEFAULT_BETA}, rrf_k={DEFAULT_RRF_K}) ==="
    )
    noise_runs: list[dict[str, float]] = []
    for i in range(NOISE_FLOOR_REPEATS):
        m = await _measure(adapter, embedder, clock, text_to_id, DEFAULT_BETA, DEFAULT_RRF_K)
        noise_runs.append(m)
        print(f"  run {i + 1}: recall={m['avg_recall']:.6f}  precision={m['avg_precision']:.6f}")
    recalls = [r["avg_recall"] for r in noise_runs]
    precisions = [r["avg_precision"] for r in noise_runs]
    recall_spread = max(recalls) - min(recalls)
    precision_spread = max(precisions) - min(precisions)
    print(f"  spread: recall={recall_spread:.6f}  precision={precision_spread:.6f}")
    print()

    # --- Experiment 1: primary sweep on the T01 harness's own corpus construction ---
    n_combos = len(BETA_RANGE) * len(RRF_K_RANGE)
    print(f"=== Experiment 1 -- primary sweep: beta x rrf_k ({n_combos} combos) ===")
    grid = await _run_grid(adapter, embedder, clock, text_to_id, BETA_RANGE, RRF_K_RANGE)

    seed_point = next(
        g for g in grid if g["beta"] == DEFAULT_BETA and g["rrf_k"] == float(DEFAULT_RRF_K)
    )
    print()
    print(
        f"Seed point (beta={DEFAULT_BETA}, rrf_k={DEFAULT_RRF_K}): "
        f"recall={seed_point['avg_recall']:.4f} precision={seed_point['avg_precision']:.4f}"
    )

    frontier = _pareto_frontier(grid)
    print()
    print("=== Pareto frontier (non-dominated on recall@10 AND precision@10) ===")
    for g in frontier:
        is_seed = (g["beta"], g["rrf_k"]) == (DEFAULT_BETA, float(DEFAULT_RRF_K))
        marker = " <- seed" if is_seed else ""
        print(
            f"  beta={g['beta']:<5} rrf_k={g['rrf_k']:<5.0f} "
            f"recall={g['avg_recall']:.4f} precision={g['avg_precision']:.4f}{marker}"
        )

    best_recall = max(grid, key=lambda g: g["avg_recall"])
    best_precision = max(grid, key=lambda g: g["avg_precision"])
    print()
    print(
        f"Best recall@10:    beta={best_recall['beta']} rrf_k={best_recall['rrf_k']:.0f} "
        f"recall={best_recall['avg_recall']:.4f} "
        f"(delta vs seed = {best_recall['avg_recall'] - seed_point['avg_recall']:+.4f})"
    )
    print(
        f"Best precision@10: beta={best_precision['beta']} rrf_k={best_precision['rrf_k']:.0f} "
        f"precision={best_precision['avg_precision']:.4f} "
        f"(delta vs seed = {best_precision['avg_precision'] - seed_point['avg_precision']:+.4f})"
    )

    # --- Diagnostic: is importance_norm actually varying across candidates? ---
    # If every record shares one save instant (recall_count=0, feedback=0.0,
    # identical last_referenced_at), compute_importance() is IDENTICAL for every
    # record, so imp_norm = imp/max_imp = 1.0 for every candidate and the
    # boost `(1 + beta * 1.0)` is a constant rescale applied uniformly --
    # provably unable to reorder anything, for ANY beta. Confirmed directly
    # against the corpus this experiment populated (not inferred from the
    # recall/precision table alone).
    ref_ats: set[datetime] = {
        rec.last_referenced_at or rec.created_at
        async for rec in await adapter.scan_records()
    }
    print()
    print(
        f"Diagnostic: distinct last_referenced_at values across the {len(CORPUS_TEXTS)}-record "
        f"corpus = {len(ref_ats)} (1 means every record has identical importance -> beta is "
        f"mathematically inert at every rrf_k, confirmed above)."
    )

    # --- Experiment 2: staggered corpus, to test beta when importance actually varies ---
    print()
    print(
        f"=== Experiment 2 -- staggered corpus (clock advances {STAGGER_STEP} per save, "
        f"beta sweep at rrf_k={DEFAULT_RRF_K}) ==="
    )
    adapter2, embedder2, clock2 = await _build_populated_adapter(stagger_step=STAGGER_STEP)
    text_to_id2 = await _text_to_id_map(adapter2)
    ref_ats2: list[datetime] = [
        rec.last_referenced_at or rec.created_at async for rec in await adapter2.scan_records()
    ]
    spread_days = (max(ref_ats2) - min(ref_ats2)).total_seconds() / 86400.0
    print(
        f"Recency spread across the staggered corpus: {spread_days:.2f} days "
        f"(vs half_life_days={EngineConfig().half_life_days})."
    )
    staggered_grid = await _run_grid(
        adapter2, embedder2, clock2, text_to_id2, BETA_RANGE, [DEFAULT_RRF_K]
    )
    staggered_seed = next(g for g in staggered_grid if g["beta"] == DEFAULT_BETA)
    staggered_best_recall = max(staggered_grid, key=lambda g: g["avg_recall"])
    staggered_best_precision = max(staggered_grid, key=lambda g: g["avg_precision"])
    print()
    print(
        f"Staggered seed (beta={DEFAULT_BETA}): recall={staggered_seed['avg_recall']:.4f} "
        f"precision={staggered_seed['avg_precision']:.4f}"
    )
    print(
        f"Staggered best recall@10:    beta={staggered_best_recall['beta']} "
        f"recall={staggered_best_recall['avg_recall']:.4f} "
        f"(delta vs staggered seed = "
        f"{staggered_best_recall['avg_recall'] - staggered_seed['avg_recall']:+.4f})"
    )
    print(
        f"Staggered best precision@10: beta={staggered_best_precision['beta']} "
        f"precision={staggered_best_precision['avg_precision']:.4f} "
        f"(delta vs staggered seed = "
        f"{staggered_best_precision['avg_precision'] - staggered_seed['avg_precision']:+.4f})"
    )

    # --- Write raw output ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_size": len(CORPUS_TEXTS),
        "n_queries": len(QUERIES),
        "k": K,
        "seed_defaults": {"beta": DEFAULT_BETA, "rrf_k": DEFAULT_RRF_K},
        "noise_floor": {
            "repeats": NOISE_FLOOR_REPEATS,
            "runs": noise_runs,
            "recall_spread": recall_spread,
            "precision_spread": precision_spread,
        },
        "experiment_1_primary": {
            "description": "T01 harness's own corpus construction (uniform timestamps).",
            "distinct_last_referenced_at_values": len(ref_ats),
            "grid": grid,
            "pareto_frontier": frontier,
        },
        "experiment_2_staggered_importance": {
            "description": (
                "Diagnostic corpus with clock advanced by stagger_step per save, so "
                "importance_norm actually varies across candidates; beta swept at "
                "the seeded rrf_k only."
            ),
            "stagger_step_hours": STAGGER_STEP.total_seconds() / 3600.0,
            "recency_spread_days": spread_days,
            "grid": staggered_grid,
        },
        "generated_by": "scripts/calibrate_beta_rrf.py",
        "note": "Measurement only -- no config default changed by this script or the task it "
        "supports.",
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nRaw ablation data written to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
