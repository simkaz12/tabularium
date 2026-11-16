"""Calibrate the dedup cosine floor over the golden pair set.

Offline-only: never imported by the gate or any production code.

Sweeps ``gray_floor`` over the hand-curated golden pairs in freeze_vectors.py and reports
the operating point that maximises recall (same-class detection rate — a same-class pair
bounces rather than passing through unflagged) subject to dedup precision >= P_FLOOR (a
bounce actually is a same-class pair, not a needless adjudication on distinct content).
The higher cosine ceiling this script used to also sweep, above which a survivor silently
auto-reinforced instead of bouncing, is gone
— every survivor at or above ``gray_floor`` bounces or goes to the Assessor now, so this
script has one threshold left to sweep, not two.

Writes a versioned per-embedder recommendation file::

    calibration/nomic-embed-text.json

which contains the recommended threshold.  Applying the recommendation to config is always
a MANUAL step (direction-gated live application is M3).

Usage::

    uv run python scripts/calibrate_thresholds.py

Output:
  - Summary table to stdout
  - calibration/nomic-embed-text.json
"""

from __future__ import annotations

import asyncio
import json
import math
import pathlib
import sys
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# Import golden set from freeze script
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from freeze_vectors import GOLDEN_PAIRS  # noqa: E402

sys.path.pop(0)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

P_FLOOR = 0.95  # target precision floor (eval-tunable seed)
FIXTURE_PATH = pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "frozen_vectors.json"
OUTPUT_PATH = pathlib.Path(__file__).parent.parent / "calibration" / "nomic-embed-text.json"

# Current seed from EngineConfig defaults
CURRENT_GRAY_FLOOR = 0.80
CURRENT_DECISION_GRAY_FLOOR = 0.70

# Sweep range
GRAY_FLOOR_RANGE = [round(x * 0.05, 2) for x in range(10, 20)]   # 0.50 .. 0.95


# ---------------------------------------------------------------------------
# Cosine helper
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Gate verdict at a given gray_floor
# ---------------------------------------------------------------------------


def _gate_verdict(cos: float, gray_floor: float) -> str:
    """Classify a gate probe at ``gray_floor`` (simplified, Fact arm only): below
    floor passes through untouched; at or above it bounces (or goes to the
    Assessor, evidence only) — no cosine value auto-merges, so a guard trip has
    nothing left to annotate a downgrade from and is not a separate verdict here."""
    return "pass" if cos < gray_floor else "bounce"


# ---------------------------------------------------------------------------
# Evaluate a gray_floor over the golden set
# ---------------------------------------------------------------------------


def _evaluate(pair_cosines: list[dict[str, object]], gray_floor: float) -> dict[str, float]:
    """Compute precision and recall at ``gray_floor``.

    ``tp``: same-class pairs that bounce (correctly flagged for adjudication).
    ``fp``: distinct_near/unrelated pairs that bounce (a needless adjudication).
    ``fn``: same-class pairs that pass (missed — silently duplicated).
    ``tn``: distinct_near/unrelated pairs that pass (correctly let through).
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for pc in pair_cosines:
        label = str(pc["label"])
        cos = float(pc["cosine"])  # type: ignore[arg-type]
        verdict = _gate_verdict(cos, gray_floor)

        if label == "same":
            if verdict == "bounce":
                tp += 1
            else:
                fn += 1
        else:  # "distinct_near" or "unrelated"
            if verdict == "bounce":
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # Load frozen vectors
    if not FIXTURE_PATH.exists():
        print(
            f"ERROR: frozen fixture not found at {FIXTURE_PATH}. "
            "Run scripts/freeze_vectors.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    vectors: dict[str, list[float]] = raw["vectors"]
    embedder_model: str = raw["embedder_model"]
    dim: int = raw["dim"]

    print(f"Calibrating over {len(GOLDEN_PAIRS)} golden pairs")
    print(f"Embedder: {embedder_model} (dim={dim})")
    print(f"P_FLOOR: {P_FLOOR}")
    print()

    # Pre-compute cosines for all golden pairs
    pair_cosines: list[dict[str, object]] = []
    for pair in GOLDEN_PAIRS:
        ta = str(pair["text_a"])
        tb = str(pair["text_b"])
        if ta not in vectors or tb not in vectors:
            print(f"WARNING: pair text not in fixture: {ta[:50]!r}")
            continue
        cos = _cosine(vectors[ta], vectors[tb])
        pair_cosines.append({
            "text_a": ta,
            "text_b": tb,
            "label": pair["label"],
            "guard_note": pair.get("guard_note"),
            "cosine": cos,
        })

    # Show cosine distribution by class
    print("=== Cosine distribution by class ===")
    for label in ["same", "distinct_near", "unrelated"]:
        class_cosines = [float(pc["cosine"]) for pc in pair_cosines if pc["label"] == label]  # type: ignore[arg-type]
        if class_cosines:
            avg = sum(class_cosines) / len(class_cosines)
            mn = min(class_cosines)
            mx = max(class_cosines)
            print(
                f"  {label:15s}: n={len(class_cosines):3d}  "
                f"min={mn:.3f}  avg={avg:.3f}  max={mx:.3f}"
            )
    print()

    # Current seed evaluation
    current = _evaluate(pair_cosines, CURRENT_GRAY_FLOOR)
    print(f"=== Current seed (gray_floor={CURRENT_GRAY_FLOOR}) ===")
    print(
        f"  precision={current['precision']:.3f}  recall={current['recall']:.3f}  "
        f"TP={current['tp']}  FP={current['fp']}  FN={current['fn']}"
    )
    meets_floor = current["precision"] >= P_FLOOR
    print(f"  precision ≥ P_FLOOR={P_FLOOR}: {'YES' if meets_floor else 'NO'}")
    print()

    # Sweep
    print("=== Sweep (gray_floor, precision >= P_FLOOR) ===")
    best: dict[str, object] | None = None
    best_recall = -1.0

    results = []
    for gf in GRAY_FLOOR_RANGE:
        ev = _evaluate(pair_cosines, gf)
        meets = ev["precision"] >= P_FLOOR
        results.append({
            "gray_floor": gf,
            "precision": ev["precision"],
            "recall": ev["recall"],
            "meets_p_floor": meets,
        })
        if meets and ev["recall"] > best_recall:
            best_recall = ev["recall"]
            best = {
                "gray_floor": gf,
                "precision": ev["precision"],
                "recall": ev["recall"],
            }

    # Print top 10 by recall (filtered to precision ≥ P_FLOOR)
    feasible = [r for r in results if r["meets_p_floor"]]
    feasible.sort(key=lambda r: float(r["recall"]) * -1)
    print(f"  {'gray_floor':>10s}  {'precision':>9s}  {'recall':>6s}")
    for r in feasible[:10]:
        flag = " ← best" if r is feasible[0] else ""
        print(f"  {r['gray_floor']:>10.2f}  {r['precision']:>9.3f}  {r['recall']:>6.3f}{flag}")

    print()
    if best is None:
        print("WARNING: no gray_floor meets P_FLOOR — P_FLOOR may be too tight.")
        sys.exit(0)

    print("=== Recommendation ===")
    print(f"  gray_floor       = {best['gray_floor']}")
    print(f"  precision        = {best['precision']:.3f}")
    print(f"  recall           = {best['recall']:.3f}")
    print()
    print("vs current seed:")
    print(f"  gray_floor       : {CURRENT_GRAY_FLOOR} → {best['gray_floor']}")
    print(f"  decision_gray_floor: {CURRENT_DECISION_GRAY_FLOOR} (not swept; apply same delta)")
    print()
    print("NOTE: recommendation is for manual review; not auto-applied.")

    # Write calibration output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output: dict[str, object] = {
        "embedder": embedder_model,
        "dim": dim,
        "generated_at": datetime.now(UTC).isoformat(),
        "golden_set_size": len(pair_cosines),
        "golden_set_by_class": {
            label: len([pc for pc in pair_cosines if pc["label"] == label])
            for label in ["same", "distinct_near", "unrelated"]
        },
        "P_floor": P_FLOOR,
        "current_seeds": {
            "gray_floor": CURRENT_GRAY_FLOOR,
            "decision_gray_floor": CURRENT_DECISION_GRAY_FLOOR,
            "precision": current["precision"],
            "recall": current["recall"],
        },
        "recommendation": {
            "gray_floor": best["gray_floor"],
            "precision": best["precision"],
            "recall": best["recall"],
        },
        "generated_by": "scripts/calibrate_thresholds.py",
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Calibration written to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
