"""Calibration harness: re-judge an explicit, labelled batch of records through the Cleaner's
relevance axis in dry-run mode, so a rubric change can be measured against a known-labelled batch
(2026-08-12 rubric-gate-retune calibration brief).

Why this exists
----------------
A 2026-08-11 relevance pass on scope ``project-a`` shelved a batch of records; 8 of them were later
judged false positives (they should have been KEEP), including a live authorization-bypass
finding. The rubric's Q1 admission question has since been retuned (the artifact
``engine/rubric.py``'s ``load_rubric()`` loads). This harness answers: does the retune recover
those 8 WITHOUT also flipping the correctly-shelved records to KEEP — a uniform leniency shift
would look like a fix and would not be one.

Why the normal CLI cannot do this
----------------------------------
``tabularium clean --scope <name> --dry-run`` selects its pool via ``list_born_since`` from the
per-scope watermark, and ``_select_relevance_pool`` (``engine/relevance.py``) accepts only
``active``/``archived`` lifecycles as candidates — a genuinely ``shelved`` record (most of this
batch) never reaches a judge that way. This harness instead calls ``run_relevance_axis`` with its
``explicit_pool`` parameter (2026-08-12 calibration-harness fix, additive to that module — ``None``
the default preserves ordinary production behaviour unchanged), which bypasses both
``list_born_since`` and ``_select_relevance_pool``'s exclusions entirely and judges the given Facts
as-is. This is the SAME judging path a real pass uses (batching, the panel, the band policy) — this
script never re-implements any of it.

Store opening
-------------
Uses ``tabularium.storage.open_adapter`` — the SAME shared path every other CLI command in this
repo already goes through identically (``doctor``, ``supersede``, ``sync``, ``clean --dry-run`` —
see ``src/tabularium/cli.py``). Opening the store at all, even for a read-only calibration run,
performs whatever standing housekeeping ``LanceDBAdapter.open()`` does (schema-migration column
backfill, crash-repair tombstone cleanup on an old on-disk table) — this script does not
special-case that.

Hard safety: this script is structurally incapable of writing to the store
----------------------------------------------------------------------------
Two independent layers, neither trusted alone:

1. ``run_relevance_axis`` is ALWAYS called with ``dry_run=True`` — hard-coded, not a CLI flag, so
   there is no argument that could ever flip it to a live run.
2. Every adapter handed to ``run_calibration`` is wrapped in ``_NoWriteGuard`` first — a proxy
   whose only real method is ``get_by_id``; every other attribute access (including anything
   ``run_relevance_axis`` might someday call that this script does not expect) raises
   ``RuntimeError`` rather than silently delegating. This is deliberately independent of layer 1:
   it is what a future accidental ``dry_run=False`` (or a future change to ``run_relevance_axis``
   that adds a call this script did not anticipate) fails LOUD against, instead of quietly writing
   to a live store.

Usage
-----
::

    uv run python scripts/calibrate_relevance_rubric.py \\
        --store /path/to/lancedb-store \\
        --scope project-a \\
        --ids-file dev-corpus/project-a-relevance-calibration-batch.json \\
        --runs 3 \\
        --out dev-corpus/project-a-relevance-calibration-results.json

``--ids-file`` is a JSON array of ``{"id": "<uuid>", "label": "keep"|"shelve"}`` — the human ground
truth: ``"keep"`` for a record that should NOT have been shelved (a false positive under the old
rubric), ``"shelve"`` for a record whose shelve was correct (the control set that must not flip).

Never run this against a real store from this repo's own test/dev environment — it needs the
target machine's local judge instruction files (``engine/judge.py``'s instruction-fingerprint
chain) and would spend real LLM calls. Local verification is the test suite
(``tests/test_calibration_harness.py``) only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid as _uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from tabularium.clock import Clock, SystemClock
from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, HypomnemaId, Scope
from tabularium.engine.relevance import run_relevance_axis

if TYPE_CHECKING:
    from tabularium.engine.judge import JudgeAdapter
    from tabularium.engine.rubric import Rubric
    from tabularium.storage.types import StorageAdapter

_log = logging.getLogger("calibrate_relevance_rubric")

# ---------------------------------------------------------------------------
# Ids file (the human ground truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelledId:
    id: HypomnemaId
    label: Literal["keep", "shelve"]


def load_ids_file(path: Path) -> list[LabelledId]:
    """Parse ``path``'s JSON array of ``{"id": ..., "label": "keep"|"shelve"}`` entries.

    Raises ``ValueError`` (never a raw ``KeyError``/``TypeError``) naming exactly which entry and
    what was wrong with it — this file is hand-authored ground truth, so a malformed entry should
    fail loud and specific, not with a stack trace pointing at generic dict-access code.

    Duplicate ids are also a hard error (a
    duplicate id silently judges the same record twice and lets ``run_calibration``'s
    ``label_by_id`` mapping overwrite by id, moving both a scoring numerator and denominator
    without anyone noticing) — rejected here, before any store read or judge call, even when both
    occurrences carry the SAME label.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array, got {type(raw).__name__}")

    out: list[LabelledId] = []
    seen: set[HypomnemaId] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {i} is not a JSON object")
        if "id" not in entry or "label" not in entry:
            raise ValueError(f"{path}: entry {i} must have both 'id' and 'label'")
        label = entry["label"]
        if label not in ("keep", "shelve"):
            raise ValueError(
                f"{path}: entry {i} has label {label!r} — must be 'keep' or 'shelve'"
            )
        try:
            record_id = HypomnemaId(_uuid.UUID(str(entry["id"])))
        except ValueError as exc:
            raise ValueError(f"{path}: entry {i} has an invalid id {entry['id']!r}") from exc
        if record_id in seen:
            raise ValueError(f"{path}: entry {i} has duplicate id {record_id!s}")
        seen.add(record_id)
        out.append(LabelledId(id=record_id, label=label))
    return out


# ---------------------------------------------------------------------------
# The no-write guard (hard safety, layer 2 — see module docstring)
# ---------------------------------------------------------------------------


class _NoWriteGuard:
    """Wraps a ``StorageAdapter`` so this harness is structurally incapable of writing to the
    store, independent of ``run_relevance_axis``'s own ``dry_run`` gate (see the module docstring's
    "Hard safety" section). ``get_by_id`` is the only call this harness — or ``run_relevance_axis``
    in ``dry_run=True`` + ``explicit_pool=...`` mode — should ever need; every other attribute
    access raises ``RuntimeError`` rather than silently delegating to the real adapter, so a future
    change to either this script or ``run_relevance_axis`` that adds an unexpected adapter call
    fails LOUD here instead of risking a write against a real store.
    """

    def __init__(self, adapter: StorageAdapter) -> None:
        self._adapter = adapter

    async def get_by_id(self, id: HypomnemaId) -> Fact | Decision:
        return await self._adapter.get_by_id(id)

    def __getattr__(self, name: str) -> Any:
        async def _refuse(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                f"calibrate_relevance_rubric: refused adapter.{name}() -- this harness only ever "
                "calls get_by_id; every other adapter call is structurally refused"
            )

        return _refuse


# ---------------------------------------------------------------------------
# Per-run, per-record verdict detail
# ---------------------------------------------------------------------------

_Band = Literal["shelve", "keep", "unclear", "contested", "batch_failed"]


@dataclass(frozen=True)
class JudgeGrade:
    identity: str
    grade: str
    why: str
    unclear: bool


@dataclass(frozen=True)
class RecordRunVerdict:
    """One record's verdict on one calibration run."""

    record_id: HypomnemaId
    run_index: int
    band: _Band
    rubric_version: str
    judge_a: JudgeGrade | None = None
    judge_b: JudgeGrade | None = None


# ---------------------------------------------------------------------------
# Per-record aggregation across runs, and the two headline scores
# ---------------------------------------------------------------------------


def majority_band(bands: Sequence[str]) -> tuple[str | None, float | None]:
    """The most-common band across ``bands``, and the fraction of ``bands`` agreeing with it
    (flip stability: ``1.0`` means every run agreed, lower means the panel flip-flopped).

    ``(None, None)`` for an empty sequence — a record with zero scored runs (every run's batch
    failed) has no majority to report, and callers must not silently coerce that to a real band.

    ``(None, None)`` ALSO for a genuine tie (an
    earlier version broke ties alphabetically and let ``score_aggregates`` count the tie-broken
    winner as a real majority — e.g. ``["shelve", "keep"]`` silently became a "keep" recovery with
    ``stability=0.5``, presenting an unstable, no-majority result as a win). A tie is deliberately
    reported as "no majority" rather than resolved by any rule, so the two headline scores in
    ``score_aggregates`` can never count a coin-flip as a verdict either way.
    """
    if not bands:
        return None, None
    counts = Counter(bands)
    top = max(counts.values())
    winners = sorted(b for b, c in counts.items() if c == top)
    if len(winners) > 1:
        return None, None
    return winners[0], top / len(bands)


@dataclass(frozen=True)
class RecordAggregate:
    record_id: HypomnemaId
    label: Literal["keep", "shelve"]
    bands: tuple[str, ...]  # one per run that produced a real verdict (excludes batch_failed)
    batch_failed_runs: int
    majority_band: str | None
    stability: float | None  # fraction of scored runs agreeing with majority_band


def aggregate_record(
    record_id: HypomnemaId,
    label: Literal["keep", "shelve"],
    run_verdicts: Sequence[RecordRunVerdict],
) -> RecordAggregate:
    scored_bands = [v.band for v in run_verdicts if v.band != "batch_failed"]
    failed = sum(1 for v in run_verdicts if v.band == "batch_failed")
    winner, stability = majority_band(scored_bands)
    return RecordAggregate(
        record_id=record_id,
        label=label,
        bands=tuple(scored_bands),
        batch_failed_runs=failed,
        majority_band=winner,
        stability=stability,
    )


@dataclass(frozen=True)
class CalibrationScore:
    """The two headline numbers this harness exists to produce — never just one (module brief:
    a uniform leniency shift would move only one of these and must not be mistaken for a fix).

    ``None`` on either fraction means the corresponding label had zero records with a majority
    band to score (e.g. an empty ``--ids-file`` half, or every run's batch failing) — reported
    honestly rather than coerced to ``0.0``, which would misreport "nothing to measure" as "measured
    and it's bad".
    """

    false_positive_recovery: float | None
    false_positive_recovered: int
    false_positive_total: int
    control_retention: float | None
    control_retained: int
    control_total: int


def score_aggregates(aggregates: Sequence[RecordAggregate]) -> CalibrationScore:
    """False-positive recovery: fraction of ``label="keep"`` records the rubric now KEEPs
    (majority band == "keep"). Control retention: fraction of ``label="shelve"`` records it still
    SHELVEs (majority band == "shelve"). Both computed only over records that produced a majority
    band at all — a record whose every run's batch failed contributes to neither numerator nor
    denominator, rather than silently counting as a miss.
    """
    keep_scored = [a for a in aggregates if a.label == "keep" and a.majority_band is not None]
    shelve_scored = [a for a in aggregates if a.label == "shelve" and a.majority_band is not None]
    recovered = sum(1 for a in keep_scored if a.majority_band == "keep")
    retained = sum(1 for a in shelve_scored if a.majority_band == "shelve")
    return CalibrationScore(
        false_positive_recovery=(recovered / len(keep_scored)) if keep_scored else None,
        false_positive_recovered=recovered,
        false_positive_total=len(keep_scored),
        control_retention=(retained / len(shelve_scored)) if shelve_scored else None,
        control_retained=retained,
        control_total=len(shelve_scored),
    )


# ---------------------------------------------------------------------------
# The full result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationResult:
    scope: str
    runs: int
    rubric_version: str
    missing_ids: tuple[HypomnemaId, ...]
    skipped_non_fact_ids: tuple[HypomnemaId, ...]
    per_run_verdicts: tuple[RecordRunVerdict, ...]
    aggregates: tuple[RecordAggregate, ...]
    score: CalibrationScore


# ---------------------------------------------------------------------------
# Orchestration — load, judge `runs` times, aggregate, score
# ---------------------------------------------------------------------------


async def run_calibration(
    adapter: StorageAdapter,
    *,
    scope: Scope,
    labelled_ids: Sequence[LabelledId],
    runs: int,
    rubric: Rubric,
    judge_a: JudgeAdapter,
    judge_b: JudgeAdapter,
    clock: Clock | None = None,
) -> CalibrationResult:
    """Fetch every labelled record once, then re-judge the whole batch ``runs`` times through
    ``run_relevance_axis`` (``dry_run=True``, ``explicit_pool=...``) — the SAME judging path a real
    pass uses. ``adapter`` is wrapped in ``_NoWriteGuard`` before anything else touches it (hard
    safety layer 2 — see the module docstring).

    A missing id (``get_by_id`` raises ``KeyError``) is reported in ``missing_ids``, never a
    crash. A found id that is not a ``Fact`` (a ``Decision``) is reported in
    ``skipped_non_fact_ids`` — the relevance axis's judging is Fact-shaped (``set_shelved`` is
    Facts-only), so a Decision has no honest band to report here either.
    """
    guarded = cast("StorageAdapter", _NoWriteGuard(adapter))
    resolved_clock = clock if clock is not None else SystemClock()

    facts: list[Fact] = []
    label_by_id: dict[HypomnemaId, Literal["keep", "shelve"]] = {}
    missing: list[HypomnemaId] = []
    skipped_non_fact: list[HypomnemaId] = []

    for entry in labelled_ids:
        try:
            record = await guarded.get_by_id(entry.id)
        except KeyError:
            missing.append(entry.id)
            continue
        if not isinstance(record, Fact):
            skipped_non_fact.append(entry.id)
            continue
        facts.append(record)
        label_by_id[entry.id] = entry.label

    # `since` plays no role when `explicit_pool` is supplied (`run_relevance_axis`'s own
    # docstring) -- a fixed epoch value, never read.
    since = datetime(1970, 1, 1, tzinfo=UTC)

    per_run_verdicts: list[RecordRunVerdict] = []
    rubric_version = rubric.version

    for run_index in range(runs):
        summary = await run_relevance_axis(
            scope=scope,
            since=since,
            adapter=guarded,
            clock=resolved_clock,
            lock=asyncio.Lock(),
            cfg=EngineConfig(),
            rubric=rubric,
            judge_a=judge_a,
            judge_b=judge_b,
            dry_run=True,
            explicit_pool=facts,
        )
        rubric_version = summary.rubric_version
        full_by_id = {fv.id: fv for fv in summary.full_verdicts}
        failed_ids = {record_id for f in summary.failed_batches for record_id in f.record_ids}

        for fact in facts:
            full = full_by_id.get(fact.id)
            if full is not None:
                per_run_verdicts.append(
                    RecordRunVerdict(
                        record_id=fact.id,
                        run_index=run_index,
                        band=full.band,
                        rubric_version=summary.rubric_version,
                        judge_a=JudgeGrade(
                            identity=full.judge_a_identity,
                            grade=full.judge_a_grade,
                            why=full.judge_a_why,
                            unclear=full.judge_a_unclear,
                        ),
                        judge_b=JudgeGrade(
                            identity=full.judge_b_identity,
                            grade=full.judge_b_grade,
                            why=full.judge_b_why,
                            unclear=full.judge_b_unclear,
                        ),
                    )
                )
                continue
            if fact.id in failed_ids:
                per_run_verdicts.append(
                    RecordRunVerdict(
                        record_id=fact.id,
                        run_index=run_index,
                        band="batch_failed",
                        rubric_version=summary.rubric_version,
                    )
                )
                continue
            # Unreachable under `run_relevance_axis`'s own contract (every explicit_pool candidate
            # lands in exactly one of full_verdicts or a failed batch) -- recorded rather than
            # silently dropped if it ever happens, so a future contract change surfaces here
            # instead of a record quietly vanishing from the report.
            _log.warning(
                "record %s produced neither a verdict nor a batch failure on run %d -- "
                "treating as batch_failed",
                fact.id,
                run_index,
            )
            per_run_verdicts.append(
                RecordRunVerdict(
                    record_id=fact.id,
                    run_index=run_index,
                    band="batch_failed",
                    rubric_version=summary.rubric_version,
                )
            )

    aggregates = [
        aggregate_record(
            fact.id,
            label_by_id[fact.id],
            [v for v in per_run_verdicts if v.record_id == fact.id],
        )
        for fact in facts
    ]
    score = score_aggregates(aggregates)

    return CalibrationResult(
        scope=str(scope),
        runs=runs,
        rubric_version=rubric_version,
        missing_ids=tuple(missing),
        skipped_non_fact_ids=tuple(skipped_non_fact),
        per_run_verdicts=tuple(per_run_verdicts),
        aggregates=tuple(aggregates),
        score=score,
    )


# ---------------------------------------------------------------------------
# Rendering + durable output (atomic write + fsync, matching
# the other scripts' conventions)
# ---------------------------------------------------------------------------


def render_summary(result: CalibrationResult) -> str:
    lines: list[str] = []
    lines.append(f"calibrate_relevance_rubric — scope {result.scope}")
    lines.append("=" * 60)
    lines.append(f"rubric: {result.rubric_version}")
    lines.append(f"runs: {result.runs}")
    if result.missing_ids:
        lines.append(f"missing ids (not found in the store): {len(result.missing_ids)}")
        for record_id in result.missing_ids:
            lines.append(f"    {record_id}")
    if result.skipped_non_fact_ids:
        lines.append(f"skipped (not a Fact): {len(result.skipped_non_fact_ids)}")
        for record_id in result.skipped_non_fact_ids:
            lines.append(f"    {record_id}")
    lines.append("")

    score = result.score
    fpr = "n/a (no keep-labelled records scored)" if score.false_positive_recovery is None else (
        f"{score.false_positive_recovery:.1%} ({score.false_positive_recovered}/"
        f"{score.false_positive_total})"
    )
    cr = "n/a (no shelve-labelled records scored)" if score.control_retention is None else (
        f"{score.control_retention:.1%} ({score.control_retained}/{score.control_total})"
    )
    lines.append(f"false-positive recovery (label=keep, now KEEPs): {fpr}")
    lines.append(f"control retention (label=shelve, still SHELVEs):  {cr}")
    lines.append("")

    lines.append("per-record:")
    for agg in result.aggregates:
        stability = "n/a" if agg.stability is None else f"{agg.stability:.0%}"
        lines.append(
            f"  {agg.record_id} label={agg.label:<6} majority={agg.majority_band!s:<9} "
            f"stability={stability:<5} bands={list(agg.bands)} "
            f"batch_failed_runs={agg.batch_failed_runs}"
        )
    lines.append("")
    return "\n".join(lines)


def _fsync_dir(d: Path) -> None:
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_results(path: Path, result: CalibrationResult) -> None:
    """Write ``result`` durably — temp file in the same directory, fsynced, atomically renamed
    onto ``path``, the leaf directory fsynced too (following the
    ``write_manifest`` conventions: this is the artifact a caller diffs across rubric revisions,
    not a throwaway log)."""
    data = asdict(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    else:
        _fsync_dir(path.parent)


# ---------------------------------------------------------------------------
# Store config (independent copy of tabularium.cli._store_config's env-var convention — this
# script does not import cli.py, same reasoning as the other scripts)
# ---------------------------------------------------------------------------


def _resolve_store_config(*, store: str) -> dict[str, object]:
    embedding_model: str = os.environ.get("TABULARIUM_MODEL", "ollama/nomic-embed-text")
    embedder_dim: int = int(os.environ.get("TABULARIUM_DIM", "768"))
    return {"store_path": store, "embedding_model": embedding_model, "dim": embedder_dim}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibration harness: re-judge an explicit, labelled batch of records through the "
            "Cleaner relevance axis in dry-run mode, so a rubric change can be measured against "
            "a known-labelled batch. Never writes to the store."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--store", required=True, help="Path to the LanceDB store.")
    parser.add_argument("--scope", required=True, help="Scope the labelled batch belongs to.")
    parser.add_argument(
        "--ids-file",
        required=True,
        type=Path,
        help='JSON array of {"id": ..., "label": "keep"|"shelve"} -- the human ground truth.',
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Judge passes over the whole batch (default 3 -- LLM seats are non-deterministic).",
    )
    parser.add_argument("--out", required=True, type=Path, help="Path to write the results JSON.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.runs < 1:
        print("error: --runs must be >= 1", file=sys.stderr)
        return 1

    try:
        labelled_ids = load_ids_file(args.ids_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not labelled_ids:
        print(f"error: {args.ids_file} contains no entries", file=sys.stderr)
        return 1

    cfg = _resolve_store_config(store=args.store)

    async def _run() -> int:
        from tabularium.engine.judge import build_judge_adapter
        from tabularium.engine.rubric import load_rubric
        from tabularium.storage import open_adapter

        adapter = await open_adapter(
            "lancedb",
            embedding_model=str(cfg["embedding_model"]),
            dim=int(str(cfg["dim"])),
            store_path=str(cfg["store_path"]),
        )
        rubric = load_rubric()
        judge_a = build_judge_adapter(rubric.judges.seat_a)
        judge_b = build_judge_adapter(rubric.judges.seat_b)

        result = await run_calibration(
            adapter,
            scope=Scope(args.scope),
            labelled_ids=labelled_ids,
            runs=args.runs,
            rubric=rubric,
            judge_a=judge_a,
            judge_b=judge_b,
        )
        write_results(args.out, result)
        print(f"Results written: {args.out}")
        print()
        print(render_summary(result))
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
