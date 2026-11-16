from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from eval_v1v2 import paths
from eval_v1v2.rows import Bucket, NoiseRow, PairRow, SaveRow, iter_jsonl, read_jsonl
from eval_v1v2.v1_engine import (
    EVAL_SCOPE,
    V1Harness,
    build_record,
    count_records,
    load_noise,
    open_v1,
    require_assessor_key,
)
from tabularium.clock import SystemClock
from tabularium.domain import HypomnemaId
from tabularium.results import SaveResult

WARM_UP_TEXT = "Warm-up note written once before the replay so the first pair is not the cold save."
UNJUDGED_WITHHELD = frozenset({"candidate_changed", "candidate_retired"})


@dataclass(frozen=True)
class Mapped:
    bucket: Bucket
    found_existing: bool
    relation_pick: str | None
    raw_outcome: str
    degraded: str | None


def _point_bucket(target: str | None, existing_id: str) -> Bucket:
    return "caught" if target == existing_id else "wrong_target"


def _judged_pick(judged: object) -> tuple[str | None, str | None]:
    if judged is None:
        return None, "assessor_absent"

    relation = judged.get("relation") if isinstance(judged, dict) else None

    if relation is None:
        return None, "assessor_absent"

    if relation.status == "unavailable":
        return None, f"assessor_unavailable:{relation.reason}"

    return relation.pick, None


def _bounce_view(result: SaveResult, existing_id: str) -> tuple[bool, str | None]:
    bounce = result.bounce

    if bounce is None:
        return False, None

    if str(bounce.candidate_id) == existing_id:
        return True, _judged_pick(bounce.judged)[0]

    for neighbour in bounce.other_candidates:
        if str(neighbour.candidate_id) == existing_id:
            return True, _judged_pick(neighbour.judged)[0]

    return False, None


def _bounce_degraded(result: SaveResult) -> str | None:
    bounce = result.bounce

    if bounce is None:
        return None

    lapse = _judged_pick(bounce.judged)[1]

    if lapse is not None:
        return lapse

    if bounce.withheld in UNJUDGED_WITHHELD:
        return f"withheld:{bounce.withheld}"

    for neighbour in bounce.other_candidates:
        if neighbour.judged is not None and (lapse := _judged_pick(neighbour.judged)[1]):
            return lapse

    return None


def map_result(result: SaveResult, existing_id: str) -> Mapped:
    outcome = result.outcome
    degraded = None if result.embedded else "embedder_unavailable"

    if outcome in {"saved", "reconciled"}:
        return Mapped("new", False, None, outcome, degraded)

    if outcome in {"reinforced", "revived"}:
        target = str(result.reinforced_id) if result.reinforced_id is not None else None
        found = target == existing_id

        raw = outcome if target is None else f"{outcome}:{target}"

        return Mapped(_point_bucket(target, existing_id), found, None, raw, degraded)

    if outcome == "already_saved":
        target = str(result.hypomnema.id) if result.hypomnema is not None else None
        found = target == existing_id

        raw = outcome if target is None else f"{outcome}:{target}"

        return Mapped(_point_bucket(target, existing_id), found, None, raw, degraded)

    bounce = result.bounce

    if bounce is None:
        raise ValueError("bounced result carries no bounce")

    found, pick = _bounce_view(result, existing_id)
    primary = str(bounce.candidate_id) == existing_id
    bucket: Bucket = "flagged" if primary else "wrong_target"
    raw = f"bounced:{bounce.verdict}:{bounce.candidate_id}"

    return Mapped(bucket, found, pick, raw, degraded or _bounce_degraded(result))


def _created_id(result: SaveResult) -> uuid.UUID | None:
    if result.outcome in {"saved", "reconciled"} and result.hypomnema is not None:
        return uuid.UUID(str(result.hypomnema.id))

    return None


async def _bury(harness: V1Harness, record_id: uuid.UUID) -> None:
    try:
        await harness.adapter.bury(HypomnemaId(record_id), "eval isolation")
    except KeyError:
        return


async def replay_pair(harness: V1Harness, pair: PairRow, base_count: int) -> SaveRow:
    present = await count_records(harness)

    if present != base_count:
        raise AssertionError(
            f"store holds {present} rows before {pair.pair_id}, base is {base_count}"
        )

    existing_id = uuid.uuid4()
    vector = await harness.embedder.embed(pair.existing)
    existing = build_record(
        record_id=existing_id,
        kind=pair.kind,
        content=pair.existing,
        scope=EVAL_SCOPE,
        created_at=SystemClock().now(),
        recall_count=0,
        lifecycle="active",
    )
    created: uuid.UUID | None = None

    await harness.adapter.upsert(existing, vector)

    try:
        started = time.perf_counter()
        result = await harness.engine.save(pair.incoming, scope=EVAL_SCOPE, kind=pair.kind)
        elapsed = (time.perf_counter() - started) * 1000
        created = _created_id(result)
        mapped = map_result(result, str(existing_id))
    finally:
        await _bury(harness, existing_id)

        if created is not None:
            await _bury(harness, created)

    return SaveRow(
        pair_id=pair.pair_id,
        system="v1",
        bucket=mapped.bucket,
        found_existing=mapped.found_existing,
        relation_pick=mapped.relation_pick,
        raw_outcome=mapped.raw_outcome,
        elapsed_ms=round(elapsed, 2),
        degraded=mapped.degraded,
    )


async def run_save(
    harness: V1Harness,
    pairs: list[PairRow],
    out: Path,
    done: frozenset[str] = frozenset(),
) -> list[SaveRow]:
    base_count = await count_records(harness)
    pending = [pair for pair in pairs if pair.pair_id not in done]
    rows: list[SaveRow] = []

    if not pending:
        return rows

    await harness.embedder.embed("warm up")
    warm = await harness.engine.save(WARM_UP_TEXT, scope=EVAL_SCOPE, kind="fact")
    warm_id = _created_id(warm)

    if warm_id is not None:
        await _bury(harness, warm_id)

    for pair in pending:
        row = await replay_pair(harness, pair, base_count)

        with out.open("a", encoding="utf-8") as handle:
            handle.write(row.model_dump_json())
            handle.write("\n")

        rows.append(row)

    return rows


def finished_ids(out: Path) -> frozenset[str]:
    if not out.exists():
        return frozenset()

    return frozenset(row.pair_id for row in iter_jsonl(out, SaveRow))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay labelled pairs through the v1 save path")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--noise", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--store", type=Path, default=None)

    return parser


async def _main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)

    require_assessor_key()

    pairs_path = args.pairs or paths.out_path("pairs.jsonl")
    noise_path = args.noise or paths.out_path("noise.jsonl")
    out = args.out or paths.out_path("save-v1.jsonl")
    pairs = read_jsonl(pairs_path, PairRow)[: args.limit]
    noise = read_jsonl(noise_path, NoiseRow)
    harness = await open_v1(args.store)
    present = await count_records(harness)

    if present not in {0, len(noise)}:
        raise SystemExit(f"store holds {present} rows, expected 0 or {len(noise)}")

    if present == 0:
        started = time.perf_counter()

        await load_noise(
            harness, noise, lambda n: print(f"seeded {n}/{len(noise)}", file=sys.stderr)
        )

        seconds = round(time.perf_counter() - started, 1)

        print(f"seeded {len(noise)} noise rows in {seconds} s", file=sys.stderr)

    rows = await run_save(harness, pairs, out, finished_ids(out))
    degraded = sum(1 for row in rows if row.degraded is not None)

    print(
        f'{{"event": "v1-save", "pairs": {len(pairs)}, "written": {len(rows)}, '
        f'"degraded": {degraded}, "store": "{harness.dir}"}}'
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(sys.argv[1:])))
