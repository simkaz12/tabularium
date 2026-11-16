from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import get_args

from pydantic import BaseModel

from eval_v1v2 import paths, targets
from eval_v1v2.rows import Bucket, KeyRow, PairRow, SaveRow, Source, System, read_jsonl

BUCKETS: tuple[str, ...] = get_args(Bucket)
RELATIONS: tuple[str, ...] = ("restatement", "replacement", "follow_up", "distinct")
RESTATEMENT: frozenset[str] = frozenset({"restatement"})
REPLACEMENT: frozenset[str] = frozenset({"replacement"})
UNRELATED: frozenset[str] = frozenset({"follow_up", "distinct"})
SCORE_NAME = "score-save.json"


class Rate(BaseModel):
    count: int
    total: int

    @property
    def share(self) -> float | None:
        return self.count / self.total if self.total else None


class RelationRow(BaseModel):
    relation: str
    total: int
    buckets: dict[str, int]
    noticed: int


class SourceRow(BaseModel):
    source: str
    scored: int
    restatement_noticed: Rate
    replacement_flagged: Rate
    false_alarm: Rate


class PointedScore(BaseModel):
    total: int = 0
    cells: dict[str, dict[str, int]] = {}
    merged_right: int = 0
    merged_lost_record: int = 0
    merged_lost_update: int = 0
    flagged_right: int = 0
    flagged_false_alarm: int = 0
    unresolved: int = 0


class SystemScore(BaseModel):
    system: str
    rows: int
    scored: int
    unkeyed: int
    by_relation: list[RelationRow]
    duplicate_caught: Rate
    duplicate_noticed: Rate
    replacement_flagged: Rate
    lost_update: Rate
    lost_record: Rate
    false_alarm: Rate
    lost_record_any: Rate = Rate(count=0, total=0)
    false_alarm_any: Rate = Rate(count=0, total=0)
    pointed: PointedScore = PointedScore()
    by_source: list[SourceRow] = []
    found_existing: Rate
    relation_pick: Rate
    degraded: Rate
    median_ms: float | None
    p95_ms: float | None
    timed_calls: int


class SaveScore(BaseModel):
    systems: list[SystemScore]


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(math.ceil(fraction * len(ordered)), 1)

    return ordered[rank - 1]


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def load_key(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    return {row.item_id: str(row.label) for row in read_jsonl(path, KeyRow)}


def _count(
    rows: Sequence[SaveRow], key: Mapping[str, str], relations: frozenset[str], bucket: str
) -> int:
    return sum(1 for row in rows if key[row.pair_id] in relations and row.bucket == bucket)


def _total(rows: Sequence[SaveRow], key: Mapping[str, str], relations: frozenset[str]) -> int:
    return sum(1 for row in rows if key[row.pair_id] in relations)


def _rate(
    rows: Sequence[SaveRow], key: Mapping[str, str], relations: frozenset[str], *buckets: str
) -> Rate:
    count = sum(_count(rows, key, relations, bucket) for bucket in buckets)

    return Rate(count=count, total=_total(rows, key, relations))


def _relation_rows(rows: Sequence[SaveRow], key: Mapping[str, str]) -> list[RelationRow]:
    table: list[RelationRow] = []

    for relation in RELATIONS:
        picked = [row for row in rows if key[row.pair_id] == relation]
        counts: Counter[str] = Counter(row.bucket for row in picked)
        table.append(
            RelationRow(
                relation=relation,
                total=len(picked),
                buckets={bucket: counts.get(bucket, 0) for bucket in BUCKETS},
                noticed=counts.get("caught", 0) + counts.get("flagged", 0),
            )
        )

    return table


def _source_rows(
    rows: Sequence[SaveRow], key: Mapping[str, str], sources: Mapping[str, str]
) -> list[SourceRow]:
    table: list[SourceRow] = []

    for source in get_args(Source):
        picked = [row for row in rows if sources.get(row.pair_id) == source]
        table.append(
            SourceRow(
                source=source,
                scored=len(picked),
                restatement_noticed=_rate(picked, key, RESTATEMENT, "caught", "flagged"),
                replacement_flagged=_rate(picked, key, REPLACEMENT, "flagged"),
                false_alarm=_rate(picked, key, UNRELATED, "flagged"),
            )
        )

    return table


def _pointed_score(pointed: Sequence[targets.Pointed]) -> PointedScore:
    cells: dict[str, dict[str, int]] = {
        action: dict.fromkeys(targets.CATEGORIES, 0) for action in ("merged", "flagged")
    }

    for row in pointed:
        cells[row.action][row.category] += 1

    merged, flagged = cells["merged"], cells["flagged"]

    return PointedScore(
        total=len(pointed),
        cells=cells,
        merged_right=merged["restatement"],
        merged_lost_record=merged["follow_up"] + merged["distinct"],
        merged_lost_update=merged["replacement"],
        flagged_right=flagged["restatement"] + flagged["replacement"],
        flagged_false_alarm=flagged["follow_up"] + flagged["distinct"],
        unresolved=merged[targets.UNRESOLVED] + flagged[targets.UNRESOLVED],
    )


def _pointed_in_key(
    pointed: Sequence[targets.Pointed],
    key: Mapping[str, str],
    action: str,
    categories: frozenset[str],
) -> int:
    return sum(
        1
        for row in pointed
        if row.pair_id in key and row.action == action and row.category in categories
    )


def score_system(
    system: System,
    rows: Sequence[SaveRow],
    key: Mapping[str, str],
    pointed: Sequence[targets.Pointed] = (),
    sources: Mapping[str, str] | None = None,
) -> SystemScore:
    mine = [row for row in rows if row.system == system]
    scored = [row for row in mine if row.pair_id in key]

    found = [row for row in scored if row.found_existing]
    times = [row.elapsed_ms for row in mine[1:]]
    picked = [row for row in found if row.relation_pick is not None]
    aimed = [row for row in pointed if row.system == system]
    lost_record = _rate(scored, key, UNRELATED, "caught")
    false_alarm = _rate(scored, key, UNRELATED, "flagged")
    correct = sum(1 for row in picked if row.relation_pick == key[row.pair_id])

    return SystemScore(
        system=system,
        rows=len(mine),
        scored=len(scored),
        unkeyed=len(mine) - len(scored),
        by_relation=_relation_rows(scored, key),
        duplicate_caught=_rate(scored, key, RESTATEMENT, "caught"),
        duplicate_noticed=_rate(scored, key, RESTATEMENT, "caught", "flagged"),
        replacement_flagged=_rate(scored, key, REPLACEMENT, "flagged"),
        lost_update=_rate(scored, key, REPLACEMENT, "caught"),
        lost_record=lost_record,
        false_alarm=false_alarm,
        lost_record_any=Rate(
            count=lost_record.count + _pointed_in_key(aimed, key, "merged", UNRELATED),
            total=len(scored),
        ),
        false_alarm_any=Rate(
            count=false_alarm.count + _pointed_in_key(aimed, key, "flagged", UNRELATED),
            total=len(scored),
        ),
        pointed=_pointed_score(aimed),
        by_source=_source_rows(scored, key, sources or {}),
        found_existing=Rate(count=len(found), total=len(scored)),
        relation_pick=Rate(count=correct, total=len(picked)),
        degraded=Rate(count=sum(1 for row in mine if row.degraded is not None), total=len(mine)),
        median_ms=median(times),
        p95_ms=percentile(times, 0.95),
        timed_calls=len(times),
    )


def score_save(
    rows: Sequence[SaveRow],
    key: Mapping[str, str],
    systems: Sequence[System] = ("v1", "v2"),
    pointed: Sequence[targets.Pointed] = (),
    sources: Mapping[str, str] | None = None,
) -> SaveScore:
    return SaveScore(
        systems=[score_system(system, rows, key, pointed, sources) for system in systems]
    )


def score_files(out_dir: Path | None = None) -> SaveScore:
    base = out_dir if out_dir is not None else paths.ensure_out_dir()
    rows: list[SaveRow] = []

    for system in ("v1", "v2"):
        path = base / f"save-{system}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path, SaveRow))

    pairs_path = base / "pairs.jsonl"
    sources = (
        {pair.pair_id: pair.source for pair in read_jsonl(pairs_path, PairRow)}
        if pairs_path.exists()
        else {}
    )
    score = score_save(
        rows,
        load_key(base / "key-relation.jsonl"),
        pointed=targets.judged_pointed(base),
        sources=sources,
    )
    (base / SCORE_NAME).write_text(
        json.dumps(score.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the save replay against the key.")
    parser.parse_args()
    score = score_files()

    for item in score.systems:
        print(f"{item.system}: scored {item.scored} of {item.rows}")


if __name__ == "__main__":
    main()
