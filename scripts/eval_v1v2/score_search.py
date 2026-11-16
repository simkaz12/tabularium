from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import get_args

from pydantic import BaseModel

from eval_v1v2 import paths
from eval_v1v2.rows import Arm, CorpusRow, LabelRow, QuestionRow, SearchRow, read_jsonl
from eval_v1v2.score_save import Rate, median, percentile

ARMS: tuple[str, ...] = get_args(Arm)
AS_CALLED_ARMS: tuple[str, ...] = ("v1", "v2_rerank", "v2_plain")
FACTS_ONLY_ARMS: tuple[str, ...] = ("v1", "v2_rerank_facts", "v2_plain_facts")
FACT = "fact"
VARIANTS: tuple[str, ...] = ("min", "claude", "codex")
BASELINE = "v1"
TOP = 10
SEED = 20261112
RESAMPLES = 1000
TIE = 1e-9
SCORE_NAME = "score-search.json"

Grades = dict[tuple[str, str], int]


class ArmScore(BaseModel):
    arm: str
    questions: int
    answered: int
    ndcg: float | None
    ndcg_questions: int
    hit1: Rate
    hit5: Rate
    precision5: Rate
    mrr: float
    mrr_questions: int
    wins: int | None
    ties: int | None
    losses: int | None
    diff_mean: float | None
    diff_low: float | None
    diff_high: float | None
    median_ms: float | None
    p95_ms: float | None
    timed_calls: int


class VariantScore(BaseModel):
    variant: str
    graded: int
    leftover: int
    unjudged: int
    arms: list[ArmScore]


class KindMix(BaseModel):
    arm: str
    facts: int
    decisions: int


class SearchScore(BaseModel):
    as_called: list[VariantScore]
    facts_only: list[VariantScore]
    kind_mix: list[KindMix]


def split_item_id(item_id: str) -> tuple[str, str]:
    question_id, record_id = item_id.rsplit(":", 1)

    return question_id, record_id.lower()


def grades_for(labels: Sequence[LabelRow], variant: str) -> Grades:
    by_item: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)

    for row in labels:
        by_item[split_item_id(row.item_id)][row.judge] = int(row.label)

    grades: Grades = {}

    for item, judged in by_item.items():
        if variant == "min":
            if len(judged) == 2:
                grades[item] = min(judged.values())
        elif variant in judged:
            grades[item] = judged[variant]

    return grades


def dcg(gains: Sequence[int]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ideal_gains(grades: Grades, question_id: str) -> list[int]:
    pool = sorted(
        (grade for (question, _), grade in grades.items() if question == question_id),
        reverse=True,
    )

    return pool[:TOP]


def ndcg(gains: Sequence[int], ideal: Sequence[int]) -> float | None:
    best = dcg(ideal)

    return dcg(gains) / best if best > 0 else None


def _ranked(rows: Sequence[SearchRow]) -> dict[str, list[SearchRow]]:
    by_question: dict[str, list[SearchRow]] = defaultdict(list)

    for row in rows:
        by_question[row.question_id].append(row)

    return {
        question: sorted(hits, key=lambda hit: hit.rank)[:TOP]
        for question, hits in by_question.items()
    }


def _gains(hits: Sequence[SearchRow], grades: Grades) -> list[int]:
    return [grades.get((hit.question_id, hit.record_id.lower()), 0) for hit in hits]


def bootstrap(
    diffs: Sequence[float], seed: int = SEED, resamples: int = RESAMPLES
) -> tuple[float, float] | None:
    if not diffs:
        return None

    rng = random.Random(seed)
    size = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=size)) / size for _ in range(resamples))
    low = means[int(0.025 * resamples)]
    high = means[int(0.975 * resamples) - 1]

    return low, high


def per_question_ndcg(
    ranked: Mapping[str, Sequence[SearchRow]], questions: Sequence[str], grades: Grades
) -> dict[str, float]:
    scores: dict[str, float] = {}

    for question in questions:
        value = ndcg(_gains(ranked.get(question, []), grades), ideal_gains(grades, question))
        if value is not None:
            scores[question] = value

    return scores


def score_arm(
    arm: str,
    rows: Sequence[SearchRow],
    questions: Sequence[str],
    grades: Grades,
    baseline: Mapping[str, float] | None,
) -> ArmScore:
    mine = [row for row in rows if row.arm == arm]
    ranked = _ranked(mine)
    scores = per_question_ndcg(ranked, questions, grades)
    gains = {question: _gains(ranked.get(question, []), grades) for question in questions}

    hit1 = sum(1 for g in gains.values() if g[:1] == [2])
    hit5 = sum(1 for g in gains.values() if 2 in g[:5])
    relevant = sum(1 for g in gains.values() for gain in g[:5] if gain >= 1)
    reciprocal = [
        next((1 / rank for rank, gain in enumerate(g, start=1) if gain == 2), 0.0)
        for g in gains.values()
    ]

    wins = ties = losses = None
    diff_mean = low = high = None

    if baseline is not None:
        shared = [question for question in scores if question in baseline]
        diffs = [scores[question] - baseline[question] for question in shared]
        wins = sum(1 for d in diffs if d > TIE)
        losses = sum(1 for d in diffs if d < -TIE)
        ties = len(diffs) - wins - losses
        diff_mean = sum(diffs) / len(diffs) if diffs else None
        interval = bootstrap(diffs)
        low, high = interval if interval is not None else (None, None)

    times = list({row.question_id: row.elapsed_ms for row in mine}.values())

    return ArmScore(
        arm=arm,
        questions=len(questions),
        answered=len([q for q in questions if q in ranked]),
        ndcg=sum(scores.values()) / len(scores) if scores else None,
        ndcg_questions=len(scores),
        hit1=Rate(count=hit1, total=len(questions)),
        hit5=Rate(count=hit5, total=len(questions)),
        precision5=Rate(count=relevant, total=5 * len(questions)),
        mrr=sum(reciprocal) / len(reciprocal) if reciprocal else 0.0,
        mrr_questions=len(reciprocal),
        wins=wins,
        ties=ties,
        losses=losses,
        diff_mean=diff_mean,
        diff_low=low,
        diff_high=high,
        median_ms=median(times),
        p95_ms=percentile(times, 0.95),
        timed_calls=len(times),
    )


def only_facts(
    rows: Sequence[SearchRow], grades: Grades, kinds: Mapping[str, str]
) -> tuple[list[SearchRow], Grades]:
    kept = [row for row in rows if kinds.get(row.record_id.lower()) == FACT]
    pool = {item: grade for item, grade in grades.items() if kinds.get(item[1]) == FACT}

    return kept, pool


def score_variant(
    variant: str,
    rows: Sequence[SearchRow],
    questions: Sequence[str],
    labels: Sequence[LabelRow],
    arms: Sequence[str],
    facts_only: bool,
    kinds: Mapping[str, str],
) -> VariantScore:
    grades = grades_for(labels, variant)

    if facts_only:
        rows, grades = only_facts(rows, grades, kinds)

    present = [arm for arm in arms if any(row.arm == arm for row in rows)]
    baseline = per_question_ndcg(_ranked([r for r in rows if r.arm == BASELINE]), questions, grades)
    seen = {
        (row.question_id, row.record_id.lower())
        for row in rows
        if row.rank <= TOP and row.arm in present
    }

    return VariantScore(
        variant=variant,
        graded=len(grades),
        leftover=len(set(grades) - seen),
        unjudged=len(seen - set(grades)),
        arms=[
            score_arm(arm, rows, questions, grades, None if arm == BASELINE else baseline)
            for arm in present
        ],
    )


def kind_mix(rows: Sequence[SearchRow], kinds: Mapping[str, str]) -> list[KindMix]:
    mix: list[KindMix] = []

    for arm in ARMS:
        hits = [row for row in rows if row.arm == arm and row.rank <= TOP]

        if hits:
            facts = sum(1 for row in hits if kinds.get(row.record_id.lower()) == FACT)
            mix.append(KindMix(arm=arm, facts=facts, decisions=len(hits) - facts))

    return mix


def score_search(
    rows: Sequence[SearchRow],
    questions: Sequence[str],
    labels: Sequence[LabelRow],
    kinds: Mapping[str, str],
    variants: Sequence[str] = VARIANTS,
) -> SearchScore:
    return SearchScore(
        as_called=[
            score_variant(v, rows, questions, labels, AS_CALLED_ARMS, False, kinds)
            for v in variants
        ],
        facts_only=[
            score_variant(v, rows, questions, labels, FACTS_ONLY_ARMS, True, kinds)
            for v in variants
        ],
        kind_mix=kind_mix(rows, kinds),
    )


def score_files(out_dir: Path | None = None) -> SearchScore:
    base = out_dir if out_dir is not None else paths.ensure_out_dir()
    rows: list[SearchRow] = []

    for arm in ARMS:
        path = base / f"search-{arm}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path, SearchRow))

    questions = [q.question_id for q in read_jsonl(base / "questions.jsonl", QuestionRow)]
    labels = read_jsonl(base / "labels-relevance.jsonl", LabelRow)
    kinds = {row.id.lower(): row.kind for row in read_jsonl(base / "corpus.jsonl", CorpusRow)}
    score = score_search(rows, questions, labels, kinds)
    (base / SCORE_NAME).write_text(
        json.dumps(score.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    return score


def main() -> None:
    argparse.ArgumentParser(
        description="Score the search arms against judged relevance."
    ).parse_args()
    score = score_files()

    for name, variants in (("as called", score.as_called), ("facts only", score.facts_only)):
        for variant in variants:
            print(f"{name} {variant.variant}: {variant.graded} graded, {variant.unjudged} unjudged")


if __name__ == "__main__":
    main()
