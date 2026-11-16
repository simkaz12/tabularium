from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eval_v1v2 import paths
from eval_v1v2.rows import CorpusRow, QuestionRow, SearchRow, read_jsonl, write_jsonl
from eval_v1v2.v1_engine import V1Harness, load_corpus, open_v1, snapshot_clock
from tabularium.results import RecallResult

RECALL_LIMIT = 10
LEGS = frozenset({"vector", "text"})
SELF_CHECK_SIZE = 20
SELF_CHECK_MIN_RANK_ONE = 18
SELF_CHECK_SEED = 20261112


class DegradedRecall(RuntimeError):
    pass


@dataclass(frozen=True)
class SelfCheck:
    sampled: int
    rank_one: int
    required: int
    ranks: list[int | None]

    @property
    def passed(self) -> bool:
        return self.sampled > 0 and self.rank_one >= self.required

    def as_json(self) -> dict[str, object]:
        return {
            "sampled": self.sampled,
            "rank_one": self.rank_one,
            "required": self.required,
            "ranks": self.ranks,
            "passed": self.passed,
        }


def require_both_legs(result: RecallResult, query: str) -> None:
    if result.degraded or not LEGS <= set(result.legs_used):
        raise DegradedRecall(
            f"recall for {query!r} ran degraded={result.degraded} reason={result.reason!r} "
            f"legs_used={result.legs_used}"
        )


def sample_known_items(
    corpus: list[CorpusRow], size: int = SELF_CHECK_SIZE, seed: int = SELF_CHECK_SEED
) -> list[CorpusRow]:
    copies = Counter((row.scope, row.content) for row in corpus)
    unique = [row for row in corpus if row.kind == "fact" and copies[(row.scope, row.content)] == 1]

    return random.Random(seed).sample(unique, min(size, len(unique)))


async def self_check(harness: V1Harness, sample: list[CorpusRow]) -> SelfCheck:
    ranks: list[int | None] = []

    for row in sample:
        result = await harness.engine.recall(row.content, scope=row.scope, limit=RECALL_LIMIT)

        require_both_legs(result, row.content)

        ids = [str(hit.id) for hit in result.hits]
        ranks.append(ids.index(row.id) + 1 if row.id in ids else None)

    return SelfCheck(
        sampled=len(sample),
        rank_one=sum(1 for rank in ranks if rank == 1),
        required=min(SELF_CHECK_MIN_RANK_ONE, len(sample)),
        ranks=ranks,
    )


async def run_search(harness: V1Harness, questions: list[QuestionRow]) -> list[SearchRow]:
    rows: list[SearchRow] = []

    if not questions:
        return rows

    first = questions[0]

    warm = await harness.engine.recall(first.query, scope=first.scope, limit=RECALL_LIMIT)

    require_both_legs(warm, first.query)

    for question in questions:
        started = time.perf_counter()
        result = await harness.engine.recall(
            question.query, scope=question.scope, limit=RECALL_LIMIT
        )
        elapsed = round((time.perf_counter() - started) * 1000, 2)

        require_both_legs(result, question.query)

        for rank, hit in enumerate(result.hits, start=1):
            rows.append(
                SearchRow(
                    question_id=question.question_id,
                    arm="v1",
                    rank=rank,
                    record_id=str(hit.id),
                    score=hit.score,
                    elapsed_ms=elapsed,
                )
            )

    return rows


def zero_hit_calls(questions: list[QuestionRow], rows: list[SearchRow]) -> int:
    return len(questions) - len({row.question_id for row in rows})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask the v1 engine the eval questions")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--store", type=Path, default=None)

    return parser


async def _main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    corpus = read_jsonl(args.corpus or paths.out_path("corpus.jsonl"), CorpusRow)
    questions = read_jsonl(args.questions or paths.out_path("questions.jsonl"), QuestionRow)[
        : args.limit
    ]
    out = args.out or paths.out_path("search-v1.jsonl")
    harness = await open_v1(args.store, use_assessor=False, clock=snapshot_clock(corpus))
    started = time.perf_counter()

    await load_corpus(
        harness, corpus, lambda n: print(f"loaded {n}/{len(corpus)}", file=sys.stderr)
    )

    print(
        f"loaded {len(corpus)} rows in {round(time.perf_counter() - started, 1)} s", file=sys.stderr
    )

    print(f"before upkeep: {await harness.adapter.maintenance_debt()}", file=sys.stderr)

    folded = time.perf_counter()

    await harness.adapter.upkeep()

    print(
        f"after upkeep ({round(time.perf_counter() - folded, 1)} s): "
        f"{await harness.adapter.maintenance_debt()}",
        file=sys.stderr,
    )

    try:
        check = await self_check(harness, sample_known_items(corpus))

        if not check.passed:
            print(
                f"known-item self-check failed: {check.rank_one} of {check.sampled} at rank 1",
                file=sys.stderr,
            )
            print(f"ranks: {check.ranks}", file=sys.stderr)
            print(json.dumps({"event": "v1-search-selfcheck", **check.as_json()}))

            return 1

        rows = await run_search(harness, questions)
    except DegradedRecall as error:
        print(f"refusing to write {out}: {error}", file=sys.stderr)

        return 1

    write_jsonl(out, rows)
    print(
        json.dumps(
            {
                "event": "v1-search",
                "questions": len(questions),
                "rows": len(rows),
                "zero_hit_calls": zero_hit_calls(questions, rows),
                "self_check": check.as_json(),
            }
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(sys.argv[1:])))
