from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from eval_v1v2 import paths
from eval_v1v2.dup_rules import SignalRow, load_signals
from eval_v1v2.judges import COVERS_PREAMBLE, Item, Outcome, run_task
from eval_v1v2.rows import PairRow, Row, read_jsonl
from eval_v1v2.second_judge import CALL_TIMEOUT, CONCURRENCY, MAX_ATTEMPTS, band_pair_ids
from tabularium.engine.judge import JudgeUnavailable, SubprocessJudgeAdapter

QUESTIONS: tuple[str, ...] = ("new_covers_old", "old_covers_new")
MODEL = "sonnet"
STOP_AFTER_FAILURES = 6
YesNo = Literal["yes", "no"]
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

SONNET_REPLY = (
    "Answer both questions for the pair below. Reply with one JSON object and nothing else: "
    '{"new_covers_old": "yes" or "no", "old_covers_new": "yes" or "no", '
    '"reason": "one short sentence naming what is missing, or empty"}.'
)


class SonnetCall(Row):
    pair_id: str
    new_covers_old: YesNo | None
    old_covers_new: YesNo | None
    reason: str | None
    elapsed_ms: float
    attempts: int
    error: str | None

    @property
    def answered(self) -> bool:
        return self.error is None and self.new_covers_old is not None


class SonnetAnswer(Row):
    new_covers_old: YesNo
    old_covers_new: YesNo
    reason: str = ""


class UnparseableAnswer(Exception):
    pass


def item_id(pair_id: str, question: str) -> str:
    return f"{pair_id}#{question}"


def split_item_id(value: str) -> tuple[str, str]:
    pair_id, _, question = value.partition("#")

    return pair_id, question


def cover_items(pairs: Sequence[PairRow]) -> list[Item]:
    return [
        Item(
            item_id=item_id(pair.pair_id, question),
            fields={"question": question, "existing": pair.existing, "incoming": pair.incoming},
        )
        for pair in pairs
        for question in QUESTIONS
    ]


def run_judges(out_dir: Path, base: Path) -> Outcome:
    pairs = read_jsonl(base / "pairs.jsonl", PairRow)
    out_dir.mkdir(parents=True, exist_ok=True)

    return run_task("covers", cover_items(pairs), out_dir=out_dir)


def render_sonnet_prompt(pair: PairRow) -> str:
    return (
        f"{COVERS_PREAMBLE}\n\n{SONNET_REPLY}\n\n"
        f"EXISTING: {pair.existing}\nINCOMING: {pair.incoming}"
    )


def parse_sonnet(text: str) -> SonnetAnswer:
    found = _JSON_OBJECT.search(text)

    if found is None:
        raise UnparseableAnswer(text[:200])

    try:
        return SonnetAnswer.model_validate(json.loads(found.group(0)))
    except (ValueError, ValidationError) as exc:
        raise UnparseableAnswer(str(exc)[:200]) from exc


async def ask_once(adapter: SubprocessJudgeAdapter, pair: PairRow, attempts: int) -> SonnetCall:
    started = time.perf_counter()
    answer: SonnetAnswer | None = None
    error: str | None = None

    try:
        answer = parse_sonnet(await adapter.call(render_sonnet_prompt(pair)))
    except (JudgeUnavailable, UnparseableAnswer) as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"

    return SonnetCall(
        pair_id=pair.pair_id,
        new_covers_old=None if answer is None else answer.new_covers_old,
        old_covers_new=None if answer is None else answer.old_covers_new,
        reason=None if answer is None else answer.reason,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        attempts=attempts,
        error=error,
    )


async def ask(adapter: SubprocessJudgeAdapter, pair: PairRow) -> SonnetCall:
    result = await ask_once(adapter, pair, 1)

    for attempt in range(2, MAX_ATTEMPTS + 1):
        if result.answered:
            break

        result = await ask_once(adapter, pair, attempt)

    return result


async def ask_all(
    adapter: SubprocessJudgeAdapter,
    pairs: Sequence[PairRow],
    sink: Callable[[SonnetCall], None],
) -> list[SonnetCall]:
    gate = asyncio.Semaphore(CONCURRENCY)
    failures = 0
    results: list[SonnetCall] = []

    async def one(pair: PairRow) -> None:
        nonlocal failures

        if failures >= STOP_AFTER_FAILURES:
            return

        async with gate:
            if failures >= STOP_AFTER_FAILURES:
                return

            result = await ask(adapter, pair)

        failures = 0 if result.answered else failures + 1
        sink(result)
        results.append(result)

    await asyncio.gather(*(one(pair) for pair in pairs))

    return results


def load_sonnet_calls(path: Path) -> dict[str, SonnetCall]:
    if not path.exists():
        return {}

    latest: dict[str, SonnetCall] = {}

    for call in read_jsonl(path, SonnetCall):
        latest[call.pair_id] = call

    return latest


def pending_pairs(
    pairs: Mapping[str, PairRow], signals: Mapping[str, SignalRow], done: Mapping[str, SonnetCall]
) -> list[PairRow]:
    return [
        pairs[pid]
        for pid in band_pair_ids(signals, None)
        if pid not in done or not done[pid].answered
    ]


def run_sonnet(out_dir: Path, base: Path) -> int:
    signals = load_signals(base / "experiments" / "dup-rules" / "signals.jsonl")
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "sonnet-calls.jsonl"
    todo = pending_pairs(pairs, signals, load_sonnet_calls(log))
    binary = shutil.which("claude")

    if binary is None:
        print("claude CLI not found", file=sys.stderr)

        return 1

    handle = log.open("a", encoding="utf-8")

    def sink(call: SonnetCall) -> None:
        handle.write(json.dumps(call.model_dump(mode="json"), ensure_ascii=False) + "\n")
        handle.flush()

    os.chdir(tempfile.mkdtemp(prefix="covers-sonnet-"))
    adapter = SubprocessJudgeAdapter(cli="claude", binary=binary, model=MODEL, timeout=CALL_TIMEOUT)
    results = asyncio.run(ask_all(adapter, todo, sink))
    handle.close()
    landed = sum(1 for call in results if call.answered)
    band = len(band_pair_ids(signals, None))
    done = sum(1 for call in load_sonnet_calls(log).values() if call.answered)
    print(f"sonnet: {landed} landed this run, {done} of {band} band pairs answered", flush=True)

    return 0 if done == band else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Containment judges and the Sonnet arm.")
    parser.add_argument("phase", choices=("judges", "sonnet"))
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=paths.OUT_DIR / "experiments" / "covers")
    args = parser.parse_args()

    if args.phase == "judges":
        outcome = run_judges(args.out_dir, args.base)
        print(
            json.dumps(
                {
                    "key_rows": len(outcome.key),
                    "unlabelled": outcome.unlabelled,
                    "raw_agreement": outcome.agreement["raw_agreement"],
                    "cohens_kappa": outcome.agreement["cohens_kappa"],
                }
            )
        )

        return

    sys.exit(run_sonnet(args.out_dir, args.base))


if __name__ == "__main__":
    main()
